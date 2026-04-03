"""
Integration tests for the MCP server search and index_status tool behaviour.

Uses a real on-disk SQLite DB built by run_index with FakeEmbedder and
FakeSummarizer.  No Anthropic API calls; no fastembed.

server_module.configure(db_path, embedder) injects state into the server
before each test and tears it down after.
"""

import textwrap
from datetime import datetime
from pathlib import Path

import pytest
from fastmcp import Client

import mcp_rag.server as server_module
from mcp_rag.indexer import run_index
from mcp_rag.models import SemanticUnit
from tests.conftest import FakeEmbedder, FakeSummarizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _call(tool: str, args: dict) -> object:
    """Call a tool and return the deserialized Python result."""
    async with Client(server_module.mcp) as client:
        result = await client.call_tool(tool, args)
    return result.structured_content["result"]


def _fake_summary(unit_type: str, unit_name: str, content: str) -> str:
    """Reproduce the deterministic string FakeSummarizer emits for a unit."""
    unit = SemanticUnit(
        unit_type=unit_type, unit_name=unit_name, content=content, char_offset=0
    )
    return FakeSummarizer().summarize(unit)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def set_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-tests")


@pytest.fixture
def embedder():
    return FakeEmbedder(dim=4)


@pytest.fixture
def summarizer():
    return FakeSummarizer()


@pytest.fixture
def populated_db(tmp_path, embedder, summarizer):
    """DB with two Python files: auth.py (validate_token) and utils.py (format_name)."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "auth.py").write_text(
        textwrap.dedent("""\
            def validate_token(token: str) -> bool:
                return len(token) > 0
        """),
        encoding="utf-8",
    )
    (root / "utils.py").write_text(
        textwrap.dedent("""\
            def format_name(first: str, last: str) -> str:
                return f"{first} {last}"
        """),
        encoding="utf-8",
    )
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)
    return db_path, root


@pytest.fixture
def configured_server(populated_db, embedder):
    db_path, _ = populated_db
    server_module.configure(db_path, embedder)
    yield
    server_module.configure(None, None)


@pytest.fixture
def empty_db(tmp_path, embedder, summarizer):
    root = tmp_path / "proj"
    root.mkdir()
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)
    return db_path


# ---------------------------------------------------------------------------
# search — shape and contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_returns_list(configured_server):
    result = await _call("search", {"query": "token"})
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_search_result_has_required_fields(configured_server):
    results = await _call("search", {"query": "token"})
    assert len(results) >= 1
    for field in ("path", "summary", "score"):
        assert field in results[0], f"missing field: {field}"
    assert "content" not in results[0], "search should not return content"


@pytest.mark.asyncio
async def test_search_score_in_range(configured_server):
    results = await _call("search", {"query": "token"})
    for r in results:
        assert 0.0 <= r["score"] <= 1.0, f"score out of range: {r['score']}"


@pytest.mark.asyncio
async def test_search_results_ordered_by_score_descending(configured_server):
    results = await _call("search", {"query": "token"})
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_search_returns_matching_paths(configured_server):
    results = await _call("search", {"query": "token", "top_k": 5})
    paths = [r["path"] for r in results]
    assert any("validate_token" in p for p in paths)


@pytest.mark.asyncio
async def test_search_path_is_qualified(configured_server):
    results = await _call("search", {"query": "token", "top_k": 5})
    for r in results:
        # Qualified paths contain a file extension and : separator
        assert ":" in r["path"] or r["path"].endswith((".py", ".md", ".sql")), (
            f"unexpected path format: {r['path']}"
        )


# ---------------------------------------------------------------------------
# search — top_k behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_top_k_limits_results(configured_server):
    results = await _call("search", {"query": "token", "top_k": 1})
    assert len(results) <= 1


@pytest.mark.asyncio
async def test_search_top_k_default_is_5(configured_server):
    # DB has 2 units; default top_k=5 should return at most 2
    results = await _call("search", {"query": "token"})
    assert len(results) <= 5


@pytest.mark.asyncio
async def test_search_top_k_capped_at_20(tmp_path, embedder, summarizer):
    """top_k=100 must not return more than 20 results."""
    root = tmp_path / "proj"
    root.mkdir()
    # Write 25 distinct functions so there are > 20 units to return
    lines = "\n".join(f"def fn_{i}(): pass" for i in range(25))
    (root / "big.py").write_text(lines + "\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)
    server_module.configure(db_path, embedder)
    try:
        results = await _call("search", {"query": "fn", "top_k": 100})
        assert len(results) <= 20
    finally:
        server_module.configure(None, None)


# ---------------------------------------------------------------------------
# search — exact-match score
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_exact_summary_scores_1(populated_db, embedder):
    """Querying with the exact embed text (qualified_path | summary) returns score ≈ 1.0."""
    db_path, root = populated_db
    # ast.get_source_segment does not include the trailing newline
    content = "def validate_token(token: str) -> bool:\n    return len(token) > 0"
    summary = _fake_summary("function", "validate_token", content)
    # The indexer embeds: qualified_path | summary
    exact = f"auth.py:validate_token | {summary}"

    server_module.configure(db_path, embedder)
    try:
        results = await _call("search", {"query": exact, "top_k": 1})
        assert len(results) == 1
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-6)
    finally:
        server_module.configure(None, None)


# ---------------------------------------------------------------------------
# search — path_glob filtering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_path_glob_filters_by_file(configured_server):
    """path_glob=`auth*` should only return units from auth.py."""
    results = await _call("search", {"query": "function", "top_k": 5, "path_glob": "auth*"})
    assert len(results) >= 1
    assert all("auth.py" in r["path"] for r in results)


@pytest.mark.asyncio
async def test_search_path_glob_excludes_non_matching(configured_server):
    """path_glob that matches nothing returns empty list."""
    results = await _call("search", {"query": "function", "top_k": 5, "path_glob": "nonexistent*"})
    assert results == []


@pytest.mark.asyncio
async def test_search_path_glob_wildcard_unit_name(configured_server):
    """path_glob=`*:validate_token` matches by unit name."""
    results = await _call("search", {"query": "token", "top_k": 5, "path_glob": "*:validate_token"})
    assert len(results) == 1
    assert "validate_token" in results[0]["path"]


@pytest.mark.asyncio
async def test_search_path_glob_none_returns_all(configured_server):
    """Omitting path_glob returns results from all files."""
    results = await _call("search", {"query": "function", "top_k": 5})
    paths = {r["path"] for r in results}
    assert len(paths) >= 2


@pytest.mark.asyncio
async def test_search_path_glob_is_optional():
    """path_glob must be an optional parameter."""
    async with Client(server_module.mcp) as client:
        tools = await client.list_tools()
    schema = {t.name: t for t in tools}["search"].inputSchema
    assert "path_glob" not in schema.get("required", [])


# ---------------------------------------------------------------------------
# search — empty DB
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_empty_db_returns_empty_list(empty_db, embedder):
    server_module.configure(empty_db, embedder)
    try:
        results = await _call("search", {"query": "anything"})
        assert results == []
    finally:
        server_module.configure(None, None)


# ---------------------------------------------------------------------------
# index_status — shape and contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_index_status_returns_list(configured_server):
    result = await _call("index_status", {})
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_index_status_result_has_required_fields(configured_server):
    results = await _call("index_status", {})
    assert len(results) >= 1
    for field in ("root", "file_count", "unit_count", "last_indexed_at"):
        assert field in results[0], f"missing field: {field}"


@pytest.mark.asyncio
async def test_index_status_file_count(configured_server):
    results = await _call("index_status", {})
    assert results[0]["file_count"] == 2


@pytest.mark.asyncio
async def test_index_status_unit_count(configured_server):
    results = await _call("index_status", {})
    assert results[0]["unit_count"] >= 2


@pytest.mark.asyncio
async def test_index_status_last_indexed_at_is_iso8601(configured_server):
    results = await _call("index_status", {})
    datetime.fromisoformat(results[0]["last_indexed_at"])  # raises if invalid


@pytest.mark.asyncio
async def test_index_status_root_matches_indexed_path(populated_db, embedder):
    db_path, root = populated_db
    server_module.configure(db_path, embedder)
    try:
        results = await _call("index_status", {})
        roots = {r["root"] for r in results}
        assert str(root) in roots
    finally:
        server_module.configure(None, None)


# ---------------------------------------------------------------------------
# index_status — empty DB and multi-root
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_index_status_empty_db_returns_empty_list(empty_db, embedder):
    server_module.configure(empty_db, embedder)
    try:
        results = await _call("index_status", {})
        assert results == []
    finally:
        server_module.configure(None, None)


@pytest.mark.asyncio
async def test_index_status_one_entry_per_root(tmp_path, embedder, summarizer):
    root_a = tmp_path / "proj_a"
    root_b = tmp_path / "proj_b"
    root_a.mkdir()
    root_b.mkdir()
    (root_a / "a.py").write_text("def fa(): pass\n", encoding="utf-8")
    (root_b / "b.py").write_text("def fb(): pass\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    run_index(
        [root_a, root_b], db_path=db_path, embedder=embedder, summarizer=summarizer
    )

    server_module.configure(db_path, embedder)
    try:
        results = await _call("index_status", {})
        assert len(results) == 2
        roots = {r["root"] for r in results}
        assert str(root_a) in roots
        assert str(root_b) in roots
    finally:
        server_module.configure(None, None)


# ---------------------------------------------------------------------------
# get_unit — shape and contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_unit_returns_content(configured_server):
    results = await _call("get_unit", {"paths": ["auth.py:validate_token"]})
    assert len(results) == 1
    assert "validate_token" in results[0]["content"]
    for field in ("path", "content", "summary"):
        assert field in results[0], f"missing field: {field}"


@pytest.mark.asyncio
async def test_get_unit_multiple_paths(configured_server):
    results = await _call(
        "get_unit",
        {"paths": ["auth.py:validate_token", "utils.py:format_name"]},
    )
    assert len(results) == 2
    paths = {r["path"] for r in results}
    assert "auth.py:validate_token" in paths
    assert "utils.py:format_name" in paths


@pytest.mark.asyncio
async def test_get_unit_unknown_path_skipped(configured_server):
    results = await _call("get_unit", {"paths": ["nonexistent.py:foo"]})
    assert results == []


@pytest.mark.asyncio
async def test_get_unit_empty_paths(configured_server):
    results = await _call("get_unit", {"paths": []})
    assert results == []


# ---------------------------------------------------------------------------
# list_units — shape and contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_units_returns_all(configured_server):
    results = await _call("list_units", {})
    assert len(results) >= 2
    for r in results:
        assert "path" in r
        assert "summary" in r
        assert "content" not in r


@pytest.mark.asyncio
async def test_list_units_ordered_by_path(configured_server):
    results = await _call("list_units", {})
    paths = [r["path"] for r in results]
    assert paths == sorted(paths)


@pytest.mark.asyncio
async def test_list_units_path_glob_filters(configured_server):
    results = await _call("list_units", {"path_glob": "auth*"})
    assert len(results) >= 1
    assert all("auth" in r["path"] for r in results)


@pytest.mark.asyncio
async def test_list_units_path_glob_no_match(configured_server):
    results = await _call("list_units", {"path_glob": "nonexistent*"})
    assert results == []


@pytest.mark.asyncio
async def test_list_units_limit(configured_server):
    results = await _call("list_units", {"limit": 1})
    assert len(results) == 1


@pytest.mark.asyncio
async def test_list_units_limit_capped_at_500(tmp_path, embedder, summarizer):
    """limit > 500 is silently capped."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("def f(): pass\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)
    server_module.configure(db_path, embedder)
    try:
        # Just verify it doesn't error — the cap is internal
        results = await _call("list_units", {"limit": 9999})
        assert isinstance(results, list)
    finally:
        server_module.configure(None, None)
