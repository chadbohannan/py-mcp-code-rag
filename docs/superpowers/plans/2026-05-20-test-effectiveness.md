# Test Effectiveness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve all eight test-effectiveness gaps identified on the `rework-mcp` branch — restore end-to-end webui coverage, tighten mocked-client assertions, and cover every CLI subcommand.

**Architecture:** Three test layers. (1) New `tests/integration/test_webui_api.py` drives the FastAPI app via `fastapi.testclient.TestClient` against a real SQLite index built with `FakeEmbedder` + `FakeSummarizer`. (2) Existing `tests/unit/test_mcp_client.py` and `tests/unit/test_cli_client.py` are tightened to use a `parse_request()` helper that compares URL paths and parsed query dicts, not substrings. (3) New tests fill smoke coverage of 11 untested CLI subcommands and the MCP `main()` transport selection.

**Tech Stack:** pytest, `fastapi.testclient.TestClient`, `urllib.parse.parse_qs`, `unittest.mock`, existing `FakeEmbedder`/`FakeSummarizer`/`make_git_project` fixtures from `tests/conftest.py`.

**Spec:** `docs/superpowers/specs/2026-05-20-test-effectiveness-design.md`

**Note on TDD:** These are *characterization tests* against existing code. The pattern is **write test → run test → it should PASS**. If a test fails, do NOT modify the test to make it pass — investigate the underlying behavior. A failure here means either the test is wrong, the underlying contract drifted, or a real bug was found. STOP and report before continuing.

---

## File Map

| Path | Action | Responsibility |
|---|---|---|
| `tests/conftest.py` | MODIFY | Add `parse_request()` helper |
| `tests/integration/conftest.py` | CREATE | `indexed_db` + `client` fixtures |
| `tests/integration/test_webui_api.py` | CREATE | 12 endpoint tests via TestClient |
| `tests/unit/test_mcp_client.py` | MODIFY | Migrate to `parse_request`; add 3 new tests |
| `tests/unit/test_cli_client.py` | MODIFY | Migrate to `parse_request`; lazy-load fix; tighten `--wait`; add 11 subcommand tests |

---

## Task 1: Foundation — parse_request helper, integration fixtures, smoke test

**Files:**
- Modify: `tests/conftest.py` (add helper at module level)
- Create: `tests/integration/conftest.py`
- Create: `tests/integration/test_webui_api.py` (smoke test only at this stage)

- [ ] **Step 1: Add `parse_request` helper to `tests/conftest.py`**

Add at the top of `tests/conftest.py` after the existing imports:

```python
import json as _json
import urllib.parse as _urlparse


def parse_request(mock_call) -> tuple[str, str, dict, dict | None]:
    """Decompose a urlopen MagicMock call into (method, path, query, json_body).

    *mock_call* is ``mock.call_args`` from a patched ``urllib.request.urlopen``.
    *query* is a dict of {name: [values]} from ``parse_qs`` — note values are
    always lists, even single-valued ones. *json_body* is ``None`` for GETs.
    """
    req = mock_call[0][0]
    parsed = _urlparse.urlparse(req.full_url)
    query = _urlparse.parse_qs(parsed.query, keep_blank_values=True)
    body = _json.loads(req.data.decode()) if req.data else None
    return req.get_method(), parsed.path, query, body
```

- [ ] **Step 2: Run existing tests to confirm helper does not break them**

Run: `~/.local/bin/uv run pytest tests/unit -q`
Expected: All tests still pass (helper is not yet used).

- [ ] **Step 3: Create `tests/integration/conftest.py`**

```python
"""Shared fixtures for the webui integration tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mcp_rag.indexer import run_index
from mcp_rag.webui import create_app
from tests.conftest import FakeEmbedder, FakeSummarizer, make_git_project


REPO_FILES = {
    "src/app.py": (
        "def login(user):\n"
        "    return user\n"
        "\n"
        "def logout():\n"
        "    return None\n"
        "\n"
        "class Session:\n"
        "    def keep_alive(self):\n"
        "        return True\n"
    ),
    "src/db.py": (
        "def connect():\n"
        "    return None\n"
        "\n"
        "def disconnect():\n"
        "    return None\n"
    ),
    "README.md": "# Overview\n\nProject docs for the test fixture repo.\n",
}


@pytest.fixture(scope="module")
def indexed_db(tmp_path_factory):
    """Build a real on-disk SQLite index once per test module."""
    db_path = tmp_path_factory.mktemp("idx") / "test.db"
    repo_root = tmp_path_factory.mktemp("repo") / "testrepo"
    make_git_project(repo_root, REPO_FILES)
    run_index(
        roots=[repo_root],
        db_path=db_path,
        embedder=FakeEmbedder(dim=4),
        summarizer=FakeSummarizer(),
    )
    return db_path


@pytest.fixture
def client(indexed_db):
    """fastapi.TestClient wired to the indexed DB.

    The summarizer_factory is supplied because create_app requires it, but no
    integration test in this file exercises the indexing endpoints.
    """
    app = create_app(
        db_path=indexed_db,
        embedder=FakeEmbedder(dim=4),
        summarizer_factory=lambda: FakeSummarizer(),
    )
    return TestClient(app)
```

- [ ] **Step 4: Create `tests/integration/test_webui_api.py` with one smoke test**

```python
"""Integration tests for the webui REST API.

Drives the FastAPI app in-process via TestClient against a real SQLite index
built with FakeEmbedder + FakeSummarizer. Replaces the deleted
test_server_search.py from before the MCP-HTTP migration.
"""

from __future__ import annotations


def test_status_endpoint_returns_populated_index(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["total_units"] > 0
    assert body["embed_count"] == body["total_units"]
    assert len(body["repos"]) == 1
    assert body["repos"][0]["repo"] == "testrepo"
```

- [ ] **Step 5: Run the smoke test**

Run: `~/.local/bin/uv run pytest tests/integration/test_webui_api.py -v`
Expected: PASS. If FAIL, the fixture is wrong — investigate `create_app` signature, `make_git_project` arguments, or sqlite-vec loading (see CLAUDE.md gotcha).

- [ ] **Step 6: Run full suite to confirm no regressions**

Run: `~/.local/bin/uv run pytest -q`
Expected: All previous tests + 1 new test pass.

- [ ] **Step 7: Commit**

```bash
git add tests/conftest.py tests/integration/conftest.py tests/integration/test_webui_api.py
git commit -m "Add parse_request helper and webui integration fixtures

Foundation for the test-effectiveness work: parse_request decomposes
urlopen MagicMock calls into (method, path, query_dict, body); the
integration conftest builds a real SQLite index once per module so
TestClient-driven endpoint tests can run hermetically."
```

---

## Task 2: Integration tests — /api/search

**Files:**
- Modify: `tests/integration/test_webui_api.py` (append)

- [ ] **Step 1: Append three search tests to `tests/integration/test_webui_api.py`**

```python
def test_search_returns_results_with_required_fields(client):
    r = client.get("/api/search", params={"q": "user login", "top_k": 5})
    assert r.status_code == 200
    results = r.json()
    assert len(results) > 0
    for entry in results:
        assert set(entry.keys()) >= {"path", "summary", "score"}
        assert isinstance(entry["score"], float)


def test_search_results_are_ordered_by_score_descending(client):
    r = client.get("/api/search", params={"q": "database connection", "top_k": 5})
    scores = [e["score"] for e in r.json()]
    assert scores == sorted(scores, reverse=True)


def test_search_globs_filter_narrows_to_markdown(client):
    r = client.get(
        "/api/search",
        params=[("q", "overview"), ("top_k", 10), ("globs", "*.md*")],
    )
    assert r.status_code == 200
    paths = [e["path"] for e in r.json()]
    assert paths, "globs filter should still return at least one match"
    assert all(".md" in p for p in paths)


def test_search_top_k_caps_returned_count(client):
    r = client.get("/api/search", params={"q": "anything", "top_k": 2})
    assert r.status_code == 200
    assert len(r.json()) <= 2
```

- [ ] **Step 2: Run the new tests**

Run: `~/.local/bin/uv run pytest tests/integration/test_webui_api.py -v`
Expected: 5 tests pass (1 smoke from Task 1 + 4 search). If `test_search_globs_filter_narrows_to_markdown` returns zero results, the FakeSummarizer text for the README section may not match "overview" semantically — pick a query that does match, or assert presence by path rather than scoring.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_webui_api.py
git commit -m "Add /api/search integration tests

Covers result shape, score ordering, top_k cap, and glob filtering."
```

---

## Task 3: Integration tests — /api/units, /api/unit, /api/units/fetch

**Files:**
- Modify: `tests/integration/test_webui_api.py` (append)

- [ ] **Step 1: Append three units tests**

```python
def test_units_listing_returns_paths_and_summaries(client):
    r = client.get("/api/units", params={"limit": 100})
    assert r.status_code == 200
    units = r.json()
    assert len(units) > 0
    assert all(set(u.keys()) >= {"path", "summary"} for u in units)
    paths = {u["path"] for u in units}
    assert any(p.endswith(":login") for p in paths)
    assert any(p.endswith(":connect") for p in paths)


def test_unit_fetch_single_returns_full_content(client):
    listing = client.get("/api/units").json()
    target = next(u["path"] for u in listing if u["path"].endswith(":login"))
    r = client.get("/api/unit", params={"path": target})
    assert r.status_code == 200
    body = r.json()
    assert body["path"] == target
    assert "def login" in body["content"]
    assert body["summary"]


def test_units_fetch_post_returns_only_matching_paths(client):
    listing = client.get("/api/units").json()
    real_path = listing[0]["path"]
    r = client.post(
        "/api/units/fetch",
        json={"paths": [real_path, "testrepo/does/not/exist:nope"]},
    )
    assert r.status_code == 200
    results = r.json()
    assert len(results) == 1
    assert results[0]["path"] == real_path
```

- [ ] **Step 2: Run the new tests**

Run: `~/.local/bin/uv run pytest tests/integration/test_webui_api.py -v`
Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_webui_api.py
git commit -m "Add /api/units, /api/unit, /api/units/fetch integration tests

Verifies listing shape, single-unit content retrieval, and that
POST /api/units/fetch silently drops paths that don't match."
```

---

## Task 4: Integration tests — /api/files, /api/repos, /api/status

**Files:**
- Modify: `tests/integration/test_webui_api.py` (append)

- [ ] **Step 1: Append three tests**

```python
def test_files_listing_returns_three_files(client):
    r = client.get("/api/files")
    assert r.status_code == 200
    files = r.json()
    paths = {f["path"] for f in files}
    assert paths == {"src/app.py", "src/db.py", "README.md"}


def test_files_glob_filter_narrows_to_markdown(client):
    r = client.get("/api/files", params={"globs": "*.md"})
    assert r.status_code == 200
    paths = [f["path"] for f in r.json()]
    assert paths == ["README.md"]


def test_repos_and_status_consistent(client):
    repos = client.get("/api/repos").json()
    assert len(repos) == 1
    assert repos[0]["name"] == "testrepo"

    status = client.get("/api/status").json()
    assert status["repos"][0]["repo"] == "testrepo"
    assert status["repos"][0]["file_count"] == 3
```

- [ ] **Step 2: Run the new tests**

Run: `~/.local/bin/uv run pytest tests/integration/test_webui_api.py -v`
Expected: All tests pass. If file count mismatches, check `discover_files` exclusion globs — the `make_git_project` helper writes only the listed files but `.git/` is created by `git_init`.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_webui_api.py
git commit -m "Add /api/files, /api/repos, /api/status integration tests

Confirms file listing, glob filtering, and that /api/status counts
agree with /api/repos."
```

---

## Task 5: Integration tests — /api/repos/staleness, /api/browse, 404 error shape

**Files:**
- Modify: `tests/integration/test_webui_api.py` (append)

- [ ] **Step 1: Append three tests**

```python
def test_staleness_fresh_after_index(client):
    r = client.get("/api/repos/staleness")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    assert rows[0]["repo"] == "testrepo"
    assert rows[0]["stale"] is False


def test_browse_root_then_drill_into_file(client):
    root = client.get("/api/browse", params={"path": ""}).json()
    assert any(node["type"] == "repo" and node["name"] == "testrepo" for node in root)

    drill = client.get("/api/browse", params={"path": "testrepo/src/app.py"}).json()
    types = {node["type"] for node in drill}
    assert "unit" in types


def test_unit_not_found_returns_404_with_detail_key(client):
    """Load-bearing cross-check: code-rag-mcp.py extracts ``detail`` from
    the JSON body on HTTPError. If FastAPI ever stops emitting that field,
    the MCP error message degrades to the raw body. This test guards the
    contract from the webui side."""
    r = client.get("/api/unit", params={"path": "no/such/path:no"})
    assert r.status_code == 404
    body = r.json()
    assert "detail" in body
    assert body["detail"] == "Unit not found"
```

- [ ] **Step 2: Run the new tests**

Run: `~/.local/bin/uv run pytest tests/integration/test_webui_api.py -v`
Expected: All 14 integration tests pass (1 smoke + 4 search + 3 units + 3 files/repos/status + 3 staleness/browse/404).

- [ ] **Step 3: Confirm timing**

Run: `~/.local/bin/uv run pytest tests/integration/test_webui_api.py --durations=0`
Expected: Total time under 5s on the Pi. If slower, accept it but note in the commit message.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_webui_api.py
git commit -m "Add staleness, browse, and 404-shape integration tests

The 404 test is the load-bearing cross-check between the webui's
HTTPException(detail=...) shape and code-rag-mcp.py's detail extraction."
```

---

## Task 6: Unit polish — parse_request migration in test_mcp_client.py

**Files:**
- Modify: `tests/unit/test_mcp_client.py`

This task is a mechanical refactor: every place that does `m.call_args[0][0].full_url` or `json.loads(req.data.decode())` becomes a `parse_request(m.call_args)` call with structured equality.

- [ ] **Step 1: Add the helper import and rewrite all existing tests**

Replace the contents of `tests/unit/test_mcp_client.py` with:

```python
"""Unit tests for code-rag-mcp.py."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import parse_request


def _load_mcp():
    path = Path(__file__).parent.parent.parent / "code-rag-mcp.py"
    spec = importlib.util.spec_from_file_location("code_rag_mcp", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mcp_mod():
    return _load_mcp()


def _fake_response(payload):
    mock = MagicMock()
    mock.__enter__.return_value.read.return_value = json.dumps(payload).encode()
    mock.__exit__.return_value = False
    return mock


# --- Read tools ------------------------------------------------------------


def test_search_builds_get_with_query_and_top_k(mcp_mod):
    payload = [{"path": "x", "summary": "s", "score": 0.9}]
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        result = asyncio.run(mcp_mod.search("test query", top_k=3))
    method, path, query, body = parse_request(m.call_args)
    assert method == "GET"
    assert path == "/api/search"
    assert query == {"q": ["test query"], "top_k": ["3"]}
    assert body is None
    assert result == payload


def test_search_omits_globs_when_none(mcp_mod):
    with patch("urllib.request.urlopen", return_value=_fake_response([])) as m:
        asyncio.run(mcp_mod.search("q"))
    _, _, query, _ = parse_request(m.call_args)
    assert "globs" not in query


def test_get_unit_posts_paths_list(mcp_mod):
    payload = [{"path": "x", "content": "code", "summary": "s"}]
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        result = asyncio.run(mcp_mod.get_unit(["x", "y"]))
    method, path, _, body = parse_request(m.call_args)
    assert method == "POST"
    assert path == "/api/units/fetch"
    assert body == {"paths": ["x", "y"]}
    assert result == payload


def test_list_units_encodes_multiple_globs_as_repeated_param(mcp_mod):
    with patch("urllib.request.urlopen", return_value=_fake_response([])) as m:
        asyncio.run(mcp_mod.list_units(globs=["*.py", "backend/*"], limit=50))
    method, path, query, _ = parse_request(m.call_args)
    assert method == "GET"
    assert path == "/api/units"
    assert query == {"globs": ["*.py", "backend/*"], "limit": ["50"]}


def test_list_files(mcp_mod):
    payload = [{"repo": "r", "path": "f.py", "indexed_at": "t"}]
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        result = asyncio.run(mcp_mod.list_files())
    method, path, query, _ = parse_request(m.call_args)
    assert method == "GET"
    assert path == "/api/files"
    assert query == {}
    assert result == payload


def test_list_repos(mcp_mod):
    payload = [{"name": "r", "root": "/r", "added_at": "t", "description": ""}]
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        result = asyncio.run(mcp_mod.list_repos())
    _, path, _, _ = parse_request(m.call_args)
    assert path == "/api/repos"
    assert result == payload


def test_index_status_returns_repos_list(mcp_mod):
    payload = {
        "repos": [
            {
                "repo": "r",
                "root": "/r",
                "file_count": 1,
                "unit_count": 2,
                "last_indexed_at": "t",
            }
        ],
        "total_units": 2,
        "embed_count": 2,
    }
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        result = asyncio.run(mcp_mod.index_status())
    _, path, _, _ = parse_request(m.call_args)
    assert path == "/api/status"
    assert result == payload["repos"]


def test_index_status_missing_repos_key_returns_empty(mcp_mod):
    """Exercises the data.get('repos', []) defensive default."""
    with patch("urllib.request.urlopen", return_value=_fake_response({})):
        result = asyncio.run(mcp_mod.index_status())
    assert result == []


def test_staleness_returns_list(mcp_mod):
    payload = [
        {
            "repo": "r",
            "root": "/r",
            "last_indexed_at": "t1",
            "last_commit_at": "t2",
            "stale": True,
            "reason": "older than HEAD",
        }
    ]
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        result = asyncio.run(mcp_mod.staleness())
    _, path, _, _ = parse_request(m.call_args)
    assert path == "/api/repos/staleness"
    assert result == payload


# --- Index control tools ---------------------------------------------------


def test_index_start_posts_paths_and_reindex(mcp_mod):
    payload = {"running": True, "last_result": None, "last_finished_at": None}
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        asyncio.run(mcp_mod.index_start(["/a", "/b"], reindex=True))
    method, path, _, body = parse_request(m.call_args)
    assert method == "POST"
    assert path == "/api/index"
    assert body == {"paths": ["/a", "/b"], "reindex": True}


def test_index_job_status_is_get(mcp_mod):
    payload = {"running": False, "last_result": "ok", "last_finished_at": "t"}
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        result = asyncio.run(mcp_mod.index_job_status())
    method, path, _, _ = parse_request(m.call_args)
    assert method == "GET"
    assert path == "/api/index/status"
    assert result == payload


def test_index_cancel_is_post(mcp_mod):
    payload = {"running": False, "last_result": "cancelled", "last_finished_at": "t"}
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        result = asyncio.run(mcp_mod.index_cancel())
    method, path, _, _ = parse_request(m.call_args)
    assert method == "POST"
    assert path == "/api/index/cancel"
    assert result == payload


# --- Error path ------------------------------------------------------------


def test_unreachable_server_raises_toolerror(mcp_mod):
    import urllib.error

    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        from fastmcp.exceptions import ToolError

        with pytest.raises(ToolError, match="cannot reach code-rag"):
            asyncio.run(mcp_mod.search("anything"))


def test_http_error_raises_toolerror(mcp_mod):
    import io
    import urllib.error

    err = urllib.error.HTTPError(
        url="http://x",
        code=422,
        msg="Unprocessable",
        hdrs=None,
        fp=io.BytesIO(b'{"detail":"bad query"}'),
    )
    with patch("urllib.request.urlopen", side_effect=err):
        from fastmcp.exceptions import ToolError

        with pytest.raises(ToolError, match="422"):
            asyncio.run(mcp_mod.search("bad"))
```

- [ ] **Step 2: Run the rewritten file**

Run: `~/.local/bin/uv run pytest tests/unit/test_mcp_client.py -v`
Expected: 14 tests pass (12 from before — one renamed to `test_list_units_encodes_multiple_globs_as_repeated_param` — plus 2 new: `test_search_omits_globs_when_none` and `test_index_status_missing_repos_key_returns_empty`). Resolves issues #2, #3, #5 from the spec.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_mcp_client.py
git commit -m "Tighten test_mcp_client.py with parse_request and missing-case tests

- Replaces full_url substring asserts with parse_request equality
- Adds multi-globs encoding test (catches list-serialization bugs)
- Adds None-globs negative test
- Adds index_status({}) defensive-default test"
```

---

## Task 7: Unit polish — test_mcp_client.py main() transport tests

**Files:**
- Modify: `tests/unit/test_mcp_client.py` (append)

- [ ] **Step 1: Append two `main()` tests**

```python
# --- main() entrypoint -----------------------------------------------------


def test_main_stdio_is_default(mcp_mod, monkeypatch):
    monkeypatch.setattr("sys.argv", ["code-rag-mcp", "--base-url", "http://h:1"])
    with patch.object(mcp_mod.mcp, "run") as run:
        mcp_mod.main()
    run.assert_called_once_with()


def test_main_http_uses_port(mcp_mod, monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["code-rag-mcp", "--http", "--port", "9999"]
    )
    with patch.object(mcp_mod.mcp, "run") as run:
        mcp_mod.main()
    run.assert_called_once_with(
        transport="streamable-http", host="127.0.0.1", port=9999
    )
```

- [ ] **Step 2: Run the new tests**

Run: `~/.local/bin/uv run pytest tests/unit/test_mcp_client.py::test_main_stdio_is_default tests/unit/test_mcp_client.py::test_main_http_uses_port -v`
Expected: Both pass. Resolves issue #8.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_mcp_client.py
git commit -m "Test code-rag-mcp main() transport selection

Locks in stdio default and --http/--port wiring."
```

---

## Task 8: Unit polish — test_cli_client.py refactor

**Files:**
- Modify: `tests/unit/test_cli_client.py`

Three changes in one task: lazy module load via fixture, parse_request migration, tighten the `--wait` test.

- [ ] **Step 1: Replace the contents of `tests/unit/test_cli_client.py`**

```python
"""Unit tests for code-rag-cli.py (HTTP client CLI)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import parse_request


def _load_cli():
    path = Path(__file__).parent.parent.parent / "code-rag-cli.py"
    spec = importlib.util.spec_from_file_location("code_rag_cli", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cli():
    return _load_cli()


def _fake_response(payload):
    mock = MagicMock()
    mock.__enter__.return_value.read.return_value = json.dumps(payload).encode()
    mock.__exit__.return_value = False
    return mock


def _fake_responses(payloads):
    return [_fake_response(p) for p in payloads]


def _run(cli, argv, payload):
    """Parse argv, mock urlopen with payload, invoke the subcommand."""
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        args = cli._build_parser().parse_args(argv)
        args.func(args, args.base_url)
    return m


# --- --base-url / CODE_RAG_URL ----------------------------------------------


def test_explicit_base_url_overrides_env(cli, monkeypatch):
    monkeypatch.setenv("CODE_RAG_URL", "http://env-host:1")
    args = cli._build_parser().parse_args(
        ["--base-url", "http://flag-host:2", "repos"]
    )
    assert args.base_url == "http://flag-host:2"


def test_env_var_overrides_default(cli, monkeypatch):
    monkeypatch.setenv("CODE_RAG_URL", "http://env-host:9999")
    args = cli._build_parser().parse_args(["repos"])
    assert args.base_url == "http://env-host:9999"


# --- --json flag ------------------------------------------------------------


def test_json_flag_outputs_raw_payload(cli, capsys):
    payload = [{"path": "x", "summary": "s", "score": 0.9}]
    _run(cli, ["--json", "search", "auth"], payload)
    assert json.loads(capsys.readouterr().out) == payload


def test_no_json_flag_uses_pretty_format(cli, capsys):
    payload = [{"path": "repo/file.py:foo", "summary": "does foo", "score": 0.75}]
    _run(cli, ["search", "auth"], payload)
    out = capsys.readouterr().out
    assert "0.7500" in out and "repo/file.py:foo" in out and "does foo" in out


# --- index --wait -----------------------------------------------------------


def test_index_wait_polls_until_finished(cli, tmp_path, capsys):
    """Three responses: initial POST + running poll + finished poll.

    Asserts exactly 2 sleep calls (after the two intermediate states), the
    final status is printed to stdout, and the dots are written to stderr.
    """
    responses = _fake_responses(
        [
            {"running": True, "last_result": None, "last_finished_at": None},
            {"running": True, "last_result": None, "last_finished_at": None},
            {"running": False, "last_result": "ok", "last_finished_at": "t"},
        ]
    )
    with (
        patch("urllib.request.urlopen", side_effect=responses),
        patch("time.sleep") as s,
    ):
        args = cli._build_parser().parse_args(["index", "--wait", str(tmp_path)])
        args.func(args, args.base_url)
    assert s.call_count == 2
    captured = capsys.readouterr()
    assert "running: False" in captured.out
    assert "last_result: ok" in captured.out
    assert captured.err.count(".") == 2


def test_index_wait_nonzero_on_failure(cli, tmp_path):
    responses = _fake_responses(
        [
            {"running": True, "last_result": None, "last_finished_at": None},
            {"running": False, "last_result": "boom", "last_finished_at": "t"},
        ]
    )
    with patch("urllib.request.urlopen", side_effect=responses), patch("time.sleep"):
        args = cli._build_parser().parse_args(["index", "--wait", str(tmp_path)])
        with pytest.raises(SystemExit) as ei:
            args.func(args, args.base_url)
    assert ei.value.code == 1
```

- [ ] **Step 2: Run the refactored file**

Run: `~/.local/bin/uv run pytest tests/unit/test_cli_client.py -v`
Expected: 7 tests pass (down from 8 — the staleness and ls tests were inlined helpers we'll re-add in Task 9 under the new pattern).

Wait — recount. Original was 8 tests. After this refactor: `test_explicit_base_url_overrides_env`, `test_env_var_overrides_default`, `test_json_flag_outputs_raw_payload`, `test_no_json_flag_uses_pretty_format`, `test_index_wait_polls_until_finished`, `test_index_wait_nonzero_on_failure` = 6 tests. The original `test_staleness_subcommand` and `test_ls_with_path_and_git_marker` are deleted here and re-added in Task 9 alongside the other 11 subcommand tests for consistency.

So Step 2 expects 6 tests pass.

- [ ] **Step 3: Run full unit suite to confirm no regressions**

Run: `~/.local/bin/uv run pytest tests/unit -q`
Expected: All pass (the temporarily-missing staleness/ls coverage will be replaced in Task 9 before merging).

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_cli_client.py
git commit -m "Refactor test_cli_client.py: lazy module load, parse_request, tighten --wait

- Lazy cli load via module-scoped fixture (matches test_mcp_client.py)
- Tightens --wait to assert exact poll count and stdout/stderr contents
- Drops staleness/ls tests temporarily; reinstated with the full
  subcommand sweep in the next commit"
```

---

## Task 9: Unit polish — add tests for the 11 untested CLI subcommands

**Files:**
- Modify: `tests/unit/test_cli_client.py` (append)

Each subcommand gets one smoke test: assert the URL `(method, path)` and one observable line in stdout. Use the `_run` helper from Task 8.

- [ ] **Step 1: Append subcommand tests**

```python
# --- CLI subcommands (one smoke test per command) --------------------------


def test_unit_subcommand(cli, capsys):
    payload = {"path": "r/f.py:foo", "content": "def foo(): pass", "summary": "s"}
    m = _run(cli, ["unit", "r/f.py:foo"], payload)
    method, path, query, _ = parse_request(m.call_args)
    assert method == "GET"
    assert path == "/api/unit"
    assert query == {"path": ["r/f.py:foo"]}
    out = capsys.readouterr().out
    assert "# r/f.py:foo" in out
    assert "def foo(): pass" in out


def test_fetch_subcommand(cli, capsys):
    payload = [
        {"path": "a", "content": "code-a", "summary": "sa"},
        {"path": "b", "content": "code-b", "summary": "sb"},
    ]
    m = _run(cli, ["fetch", "a", "b"], payload)
    method, path, _, body = parse_request(m.call_args)
    assert method == "POST"
    assert path == "/api/units/fetch"
    assert body == {"paths": ["a", "b"]}
    out = capsys.readouterr().out
    assert "code-a" in out and "code-b" in out
    assert "---" in out  # separator between units


def test_units_subcommand_with_globs(cli, capsys):
    payload = [{"path": "r/x.py:foo", "summary": "does foo"}]
    m = _run(cli, ["units", "--limit", "5", "--glob", "*.py:*"], payload)
    method, path, query, _ = parse_request(m.call_args)
    assert method == "GET"
    assert path == "/api/units"
    assert query == {"limit": ["5"], "globs": ["*.py:*"]}
    assert "r/x.py:foo" in capsys.readouterr().out


def test_files_subcommand(cli, capsys):
    payload = [{"repo": "r", "path": "f.py", "indexed_at": "2026-01-01"}]
    m = _run(cli, ["files"], payload)
    _, path, _, _ = parse_request(m.call_args)
    assert path == "/api/files"
    out = capsys.readouterr().out
    assert "r/f.py" in out and "2026-01-01" in out


def test_repos_subcommand(cli, capsys):
    payload = [{"name": "r", "root": "/r", "added_at": "2026-01-01", "description": ""}]
    m = _run(cli, ["repos"], payload)
    _, path, _, _ = parse_request(m.call_args)
    assert path == "/api/repos"
    assert "r" in capsys.readouterr().out


def test_status_subcommand(cli, capsys):
    payload = {
        "repos": [
            {
                "repo": "r",
                "root": "/r",
                "file_count": 3,
                "unit_count": 7,
                "last_indexed_at": "2026-01-01",
            }
        ],
        "total_units": 7,
        "embed_count": 7,
    }
    m = _run(cli, ["status"], payload)
    _, path, _, _ = parse_request(m.call_args)
    assert path == "/api/status"
    out = capsys.readouterr().out
    assert "total_units: 7" in out
    assert "embed_count: 7" in out


def test_browse_subcommand(cli, capsys):
    payload = [
        {
            "type": "repo",
            "name": "r",
            "path": "r",
            "summary": "",
            "has_children": True,
        }
    ]
    m = _run(cli, ["browse"], payload)
    _, path, query, _ = parse_request(m.call_args)
    assert path == "/api/browse"
    assert query == {"path": [""]}
    assert "repo" in capsys.readouterr().out


def test_index_subcommand_no_wait(cli, tmp_path, capsys):
    payload = {"running": True, "last_result": None, "last_finished_at": None}
    m = _run(cli, ["index", str(tmp_path)], payload)
    method, path, _, body = parse_request(m.call_args)
    assert method == "POST"
    assert path == "/api/index"
    assert body == {"paths": [str(tmp_path)], "reindex": False}
    assert "running: True" in capsys.readouterr().out


def test_index_status_subcommand(cli, capsys):
    payload = {"running": False, "last_result": "ok", "last_finished_at": "t"}
    m = _run(cli, ["index-status"], payload)
    method, path, _, _ = parse_request(m.call_args)
    assert method == "GET"
    assert path == "/api/index/status"
    assert "running: False" in capsys.readouterr().out


def test_index_cancel_subcommand(cli, capsys):
    payload = {"running": False, "last_result": "cancelled", "last_finished_at": "t"}
    m = _run(cli, ["index-cancel"], payload)
    method, path, _, _ = parse_request(m.call_args)
    assert method == "POST"
    assert path == "/api/index/cancel"
    assert "running: False" in capsys.readouterr().out


def test_clear_repo_subcommand(cli, capsys):
    payload = {"ok": True, "repo": "myrepo"}
    m = _run(cli, ["clear-repo", "myrepo"], payload)
    method, path, query, _ = parse_request(m.call_args)
    assert method == "POST"
    assert path == "/api/clear_repo"
    assert query == {"repo": ["myrepo"]}
    assert "cleared: myrepo" in capsys.readouterr().out


def test_staleness_subcommand(cli, capsys):
    payload = [
        {
            "repo": "alpha",
            "root": "/r",
            "last_indexed_at": "t1",
            "last_commit_at": "t2",
            "stale": True,
            "reason": "older than HEAD",
        }
    ]
    m = _run(cli, ["staleness"], payload)
    _, path, _, _ = parse_request(m.call_args)
    assert path == "/api/repos/staleness"
    out = capsys.readouterr().out
    assert "alpha" in out and "older than HEAD" in out


def test_ls_subcommand_marks_git_repo(cli, capsys):
    payload = {
        "path": "/home/u/r",
        "parent": "/home/u",
        "is_git": True,
        "dirs": [{"name": "src", "path": "/home/u/r/src"}],
    }
    m = _run(cli, ["ls", "/home/u/r"], payload)
    _, path, query, _ = parse_request(m.call_args)
    assert path == "/api/ls"
    assert query == {"path": ["/home/u/r"]}
    out = capsys.readouterr().out
    assert "*" in out and "src" in out
```

- [ ] **Step 2: Run all CLI tests**

Run: `~/.local/bin/uv run pytest tests/unit/test_cli_client.py -v`
Expected: 6 (from Task 8) + 13 (here — 11 untested commands + staleness + ls re-added) = 19 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_cli_client.py
git commit -m "Add smoke tests for all CLI subcommands

Covers the 11 previously untested subcommands plus the restored
staleness and ls tests. Each asserts (method, path, query/body)
and one line of stdout."
```

---

## Task 10: Final verification

- [ ] **Step 1: Full test suite**

Run: `~/.local/bin/uv run pytest -q`
Expected: All tests pass.

- [ ] **Step 2: Lint**

Run: `make lint`
Expected: green.

- [ ] **Step 3: Diff stats vs master**

Run: `git diff master..HEAD --stat -- tests/`
Expected: tests/ delta is net non-negative vs master (we are rebuilding what was deleted, not just polishing).

- [ ] **Step 4: Confirm acceptance criteria**

| Criterion | How to verify |
|---|---|
| `make test` passes | Step 1 above |
| URL substring asserts replaced | `grep -n "full_url" tests/unit/test_mcp_client.py tests/unit/test_cli_client.py` returns nothing |
| Every CLI subparser covered | `grep -c "^def test_" tests/unit/test_cli_client.py` returns ≥ 19 |
| Integration suite under 5s on Pi | `~/.local/bin/uv run pytest tests/integration/test_webui_api.py --durations=0` |
| `make lint` green | Step 2 above |

- [ ] **Step 5: Push and open PR (only when user asks)**

Do NOT push without explicit user instruction.
