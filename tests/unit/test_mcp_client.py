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
