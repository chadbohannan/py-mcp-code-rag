"""Unit tests for code-rag-mcp.py."""
from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


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

def test_search_builds_search_url(mcp_mod):
    payload = [{"path": "x", "summary": "s", "score": 0.9}]
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        result = asyncio.run(mcp_mod.search("test query", top_k=3))
    url = m.call_args[0][0].full_url
    assert "/api/search" in url and "top_k=3" in url
    assert result[0]["path"] == "x"


def test_get_unit_uses_post(mcp_mod):
    payload = [{"path": "x", "content": "code", "summary": "s"}]
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        result = asyncio.run(mcp_mod.get_unit(["x", "y"]))
    req = m.call_args[0][0]
    assert req.get_method() == "POST"
    assert "/api/units/fetch" in req.full_url
    assert json.loads(req.data.decode()) == {"paths": ["x", "y"]}
    assert result == payload


def test_list_units_passes_globs_and_limit(mcp_mod):
    with patch("urllib.request.urlopen", return_value=_fake_response([])) as m:
        asyncio.run(mcp_mod.list_units(globs=["*.py"], limit=50))
    url = m.call_args[0][0].full_url
    assert "/api/units" in url and "globs=%2A.py" in url and "limit=50" in url


def test_list_files(mcp_mod):
    payload = [{"repo": "r", "path": "f.py", "indexed_at": "t"}]
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        result = asyncio.run(mcp_mod.list_files())
    assert "/api/files" in m.call_args[0][0].full_url
    assert result == payload


def test_list_repos(mcp_mod):
    payload = [{"name": "r", "root": "/r", "added_at": "t", "description": ""}]
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        result = asyncio.run(mcp_mod.list_repos())
    assert "/api/repos" in m.call_args[0][0].full_url
    assert result == payload


def test_index_status_returns_repos_list(mcp_mod):
    payload = {
        "repos": [{"repo": "r", "root": "/r", "file_count": 1,
                   "unit_count": 2, "last_indexed_at": "t"}],
        "total_units": 2, "embed_count": 2,
    }
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        result = asyncio.run(mcp_mod.index_status())
    assert "/api/status" in m.call_args[0][0].full_url
    assert result == payload["repos"]


# --- Index control tools ---------------------------------------------------

def test_index_start_posts_paths_and_reindex(mcp_mod):
    payload = {"running": True, "last_result": None, "last_finished_at": None}
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        asyncio.run(mcp_mod.index_start(["/a", "/b"], reindex=True))
    req = m.call_args[0][0]
    assert req.get_method() == "POST" and "/api/index" in req.full_url
    assert json.loads(req.data.decode()) == {"paths": ["/a", "/b"], "reindex": True}


def test_index_job_status_is_get(mcp_mod):
    payload = {"running": False, "last_result": "ok", "last_finished_at": "t"}
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        result = asyncio.run(mcp_mod.index_job_status())
    req = m.call_args[0][0]
    assert req.get_method() == "GET" and "/api/index/status" in req.full_url
    assert result == payload


def test_index_cancel_is_post(mcp_mod):
    payload = {"running": False, "last_result": "cancelled", "last_finished_at": "t"}
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        result = asyncio.run(mcp_mod.index_cancel())
    req = m.call_args[0][0]
    assert req.get_method() == "POST" and "/api/index/cancel" in req.full_url
    assert result == payload


# --- Error path ------------------------------------------------------------

def test_unreachable_server_raises_toolerror(mcp_mod):
    import urllib.error
    with patch("urllib.request.urlopen",
               side_effect=urllib.error.URLError("connection refused")):
        from fastmcp.exceptions import ToolError
        with pytest.raises(ToolError, match="cannot reach code-rag"):
            asyncio.run(mcp_mod.search("anything"))


def test_http_error_raises_toolerror(mcp_mod):
    import io
    import urllib.error
    err = urllib.error.HTTPError(
        url="http://x", code=422, msg="Unprocessable",
        hdrs=None, fp=io.BytesIO(b'{"detail":"bad query"}')
    )
    with patch("urllib.request.urlopen", side_effect=err):
        from fastmcp.exceptions import ToolError
        with pytest.raises(ToolError, match="422"):
            asyncio.run(mcp_mod.search("bad"))
