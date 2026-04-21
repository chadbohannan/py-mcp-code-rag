"""Unit tests for the CLI client (mcp_rag.client).

All HTTP calls are monkeypatched via urllib so no server is needed.
"""

from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from importlib import import_module

_rag_cli = import_module("code-rag-cli")
main = _rag_cli.main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _argv(*args):
    return ["code-rag-cli"] + list(args)


def _mock_urlopen(response_data, status=200):
    """Return a context-manager mock for urllib.request.urlopen."""
    resp = MagicMock()
    resp.read.return_value = json.dumps(response_data).encode()
    resp.status = status
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return patch("mcp_rag.client.urllib.request.urlopen", return_value=resp)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_client_search(monkeypatch, capsys):
    data = [
        {"path": "repo/file.py:func", "summary": "Does stuff", "score": 0.95},
        {"path": "repo/other.py:cls", "summary": "Other thing", "score": 0.80},
    ]
    monkeypatch.setattr("sys.argv", _argv("search", "find functions"))
    with _mock_urlopen(data):
        main()
    out = capsys.readouterr().out
    assert "0.9500" in out
    assert "repo/file.py:func" in out
    assert "Does stuff" in out


def test_client_search_with_globs(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", _argv("search", "--glob", "*.py", "query"))
    with _mock_urlopen([]) as mock_open:
        main()
    url = mock_open.call_args[0][0]
    assert "globs" in url
    assert "%2A.py" in url or "*.py" in url


def test_client_search_top_k(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", _argv("search", "--top-k", "10", "query"))
    with _mock_urlopen([]) as mock_open:
        main()
    url = mock_open.call_args[0][0]
    assert "top_k=10" in url


# ---------------------------------------------------------------------------
# unit
# ---------------------------------------------------------------------------


def test_client_unit(monkeypatch, capsys):
    data = {"path": "repo/f.py:func", "content": "def func(): pass", "summary": "A function"}
    monkeypatch.setattr("sys.argv", _argv("unit", "repo/f.py:func"))
    with _mock_urlopen(data):
        main()
    out = capsys.readouterr().out
    assert "# repo/f.py:func" in out
    assert "def func(): pass" in out


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------


def test_client_fetch(monkeypatch, capsys):
    data = [
        {"path": "repo/a.py:f", "content": "def f(): ...", "summary": "f"},
        {"path": "repo/b.py:g", "content": "def g(): ...", "summary": "g"},
    ]
    monkeypatch.setattr("sys.argv", _argv("fetch", "repo/a.py:f", "repo/b.py:g"))
    with _mock_urlopen(data):
        main()
    out = capsys.readouterr().out
    assert "repo/a.py:f" in out
    assert "repo/b.py:g" in out
    assert "---" in out


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------


def test_client_units(monkeypatch, capsys):
    data = [{"path": "repo/f.py:func", "summary": "A function"}]
    monkeypatch.setattr("sys.argv", _argv("units"))
    with _mock_urlopen(data):
        main()
    out = capsys.readouterr().out
    assert "repo/f.py:func" in out
    assert "A function" in out


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------


def test_client_files(monkeypatch, capsys):
    data = [{"repo": "myrepo", "path": "src/main.py", "indexed_at": "2025-01-01"}]
    monkeypatch.setattr("sys.argv", _argv("files"))
    with _mock_urlopen(data):
        main()
    out = capsys.readouterr().out
    assert "myrepo/src/main.py" in out


# ---------------------------------------------------------------------------
# repos
# ---------------------------------------------------------------------------


def test_client_repos(monkeypatch, capsys):
    data = [{"name": "myrepo", "root": "/code/myrepo", "added_at": "2025-01-01", "description": ""}]
    monkeypatch.setattr("sys.argv", _argv("repos"))
    with _mock_urlopen(data):
        main()
    out = capsys.readouterr().out
    assert "myrepo" in out
    assert "/code/myrepo" in out


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def test_client_status(monkeypatch, capsys):
    data = {
        "repos": [
            {"repo": "r", "root": "/r", "file_count": 10, "unit_count": 50, "last_indexed_at": "2025-01-01"}
        ],
        "total_units": 50,
        "embed_count": 50,
    }
    monkeypatch.setattr("sys.argv", _argv("status"))
    with _mock_urlopen(data):
        main()
    out = capsys.readouterr().out
    assert "total_units: 50" in out
    assert "files=10" in out


# ---------------------------------------------------------------------------
# browse
# ---------------------------------------------------------------------------


def test_client_browse(monkeypatch, capsys):
    data = [
        {"type": "repo", "name": "myrepo", "path": "myrepo", "summary": "", "has_children": True, "unit_type": None}
    ]
    monkeypatch.setattr("sys.argv", _argv("browse"))
    with _mock_urlopen(data):
        main()
    out = capsys.readouterr().out
    assert "repo\tmyrepo" in out


def test_client_browse_with_path(monkeypatch, capsys):
    data = []
    monkeypatch.setattr("sys.argv", _argv("browse", "myrepo/src"))
    with _mock_urlopen(data) as mock_open:
        main()
    url = mock_open.call_args[0][0]
    assert "path=myrepo" in url and "src" in url


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------


def test_client_index_start(monkeypatch, capsys):
    data = {"running": True, "last_result": None, "last_finished_at": None}
    monkeypatch.setattr("sys.argv", _argv("index", "/some/path"))
    with _mock_urlopen(data):
        main()
    out = capsys.readouterr().out
    assert "running: True" in out


# ---------------------------------------------------------------------------
# index-status
# ---------------------------------------------------------------------------


def test_client_index_status(monkeypatch, capsys):
    data = {"running": False, "last_result": "ok", "last_finished_at": "2025-01-01T00:00:00"}
    monkeypatch.setattr("sys.argv", _argv("index-status"))
    with _mock_urlopen(data):
        main()
    out = capsys.readouterr().out
    assert "running: False" in out
    assert "last_result: ok" in out


# ---------------------------------------------------------------------------
# index-cancel
# ---------------------------------------------------------------------------


def test_client_index_cancel(monkeypatch, capsys):
    data = {"running": False, "last_result": None, "last_finished_at": None}
    monkeypatch.setattr("sys.argv", _argv("index-cancel"))
    with _mock_urlopen(data):
        main()
    out = capsys.readouterr().out
    assert "running: False" in out


# ---------------------------------------------------------------------------
# clear-repo
# ---------------------------------------------------------------------------


def test_client_clear_repo(monkeypatch, capsys):
    data = {"ok": True, "repo": "myrepo"}
    monkeypatch.setattr("sys.argv", _argv("clear-repo", "myrepo"))
    with _mock_urlopen(data):
        main()
    out = capsys.readouterr().out
    assert "cleared: myrepo" in out


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------


def test_client_http_error(monkeypatch, capsys):
    import urllib.error

    monkeypatch.setattr("sys.argv", _argv("repos"))
    error = urllib.error.HTTPError(
        url="http://localhost:8080/api/repos",
        code=500,
        msg="Server Error",
        hdrs={},
        fp=BytesIO(json.dumps({"detail": "something broke"}).encode()),
    )
    with patch("mcp_rag.client.urllib.request.urlopen", side_effect=error):
        with pytest.raises(SystemExit) as ei:
            main()
    assert ei.value.code == 1
    assert "something broke" in capsys.readouterr().err


def test_client_connection_error(monkeypatch, capsys):
    import urllib.error

    monkeypatch.setattr("sys.argv", _argv("repos"))
    error = urllib.error.URLError("Connection refused")
    with patch("mcp_rag.client.urllib.request.urlopen", side_effect=error):
        with pytest.raises(SystemExit) as ei:
            main()
    assert ei.value.code == 1
    assert "cannot reach server" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# base-url flag
# ---------------------------------------------------------------------------


def test_client_custom_base_url(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", _argv("--base-url", "http://myhost:9090", "repos"))
    with _mock_urlopen([]) as mock_open:
        main()
    url = mock_open.call_args[0][0]
    assert url.startswith("http://myhost:9090/")
