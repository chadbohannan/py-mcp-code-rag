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
    args = cli._build_parser().parse_args(["--base-url", "http://flag-host:2", "repos"])
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
