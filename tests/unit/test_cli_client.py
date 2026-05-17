"""Unit tests for code-rag-cli.py (HTTP client CLI)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _load_cli():
    path = Path(__file__).parent.parent.parent / "code-rag-cli.py"
    spec = importlib.util.spec_from_file_location("code_rag_cli", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cli = _load_cli()


def _fake_response(payload):
    mock = MagicMock()
    mock.__enter__.return_value.read.return_value = json.dumps(payload).encode()
    mock.__exit__.return_value = False
    return mock


def _fake_responses(payloads):
    return [_fake_response(p) for p in payloads]


# --- --base-url / CODE_RAG_URL ----------------------------------------------

def test_explicit_base_url_overrides_env(monkeypatch):
    monkeypatch.setenv("CODE_RAG_URL", "http://env-host:1")
    args = cli._build_parser().parse_args(["--base-url", "http://flag-host:2", "repos"])
    assert args.base_url == "http://flag-host:2"


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("CODE_RAG_URL", "http://env-host:9999")
    args = cli._build_parser().parse_args(["repos"])
    assert args.base_url == "http://env-host:9999"


# --- --json flag ------------------------------------------------------------

def test_json_flag_outputs_raw_payload(capsys):
    payload = [{"path": "x", "summary": "s", "score": 0.9}]
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)):
        args = cli._build_parser().parse_args(["--json", "search", "auth"])
        args.func(args, args.base_url)
    assert json.loads(capsys.readouterr().out) == payload


def test_no_json_flag_uses_pretty_format(capsys):
    payload = [{"path": "repo/file.py:foo", "summary": "does foo", "score": 0.75}]
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)):
        args = cli._build_parser().parse_args(["search", "auth"])
        args.func(args, args.base_url)
    out = capsys.readouterr().out
    assert "0.7500" in out and "repo/file.py:foo" in out and "does foo" in out


# --- staleness --------------------------------------------------------------

def test_staleness_subcommand(capsys):
    payload = [{
        "repo": "alpha", "root": "/r", "last_indexed_at": "t1",
        "last_commit_at": "t2", "stale": True, "reason": "older than HEAD",
    }]
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        args = cli._build_parser().parse_args(["staleness"])
        args.func(args, args.base_url)
    assert "/api/repos/staleness" in m.call_args[0][0].full_url
    out = capsys.readouterr().out
    assert "alpha" in out and "older than HEAD" in out


# --- ls ---------------------------------------------------------------------

def test_ls_with_path_and_git_marker(capsys):
    payload = {
        "path": "/home/u/r", "parent": "/home/u", "is_git": True,
        "dirs": [{"name": "src", "path": "/home/u/r/src"}],
    }
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        args = cli._build_parser().parse_args(["ls", "/home/u/r"])
        args.func(args, args.base_url)
    assert "/api/ls" in m.call_args[0][0].full_url
    out = capsys.readouterr().out
    assert "*" in out and "src" in out


# --- index --wait -----------------------------------------------------------

def test_index_wait_polls_and_exits_ok(tmp_path, capsys):
    responses = _fake_responses([
        {"running": True,  "last_result": None, "last_finished_at": None},
        {"running": True,  "last_result": None, "last_finished_at": None},
        {"running": False, "last_result": "ok", "last_finished_at": "t"},
    ])
    with patch("urllib.request.urlopen", side_effect=responses), patch("time.sleep") as s:
        args = cli._build_parser().parse_args(["index", "--wait", str(tmp_path)])
        args.func(args, args.base_url)
    assert s.called
    assert "." in capsys.readouterr().err


def test_index_wait_nonzero_on_failure(tmp_path):
    responses = _fake_responses([
        {"running": True,  "last_result": None, "last_finished_at": None},
        {"running": False, "last_result": "boom", "last_finished_at": "t"},
    ])
    with patch("urllib.request.urlopen", side_effect=responses), patch("time.sleep"):
        args = cli._build_parser().parse_args(["index", "--wait", str(tmp_path)])
        with pytest.raises(SystemExit) as ei:
            args.func(args, args.base_url)
    assert ei.value.code == 1
