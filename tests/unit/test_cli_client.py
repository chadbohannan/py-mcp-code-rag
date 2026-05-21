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
