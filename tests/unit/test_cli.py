"""Unit tests for the CLI entry point (mcp_rag.__main__).

All external I/O (run_index, FastEmbedder, AnthropicSummarizer, mcp.run,
_read_embed_meta) is monkeypatched so no files, network, or servers are
touched.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcp_rag.__main__ import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _argv(*args):
    return ["mcp-rag"] + list(args)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_embedder(monkeypatch):
    inst = MagicMock()
    inst.model = "fake-model"
    inst.dim = 4
    cls = MagicMock(return_value=inst)
    monkeypatch.setattr("mcp_rag.__main__.FastEmbedder", cls)
    return cls, inst


@pytest.fixture
def mock_summarizer(monkeypatch):
    inst = MagicMock()
    cls = MagicMock(return_value=inst)
    monkeypatch.setattr("mcp_rag.__main__.AnthropicSummarizer", cls)
    return inst


@pytest.fixture
def mock_run_index(monkeypatch):
    m = MagicMock()
    monkeypatch.setattr("mcp_rag.__main__.run_index", m)
    return m


@pytest.fixture
def mock_server(monkeypatch):
    m = MagicMock()
    monkeypatch.setattr("mcp_rag.__main__.server", m)
    return m


@pytest.fixture
def mock_mcp(monkeypatch):
    m = MagicMock()
    monkeypatch.setattr("mcp_rag.__main__.mcp", m)
    return m


@pytest.fixture
def mock_read_meta(monkeypatch):
    m = MagicMock(return_value=("fake-model", 4))
    monkeypatch.setattr("mcp_rag.__main__._read_embed_meta", m)
    return m


# ---------------------------------------------------------------------------
# index subcommand — argument wiring
# ---------------------------------------------------------------------------

def test_index_calls_run_index(
    tmp_path, monkeypatch, mock_embedder, mock_summarizer, mock_run_index
):
    monkeypatch.setattr("sys.argv", _argv("index", str(tmp_path)))
    main()
    mock_run_index.assert_called_once()


def test_index_passes_resolved_path(
    tmp_path, monkeypatch, mock_embedder, mock_summarizer, mock_run_index
):
    monkeypatch.setattr("sys.argv", _argv("index", str(tmp_path)))
    main()
    roots = mock_run_index.call_args.kwargs["roots"]
    assert tmp_path.resolve() in roots


def test_index_default_db_is_index_db(
    tmp_path, monkeypatch, mock_embedder, mock_summarizer, mock_run_index
):
    monkeypatch.setattr("sys.argv", _argv("index", str(tmp_path)))
    main()
    assert mock_run_index.call_args.kwargs["db_path"] == Path("index.db")


def test_index_custom_db(
    tmp_path, monkeypatch, mock_embedder, mock_summarizer, mock_run_index
):
    monkeypatch.setattr(
        "sys.argv", _argv("index", "--db", str(tmp_path / "custom.db"), str(tmp_path))
    )
    main()
    assert mock_run_index.call_args.kwargs["db_path"] == tmp_path / "custom.db"


def test_index_reindex_false_by_default(
    tmp_path, monkeypatch, mock_embedder, mock_summarizer, mock_run_index
):
    monkeypatch.setattr("sys.argv", _argv("index", str(tmp_path)))
    main()
    assert mock_run_index.call_args.kwargs["reindex"] is False


def test_index_reindex_flag(
    tmp_path, monkeypatch, mock_embedder, mock_summarizer, mock_run_index
):
    monkeypatch.setattr("sys.argv", _argv("index", "--reindex", str(tmp_path)))
    main()
    assert mock_run_index.call_args.kwargs["reindex"] is True


def test_index_default_embed_model(
    tmp_path, monkeypatch, mock_embedder, mock_summarizer, mock_run_index
):
    from mcp_rag.embedder import DEFAULT_MODEL
    monkeypatch.setattr("sys.argv", _argv("index", str(tmp_path)))
    main()
    mock_embedder[0].assert_called_once_with(model_name=DEFAULT_MODEL)


def test_index_custom_embed_model(
    tmp_path, monkeypatch, mock_embedder, mock_summarizer, mock_run_index
):
    monkeypatch.setattr(
        "sys.argv", _argv("index", "--embed-model", "my/model", str(tmp_path))
    )
    main()
    mock_embedder[0].assert_called_once_with(model_name="my/model")


# ---------------------------------------------------------------------------
# index subcommand — error handling
# ---------------------------------------------------------------------------

def test_index_abort_error_exits_1(
    tmp_path, monkeypatch, mock_embedder, mock_summarizer, mock_run_index
):
    from mcp_rag.indexer import IndexAbortError
    mock_run_index.side_effect = IndexAbortError("no key")
    monkeypatch.setattr("sys.argv", _argv("index", str(tmp_path)))
    with pytest.raises(SystemExit) as ei:
        main()
    assert ei.value.code == 1


# ---------------------------------------------------------------------------
# serve subcommand
# ---------------------------------------------------------------------------

def test_serve_configures_server(
    monkeypatch, mock_embedder, mock_server, mock_mcp, mock_read_meta
):
    monkeypatch.setattr("sys.argv", _argv("serve"))
    main()
    mock_server.configure.assert_called_once()


def test_serve_calls_mcp_run(
    monkeypatch, mock_embedder, mock_server, mock_mcp, mock_read_meta
):
    monkeypatch.setattr("sys.argv", _argv("serve"))
    main()
    mock_mcp.run.assert_called_once()


def test_serve_uses_stdio_by_default(
    monkeypatch, mock_embedder, mock_server, mock_mcp, mock_read_meta
):
    monkeypatch.setattr("sys.argv", _argv("serve"))
    main()
    call_kwargs = mock_mcp.run.call_args.kwargs
    assert call_kwargs.get("transport") != "streamable-http"


def test_serve_http_flag_uses_http_transport(
    monkeypatch, mock_embedder, mock_server, mock_mcp, mock_read_meta
):
    monkeypatch.setattr("sys.argv", _argv("serve", "--http"))
    main()
    call_kwargs = mock_mcp.run.call_args.kwargs
    assert call_kwargs.get("transport") == "streamable-http"


def test_serve_http_binds_localhost(
    monkeypatch, mock_embedder, mock_server, mock_mcp, mock_read_meta
):
    monkeypatch.setattr("sys.argv", _argv("serve", "--http"))
    main()
    assert mock_mcp.run.call_args.kwargs.get("host") == "127.0.0.1"


def test_serve_default_port_is_8000(
    monkeypatch, mock_embedder, mock_server, mock_mcp, mock_read_meta
):
    monkeypatch.setattr("sys.argv", _argv("serve", "--http"))
    main()
    assert mock_mcp.run.call_args.kwargs.get("port") == 8000


def test_serve_custom_port(
    monkeypatch, mock_embedder, mock_server, mock_mcp, mock_read_meta
):
    monkeypatch.setattr("sys.argv", _argv("serve", "--http", "--port", "9000"))
    main()
    assert mock_mcp.run.call_args.kwargs.get("port") == 9000


def test_serve_custom_db(
    tmp_path, monkeypatch, mock_embedder, mock_server, mock_mcp, mock_read_meta
):
    db = tmp_path / "my.db"
    monkeypatch.setattr("sys.argv", _argv("serve", "--db", str(db)))
    main()
    db_arg = mock_server.configure.call_args.args[0]
    assert db_arg == db


# ---------------------------------------------------------------------------
# combined mode
# ---------------------------------------------------------------------------

def test_combined_indexes_when_db_absent(
    tmp_path, monkeypatch, mock_embedder, mock_summarizer, mock_server,
    mock_mcp, mock_run_index, mock_read_meta
):
    db = tmp_path / "index.db"  # does not exist
    monkeypatch.setattr("sys.argv", _argv("--db", str(db), str(tmp_path)))
    main()
    mock_run_index.assert_called_once()
    mock_mcp.run.assert_called_once()


def test_combined_skips_index_when_db_present(
    tmp_path, monkeypatch, mock_embedder, mock_summarizer, mock_server,
    mock_mcp, mock_run_index, mock_read_meta
):
    db = tmp_path / "index.db"
    db.touch()  # DB exists
    monkeypatch.setattr("sys.argv", _argv("--db", str(db), str(tmp_path)))
    main()
    mock_run_index.assert_not_called()
    mock_mcp.run.assert_called_once()


def test_combined_serve_only_when_no_paths(
    tmp_path, monkeypatch, mock_embedder, mock_server, mock_mcp,
    mock_run_index, mock_read_meta
):
    """No paths given → go straight to serve even if DB absent."""
    db = tmp_path / "index.db"
    monkeypatch.setattr("sys.argv", _argv("--db", str(db)))
    main()
    mock_run_index.assert_not_called()
    mock_mcp.run.assert_called_once()
