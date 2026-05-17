"""Unit tests for the CLI entry point (mcp_rag.__main__).

All external I/O (run_index, FastEmbedder, AnthropicSummarizer) is
monkeypatched so no files, network, or servers are touched.
"""

from pathlib import Path
from unittest.mock import MagicMock

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
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-tests")
    inst = MagicMock()
    cls = MagicMock(return_value=inst)
    monkeypatch.setattr("mcp_rag.__main__.AnthropicSummarizer", cls)
    return inst


@pytest.fixture
def mock_ollama_summarizer(monkeypatch):
    inst = MagicMock()
    cls = MagicMock(return_value=inst)
    monkeypatch.setattr("mcp_rag.__main__.OllamaSummarizer", cls)
    return cls, inst


@pytest.fixture
def mock_run_index(monkeypatch):
    m = MagicMock()
    monkeypatch.setattr("mcp_rag.__main__.run_index", m)
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
# index subcommand — summarizer selection
# ---------------------------------------------------------------------------


def test_index_default_summarizer_is_ollama(
    tmp_path,
    monkeypatch,
    mock_embedder,
    mock_ollama_summarizer,
    mock_run_index,
):
    monkeypatch.setattr("sys.argv", _argv("index", str(tmp_path)))
    main()
    mock_run_index.assert_called_once()
    _, inst = mock_ollama_summarizer
    summarizer_arg = mock_run_index.call_args.kwargs["summarizer"]
    assert summarizer_arg is inst


def test_index_ollama_summarizer_flag(
    tmp_path, monkeypatch, mock_embedder, mock_ollama_summarizer, mock_run_index
):
    monkeypatch.setattr(
        "sys.argv", _argv("index", "--summarizer", "ollama", str(tmp_path))
    )
    main()
    mock_run_index.assert_called_once()
    summarizer_arg = mock_run_index.call_args.kwargs["summarizer"]
    assert summarizer_arg is mock_ollama_summarizer[1]


def test_index_ollama_model_flag(
    tmp_path, monkeypatch, mock_embedder, mock_ollama_summarizer, mock_run_index
):
    monkeypatch.setattr(
        "sys.argv",
        _argv(
            "index",
            "--summarizer",
            "ollama",
            "--ollama-model",
            "mymodel",
            str(tmp_path),
        ),
    )
    main()
    cls, _ = mock_ollama_summarizer
    call_kwargs = cls.call_args.kwargs
    assert call_kwargs["model"] == "mymodel"


def test_index_no_api_key_exits_1(tmp_path, monkeypatch, mock_embedder, mock_run_index):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "sys.argv", _argv("index", "--summarizer", "anthropic", str(tmp_path))
    )
    with pytest.raises(SystemExit) as ei:
        main()
    assert ei.value.code == 1
    mock_run_index.assert_not_called()


def test_index_ollama_no_api_key_required(
    tmp_path, monkeypatch, mock_embedder, mock_ollama_summarizer, mock_run_index
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "sys.argv", _argv("index", "--summarizer", "ollama", str(tmp_path))
    )
    main()  # must not raise or exit
    mock_run_index.assert_called_once()
