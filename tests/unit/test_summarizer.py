"""Unit tests for mcp_rag.summarizer — AnthropicSummarizer.

anthropic.Anthropic is monkeypatched so no API calls are made.
time.sleep and random.uniform are patched to keep tests instant.
"""

import pytest
from unittest.mock import MagicMock

from mcp_rag.models import SemanticUnit
from mcp_rag.summarizer import (
    AnthropicSummarizer,
    OllamaSummarizer,
)

MODEL = AnthropicSummarizer.MODEL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(unit_type="function", unit_name="my_func", content="def my_func(): pass"):
    return SemanticUnit(
        unit_type=unit_type, unit_name=unit_name, content=content, char_offset=0
    )


def _response(text="summary text"):
    resp = MagicMock()
    resp.content[0].text = text
    return resp


def _status_error(status_code: int) -> Exception:
    exc = Exception(f"HTTP {status_code}")
    exc.status_code = status_code
    return exc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr("anthropic.Anthropic", lambda: client)
    return client


@pytest.fixture
def summarizer(mock_client):
    return AnthropicSummarizer()


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr("mcp_rag.summarizer.time.sleep", lambda _: None)
    monkeypatch.setattr("mcp_rag.summarizer.random.uniform", lambda a, b: 0.0)


# ---------------------------------------------------------------------------
# API call shape
# ---------------------------------------------------------------------------


def test_uses_correct_model(mock_client, summarizer):
    mock_client.messages.create.return_value = _response()
    summarizer.summarize(_unit())
    kwargs = mock_client.messages.create.call_args.kwargs
    assert kwargs["model"] == MODEL


def test_max_tokens_is_128(mock_client, summarizer):
    mock_client.messages.create.return_value = _response()
    summarizer.summarize(_unit())
    kwargs = mock_client.messages.create.call_args.kwargs
    assert kwargs["max_tokens"] == 128


def test_message_role_is_user(mock_client, summarizer):
    mock_client.messages.create.return_value = _response()
    summarizer.summarize(_unit())
    kwargs = mock_client.messages.create.call_args.kwargs
    assert kwargs["messages"][0]["role"] == "user"


# ---------------------------------------------------------------------------
# Prompt content
# ---------------------------------------------------------------------------


def test_prompt_includes_unit_type(mock_client, summarizer):
    mock_client.messages.create.return_value = _response()
    summarizer.summarize(_unit(unit_type="class"))
    prompt = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "class" in prompt


def test_prompt_includes_source_content(mock_client, summarizer):
    mock_client.messages.create.return_value = _response()
    summarizer.summarize(_unit(content="def foo(): return 42"))
    prompt = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "def foo(): return 42" in prompt


def test_prompt_handles_anonymous_unit(mock_client, summarizer):
    """unit_name=None must not crash prompt construction."""
    mock_client.messages.create.return_value = _response()
    unit = SemanticUnit(
        unit_type="sql", unit_name=None, content="SELECT 1", char_offset=0
    )
    summarizer.summarize(unit)  # must not raise


# ---------------------------------------------------------------------------
# Return value
# ---------------------------------------------------------------------------


def test_returns_response_text(mock_client, summarizer):
    mock_client.messages.create.return_value = _response("my generated summary")
    assert summarizer.summarize(_unit()) == "my generated summary"


# ---------------------------------------------------------------------------
# Retry — success after transient errors
# ---------------------------------------------------------------------------


def test_retries_on_429(mock_client, summarizer, no_sleep):
    mock_client.messages.create.side_effect = [
        _status_error(429),
        _response("ok"),
    ]
    assert summarizer.summarize(_unit()) == "ok"
    assert mock_client.messages.create.call_count == 2


def test_retries_on_529(mock_client, summarizer, no_sleep):
    mock_client.messages.create.side_effect = [
        _status_error(529),
        _response("ok"),
    ]
    assert summarizer.summarize(_unit()) == "ok"
    assert mock_client.messages.create.call_count == 2


def test_retries_on_500(mock_client, summarizer, no_sleep):
    mock_client.messages.create.side_effect = [
        _status_error(500),
        _response("ok"),
    ]
    assert summarizer.summarize(_unit()) == "ok"
    assert mock_client.messages.create.call_count == 2


def test_retries_up_to_three_times(mock_client, summarizer, no_sleep):
    mock_client.messages.create.side_effect = [
        _status_error(429),
        _status_error(429),
        _status_error(429),
        _response("ok"),
    ]
    assert summarizer.summarize(_unit()) == "ok"
    assert mock_client.messages.create.call_count == 4


# ---------------------------------------------------------------------------
# Retry — exhaustion and non-retryable errors
# ---------------------------------------------------------------------------


def test_raises_after_max_retries_exhausted(mock_client, summarizer, no_sleep):
    exc = _status_error(429)
    mock_client.messages.create.side_effect = exc
    with pytest.raises(Exception) as ei:
        summarizer.summarize(_unit())
    assert ei.value is exc
    assert mock_client.messages.create.call_count == 4  # 1 initial + 3 retries


def test_raises_immediately_on_non_retryable_4xx(mock_client, summarizer, no_sleep):
    exc = _status_error(400)
    mock_client.messages.create.side_effect = exc
    with pytest.raises(Exception) as ei:
        summarizer.summarize(_unit())
    assert ei.value is exc
    assert mock_client.messages.create.call_count == 1


def test_raises_immediately_on_401(mock_client, summarizer, no_sleep):
    exc = _status_error(401)
    mock_client.messages.create.side_effect = exc
    with pytest.raises(Exception):
        summarizer.summarize(_unit())
    assert mock_client.messages.create.call_count == 1


# ---------------------------------------------------------------------------
# OllamaSummarizer fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ollama_client(monkeypatch):
    import sys

    client = MagicMock()
    ollama_mod = MagicMock()
    ollama_mod.Client.return_value = client
    monkeypatch.setitem(sys.modules, "ollama", ollama_mod)
    return client


@pytest.fixture
def ollama_summarizer(mock_ollama_client):
    return OllamaSummarizer(model="test-model")


# ---------------------------------------------------------------------------
# OllamaSummarizer tests
# ---------------------------------------------------------------------------


def test_ollama_uses_specified_model(mock_ollama_client, ollama_summarizer):
    mock_ollama_client.chat.return_value = MagicMock(
        message=MagicMock(content="summary")
    )
    ollama_summarizer.summarize(_unit())
    call_kwargs = mock_ollama_client.chat.call_args.kwargs
    assert call_kwargs["model"] == "test-model"


def test_ollama_message_role_is_user(mock_ollama_client, ollama_summarizer):
    mock_ollama_client.chat.return_value = MagicMock(
        message=MagicMock(content="summary")
    )
    ollama_summarizer.summarize(_unit())
    call_kwargs = mock_ollama_client.chat.call_args.kwargs
    assert call_kwargs["messages"][0]["role"] == "user"


def test_ollama_prompt_includes_unit_type(mock_ollama_client, ollama_summarizer):
    mock_ollama_client.chat.return_value = MagicMock(
        message=MagicMock(content="summary")
    )
    ollama_summarizer.summarize(_unit(unit_type="class"))
    call_kwargs = mock_ollama_client.chat.call_args.kwargs
    assert "class" in call_kwargs["messages"][0]["content"]


def test_ollama_prompt_includes_source_content(mock_ollama_client, ollama_summarizer):
    mock_ollama_client.chat.return_value = MagicMock(
        message=MagicMock(content="summary")
    )
    ollama_summarizer.summarize(_unit(content="def foo(): return 42"))
    call_kwargs = mock_ollama_client.chat.call_args.kwargs
    assert "def foo(): return 42" in call_kwargs["messages"][0]["content"]


def test_ollama_returns_response_text(mock_ollama_client, ollama_summarizer):
    mock_ollama_client.chat.return_value = MagicMock(
        message=MagicMock(content="ollama summary")
    )
    assert ollama_summarizer.summarize(_unit()) == "ollama summary"


def test_ollama_default_model_is_gemma4(monkeypatch):
    import sys

    client = MagicMock()
    ollama_mod = MagicMock()
    ollama_mod.Client.return_value = client
    monkeypatch.setitem(sys.modules, "ollama", ollama_mod)
    s = OllamaSummarizer()
    client.chat.return_value = MagicMock(message=MagicMock(content="ok"))
    s.summarize(_unit())
    assert client.chat.call_args.kwargs["model"] == "gemma4:latest"


def test_ollama_handles_anonymous_unit(mock_ollama_client, ollama_summarizer):
    """unit_name=None must not crash prompt construction."""
    mock_ollama_client.chat.return_value = MagicMock(message=MagicMock(content="ok"))
    unit = SemanticUnit(
        unit_type="sql", unit_name=None, content="SELECT 1", char_offset=0
    )
    ollama_summarizer.summarize(unit)  # must not raise
