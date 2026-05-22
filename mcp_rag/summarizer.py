"""Summarizer implementations for mcp-rag."""

from __future__ import annotations

import random
import sys
import time

from mcp_rag.models import SemanticUnit

_MAX_TOKENS = 128
_RETRY_DELAYS = [1, 2, 4]  # seconds before each retry attempt
_JITTER = 0.2
_RETRY_STATUSES = frozenset({429, 529})

DEFAULT_OLLAMA_MODEL = "gemma4:latest"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# Ordered preference list used when the requested model isn't installed.
# First match wins; if none match, the first available model on the server
# is used as a last resort.
FALLBACK_OLLAMA_MODELS: tuple[str, ...] = (
    "gemma4:latest",
    "gemma4:e4b",
    "gemma4:e2b",
    "qwen3.5:4b",
    "qwen3.5:2b",
    "qwen3:1.7b",
    "llama3.2:3b",
    "llama3.2:1b",
)


class AnthropicSummarizer:
    """Summarizer backed by the Anthropic API (claude-haiku).

    Retries on 429, 529, and 5xx responses with exponential backoff ±20%
    jitter. Other errors are raised immediately.
    """

    MODEL = "claude-haiku-4-5-20251001"

    def __init__(self) -> None:
        import anthropic

        self._client = anthropic.Anthropic()

    def summarize(self, unit: SemanticUnit) -> str:
        prompt = _build_prompt(unit)
        last_exc: Exception | None = None

        for attempt in range(len(_RETRY_DELAYS) + 1):
            if last_exc is not None:
                delay = _RETRY_DELAYS[attempt - 1]
                jitter = 1.0 + random.uniform(-_JITTER, _JITTER)
                time.sleep(delay * jitter)
            try:
                response = self._client.messages.create(
                    model=self.MODEL,
                    max_tokens=_MAX_TOKENS,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                retryable = status is not None and (
                    status in _RETRY_STATUSES or status >= 500
                )
                if not retryable or attempt == len(_RETRY_DELAYS):
                    raise
                last_exc = exc

        raise AssertionError("unreachable")  # pragma: no cover


class OllamaSummarizer:
    """Summarizer backed by a local Ollama server."""

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
    ) -> None:
        import ollama  # lazy import — optional dependency

        self._client = ollama.Client(host=host)
        self._model = _resolve_ollama_model(self._client, model)

    def summarize(self, unit: SemanticUnit) -> str:
        response = self._client.chat(
            model=self._model,
            messages=[{"role": "user", "content": _build_prompt(unit)}],
            options={"num_predict": _MAX_TOKENS},
            think=False,
        )
        content = response.message.content
        # Some thinking models (e.g. gemma4) may put output in the thinking
        # field and return empty content even with think=False; fall back.
        if not content and getattr(response.message, "thinking", None):
            content = response.message.thinking
        return content or ""


def _list_available_models(client) -> list[str]:
    """Return the list of model names installed on the Ollama server.

    Returns [] if the server is unreachable or returns an unexpected shape;
    callers treat that as "skip validation and trust the requested model."
    """
    try:
        resp = client.list()
        models = getattr(resp, "models", None) or []
        names: list[str] = []
        for m in models:
            name = getattr(m, "model", None) or getattr(m, "name", None)
            if isinstance(name, str):
                names.append(name)
        return names
    except Exception:
        return []


def _resolve_ollama_model(client, requested: str) -> str:
    """Pick a usable model, falling back through FALLBACK_OLLAMA_MODELS.

    1. If the requested model is installed, use it.
    2. Otherwise try each name in FALLBACK_OLLAMA_MODELS in order.
    3. Otherwise use the first model the server reports.
    4. If the server reports no models at all, raise.

    If listing fails entirely (empty list returned), we trust `requested`
    and let any subsequent chat() call surface the real error.
    """
    available = _list_available_models(client)
    if not available:
        return requested  # can't validate; defer to chat() to error out

    available_set = set(available)
    if requested in available_set:
        return requested

    for candidate in FALLBACK_OLLAMA_MODELS:
        if candidate in available_set:
            print(
                f"ollama: requested model {requested!r} not installed; "
                f"falling back to {candidate!r}",
                file=sys.stderr,
            )
            return candidate

    chosen = available[0]
    print(
        f"ollama: requested model {requested!r} not installed and no "
        f"preferred fallback present; using {chosen!r}. "
        f"Available: {', '.join(available)}",
        file=sys.stderr,
    )
    return chosen


_STYLE_SHORT = "Be direct, no preamble. 2 sentences max. No headings, no bullet points."
_STYLE_DENSE = (
    "2-3 sentences, terse and dense. No preamble, no headings, no bullet points."
)
_FALLBACK = "Summarize this {unit_type}. " + _STYLE_SHORT

_PROMPTS: dict[str, str] = {
    "directory": f"Summarize this directory's purpose and what it contains based on its files and subdirectories below. {_STYLE_DENSE}",
    "module": f"Summarize this file's purpose, key exports, and role relative to the modules it depends on. {_STYLE_DENSE}",
    "function": f"What does this function compute or perform based on its signature and body. {_STYLE_SHORT}",
    "method": f"What does this method do based on its signature and body, what state does it read or modify. {_STYLE_SHORT}",
    "class": f"What responsibility does this class encapsulate based on its definition. {_STYLE_SHORT}",
    "struct": f"Describe this struct's purpose, its key fields, and what domain it models. {_STYLE_SHORT}",
    "interface": f"Describe what contract this interface defines and what operations it requires. {_STYLE_SHORT}",
    "enum": f"What concept or domain do these enum values model. {_STYLE_SHORT}",
}


def _build_prompt(unit: SemanticUnit) -> str:
    instruction = _PROMPTS.get(
        unit.unit_type, _FALLBACK.format(unit_type=unit.unit_type)
    )
    return f"{instruction}\n\n{unit.content}"
