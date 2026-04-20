"""Summarizer implementations for mcp-rag."""

from __future__ import annotations

import random
import time

from mcp_rag.models import SemanticUnit

_MAX_TOKENS = 128
_RETRY_DELAYS = [1, 2, 4]  # seconds before each retry attempt
_JITTER = 0.2
_RETRY_STATUSES = frozenset({429, 529})

DEFAULT_OLLAMA_MODEL = "gemma4:latest"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


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
        self._model = model

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


def _build_prompt(unit: SemanticUnit) -> str:
    if unit.unit_type == "directory":
        return (
            "Summarize this directory's purpose and what it contains based on its "
            "files and subdirectories below. 2-3 sentences, terse and dense. "
            "No preamble, no headings, no bullet points.\n\n"
            f"{unit.content}"
        )
    if unit.unit_type == "module":
        return (
            "Summarize this file's purpose, key exports, and role relative to the "
            "modules it depends on. 2-3 sentences, terse and dense. "
            "No preamble, no headings, no bullet points.\n\n"
            f"{unit.content}"
        )
    return (
        f"Summarize this {unit.unit_type} in 2-3 sentences. "
        f"Say what it does and why using terse, dense natural language a developer would "
        f"search for. No preamble, no headings, no bullet points.\n\n"
        f"{unit.content}"
    )
