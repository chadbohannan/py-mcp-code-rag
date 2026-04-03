"""Summarizer implementations for mcp-rag."""

from __future__ import annotations

import random
import time

from mcp_rag.models import SemanticUnit, relative_path

_MAX_TOKENS = 256
_RETRY_DELAYS = [1, 4, 16]  # seconds before each retry attempt
_JITTER = 0.2
_RETRY_STATUSES = frozenset({429, 529})

DEFAULT_OLLAMA_MODEL = "gemma4"
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
        )
        return response.message.content


def _build_prompt(unit: SemanticUnit) -> str:
    name_clause = f" named `{unit.unit_name}`" if unit.unit_name else ""

    path_clause = ""
    if unit.file_path is not None:
        base = unit.root if unit.root is not None else unit.file_path.parent
        path_clause = f"File: {relative_path(unit.file_path, base)}\n\n"

    return (
        f"You are indexing a codebase for semantic search. "
        f"Write a dense, searchable description of the following "
        f"{unit.unit_type}{name_clause}.\n\n"
        f"{path_clause}"
        f"Describe: what it does, what problem it solves, "
        f"key inputs/outputs/parameters, and any important constraints "
        f"or design patterns. Use natural language that will match "
        f"developer questions about this code.\n\n"
        f"{unit.content}"
    )
