"""Real fastembed-backed Embedder implementation."""

from __future__ import annotations

import math
import warnings

DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5-Q"


class EmbedderLoadError(RuntimeError):
    """Raised when the embedding model cannot be loaded."""


class FastEmbedder:
    """Embedder backed by fastembed (ONNX Runtime, in-process).

    Normalizes every output vector to unit length so that cosine similarity
    reduces to a dot product and sqlite-vec's cosine distance is in [0, 2].
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from pathlib import Path

        from fastembed import TextEmbedding

        self.model = model_name
        cache_dir = Path.home() / ".cache" / "fastembed"
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self._fe = TextEmbedding(model_name, cache_dir=str(cache_dir))
            # Probe dimension once at construction time
            probe = next(iter(self._fe.embed(["probe"])))
            self.dim = len(probe)
        except Exception as exc:
            raise EmbedderLoadError(
                f"Failed to load embedding model '{model_name}'.\n"
                f"The model cache may be corrupt. Try clearing it:\n"
                f"  rm -rf {cache_dir}"
            ) from exc

    def embed(self, text: str) -> list[float]:
        raw = next(iter(self._fe.embed([text])))
        vec = [float(x) for x in raw]
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:
            return vec
        return [x / norm for x in vec]
