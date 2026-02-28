"""Real fastembed-backed Embedder implementation."""
from __future__ import annotations

import math

DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5-Q"


class FastEmbedder:
    """Embedder backed by fastembed (ONNX Runtime, in-process).

    Normalizes every output vector to unit length so that cosine similarity
    reduces to a dot product and sqlite-vec's cosine distance is in [0, 2].
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from fastembed import TextEmbedding

        self.model = model_name
        self._fe = TextEmbedding(model_name)
        # Probe dimension once at construction time
        probe = next(iter(self._fe.embed(["probe"])))
        self.dim = len(probe)

    def embed(self, text: str) -> list[float]:
        raw = next(iter(self._fe.embed([text])))
        vec = [float(x) for x in raw]
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:
            return vec
        return [x / norm for x in vec]
