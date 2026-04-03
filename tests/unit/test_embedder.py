"""Unit tests for mcp_rag.embedder — FastEmbedder wrapping fastembed.

fastembed.TextEmbedding is monkeypatched so no model is downloaded.
"""

import math

import numpy as np
import pytest

from mcp_rag.embedder import DEFAULT_MODEL, FastEmbedder

_FAKE_DIM = 4
_FAKE_VEC = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)  # non-unit length


class MockTextEmbedding:
    """Drop-in replacement for fastembed.TextEmbedding."""

    def __init__(self, model_name: str = DEFAULT_MODEL, **kwargs):
        self.model_name = model_name

    def embed(self, texts):
        for _ in texts:
            yield _FAKE_VEC.copy()


@pytest.fixture(autouse=True)
def patch_fastembed(monkeypatch):
    monkeypatch.setattr("fastembed.TextEmbedding", MockTextEmbedding)


# ---------------------------------------------------------------------------
# Protocol attributes
# ---------------------------------------------------------------------------


def test_has_model_attribute():
    assert FastEmbedder().model == DEFAULT_MODEL


def test_custom_model_name_stored():
    assert FastEmbedder(model_name="my/model").model == "my/model"


def test_has_dim_attribute():
    assert FastEmbedder().dim == _FAKE_DIM


def test_dim_matches_embed_output_length():
    e = FastEmbedder()
    assert len(e.embed("probe")) == e.dim


# ---------------------------------------------------------------------------
# embed() return type and shape
# ---------------------------------------------------------------------------


def test_embed_returns_list():
    assert isinstance(FastEmbedder().embed("hello"), list)


def test_embed_returns_floats():
    result = FastEmbedder().embed("hello")
    assert all(isinstance(x, float) for x in result)


def test_embed_correct_length():
    assert len(FastEmbedder().embed("hello")) == _FAKE_DIM


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_embed_is_unit_length():
    v = FastEmbedder().embed("hello")
    norm = math.sqrt(sum(x * x for x in v))
    assert norm == pytest.approx(1.0, abs=1e-5)


def test_embed_normalises_non_unit_input():
    # MockTextEmbedding returns [1,2,3,4] (norm=sqrt(30)); output must be unit
    v = FastEmbedder().embed("anything")
    expected_norm = math.sqrt(30)
    assert v[0] == pytest.approx(1.0 / expected_norm, abs=1e-5)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_embed_is_deterministic():
    e = FastEmbedder()
    assert e.embed("hello") == e.embed("hello")
