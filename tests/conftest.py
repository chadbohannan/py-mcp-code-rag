"""
Shared test fixtures for all test tiers.

FakeEmbedder  — deterministic unit-length vectors, no fastembed dependency.
FakeSummarizer — deterministic strings + call log, no Anthropic API dependency.
"""

import hashlib
import math
import textwrap
from pathlib import Path

import pytest

from mcp_rag.models import SemanticUnit


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """
    Deterministic unit-length embedder.

    MD5-seeds a float vector of ``dim`` dimensions from the input text, then
    normalises it to unit length.  Same text always produces the same vector;
    different texts almost certainly produce different vectors.
    """

    model: str = "fake-model"

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        digest = hashlib.md5(text.encode()).digest()
        # Map each byte to a signed float in [-128, 127]
        floats = [float(digest[i % len(digest)]) - 128.0 for i in range(self.dim)]
        norm = math.sqrt(sum(x * x for x in floats))
        if norm == 0:
            return [1.0] + [0.0] * (self.dim - 1)
        return [x / norm for x in floats]


class FakeSummarizer:
    """
    Deterministic summarizer that keeps a call log.

    Returns a stable string derived from the unit so tests can assert on the
    stored summary text.  ``calls`` accumulates every unit passed in, enabling
    assertions like "the second run did not re-summarize unchanged units".
    """

    def __init__(self) -> None:
        self.calls: list[SemanticUnit] = []

    def summarize(self, unit: SemanticUnit) -> str:
        self.calls.append(unit)
        name = unit.unit_name or "(anonymous)"
        return f"summary:{unit.unit_type}:{name}:{unit.content_md5[:8]}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=4)


@pytest.fixture
def fake_summarizer() -> FakeSummarizer:
    return FakeSummarizer()


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """A fresh path (not yet created) for each test's index.db."""
    return tmp_path / "index.db"


@pytest.fixture
def sample_project_dir(tmp_path: Path) -> Path:
    """
    A small realistic project directory written to disk.

    Structure
    ---------
    src/models.py       — one class with two methods
    README.md           — two heading sections
    queries/report.sql  — small SQL file < 4 KB
    docs/skipped.bin    — 8 null bytes (binary → skipped)
    src/huge.sql        — 4,097 bytes (over 4 KB limit → skipped)
    """
    root = tmp_path / "project"

    # src/models.py
    (root / "src").mkdir(parents=True)
    (root / "src" / "models.py").write_text(
        textwrap.dedent("""\
            class User:
                \"\"\"Represents an application user.\"\"\"

                def __init__(self, name: str, email: str) -> None:
                    self.name = name
                    self.email = email

                def display_name(self) -> str:
                    return f"{self.name} <{self.email}>"
        """),
        encoding="utf-8",
    )

    # README.md
    (root / "README.md").write_text(
        textwrap.dedent("""\
            # Overview

            This project provides a sample codebase for testing the indexer.

            # Installation

            Run `uv sync` to install dependencies.
        """),
        encoding="utf-8",
    )

    # queries/report.sql
    (root / "queries").mkdir()
    (root / "queries" / "report.sql").write_text(
        "SELECT id, name FROM users WHERE active = 1;",
        encoding="utf-8",
    )

    # docs/skipped.bin — binary file
    (root / "docs").mkdir()
    (root / "docs" / "skipped.bin").write_bytes(b"\x00" * 8)

    # src/huge.sql — over 4 KB limit
    (root / "src" / "huge.sql").write_text(
        "-- x\n" * 820, encoding="utf-8"
    )  # ~4,920 bytes

    return root
