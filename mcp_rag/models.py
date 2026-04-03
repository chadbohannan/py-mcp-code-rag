"""Core data models for mcp-rag."""

import hashlib
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class SemanticUnit:
    """A single indexable unit extracted from a source file.

    ``content_md5`` is derived automatically from ``content`` at construction
    time and used for per-unit change detection during incremental indexing.
    ``summary`` is filled in by the Summarizer after parsing.
    ``file_path`` and ``root`` are set by the indexer after parsing so that
    summarizers can include path context in their prompts.
    """

    unit_type: str  # function | class | method | paragraph | sql | ...
    unit_name: str | None  # identifier; None if not applicable
    content: str  # original source, returned to MCP callers
    char_offset: int  # Unicode character offset in source file
    content_md5: str = field(init=False)
    summary: str = ""
    file_path: Path | None = None
    root: Path | None = None

    def __post_init__(self) -> None:
        self.content_md5 = hashlib.md5(self.content.encode()).hexdigest()

    @property
    def qualified_path(self) -> str:
        """Build the full qualified path: ``relative/file.py:Class:method``.

        The file's relative path uses ``/`` separators; the unit hierarchy
        uses ``:`` so there is no ambiguity at the file boundary.
        """
        parts: list[str] = []
        if self.file_path is not None:
            parts.append(str(relative_path(self.file_path, self.root)))
        if self.unit_name:
            parts.append(self.unit_name)
        return ":".join(parts)


# ---------------------------------------------------------------------------
# Test-seam protocols
# ---------------------------------------------------------------------------


class Embedder(Protocol):
    dim: int
    model: str

    def embed(self, text: str) -> list[float]: ...


class Summarizer(Protocol):
    def summarize(self, unit: SemanticUnit) -> str: ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def relative_path(file_path: Path, root: Path | None) -> Path:
    """Return *file_path* relative to *root*, falling back to *file_path* itself."""
    if root is None:
        return file_path
    try:
        return file_path.relative_to(root)
    except ValueError:
        return file_path


def encode_embedding(embedding: list[float]) -> bytes:
    """Pack a float vector into the binary format expected by sqlite-vec."""
    return struct.pack(f"{len(embedding)}f", *embedding)
