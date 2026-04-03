"""Core data models for mcp-rag."""

import hashlib
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


# ---------------------------------------------------------------------------
# Test-seam protocols
# ---------------------------------------------------------------------------


class Embedder(Protocol):
    dim: int
    model: str

    def embed(self, text: str) -> list[float]: ...


class Summarizer(Protocol):
    def summarize(self, unit: SemanticUnit) -> str: ...
