"""File parsers for mcp-rag.

Each parser takes source text (a str) and returns a list of SemanticUnit
objects in source order.  parse_file dispatches by extension after a binary
check.
"""

import ast
import json
import logging
import shutil
import subprocess
from pathlib import Path

from mcp_rag.models import SemanticUnit

logger = logging.getLogger(__name__)

_GO_PARSER = Path(__file__).parent / "go_parser" / "main.go"
_go_warned = False  # emit the "go not in PATH" warning at most once

_SQL_SIZE_LIMIT = 4096  # bytes; files strictly over this are skipped
_BINARY_CHECK_BYTES = 512


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------


def parse_python(source: str) -> list[SemanticUnit]:
    """Parse a Python source string into SemanticUnits.

    Extracts module-level functions, classes, and methods (immediate children
    of classes).  Nested functions and nested classes are not extracted.
    Returns [] on syntax error without raising.
    """
    if not source:
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    lines = source.splitlines(keepends=True)

    def _char_offset(node: ast.AST) -> int:
        return sum(len(lines[i]) for i in range(node.lineno - 1)) + node.col_offset  # type: ignore[attr-defined]

    units: list[SemanticUnit] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            content = ast.get_source_segment(source, node) or ""
            if content:
                units.append(
                    SemanticUnit(
                        unit_type="function",
                        unit_name=node.name,
                        content=content,
                        char_offset=_char_offset(node),
                    )
                )

        elif isinstance(node, ast.ClassDef):
            content = ast.get_source_segment(source, node) or ""
            if content:
                units.append(
                    SemanticUnit(
                        unit_type="class",
                        unit_name=node.name,
                        content=content,
                        char_offset=_char_offset(node),
                    )
                )
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_content = ast.get_source_segment(source, item) or ""
                    if method_content:
                        units.append(
                            SemanticUnit(
                                unit_type="method",
                                unit_name=item.name,
                                content=method_content,
                                char_offset=_char_offset(item),
                            )
                        )

    return sorted(units, key=lambda u: u.char_offset)


# ---------------------------------------------------------------------------
# Markdown / MDX
# ---------------------------------------------------------------------------


def parse_markdown(source: str) -> list[SemanticUnit]:
    """Parse a Markdown source string into SemanticUnits.

    Each ATX heading (#, ##, …) starts a new section.  Text before the first
    heading is emitted as a preamble unit (unit_name=None).  Empty sections
    are dropped.  unit_name is the heading text stripped of leading # and
    surrounding whitespace.
    """
    if not source:
        return []

    sections: list[tuple[int, str | None, list[str]]] = []  # (offset, name, lines)
    current_offset = 0
    current_name: str | None = None
    current_lines: list[str] = []
    char_pos = 0

    for line in source.splitlines(keepends=True):
        if line.startswith("#"):
            # Flush previous section
            content = "".join(current_lines).strip()
            if content:
                sections.append((current_offset, current_name, current_lines[:]))
            # Start new section at this heading
            current_name = line.lstrip("#").strip()
            current_offset = char_pos
            current_lines = [line]
        else:
            current_lines.append(line)
        char_pos += len(line)

    # Flush final section
    content = "".join(current_lines).strip()
    if content:
        sections.append((current_offset, current_name, current_lines[:]))

    return [
        SemanticUnit(
            unit_type="paragraph",
            unit_name=name,
            content="".join(lines).strip(),
            char_offset=offset,
        )
        for offset, name, lines in sections
        if "".join(lines).strip()
    ]


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------


def parse_sql(source: str) -> list[SemanticUnit]:
    """Parse a SQL source string into a single SemanticUnit.

    Returns [] if the source exceeds _SQL_SIZE_LIMIT bytes (UTF-8 encoded).
    """
    if len(source.encode()) > _SQL_SIZE_LIMIT:
        return []
    return [
        SemanticUnit(
            unit_type="sql",
            unit_name=None,
            content=source,
            char_offset=0,
        )
    ]


# ---------------------------------------------------------------------------
# Go
# ---------------------------------------------------------------------------


def parse_go(path: Path) -> list[SemanticUnit]:
    """Parse a Go source file into SemanticUnits via the bundled go_parser helper.

    Requires `go` in PATH.  Returns [] (with a one-time warning) if `go` is
    absent or if the helper exits non-zero.
    """
    global _go_warned
    if shutil.which("go") is None:
        if not _go_warned:
            logger.warning("'go' not found in PATH — .go files will not be indexed")
            _go_warned = True
        return []

    # "--" separates go source files from program arguments so that `go run`
    # does not treat the target .go file as a source file to compile.
    result = subprocess.run(
        ["go", "run", str(_GO_PARSER), "--", str(path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.warning("skipping %s — go parser error (see stderr)", path)
        return []

    data = json.loads(result.stdout)
    units = []
    for item in data:
        units.append(
            SemanticUnit(
                unit_type=item["unit_type"],
                unit_name=item.get("unit_name"),
                content=item["content"],
                char_offset=item["char_offset"],
            )
        )
    return units


# ---------------------------------------------------------------------------
# File dispatcher
# ---------------------------------------------------------------------------


def _is_binary(path: Path) -> bool:
    """Return True if the first 512 bytes of the file contain a null byte."""
    try:
        with open(path, "rb") as fh:
            chunk = fh.read(_BINARY_CHECK_BYTES)
        return b"\x00" in chunk
    except OSError:
        return True


def parse_file(path: Path) -> list[SemanticUnit]:
    """Dispatch to the appropriate parser based on file extension.

    Returns [] for binary files and unsupported extensions without raising.
    """
    if _is_binary(path):
        return []

    suffix = path.suffix.lower()

    if suffix == ".py":
        return parse_python(path.read_text(encoding="utf-8", errors="replace"))
    if suffix == ".go":
        return parse_go(path)
    if suffix in (".md", ".mdx"):
        return parse_markdown(path.read_text(encoding="utf-8", errors="replace"))
    if suffix == ".sql":
        return parse_sql(path.read_text(encoding="utf-8", errors="replace"))

    logger.debug("skipping %s: unsupported extension %r", path, suffix)
    return []
