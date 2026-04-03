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
import warnings
from pathlib import Path

from mcp_rag.models import SemanticUnit

logger = logging.getLogger(__name__)

_GO_PARSER = Path(__file__).parent / "go_parser" / "main.go"

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
                                unit_name=f"{node.name}:{item.name}",
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
    # Track heading stack for hierarchical names: [(level, text), ...]
    heading_stack: list[tuple[int, str]] = []

    for line in source.splitlines(keepends=True):
        if line.startswith("#"):
            # Flush previous section
            content = "".join(current_lines).strip()
            if content:
                sections.append((current_offset, current_name, current_lines[:]))
            # Parse heading level and text
            stripped = line.lstrip("#")
            level = len(line) - len(stripped)
            heading_text = stripped.strip()
            # Pop headings at same or deeper level
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            # Build hierarchical name
            if heading_stack:
                current_name = ":".join(h[1] for h in heading_stack) + ":" + heading_text
            else:
                current_name = heading_text
            heading_stack.append((level, heading_text))
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
# C / C++
# ---------------------------------------------------------------------------

_C_EXTENSIONS = frozenset({".c", ".h"})
_CPP_EXTENSIONS = frozenset({".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx", ".ino"})


def _get_ts_c_language():
    """Return the tree-sitter C Language object, or None if unavailable."""
    try:
        import tree_sitter
        import tree_sitter_c

        return tree_sitter.Language(tree_sitter_c.language())
    except Exception:
        return None


def _get_ts_cpp_language():
    """Return the tree-sitter C++ Language object, or None if unavailable."""
    try:
        import tree_sitter
        import tree_sitter_cpp

        return tree_sitter.Language(tree_sitter_cpp.language())
    except Exception:
        return None


def _ts_node_text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode(errors="replace")


def _ts_find_child_by_type(node, *type_names: str):
    """Return the first child matching any of the given type names, or None."""
    for child in node.children:
        if child.type in type_names:
            return child
    return None


def _extract_c_cpp_units(tree, source_bytes: bytes) -> list[SemanticUnit]:
    """Walk a tree-sitter parse tree and extract semantic units for C/C++."""
    units: list[SemanticUnit] = []

    def _walk(node, class_name: str | None = None):
        for child in node.children:
            if child.type == "function_definition":
                declarator = _ts_find_child_by_type(child, "function_declarator")
                if declarator is None:
                    declarator = _ts_find_child_by_type(child, "pointer_declarator")
                name = None
                if declarator is not None:
                    ident = _ts_find_child_by_type(
                        declarator, "identifier", "field_identifier",
                        "destructor_name", "qualified_identifier",
                    )
                    if ident is not None:
                        name = _ts_node_text(ident, source_bytes)

                if class_name:
                    unit_type = "method"
                    unit_name = f"{class_name}:{name}" if name else class_name
                else:
                    unit_type = "function"
                    unit_name = name

                units.append(
                    SemanticUnit(
                        unit_type=unit_type,
                        unit_name=unit_name,
                        content=_ts_node_text(child, source_bytes),
                        char_offset=child.start_byte,
                    )
                )

            elif child.type == "struct_specifier":
                body = _ts_find_child_by_type(child, "field_declaration_list")
                if body is None:
                    continue
                name_node = _ts_find_child_by_type(child, "type_identifier")
                name = _ts_node_text(name_node, source_bytes) if name_node else None
                units.append(
                    SemanticUnit(
                        unit_type="struct",
                        unit_name=name,
                        content=_ts_node_text(child, source_bytes),
                        char_offset=child.start_byte,
                    )
                )

            elif child.type == "class_specifier":
                body = _ts_find_child_by_type(child, "field_declaration_list")
                if body is None:
                    continue
                name_node = _ts_find_child_by_type(child, "type_identifier")
                name = _ts_node_text(name_node, source_bytes) if name_node else None
                units.append(
                    SemanticUnit(
                        unit_type="class",
                        unit_name=name,
                        content=_ts_node_text(child, source_bytes),
                        char_offset=child.start_byte,
                    )
                )
                if body is not None:
                    _walk(body, class_name=name)

            elif child.type == "enum_specifier":
                body = _ts_find_child_by_type(child, "enumerator_list")
                if body is None:
                    continue
                name_node = _ts_find_child_by_type(child, "type_identifier")
                name = _ts_node_text(name_node, source_bytes) if name_node else None
                units.append(
                    SemanticUnit(
                        unit_type="enum",
                        unit_name=name,
                        content=_ts_node_text(child, source_bytes),
                        char_offset=child.start_byte,
                    )
                )

            elif child.type in (
                "declaration",
                "declaration_list",
                "type_definition",
                "template_declaration",
                "namespace_definition",
                "linkage_specification",
                "preproc_ifdef",
                "preproc_ifndef",
                "preproc_if",
                "preproc_else",
                "preproc_elif",
            ):
                _walk(child, class_name=class_name)

    _walk(tree.root_node)
    return sorted(units, key=lambda u: u.char_offset)


def parse_c(source: str) -> list[SemanticUnit]:
    """Parse C source into SemanticUnits using tree-sitter.

    Returns [] with a warning if tree-sitter-c is not installed.
    """
    if not source:
        return []

    lang = _get_ts_c_language()
    if lang is None:
        warnings.warn(
            "tree-sitter-c not installed — .c/.h files will not be indexed",
            stacklevel=2,
        )
        return []

    import tree_sitter

    parser = tree_sitter.Parser()
    parser.language = lang
    source_bytes = source.encode()
    tree = parser.parse(source_bytes)
    return _extract_c_cpp_units(tree, source_bytes)


def parse_cpp(source: str) -> list[SemanticUnit]:
    """Parse C++ source into SemanticUnits using tree-sitter.

    Returns [] with a warning if tree-sitter-cpp is not installed.
    """
    if not source:
        return []

    lang = _get_ts_cpp_language()
    if lang is None:
        warnings.warn(
            "tree-sitter-cpp not installed — C++ files will not be indexed",
            stacklevel=2,
        )
        return []

    import tree_sitter

    parser = tree_sitter.Parser()
    parser.language = lang
    source_bytes = source.encode()
    tree = parser.parse(source_bytes)
    return _extract_c_cpp_units(tree, source_bytes)


# ---------------------------------------------------------------------------
# Go
# ---------------------------------------------------------------------------


def parse_go(path: Path) -> list[SemanticUnit]:
    """Parse a Go source file into SemanticUnits via the bundled go_parser helper.

    Requires `go` in PATH.  Returns [] (with a one-time warning) if `go` is
    absent or if the helper exits non-zero.
    """
    if shutil.which("go") is None:
        warnings.warn(
            "'go' not found in PATH — .go files will not be indexed",
            stacklevel=2,
        )
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
    if suffix in _C_EXTENSIONS:
        return parse_c(path.read_text(encoding="utf-8", errors="replace"))
    if suffix in _CPP_EXTENSIONS:
        return parse_cpp(path.read_text(encoding="utf-8", errors="replace"))
    if suffix in (".md", ".mdx"):
        return parse_markdown(path.read_text(encoding="utf-8", errors="replace"))
    if suffix == ".sql":
        return parse_sql(path.read_text(encoding="utf-8", errors="replace"))

    logger.debug("skipping %s: unsupported extension %r", path, suffix)
    return []
