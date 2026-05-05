"""File parsers for mcp-rag.

Each parser takes source text (a str) and returns a list of SemanticUnit
objects in source order.  parse_file dispatches by extension after a binary
check.
"""

import ast
import logging
import re
import warnings
from pathlib import Path

from mcp_rag.models import SemanticUnit

logger = logging.getLogger(__name__)

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
    fence_marker: str | None = None  # tracks which delimiter opened the block

    for line in source.splitlines(keepends=True):
        stripped_line = line.strip()
        if fence_marker is None:
            if stripped_line.startswith("```"):
                fence_marker = "```"
            elif stripped_line.startswith("~~~"):
                fence_marker = "~~~"
        elif (
            stripped_line.startswith(fence_marker)
            and stripped_line.rstrip(fence_marker[0]) == ""
        ):
            fence_marker = None
        if line.startswith("#") and fence_marker is None:
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
                current_name = (
                    ":".join(h[1] for h in heading_stack) + ":" + heading_text
                )
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

    units = []
    for offset, name, lines in sections:
        content = "".join(lines).strip()
        if not content:
            continue
        # Derive heading level from the number of '#' chars on the first line.
        first_line = lines[0] if lines else ""
        level = len(first_line) - len(first_line.lstrip("#"))
        unit_type = f"h{level}" if level else "paragraph"
        units.append(
            SemanticUnit(
                unit_type=unit_type,
                unit_name=name,
                content=content,
                char_offset=offset,
            )
        )
    return units


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
                        declarator,
                        "identifier",
                        "field_identifier",
                        "destructor_name",
                        "qualified_identifier",
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
# JavaScript / TypeScript
# ---------------------------------------------------------------------------

_JS_EXTENSIONS = frozenset({".js", ".jsx", ".mjs", ".cjs"})
_TS_EXTENSIONS = frozenset({".ts", ".tsx", ".mts", ".cts"})


def _get_ts_javascript_language():
    """Return the tree-sitter JavaScript Language object, or None if unavailable."""
    try:
        import tree_sitter
        import tree_sitter_javascript

        return tree_sitter.Language(tree_sitter_javascript.language())
    except Exception:
        return None


def _get_ts_typescript_language():
    """Return the tree-sitter TypeScript Language object, or None if unavailable."""
    try:
        import tree_sitter
        import tree_sitter_typescript

        return tree_sitter.Language(tree_sitter_typescript.language_typescript())
    except Exception:
        return None


def _get_ts_tsx_language():
    """Return the tree-sitter TSX Language object, or None if unavailable."""
    try:
        import tree_sitter
        import tree_sitter_typescript

        return tree_sitter.Language(tree_sitter_typescript.language_tsx())
    except Exception:
        return None


def _extract_js_ts_units(tree, source_bytes: bytes) -> list[SemanticUnit]:
    """Walk a tree-sitter parse tree and extract semantic units for JS/TS."""
    units: list[SemanticUnit] = []

    def _walk(node, class_name: str | None = None):
        for child in node.children:
            if child.type == "function_declaration":
                name_node = _ts_find_child_by_type(child, "identifier")
                name = _ts_node_text(name_node, source_bytes) if name_node else None
                units.append(
                    SemanticUnit(
                        unit_type="function",
                        unit_name=name,
                        content=_ts_node_text(child, source_bytes),
                        char_offset=child.start_byte,
                    )
                )

            elif child.type == "class_declaration":
                name_node = _ts_find_child_by_type(
                    child, "type_identifier", "identifier"
                )
                name = _ts_node_text(name_node, source_bytes) if name_node else None
                units.append(
                    SemanticUnit(
                        unit_type="class",
                        unit_name=name,
                        content=_ts_node_text(child, source_bytes),
                        char_offset=child.start_byte,
                    )
                )
                body = _ts_find_child_by_type(child, "class_body")
                if body is not None:
                    _walk(body, class_name=name)

            elif child.type == "method_definition":
                name_node = _ts_find_child_by_type(
                    child,
                    "property_identifier",
                    "identifier",
                    "computed_property_name",
                )
                name = _ts_node_text(name_node, source_bytes) if name_node else None
                if class_name:
                    unit_name = f"{class_name}:{name}" if name else class_name
                else:
                    unit_name = name
                units.append(
                    SemanticUnit(
                        unit_type="method",
                        unit_name=unit_name,
                        content=_ts_node_text(child, source_bytes),
                        char_offset=child.start_byte,
                    )
                )

            elif child.type in ("lexical_declaration", "variable_declaration"):
                # Extract arrow functions and function expressions assigned to variables
                for decl in child.children:
                    if decl.type == "variable_declarator":
                        value = _ts_find_child_by_type(
                            decl,
                            "arrow_function",
                            "function_expression",
                        )
                        if value is not None:
                            name_node = _ts_find_child_by_type(decl, "identifier")
                            name = (
                                _ts_node_text(name_node, source_bytes)
                                if name_node
                                else None
                            )
                            units.append(
                                SemanticUnit(
                                    unit_type="function",
                                    unit_name=name,
                                    content=_ts_node_text(child, source_bytes),
                                    char_offset=child.start_byte,
                                )
                            )

            elif child.type == "interface_declaration":
                name_node = _ts_find_child_by_type(
                    child, "type_identifier", "identifier"
                )
                name = _ts_node_text(name_node, source_bytes) if name_node else None
                units.append(
                    SemanticUnit(
                        unit_type="interface",
                        unit_name=name,
                        content=_ts_node_text(child, source_bytes),
                        char_offset=child.start_byte,
                    )
                )

            elif child.type == "type_alias_declaration":
                name_node = _ts_find_child_by_type(
                    child, "type_identifier", "identifier"
                )
                name = _ts_node_text(name_node, source_bytes) if name_node else None
                units.append(
                    SemanticUnit(
                        unit_type="type",
                        unit_name=name,
                        content=_ts_node_text(child, source_bytes),
                        char_offset=child.start_byte,
                    )
                )

            elif child.type == "enum_declaration":
                name_node = _ts_find_child_by_type(child, "identifier")
                name = _ts_node_text(name_node, source_bytes) if name_node else None
                units.append(
                    SemanticUnit(
                        unit_type="enum",
                        unit_name=name,
                        content=_ts_node_text(child, source_bytes),
                        char_offset=child.start_byte,
                    )
                )

            elif child.type == "export_statement":
                _walk(child, class_name=class_name)

    _walk(tree.root_node)
    return sorted(units, key=lambda u: u.char_offset)


def parse_javascript(source: str) -> list[SemanticUnit]:
    """Parse JavaScript source into SemanticUnits using tree-sitter."""
    if not source:
        return []

    lang = _get_ts_javascript_language()
    if lang is None:
        warnings.warn(
            "tree-sitter-javascript not installed — JS files will not be indexed",
            stacklevel=2,
        )
        return []

    import tree_sitter

    parser = tree_sitter.Parser()
    parser.language = lang
    source_bytes = source.encode()
    tree = parser.parse(source_bytes)
    return _extract_js_ts_units(tree, source_bytes)


def parse_typescript(source: str, tsx: bool = False) -> list[SemanticUnit]:
    """Parse TypeScript/TSX source into SemanticUnits using tree-sitter."""
    if not source:
        return []

    lang = _get_ts_tsx_language() if tsx else _get_ts_typescript_language()
    label = "TSX" if tsx else "TypeScript"
    if lang is None:
        warnings.warn(
            f"tree-sitter-typescript not installed — {label} files will not be indexed",
            stacklevel=2,
        )
        return []

    import tree_sitter

    parser = tree_sitter.Parser()
    parser.language = lang
    source_bytes = source.encode()
    tree = parser.parse(source_bytes)
    return _extract_js_ts_units(tree, source_bytes)


# ---------------------------------------------------------------------------
# Java
# ---------------------------------------------------------------------------

_JAVA_EXTENSIONS = frozenset({".java"})


def _get_ts_java_language():
    """Return the tree-sitter Java Language object, or None if unavailable."""
    try:
        import tree_sitter
        import tree_sitter_java

        return tree_sitter.Language(tree_sitter_java.language())
    except Exception:
        return None


def _extract_java_units(tree, source_bytes: bytes) -> list[SemanticUnit]:
    """Walk a tree-sitter parse tree and extract semantic units for Java."""
    units: list[SemanticUnit] = []

    def _walk(node, class_name: str | None = None):
        for child in node.children:
            if child.type in ("class_declaration", "record_declaration"):
                name_node = _ts_find_child_by_type(child, "identifier")
                name = _ts_node_text(name_node, source_bytes) if name_node else None
                units.append(
                    SemanticUnit(
                        unit_type="class",
                        unit_name=name,
                        content=_ts_node_text(child, source_bytes),
                        char_offset=child.start_byte,
                    )
                )
                body = _ts_find_child_by_type(
                    child, "class_body", "record_declaration_body"
                )
                if body is not None:
                    _walk(body, class_name=name)

            elif child.type == "interface_declaration":
                name_node = _ts_find_child_by_type(child, "identifier")
                name = _ts_node_text(name_node, source_bytes) if name_node else None
                units.append(
                    SemanticUnit(
                        unit_type="interface",
                        unit_name=name,
                        content=_ts_node_text(child, source_bytes),
                        char_offset=child.start_byte,
                    )
                )
                body = _ts_find_child_by_type(child, "interface_body")
                if body is not None:
                    _walk(body, class_name=name)

            elif child.type == "enum_declaration":
                name_node = _ts_find_child_by_type(child, "identifier")
                name = _ts_node_text(name_node, source_bytes) if name_node else None
                units.append(
                    SemanticUnit(
                        unit_type="enum",
                        unit_name=name,
                        content=_ts_node_text(child, source_bytes),
                        char_offset=child.start_byte,
                    )
                )
                body = _ts_find_child_by_type(child, "enum_body")
                if body is not None:
                    _walk(body, class_name=name)

            elif child.type == "enum_body_declarations":
                _walk(child, class_name=class_name)

            elif child.type == "method_declaration":
                name_node = _ts_find_child_by_type(child, "identifier")
                name = _ts_node_text(name_node, source_bytes) if name_node else None
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

            elif child.type == "constructor_declaration":
                name_node = _ts_find_child_by_type(child, "identifier")
                name = _ts_node_text(name_node, source_bytes) if name_node else None
                if class_name:
                    unit_name = f"{class_name}:{name}" if name else class_name
                else:
                    unit_name = name
                units.append(
                    SemanticUnit(
                        unit_type="method",
                        unit_name=unit_name,
                        content=_ts_node_text(child, source_bytes),
                        char_offset=child.start_byte,
                    )
                )

            elif child.type in ("program", "package_declaration"):
                _walk(child, class_name=class_name)

    _walk(tree.root_node)
    return sorted(units, key=lambda u: u.char_offset)


def parse_java(source: str) -> list[SemanticUnit]:
    """Parse Java source into SemanticUnits using tree-sitter.

    Returns [] with a warning if tree-sitter-java is not installed.
    """
    if not source:
        return []

    lang = _get_ts_java_language()
    if lang is None:
        warnings.warn(
            "tree-sitter-java not installed — .java files will not be indexed",
            stacklevel=2,
        )
        return []

    import tree_sitter

    parser = tree_sitter.Parser()
    parser.language = lang
    source_bytes = source.encode()
    tree = parser.parse(source_bytes)
    return _extract_java_units(tree, source_bytes)


# ---------------------------------------------------------------------------
# Go
# ---------------------------------------------------------------------------


def _get_ts_go_language():
    """Return the tree-sitter Go Language object, or None if unavailable."""
    try:
        import tree_sitter
        import tree_sitter_go

        return tree_sitter.Language(tree_sitter_go.language())
    except Exception:
        return None


def _extract_go_units(tree, source_bytes: bytes) -> list[SemanticUnit]:
    """Walk a tree-sitter parse tree and extract semantic units for Go."""
    units: list[SemanticUnit] = []

    for child in tree.root_node.children:
        if child.type == "function_declaration":
            name_node = _ts_find_child_by_type(child, "identifier")
            name = _ts_node_text(name_node, source_bytes) if name_node else None
            units.append(
                SemanticUnit(
                    unit_type="function",
                    unit_name=name,
                    content=_ts_node_text(child, source_bytes),
                    char_offset=child.start_byte,
                )
            )

        elif child.type == "method_declaration":
            # receiver is the first parameter_list: (s *Server)
            recv_list = _ts_find_child_by_type(child, "parameter_list")
            receiver_type = None
            if recv_list is not None:
                param = _ts_find_child_by_type(recv_list, "parameter_declaration")
                if param is not None:
                    type_node = _ts_find_child_by_type(
                        param, "pointer_type", "type_identifier"
                    )
                    if type_node is not None:
                        if type_node.type == "pointer_type":
                            ident = _ts_find_child_by_type(type_node, "type_identifier")
                            receiver_type = (
                                _ts_node_text(ident, source_bytes) if ident else None
                            )
                        else:
                            receiver_type = _ts_node_text(type_node, source_bytes)
            name_node = _ts_find_child_by_type(child, "field_identifier")
            method_name = _ts_node_text(name_node, source_bytes) if name_node else None
            unit_name = (
                f"{receiver_type}:{method_name}"
                if receiver_type and method_name
                else method_name
            )
            units.append(
                SemanticUnit(
                    unit_type="method",
                    unit_name=unit_name,
                    content=_ts_node_text(child, source_bytes),
                    char_offset=child.start_byte,
                )
            )

        elif child.type == "type_declaration":
            for spec in child.children:
                if spec.type != "type_spec":
                    continue
                name_node = _ts_find_child_by_type(spec, "type_identifier")
                name = _ts_node_text(name_node, source_bytes) if name_node else None
                body = _ts_find_child_by_type(spec, "struct_type", "interface_type")
                if body is None:
                    continue
                unit_type = "struct" if body.type == "struct_type" else "interface"
                units.append(
                    SemanticUnit(
                        unit_type=unit_type,
                        unit_name=name,
                        content=_ts_node_text(child, source_bytes),
                        char_offset=child.start_byte,
                    )
                )

    return sorted(units, key=lambda u: u.char_offset)


def parse_go(source: str) -> list[SemanticUnit]:
    """Parse a Go source string into SemanticUnits using tree-sitter.

    Returns [] with a warning if tree-sitter-go is not installed.
    """
    if not source:
        return []

    lang = _get_ts_go_language()
    if lang is None:
        warnings.warn(
            "tree-sitter-go not installed — .go files will not be indexed",
            stacklevel=2,
        )
        return []

    import tree_sitter

    parser = tree_sitter.Parser()
    parser.language = lang
    source_bytes = source.encode()
    tree = parser.parse(source_bytes)
    return _extract_go_units(tree, source_bytes)


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


# ---------------------------------------------------------------------------
# Terraform / HCL
# ---------------------------------------------------------------------------

_TF_EXTENSIONS = frozenset({".tf", ".tfvars"})

# Matches top-level HCL block headers, e.g.:
#   resource "aws_instance" "web" {
#   variable "region" {
#   locals {
_TF_BLOCK_RE = re.compile(
    r"^[ \t]*"
    r"(resource|variable|output|module|data|locals|provider|terraform|moved|import|check)"
    r'(?:[ \t]+"([^"]*)")?'
    r'(?:[ \t]+"([^"]*)")?'
    r"[ \t]*\{",
    re.MULTILINE,
)


def _tf_find_block_end(source: str, open_brace: int) -> int:
    """Return the index just past the closing brace matching open_brace."""
    depth = 0
    i = open_brace
    in_string = False
    escape_next = False
    while i < len(source):
        ch = source[i]
        if escape_next:
            escape_next = False
        elif in_string:
            if ch == "\\":
                escape_next = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
        i += 1
    return len(source)


def parse_terraform(source: str, is_tfvars: bool = False) -> list[SemanticUnit]:
    """Parse a Terraform HCL source string into SemanticUnits.

    For .tf files: each top-level block (resource, variable, output, module,
    data, locals, provider, terraform, moved, import, check) becomes one unit.
    unit_name is the dot-joined block labels, e.g. "aws_instance.web" for
    ``resource "aws_instance" "web" { ... }``.

    For .tfvars files: the entire file is returned as a single "tfvars" unit
    (analogous to the SQL parser's single-unit approach).
    """
    if not source.strip():
        return []

    if is_tfvars:
        return [
            SemanticUnit(
                unit_type="tfvars",
                unit_name=None,
                content=source,
                char_offset=0,
            )
        ]

    units = []
    for match in _TF_BLOCK_RE.finditer(source):
        block_type = match.group(1)
        label1 = match.group(2)
        label2 = match.group(3)

        if label1 and label2:
            unit_name = f"{label1}.{label2}"
        elif label1:
            unit_name = label1
        else:
            unit_name = None

        # The opening brace is the last character of the match.
        open_brace = match.end() - 1
        end = _tf_find_block_end(source, open_brace)
        content = source[match.start() : end].strip()

        units.append(
            SemanticUnit(
                unit_type=block_type,
                unit_name=unit_name,
                content=content,
                char_offset=match.start(),
            )
        )

    return units


# ---------------------------------------------------------------------------
# OpenSCAD
# ---------------------------------------------------------------------------
# tree-sitter-openscad exists at https://github.com/bollian/tree-sitter-openscad
# but has no PyPI Python package as of 2026-04.  Using regex fallback until
# a pip-installable binding is published.

_OPENSCAD_EXTENSIONS = frozenset({".scad"})

# Matches top-level module or function declarations.
# module foo(...) {   — braced body
# function foo(...) = expr;  — expression body (no brace)
_OPENSCAD_DECL_RE = re.compile(
    r"^[ \t]*(module|function)[ \t]+([A-Za-z_][A-Za-z0-9_]*)[ \t]*\(",
    re.MULTILINE,
)


def _openscad_find_body_end(source: str, start: int) -> int:
    """Return the index just past the end of an OpenSCAD declaration body.

    For module declarations the body is a brace-delimited block.
    For function declarations the body ends at the first ';' outside parens/brackets.
    Starts scanning from *start*, which should be positioned at or before the
    opening '{' or '=' of the body.
    """
    i = start
    n = len(source)
    in_string = False
    escape_next = False

    # Advance to the opening '{' or '='
    while i < n and source[i] not in ("{", "=", ";"):
        i += 1

    if i >= n:
        return n

    if source[i] == "{":
        # Brace-matched block (module body)
        depth = 0
        while i < n:
            ch = source[i]
            if escape_next:
                escape_next = False
            elif in_string:
                if ch == "\\":
                    escape_next = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return i + 1
            i += 1
        return n
    else:
        # '=' expression ending at ';' (function body)
        paren_depth = 0
        bracket_depth = 0
        while i < n:
            ch = source[i]
            if escape_next:
                escape_next = False
            elif in_string:
                if ch == "\\":
                    escape_next = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "(":
                    paren_depth += 1
                elif ch == ")":
                    paren_depth -= 1
                elif ch == "[":
                    bracket_depth += 1
                elif ch == "]":
                    bracket_depth -= 1
                elif ch == ";" and paren_depth == 0 and bracket_depth == 0:
                    return i + 1
            i += 1
        return n


def parse_openscad(source: str) -> list[SemanticUnit]:
    """Parse an OpenSCAD source string into SemanticUnits.

    Extracts top-level module and function declarations.
    """
    if not source.strip():
        return []

    units: list[SemanticUnit] = []
    for match in _OPENSCAD_DECL_RE.finditer(source):
        decl_type = match.group(1)  # "module" or "function"
        name = match.group(2)
        unit_type = "module" if decl_type == "module" else "function"

        # Find where the parameter list closes, then locate the body
        end = _openscad_find_body_end(source, match.end())
        content = source[match.start() : end].strip()
        if content:
            units.append(
                SemanticUnit(
                    unit_type=unit_type,
                    unit_name=name,
                    content=content,
                    char_offset=match.start(),
                )
            )

    return units


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
        return parse_go(path.read_text(encoding="utf-8", errors="replace"))
    if suffix in _C_EXTENSIONS:
        return parse_c(path.read_text(encoding="utf-8", errors="replace"))
    if suffix in _CPP_EXTENSIONS:
        return parse_cpp(path.read_text(encoding="utf-8", errors="replace"))
    if suffix in _JS_EXTENSIONS:
        return parse_javascript(path.read_text(encoding="utf-8", errors="replace"))
    if suffix in _TS_EXTENSIONS:
        return parse_typescript(
            path.read_text(encoding="utf-8", errors="replace"),
            tsx=(suffix in (".tsx",)),
        )
    if suffix in _JAVA_EXTENSIONS:
        return parse_java(path.read_text(encoding="utf-8", errors="replace"))
    if suffix in (".md", ".mdx"):
        return parse_markdown(path.read_text(encoding="utf-8", errors="replace"))
    if suffix == ".sql":
        return parse_sql(path.read_text(encoding="utf-8", errors="replace"))
    if suffix in _TF_EXTENSIONS:
        return parse_terraform(
            path.read_text(encoding="utf-8", errors="replace"),
            is_tfvars=(suffix == ".tfvars"),
        )
    if suffix in _OPENSCAD_EXTENSIONS:
        return parse_openscad(path.read_text(encoding="utf-8", errors="replace"))

    logger.debug("skipping %s: unsupported extension %r", path, suffix)
    return []
