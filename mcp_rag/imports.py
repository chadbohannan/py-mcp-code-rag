"""Per-language import extraction and resolution for mcp-rag.

Each extractor takes source text and returns raw import strings.
``resolve_imports`` maps those strings to file paths within the repo
on a best-effort basis — unresolved imports are silently dropped.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------


def _extract_python_imports(source: str, filename: str = "<unknown>") -> list[str]:
    """Extract module-level import targets from Python source.

    Returns dotted module names (e.g. ``["os.path", "mcp_rag.db"]``).
    """
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return []

    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.append(node.module)
    return modules


def _resolve_python_import(module: str, repo_root: Path, repo_files: set[Path]) -> Path | None:
    """Try to resolve a dotted Python module to a file in the repo."""
    parts = module.split(".")
    # Try as a direct module file: foo/bar.py
    candidate = repo_root / "/".join(parts)
    py_file = candidate.with_suffix(".py")
    if py_file.resolve() in repo_files:
        return py_file.resolve()
    # Try as a package: foo/bar/__init__.py
    init_file = candidate / "__init__.py"
    if init_file.resolve() in repo_files:
        return init_file.resolve()
    return None


# ---------------------------------------------------------------------------
# JavaScript / TypeScript
# ---------------------------------------------------------------------------

# Matches: import ... from 'path' / import ... from "path"
# Also: require('path') / require("path")
_JS_IMPORT_RE = re.compile(
    r"""(?:"""
    r"""import\s+(?:.*?\s+from\s+)?['"]([^'"]+)['"]"""
    r"""|"""
    r"""require\s*\(\s*['"]([^'"]+)['"]\s*\)"""
    r""")""",
    re.MULTILINE,
)


def _extract_js_ts_imports(source: str) -> list[str]:
    """Extract import paths from JS/TS source."""
    paths: list[str] = []
    for m in _JS_IMPORT_RE.finditer(source):
        path = m.group(1) or m.group(2)
        if path:
            paths.append(path)
    return paths


_JS_EXTENSIONS = (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts")


def _resolve_js_ts_import(
    import_path: str, file_path: Path, repo_root: Path, repo_files: set[Path]
) -> Path | None:
    """Resolve a JS/TS import path to a file in the repo."""
    # Skip bare specifiers (package imports like 'react', '@foo/bar')
    if not import_path.startswith("."):
        return None

    base = file_path.parent / import_path

    # Try exact path first
    if base.resolve() in repo_files:
        return base.resolve()

    # Try with extensions
    for ext in _JS_EXTENSIONS:
        candidate = base.with_suffix(ext)
        if candidate.resolve() in repo_files:
            return candidate.resolve()

    # Try index files
    for ext in _JS_EXTENSIONS:
        candidate = base / f"index{ext}"
        if candidate.resolve() in repo_files:
            return candidate.resolve()

    return None


# ---------------------------------------------------------------------------
# C / C++
# ---------------------------------------------------------------------------

# Matches: #include "path" (quoted includes only, not <system>)
_C_INCLUDE_RE = re.compile(r'#\s*include\s+"([^"]+)"')


def _extract_c_cpp_imports(source: str) -> list[str]:
    """Extract quoted #include paths from C/C++ source."""
    return _C_INCLUDE_RE.findall(source)


def _resolve_c_cpp_import(
    include_path: str, file_path: Path, repo_root: Path, repo_files: set[Path]
) -> Path | None:
    """Resolve a quoted #include to a file in the repo."""
    # Try relative to the including file
    candidate = (file_path.parent / include_path).resolve()
    if candidate in repo_files:
        return candidate

    # Try relative to repo root
    candidate = (repo_root / include_path).resolve()
    if candidate in repo_files:
        return candidate

    return None


# ---------------------------------------------------------------------------
# Java
# ---------------------------------------------------------------------------

_JAVA_IMPORT_RE = re.compile(r"^\s*import\s+([\w.]+)\s*;", re.MULTILINE)


def _extract_java_imports(source: str) -> list[str]:
    """Extract import targets from Java source."""
    return _JAVA_IMPORT_RE.findall(source)


def _resolve_java_import(
    import_path: str, repo_root: Path, repo_files: set[Path]
) -> Path | None:
    """Resolve a Java import to a file in the repo."""
    # com.foo.Bar → com/foo/Bar.java
    parts = import_path.split(".")
    candidate = repo_root / "/".join(parts)
    java_file = candidate.with_suffix(".java")
    if java_file.resolve() in repo_files:
        return java_file.resolve()
    return None


# ---------------------------------------------------------------------------
# Go
# ---------------------------------------------------------------------------

_GO_IMPORT_RE = re.compile(r'"([^"]+)"')
_GO_IMPORT_BLOCK_RE = re.compile(r"import\s*\((.*?)\)", re.DOTALL)
_GO_IMPORT_SINGLE_RE = re.compile(r'import\s+"([^"]+)"')


def _extract_go_imports(source: str) -> list[str]:
    """Extract import paths from Go source."""
    paths: list[str] = []
    # Grouped imports: import ( ... )
    for block in _GO_IMPORT_BLOCK_RE.finditer(source):
        for m in _GO_IMPORT_RE.finditer(block.group(1)):
            paths.append(m.group(1))
    # Single imports: import "path"
    for m in _GO_IMPORT_SINGLE_RE.finditer(source):
        paths.append(m.group(1))
    return paths


def _resolve_go_import(
    import_path: str,
    repo_root: Path,
    go_files_by_dir: dict[Path, Path],
) -> Path | None:
    """Resolve a Go import path to a file in the repo.

    Go imports are package-level, not file-level. We look for any .go file
    in a directory matching the import path suffix and return the first match.

    *go_files_by_dir* maps resolved directory paths to a representative .go
    file in that directory (pre-built by the caller for O(1) lookups).
    """
    # e.g. "github.com/user/repo/pkg/foo" → try "pkg/foo", then "foo"
    parts = import_path.split("/")
    for start in range(len(parts)):
        candidate_dir = (repo_root / "/".join(parts[start:])).resolve()
        match = go_files_by_dir.get(candidate_dir)
        if match is not None:
            return match
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_PY_EXTENSIONS = frozenset({".py"})
_JS_TS_EXTENSIONS = frozenset({".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"})
_C_CPP_EXTENSIONS = frozenset({
    ".c", ".h", ".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx", ".ino",
})
_JAVA_EXTENSIONS = frozenset({".java"})
_GO_EXTENSIONS = frozenset({".go"})


def extract_and_resolve_imports(
    file_path: Path,
    source: str,
    repo_root: Path,
    repo_files: set[Path],
) -> list[Path]:
    """Extract imports from a file and resolve them to paths within the repo.

    Returns a deduplicated list of resolved absolute file paths.
    Unresolvable imports are silently dropped.
    """
    suffix = file_path.suffix.lower()
    resolved: list[Path] = []

    if suffix in _PY_EXTENSIONS:
        for mod in _extract_python_imports(source, filename=str(file_path)):
            r = _resolve_python_import(mod, repo_root, repo_files)
            if r is not None:
                resolved.append(r)

    elif suffix in _JS_TS_EXTENSIONS:
        for imp in _extract_js_ts_imports(source):
            r = _resolve_js_ts_import(imp, file_path, repo_root, repo_files)
            if r is not None:
                resolved.append(r)

    elif suffix in _C_CPP_EXTENSIONS:
        for inc in _extract_c_cpp_imports(source):
            r = _resolve_c_cpp_import(inc, file_path, repo_root, repo_files)
            if r is not None:
                resolved.append(r)

    elif suffix in _JAVA_EXTENSIONS:
        for imp in _extract_java_imports(source):
            r = _resolve_java_import(imp, repo_root, repo_files)
            if r is not None:
                resolved.append(r)

    elif suffix in _GO_EXTENSIONS:
        # Build a dir→file index for O(1) lookups instead of scanning repo_files
        go_files_by_dir: dict[Path, Path] = {}
        for f in repo_files:
            if f.suffix == ".go":
                go_files_by_dir.setdefault(f.parent.resolve(), f)
        for imp in _extract_go_imports(source):
            r = _resolve_go_import(imp, repo_root, go_files_by_dir)
            if r is not None:
                resolved.append(r)

    # Deduplicate while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in resolved:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique
