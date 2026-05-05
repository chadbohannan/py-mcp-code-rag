"""Unit tests for import extraction and resolution."""

from mcp_rag.imports import (
    _extract_python_imports,
    _extract_js_ts_imports,
    _extract_c_cpp_imports,
    _extract_java_imports,
    _extract_go_imports,
    _resolve_python_import,
    _resolve_js_ts_import,
    extract_and_resolve_imports,
)


# ---------------------------------------------------------------------------
# Python extraction
# ---------------------------------------------------------------------------


def test_python_import():
    assert _extract_python_imports("import os") == ["os"]


def test_python_from_import():
    assert _extract_python_imports("from os.path import join") == ["os.path"]


def test_python_multiple_imports():
    source = "import os\nimport sys\nfrom pathlib import Path\n"
    result = _extract_python_imports(source)
    assert "os" in result
    assert "sys" in result
    assert "pathlib" in result


def test_python_syntax_error_returns_empty():
    assert _extract_python_imports("def (broken") == []


def test_python_no_imports():
    assert _extract_python_imports("x = 1\n") == []


# ---------------------------------------------------------------------------
# Python resolution
# ---------------------------------------------------------------------------


def test_python_resolve_direct_file(tmp_path):
    (tmp_path / "foo").mkdir()
    target = tmp_path / "foo" / "bar.py"
    target.write_text("x = 1")
    repo_files = {target.resolve()}
    result = _resolve_python_import("foo.bar", tmp_path, repo_files)
    assert result == target.resolve()


def test_python_resolve_package_init(tmp_path):
    (tmp_path / "pkg").mkdir()
    init = tmp_path / "pkg" / "__init__.py"
    init.write_text("")
    repo_files = {init.resolve()}
    result = _resolve_python_import("pkg", tmp_path, repo_files)
    assert result == init.resolve()


def test_python_resolve_not_found(tmp_path):
    result = _resolve_python_import("nonexistent", tmp_path, set())
    assert result is None


# ---------------------------------------------------------------------------
# JS/TS extraction
# ---------------------------------------------------------------------------


def test_js_import_from():
    source = "import { foo } from './utils';"
    result = _extract_js_ts_imports(source)
    assert "./utils" in result


def test_js_require():
    source = "const foo = require('./utils');"
    result = _extract_js_ts_imports(source)
    assert "./utils" in result


def test_js_bare_import():
    source = "import 'react';"
    result = _extract_js_ts_imports(source)
    assert "react" in result


def test_js_multiple_imports():
    source = (
        "import { a } from './a';\nimport b from './b';\nconst c = require('./c');\n"
    )
    result = _extract_js_ts_imports(source)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# JS/TS resolution
# ---------------------------------------------------------------------------


def test_js_resolve_relative_with_extension(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    target = src / "utils.js"
    target.write_text("")
    main = src / "main.js"
    repo_files = {target.resolve()}
    result = _resolve_js_ts_import("./utils", main, tmp_path, repo_files)
    assert result == target.resolve()


def test_js_resolve_bare_specifier_skipped(tmp_path):
    main = tmp_path / "main.js"
    result = _resolve_js_ts_import("react", main, tmp_path, set())
    assert result is None


def test_js_resolve_index_file(tmp_path):
    src = tmp_path / "src"
    comp = src / "components"
    comp.mkdir(parents=True)
    index = comp / "index.ts"
    index.write_text("")
    main = src / "main.ts"
    repo_files = {index.resolve()}
    result = _resolve_js_ts_import("./components", main, tmp_path, repo_files)
    assert result == index.resolve()


# ---------------------------------------------------------------------------
# C/C++ extraction
# ---------------------------------------------------------------------------


def test_c_quoted_include():
    source = '#include "myheader.h"\n#include <stdio.h>\n'
    result = _extract_c_cpp_imports(source)
    assert result == ["myheader.h"]


def test_c_no_system_includes():
    source = "#include <stdlib.h>\n"
    assert _extract_c_cpp_imports(source) == []


# ---------------------------------------------------------------------------
# Java extraction
# ---------------------------------------------------------------------------


def test_java_import():
    source = "import com.example.Foo;\nimport java.util.List;\n"
    result = _extract_java_imports(source)
    assert "com.example.Foo" in result
    assert "java.util.List" in result


# ---------------------------------------------------------------------------
# Go extraction
# ---------------------------------------------------------------------------


def test_go_single_import():
    source = 'import "fmt"\n'
    result = _extract_go_imports(source)
    assert "fmt" in result


def test_go_grouped_imports():
    source = 'import (\n\t"fmt"\n\t"os"\n)\n'
    result = _extract_go_imports(source)
    assert "fmt" in result
    assert "os" in result


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def test_extract_and_resolve_python(tmp_path):
    (tmp_path / "lib").mkdir()
    target = tmp_path / "lib" / "__init__.py"
    target.write_text("")
    utils = tmp_path / "lib" / "utils.py"
    utils.write_text("def helper(): pass")

    source_file = tmp_path / "main.py"
    # `from lib.utils import helper` → module is "lib.utils" → resolves to lib/utils.py
    source_file.write_text("from lib.utils import helper\n")

    repo_files = {target.resolve(), utils.resolve(), source_file.resolve()}
    result = extract_and_resolve_imports(
        source_file, source_file.read_text(), tmp_path, repo_files
    )
    assert utils.resolve() in result


def test_extract_and_resolve_unsupported_extension(tmp_path):
    source_file = tmp_path / "data.sql"
    source_file.write_text("SELECT 1;")
    result = extract_and_resolve_imports(
        source_file, source_file.read_text(), tmp_path, set()
    )
    assert result == []


def test_deduplication(tmp_path):
    (tmp_path / "lib.py").write_text("")
    source = "import lib\nimport lib\n"
    source_file = tmp_path / "main.py"
    repo_files = {(tmp_path / "lib.py").resolve()}
    result = extract_and_resolve_imports(source_file, source, tmp_path, repo_files)
    assert len(result) == 1
