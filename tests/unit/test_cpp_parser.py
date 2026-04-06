"""Unit tests for the C/C++ parser integration in mcp_rag.parsers.

Tests use real tree-sitter parsing (no subprocess mocking needed since
the parser runs in-process).
"""

import hashlib
import textwrap

import pytest

from mcp_rag.parsers import parse_c, parse_cpp, parse_file


# ---------------------------------------------------------------------------
# C parser — functions
# ---------------------------------------------------------------------------


def test_parse_c_function():
    source = textwrap.dedent("""\
        int add(int a, int b) {
            return a + b;
        }
    """)
    units = parse_c(source)
    assert len(units) == 1
    assert units[0].unit_type == "function"
    assert units[0].unit_name == "add"


def test_parse_c_multiple_functions():
    source = textwrap.dedent("""\
        void foo() {}
        void bar() {}
    """)
    units = parse_c(source)
    assert len(units) == 2
    names = {u.unit_name for u in units}
    assert names == {"foo", "bar"}


# ---------------------------------------------------------------------------
# C parser — structs
# ---------------------------------------------------------------------------


def test_parse_c_struct():
    source = textwrap.dedent("""\
        struct Point {
            int x;
            int y;
        };
    """)
    units = parse_c(source)
    assert len(units) == 1
    assert units[0].unit_type == "struct"
    assert units[0].unit_name == "Point"


def test_parse_c_forward_declaration_skipped():
    source = "struct Opaque;\n"
    units = parse_c(source)
    assert units == []


# ---------------------------------------------------------------------------
# C parser — enums
# ---------------------------------------------------------------------------


def test_parse_c_enum():
    source = textwrap.dedent("""\
        enum Color {
            RED,
            GREEN,
            BLUE
        };
    """)
    units = parse_c(source)
    assert len(units) == 1
    assert units[0].unit_type == "enum"
    assert units[0].unit_name == "Color"


def test_parse_c_enum_forward_decl_skipped():
    source = "enum Status;\n"
    units = parse_c(source)
    assert units == []


# ---------------------------------------------------------------------------
# C parser — field values
# ---------------------------------------------------------------------------


def test_parse_c_char_offset():
    source = "/* comment */\nint foo() { return 0; }\n"
    units = parse_c(source)
    assert len(units) == 1
    assert units[0].char_offset == source.index("int foo")


def test_parse_c_content_preserved():
    func_text = "int square(int n) {\n    return n * n;\n}"
    source = func_text + "\n"
    units = parse_c(source)
    assert units[0].content == func_text


def test_parse_c_content_md5():
    source = "void noop() {}\n"
    units = parse_c(source)
    expected_md5 = hashlib.md5(units[0].content.encode()).hexdigest()
    assert units[0].content_md5 == expected_md5


def test_parse_c_empty_source():
    assert parse_c("") == []


# ---------------------------------------------------------------------------
# C++ parser — classes
# ---------------------------------------------------------------------------


def test_parse_cpp_class():
    source = textwrap.dedent("""\
        class Widget {
        public:
            int value;
        };
    """)
    units = parse_cpp(source)
    assert any(u.unit_type == "class" and u.unit_name == "Widget" for u in units)


def test_parse_cpp_class_with_method():
    source = textwrap.dedent("""\
        class Counter {
        public:
            void increment() {
                count++;
            }
            int count;
        };
    """)
    units = parse_cpp(source)
    types = {u.unit_type for u in units}
    assert "class" in types
    assert "method" in types
    method = next(u for u in units if u.unit_type == "method")
    assert method.unit_name == "Counter:increment"


# ---------------------------------------------------------------------------
# C++ parser — functions and structs
# ---------------------------------------------------------------------------


def test_parse_cpp_function():
    source = textwrap.dedent("""\
        int factorial(int n) {
            if (n <= 1) return 1;
            return n * factorial(n - 1);
        }
    """)
    units = parse_cpp(source)
    assert len(units) == 1
    assert units[0].unit_type == "function"
    assert units[0].unit_name == "factorial"


def test_parse_cpp_struct():
    source = textwrap.dedent("""\
        struct Vec3 {
            float x, y, z;
        };
    """)
    units = parse_cpp(source)
    assert any(u.unit_type == "struct" and u.unit_name == "Vec3" for u in units)


def test_parse_cpp_enum():
    source = textwrap.dedent("""\
        enum class Direction {
            Up,
            Down,
            Left,
            Right
        };
    """)
    units = parse_cpp(source)
    assert any(u.unit_type == "enum" and u.unit_name == "Direction" for u in units)


def test_parse_cpp_empty_source():
    assert parse_cpp("") == []


# ---------------------------------------------------------------------------
# C++ parser — namespace and template
# ---------------------------------------------------------------------------


def test_parse_cpp_function_in_namespace():
    source = textwrap.dedent("""\
        namespace math {
            int add(int a, int b) {
                return a + b;
            }
        }
    """)
    units = parse_cpp(source)
    assert len(units) == 1
    assert units[0].unit_type == "function"
    assert units[0].unit_name == "add"


def test_parse_cpp_template_function():
    source = textwrap.dedent("""\
        template<typename T>
        T max_val(T a, T b) {
            return a > b ? a : b;
        }
    """)
    units = parse_cpp(source)
    assert len(units) == 1
    assert units[0].unit_type == "function"


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


def test_parse_c_missing_treesitter_returns_empty(monkeypatch):
    import mcp_rag.parsers as parsers_mod

    monkeypatch.setattr(parsers_mod, "_get_ts_c_language", lambda: None)
    with pytest.warns(UserWarning, match="tree-sitter-c"):
        result = parse_c("int foo() {}\n")
    assert result == []


def test_parse_cpp_missing_treesitter_returns_empty(monkeypatch):
    import mcp_rag.parsers as parsers_mod

    monkeypatch.setattr(parsers_mod, "_get_ts_cpp_language", lambda: None)
    with pytest.warns(UserWarning, match="tree-sitter-cpp"):
        result = parse_cpp("void bar() {}\n")
    assert result == []


# ---------------------------------------------------------------------------
# parse_file dispatch
# ---------------------------------------------------------------------------


def test_parse_file_c_extension(tmp_path):
    f = tmp_path / "hello.c"
    f.write_text("int main() { return 0; }\n")
    units = parse_file(f)
    assert any(u.unit_name == "main" for u in units)


def test_parse_file_h_extension(tmp_path):
    f = tmp_path / "header.h"
    f.write_text("struct Cfg { int val; };\n")
    units = parse_file(f)
    assert any(u.unit_type == "struct" for u in units)


def test_parse_file_cpp_extension(tmp_path):
    f = tmp_path / "app.cpp"
    f.write_text("void run() {}\n")
    units = parse_file(f)
    assert any(u.unit_name == "run" for u in units)


def test_parse_file_hpp_extension(tmp_path):
    f = tmp_path / "util.hpp"
    f.write_text("class Util { void go() {} };\n")
    units = parse_file(f)
    assert any(u.unit_type == "class" for u in units)


def test_parse_file_cc_extension(tmp_path):
    f = tmp_path / "lib.cc"
    f.write_text("int compute() { return 42; }\n")
    units = parse_file(f)
    assert any(u.unit_name == "compute" for u in units)


def test_parse_file_c_binary_skipped(tmp_path):
    f = tmp_path / "weird.c"
    f.write_bytes(b"\x00" * 20 + b"int main() {}\n")
    assert parse_file(f) == []
