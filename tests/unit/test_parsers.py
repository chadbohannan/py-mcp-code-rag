"""Unit tests for mcp_rag.parsers — pure parsing logic, no I/O."""

import textwrap


from mcp_rag.parsers import parse_file, parse_markdown, parse_python, parse_sql


# ---------------------------------------------------------------------------
# Python parser
# ---------------------------------------------------------------------------


def test_parse_python_empty_file():
    assert parse_python("") == []


def test_parse_python_function():
    source = textwrap.dedent("""\
        def foo():
            return 42
    """)
    units = parse_python(source)
    funcs = [u for u in units if u.unit_type == "function" and u.unit_name == "foo"]
    assert len(funcs) == 1


def test_parse_python_function_content_contains_def():
    source = textwrap.dedent("""\
        def greet(name: str) -> str:
            return f"Hello, {name}"
    """)
    units = parse_python(source)
    func = next(u for u in units if u.unit_name == "greet")
    assert "def greet" in func.content
    assert "return" in func.content


def test_parse_python_char_offset_is_correct():
    source = textwrap.dedent("""\
        x = 1

        def my_func():
            pass
    """)
    units = parse_python(source)
    func = next(u for u in units if u.unit_name == "my_func")
    assert func.char_offset == source.index("def my_func")


def test_parse_python_class():
    source = textwrap.dedent("""\
        class MyClass:
            pass
    """)
    units = parse_python(source)
    classes = [u for u in units if u.unit_type == "class" and u.unit_name == "MyClass"]
    assert len(classes) == 1


def test_parse_python_method():
    source = textwrap.dedent("""\
        class Calc:
            def add(self, a, b):
                return a + b
    """)
    units = parse_python(source)
    methods = [u for u in units if u.unit_type == "method" and u.unit_name == "add"]
    assert len(methods) == 1


def test_parse_python_class_and_method_are_separate_units():
    source = textwrap.dedent("""\
        class Service:
            def run(self):
                pass
    """)
    units = parse_python(source)
    types = {u.unit_type for u in units}
    assert "class" in types
    assert "method" in types


def test_parse_python_multiple_functions():
    source = textwrap.dedent("""\
        def alpha():
            pass

        def beta():
            pass

        def gamma():
            pass
    """)
    units = parse_python(source)
    funcs = [u for u in units if u.unit_type == "function"]
    names = [u.unit_name for u in funcs]
    assert "alpha" in names
    assert "beta" in names
    assert "gamma" in names
    assert len(funcs) == 3


def test_parse_python_function_order_matches_source():
    source = textwrap.dedent("""\
        def first():
            pass

        def second():
            pass
    """)
    units = parse_python(source)
    funcs = [u for u in units if u.unit_type == "function"]
    assert funcs[0].unit_name == "first"
    assert funcs[1].unit_name == "second"
    assert funcs[0].char_offset < funcs[1].char_offset


def test_parse_python_nested_function_not_a_top_unit():
    source = textwrap.dedent("""\
        def outer():
            def inner():
                pass
            return inner
    """)
    units = parse_python(source)
    names = [u.unit_name for u in units]
    assert "inner" not in names


def test_parse_python_syntax_error_returns_empty():
    units = parse_python("def broken(:\n    pass")
    assert units == []


def test_parse_python_content_md5_set():
    source = "def compute(): return 1\n"
    units = parse_python(source)
    func = next(u for u in units if u.unit_name == "compute")
    import hashlib

    assert func.content_md5 == hashlib.md5(func.content.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Markdown parser
# ---------------------------------------------------------------------------


def test_parse_markdown_empty_string():
    assert parse_markdown("") == []


def test_parse_markdown_single_heading_with_text():
    source = textwrap.dedent("""\
        # Introduction

        This section introduces the topic.
    """)
    units = parse_markdown(source)
    assert len(units) >= 1
    assert any("Introduction" in (u.unit_name or u.content) for u in units)


def test_parse_markdown_multiple_headings_produce_multiple_units():
    source = textwrap.dedent("""\
        # Alpha

        First section.

        # Beta

        Second section.
    """)
    units = parse_markdown(source)
    assert len(units) >= 2


def test_parse_markdown_unit_name_is_heading_text():
    source = textwrap.dedent("""\
        # My Section

        Some content here.
    """)
    units = parse_markdown(source)
    assert any(u.unit_name == "My Section" for u in units)


def test_parse_markdown_char_offset_increases_monotonically():
    source = textwrap.dedent("""\
        # First

        First content.

        # Second

        Second content.
    """)
    units = parse_markdown(source)
    offsets = [u.char_offset for u in units]
    assert offsets == sorted(offsets)


def test_parse_markdown_content_before_first_heading():
    source = textwrap.dedent("""\
        Some preamble text.

        # First Heading

        Content under heading.
    """)
    units = parse_markdown(source)
    # Preamble should produce at least one unit
    preamble_units = [u for u in units if u.char_offset == 0]
    assert len(preamble_units) >= 1


def test_parse_markdown_heading_content_includes_text():
    source = textwrap.dedent("""\
        # Setup

        Install with `pip install foo`.
    """)
    units = parse_markdown(source)
    setup_unit = next(u for u in units if u.unit_name == "Setup")
    assert "Install" in setup_unit.content


def test_parse_markdown_content_md5_set():
    source = "# Hello\n\nWorld.\n"
    units = parse_markdown(source)
    import hashlib

    for unit in units:
        assert unit.content_md5 == hashlib.md5(unit.content.encode()).hexdigest()


# ---------------------------------------------------------------------------
# SQL parser
# ---------------------------------------------------------------------------


def test_parse_sql_small_file():
    source = "SELECT id, name FROM users WHERE active = 1;"
    units = parse_sql(source)
    assert len(units) == 1
    assert units[0].unit_type == "sql"
    assert units[0].content == source


def test_parse_sql_unit_name_is_none():
    source = "SELECT 1"
    units = parse_sql(source)
    assert units[0].unit_name is None


def test_parse_sql_char_offset_is_zero():
    source = "SELECT 1"
    units = parse_sql(source)
    assert units[0].char_offset == 0


def test_parse_sql_exactly_4096_bytes():
    # Boundary: exactly 4096 bytes should be included
    source = "x" * 4096
    units = parse_sql(source)
    assert len(units) == 1


def test_parse_sql_over_4096_bytes_skipped():
    source = "x" * 4097
    units = parse_sql(source)
    assert units == []


def test_parse_sql_content_md5_set():
    source = "SELECT 1"
    units = parse_sql(source)
    import hashlib

    assert units[0].content_md5 == hashlib.md5(source.encode()).hexdigest()


# ---------------------------------------------------------------------------
# parse_file dispatch
# ---------------------------------------------------------------------------


def test_parse_file_py_dispatches_to_python_parser(tmp_path):
    py_file = tmp_path / "example.py"
    source = "def hello(): return 'hi'\n"
    py_file.write_text(source, encoding="utf-8")
    units = parse_file(py_file)
    assert any(u.unit_name == "hello" for u in units)


def test_parse_file_md_dispatches_to_markdown_parser(tmp_path):
    md_file = tmp_path / "README.md"
    md_file.write_text("# Title\n\nSome content.\n", encoding="utf-8")
    units = parse_file(md_file)
    assert len(units) >= 1


def test_parse_file_mdx_dispatches_to_markdown_parser(tmp_path):
    mdx_file = tmp_path / "page.mdx"
    mdx_file.write_text("# Title\n\nSome content.\n", encoding="utf-8")
    units = parse_file(mdx_file)
    assert len(units) >= 1


def test_parse_file_sql_dispatches_to_sql_parser(tmp_path):
    sql_file = tmp_path / "query.sql"
    sql_file.write_text("SELECT 1;", encoding="utf-8")
    units = parse_file(sql_file)
    assert len(units) == 1
    assert units[0].unit_type == "sql"


def test_parse_file_binary_returns_empty(tmp_path):
    bin_file = tmp_path / "data.bin"
    bin_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 512)
    units = parse_file(bin_file)
    assert units == []


def test_parse_file_unknown_extension_returns_empty(tmp_path):
    unknown = tmp_path / "data.xyz"
    unknown.write_text("some random content", encoding="utf-8")
    units = parse_file(unknown)
    assert units == []


def test_parse_file_unknown_extension_does_not_raise(tmp_path):
    unknown = tmp_path / "config.toml"
    unknown.write_text("[settings]\nfoo = 1\n", encoding="utf-8")
    # Should not raise, just return empty
    result = parse_file(unknown)
    assert isinstance(result, list)


def test_parse_file_binary_check_uses_first_512_bytes(tmp_path):
    # A file that has a null byte only at position 511 (within the first 512)
    # should still be detected as binary
    content = b"a" * 511 + b"\x00" + b"b" * 100
    f = tmp_path / "tricky.py"
    f.write_bytes(content)
    units = parse_file(f)
    assert units == []
