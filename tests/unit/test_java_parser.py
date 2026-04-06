"""Unit tests for the Java parser integration in mcp_rag.parsers."""

import hashlib
import textwrap

import pytest

from mcp_rag.parsers import parse_java, parse_file


# ---------------------------------------------------------------------------
# Java parser — classes
# ---------------------------------------------------------------------------


def test_parse_java_class():
    source = textwrap.dedent("""\
        public class Hello {
            int value;
        }
    """)
    units = parse_java(source)
    assert any(u.unit_type == "class" and u.unit_name == "Hello" for u in units)


def test_parse_java_class_with_method():
    source = textwrap.dedent("""\
        public class Counter {
            private int count;

            public void increment() {
                count++;
            }
        }
    """)
    units = parse_java(source)
    types = {u.unit_type for u in units}
    assert "class" in types
    assert "method" in types
    method = next(u for u in units if u.unit_type == "method")
    assert method.unit_name == "Counter:increment"


def test_parse_java_constructor():
    source = textwrap.dedent("""\
        public class Point {
            int x, y;

            public Point(int x, int y) {
                this.x = x;
                this.y = y;
            }
        }
    """)
    units = parse_java(source)
    ctor = next(u for u in units if u.unit_type == "method" and "Point:Point" in (u.unit_name or ""))
    assert ctor is not None


# ---------------------------------------------------------------------------
# Java parser — interfaces
# ---------------------------------------------------------------------------


def test_parse_java_interface():
    source = textwrap.dedent("""\
        public interface Runnable {
            void run();
        }
    """)
    units = parse_java(source)
    assert any(u.unit_type == "interface" and u.unit_name == "Runnable" for u in units)


# ---------------------------------------------------------------------------
# Java parser — enums
# ---------------------------------------------------------------------------


def test_parse_java_enum():
    source = textwrap.dedent("""\
        public enum Color {
            RED,
            GREEN,
            BLUE
        }
    """)
    units = parse_java(source)
    assert any(u.unit_type == "enum" and u.unit_name == "Color" for u in units)


def test_parse_java_enum_with_method():
    source = textwrap.dedent("""\
        public enum Planet {
            EARTH(5.976e+24);

            private final double mass;

            Planet(double mass) {
                this.mass = mass;
            }

            public double getMass() {
                return mass;
            }
        }
    """)
    units = parse_java(source)
    assert any(u.unit_type == "enum" and u.unit_name == "Planet" for u in units)
    assert any(u.unit_type == "method" and u.unit_name == "Planet:getMass" for u in units)


# ---------------------------------------------------------------------------
# Java parser — field values
# ---------------------------------------------------------------------------


def test_parse_java_char_offset():
    source = "/* comment */\npublic class Foo {}\n"
    units = parse_java(source)
    assert len(units) == 1
    assert units[0].char_offset == source.index("public class Foo")


def test_parse_java_content_preserved():
    source = textwrap.dedent("""\
        public class App {
            public static void main(String[] args) {
                System.out.println("Hello");
            }
        }
    """)
    units = parse_java(source)
    cls = next(u for u in units if u.unit_type == "class")
    assert "public class App" in cls.content


def test_parse_java_content_md5():
    source = "public class Empty {}\n"
    units = parse_java(source)
    expected_md5 = hashlib.md5(units[0].content.encode()).hexdigest()
    assert units[0].content_md5 == expected_md5


def test_parse_java_empty_source():
    assert parse_java("") == []


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


def test_parse_java_missing_treesitter_returns_empty(monkeypatch):
    import mcp_rag.parsers as parsers_mod

    monkeypatch.setattr(parsers_mod, "_get_ts_java_language", lambda: None)
    with pytest.warns(UserWarning, match="tree-sitter-java"):
        result = parse_java("public class Foo {}\n")
    assert result == []


# ---------------------------------------------------------------------------
# parse_file dispatch
# ---------------------------------------------------------------------------


def test_parse_file_java_extension(tmp_path):
    f = tmp_path / "Main.java"
    f.write_text("public class Main { public static void main(String[] args) {} }\n")
    units = parse_file(f)
    assert any(u.unit_name == "Main" for u in units)


def test_parse_file_java_binary_skipped(tmp_path):
    f = tmp_path / "Bad.java"
    f.write_bytes(b"\x00" * 20 + b"public class Bad {}\n")
    assert parse_file(f) == []
