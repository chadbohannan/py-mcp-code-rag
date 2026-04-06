"""Unit tests for the Go parser (tree-sitter implementation) in mcp_rag.parsers."""

import hashlib
import warnings

import pytest

from mcp_rag.parsers import parse_file, parse_go


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PACKAGE = "package main\n\n"

_FUNC_SRC = _PACKAGE + "func Hello() {}\n"
_METHOD_SRC = _PACKAGE + "func (s *Srv) Run() {}\n"
_STRUCT_SRC = _PACKAGE + "type Config struct { Port int }\n"
_INTERFACE_SRC = _PACKAGE + "type Handler interface { Handle() }\n"
_MULTI_SRC = (
    _PACKAGE
    + "type Server struct { port int }\n"
    + "func New() *Server { return nil }\n"
)


# ---------------------------------------------------------------------------
# Empty / no-op inputs
# ---------------------------------------------------------------------------


def test_parse_go_empty_string_returns_empty():
    assert parse_go("") == []


def test_parse_go_package_only_returns_empty():
    assert parse_go("package main\n") == []


# ---------------------------------------------------------------------------
# Missing tree-sitter-go
# ---------------------------------------------------------------------------


def test_parse_go_missing_package_warns(monkeypatch):
    monkeypatch.setattr("mcp_rag.parsers._get_ts_go_language", lambda: None)
    with pytest.warns(UserWarning, match="tree-sitter-go"):
        result = parse_go(_FUNC_SRC)
    assert result == []


# ---------------------------------------------------------------------------
# Unit types
# ---------------------------------------------------------------------------


def test_parse_go_returns_function():
    units = parse_go(_FUNC_SRC)
    assert len(units) == 1
    assert units[0].unit_type == "function"
    assert units[0].unit_name == "Hello"


def test_parse_go_returns_method():
    units = parse_go(_METHOD_SRC)
    assert len(units) == 1
    assert units[0].unit_type == "method"
    assert units[0].unit_name == "Srv:Run"


def test_parse_go_returns_struct():
    units = parse_go(_STRUCT_SRC)
    assert len(units) == 1
    assert units[0].unit_type == "struct"
    assert units[0].unit_name == "Config"


def test_parse_go_returns_interface():
    units = parse_go(_INTERFACE_SRC)
    assert len(units) == 1
    assert units[0].unit_type == "interface"
    assert units[0].unit_name == "Handler"


# ---------------------------------------------------------------------------
# Field values
# ---------------------------------------------------------------------------


def test_parse_go_content_preserved():
    units = parse_go(_FUNC_SRC)
    assert "func Hello()" in units[0].content


def test_parse_go_char_offset_nonzero():
    units = parse_go(_FUNC_SRC)
    assert units[0].char_offset > 0


def test_parse_go_content_md5_set():
    units = parse_go(_FUNC_SRC)
    expected = hashlib.md5(units[0].content.encode()).hexdigest()
    assert units[0].content_md5 == expected


def test_parse_go_multiple_units():
    units = parse_go(_MULTI_SRC)
    assert len(units) == 2
    types = {u.unit_type for u in units}
    assert types == {"struct", "function"}


def test_parse_go_units_sorted_by_offset():
    units = parse_go(_MULTI_SRC)
    offsets = [u.char_offset for u in units]
    assert offsets == sorted(offsets)


# ---------------------------------------------------------------------------
# parse_file dispatch
# ---------------------------------------------------------------------------


def test_parse_file_go_dispatches_to_go_parser(tmp_path):
    f = tmp_path / "main.go"
    f.write_text("package main\nfunc main() {}\n")
    units = parse_file(f)
    assert any(u.unit_name == "main" for u in units)


def test_parse_file_go_binary_skipped(tmp_path):
    f = tmp_path / "weird.go"
    f.write_bytes(b"\x00" * 20 + b"package main\n")
    assert parse_file(f) == []
