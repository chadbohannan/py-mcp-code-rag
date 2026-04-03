"""Unit tests for the Go parser integration in mcp_rag.parsers.

subprocess.run and shutil.which are monkeypatched so no real Go subprocess
is spawned.  The _go_warned module flag is reset before each test via autouse
fixture so warning-once logic is deterministic.
"""

import hashlib
import json
import logging
from unittest.mock import MagicMock

import pytest

import mcp_rag.parsers as parsers_module
from mcp_rag.parsers import parse_file, parse_go


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_ok(stdout: str = "[]") -> MagicMock:
    r = MagicMock()
    r.returncode = 0
    r.stdout = stdout
    r.stderr = ""
    return r


def _run_fail(stderr: str = "parse error") -> MagicMock:
    r = MagicMock()
    r.returncode = 1
    r.stdout = ""
    r.stderr = stderr
    return r


def _units(*items) -> str:
    return json.dumps(list(items))


def _unit(
    unit_type="function", unit_name="Foo", content="func Foo() {}", char_offset=0
):
    return {
        "unit_type": unit_type,
        "unit_name": unit_name,
        "content": content,
        "char_offset": char_offset,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_go_warned():
    """Reset _go_warned before/after each test to isolate warning-once logic."""
    parsers_module._go_warned = False
    yield
    parsers_module._go_warned = False


@pytest.fixture
def go_available(monkeypatch):
    monkeypatch.setattr(
        "shutil.which", lambda cmd: "/usr/bin/go" if cmd == "go" else None
    )


@pytest.fixture
def go_missing(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda cmd: None)


# ---------------------------------------------------------------------------
# go not in PATH
# ---------------------------------------------------------------------------


def test_parse_go_missing_binary_returns_empty(tmp_path, go_missing):
    f = tmp_path / "main.go"
    f.write_text("package main\n")
    assert parse_go(f) == []


def test_parse_go_missing_binary_logs_warning(tmp_path, go_missing, caplog):
    f = tmp_path / "main.go"
    f.write_text("package main\n")
    with caplog.at_level(logging.WARNING, logger="mcp_rag.parsers"):
        parse_go(f)
    assert "go" in caplog.text.lower()
    assert "PATH" in caplog.text


def test_parse_go_missing_binary_warns_only_once(tmp_path, go_missing, caplog):
    f = tmp_path / "main.go"
    f.write_text("package main\n")
    with caplog.at_level(logging.WARNING, logger="mcp_rag.parsers"):
        parse_go(f)
        parse_go(f)
    assert caplog.text.count("PATH") == 1


# ---------------------------------------------------------------------------
# subprocess failure
# ---------------------------------------------------------------------------


def test_parse_go_subprocess_failure_returns_empty(tmp_path, go_available, monkeypatch):
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: _run_fail())
    f = tmp_path / "bad.go"
    f.write_text("package main\nsyntax {{{\n")
    assert parse_go(f) == []


def test_parse_go_subprocess_failure_logs_warning(
    tmp_path, go_available, monkeypatch, caplog
):
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: _run_fail())
    f = tmp_path / "bad.go"
    f.write_text("package main\n")
    with caplog.at_level(logging.WARNING, logger="mcp_rag.parsers"):
        parse_go(f)
    assert "bad.go" in caplog.text


# ---------------------------------------------------------------------------
# Successful parse — unit types
# ---------------------------------------------------------------------------


def test_parse_go_returns_function(tmp_path, go_available, monkeypatch):
    data = _units(
        _unit(unit_type="function", unit_name="Hello", content="func Hello() {}")
    )
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: _run_ok(data))
    f = tmp_path / "hello.go"
    f.write_text("package main\n")
    units = parse_go(f)
    assert len(units) == 1
    assert units[0].unit_type == "function"
    assert units[0].unit_name == "Hello"


def test_parse_go_returns_method(tmp_path, go_available, monkeypatch):
    data = _units(
        _unit(unit_type="method", unit_name="Run", content="func (s *Srv) Run() {}")
    )
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: _run_ok(data))
    f = tmp_path / "srv.go"
    f.write_text("package main\n")
    units = parse_go(f)
    assert units[0].unit_type == "method"


def test_parse_go_returns_struct(tmp_path, go_available, monkeypatch):
    data = _units(
        _unit(unit_type="struct", unit_name="Config", content="type Config struct {}")
    )
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: _run_ok(data))
    f = tmp_path / "cfg.go"
    f.write_text("package main\n")
    units = parse_go(f)
    assert units[0].unit_type == "struct"


def test_parse_go_returns_interface(tmp_path, go_available, monkeypatch):
    data = _units(
        _unit(
            unit_type="interface",
            unit_name="Handler",
            content="type Handler interface {}",
        )
    )
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: _run_ok(data))
    f = tmp_path / "iface.go"
    f.write_text("package main\n")
    units = parse_go(f)
    assert units[0].unit_type == "interface"


# ---------------------------------------------------------------------------
# Successful parse — field values
# ---------------------------------------------------------------------------


def test_parse_go_null_unit_name(tmp_path, go_available, monkeypatch):
    item = _unit()
    item["unit_name"] = None
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: _run_ok(json.dumps([item])))
    f = tmp_path / "anon.go"
    f.write_text("package main\n")
    units = parse_go(f)
    assert units[0].unit_name is None


def test_parse_go_char_offset_preserved(tmp_path, go_available, monkeypatch):
    data = _units(_unit(char_offset=128))
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: _run_ok(data))
    f = tmp_path / "off.go"
    f.write_text("package main\n")
    units = parse_go(f)
    assert units[0].char_offset == 128


def test_parse_go_content_preserved(tmp_path, go_available, monkeypatch):
    content = "func Bar(x int) int { return x * 2 }"
    data = _units(_unit(content=content))
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: _run_ok(data))
    f = tmp_path / "bar.go"
    f.write_text("package main\n")
    units = parse_go(f)
    assert units[0].content == content


def test_parse_go_content_md5_set(tmp_path, go_available, monkeypatch):
    content = "func Baz() {}"
    data = _units(_unit(content=content))
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: _run_ok(data))
    f = tmp_path / "baz.go"
    f.write_text("package main\n")
    units = parse_go(f)
    assert units[0].content_md5 == hashlib.md5(content.encode()).hexdigest()


def test_parse_go_multiple_units(tmp_path, go_available, monkeypatch):
    data = _units(
        _unit(unit_type="struct", unit_name="Server", content="type Server struct {}"),
        _unit(
            unit_type="function",
            unit_name="New",
            content="func New() *Server { return nil }",
            char_offset=30,
        ),
    )
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: _run_ok(data))
    f = tmp_path / "server.go"
    f.write_text("package main\n")
    units = parse_go(f)
    assert len(units) == 2


def test_parse_go_empty_result(tmp_path, go_available, monkeypatch):
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: _run_ok("[]"))
    f = tmp_path / "pkg.go"
    f.write_text("package main\n")
    assert parse_go(f) == []


# ---------------------------------------------------------------------------
# parse_file dispatch
# ---------------------------------------------------------------------------


def test_parse_file_go_dispatches_to_go_parser(tmp_path, go_available, monkeypatch):
    data = _units(
        _unit(unit_type="function", unit_name="Main", content="func main() {}")
    )
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: _run_ok(data))
    f = tmp_path / "main.go"
    f.write_text("package main\nfunc main() {}\n")
    units = parse_file(f)
    assert any(u.unit_name == "Main" for u in units)


def test_parse_file_go_binary_skipped(tmp_path, go_available, monkeypatch):
    """A .go file with null bytes is detected as binary and skipped."""
    f = tmp_path / "weird.go"
    f.write_bytes(b"\x00" * 20 + b"package main\n")
    units = parse_file(f)
    assert units == []
