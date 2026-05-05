"""Unit tests for the JavaScript/TypeScript parser integration in mcp_rag.parsers.

Tests use real tree-sitter parsing (no subprocess mocking needed since
the parser runs in-process).
"""

import hashlib
import textwrap

import pytest

from mcp_rag.parsers import parse_javascript, parse_typescript, parse_file


# ---------------------------------------------------------------------------
# JavaScript — functions
# ---------------------------------------------------------------------------


def test_parse_js_function():
    source = textwrap.dedent("""\
        function greet(name) {
            return 'Hello ' + name;
        }
    """)
    units = parse_javascript(source)
    assert len(units) == 1
    assert units[0].unit_type == "function"
    assert units[0].unit_name == "greet"


def test_parse_js_multiple_functions():
    source = textwrap.dedent("""\
        function foo() {}
        function bar() {}
    """)
    units = parse_javascript(source)
    assert len(units) == 2
    names = {u.unit_name for u in units}
    assert names == {"foo", "bar"}


def test_parse_js_arrow_function():
    source = "const add = (a, b) => a + b;\n"
    units = parse_javascript(source)
    assert len(units) == 1
    assert units[0].unit_type == "function"
    assert units[0].unit_name == "add"


def test_parse_js_function_expression():
    source = "const multiply = function(a, b) { return a * b; };\n"
    units = parse_javascript(source)
    assert len(units) == 1
    assert units[0].unit_type == "function"
    assert units[0].unit_name == "multiply"


# ---------------------------------------------------------------------------
# JavaScript — classes
# ---------------------------------------------------------------------------


def test_parse_js_class():
    source = textwrap.dedent("""\
        class Animal {
            constructor(name) {
                this.name = name;
            }
        }
    """)
    units = parse_javascript(source)
    assert any(u.unit_type == "class" and u.unit_name == "Animal" for u in units)


def test_parse_js_class_with_methods():
    source = textwrap.dedent("""\
        class Counter {
            constructor() {
                this.count = 0;
            }
            increment() {
                this.count++;
            }
        }
    """)
    units = parse_javascript(source)
    types = {u.unit_type for u in units}
    assert "class" in types
    assert "method" in types
    method = next(
        u
        for u in units
        if u.unit_type == "method" and "increment" in (u.unit_name or "")
    )
    assert method.unit_name == "Counter:increment"


# ---------------------------------------------------------------------------
# JavaScript — exports
# ---------------------------------------------------------------------------


def test_parse_js_exported_function():
    source = "export function handler() { return 1; }\n"
    units = parse_javascript(source)
    assert len(units) == 1
    assert units[0].unit_type == "function"
    assert units[0].unit_name == "handler"


def test_parse_js_exported_class():
    source = textwrap.dedent("""\
        export class Service {
            run() {}
        }
    """)
    units = parse_javascript(source)
    assert any(u.unit_type == "class" and u.unit_name == "Service" for u in units)


# ---------------------------------------------------------------------------
# JavaScript — field values
# ---------------------------------------------------------------------------


def test_parse_js_char_offset():
    source = "// comment\nfunction foo() { return 0; }\n"
    units = parse_javascript(source)
    assert len(units) == 1
    assert units[0].char_offset == source.index("function foo")


def test_parse_js_content_md5():
    source = "function noop() {}\n"
    units = parse_javascript(source)
    expected_md5 = hashlib.md5(units[0].content.encode()).hexdigest()
    assert units[0].content_md5 == expected_md5


def test_parse_js_empty_source():
    assert parse_javascript("") == []


# ---------------------------------------------------------------------------
# TypeScript — interfaces and types
# ---------------------------------------------------------------------------


def test_parse_ts_interface():
    source = textwrap.dedent("""\
        interface User {
            id: number;
            name: string;
        }
    """)
    units = parse_typescript(source)
    assert len(units) == 1
    assert units[0].unit_type == "interface"
    assert units[0].unit_name == "User"


def test_parse_ts_type_alias():
    source = "type ID = string | number;\n"
    units = parse_typescript(source)
    assert len(units) == 1
    assert units[0].unit_type == "type"
    assert units[0].unit_name == "ID"


def test_parse_ts_enum():
    source = textwrap.dedent("""\
        enum Color {
            Red,
            Green,
            Blue,
        }
    """)
    units = parse_typescript(source)
    assert len(units) == 1
    assert units[0].unit_type == "enum"
    assert units[0].unit_name == "Color"


# ---------------------------------------------------------------------------
# TypeScript — functions and classes
# ---------------------------------------------------------------------------


def test_parse_ts_function():
    source = textwrap.dedent("""\
        function fetchUser(id: number): User {
            return { id, name: "test" };
        }
    """)
    units = parse_typescript(source)
    assert len(units) == 1
    assert units[0].unit_type == "function"
    assert units[0].unit_name == "fetchUser"


def test_parse_ts_class_with_method():
    source = textwrap.dedent("""\
        class UserService {
            getUser(id: number): User {
                return fetchUser(id);
            }
        }
    """)
    units = parse_typescript(source)
    types = {u.unit_type for u in units}
    assert "class" in types
    assert "method" in types
    method = next(u for u in units if u.unit_type == "method")
    assert method.unit_name == "UserService:getUser"


def test_parse_ts_arrow_function():
    source = "const handler = async (req: Request): Promise<Response> => { return new Response('ok'); };\n"
    units = parse_typescript(source)
    assert len(units) == 1
    assert units[0].unit_type == "function"
    assert units[0].unit_name == "handler"


def test_parse_ts_exported_interface():
    source = textwrap.dedent("""\
        export interface Config {
            port: number;
            host: string;
        }
    """)
    units = parse_typescript(source)
    assert len(units) == 1
    assert units[0].unit_type == "interface"
    assert units[0].unit_name == "Config"


def test_parse_ts_empty_source():
    assert parse_typescript("") == []


# ---------------------------------------------------------------------------
# TSX
# ---------------------------------------------------------------------------


def test_parse_tsx_component():
    source = textwrap.dedent("""\
        interface Props {
            name: string;
        }

        export function Greeting({ name }: Props) {
            return <h1>Hello {name}</h1>;
        }
    """)
    units = parse_typescript(source, tsx=True)
    assert any(u.unit_type == "interface" and u.unit_name == "Props" for u in units)
    assert any(u.unit_type == "function" and u.unit_name == "Greeting" for u in units)


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


def test_parse_js_missing_treesitter_returns_empty(monkeypatch):
    import mcp_rag.parsers as parsers_mod

    monkeypatch.setattr(parsers_mod, "_get_ts_javascript_language", lambda: None)
    with pytest.warns(UserWarning, match="tree-sitter-javascript"):
        result = parse_javascript("function foo() {}\n")
    assert result == []


def test_parse_ts_missing_treesitter_returns_empty(monkeypatch):
    import mcp_rag.parsers as parsers_mod

    monkeypatch.setattr(parsers_mod, "_get_ts_typescript_language", lambda: None)
    with pytest.warns(UserWarning, match="tree-sitter-typescript"):
        result = parse_typescript("function foo() {}\n")
    assert result == []


def test_parse_tsx_missing_treesitter_returns_empty(monkeypatch):
    import mcp_rag.parsers as parsers_mod

    monkeypatch.setattr(parsers_mod, "_get_ts_tsx_language", lambda: None)
    with pytest.warns(UserWarning, match="tree-sitter-typescript"):
        result = parse_typescript("function foo() {}\n", tsx=True)
    assert result == []


# ---------------------------------------------------------------------------
# parse_file dispatch
# ---------------------------------------------------------------------------


def test_parse_file_js_extension(tmp_path):
    f = tmp_path / "app.js"
    f.write_text("function main() { return 0; }\n")
    units = parse_file(f)
    assert any(u.unit_name == "main" for u in units)


def test_parse_file_jsx_extension(tmp_path):
    f = tmp_path / "component.jsx"
    f.write_text("function App() { return null; }\n")
    units = parse_file(f)
    assert any(u.unit_name == "App" for u in units)


def test_parse_file_mjs_extension(tmp_path):
    f = tmp_path / "module.mjs"
    f.write_text("export function init() {}\n")
    units = parse_file(f)
    assert any(u.unit_name == "init" for u in units)


def test_parse_file_ts_extension(tmp_path):
    f = tmp_path / "service.ts"
    f.write_text("function serve() {}\n")
    units = parse_file(f)
    assert any(u.unit_name == "serve" for u in units)


def test_parse_file_tsx_extension(tmp_path):
    f = tmp_path / "page.tsx"
    f.write_text("export function Page() { return <div/>; }\n")
    units = parse_file(f)
    assert any(u.unit_name == "Page" for u in units)


def test_parse_file_js_binary_skipped(tmp_path):
    f = tmp_path / "weird.js"
    f.write_bytes(b"\x00" * 20 + b"function main() {}\n")
    assert parse_file(f) == []
