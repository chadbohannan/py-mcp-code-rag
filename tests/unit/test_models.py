"""Unit tests for mcp_rag.models — SemanticUnit dataclass."""

import hashlib
from pathlib import Path

from mcp_rag.models import SemanticUnit


def test_semantic_unit_fields():
    unit = SemanticUnit(
        unit_type="function",
        unit_name="foo",
        content="def foo(): pass",
        char_offset=0,
    )
    assert unit.unit_type == "function"
    assert unit.unit_name == "foo"
    assert unit.content == "def foo(): pass"
    assert unit.char_offset == 0


def test_content_md5_is_derived_from_content():
    content = "def foo(): pass"
    unit = SemanticUnit(
        unit_type="function", unit_name="foo", content=content, char_offset=0
    )
    expected = hashlib.md5(content.encode()).hexdigest()
    assert unit.content_md5 == expected


def test_content_md5_changes_with_content():
    unit_a = SemanticUnit(
        unit_type="function", unit_name="foo", content="def foo(): pass", char_offset=0
    )
    unit_b = SemanticUnit(
        unit_type="function",
        unit_name="foo",
        content="def foo(): return 1",
        char_offset=0,
    )
    assert unit_a.content_md5 != unit_b.content_md5


def test_unit_name_nullable():
    unit = SemanticUnit(
        unit_type="sql", unit_name=None, content="SELECT 1", char_offset=0
    )
    assert unit.unit_name is None


def test_semantic_unit_key_tuple_usable_as_dict_key():
    unit = SemanticUnit(
        unit_type="method",
        unit_name="save",
        content="def save(self): ...",
        char_offset=42,
    )
    key = (unit.unit_type, unit.unit_name, unit.char_offset)
    d = {key: unit}
    assert d[("method", "save", 42)] is unit


def test_semantic_unit_equality():
    u1 = SemanticUnit(
        unit_type="function", unit_name="bar", content="def bar(): ...", char_offset=10
    )
    u2 = SemanticUnit(
        unit_type="function", unit_name="bar", content="def bar(): ...", char_offset=10
    )
    assert u1 == u2


def test_semantic_unit_summary_defaults_to_empty():
    unit = SemanticUnit(
        unit_type="function", unit_name="x", content="def x(): pass", char_offset=0
    )
    assert unit.summary == ""


def test_semantic_unit_summary_settable():
    unit = SemanticUnit(
        unit_type="function",
        unit_name="x",
        content="def x(): pass",
        char_offset=0,
        summary="Does nothing.",
    )
    assert unit.summary == "Does nothing."


# ---------------------------------------------------------------------------
# qualified_path property
# ---------------------------------------------------------------------------


def test_qualified_path_with_file_and_unit_name():
    root = Path("/project")
    unit = SemanticUnit(
        unit_type="method",
        unit_name="Router:send",
        content="def send(): pass",
        char_offset=0,
        file_path=root / "src" / "net.py",
        root=root,
    )
    assert unit.qualified_path == "src/net.py:Router:send"


def test_qualified_path_with_file_no_unit_name():
    root = Path("/project")
    unit = SemanticUnit(
        unit_type="sql",
        unit_name=None,
        content="SELECT 1",
        char_offset=0,
        file_path=root / "query.sql",
        root=root,
    )
    assert unit.qualified_path == "query.sql"


def test_qualified_path_no_file_no_name():
    unit = SemanticUnit(
        unit_type="function",
        unit_name=None,
        content="def f(): pass",
        char_offset=0,
    )
    assert unit.qualified_path == ""


def test_qualified_path_uses_colon_delimiter():
    root = Path("/r")
    unit = SemanticUnit(
        unit_type="function",
        unit_name="foo",
        content="def foo(): pass",
        char_offset=0,
        file_path=root / "mod.py",
        root=root,
    )
    # File path separated from unit name by single colon
    assert unit.qualified_path == "mod.py:foo"
    assert "::" not in unit.qualified_path
