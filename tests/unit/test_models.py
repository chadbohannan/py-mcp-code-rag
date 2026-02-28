"""Unit tests for mcp_rag.models — SemanticUnit dataclass."""
import hashlib

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
    unit = SemanticUnit(unit_type="function", unit_name="foo", content=content, char_offset=0)
    expected = hashlib.md5(content.encode()).hexdigest()
    assert unit.content_md5 == expected


def test_content_md5_changes_with_content():
    unit_a = SemanticUnit(unit_type="function", unit_name="foo", content="def foo(): pass", char_offset=0)
    unit_b = SemanticUnit(unit_type="function", unit_name="foo", content="def foo(): return 1", char_offset=0)
    assert unit_a.content_md5 != unit_b.content_md5


def test_unit_name_nullable():
    unit = SemanticUnit(unit_type="sql", unit_name=None, content="SELECT 1", char_offset=0)
    assert unit.unit_name is None


def test_semantic_unit_key_tuple_usable_as_dict_key():
    unit = SemanticUnit(unit_type="method", unit_name="save", content="def save(self): ...", char_offset=42)
    key = (unit.unit_type, unit.unit_name, unit.char_offset)
    d = {key: unit}
    assert d[("method", "save", 42)] is unit


def test_semantic_unit_equality():
    u1 = SemanticUnit(unit_type="function", unit_name="bar", content="def bar(): ...", char_offset=10)
    u2 = SemanticUnit(unit_type="function", unit_name="bar", content="def bar(): ...", char_offset=10)
    assert u1 == u2


def test_semantic_unit_summary_defaults_to_empty():
    unit = SemanticUnit(unit_type="function", unit_name="x", content="def x(): pass", char_offset=0)
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
