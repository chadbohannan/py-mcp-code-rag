"""Unit tests for mcp_rag.reconcile — unit-level diff logic.

``diff_units`` takes the existing DB units and freshly parsed units, then
returns three buckets:
  - to_keep   : StoredUnit objects that are unchanged (retain row + embedding)
  - to_add    : SemanticUnit objects that are new or changed (insert after summarise+embed)
  - to_delete : StoredUnit objects that are removed or changed (delete from DB)
"""

import hashlib
from pathlib import Path


from mcp_rag.models import SemanticUnit
from mcp_rag.reconcile import StoredUnit, diff_units

_ROOT = Path("/project")
_FILE = _ROOT / "mod.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stored(
    id: int, path: str, content: str, char_offset: int
) -> StoredUnit:
    return StoredUnit(
        id=id,
        path=path,
        content_md5=hashlib.md5(content.encode()).hexdigest(),
        char_offset=char_offset,
    )


def _incoming(
    unit_type: str, unit_name: str | None, content: str, char_offset: int
) -> SemanticUnit:
    return SemanticUnit(
        unit_type=unit_type,
        unit_name=unit_name,
        content=content,
        char_offset=char_offset,
        file_path=_FILE,
        root=_ROOT,
    )


# ---------------------------------------------------------------------------
# Empty cases
# ---------------------------------------------------------------------------


def test_diff_empty_both_sides():
    keep, add, delete = diff_units([], [])
    assert keep == []
    assert add == []
    assert delete == []


def test_diff_all_new_no_existing():
    incoming = [
        _incoming("function", "foo", "def foo(): pass", 0),
        _incoming("function", "bar", "def bar(): pass", 20),
    ]
    keep, add, delete = diff_units([], incoming)
    assert keep == []
    assert len(add) == 2
    assert delete == []


def test_diff_no_incoming_all_deleted():
    existing = [
        _stored(1, "mod.py:foo", "def foo(): pass", 0),
        _stored(2, "mod.py:bar", "def bar(): pass", 20),
    ]
    keep, add, delete = diff_units(existing, [])
    assert keep == []
    assert add == []
    assert len(delete) == 2


# ---------------------------------------------------------------------------
# Unchanged units
# ---------------------------------------------------------------------------


def test_diff_unchanged_unit_kept():
    content = "def foo(): return 1"
    existing = [_stored(7, "mod.py:foo", content, 0)]
    incoming = [_incoming("function", "foo", content, 0)]

    keep, add, delete = diff_units(existing, incoming)
    assert len(keep) == 1
    assert keep[0].id == 7
    assert add == []
    assert delete == []


def test_diff_to_keep_preserves_stored_unit_id():
    content = "def x(): pass"
    existing = [_stored(99, "mod.py:x", content, 5)]
    incoming = [_incoming("function", "x", content, 5)]

    keep, _, _ = diff_units(existing, incoming)
    assert keep[0].id == 99


# ---------------------------------------------------------------------------
# Changed units (same key, different content)
# ---------------------------------------------------------------------------


def test_diff_changed_content_triggers_replacement():
    existing = [_stored(1, "mod.py:compute", "def compute(): return 0", 0)]
    incoming = [_incoming("function", "compute", "def compute(): return 42", 0)]

    keep, add, delete = diff_units(existing, incoming)
    assert keep == []
    assert len(add) == 1
    assert len(delete) == 1
    assert delete[0].id == 1
    assert add[0].unit_name == "compute"


# ---------------------------------------------------------------------------
# Removed units
# ---------------------------------------------------------------------------


def test_diff_removed_unit_deleted():
    existing = [
        _stored(1, "mod.py:keep_me", "def keep_me(): pass", 0),
        _stored(2, "mod.py:removed", "def removed(): pass", 20),
    ]
    incoming = [_incoming("function", "keep_me", "def keep_me(): pass", 0)]

    keep, add, delete = diff_units(existing, incoming)
    assert len(keep) == 1
    assert keep[0].id == 1
    assert add == []
    assert len(delete) == 1
    assert delete[0].id == 2


# ---------------------------------------------------------------------------
# Added units
# ---------------------------------------------------------------------------


def test_diff_added_unit_inserted():
    existing = [_stored(1, "mod.py:old", "def old(): pass", 0)]
    incoming = [
        _incoming("function", "old", "def old(): pass", 0),
        _incoming("function", "new", "def new(): pass", 20),
    ]
    keep, add, delete = diff_units(existing, incoming)
    assert len(keep) == 1
    assert len(add) == 1
    assert add[0].unit_name == "new"
    assert delete == []


# ---------------------------------------------------------------------------
# Key is the (path, char_offset) pair
# ---------------------------------------------------------------------------


def test_diff_key_path_change():
    # Same offset, different qualified path → different key
    content = "def foo(): pass"
    existing = [_stored(1, "mod.py:foo", content, 0)]
    incoming = [_incoming("function", "bar", content, 0)]

    keep, add, delete = diff_units(existing, incoming)
    assert keep == []
    assert len(add) == 1
    assert len(delete) == 1


def test_diff_key_char_offset_change():
    # Same path, different offset → different key
    content = "def foo(): pass"
    existing = [_stored(1, "mod.py:foo", content, 0)]
    incoming = [_incoming("function", "foo", content, 10)]

    keep, add, delete = diff_units(existing, incoming)
    assert keep == []
    assert len(add) == 1
    assert len(delete) == 1


# ---------------------------------------------------------------------------
# Mixed scenario
# ---------------------------------------------------------------------------


def test_diff_multiple_mixed_changes():
    existing = [
        _stored(1, "mod.py:unchanged", "def unchanged(): pass", 0),
        _stored(2, "mod.py:modified", "def modified(): return 0", 30),
        _stored(3, "mod.py:deleted", "def deleted(): pass", 60),
    ]
    incoming = [
        _incoming("function", "unchanged", "def unchanged(): pass", 0),  # same
        _incoming("function", "modified", "def modified(): return 99", 30),  # changed
        _incoming("function", "added", "def added(): pass", 90),  # new
    ]
    keep, add, delete = diff_units(existing, incoming)

    keep_ids = {u.id for u in keep}
    delete_ids = {u.id for u in delete}
    add_names = {u.unit_name for u in add}

    assert keep_ids == {1}  # unchanged retained
    assert delete_ids == {2, 3}  # modified old + deleted
    assert add_names == {"modified", "added"}  # modified new + newly added


# ---------------------------------------------------------------------------
# Null unit_name in key
# ---------------------------------------------------------------------------


def test_diff_null_unit_name_as_key():
    content = "SELECT 1"
    existing = [_stored(5, "mod.py", content, 0)]
    incoming = [_incoming("sql", None, content, 0)]

    keep, add, delete = diff_units(existing, incoming)
    assert len(keep) == 1
    assert keep[0].id == 5
    assert add == []
    assert delete == []
