"""Unit-level reconciliation diff for incremental indexing.

``diff_units`` compares the units already stored in the DB against the freshly
parsed units from a file and returns three buckets:

- to_keep   — StoredUnit rows whose key AND content_md5 are unchanged; retain
              the existing DB row and embedding as-is.
- to_add    — SemanticUnit objects that are new or whose content changed;
              caller must summarise, embed, and insert them.
- to_delete — StoredUnit rows that have no matching incoming unit, or whose
              content changed (old row replaced by new); caller must DELETE.

The reconciliation key is the triple ``(unit_type, unit_name, char_offset)``.
"""
from dataclasses import dataclass

from mcp_rag.models import SemanticUnit

# Type alias for the reconciliation key
_Key = tuple[str, str | None, int]


@dataclass
class StoredUnit:
    """Lightweight representation of a unit row already in the DB."""
    id: int
    unit_type: str
    unit_name: str | None
    content_md5: str
    char_offset: int


def _key(unit_type: str, unit_name: str | None, char_offset: int) -> _Key:
    return (unit_type, unit_name, char_offset)


def diff_units(
    existing: list[StoredUnit],
    incoming: list[SemanticUnit],
) -> tuple[list[StoredUnit], list[SemanticUnit], list[StoredUnit]]:
    """Compute the diff between stored and freshly parsed units.

    Returns ``(to_keep, to_add, to_delete)``.
    """
    # Build a lookup of existing units by reconciliation key
    existing_by_key: dict[_Key, StoredUnit] = {
        _key(u.unit_type, u.unit_name, u.char_offset): u
        for u in existing
    }

    to_keep: list[StoredUnit] = []
    to_add: list[SemanticUnit] = []
    to_delete: list[StoredUnit] = []
    matched_keys: set[_Key] = set()

    for new_unit in incoming:
        k = _key(new_unit.unit_type, new_unit.unit_name, new_unit.char_offset)
        stored = existing_by_key.get(k)

        if stored is not None and stored.content_md5 == new_unit.content_md5:
            # Identical — keep existing row and embedding
            to_keep.append(stored)
        else:
            # New or content-changed — needs summarise + embed + insert
            to_add.append(new_unit)
            if stored is not None:
                # Old row must be removed before the new one is inserted
                to_delete.append(stored)

        matched_keys.add(k)

    # Any existing unit not matched by an incoming unit has been removed
    for stored in existing:
        k = _key(stored.unit_type, stored.unit_name, stored.char_offset)
        if k not in matched_keys:
            to_delete.append(stored)

    return to_keep, to_add, to_delete
