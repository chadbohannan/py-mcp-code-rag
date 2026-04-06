"""Unit-level reconciliation diff for incremental indexing.

``diff_units`` compares the units already stored in the DB against the freshly
parsed units from a file and returns three buckets:

- to_keep   — StoredUnit rows whose key AND content_md5 are unchanged; retain
              the existing DB row and embedding as-is.
- to_add    — SemanticUnit objects that are new or whose content changed;
              caller must summarise, embed, and insert them.
- to_delete — StoredUnit rows that have no matching incoming unit, or whose
              content changed (old row replaced by new); caller must DELETE.

The reconciliation key is the pair ``(path, char_offset)``.
"""

from dataclasses import dataclass

from mcp_rag.models import SemanticUnit

# Type alias for the reconciliation key
_Key = tuple[str, int]


@dataclass
class StoredUnit:
    """Lightweight representation of a unit row already in the DB."""

    id: int
    path: str
    content_md5: str
    char_offset: int


def diff_units(
    existing: list[StoredUnit],
    incoming: list[SemanticUnit],
) -> tuple[list[StoredUnit], list[SemanticUnit], list[StoredUnit]]:
    """Compute the diff between stored and freshly parsed units.

    Returns ``(to_keep, to_add, to_delete)``.
    """
    existing_by_key: dict[_Key, StoredUnit] = {
        (u.path, u.char_offset): u for u in existing
    }

    to_keep: list[StoredUnit] = []
    to_add: list[SemanticUnit] = []
    to_delete: list[StoredUnit] = []
    matched_keys: set[_Key] = set()

    for new_unit in incoming:
        k: _Key = (new_unit.qualified_path, new_unit.char_offset)
        stored = existing_by_key.get(k)

        if stored is not None and stored.content_md5 == new_unit.content_md5:
            to_keep.append(stored)
        else:
            to_add.append(new_unit)
            if stored is not None:
                to_delete.append(stored)

        matched_keys.add(k)

    for stored in existing:
        k = (stored.path, stored.char_offset)
        if k not in matched_keys:
            to_delete.append(stored)

    return to_keep, to_add, to_delete
