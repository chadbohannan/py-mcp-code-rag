"""Slice definitions: groups of unit_types that share a prompting seam.

A slice is the unit of per-type championing. Each slice has its own
champion variant and its own per-slice retrieval metrics. A candidate
variant can win one slice while leaving others unchanged.

Design notes
------------
- ``leaf_code``: source-code units whose input to the summarizer is raw
  source text. All share the "code → English" task.
- ``markdown``: prose and heading units whose input is authored natural
  language. The task is compression, not translation.
- ``rollup``: units whose input is *other summaries* (module and
  directory rollups). The task is synthesis over already-summarized
  children; fundamentally different from leaf summarization.

Bucketing rule
--------------
Every ``unit_type`` present in the corpus must belong to exactly one
slice. When the parser adds a new unit_type, add it here explicitly —
the harness refuses to score unknown types so regressions are loud.

A query contributes to a slice if ANY of its ``must_include`` paths has
a type in that slice. One query can contribute to multiple slices,
each with its own best-rank computation restricted to in-slice answers.
"""

from __future__ import annotations

SLICES: dict[str, frozenset[str]] = {
    "leaf_code": frozenset(
        {"function", "method", "class", "struct", "interface", "enum"}
    ),
    "markdown": frozenset({"paragraph", "h1", "h2", "h3", "h4", "h5", "h6"}),
    "rollup": frozenset({"module", "directory"}),
}

# Every unit_type in the corpus should map into exactly one slice.
_TYPE_TO_SLICE: dict[str, str] = {
    t: name for name, types in SLICES.items() for t in types
}


def slice_for_type(unit_type: str) -> str | None:
    """Return the slice name for a unit_type, or None if unmapped.

    Unmapped types should be audited: either assign them to a slice or
    extend the mapping. The scorer treats unmapped types as a hard error.
    """
    return _TYPE_TO_SLICE.get(unit_type)


# Minimum per-slice sample sizes before a slice's ratchet is armed.
# Below this, the harness reports the metric but refuses to promote.
MIN_DEV_SLICE_N = 5
MIN_HELDOUT_SLICE_N = 20
