"""Composite variant: dispatches per-slice to each slice's champion.

Rationale
---------
Different slices (``leaf_code``, ``markdown``, ``rollup``) may be won
by different variants. Deploying "just the latest winner" loses the
gains on slices it didn't improve. This variant reads ``eval/CHAMPION``
and routes each unit through the ``build_prompt`` of whichever variant
currently holds that slice's champion.

Build-time resolution
---------------------
The CHAMPION map is read ONCE at import time. Re-indexing with this
variant after a ratchet promotion requires re-importing (fresh CLI
invocation). This is intentional: an index built with ``composite`` is
only meaningful relative to the champion set at build time, so freezing
the resolution there keeps the index reproducible.

Receipts for runs built with composite should be scored and ratcheted
normally — composite can itself become the champion of any slice it
wins, at which point the next composite build pins to the new state.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from mcp_rag.models import SemanticUnit

from eval.harness.slices import SLICES, slice_for_type

ID = "composite"
DESCRIPTION = "Per-slice dispatch to each slice's champion variant."

_CHAMPION_FILE = Path(__file__).resolve().parent.parent / "CHAMPION"
_REPO_ROOT = _CHAMPION_FILE.parent.parent  # paths in CHAMPION are repo-relative


def _load_champion_variant_ids() -> dict[str, str]:
    """Return ``{slice: variant_id}`` read from the champion map."""
    if not _CHAMPION_FILE.exists():
        raise RuntimeError(
            f"composite variant requires {_CHAMPION_FILE} to exist. "
            f"Bootstrap the ratchet first."
        )
    mapping = json.loads(_CHAMPION_FILE.read_text())
    if not isinstance(mapping, dict):
        raise RuntimeError(f"CHAMPION must be a JSON object, got {type(mapping)}")
    out: dict[str, str] = {}
    for slice_name in SLICES:
        receipt_path = mapping.get(slice_name)
        if not receipt_path:
            raise RuntimeError(
                f"CHAMPION has no entry for slice '{slice_name}'. "
                f"Run the ratchet (or --bootstrap) to populate it."
            )
        rp = Path(receipt_path)
        if not rp.is_absolute():
            rp = _REPO_ROOT / rp
        receipt = json.loads(rp.read_text())
        out[slice_name] = receipt["variant_id"]
    return out


def _resolve_builders() -> dict[str, callable]:
    """Map each ``unit_type`` to its winning variant's build_prompt.

    A variant module may declare ``SLICE = "<slice_name>"`` to mark
    itself as slice-scoped. When present, composite validates that the
    variant is being used for the slice it declares — mis-labeling a
    slice-scoped variant as some other slice's champion is a loud error
    at load time rather than a silent miscompile of summaries.
    """
    variant_by_slice = _load_champion_variant_ids()
    loaded: dict[str, callable] = {}
    for slice_name, variant_id in variant_by_slice.items():
        if variant_id == ID:
            raise RuntimeError(
                "composite variant cannot recursively dispatch to itself; "
                "the underlying champion must be a leaf variant."
            )
        module = importlib.import_module(f"eval.variants.{variant_id}")
        declared = getattr(module, "SLICE", None)
        if declared is not None and declared != slice_name:
            raise RuntimeError(
                f"variant '{variant_id}' declares SLICE={declared!r} but "
                f"CHAMPION maps it to slice {slice_name!r}. A slice-scoped "
                f"variant can only be champion of its declared slice."
            )
        loaded[slice_name] = module.build_prompt

    per_type: dict[str, callable] = {}
    for slice_name, types in SLICES.items():
        for t in types:
            per_type[t] = loaded[slice_name]
    return per_type


_BUILDERS = _resolve_builders()


def build_prompt(unit: SemanticUnit) -> str:
    builder = _BUILDERS.get(unit.unit_type)
    if builder is None:
        slice_name = slice_for_type(unit.unit_type)
        raise RuntimeError(
            f"composite: no builder for unit_type '{unit.unit_type}' "
            f"(slice={slice_name})"
        )
    return builder(unit)
