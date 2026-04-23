"""Write a slice-scoped candidate variant file from a prompt spec.

A slice-scoped variant handles exactly one slice's ``unit_type``s. The
generator produces instruction text per unit_type; this helper wraps it
into a loadable variant file at ``eval/variants/<candidate_id>.py``.

Generated variants declare ``SLICE = "<slice_name>"`` so composite.py
rejects them if they're ever mapped to the wrong slice's CHAMPION slot.
They also record ``PARENT_RECEIPT`` (the champion receipt the candidate
was mutated from) and ``GENERATED_AT`` for lineage auditing.

Design guarantees
-----------------
- **Ruff-clean output.** String literals are emitted via ``json.dumps``
  (always double-quoted, always correctly escaped). The ``_PROMPTS``
  dict is rendered with trailing commas. Generated files pass
  ``ruff format --check`` without post-processing.
- **Reserved names guarded.** Candidate ids that collide with variant
  infrastructure modules (``composite``, ``make_candidate``, etc.) are
  refused so a generator can't accidentally overwrite them.
- **Deterministic output.** Dict entries are sorted by unit_type so a
  re-run with identical inputs produces byte-identical output
  (modulo ``GENERATED_AT``), which is easier to review.

Intended caller: the prompt-mutation generator (outer-loop). Also
usable by hand for quick experiments.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from eval.harness.slices import SLICES

VARIANTS_DIR = Path(__file__).resolve().parent
_ID_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# Module names in eval/variants/ that are infrastructure, not candidates.
# A generator using a reserved id would overwrite these files. Keep in
# sync with the directory — add new infra modules here when they land.
RESERVED_IDS: frozenset[str] = frozenset(
    {"composite", "make_candidate", "v0_baseline", "__init__"}
)


def _render(
    candidate_id: str,
    slice_name: str,
    prompts: dict[str, str],
    description: str,
    parent_receipt: str | None,
    generated_at: str,
) -> str:
    dict_lines = ["_PROMPTS: dict[str, str] = {"]
    for unit_type in sorted(prompts):
        dict_lines.append(
            f"    {json.dumps(unit_type)}: {json.dumps(prompts[unit_type])},"
        )
    dict_lines.append("}")
    prompts_block = "\n".join(dict_lines)

    parent_literal = (
        json.dumps(parent_receipt) if parent_receipt is not None else "None"
    )
    docstring_desc = description or "No description provided."

    return f'''"""Generated slice-scoped candidate: {candidate_id}.

{docstring_desc}
"""

from __future__ import annotations

from mcp_rag.models import SemanticUnit

ID = {json.dumps(candidate_id)}
DESCRIPTION = {json.dumps(description)}
SLICE = {json.dumps(slice_name)}
PARENT_RECEIPT: str | None = {parent_literal}
GENERATED_AT = {json.dumps(generated_at)}

{prompts_block}


def build_prompt(unit: SemanticUnit) -> str:
    instruction = _PROMPTS.get(unit.unit_type)
    if instruction is None:
        raise RuntimeError(
            f"variant {{ID!r}} (slice={{SLICE!r}}) has no prompt for "
            f"unit_type {{unit.unit_type!r}}"
        )
    return f"{{instruction}}\\n\\n{{unit.content}}"
'''


def make_candidate(
    candidate_id: str,
    slice_name: str,
    prompts: dict[str, str],
    description: str = "",
    parent_receipt: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a slice-scoped variant file, return its path.

    ``prompts`` must cover every ``unit_type`` in ``SLICES[slice_name]``
    exactly — no missing keys, no extras. The generator has to be
    explicit about which types it intends to handle.

    ``parent_receipt`` is the champion receipt path the candidate was
    mutated from (recorded as ``PARENT_RECEIPT`` in the generated file).
    Pass ``None`` for hand-written experiments with no lineage.
    """
    if not _ID_RE.match(candidate_id):
        raise ValueError(
            f"candidate_id must be a valid Python identifier, "
            f"got {candidate_id!r}"
        )
    if candidate_id in RESERVED_IDS:
        raise ValueError(
            f"candidate_id {candidate_id!r} is reserved for variant "
            f"infrastructure; pick a different id"
        )
    if slice_name not in SLICES:
        raise ValueError(
            f"unknown slice {slice_name!r}; known: {list(SLICES)}"
        )
    expected = set(SLICES[slice_name])
    got = set(prompts)
    if got != expected:
        missing = expected - got
        extra = got - expected
        raise ValueError(
            f"prompts keys must exactly match unit_types in slice "
            f"{slice_name!r}. missing={sorted(missing)} extra={sorted(extra)}"
        )

    out_path = VARIANTS_DIR / f"{candidate_id}.py"
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"{out_path} already exists; pass overwrite=True to replace"
        )
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out_path.write_text(
        _render(
            candidate_id,
            slice_name,
            prompts,
            description,
            parent_receipt,
            generated_at,
        )
    )
    return out_path
