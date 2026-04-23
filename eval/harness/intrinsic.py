"""Intrinsic (label-free) summary-quality metrics.

These are cheap heuristics computed directly over the stored summaries
and their source content. They do not require a query set, and they
run in seconds over the full corpus. They serve two roles:

1. A fast per-commit signal while iterating on prompt variants. Run
   this after re-indexing a subset to see if obvious quality issues
   changed, without spending tokens on the full retrieval eval.
2. Guardrails for the ratchet. A variant that helps retrieval but
   tanks grounding is probably reward-hacking; the ratchet can refuse
   the promotion. (Guardrail integration is a follow-up; this module
   only computes the numbers.)

Metrics
-------
- ``banned_preamble_rate``  : fraction of summaries beginning with a
  wasted lead-in ("This function…", "This method…", etc.).
- ``name_restatement``      : fraction of summary tokens that merely
  restate the unit's name (a summary of ``parsePacketBytes`` that leads
  with "The parsePacketBytes method parses packet bytes" scores high).
- ``symbol_grounding``      : fraction of identifier-shaped tokens in
  the summary that also appear in the unit's source content. Low scores
  flag hallucinations (the "MCP → Microservices" failure pattern).
- ``length_chars`` / ``length_sentences`` : distribution diagnostics.

All metrics aggregate per slice (see eval/harness/slices.py) because
what "good" looks like differs between code, markdown, and rollups.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import statistics
from pathlib import Path
from typing import Any

import sqlite_vec

from eval.harness.slices import SLICES, slice_for_type

BANNED_PREFIXES = (
    "this function",
    "this method",
    "this class",
    "this module",
    "this file",
    "this directory",
    "this interface",
    "this struct",
    "this enum",
    "this test",
    "this script",
    "this code",
)

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|[_\-\s]+")
_SENTENCE_END_RE = re.compile(r"[.!?]+\s+|[.!?]+$")


def banned_preamble(summary: str) -> bool:
    s = summary.strip().lower()
    return s.startswith(BANNED_PREFIXES)


def _split_identifier(name: str) -> list[str]:
    """Split camelCase / snake_case / kebab-case into lowercase words."""
    parts = _CAMEL_SPLIT_RE.split(name)
    return [p.lower() for p in parts if p and len(p) >= 2]


def name_restatement(summary: str, unit_name: str | None) -> float:
    """Fraction of name-words present in the first 10 summary words.

    Intuition: leading a summary with a paraphrase of its own name is
    wasted. Computed over the first 10 words because restatement in
    later sentences is often legitimate context.
    """
    if not unit_name:
        return 0.0
    name_words = set(_split_identifier(unit_name))
    if not name_words:
        return 0.0
    head = summary.strip().lower().split()[:10]
    head_words = {re.sub(r"[^a-z0-9]", "", w) for w in head}
    head_words.discard("")
    hits = name_words & head_words
    return len(hits) / len(name_words)


def _summary_identifiers(summary: str) -> set[str]:
    """Extract identifier-shaped tokens: backticked terms + CamelCase."""
    idents: set[str] = set()
    for m in _BACKTICK_RE.finditer(summary):
        for p in _split_identifier(m.group(1)):
            idents.add(p)
    # CamelCase-looking words: 2+ uppercase runs or mixed case
    for w in re.findall(r"\b[A-Z][A-Za-z0-9]{2,}\b", summary):
        for p in _split_identifier(w):
            idents.add(p)
    return idents


def symbol_grounding(summary: str, source: str) -> float | None:
    """Fraction of summary identifiers present in source.

    Returns ``None`` ("not applicable") when either:
    - the summary contains no identifier-shaped tokens, OR
    - the source is empty — this is the case for rollup units in the
      current DB schema, which build their input text from child
      summaries at index time and do not persist it. To measure rollup
      grounding, the intrinsic runner would need to reconstruct that
      input. Tracked as a known gap.

    ``None`` is distinct from 0.0, which means "identifiers exist but
    none appear in the source" — the canonical hallucination signal.
    """
    idents = _summary_identifiers(summary)
    if not idents:
        return None
    if not source or not source.strip():
        return None
    src_lower = source.lower()
    grounded = sum(1 for i in idents if i in src_lower)
    return grounded / len(idents)


def sentence_count(summary: str) -> int:
    s = summary.strip()
    if not s:
        return 0
    # count sentence terminators; last sentence may lack one
    ends = _SENTENCE_END_RE.findall(s)
    return max(1, len(ends)) if not s.endswith((".", "!", "?")) else max(1, len(ends))


def _open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def compute_intrinsic(db_path: Path) -> dict[str, Any]:
    """Compute per-slice intrinsic aggregates over all non-empty summaries."""
    conn = _open_db(db_path)
    rows = conn.execute(
        "SELECT unit_type, unit_name, content, summary FROM units"
    ).fetchall()

    # per-slice accumulators
    agg: dict[str, dict[str, list]] = {
        s: {
            "banned": [],
            "name_rest": [],
            "grounding": [],
            "length_chars": [],
            "length_sentences": [],
        }
        for s in SLICES
    }
    unmapped: dict[str, int] = {}
    empty = 0

    for unit_type, unit_name, content, summary in rows:
        if not summary:
            empty += 1
            continue
        slice_name = slice_for_type(unit_type)
        if slice_name is None:
            unmapped[unit_type] = unmapped.get(unit_type, 0) + 1
            continue
        bucket = agg[slice_name]
        bucket["banned"].append(1.0 if banned_preamble(summary) else 0.0)
        bucket["name_rest"].append(name_restatement(summary, unit_name))
        g = symbol_grounding(summary, content or "")
        if g is not None:
            bucket["grounding"].append(g)
        bucket["length_chars"].append(len(summary))
        bucket["length_sentences"].append(sentence_count(summary))

    def _p(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        return statistics.quantiles(values, n=100)[int(q) - 1] if len(values) >= 2 else values[0]

    def _mean(values: list[float]) -> float:
        return statistics.fmean(values) if values else 0.0

    out: dict[str, Any] = {"empty_summaries": empty, "unmapped_types": unmapped, "by_slice": {}}
    for slice_name, bucket in agg.items():
        n = len(bucket["banned"])
        grounding_n = len(bucket["grounding"])
        out["by_slice"][slice_name] = {
            "n": n,
            "banned_preamble_rate": _mean(bucket["banned"]),
            "name_restatement_mean": _mean(bucket["name_rest"]),
            "symbol_grounding_mean": (
                _mean(bucket["grounding"]) if grounding_n else None
            ),
            "symbol_grounding_n": grounding_n,
            "length_chars_p50": _p(bucket["length_chars"], 50),
            "length_chars_p95": _p(bucket["length_chars"], 95),
            "length_sentences_mean": _mean(bucket["length_sentences"]),
        }
    return out


def _format(report: dict[str, Any]) -> str:
    lines = []
    if report["unmapped_types"]:
        lines.append(f"!! unmapped unit_types: {report['unmapped_types']}")
    lines.append(f"empty summaries skipped: {report['empty_summaries']}")
    lines.append("")
    lines.append(
        f"{'slice':<11s} {'n':>5s} {'banned':>8s} {'namerest':>9s} "
        f"{'ground':>8s} {'chars_p50':>10s} {'chars_p95':>10s} {'sent_mean':>10s}"
    )
    for slice_name, s in report["by_slice"].items():
        ground = (
            f"{s['symbol_grounding_mean']:>8.3f}"
            if s["symbol_grounding_mean"] is not None
            else f"{'N/A':>8s}"
        )
        lines.append(
            f"{slice_name:<11s} {s['n']:>5d} "
            f"{s['banned_preamble_rate']:>8.3f} "
            f"{s['name_restatement_mean']:>9.3f} "
            f"{ground} "
            f"{s['length_chars_p50']:>10.0f} "
            f"{s['length_chars_p95']:>10.0f} "
            f"{s['length_sentences_mean']:>10.2f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    report = compute_intrinsic(args.db)
    if args.json:
        import json

        print(json.dumps(report, indent=2))
    else:
        print(_format(report))
        if report["unmapped_types"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
