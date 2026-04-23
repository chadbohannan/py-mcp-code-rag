"""Score a variant's index against the eval query sets.

Writes a machine-readable run receipt to eval/runs/ that downstream
tooling (ratchet.py, analysis) consumes.

Usage:
    uv run python -m eval.harness.score \\
        --db aln.db \\
        --variant-id v0_baseline

Scoring rules:
- ``must_include`` treated as a set. A query is a "hit" if any member
  appears in top_k. MRR uses the rank of the best-ranked member; 0 when
  no member is in top_k.
- Per-slice scoring: each query contributes to every slice for which it
  has at least one ``must_include`` answer of a type in that slice. The
  slice's rank is the best rank among in-slice answers.
- ``must_not_include`` scored as adversarial false-positive rate: the
  fraction of queries where any listed path appears in top_k.
- Smoke queries pass when a must_include member ranks #1.

The receipt includes per-query detail (including per-hit unit_type and
per-slice rank) so ratchet.py can bootstrap CIs per slice without
re-running retrieval.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import sqlite_vec

from eval.harness.intrinsic import compute_intrinsic
from eval.harness.slices import SLICES, slice_for_type
from mcp_rag.embedder import FastEmbedder
from mcp_rag.queries import search

EVAL_DIR = Path(__file__).resolve().parent.parent
QUERIES_DIR = EVAL_DIR / "queries"
RUNS_DIR = EVAL_DIR / "runs"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _lookup_types(conn: sqlite3.Connection, paths: list[str]) -> dict[str, str]:
    """Fetch unit_type for a list of qualified paths. Missing paths are
    silently skipped — a must_include path that no longer exists in the
    index is a labeling issue, not a scoring crash.
    """
    if not paths:
        return {}
    placeholders = ",".join("?" for _ in paths)
    rows = conn.execute(
        f"SELECT path, unit_type FROM units WHERE path IN ({placeholders})",
        paths,
    ).fetchall()
    return {p: t for p, t in rows}


def _best_rank_in_paths(
    hit_paths: list[str], target_paths: set[str]
) -> int:
    """1-indexed rank of best-ranked target in hit_paths, 0 if none."""
    for i, p in enumerate(hit_paths, 1):
        if p in target_paths:
            return i
    return 0


def _annotate_hits(
    conn: sqlite3.Connection, hits: list[dict]
) -> tuple[list[str], list[str]]:
    """Return parallel lists of hit paths and their unit_types."""
    paths = [h["path"] for h in hits]
    types = _lookup_types(conn, paths)
    return paths, [types.get(p, "UNKNOWN") for p in paths]


def _score_retrieval(
    conn: sqlite3.Connection,
    embedder: FastEmbedder,
    queries: list[dict],
) -> dict[str, Any]:
    """Score a must_include-style split (dev, heldout) with per-slice breakdown."""
    per_query = []
    for q in queries:
        hits = search(conn, embedder, q["query"], top_k=q.get("top_k", 10))
        hit_paths, _hit_types = _annotate_hits(conn, hits)

        mi_paths = q.get("must_include", []) or []
        mi_types = _lookup_types(conn, mi_paths)

        overall_rank = _best_rank_in_paths(hit_paths, set(mi_paths))

        slice_ranks: dict[str, dict[str, Any]] = {}
        for slice_name, types in SLICES.items():
            in_slice = {
                p for p in mi_paths if mi_types.get(p) in types
            }
            if not in_slice:
                continue  # query doesn't contribute to this slice
            r = _best_rank_in_paths(hit_paths, in_slice)
            slice_ranks[slice_name] = {
                "rank": r,
                "reciprocal_rank": 1.0 / r if r > 0 else 0.0,
                "in_slice_n": len(in_slice),
            }

        adv_hit = any(
            p in set(q.get("must_not_include", []) or []) for p in hit_paths
        )

        per_query.append(
            {
                "id": q["id"],
                "rank": overall_rank,
                "reciprocal_rank": 1.0 / overall_rank if overall_rank > 0 else 0.0,
                "recall_at_1": overall_rank == 1,
                "recall_at_5": 0 < overall_rank <= 5,
                "recall_at_10": 0 < overall_rank <= 10,
                "adversarial_hit": adv_hit,
                "top_paths": hit_paths,
                "must_include_types": {p: mi_types.get(p, "UNKNOWN") for p in mi_paths},
                "slice_ranks": slice_ranks,
            }
        )

    n = len(per_query) or 1
    overall = {
        "n": len(per_query),
        "mrr_at_10": sum(r["reciprocal_rank"] for r in per_query) / n,
        "recall_at_1": sum(r["recall_at_1"] for r in per_query) / n,
        "recall_at_5": sum(r["recall_at_5"] for r in per_query) / n,
        "recall_at_10": sum(r["recall_at_10"] for r in per_query) / n,
        "adversarial_fpr": sum(r["adversarial_hit"] for r in per_query) / n,
    }

    by_slice: dict[str, dict[str, Any]] = {}
    for slice_name in SLICES:
        pq_in = [
            r for r in per_query if slice_name in r["slice_ranks"]
        ]
        sn = len(pq_in) or 1
        if pq_in:
            by_slice[slice_name] = {
                "n": len(pq_in),
                "mrr_at_10": sum(
                    r["slice_ranks"][slice_name]["reciprocal_rank"] for r in pq_in
                ) / sn,
                "recall_at_1": sum(
                    r["slice_ranks"][slice_name]["rank"] == 1 for r in pq_in
                ) / sn,
                "recall_at_10": sum(
                    0 < r["slice_ranks"][slice_name]["rank"] <= 10 for r in pq_in
                ) / sn,
            }
        else:
            by_slice[slice_name] = {
                "n": 0,
                "mrr_at_10": 0.0,
                "recall_at_1": 0.0,
                "recall_at_10": 0.0,
            }

    return {**overall, "by_slice": by_slice, "per_query": per_query}


def _score_adversarial(
    conn: sqlite3.Connection,
    embedder: FastEmbedder,
    queries: list[dict],
) -> dict[str, Any]:
    per_query = []
    for q in queries:
        hits = search(conn, embedder, q["query"], top_k=q.get("top_k", 10))
        hit_paths = [h["path"] for h in hits]
        adv_hit = any(
            p in set(q.get("must_not_include", []) or []) for p in hit_paths
        )
        per_query.append(
            {"id": q["id"], "adversarial_hit": adv_hit, "top_paths": hit_paths}
        )
    n = len(per_query) or 1
    return {
        "n": len(per_query),
        "adversarial_fpr": sum(r["adversarial_hit"] for r in per_query) / n,
        "per_query": per_query,
    }


def _score_smoke(
    conn: sqlite3.Connection,
    embedder: FastEmbedder,
    queries: list[dict],
) -> dict[str, Any]:
    per_query = []
    for q in queries:
        hits = search(conn, embedder, q["query"], top_k=q.get("top_k", 3))
        hit_paths = [h["path"] for h in hits]
        rank = _best_rank_in_paths(hit_paths, set(q.get("must_include", []) or []))
        per_query.append(
            {
                "id": q["id"],
                "rank": rank,
                "passed": rank == 1,
                "top_paths": hit_paths,
            }
        )
    n = len(per_query) or 1
    return {
        "n": len(per_query),
        "pass_rate": sum(r["passed"] for r in per_query) / n,
        "per_query": per_query,
    }


def _git_sha(repo_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _corpus_shas(conn: sqlite3.Connection) -> dict[str, str | None]:
    rows = conn.execute("SELECT name, root FROM repos ORDER BY name").fetchall()
    return {name: _git_sha(Path(root)) for name, root in rows}


def _audit_unknown_types(conn: sqlite3.Connection) -> list[str]:
    """Return unit_types present in the corpus but not mapped to any slice."""
    rows = conn.execute(
        "SELECT DISTINCT unit_type FROM units WHERE unit_type IS NOT NULL"
    ).fetchall()
    return sorted(t for (t,) in rows if slice_for_type(t) is None)


def _open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def run_score(db_path: Path, variant_id: str, out_path: Path) -> dict[str, Any]:
    conn = _open_db(db_path)
    embedder = FastEmbedder()

    unknown = _audit_unknown_types(conn)
    if unknown:
        raise RuntimeError(
            f"unit_types present in index but not mapped to a slice: {unknown}. "
            f"Add them to eval/harness/slices.py before scoring."
        )

    dev = _load_jsonl(QUERIES_DIR / "dev.jsonl")
    heldout = _load_jsonl(QUERIES_DIR / "heldout.jsonl")
    adversarial = _load_jsonl(QUERIES_DIR / "adversarial.jsonl")
    smoke = _load_jsonl(QUERIES_DIR / "smoke.jsonl")

    receipt: dict[str, Any] = {
        "variant_id": variant_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "db_path": str(db_path),
        "db_mtime": db_path.stat().st_mtime,
        "corpus": _corpus_shas(conn),
        "slices": list(SLICES.keys()),
        "splits": {
            "dev": _score_retrieval(conn, embedder, dev),
            "heldout": _score_retrieval(conn, embedder, heldout),
            "adversarial": _score_adversarial(conn, embedder, adversarial),
            "smoke": _score_smoke(conn, embedder, smoke),
        },
        "intrinsic": compute_intrinsic(db_path),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def _format_summary(receipt: dict[str, Any]) -> str:
    dev = receipt["splits"]["dev"]
    adv = receipt["splits"]["adversarial"]
    smoke = receipt["splits"]["smoke"]
    held = receipt["splits"]["heldout"]
    lines = [
        f"variant: {receipt['variant_id']}",
        f"corpus:  {receipt['corpus']}",
        "",
        f"dev         (n={dev['n']:>3}): MRR@10={dev['mrr_at_10']:.3f}  "
        f"R@1={dev['recall_at_1']:.3f}  R@10={dev['recall_at_10']:.3f}",
    ]
    for slice_name in SLICES:
        s = dev["by_slice"][slice_name]
        lines.append(
            f"  └─ {slice_name:10s} (n={s['n']:>3}): MRR@10={s['mrr_at_10']:.3f}  "
            f"R@1={s['recall_at_1']:.3f}  R@10={s['recall_at_10']:.3f}"
        )
    lines.append(
        f"heldout     (n={held['n']:>3}): MRR@10={held['mrr_at_10']:.3f}"
    )
    for slice_name in SLICES:
        s = held["by_slice"][slice_name]
        lines.append(
            f"  └─ {slice_name:10s} (n={s['n']:>3}): MRR@10={s['mrr_at_10']:.3f}"
        )
    lines.append(f"adversarial (n={adv['n']:>3}): FPR={adv['adversarial_fpr']:.3f}")
    lines.append(f"smoke       (n={smoke['n']:>3}): pass_rate={smoke['pass_rate']:.3f}")
    lines.append("")
    lines.append("intrinsic (per slice):")
    for slice_name, s in receipt["intrinsic"]["by_slice"].items():
        ground = (
            f"{s['symbol_grounding_mean']:.3f}"
            if s["symbol_grounding_mean"] is not None
            else "N/A"
        )
        lines.append(
            f"  {slice_name:<11s} n={s['n']:>4d}  "
            f"banned={s['banned_preamble_rate']:.3f}  "
            f"namerest={s['name_restatement_mean']:.3f}  "
            f"ground={ground}  "
            f"chars_p50={s['length_chars_p50']:.0f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument(
        "--variant-id",
        required=True,
        help="Matches eval/variants/<id>.py; used in the run filename.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: eval/runs/<variant_id>-<timestamp>.json)",
    )
    parser.add_argument(
        "--print-path",
        action="store_true",
        help="Print only the receipt path to stdout; send summary to stderr. "
        "For use in shell pipelines that chain score → ratchet.",
    )
    args = parser.parse_args(argv)

    out = args.out
    if out is None:
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        out = RUNS_DIR / f"{args.variant_id}-{ts}.json"

    receipt = run_score(args.db, args.variant_id, out)
    if args.print_path:
        print(_format_summary(receipt), file=sys.stderr)
        print(f"\nreceipt: {out}", file=sys.stderr)
        print(str(out))
    else:
        print(_format_summary(receipt))
        print(f"\nreceipt: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
