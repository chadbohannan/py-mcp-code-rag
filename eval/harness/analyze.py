"""Extract failure cases from a run receipt for mutator consumption.

Given a receipt, emit the queries that missed in a given slice along
with: the must_include answers (path + summary + source), and the
top_paths the retriever actually preferred (with summaries). This
exposes *why* the retriever picked the wrong thing — the key diagnostic
a prompt mutator needs.

Query text is not stored in the receipt (only ids), so this re-reads
the query files in eval/queries/. Unit source/summary is fetched from
the DB recorded in the receipt.

Usage:
    uv run python -m eval.harness.analyze \\
        --receipt eval/runs/v0_baseline-<ts>.json \\
        --slice leaf_code --split heldout --top-n 10

Output format defaults to markdown (human + LLM friendly). `--format
jsonl` emits one JSON object per failure for programmatic consumption.

This is the inner-loop diagnostic; the outer-loop mutator feeds this
output into a prompt-variant generator.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import sqlite_vec

from eval.harness.slices import SLICES

EVAL_DIR = Path(__file__).resolve().parent.parent
QUERIES_DIR = EVAL_DIR / "queries"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]


def _open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def _fetch_units(
    conn: sqlite3.Connection, paths: list[str]
) -> dict[str, dict[str, Any]]:
    if not paths:
        return {}
    placeholders = ",".join("?" for _ in paths)
    rows = conn.execute(
        f"SELECT path, unit_type, summary, content FROM units "
        f"WHERE path IN ({placeholders})",
        paths,
    ).fetchall()
    return {
        p: {"unit_type": t, "summary": s, "content": c} for p, t, s, c in rows
    }


def _query_text_by_id(split: str) -> dict[str, str]:
    queries = _load_jsonl(QUERIES_DIR / f"{split}.jsonl")
    return {q["id"]: q["query"] for q in queries}


def extract_misses(
    receipt: dict[str, Any],
    split: str,
    slice_name: str,
    top_n: int,
    miss_threshold: int = 1,
) -> list[dict[str, Any]]:
    """Return queries that contributed to the slice but ranked worse than
    miss_threshold (rank 0 = not in top_k; rank > miss_threshold = demoted).

    Sorted worst-first (rank 0 first, then by descending rank).
    """
    if slice_name not in SLICES:
        raise ValueError(f"unknown slice {slice_name!r}; known: {list(SLICES)}")
    per_query = receipt["splits"][split]["per_query"]
    misses = []
    for r in per_query:
        sr = r["slice_ranks"].get(slice_name)
        if sr is None:
            continue
        rank = sr["rank"]
        if rank == 0 or rank > miss_threshold:
            misses.append({**r, "_slice_rank": rank})
    # rank 0 (not found) is worst; otherwise larger rank is worse.
    misses.sort(key=lambda r: (r["_slice_rank"] != 0, r["_slice_rank"]))
    return misses[:top_n]


def enrich_miss(
    miss: dict[str, Any],
    query_text: str,
    slice_name: str,
    conn: sqlite3.Connection,
) -> dict[str, Any]:
    mi_types = miss["must_include_types"]
    in_slice_mi = [p for p, t in mi_types.items() if t in SLICES[slice_name]]
    top_paths = miss["top_paths"]
    wanted_units = _fetch_units(conn, in_slice_mi)
    top_units = _fetch_units(conn, top_paths)
    return {
        "id": miss["id"],
        "query": query_text,
        "slice": slice_name,
        "slice_rank": miss["_slice_rank"],
        "wanted": [
            {
                "path": p,
                "unit_type": wanted_units.get(p, {}).get("unit_type"),
                "summary": wanted_units.get(p, {}).get("summary"),
                "content": wanted_units.get(p, {}).get("content"),
            }
            for p in in_slice_mi
        ],
        "top_paths": [
            {
                "rank": i + 1,
                "path": p,
                "unit_type": top_units.get(p, {}).get("unit_type"),
                "summary": top_units.get(p, {}).get("summary"),
            }
            for i, p in enumerate(top_paths)
        ],
    }


def _format_md(misses: list[dict[str, Any]], receipt: dict[str, Any]) -> str:
    lines = [
        f"# Failure analysis — {receipt['variant_id']}",
        f"receipt: {receipt.get('db_path', '?')}",
        f"timestamp: {receipt.get('timestamp', '?')}",
        "",
    ]
    for i, m in enumerate(misses, 1):
        rank_s = "not in top_k" if m["slice_rank"] == 0 else f"rank {m['slice_rank']}"
        lines.append(f"## {i}. `{m['id']}` — slice={m['slice']} ({rank_s})")
        lines.append(f"**query:** {m['query']}")
        lines.append("")
        lines.append("### wanted (must_include in slice)")
        for w in m["wanted"]:
            lines.append(f"- `{w['path']}` ({w['unit_type']})")
            lines.append(f"  - summary: {w['summary']!r}")
            if w.get("content"):
                src = w["content"]
                if len(src) > 400:
                    src = src[:400] + "…"
                lines.append(f"  - source:\n    ```\n    {src}\n    ```")
        lines.append("")
        lines.append("### retriever's top_paths (what it preferred)")
        for t in m["top_paths"][:5]:
            lines.append(
                f"{t['rank']}. `{t['path']}` ({t['unit_type']}): "
                f"{t['summary']!r}"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument(
        "--split", default="heldout", choices=["dev", "heldout"]
    )
    parser.add_argument(
        "--slice",
        dest="slice_name",
        default=None,
        help="Slice to analyze. Omit to emit failures for every slice.",
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--miss-threshold",
        type=int,
        default=1,
        help="Rank worse than this counts as a miss (default 1: anything "
        "not at rank 1). Use 3 to focus on 'not in top 3', etc.",
    )
    parser.add_argument("--format", default="md", choices=["md", "jsonl"])
    args = parser.parse_args(argv)

    receipt = json.loads(args.receipt.read_text())
    db_path = Path(receipt["db_path"])
    if not db_path.exists():
        raise SystemExit(
            f"db_path from receipt does not exist: {db_path}. "
            f"Re-run the variant or edit the receipt's db_path."
        )

    query_texts = _query_text_by_id(args.split)
    conn = _open_db(db_path)

    slices_to_emit = (
        [args.slice_name] if args.slice_name else list(SLICES.keys())
    )
    all_enriched: list[dict[str, Any]] = []
    for slice_name in slices_to_emit:
        raw = extract_misses(
            receipt, args.split, slice_name, args.top_n, args.miss_threshold
        )
        for miss in raw:
            qtext = query_texts.get(miss["id"], "<query text missing>")
            all_enriched.append(enrich_miss(miss, qtext, slice_name, conn))

    if args.format == "jsonl":
        for m in all_enriched:
            print(json.dumps(m))
    else:
        print(_format_md(all_enriched, receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
