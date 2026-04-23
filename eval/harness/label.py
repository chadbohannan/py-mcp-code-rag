"""Interactive CLI to grow the eval query set against a live index.

Usage:
    uv run python -m eval.harness.label \\
        --db aln.db \\
        --out eval/queries/dev.jsonl \\
        --query "how is a byte buffer parsed into a packet"

Shows the top-k search hits from the specified index and, for each,
prompts:

    [y] relevant        -> adds path to must_include
    [n] not relevant    -> skip (default)
    [x] adversarial     -> adds path to must_not_include
    [q] quit            -> abort without writing

After tagging, prompts for a stable id, comma-separated tags, and a
note. Appends one JSON line to --out. If the chosen id already exists
in --out, the record is rejected so prior labels are never silently
overwritten.

Query-set schema (one JSON object per line):

    {
      "id":               "q001",          # stable; bump prefix on schema change
      "query":            "...",           # natural-language query
      "must_include":     ["path", ...],   # any in top_k counts as a retrieval hit
      "must_not_include": ["path", ...],   # appearance in top_k is a FPR hit
      "top_k":            10,
      "tags":             ["..."],
      "notes":            "..."
    }

The harness treats must_include as a set: rank of the best-ranked member
gives MRR; presence of any member in top_k gives Recall@k. must_not_include
is scored as adversarial false-positive rate.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import sqlite_vec

from mcp_rag.embedder import FastEmbedder
from mcp_rag.queries import search


def _existing_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    ids: set[str] = set()
    for line in out_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids.add(json.loads(line)["id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return ids


def _prompt_label(idx: int, path: str, summary: str) -> str:
    print(f"\n[{idx}] {path}")
    print(f"    {summary}")
    while True:
        choice = input("    [y]es / [n]o / [x]adversarial / [q]uit: ").strip().lower()
        if choice in {"y", "n", "x", "q", ""}:
            return choice or "n"
        print("    unrecognized; please answer y/n/x/q")


def _open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", required=True, type=Path, help="Path to index SQLite DB")
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Target jsonl to append to (e.g. eval/queries/dev.jsonl)",
    )
    parser.add_argument("--query", required=True, help="Natural-language query to label")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)

    conn = _open_db(args.db)
    embedder = FastEmbedder()

    hits = search(conn, embedder, args.query, top_k=args.top_k)
    if not hits:
        print("No hits. Is the DB populated?", file=sys.stderr)
        return 2

    must_include: list[str] = []
    must_not_include: list[str] = []
    for i, hit in enumerate(hits, 1):
        choice = _prompt_label(i, hit["path"], hit["summary"] or "(no summary)")
        if choice == "q":
            print("Aborted; nothing written.")
            return 1
        if choice == "y":
            must_include.append(hit["path"])
        elif choice == "x":
            must_not_include.append(hit["path"])

    print()
    qid = input("id (e.g. q016): ").strip()
    if not qid:
        print("No id supplied; aborting.", file=sys.stderr)
        return 1
    if qid in _existing_ids(args.out):
        print(f"id {qid!r} already present in {args.out}; refusing to overwrite.", file=sys.stderr)
        return 1
    tags_raw = input("tags (comma-separated, optional): ").strip()
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
    notes = input("notes (optional): ").strip()

    record = {
        "id": qid,
        "query": args.query,
        "must_include": must_include,
        "must_not_include": must_not_include,
        "top_k": args.top_k,
        "tags": tags,
        "notes": notes,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\nAppended {qid} to {args.out}")
    print(f"  must_include:     {len(must_include)}")
    print(f"  must_not_include: {len(must_not_include)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
