"""Re-index a corpus under a candidate prompt variant.

Given a source index DB (for the list of repo roots) and a variant id,
this driver spawns ``code-rag index --reindex --prompt-variant <id>``
against a fresh output DB. The parser, embedder, summarizer backend, and
chunking logic are held fixed — only the prompt changes. That is what
lets the ratchet attribute held-out MRR deltas to the prompt.

Usage:
    uv run python -m eval.harness.rebuild \\
        --src-db index.db \\
        --variant v1_banned_preamble \\
        --out-db eval/dbs/v1_banned_preamble.db

The output DB path is the one passed to ``eval.harness.score --db``.
"""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path


def _read_repo_roots(src_db: Path) -> list[Path]:
    conn = sqlite3.connect(src_db)
    try:
        rows = conn.execute("SELECT root FROM repos ORDER BY name").fetchall()
    finally:
        conn.close()
    return [Path(r[0]) for r in rows]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src-db", required=True, type=Path)
    parser.add_argument("--variant", required=True, help="variant id under eval/variants/")
    parser.add_argument("--out-db", required=True, type=Path)
    parser.add_argument(
        "--summarizer",
        choices=["ollama", "anthropic"],
        default="ollama",
    )
    parser.add_argument("--ollama-model", default=None)
    parser.add_argument("--ollama-host", default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite out-db if it already exists.",
    )
    args = parser.parse_args(argv)

    if not args.src_db.exists():
        print(f"error: source DB not found: {args.src_db}", file=sys.stderr)
        return 1

    if args.out_db.exists():
        if not args.force:
            print(
                f"error: {args.out_db} exists; pass --force to overwrite.",
                file=sys.stderr,
            )
            return 1
        args.out_db.unlink()

    roots = _read_repo_roots(args.src_db)
    if not roots:
        print(f"error: no repos recorded in {args.src_db}", file=sys.stderr)
        return 1
    missing = [p for p in roots if not p.exists()]
    if missing:
        print(
            f"error: repo roots in src-db no longer exist: {missing}",
            file=sys.stderr,
        )
        return 1

    cmd = [
        "uv",
        "run",
        "code-rag",
        "index",
        "--reindex",
        "--db",
        str(args.out_db),
        "--prompt-variant",
        args.variant,
        "--summarizer",
        args.summarizer,
    ]
    if args.ollama_model:
        cmd += ["--ollama-model", args.ollama_model]
    if args.ollama_host:
        cmd += ["--ollama-host", args.ollama_host]
    cmd += [str(p) for p in roots]

    print(f"rebuild: variant={args.variant}  out={args.out_db}")
    print(f"roots: {[str(p) for p in roots]}")
    print(f"$ {' '.join(cmd)}")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
