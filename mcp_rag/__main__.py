"""mcp-rag CLI entry point."""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

from mcp_rag import server
from mcp_rag.embedder import DEFAULT_MODEL, FastEmbedder
from mcp_rag.indexer import IndexAbortError, run_index
from mcp_rag.server import mcp
from mcp_rag.summarizer import AgentSummarizer, AnthropicSummarizer, OllamaSummarizer

_DEFAULT_DB = Path("index.db")
_SUBCOMMANDS = {"index", "serve"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_embed_meta(db_path: Path) -> tuple[str, int]:
    """Read embed_model and embed_dim from an existing index DB.

    Falls back to the default model/dim when the DB is absent or has no meta,
    so that `serve` works gracefully before a first `index` run.
    """
    try:
        conn = sqlite3.connect(str(db_path))
        meta = dict(conn.execute("SELECT key, value FROM mcp_rag_meta").fetchall())
        conn.close()
        return meta["embed_model"], int(meta["embed_dim"])
    except Exception:
        return DEFAULT_MODEL, 768


def _do_index(
    roots: list[Path],
    db_path: Path,
    embed_model: str,
    summarizer_type: str,
    ollama_model: str,
    agent_timeout: int,
    reindex: bool,
) -> None:
    embedder = FastEmbedder(model_name=embed_model)
    if summarizer_type == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise IndexAbortError(
                "ANTHROPIC_API_KEY is not set. Export it before running mcp-rag index."
            )
        summarizer = AnthropicSummarizer()
    elif summarizer_type == "ollama":
        summarizer = OllamaSummarizer(model=ollama_model)
    else:
        summarizer = AgentSummarizer(command=summarizer_type, timeout=agent_timeout)
    run_index(
        roots=roots,
        db_path=db_path,
        embedder=embedder,
        summarizer=summarizer,
        reindex=reindex,
    )


def _do_serve(db_path: Path, http: bool = False, port: int = 8000) -> None:
    embed_model, _ = _read_embed_meta(db_path)
    embedder = FastEmbedder(model_name=embed_model)
    server.configure(db_path, embedder)
    if http:
        mcp.run(transport="streamable-http", host="127.0.0.1", port=port)
    else:
        mcp.run()


# ---------------------------------------------------------------------------
# Argument parsers
# ---------------------------------------------------------------------------

def _make_index_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mcp-rag index")
    p.add_argument("paths", nargs="+", type=Path, metavar="PATH")
    p.add_argument("--reindex", action="store_true")
    p.add_argument("--embed-model", default=DEFAULT_MODEL, dest="embed_model")
    p.add_argument("--db", type=Path, default=_DEFAULT_DB)
    p.add_argument("--summarizer", choices=["anthropic", "ollama", "claude", "pi"], default="anthropic")
    p.add_argument("--ollama-model", default="llama3.2", dest="ollama_model")
    p.add_argument("--agent-timeout", type=int, default=120, dest="agent_timeout",
                   metavar="SECONDS", help="timeout per agent subprocess call (default: 120)")
    return p


def _make_serve_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mcp-rag serve")
    p.add_argument("--http", action="store_true")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--db", type=Path, default=_DEFAULT_DB)
    return p


def _make_combined_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mcp-rag")
    p.add_argument("paths", nargs="*", type=Path, metavar="PATH")
    p.add_argument("--db", type=Path, default=_DEFAULT_DB)
    p.add_argument("--embed-model", default=DEFAULT_MODEL, dest="embed_model")
    p.add_argument("--summarizer", choices=["anthropic", "ollama", "claude", "pi"], default="anthropic")
    p.add_argument("--ollama-model", default="llama3.2", dest="ollama_model")
    p.add_argument("--agent-timeout", type=int, default=120, dest="agent_timeout",
                   metavar="SECONDS", help="timeout per agent subprocess call (default: 120)")
    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    argv = sys.argv[1:]

    # Determine mode by inspecting the first non-flag argument
    first_pos = next((a for a in argv if not a.startswith("-")), None)
    # Also skip values that follow a --flag (e.g. --db /path)
    positionals = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg.startswith("--"):
            skip_next = "=" not in arg  # --flag VALUE (two tokens)
        elif not arg.startswith("-"):
            positionals.append(arg)
    first_pos = positionals[0] if positionals else None

    if first_pos == "index":
        args = _make_index_parser().parse_args(argv[argv.index("index") + 1 :])
        _run_index_cmd(args)
    elif first_pos == "serve":
        args = _make_serve_parser().parse_args(argv[argv.index("serve") + 1 :])
        _run_serve_cmd(args)
    else:
        args = _make_combined_parser().parse_args(argv)
        _run_combined_cmd(args)


def _run_index_cmd(args: argparse.Namespace) -> None:
    try:
        _do_index(
            roots=[p.resolve() for p in args.paths],
            db_path=args.db,
            embed_model=args.embed_model,
            summarizer_type=args.summarizer,
            ollama_model=args.ollama_model,
            agent_timeout=args.agent_timeout,
            reindex=args.reindex,
        )
    except IndexAbortError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


def _run_serve_cmd(args: argparse.Namespace) -> None:
    _do_serve(db_path=args.db, http=args.http, port=args.port)


def _run_combined_cmd(args: argparse.Namespace) -> None:
    try:
        if args.paths and not args.db.exists():
            _do_index(
                roots=[p.resolve() for p in args.paths],
                db_path=args.db,
                embed_model=args.embed_model,
                summarizer_type=args.summarizer,
                ollama_model=args.ollama_model,
                agent_timeout=args.agent_timeout,
                reindex=False,
            )
        _do_serve(db_path=args.db)
    except IndexAbortError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
