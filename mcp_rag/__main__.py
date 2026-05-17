"""code-rag CLI entry point."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

from mcp_rag.embedder import DEFAULT_MODEL, EmbedderLoadError, FastEmbedder
from mcp_rag.indexer import DEFAULT_EXCLUDE_GLOBS, IndexAbortError, run_index
from mcp_rag.summarizer import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    AnthropicSummarizer,
    OllamaSummarizer,
)

_DEFAULT_DB = Path("index.db")


def _read_embed_meta(db_path: Path) -> tuple[str, int]:
    try:
        conn = sqlite3.connect(str(db_path))
        meta = dict(conn.execute("SELECT key, value FROM metadata").fetchall())
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
    ollama_host: str,
    reindex: bool,
    exclude_globs: tuple[str, ...] = DEFAULT_EXCLUDE_GLOBS,
) -> None:
    embedder = FastEmbedder(model_name=embed_model)
    if summarizer_type == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise IndexAbortError(
                "ANTHROPIC_API_KEY is not set. Export it before running code-rag index."
            )
        summarizer = AnthropicSummarizer()
    else:
        summarizer = OllamaSummarizer(model=ollama_model, host=ollama_host)
    run_index(
        roots=roots,
        db_path=db_path,
        embedder=embedder,
        summarizer=summarizer,
        reindex=reindex,
        exclude_globs=exclude_globs,
    )


def _resolve_exclude_globs(args: argparse.Namespace) -> tuple[str, ...]:
    if args.no_default_excludes:
        return ()
    if args.exclude:
        return tuple(args.exclude)
    return DEFAULT_EXCLUDE_GLOBS


def _add_exclude_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--exclude",
        action="append",
        metavar="GLOB",
        help="Exclude files matching GLOB (repeatable; replaces defaults). "
        f"Defaults: {', '.join(DEFAULT_EXCLUDE_GLOBS)}",
    )
    p.add_argument(
        "--no-default-excludes",
        action="store_true",
        dest="no_default_excludes",
        help="Disable the built-in exclude patterns for generated files",
    )


def _add_index_parser(sub) -> None:
    p = sub.add_parser("index", help="Index one or more directories")
    p.add_argument("paths", nargs="+", type=Path, metavar="PATH")
    p.add_argument("--reindex", action="store_true")
    p.add_argument("--embed-model", default=DEFAULT_MODEL, dest="embed_model")
    p.add_argument("--db", type=Path, default=_DEFAULT_DB)
    p.add_argument("--summarizer", choices=["anthropic", "ollama"], default="ollama")
    p.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL, dest="ollama_model")
    p.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST, dest="ollama_host")
    _add_exclude_args(p)


def _add_webui_parser(sub) -> None:
    p = sub.add_parser("webui", help="Run the REST API + web UI server")
    p.add_argument("--db", type=Path, default=_DEFAULT_DB)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--embed-model", default=None, dest="embed_model")
    p.add_argument("--summarizer", choices=["anthropic", "ollama"], default="ollama")
    p.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL, dest="ollama_model")
    p.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST, dest="ollama_host")
    _add_exclude_args(p)


def main() -> None:
    parser = argparse.ArgumentParser(prog="code-rag")
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_index_parser(sub)
    _add_webui_parser(sub)
    args = parser.parse_args()
    if args.cmd == "index":
        _run_index_cmd(args)
    elif args.cmd == "webui":
        _run_webui_cmd(args)


def _run_index_cmd(args: argparse.Namespace) -> None:
    for p in args.paths:
        if not p.exists():
            print(f"error: path does not exist: {p}", file=sys.stderr)
            sys.exit(1)
    try:
        _do_index(
            roots=[p.resolve() for p in args.paths],
            db_path=args.db,
            embed_model=args.embed_model,
            summarizer_type=args.summarizer,
            ollama_model=args.ollama_model,
            ollama_host=args.ollama_host,
            reindex=args.reindex,
            exclude_globs=_resolve_exclude_globs(args),
        )
    except (IndexAbortError, EmbedderLoadError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print(
            "\nInterrupted — partially indexed files will be retried on next run.",
            file=sys.stderr,
        )
        sys.exit(130)


def _run_webui_cmd(args: argparse.Namespace) -> None:
    import uvicorn

    from mcp_rag.webui import create_app

    if args.embed_model:
        embed_model = args.embed_model
    else:
        embed_model, _ = _read_embed_meta(args.db)

    try:
        embedder = FastEmbedder(model_name=embed_model)
    except EmbedderLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    sum_type = args.summarizer
    ollama_model = args.ollama_model
    ollama_host = args.ollama_host

    def make_summarizer():
        if sum_type == "anthropic":
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise IndexAbortError("ANTHROPIC_API_KEY is not set.")
            return AnthropicSummarizer()
        return OllamaSummarizer(model=ollama_model, host=ollama_host)

    app = create_app(
        db_path=args.db,
        embedder=embedder,
        summarizer_factory=make_summarizer,
        exclude_globs=_resolve_exclude_globs(args),
    )
    print(f"code-rag web UI: http://{args.host}:{args.port}", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", ws="wsproto")


if __name__ == "__main__":
    main()
