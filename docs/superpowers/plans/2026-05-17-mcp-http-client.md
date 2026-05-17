# HTTP-based MCP Client + CLI Completion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the MCP interface from direct-SQLite access to talking to the FastAPI HTTP service. Finish `code-rag-cli.py`. Delete `mcp_rag/server.py` and the `serve`/combined subcommands.

**Architecture:** Two standalone scripts at repo root (`code-rag-cli.py`, `code-rag-mcp.py`) speak HTTP to `webui.py`. The CLI is stdlib-only; the MCP server adds `fastmcp`. HTTP helpers are intentionally duplicated between the scripts so each stays self-contained.

**Tech Stack:** Python 3.12, `fastmcp` (existing dep), `urllib` (stdlib), `argparse` (stdlib), `pytest` + `unittest.mock` for tests.

**Spec:** `docs/superpowers/specs/2026-05-17-mcp-http-client-design.md`

---

## File Inventory

**Modified:**
- `mcp_rag/__main__.py` — drop `serve`/combined modes, switch to subparsers
- `code-rag-cli.py` — add 5 features
- `code-rag.sh` — usage text fix
- `tests/unit/test_cli.py` — delete `serve` + combined tests

**Created:**
- `code-rag-mcp.py`
- `tests/unit/test_cli_client.py`
- `tests/unit/test_mcp_client.py`

**Deleted:**
- `mcp_rag/server.py`
- `tests/integration/test_mcp_server.py`
- `tests/integration/test_server_search.py`

---

## Task 1: Cleanup — refactor `__main__.py`, prune tests, delete `server.py`

**Files:**
- Modify: `mcp_rag/__main__.py` (full rewrite)
- Modify: `tests/unit/test_cli.py` (delete sections)
- Delete: `mcp_rag/server.py`, `tests/integration/test_mcp_server.py`, `tests/integration/test_server_search.py`

- [ ] **Step 1: Replace `mcp_rag/__main__.py`**

Write the file to exactly:

```python
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
```

Removed vs. the old file: `from mcp_rag import server`, `from mcp_rag.server import mcp`, `_do_serve`, `_make_serve_parser`, `_run_serve_cmd`, `_make_combined_parser`, `_run_combined_cmd`, and the hand-rolled positional-arg sniff in `main()`. `argparse` subparsers handle dispatch.

- [ ] **Step 2: Prune `tests/unit/test_cli.py`**

Delete from `tests/unit/test_cli.py`:
- The `mock_server`, `mock_mcp`, and `mock_read_meta` fixtures.
- All `test_serve_*` functions (8 of them) under the `# serve subcommand` header.
- All `test_combined_*` functions (3 of them) under the `# combined mode` header.
- `test_combined_ollama_summarizer` at the end of the file.

Survivors: only the index-subcommand tests (argument wiring, error handling, summarizer selection).

Replace the file's top docstring with:

```python
"""Unit tests for the CLI entry point (mcp_rag.__main__).

All external I/O (run_index, FastEmbedder, AnthropicSummarizer) is
monkeypatched so no files, network, or servers are touched.
"""
```

- [ ] **Step 3: Delete `server.py` and its integration tests**

```bash
rm mcp_rag/server.py
rm tests/integration/test_mcp_server.py
rm tests/integration/test_server_search.py
```

- [ ] **Step 4: Sanity check — no stale references**

Run: `grep -rn "mcp_rag.server\|from mcp_rag import server" --include='*.py' .`
Expected: no matches.

- [ ] **Step 5: Test + commit**

Run: `uv run pytest tests/unit tests/integration -x -q`
Expected: PASS.

Run: `uv run code-rag --help`
Expected: usage shows `{index,webui}` only.

```bash
git add -A mcp_rag/__main__.py mcp_rag/server.py tests/unit/test_cli.py \
        tests/integration/test_mcp_server.py tests/integration/test_server_search.py
git commit -m "Remove direct-DB MCP server and combined subcommand"
```

---

## Task 2: Finish `code-rag-cli.py` — env var, `--json`, `staleness`, `ls`, `--wait`

**Files:**
- Modify: `code-rag-cli.py`
- Create: `tests/unit/test_cli_client.py`

- [ ] **Step 1: Write the test file**

Create `tests/unit/test_cli_client.py`:

```python
"""Unit tests for code-rag-cli.py (HTTP client CLI)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _load_cli():
    path = Path(__file__).parent.parent.parent / "code-rag-cli.py"
    spec = importlib.util.spec_from_file_location("code_rag_cli", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cli = _load_cli()


def _fake_response(payload):
    mock = MagicMock()
    mock.__enter__.return_value.read.return_value = json.dumps(payload).encode()
    mock.__exit__.return_value = False
    return mock


def _fake_responses(payloads):
    return [_fake_response(p) for p in payloads]


# --- --base-url / CODE_RAG_URL ----------------------------------------------

def test_explicit_base_url_overrides_env(monkeypatch):
    monkeypatch.setenv("CODE_RAG_URL", "http://env-host:1")
    args = cli._build_parser().parse_args(["--base-url", "http://flag-host:2", "repos"])
    assert args.base_url == "http://flag-host:2"


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("CODE_RAG_URL", "http://env-host:9999")
    args = cli._build_parser().parse_args(["repos"])
    assert args.base_url == "http://env-host:9999"


# --- --json flag ------------------------------------------------------------

def test_json_flag_outputs_raw_payload(capsys):
    payload = [{"path": "x", "summary": "s", "score": 0.9}]
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)):
        args = cli._build_parser().parse_args(["--json", "search", "auth"])
        args.func(args, args.base_url)
    assert json.loads(capsys.readouterr().out) == payload


def test_no_json_flag_uses_pretty_format(capsys):
    payload = [{"path": "repo/file.py:foo", "summary": "does foo", "score": 0.75}]
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)):
        args = cli._build_parser().parse_args(["search", "auth"])
        args.func(args, args.base_url)
    out = capsys.readouterr().out
    assert "0.7500" in out and "repo/file.py:foo" in out and "does foo" in out


# --- staleness --------------------------------------------------------------

def test_staleness_subcommand(capsys):
    payload = [{
        "repo": "alpha", "root": "/r", "last_indexed_at": "t1",
        "last_commit_at": "t2", "stale": True, "reason": "older than HEAD",
    }]
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        args = cli._build_parser().parse_args(["staleness"])
        args.func(args, args.base_url)
    assert "/api/repos/staleness" in m.call_args[0][0].full_url
    out = capsys.readouterr().out
    assert "alpha" in out and "older than HEAD" in out


# --- ls ---------------------------------------------------------------------

def test_ls_with_path_and_git_marker(capsys):
    payload = {
        "path": "/home/u/r", "parent": "/home/u", "is_git": True,
        "dirs": [{"name": "src", "path": "/home/u/r/src"}],
    }
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        args = cli._build_parser().parse_args(["ls", "/home/u/r"])
        args.func(args, args.base_url)
    assert "/api/ls" in m.call_args[0][0].full_url
    out = capsys.readouterr().out
    assert "*" in out and "src" in out


# --- index --wait -----------------------------------------------------------

def test_index_wait_polls_and_exits_ok(tmp_path, capsys):
    responses = _fake_responses([
        {"running": True,  "last_result": None, "last_finished_at": None},
        {"running": True,  "last_result": None, "last_finished_at": None},
        {"running": False, "last_result": "ok", "last_finished_at": "t"},
    ])
    with patch("urllib.request.urlopen", side_effect=responses), patch("time.sleep") as s:
        args = cli._build_parser().parse_args(["index", "--wait", str(tmp_path)])
        args.func(args, args.base_url)
    assert s.called
    assert "." in capsys.readouterr().err


def test_index_wait_nonzero_on_failure(tmp_path):
    responses = _fake_responses([
        {"running": True,  "last_result": None, "last_finished_at": None},
        {"running": False, "last_result": "boom", "last_finished_at": "t"},
    ])
    with patch("urllib.request.urlopen", side_effect=responses), patch("time.sleep"):
        args = cli._build_parser().parse_args(["index", "--wait", str(tmp_path)])
        with pytest.raises(SystemExit) as ei:
            args.func(args, args.base_url)
    assert ei.value.code == 1
```

- [ ] **Step 2: Apply the five changes to `code-rag-cli.py`**

Edit `code-rag-cli.py`:

**(a) Imports.** Add `import os` and `import time` to the imports block at the top.

**(b) Env var + `--json` in `_build_parser()`.** Replace the existing `--base-url` arg and insert `--json` immediately after it:

```python
    p.add_argument(
        "--base-url",
        default=os.environ.get("CODE_RAG_URL", "http://localhost:8081"),
        dest="base_url",
        help="Base URL of the code-rag webui server (env: CODE_RAG_URL)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit raw JSON instead of human-readable output",
    )
```

**(c) `_emit` helper.** Add this helper right after `_post`:

```python
def _emit(data: Any, args: argparse.Namespace) -> bool:
    """If --json is set, dump raw JSON and return True; else return False."""
    if args.json_output:
        print(json.dumps(data, indent=2))
        return True
    return False
```

**(d) Route every `_cmd_*` through `_emit`.** Each command captures the response once, returns early if `_emit` handled it, otherwise prints its formatted output. Apply this pattern to all 12 existing commands. Example for `_cmd_search`:

```python
def _cmd_search(args: argparse.Namespace, base: str) -> None:
    params: dict = {"q": args.query, "top_k": args.top_k}
    if args.globs:
        params["globs"] = args.globs
    data = _get(base, "/api/search", params)
    if _emit(data, args):
        return
    for r in data:
        print(f"{r['score']:.4f}\t{r['path']}")
        print(f"  {r['summary']}")
```

Apply the same shape to `_cmd_unit`, `_cmd_fetch`, `_cmd_units`, `_cmd_files`, `_cmd_repos`, `_cmd_status`, `_cmd_browse`, `_cmd_index_start`, `_cmd_index_status`, `_cmd_index_cancel`, `_cmd_clear_repo`: capture into `data`, early-return on `_emit`, then format.

**(e) Add `staleness` and `ls` commands.** Add these functions:

```python
def _cmd_staleness(args: argparse.Namespace, base: str) -> None:
    data = _get(base, "/api/repos/staleness")
    if _emit(data, args):
        return
    for r in data:
        print(
            f"{r['repo']}\t"
            f"indexed={r.get('last_indexed_at') or 'never'}\t"
            f"head={r.get('last_commit_at') or 'unknown'}\t"
            f"stale={r['stale']}\t{r['reason']}"
        )


def _cmd_ls(args: argparse.Namespace, base: str) -> None:
    params: dict = {"path": args.path} if args.path else {}
    data = _get(base, "/api/ls", params)
    if _emit(data, args):
        return
    marker = " *" if data.get("is_git") else ""
    print(f"{data['path']}{marker}")
    for entry in data.get("dirs", []):
        print(entry["name"])
```

Register both subparsers in `_build_parser()`:

```python
    s = sub.add_parser("staleness", help="Show per-repo index freshness vs HEAD")
    s.set_defaults(func=_cmd_staleness)

    s = sub.add_parser("ls", help="List filesystem directories (server-side)")
    s.add_argument("path", nargs="?", default="", help="Absolute path; defaults to server home")
    s.set_defaults(func=_cmd_ls)
```

**(f) `--wait` on the `index` subcommand.** In `_build_parser()`, add `--wait` to the existing `index` parser:

```python
    s.add_argument(
        "--wait",
        action="store_true",
        help="Block until the job completes; exit non-zero on failure",
    )
```

Replace `_cmd_index_start`:

```python
def _cmd_index_start(args: argparse.Namespace, base: str) -> None:
    status = _post(base, "/api/index", {"paths": args.paths, "reindex": args.reindex})
    if not args.wait:
        if _emit(status, args):
            return
        _print_job_status(status)
        return
    while status.get("running"):
        time.sleep(2)
        sys.stderr.write(".")
        sys.stderr.flush()
        status = _get(base, "/api/index/status")
    sys.stderr.write("\n")
    if not _emit(status, args):
        _print_job_status(status)
    if status.get("last_result") != "ok":
        sys.exit(1)
```

- [ ] **Step 3: Run tests + commit**

Run: `uv run pytest tests/unit/test_cli_client.py -v`
Expected: PASS.

```bash
git add code-rag-cli.py tests/unit/test_cli_client.py
git commit -m "Finish code-rag-cli: env var, --json, staleness, ls, --wait"
```

---

## Task 3: Fix `code-rag.sh` usage text

**Files:**
- Modify: `code-rag.sh`

- [ ] **Step 1: Edit and commit**

In `code-rag.sh`, change:

```
  --base-url URL               Base URL (default: http://localhost:8080)
```

to:

```
  --base-url URL               Base URL (default: http://localhost:8081)
```

```bash
git add code-rag.sh
git commit -m "Fix default URL in code-rag.sh usage text"
```

---

## Task 4: Create `code-rag-mcp.py` with all 9 tools

**Files:**
- Create: `code-rag-mcp.py`
- Create: `tests/unit/test_mcp_client.py`

- [ ] **Step 1: Write `code-rag-mcp.py`**

Create at the repo root:

```python
#!/usr/bin/env python3
"""MCP server that proxies code-rag tools to the HTTP service.

Stdio MCP by default; pass --http to run as a streamable-HTTP MCP server.
Talks to a running `code-rag webui` instance over REST.

Usage:
    python code-rag-mcp.py --base-url http://localhost:8081
    CODE_RAG_URL=http://host:8081 python code-rag-mcp.py
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

mcp = FastMCP("code-rag")

BASE_URL = "http://localhost:8081"  # set by main()


# --- HTTP helpers (duplicated from code-rag-cli.py, intentionally) ---------


def _url(path: str, params: dict | None = None) -> str:
    url = BASE_URL.rstrip("/") + path
    if params:
        flat: list[tuple[str, str]] = []
        for k, v in params.items():
            if isinstance(v, list):
                flat.extend((k, str(item)) for item in v)
            elif v is not None:
                flat.append((k, str(v)))
        if flat:
            url += "?" + urllib.parse.urlencode(flat)
    return url


def _request(url: str, data: bytes | None = None) -> Any:
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            detail = json.loads(body).get("detail", body)
        except Exception:
            detail = body
        raise ToolError(f"{e.code} {detail}")
    except urllib.error.URLError as e:
        raise ToolError(f"cannot reach code-rag at {BASE_URL}: {e.reason}")


def _get(path: str, params: dict | None = None) -> Any:
    return _request(_url(path, params))


def _post(path: str, body: dict | None = None, params: dict | None = None) -> Any:
    return _request(_url(path, params), json.dumps(body or {}).encode())


# --- Tools -----------------------------------------------------------------


@mcp.tool
async def search(
    query: str, top_k: int = 5, globs: list[str] | None = None
) -> list[dict]:
    """Search the indexed codebase using natural language.

    Every indexed unit (function, class, markdown section, etc.) has a
    pre-computed natural-language summary.  Queries are matched against these
    summaries via vector similarity, so **ask questions the way you'd ask a
    colleague** — e.g. "how does authentication work?" rather than keyword
    fragments.

    Results include the qualified path (``repo/file.py:Class:method``), the
    human-readable summary, and a relevance score in [0.0, 1.0] (higher is
    better).  top_k is capped at 20.  Use ``get_unit`` to retrieve full
    source content for specific paths.

    **Recommended workflow:**
    - Start with ``*.md`` to find authored documentation and module overviews.
    - Narrow into code (``*.py``, ``*.h``, ``*.cpp``) only after you have
      the big picture from docs.
    - Use semantic questions, not grep-style keywords — the index understands
      intent, not just tokens.

    Use globs to filter by qualified path with SQLite GLOB syntax.  Multiple
    globs are AND'd together — all must match.
    """
    params: dict = {"q": query, "top_k": top_k}
    if globs:
        params["globs"] = globs
    return _get("/api/search", params)


@mcp.tool
async def get_unit(paths: list[str]) -> list[dict]:
    """Retrieve the full source content of one or more indexed units by
    qualified path.

    Use this after ``search`` or ``list_units`` to read the actual code for
    specific results.  Paths must match exactly (use the ``path`` values
    returned by those tools).

    Returns the qualified path, source content, and summary for each matched
    path.  Paths that do not match any indexed unit are silently skipped.
    """
    return _post("/api/units/fetch", {"paths": paths})


@mcp.tool
async def list_units(
    globs: list[str] | None = None, limit: int = 100
) -> list[dict]:
    """List semantic units (functions, classes, methods, sections, etc.).

    Returns the qualified path (``repo/file.py:Class:method``) and summary for
    each unit, ordered alphabetically by path.  Use this to understand the
    structure of a file, module, or the entire codebase without fetching
    full source content.

    Use globs to filter by qualified path with SQLite GLOB syntax. Multiple
    globs are AND'd together.

    - ``["backend/*"]``          — all units in the backend repo
    - ``["*.js:*"]``             — all JS units
    - ``["*:Router:*"]``         — all Router members across languages
    - ``["backend/*", "*.py:*"]`` — Python units in backend only
    """
    params: dict = {"limit": limit}
    if globs:
        params["globs"] = globs
    return _get("/api/units", params)


@mcp.tool
async def list_files(globs: list[str] | None = None) -> list[dict]:
    """List files that have been indexed.

    Returns the repo name, relative file path, and last-indexed timestamp for
    every file in the index.  Call this early to understand what content is
    available before searching.

    Use globs to filter by file path with SQLite GLOB syntax. Multiple globs
    are AND'd together (e.g. ``["backend/*", "*.py"]``).
    """
    params: dict = {}
    if globs:
        params["globs"] = globs
    return _get("/api/files", params)


@mcp.tool
async def list_repos() -> list[dict]:
    """List all indexed repositories.

    Returns the repo name, absolute root path, and git description for each
    repository in the index.
    """
    return _get("/api/repos")


@mcp.tool
async def index_status() -> list[dict]:
    """Return per-repo index health: file count, unit count, last-indexed time.

    Call this before any other RAG tool to confirm the index is populated
    and fresh. If ``unit_count`` is 0 or ``last_indexed_at`` is stale,
    search results will be empty or incomplete.
    """
    data = _get("/api/status")
    return data.get("repos", [])


@mcp.tool
async def index_start(paths: list[str], reindex: bool = False) -> dict:
    """Enqueue paths for indexing on the shared service.

    Returns immediately with the job status (running/last_result/last_finished_at).
    Use ``index_job_status`` to poll for completion. Use ``index_cancel`` to abort.

    Indexing is long-running and calls the summarizer (LLM) and embedder per
    unit. Do not call without an explicit user request — it can take many
    minutes and may consume API quota.
    """
    return _post("/api/index", {"paths": paths, "reindex": reindex})


@mcp.tool
async def index_job_status() -> dict:
    """Poll the current state of the indexing job.

    Returns ``running`` (bool), ``last_result`` (status string from the last
    completed job, e.g. "ok" or an error message), and ``last_finished_at``.
    Distinct from ``index_status``, which reports per-repo counts.
    """
    return _get("/api/index/status")


@mcp.tool
async def index_cancel() -> dict:
    """Signal the running indexing job to cancel.

    Returns the post-cancel job status. Files already indexed are kept.
    No-op when no job is running.
    """
    return _post("/api/index/cancel")


# --- Entry point -----------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(prog="code-rag-mcp")
    p.add_argument(
        "--base-url",
        default=os.environ.get("CODE_RAG_URL", "http://localhost:8081"),
        help="Base URL of the code-rag webui server (env: CODE_RAG_URL)",
    )
    p.add_argument(
        "--http",
        action="store_true",
        help="Run as streamable-HTTP MCP server (default: stdio)",
    )
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    global BASE_URL
    BASE_URL = args.base_url
    if args.http:
        mcp.run(transport="streamable-http", host="127.0.0.1", port=args.port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
```

Make it executable: `chmod +x code-rag-mcp.py`

- [ ] **Step 2: Write `tests/unit/test_mcp_client.py`**

```python
"""Unit tests for code-rag-mcp.py."""
from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _load_mcp():
    path = Path(__file__).parent.parent.parent / "code-rag-mcp.py"
    spec = importlib.util.spec_from_file_location("code_rag_mcp", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mcp_mod():
    return _load_mcp()


def _fake_response(payload):
    mock = MagicMock()
    mock.__enter__.return_value.read.return_value = json.dumps(payload).encode()
    mock.__exit__.return_value = False
    return mock


# --- Read tools ------------------------------------------------------------

def test_search_builds_search_url(mcp_mod):
    payload = [{"path": "x", "summary": "s", "score": 0.9}]
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        result = asyncio.run(mcp_mod.search("test query", top_k=3))
    url = m.call_args[0][0].full_url
    assert "/api/search" in url and "top_k=3" in url
    assert result[0]["path"] == "x"


def test_get_unit_uses_post(mcp_mod):
    payload = [{"path": "x", "content": "code", "summary": "s"}]
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        result = asyncio.run(mcp_mod.get_unit(["x", "y"]))
    req = m.call_args[0][0]
    assert req.method == "POST"
    assert "/api/units/fetch" in req.full_url
    assert json.loads(req.data.decode()) == {"paths": ["x", "y"]}
    assert result == payload


def test_list_units_passes_globs_and_limit(mcp_mod):
    with patch("urllib.request.urlopen", return_value=_fake_response([])) as m:
        asyncio.run(mcp_mod.list_units(globs=["*.py"], limit=50))
    url = m.call_args[0][0].full_url
    assert "/api/units" in url and "globs=%2A.py" in url and "limit=50" in url


def test_list_files(mcp_mod):
    payload = [{"repo": "r", "path": "f.py", "indexed_at": "t"}]
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        result = asyncio.run(mcp_mod.list_files())
    assert "/api/files" in m.call_args[0][0].full_url
    assert result == payload


def test_list_repos(mcp_mod):
    payload = [{"name": "r", "root": "/r", "added_at": "t", "description": ""}]
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        result = asyncio.run(mcp_mod.list_repos())
    assert "/api/repos" in m.call_args[0][0].full_url
    assert result == payload


def test_index_status_returns_repos_list(mcp_mod):
    payload = {
        "repos": [{"repo": "r", "root": "/r", "file_count": 1,
                   "unit_count": 2, "last_indexed_at": "t"}],
        "total_units": 2, "embed_count": 2,
    }
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        result = asyncio.run(mcp_mod.index_status())
    assert "/api/status" in m.call_args[0][0].full_url
    assert result == payload["repos"]


# --- Index control tools ---------------------------------------------------

def test_index_start_posts_paths_and_reindex(mcp_mod):
    payload = {"running": True, "last_result": None, "last_finished_at": None}
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        asyncio.run(mcp_mod.index_start(["/a", "/b"], reindex=True))
    req = m.call_args[0][0]
    assert req.method == "POST" and "/api/index" in req.full_url
    assert json.loads(req.data.decode()) == {"paths": ["/a", "/b"], "reindex": True}


def test_index_job_status_is_get(mcp_mod):
    payload = {"running": False, "last_result": "ok", "last_finished_at": "t"}
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        result = asyncio.run(mcp_mod.index_job_status())
    req = m.call_args[0][0]
    assert req.get_method() == "GET" and "/api/index/status" in req.full_url
    assert result == payload


def test_index_cancel_is_post(mcp_mod):
    payload = {"running": False, "last_result": "cancelled", "last_finished_at": "t"}
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        result = asyncio.run(mcp_mod.index_cancel())
    req = m.call_args[0][0]
    assert req.method == "POST" and "/api/index/cancel" in req.full_url
    assert result == payload


# --- Error path ------------------------------------------------------------

def test_unreachable_server_raises_toolerror(mcp_mod):
    import urllib.error
    with patch("urllib.request.urlopen",
               side_effect=urllib.error.URLError("connection refused")):
        from fastmcp.exceptions import ToolError
        with pytest.raises(ToolError, match="cannot reach code-rag"):
            asyncio.run(mcp_mod.search("anything"))
```

- [ ] **Step 3: Run tests + commit**

Run: `uv run pytest tests/unit/test_mcp_client.py -v`
Expected: PASS.

**If a test fails with `TypeError: ... not callable` on `mcp_mod.search(...)`:** fastmcp's `@mcp.tool` returned a `Tool` wrapper instead of the original function. Fix by importing the inner callable: replace every `mcp_mod.<tool>` in the tests with `mcp_mod.<tool>.fn` (or whatever attribute the installed fastmcp uses — `print(dir(mcp_mod.search))` will show it). One-shot find/replace across the test file.

```bash
git add code-rag-mcp.py tests/unit/test_mcp_client.py
git commit -m "Add code-rag-mcp.py with 9 HTTP-proxied tools"
```

---

## Task 5: Final verification

- [ ] **Step 1: Lint + full test suite**

Run: `make lint && make test`
Expected: PASS. If `ruff format --check` fails, run `make format`, review the diff, commit with message `"Format pass"`.

- [ ] **Step 2: Smoke test the CLIs**

Run: `uv run code-rag --help`
Expected: usage shows `{index,webui}` only.

Run: `python code-rag-cli.py --help`
Expected: usage lists `search, unit, fetch, units, files, repos, status, browse, index, index-status, index-cancel, clear-repo, staleness, ls`.

Run: `python code-rag-cli.py --json repos 2>&1 | head -3`
Expected: error message about unreachable server (confirms the connection-refused path works when no webui is running). If a webui happens to be running on 8081, you'll get a JSON array — that's also fine.

- [ ] **Step 3: Confirm cleanup**

Run: `ls mcp_rag/server.py 2>&1`
Expected: `No such file or directory`.

Run: `grep -rn "mcp_rag.server\|from mcp_rag import server" --include='*.py' .`
Expected: no matches.

---

## Self-Review

- **Spec coverage:** Task 1 covers `__main__` refactor + deletions; Task 2 covers all five CLI completions; Task 3 covers the bash usage fix; Task 4 covers the new MCP with all 9 tools; Task 5 covers verification.
- **Naming consistency:** `index_status` (per-repo) and `index_job_status` (job state) are used in matching places across Task 4. `_emit` is introduced once in Task 2 and re-used by all commands.
- **Known variability:** fastmcp's `@mcp.tool` may or may not preserve the underlying callable. Task 4 Step 3 includes the one-shot fix if it's wrapped.
