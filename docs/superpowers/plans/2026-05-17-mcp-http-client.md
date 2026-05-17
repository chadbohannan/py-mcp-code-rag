# HTTP-based MCP Client + CLI Completion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the MCP interface from direct-SQLite access to talking to the FastAPI HTTP service. Finish `code-rag-cli.py`. Delete `mcp_rag/server.py` and the `serve`/combined subcommands.

**Architecture:** Two standalone scripts at repo root (`code-rag-cli.py`, `code-rag-mcp.py`) both speak HTTP to `webui.py`. The webui keeps owning SQLite via `queries.py`. The Python CLI uses stdlib only; the MCP server uses stdlib + `fastmcp`. HTTP helper code is intentionally duplicated between the two scripts so each remains self-contained.

**Tech Stack:** Python 3.12, `fastmcp` (existing dep), `urllib` (stdlib), `argparse` (stdlib), `pytest` + `unittest.mock` for tests.

**Spec:** `docs/superpowers/specs/2026-05-17-mcp-http-client-design.md`

---

## File Inventory

**Modified:**
- `mcp_rag/__main__.py` — drop `serve`/combined modes, switch to subparsers
- `code-rag-cli.py` — add 5 features
- `code-rag.sh` — usage text fix
- `tests/unit/test_cli.py` — delete `serve` + combined-mode tests

**Created:**
- `code-rag-mcp.py` — new MCP server (root)
- `tests/unit/test_cli_client.py` — tests for `code-rag-cli.py`
- `tests/unit/test_mcp_client.py` — tests for `code-rag-mcp.py`

**Deleted:**
- `mcp_rag/server.py`
- `tests/integration/test_mcp_server.py`
- `tests/integration/test_server_search.py`

---

## Task 1: Refactor `mcp_rag/__main__.py` — remove `serve` and combined mode

**Files:**
- Modify: `mcp_rag/__main__.py` (full rewrite)

- [ ] **Step 1: Replace `mcp_rag/__main__.py` with the new content**

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

Key changes from the old file: no `from mcp_rag import server`, no `from mcp_rag.server import mcp`, no `_do_serve`, no `_make_serve_parser` / `_run_serve_cmd`, no `_make_combined_parser` / `_run_combined_cmd`, no hand-rolled positional sniff in `main()`. `argparse`'s subparsers do the dispatch and provide `--help` for each subcommand.

- [ ] **Step 2: Verify `serve` is gone but `index` and `webui` still parse**

Run: `uv run code-rag --help`
Expected: usage shows `{index,webui}` only, no `serve`.

Run: `uv run code-rag index --help`
Expected: standard index help text.

Run: `uv run code-rag serve --help`
Expected: error — `invalid choice: 'serve'`.

- [ ] **Step 3: Commit**

```bash
git add mcp_rag/__main__.py
git commit -m "Refactor __main__ to drop serve and combined modes"
```

---

## Task 2: Prune `tests/unit/test_cli.py` — remove `serve` + combined tests

**Files:**
- Modify: `tests/unit/test_cli.py`

- [ ] **Step 1: Delete the `mock_server` fixture and the `serve`/combined tests**

Delete:
- The `mock_server` fixture (lines ~64-68 in the original — the `@pytest.fixture` decorated `def mock_server(monkeypatch):` and its body).
- The entire `# serve subcommand` section header and its 8 test functions: `test_serve_configures_server`, `test_serve_calls_mcp_run`, `test_serve_uses_stdio_by_default`, `test_serve_http_flag_uses_http_transport`, `test_serve_http_binds_localhost`, `test_serve_default_port_is_8000`, `test_serve_custom_port`, `test_serve_custom_db`.
- The entire `# combined mode` section header and its 3 test functions: `test_combined_indexes_when_db_absent`, `test_combined_skips_index_when_db_present`, `test_combined_serve_only_when_no_paths`.
- The `test_combined_ollama_summarizer` function at the bottom of the file.

Surviving tests in `test_cli.py`: only the `index subcommand — argument wiring`, `index subcommand — error handling`, and `index subcommand — summarizer selection` test functions (everything that exercises `index` and uses only `mock_embedder`, `mock_summarizer`, `mock_ollama_summarizer`, `mock_run_index`).

Also delete the `mock_mcp` and `mock_read_meta` fixtures if no surviving test references them. (Quick check: grep `mock_mcp\|mock_read_meta` in the file after deleting; if no hits, drop the fixtures.)

- [ ] **Step 2: Update the module docstring**

Replace the file's top docstring with:

```python
"""Unit tests for the CLI entry point (mcp_rag.__main__).

All external I/O (run_index, FastEmbedder, AnthropicSummarizer) is
monkeypatched so no files, network, or servers are touched.
"""
```

- [ ] **Step 3: Run the unit tests for this file**

Run: `uv run pytest tests/unit/test_cli.py -v`
Expected: PASS. All `serve_*` and `combined_*` tests are gone; only `index_*` tests remain and pass.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_cli.py
git commit -m "Drop serve/combined test cases from test_cli"
```

---

## Task 3: Delete `mcp_rag/server.py` and its integration tests

**Files:**
- Delete: `mcp_rag/server.py`
- Delete: `tests/integration/test_mcp_server.py`
- Delete: `tests/integration/test_server_search.py`

- [ ] **Step 1: Confirm nothing else imports `mcp_rag.server`**

Run: `grep -rn "mcp_rag.server\|from mcp_rag import server" --include='*.py' .`
Expected: no matches (after Task 1 and Task 2). If any match shows up, stop and fix that file before deleting.

- [ ] **Step 2: Delete the three files**

```bash
rm mcp_rag/server.py
rm tests/integration/test_mcp_server.py
rm tests/integration/test_server_search.py
```

- [ ] **Step 3: Run the test suite**

Run: `uv run pytest tests/unit tests/integration -x -q`
Expected: PASS. No test references the deleted `mcp_rag.server` module.

- [ ] **Step 4: Commit**

```bash
git add -A mcp_rag/server.py tests/integration/test_mcp_server.py tests/integration/test_server_search.py
git commit -m "Delete direct-DB MCP server"
```

---

## Task 4: `code-rag-cli.py` — `CODE_RAG_URL` env var fallback

**Files:**
- Modify: `code-rag-cli.py` — `_build_parser()` near line 173
- Test: `tests/unit/test_cli_client.py` (new file)

- [ ] **Step 1: Create the new test file with the env-var test**

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


def test_default_base_url_uses_env_var(monkeypatch):
    monkeypatch.setenv("CODE_RAG_URL", "http://example.com:9999")
    args = cli._build_parser().parse_args(["repos"])
    assert args.base_url == "http://example.com:9999"


def test_explicit_flag_overrides_env_var(monkeypatch):
    monkeypatch.setenv("CODE_RAG_URL", "http://example.com:9999")
    args = cli._build_parser().parse_args(
        ["--base-url", "http://other:1111", "repos"]
    )
    assert args.base_url == "http://other:1111"


def test_default_base_url_without_env(monkeypatch):
    monkeypatch.delenv("CODE_RAG_URL", raising=False)
    args = cli._build_parser().parse_args(["repos"])
    assert args.base_url == "http://localhost:8081"
```

- [ ] **Step 2: Run the tests — expect FAIL**

Run: `uv run pytest tests/unit/test_cli_client.py -v`
Expected: FAIL. `test_default_base_url_uses_env_var` will show `assert 'http://localhost:8081' == 'http://example.com:9999'`.

- [ ] **Step 3: Apply the env-var fallback in `code-rag-cli.py`**

In `code-rag-cli.py`, locate this block in `_build_parser()`:

```python
    p.add_argument(
        "--base-url",
        default="http://localhost:8081",
        dest="base_url",
        help="Base URL of the code-rag webui server",
    )
```

Replace with:

```python
    p.add_argument(
        "--base-url",
        default=os.environ.get("CODE_RAG_URL", "http://localhost:8081"),
        dest="base_url",
        help="Base URL of the code-rag webui server (env: CODE_RAG_URL)",
    )
```

Add `import os` to the imports at the top of the file (it's not currently imported).

- [ ] **Step 4: Run the tests — expect PASS**

Run: `uv run pytest tests/unit/test_cli_client.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code-rag-cli.py tests/unit/test_cli_client.py
git commit -m "Add CODE_RAG_URL env-var fallback to code-rag-cli"
```

---

## Task 5: `code-rag-cli.py` — `--json` raw output flag

**Files:**
- Modify: `code-rag-cli.py` — `_build_parser()`, every `_cmd_*` function, `main()`
- Modify: `tests/unit/test_cli_client.py` — add tests

- [ ] **Step 1: Add tests for `--json` output**

Append to `tests/unit/test_cli_client.py`:

```python
def _fake_response(payload):
    """Build a context-manager fake for urllib.request.urlopen()."""
    mock = MagicMock()
    mock.__enter__.return_value.read.return_value = json.dumps(payload).encode()
    mock.__exit__.return_value = False
    return mock


def test_json_flag_outputs_raw_json(capsys):
    payload = [{"path": "x", "summary": "s", "score": 0.9}]
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)):
        args = cli._build_parser().parse_args(["--json", "search", "auth"])
        args.func(args, args.base_url)
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed == payload


def test_no_json_flag_uses_pretty_format(capsys):
    payload = [{"path": "repo/file.py:foo", "summary": "does foo", "score": 0.75}]
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)):
        args = cli._build_parser().parse_args(["search", "auth"])
        args.func(args, args.base_url)
    out = capsys.readouterr().out
    assert "0.7500" in out
    assert "repo/file.py:foo" in out
    assert "does foo" in out
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `uv run pytest tests/unit/test_cli_client.py::test_json_flag_outputs_raw_json -v`
Expected: FAIL. The `--json` flag doesn't exist yet, so argparse raises and the test surfaces an exit.

- [ ] **Step 3: Add `--json` flag to the parser**

In `_build_parser()` in `code-rag-cli.py`, immediately after the `--base-url` argument, add:

```python
    p.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit raw JSON instead of human-readable output",
    )
```

- [ ] **Step 4: Add a JSON-aware print helper and route every `_cmd_*` through it**

Near the top of `code-rag-cli.py`, after the existing `_request` helper, add:

```python
def _emit(data: Any, args: argparse.Namespace) -> None:
    """If --json was passed, dump raw JSON; otherwise let the caller print
    its own formatted output. Returns True when the caller should skip
    its formatting block."""
    if args.json_output:
        print(json.dumps(data, indent=2))
        return True
    return False
```

Then in every `_cmd_*` function, capture the response and skip the pretty path when `--json` is set. Pattern:

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

Apply the same `data = ... ; if _emit(data, args): return ; <pretty>` shape to every other `_cmd_*` function (search, unit, fetch, units, files, repos, status, browse, index_start, index_status, index_cancel, clear_repo). For `_cmd_fetch`, capture the POST result once before iterating.

- [ ] **Step 5: Run the new tests — expect PASS**

Run: `uv run pytest tests/unit/test_cli_client.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add code-rag-cli.py tests/unit/test_cli_client.py
git commit -m "Add --json raw output flag to code-rag-cli"
```

---

## Task 6: `code-rag-cli.py` — `staleness` subcommand

**Files:**
- Modify: `code-rag-cli.py` — add `_cmd_staleness` and register the subparser
- Modify: `tests/unit/test_cli_client.py` — add test

- [ ] **Step 1: Add test**

Append to `tests/unit/test_cli_client.py`:

```python
def test_staleness_subcommand(capsys):
    payload = [
        {
            "repo": "alpha",
            "root": "/tmp/alpha",
            "last_indexed_at": "2026-05-15T10:00:00",
            "last_commit_at": "2026-05-16T11:00:00",
            "stale": True,
            "reason": "index older than HEAD",
        }
    ]
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        args = cli._build_parser().parse_args(["staleness"])
        args.func(args, args.base_url)
    req = m.call_args[0][0]
    assert "/api/repos/staleness" in req.full_url
    out = capsys.readouterr().out
    assert "alpha" in out
    assert "True" in out or "stale" in out
    assert "index older than HEAD" in out
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `uv run pytest tests/unit/test_cli_client.py::test_staleness_subcommand -v`
Expected: FAIL with `argparse` error: `invalid choice: 'staleness'`.

- [ ] **Step 3: Add the command implementation and register the subparser**

In `code-rag-cli.py`, add this function alongside the other `_cmd_*` functions:

```python
def _cmd_staleness(_args: argparse.Namespace, base: str) -> None:
    data = _get(base, "/api/repos/staleness")
    if _emit(data, _args):
        return
    for r in data:
        print(
            f"{r['repo']}\t"
            f"indexed={r.get('last_indexed_at') or 'never'}\t"
            f"head={r.get('last_commit_at') or 'unknown'}\t"
            f"stale={r['stale']}\t"
            f"{r['reason']}"
        )
```

Register it in `_build_parser()` (alongside the other `sub.add_parser` calls):

```python
    s = sub.add_parser("staleness", help="Show per-repo index freshness vs HEAD")
    s.set_defaults(func=_cmd_staleness)
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `uv run pytest tests/unit/test_cli_client.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code-rag-cli.py tests/unit/test_cli_client.py
git commit -m "Add staleness subcommand to code-rag-cli"
```

---

## Task 7: `code-rag-cli.py` — `ls` subcommand

**Files:**
- Modify: `code-rag-cli.py` — add `_cmd_ls` and register the subparser
- Modify: `tests/unit/test_cli_client.py` — add test

- [ ] **Step 1: Add test**

Append to `tests/unit/test_cli_client.py`:

```python
def test_ls_subcommand_without_path(capsys):
    payload = {
        "path": "/home/u",
        "parent": "/home",
        "is_git": False,
        "dirs": [
            {"name": "code", "path": "/home/u/code"},
            {"name": "docs", "path": "/home/u/docs"},
        ],
    }
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        args = cli._build_parser().parse_args(["ls"])
        args.func(args, args.base_url)
    req = m.call_args[0][0]
    assert "/api/ls" in req.full_url
    out = capsys.readouterr().out
    assert "/home/u" in out
    assert "code" in out
    assert "docs" in out


def test_ls_subcommand_with_path_and_git_flag(capsys):
    payload = {
        "path": "/home/u/repo",
        "parent": "/home/u",
        "is_git": True,
        "dirs": [{"name": "src", "path": "/home/u/repo/src"}],
    }
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        args = cli._build_parser().parse_args(["ls", "/home/u/repo"])
        args.func(args, args.base_url)
    req = m.call_args[0][0]
    assert "path=%2Fhome%2Fu%2Frepo" in req.full_url
    out = capsys.readouterr().out
    assert "*" in out  # git marker present
    assert "src" in out
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `uv run pytest tests/unit/test_cli_client.py -k "test_ls" -v`
Expected: FAIL — `invalid choice: 'ls'`.

- [ ] **Step 3: Add the command implementation and register the subparser**

Add to `code-rag-cli.py`:

```python
def _cmd_ls(args: argparse.Namespace, base: str) -> None:
    params: dict = {}
    if args.path:
        params["path"] = args.path
    data = _get(base, "/api/ls", params)
    if _emit(data, args):
        return
    marker = " *" if data.get("is_git") else ""
    print(f"{data['path']}{marker}")
    for entry in data.get("dirs", []):
        print(entry["name"])
```

Register the subparser:

```python
    s = sub.add_parser("ls", help="List filesystem directories (server-side)")
    s.add_argument("path", nargs="?", default="", help="Absolute path; defaults to server home")
    s.set_defaults(func=_cmd_ls)
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `uv run pytest tests/unit/test_cli_client.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code-rag-cli.py tests/unit/test_cli_client.py
git commit -m "Add ls subcommand to code-rag-cli"
```

---

## Task 8: `code-rag-cli.py` — blocking `index --wait`

**Files:**
- Modify: `code-rag-cli.py` — `_cmd_index_start` and the `index` subparser
- Modify: `tests/unit/test_cli_client.py` — add tests

- [ ] **Step 1: Add tests for `--wait`**

Append to `tests/unit/test_cli_client.py`:

```python
def _fake_responses(payloads):
    """Sequential context-manager fakes for urlopen()."""
    mocks = []
    for p in payloads:
        m = MagicMock()
        m.__enter__.return_value.read.return_value = json.dumps(p).encode()
        m.__exit__.return_value = False
        mocks.append(m)
    return mocks


def test_index_wait_polls_until_done(monkeypatch, capsys, tmp_path):
    responses = _fake_responses([
        {"running": True,  "last_result": None, "last_finished_at": None},  # POST /api/index
        {"running": True,  "last_result": None, "last_finished_at": None},  # poll 1
        {"running": False, "last_result": "ok", "last_finished_at": "2026-05-17T12:00"},  # poll 2
    ])
    with patch("urllib.request.urlopen", side_effect=responses), \
         patch("time.sleep") as sleep_mock:
        args = cli._build_parser().parse_args(["index", "--wait", str(tmp_path)])
        args.func(args, args.base_url)
    assert sleep_mock.called
    err = capsys.readouterr().err
    assert "." in err  # liveness dots on stderr


def test_index_wait_nonzero_exit_on_failure(tmp_path):
    responses = _fake_responses([
        {"running": True,  "last_result": None, "last_finished_at": None},
        {"running": False, "last_result": "boom", "last_finished_at": "now"},
    ])
    with patch("urllib.request.urlopen", side_effect=responses), \
         patch("time.sleep"):
        args = cli._build_parser().parse_args(["index", "--wait", str(tmp_path)])
        with pytest.raises(SystemExit) as ei:
            args.func(args, args.base_url)
        assert ei.value.code == 1


def test_index_without_wait_returns_immediately(tmp_path):
    """Default behavior: POST and print job status, no polling."""
    responses = _fake_responses([
        {"running": True, "last_result": None, "last_finished_at": None},
    ])
    with patch("urllib.request.urlopen", side_effect=responses) as m, \
         patch("time.sleep") as sleep_mock:
        args = cli._build_parser().parse_args(["index", str(tmp_path)])
        args.func(args, args.base_url)
    assert m.call_count == 1
    sleep_mock.assert_not_called()
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `uv run pytest tests/unit/test_cli_client.py -k "index_wait or index_without_wait" -v`
Expected: FAIL — `--wait` not recognized.

- [ ] **Step 3: Add `import time` and `--wait` flag**

In `code-rag-cli.py`:

1. Add `import time` to the top imports.

2. In `_build_parser()`, modify the `index` subparser block:

```python
    s = sub.add_parser("index", help="Start an indexing job")
    s.add_argument("paths", nargs="+", help="Directory paths to index")
    s.add_argument("--reindex", action="store_true")
    s.add_argument(
        "--wait",
        action="store_true",
        help="Block until the job completes; exit non-zero on failure",
    )
    s.set_defaults(func=_cmd_index_start)
```

3. Replace `_cmd_index_start` with:

```python
def _cmd_index_start(args: argparse.Namespace, base: str) -> None:
    status = _post(base, "/api/index", {"paths": args.paths, "reindex": args.reindex})
    if not args.wait:
        if _emit(status, args):
            return
        _print_job_status(status)
        return
    # Block: poll /api/index/status until running is false.
    while status.get("running"):
        time.sleep(2)
        sys.stderr.write(".")
        sys.stderr.flush()
        status = _get(base, "/api/index/status")
    sys.stderr.write("\n")
    if _emit(status, args):
        if status.get("last_result") != "ok":
            sys.exit(1)
        return
    _print_job_status(status)
    if status.get("last_result") != "ok":
        sys.exit(1)
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `uv run pytest tests/unit/test_cli_client.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code-rag-cli.py tests/unit/test_cli_client.py
git commit -m "Add --wait flag to code-rag-cli index"
```

---

## Task 9: Fix `code-rag.sh` usage text

**Files:**
- Modify: `code-rag.sh` — usage block

- [ ] **Step 1: Correct the default URL in the usage text**

In `code-rag.sh`, change:

```
  --base-url URL               Base URL (default: http://localhost:8080)
```

to:

```
  --base-url URL               Base URL (default: http://localhost:8081)
```

- [ ] **Step 2: Verify the help still renders**

Run: `bash code-rag.sh --help | grep base-url`
Expected: shows `default: http://localhost:8081`.

- [ ] **Step 3: Commit**

```bash
git add code-rag.sh
git commit -m "Fix default URL in code-rag.sh usage text"
```

---

## Task 10: Create `code-rag-mcp.py` skeleton + `search` tool

**Files:**
- Create: `code-rag-mcp.py` (root)
- Create: `tests/unit/test_mcp_client.py`

- [ ] **Step 1: Write the first failing test**

Create `tests/unit/test_mcp_client.py`:

```python
"""Unit tests for code-rag-mcp.py (HTTP-based MCP server)."""
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


def _call(tool):
    """Resolve a callable from either a plain async function or a fastmcp Tool.

    fastmcp's @mcp.tool decorator may return either the original function or
    a Tool wrapper depending on version. Tests use this helper so the same
    test code works with either.
    """
    if callable(tool):
        return tool
    # fastmcp Tool wrapper: expose .fn or .run
    return getattr(tool, "fn", None) or tool.run


def test_search_tool_hits_search_endpoint(mcp_mod):
    payload = [{"path": "x", "summary": "s", "score": 0.9}]
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        result = asyncio.run(_call(mcp_mod.search)("test query", top_k=3))
    req = m.call_args[0][0]
    assert "/api/search" in req.full_url
    assert "q=test" in req.full_url
    assert "top_k=3" in req.full_url
    assert result[0]["path"] == "x"
```

- [ ] **Step 2: Run the test — expect FAIL (module not found)**

Run: `uv run pytest tests/unit/test_mcp_client.py -v`
Expected: FAIL — `code-rag-mcp.py` does not exist.

- [ ] **Step 3: Write the script**

Create `code-rag-mcp.py` at the repo root:

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
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

mcp = FastMCP("code-rag")

# Set by main() from --base-url or CODE_RAG_URL.
BASE_URL = "http://localhost:8081"


# ---------------------------------------------------------------------------
# HTTP helpers (intentionally duplicated from code-rag-cli.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


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
    (e.g. ``["backend/*", "*.py:*"]`` → only Python units in the backend repo).
    """
    params: dict = {"q": query, "top_k": top_k}
    if globs:
        params["globs"] = globs
    return _get("/api/search", params)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


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

Make it executable:

```bash
chmod +x code-rag-mcp.py
```

- [ ] **Step 4: Run the test — expect PASS**

Run: `uv run pytest tests/unit/test_mcp_client.py -v`
Expected: PASS.

If the test fails with `TypeError: object Tool is not callable`, the `_call(...)` helper is already in place — it will fall back to `tool.fn` or `tool.run`. If neither attribute resolves either, inspect the failing tool object (`print(dir(mcp_mod.search))`) and update `_call` to use the correct attribute for the installed fastmcp version.

- [ ] **Step 5: Commit**

```bash
git add code-rag-mcp.py tests/unit/test_mcp_client.py
git commit -m "Add code-rag-mcp.py skeleton with search tool"
```

---

## Task 11: MCP read tools — `get_unit`, `list_units`, `list_files`, `list_repos`, `index_status`

**Files:**
- Modify: `code-rag-mcp.py` — add 5 tools
- Modify: `tests/unit/test_mcp_client.py` — add tests

- [ ] **Step 1: Add tests for the 5 tools**

Append to `tests/unit/test_mcp_client.py`:

```python
def test_get_unit_uses_post(mcp_mod):
    payload = [{"path": "x", "content": "code", "summary": "s"}]
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        result = asyncio.run(_call(mcp_mod.get_unit)(["x", "y"]))
    req = m.call_args[0][0]
    assert req.method == "POST"
    assert "/api/units/fetch" in req.full_url
    body = json.loads(req.data.decode())
    assert body == {"paths": ["x", "y"]}
    assert result == payload


def test_list_units_with_globs(mcp_mod):
    payload = [{"path": "a", "summary": "s"}]
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        asyncio.run(_call(mcp_mod.list_units)(globs=["*.py"], limit=50))
    req = m.call_args[0][0]
    assert "/api/units" in req.full_url
    assert "globs=%2A.py" in req.full_url
    assert "limit=50" in req.full_url


def test_list_files(mcp_mod):
    payload = [{"repo": "r", "path": "f.py", "indexed_at": "now"}]
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        result = asyncio.run(_call(mcp_mod.list_files)())
    req = m.call_args[0][0]
    assert "/api/files" in req.full_url
    assert result == payload


def test_list_repos(mcp_mod):
    payload = [{"name": "r", "root": "/r", "added_at": "now", "description": ""}]
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        result = asyncio.run(_call(mcp_mod.list_repos)())
    req = m.call_args[0][0]
    assert "/api/repos" in req.full_url
    assert result == payload


def test_index_status_returns_repos_list(mcp_mod):
    payload = {
        "repos": [{"repo": "r", "root": "/r", "file_count": 1, "unit_count": 2,
                   "last_indexed_at": "now"}],
        "total_units": 2,
        "embed_count": 2,
    }
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        result = asyncio.run(_call(mcp_mod.index_status)())
    req = m.call_args[0][0]
    assert "/api/status" in req.full_url
    # MCP tool should expose the per-repo list (matching old tool semantics).
    assert result == payload["repos"]
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `uv run pytest tests/unit/test_mcp_client.py -v`
Expected: 5 failures — these tools don't exist yet.

- [ ] **Step 3: Add the 5 tool definitions to `code-rag-mcp.py`**

Add after the `search` tool:

```python
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
    """List semantic units (functions, classes, methods, sections, etc.) in the
    index.

    Returns the qualified path (``repo/file.py:Class:method``) and summary for
    each unit, ordered alphabetically by path.  Use this to understand the
    structure of a file, module, or the entire codebase without fetching
    full source content.

    Use globs to filter by qualified path with SQLite GLOB syntax.  Multiple
    globs are AND'd together.  The qualified path starts with the repo name
    then the relative file path:

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

    Use globs to filter by file path with SQLite GLOB syntax.  Multiple globs
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
    """Return the current state of the index.

    Call this before any other RAG tool to confirm the index is populated
    and fresh.  If ``unit_count`` is 0 or ``last_indexed_at`` is stale,
    search results will be empty or incomplete.

    Reports per-repo file count, semantic unit count, and the timestamp of
    the most recent indexing run.
    """
    data = _get("/api/status")
    return data.get("repos", [])
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `uv run pytest tests/unit/test_mcp_client.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code-rag-mcp.py tests/unit/test_mcp_client.py
git commit -m "Add read tools to code-rag-mcp: get_unit, list_units, list_files, list_repos, index_status"
```

---

## Task 12: MCP index-control tools — `index_start`, `index_job_status`, `index_cancel`

**Files:**
- Modify: `code-rag-mcp.py`
- Modify: `tests/unit/test_mcp_client.py`

- [ ] **Step 1: Add tests**

Append to `tests/unit/test_mcp_client.py`:

```python
def test_index_start_posts_paths(mcp_mod):
    payload = {"running": True, "last_result": None, "last_finished_at": None}
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        result = asyncio.run(_call(mcp_mod.index_start)(["/repo/a", "/repo/b"]))
    req = m.call_args[0][0]
    assert req.method == "POST"
    assert "/api/index" in req.full_url
    assert "/api/index/" not in req.full_url
    body = json.loads(req.data.decode())
    assert body == {"paths": ["/repo/a", "/repo/b"], "reindex": False}
    assert result["running"] is True


def test_index_start_with_reindex(mcp_mod):
    payload = {"running": True, "last_result": None, "last_finished_at": None}
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        asyncio.run(_call(mcp_mod.index_start)(["/repo"], reindex=True))
    body = json.loads(m.call_args[0][0].data.decode())
    assert body["reindex"] is True


def test_index_job_status(mcp_mod):
    payload = {"running": False, "last_result": "ok", "last_finished_at": "now"}
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        result = asyncio.run(_call(mcp_mod.index_job_status)())
    req = m.call_args[0][0]
    assert req.get_method() == "GET"
    assert "/api/index/status" in req.full_url
    assert result == payload


def test_index_cancel(mcp_mod):
    payload = {"running": False, "last_result": "cancelled", "last_finished_at": "now"}
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        result = asyncio.run(_call(mcp_mod.index_cancel)())
    req = m.call_args[0][0]
    assert req.method == "POST"
    assert "/api/index/cancel" in req.full_url
    assert result == payload
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `uv run pytest tests/unit/test_mcp_client.py -k "index_start or index_job_status or index_cancel" -v`
Expected: 4 failures — these tools don't exist yet.

- [ ] **Step 3: Add the tool definitions**

Add to `code-rag-mcp.py` after the read tools:

```python
@mcp.tool
async def index_start(paths: list[str], reindex: bool = False) -> dict:
    """Enqueue paths for indexing on the shared service.

    Returns immediately with the job status (running/last_result/last_finished_at).
    Use ``index_job_status`` to poll for completion. Use ``index_cancel`` to abort.

    Indexing is a long-running operation that calls the summarizer (LLM) and
    embedder once per unchanged-md5 unit. Do not call this without an
    explicit user request — it can take many minutes and may consume API
    quota.
    """
    return _post("/api/index", {"paths": paths, "reindex": reindex})


@mcp.tool
async def index_job_status() -> dict:
    """Poll the current state of the indexing job.

    Returns ``running`` (bool), ``last_result`` (status string from the last
    completed job, e.g. "ok" or an error message), and ``last_finished_at``
    (ISO timestamp). Distinct from ``index_status``, which reports per-repo
    file/unit counts rather than job state.
    """
    return _get("/api/index/status")


@mcp.tool
async def index_cancel() -> dict:
    """Signal the running indexing job to cancel.

    Returns the post-cancel job status. The job stops at the next safe point;
    files already indexed are kept in the DB. Calling this when no job is
    running is a no-op.
    """
    return _post("/api/index/cancel")
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `uv run pytest tests/unit/test_mcp_client.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code-rag-mcp.py tests/unit/test_mcp_client.py
git commit -m "Add index control tools to code-rag-mcp"
```

---

## Task 13: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Lint and format**

Run: `make lint`
Expected: no errors. If `ruff format --check` complains, run `make format` and review the diff before committing.

- [ ] **Step 2: Full test suite**

Run: `make test`
Expected: PASS. All unit and integration tests pass (including the new `test_cli_client.py` and `test_mcp_client.py`).

- [ ] **Step 3: Smoke test the CLI scripts**

Run: `uv run code-rag --help`
Expected: usage shows `{index,webui}` subcommands.

Run: `python code-rag-cli.py --help`
Expected: usage shows the full subcommand list including `staleness` and `ls`.

Run: `python code-rag-cli.py --json repos 2>/dev/null || echo "expected: cannot reach server"`
Expected: prints the error path because no webui is running — confirms the connection-refused path works.

- [ ] **Step 4: Confirm `mcp_rag/server.py` is gone**

Run: `ls mcp_rag/server.py 2>&1`
Expected: `No such file or directory`.

Run: `grep -rn "mcp_rag.server\|from mcp_rag import server" --include='*.py' .`
Expected: no matches.

- [ ] **Step 5: Final commit (if any lint/format changes)**

Only needed if `make format` produced edits in Step 1.

```bash
git add -u
git commit -m "Format pass after MCP/CLI refactor"
```

- [ ] **Step 6: Update `skill.md` if the API has drifted**

Run: `make skill`
Expected: if `skill.md` shows a diff, the regen happens. If clean, no change. (No new endpoints were added in this work, so `skill.md` should be unchanged — confirm via `git diff skill.md`.)

If `skill.md` did change, commit it:

```bash
git add skill.md SKILL.md 2>/dev/null
git commit -m "Regenerate skill.md"
```

---

## Self-Review Notes

**Spec coverage:** every spec section maps to a task.
- Architecture diagram → tasks 1, 3, 10
- CLI completions (5 features) → tasks 4–8
- `code-rag-mcp.py` (9 tools) → tasks 10–12
- `__main__.py` refactor → task 1
- Deletions → tasks 2, 3
- Tests → tasks 2, 4–8, 10–12
- `code-rag.sh` usage fix → task 9
- Final verification of lint/test/smoke → task 13

**Naming consistency check:**
- `index_status` (per-repo counts) and `index_job_status` (job state) are distinct and consistently named in tasks 11 and 12.
- `_emit` helper introduced in task 5 is referenced (without re-defining) in tasks 6, 7, 8.
- `_fake_response`/`_fake_responses`/`_call` helpers are defined in tasks 5 and 10 and reused in later tasks.

**Potential snag flagged in task 10:** fastmcp's `@mcp.tool` may or may not preserve the underlying callable depending on version. The `_call(...)` helper in `test_mcp_client.py` handles both shapes; the plan calls this out in Task 10 Step 4 with a fallback action.
