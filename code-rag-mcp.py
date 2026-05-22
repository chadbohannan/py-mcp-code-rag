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
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
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
async def list_units(globs: list[str] | None = None, limit: int = 100) -> list[dict]:
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
async def staleness() -> list[dict]:
    """Compare each indexed repo's last_indexed_at against its git HEAD commit time.

    Returns one row per repo with ``stale`` (bool) and ``reason`` (human-readable
    string explaining why). Call this when you suspect the index may be out of
    date — e.g. after the user mentions a recent commit, or before starting a
    fresh investigation in a repo. If any repo is stale, recommend running
    ``index_start`` on its root path before searching.
    """
    return _get("/api/repos/staleness")


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


# --- Interactive setup -----------------------------------------------------


def _prompt(question: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{question}{suffix}: ").strip()
    except EOFError:
        return default
    return answer or default


def _probe_webui(base_url: str) -> tuple[bool, str]:
    """Hit GET /api/status. Returns (reachable, summary_or_error)."""
    try:
        with urllib.request.urlopen(
            base_url.rstrip("/") + "/api/status", timeout=3
        ) as resp:
            data = json.loads(resp.read())
        repos = data.get("repos", [])
        return True, f"{len(repos)} repo(s) indexed"
    except Exception as exc:
        return False, str(exc)


def _build_command(script_dir: Path, script_path: Path, base_url: str) -> list[str]:
    """Construct the argv used by MCP hosts to launch this server.

    Prefer `uv run --directory <dir> python <script>` when uv is on PATH and
    the project has a pyproject.toml — that matches the Makefile and keeps
    dependencies resolved correctly. Otherwise fall back to the current
    interpreter.
    """
    if shutil.which("uv") and (script_dir / "pyproject.toml").exists():
        return [
            "uv",
            "run",
            "--directory",
            str(script_dir),
            "python",
            str(script_path),
            "--base-url",
            base_url,
        ]
    return [sys.executable, str(script_path), "--base-url", base_url]


def _install_claude_code(command: list[str]) -> None:
    if not shutil.which("claude"):
        print(
            "  ! `claude` CLI not found on PATH. Run this manually once it's installed:"
        )
        print("    " + " ".join(["claude", "mcp", "add", "--transport", "stdio",
                                  "-s", "user", "code-rag", "--", *command]))
        return
    argv = ["claude", "mcp", "add", "--transport", "stdio", "-s", "user",
            "code-rag", "--", *command]
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode == 0:
        print("  ✓ registered with Claude Code (user scope)")
    else:
        print(f"  ! claude mcp add failed: {result.stderr.strip() or result.stdout.strip()}")


def _install_pi_agent(command: list[str]) -> None:
    mcp_json = Path.home() / ".pi" / "agent" / "mcp.json"
    mcp_json.parent.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(mcp_json.read_text()) if mcp_json.exists() else {}
    cfg.setdefault("mcpServers", {})["code-rag"] = {
        "command": command[0],
        "args": command[1:],
    }
    mcp_json.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"  ✓ wrote {mcp_json}")


def _run_setup(default_base_url: str) -> int:
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent

    print("code-rag MCP — interactive setup")
    print("=" * 32)
    print(
        "This script is normally launched over stdio by an MCP host\n"
        "(Claude Code, pi-agent, etc.). Since you're running it from a\n"
        "terminal, let's wire it up.\n"
    )

    base_url = _prompt("Base URL of the code-rag webui", default_base_url)
    ok, summary = _probe_webui(base_url)
    if ok:
        print(f"  ✓ reachable — {summary}\n")
    else:
        print(f"  ! could not reach {base_url}: {summary}")
        if _prompt("Continue anyway? (y/N)", "n").lower() != "y":
            print("aborted.")
            return 1
        print()

    print("Which hosts should I configure?")
    print("  1) Claude Code")
    print("  2) pi-agent")
    print("  3) Both")
    print("  4) Just print the launch command (don't write any config)")
    choice = _prompt("Choose", "3")

    command = _build_command(script_dir, script_path, base_url)

    if choice == "4":
        print("\nLaunch command for manual configuration:")
        print("  " + " ".join(command))
        return 0

    if choice in ("1", "3"):
        print("\nConfiguring Claude Code...")
        _install_claude_code(command)
    if choice in ("2", "3"):
        print("\nConfiguring pi-agent...")
        _install_pi_agent(command)

    print("\nDone. Restart your MCP host to pick up the new server.")
    return 0


# --- Entry point -----------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        prog="code-rag-mcp",
        description="MCP server that proxies code-rag tools to the webui. "
        "Run with --setup (or from a terminal) for an interactive integration helper.",
    )
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
    p.add_argument(
        "--setup",
        action="store_true",
        help="Run the interactive setup wizard for Claude Code / pi-agent.",
    )
    args = p.parse_args()

    # Auto-enter setup if invoked from a terminal without an MCP host on the
    # pipe. --http is a server mode, so skip setup there.
    auto_setup = sys.stdin.isatty() and sys.stdout.isatty() and not args.http
    if args.setup or auto_setup:
        sys.exit(_run_setup(args.base_url))

    global BASE_URL
    BASE_URL = args.base_url
    if args.http:
        mcp.run(transport="streamable-http", host="127.0.0.1", port=args.port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
