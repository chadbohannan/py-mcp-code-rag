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
