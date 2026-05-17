# Design: HTTP-based MCP client and CLI completion

**Date:** 2026-05-17
**Branch:** `rework-mcp`
**Status:** Approved

## Goal

Crystallize the shared-service paradigm: the HTTP service (`webui.py`) becomes the only data path. Both the human-facing CLI and the LLM-facing MCP server become thin HTTP clients on top of it. The legacy direct-DB MCP server is removed.

Concretely:

1. Finish `code-rag-cli.py` (HTTP CLI client).
2. Add `code-rag-mcp.py` (HTTP MCP server) parallel to it.
3. Refactor `mcp_rag/__main__.py` to a two-subcommand dispatcher (`index`, `webui`).
4. Delete `mcp_rag/server.py` and its tests.

## Architecture

```
                ┌──────────────────────────┐
                │  webui.py (FastAPI)      │   single source of truth
                │  + queries.py + db.py    │   reads/writes SQLite
                └────────────┬─────────────┘
                             │ HTTP REST
              ┌──────────────┼──────────────┐
              ▼                             ▼
   code-rag-cli.py                  code-rag-mcp.py
   (stdlib only)                    (stdlib + fastmcp)
   human/script use                 LLM agent use
```

Both client scripts live at repo root, are standalone (copyable to another machine without installing the package), and carry their own copy of the ~50-line stdlib HTTP helper block. The duplication is intentional — it keeps each script self-contained.

## `code-rag-cli.py` completions

The current script already covers 12 of the 14 REST endpoints. Five changes complete it:

### Missing endpoints

- `code-rag-cli.py staleness` → `GET /api/repos/staleness`. Prints `repo<TAB>last_indexed<TAB>last_commit<TAB>stale<TAB>reason`.
- `code-rag-cli.py ls [PATH]` → `GET /api/ls`. Defaults to home directory; prints one dir per line; suffixes the listed root with ` *` if `is_git=true`.

### `--json` raw output flag

Top-level flag (before subcommand) that bypasses the pretty-printer and emits `json.dumps(data, indent=2)`. Applies to every subcommand uniformly.

```
code-rag-cli.py --json search "auth" | jq '.[] | select(.score > 0.7)'
```

### Blocking `index --wait`

After `POST /api/index` returns 202, poll `GET /api/index/status` every 2s until `running=false`. Print a `.` per poll to stderr for liveness. On completion, print `last_result`. Exit non-zero if `last_result != "ok"`.

### `CODE_RAG_URL` env var fallback

`--base-url` default becomes `os.environ.get("CODE_RAG_URL", "http://localhost:8081")`. Explicit `--base-url` overrides the env var.

### Bonus: usage-text fix in `code-rag.sh`

`code-rag.sh` line 234 says `default: http://localhost:8080` but the actual default is `8081`. Correct the usage string.

## `code-rag-mcp.py` (new)

Standalone script at repo root. Layout:

```python
#!/usr/bin/env python3
"""MCP server that proxies code-rag tools to the HTTP service."""

import argparse, json, os, sys
import urllib.error, urllib.parse, urllib.request
from typing import Any

from fastmcp import FastMCP

mcp = FastMCP("code-rag")
BASE_URL = "http://localhost:8081"   # set by main() from --base-url / env

# --- HTTP helpers (duplicated from CLI, intentionally) ---
def _url(path: str, params: dict | None = None) -> str: ...
def _get(path: str, params: dict | None = None) -> Any: ...
def _post(path: str, body: dict | None = None, params: dict | None = None) -> Any: ...
```

### Tool surface (9 tools)

| Tool | Endpoint | Notes |
|---|---|---|
| `search` | `GET /api/search` | Natural-language search; docstring preserved from current `server.py` |
| `get_unit` | `POST /api/units/fetch` | Batches via POST to avoid URL-length limits |
| `list_units` | `GET /api/units` | |
| `list_files` | `GET /api/files` | |
| `list_repos` | `GET /api/repos` | |
| `index_status` | `GET /api/status` | Per-repo file/unit counts (matches current direct-DB tool semantics) |
| `index_start` | `POST /api/index` | Returns 202 immediately; agent polls `index_job_status` for completion |
| `index_job_status` | `GET /api/index/status` | Distinct from `index_status` — this is job-level state |
| `index_cancel` | `POST /api/index/cancel` | |

**Naming decision:** keep `index_status` (per-repo counts) as the same name as today, and introduce `index_job_status` (job state) as a new name. This preserves the current tool surface for any agent prompts already wired to `index_status`.

### Error handling

- HTTP 4xx/5xx → raise `fastmcp.exceptions.ToolError` with the upstream `detail` message.
- Connection refused / unreachable → `ToolError(f"cannot reach code-rag at {BASE_URL}: {reason}")`.

### Entry point

```python
def main():
    p = argparse.ArgumentParser(prog="code-rag-mcp")
    p.add_argument("--base-url",
                   default=os.environ.get("CODE_RAG_URL", "http://localhost:8081"))
    p.add_argument("--http", action="store_true",
                   help="Run as streamable-HTTP MCP server (default: stdio)")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    global BASE_URL
    BASE_URL = args.base_url
    if args.http:
        mcp.run(transport="streamable-http", host="127.0.0.1", port=args.port)
    else:
        mcp.run()  # stdio
```

## `mcp_rag/__main__.py` refactor

Simplified to two subcommands; argparse subparsers replace the hand-rolled positional sniffing. Goes from ~290 lines to ~120.

**Removed:**

- `_do_serve()` (and the `from mcp_rag import server` import)
- `_make_serve_parser`, `_make_combined_parser`
- `_run_serve_cmd`, `_run_combined_cmd`
- The positional-arg sniffing loop in `main()` that decided between subcommand vs. combined mode

**Kept:**

- `_read_embed_meta()` — webui still uses it to pick an embedder when `--embed-model` isn't passed.

**Result skeleton:**

```python
def main():
    parser = argparse.ArgumentParser(prog="code-rag")
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_index_parser(sub)
    _add_webui_parser(sub)
    args = parser.parse_args()
    if args.cmd == "index":
        _run_index_cmd(args)
    elif args.cmd == "webui":
        _run_webui_cmd(args)
```

## File deletions

- `mcp_rag/server.py` — entire file.
- Any test in `tests/` that imports `mcp_rag.server` or asserts against its tools (identified via grep; expected to be a small set in `tests/integration/`).

## Dependencies / packaging

- `pyproject.toml`: no changes. `fastmcp` stays in deps (used by `code-rag-mcp.py`). `code-rag` script entry stays `mcp_rag.__main__:main`.
- `Makefile`: no changes.
- `code-rag.sh`: only the usage-text fix in Section 2.

## Testing

### New tests

**`tests/unit/test_cli_client.py`** — covers `code-rag-cli.py`:

- URL construction with repeated `globs` params.
- `--json` raw passthrough.
- `--base-url` precedence over `CODE_RAG_URL` env var.
- `index --wait`: mock two `running=true` then one `running=false` response; assert dots printed and exit code reflects `last_result`.
- 4xx/5xx response → stderr + exit 1.
- Connection refused → exit 1.

**`tests/unit/test_mcp_client.py`** — covers `code-rag-mcp.py`:

- For each of the 9 tools, assert it hits the right endpoint with the right params/body.
- `BASE_URL` plumbing from `--base-url` and `CODE_RAG_URL`.
- HTTP error → `ToolError` with upstream detail.
- Tool functions are `async`; invoked via `asyncio.run` in tests.

### Mocking approach

Both test files patch `urllib.request.urlopen` (and `urllib.error.URLError` paths) to feed canned responses. No real network, no real DB.

### Script importability

Scripts live at repo root, not under `mcp_rag/`. Tests import them via `importlib.util.spec_from_file_location` so the scripts remain runnable as `python code-rag-cli.py …` while still being unit-testable.

### Removed tests

Anything in `tests/integration/` that boots `mcp_rag.server` or asserts against the direct-DB MCP tools is deleted. The HTTP-MCP path is covered by the unit tests above. No end-to-end test that boots the webui in a subprocess is added — out of scope (slow; the unit mocks plus existing webui coverage are sufficient).

## Out of scope

- End-to-end test that launches `webui.py` in a subprocess and exercises `code-rag-mcp.py` against it.
- Auth/auth between MCP client and HTTP server (single-user local deployment for now).
- Shared HTTP helper module — intentionally rejected; duplication preserves standalone-ness.
- Renaming `index_status` → `repo_status` for clarity — would be a breaking change for agents using the current tool name.

## Migration notes

Users who launched the MCP via `code-rag serve --db ./index.db` must switch to:

```
# in one terminal
uv run code-rag webui --db ./index.db --port 8081

# in another (or in MCP client config)
python code-rag-mcp.py --base-url http://localhost:8081
```

The `code-rag serve` subcommand is removed without deprecation warning (small user base, branch is pre-release).
