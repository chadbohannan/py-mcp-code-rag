# Cleanup: Post-MCP-Migration Doc & Deploy Sync — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the Makefile, scripts, docs, and bash CLI into alignment with the HTTP-based MCP architecture introduced on `rework-mcp`. Add the `staleness` tool to the new MCP server and tighten lint coverage.

**Architecture:** No new architecture — this plan synchronises supporting artefacts with the code that already shipped. The reference implementation is `code-rag-cli.py`, `code-rag-mcp.py`, and `mcp_rag/__main__.py` as they currently stand on the branch.

**Context:** Built on top of `43ced44`. The previous plan (`2026-05-17-mcp-http-client.md`) replaced the direct-DB MCP server with `code-rag-mcp.py` (HTTP client) but did not touch the Makefile, scripts, or documentation. A holistic review surfaced that `make serve`, `make add-claude-mcp`, `make add-pi-mcp`, and the README's Quickstart all still reference the removed `code-rag serve` subcommand — users following the README cannot start the MCP.

**Branch:** `rework-mcp`

---

## File Inventory

**Modified:**
- `Makefile` — fix `serve`/`add-claude-mcp` targets, add lint scope
- `scripts/add_pi_mcp.py` — change registered MCP command
- `README.md` — Quickstart, CLI reference, Serving section, agent-integration table, MCP-tools section
- `CLAUDE.md` — remove `serve` mode references, update architecture
- `overview.md` — drop `serve mode` / `combined mode` from runtime design
- `design_spec.md` — update CLI section, remove Combined-mode Behaviour section
- `code-rag.sh` — bring to parity with `code-rag-cli.py` (env var, `--json`, `staleness`, `ls`, `--wait`)
- `code-rag-cli.py` — set executable bit (file mode only)
- `code-rag-mcp.py` — add `staleness` tool
- `tests/unit/test_mcp_client.py` — add `staleness` test

---

## Task 1: Fix runtime breakage — Makefile + `add_pi_mcp.py` + executable bit

**Files:**
- Modify: `Makefile` (lines 1, 39-41, 47-49, 131 of the make-targets table — see below)
- Modify: `scripts/add_pi_mcp.py` (line 21)
- Modify: `code-rag-cli.py` (file mode 0664 → 0755)

- [ ] **Step 1: Replace the broken `Makefile` targets**

Edit `Makefile`. Replace the `serve` target (lines 39-41):

```makefile
# Start the MCP stdio server (HTTP client to webui). Usage: make mcp BASE_URL=http://host:8081
mcp:
	~/.local/bin/uv run python code-rag-mcp.py --base-url $(or $(BASE_URL),http://localhost:8081)
```

Replace the `add-claude-mcp` target (lines 47-49):

```makefile
# Register this server with Claude Code (run once after cloning). Usage: make add-claude-mcp BASE_URL=http://host:8081
add-claude-mcp:
	claude mcp add --transport stdio -s user code-rag -- uv run --directory $(DIR) python $(DIR)/code-rag-mcp.py --base-url $(or $(BASE_URL),http://localhost:8081)
```

Update the `.PHONY` line (line 1) — replace `serve` with `mcp`:

```makefile
.PHONY: install test test-unit test-integration lint format index reindex mcp webui skill clean add-claude-mcp remove-claude-mcp add-pi-mcp remove-pi-mcp
```

- [ ] **Step 2: Fix `scripts/add_pi_mcp.py`**

Replace line 14-22 of `scripts/add_pi_mcp.py`:

```python
    directory, db_path = sys.argv[1], sys.argv[2]
    base_url = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8081"
    mcp_json = Path.home() / ".pi" / "agent" / "mcp.json"

    mcp_json.parent.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(mcp_json.read_text()) if mcp_json.exists() else {}
    cfg.setdefault("mcpServers", {})["code-rag"] = {
        "command": "uv",
        "args": ["run", "--directory", directory, "python",
                 f"{directory}/code-rag-mcp.py", "--base-url", base_url],
    }
```

Update the `usage:` line (line 11):

```python
        print("usage: add_pi_mcp.py <directory> <db-path> [base-url]", file=sys.stderr)
```

Then update the `add-pi-mcp` target in `Makefile` (line 58) to pass through `BASE_URL`:

```makefile
add-pi-mcp:
	pi install npm:pi-mcp-adapter
	python3 scripts/add_pi_mcp.py $(DIR) $(abspath $(or $(DB),index.db)) $(or $(BASE_URL),http://localhost:8081)
```

The `db-path` argument is now unused by the MCP config but kept for backward compat — `add_pi_mcp.py` still accepts and ignores it.

- [ ] **Step 3: Set executable bit on `code-rag-cli.py`**

```bash
chmod +x /home/kali/workspace/py-mcp-code-rag/code-rag-cli.py
```

- [ ] **Step 4: Verify**

Run: `make mcp --dry-run 2>&1 | head -5`
Expected: shows the new `uv run python code-rag-mcp.py ...` command (no execution because no webui is running).

Run: `make add-claude-mcp --dry-run 2>&1 | head -5`
Expected: shows the new `claude mcp add ... code-rag-mcp.py ...` command.

Run: `ls -la code-rag-cli.py code-rag-mcp.py | awk '{print $1, $NF}'`
Expected: both files show `-rwxrwxr-x` (or `-rwxr-xr-x`) executable bits.

- [ ] **Step 5: Commit**

```bash
git add Makefile scripts/add_pi_mcp.py
git update-index --chmod=+x code-rag-cli.py
git add code-rag-cli.py
git commit -m "Fix make targets and add_pi_mcp.py for HTTP-based MCP"
```

---

## Task 2: Update `README.md`

**Files:**
- Modify: `README.md` (full pass over Quickstart, CLI, Serving, Agent integration, MCP tools, Make targets, Architecture)

- [ ] **Step 1: Replace the Quickstart section**

In `README.md`, replace the Quickstart block (lines 29-40):

````markdown
## Quickstart

```bash
# Install dependencies
make install

# Index a codebase
make index SRC=../my-project DB=./index.db

# Start the REST API + web UI (primary interface)
make webui DB=./index.db

# In another terminal, start the MCP server pointing at the web UI
make mcp
```
````

- [ ] **Step 2: Rewrite the CLI reference**

Replace lines 52-57:

````markdown
### CLI

```
code-rag index [paths...] [options]    Build or update the index
code-rag webui [options]               Start the REST API + web UI
code-rag-cli.py SUBCOMMAND [args]      Stdlib CLI client to a running web UI
code-rag-mcp.py [options]              MCP server proxying tools to the web UI
```
````

- [ ] **Step 3: Replace the "Serving" section with "Web UI" and "MCP server"**

Replace lines 91-104:

````markdown
### Web UI (REST API)

The primary interface. Serves the REST API (documented in `SKILL.md`) and the browser UI.

```bash
code-rag webui --db index.db --port 8081
```

**Webui options:**

| Flag | Default | Description |
|---|---|---|
| `--db PATH` | `./index.db` | Index file location |
| `--host HOST` | `0.0.0.0` | Bind address |
| `--port N` | `8080` | Listen port |
| `--embed-model MODEL` | (from DB) | Override the embedding model |
| `--summarizer {anthropic,ollama}` | `ollama` | Summarization backend used for reindex jobs triggered via the web UI |

### MCP server

A standalone script that proxies MCP tool calls to a running web UI over REST. Stdio by default; pass `--http` for streamable-HTTP transport.

```bash
# Stdio MCP server (default — for direct agent stdin/stdout integration)
python code-rag-mcp.py --base-url http://localhost:8081

# Or via the env var
CODE_RAG_URL=http://localhost:8081 python code-rag-mcp.py
```

**MCP server options:**

| Flag | Default | Description |
|---|---|---|
| `--base-url URL` | `http://localhost:8081` | Web UI to talk to (env: `CODE_RAG_URL`) |
| `--http` | off | Run as streamable-HTTP MCP server on `127.0.0.1` |
| `--port N` | `8000` | Port for `--http` mode |

### Standalone CLI client

For shell scripts and human use, `code-rag-cli.py` wraps the REST API in plain-text output:

```bash
python code-rag-cli.py search "how does authentication work?"
python code-rag-cli.py --json repos | jq '.[].name'
python code-rag-cli.py index /path/to/repo --wait
```

Subcommands: `search`, `unit`, `fetch`, `units`, `files`, `repos`, `status`, `browse`, `index`, `index-status`, `index-cancel`, `clear-repo`, `staleness`, `ls`. The top-level `--json` flag emits raw JSON from any subcommand.
````

- [ ] **Step 4: Update the agent-integration table and system prompt**

Replace lines 105-120:

````markdown
### Agent integration

Register `code-rag-mcp.py` with your agent of choice. Both commands accept `BASE_URL=` to point at a non-default web UI.

| Agent | Register | Unregister |
|---|---|---|
| Claude Code | `make add-claude-mcp` | `make remove-claude-mcp` |
| pi-agent | `make add-pi-mcp` | `make remove-pi-mcp` |

The web UI must be running for the MCP server to work — start it once on the host that owns the index, then point agents at it.

#### System prompt

For best results, include the following in your agent's system prompt or persona config:

> code-rag is a RAG server for an index of code repositories indexed to accelerate design, debugging, and discovery of relevant code prior to the use of filesystem tools. Use code-rag for vague or exploratory queries about a codebase; start with `index_status` then discover relevant code using the search tool with natural language topic descriptions.

For agents that support per-project instruction files (e.g. `AGENTS.md` for Claude Code), place this text there so it applies automatically whenever you work in an indexed repo.
````

- [ ] **Step 5: Update the Make targets table**

Replace line 131 (`serve` row) with:

```markdown
| `mcp` | Start MCP server (stdio) | `make mcp BASE_URL=http://host:8081` |
| `webui` | Start REST API + web UI | `make webui DB=my.db PORT=8081` |
```

(Remove the old `serve` row entirely. The `webui` row already exists — verify it's present and matches.)

- [ ] **Step 6: Replace the "MCP tools" section**

Replace lines 161-174 (the `search` and `index_status` parameter tables):

````markdown
## MCP tools

`code-rag-mcp.py` exposes 10 tools, all proxied to the REST API documented in `SKILL.md`:

| Tool | Purpose |
|---|---|
| `search` | Natural-language vector search; returns ranked unit summaries |
| `get_unit` | Fetch full source for one or more qualified paths |
| `list_units` | List indexed units with optional glob filter |
| `list_files` | List indexed files with optional glob filter |
| `list_repos` | List indexed repositories |
| `index_status` | Per-repo file/unit counts and last-indexed timestamp |
| `staleness` | Per-repo index freshness vs. each repo's git HEAD |
| `index_start` | Enqueue paths for indexing (returns immediately) |
| `index_job_status` | Poll the running indexing job |
| `index_cancel` | Signal the running job to cancel |

Full parameter and response schemas are in `SKILL.md` (generated from the live OpenAPI spec).
````

- [ ] **Step 7: Update the Architecture section**

Replace lines 176-182:

````markdown
## Architecture

- **Web UI** (`mcp_rag/webui.py`): FastAPI ASGI app — REST API + browser UI; single source of truth for SQLite reads/writes
- **MCP server** (`code-rag-mcp.py`): standalone script proxying 10 MCP tools over HTTP to the web UI
- **CLI client** (`code-rag-cli.py`): standalone stdlib-only script for human/script use
- **Embeddings**: [fastembed](https://github.com/qdrant/fastembed) in-process via ONNX Runtime (`nomic-ai/nomic-embed-text-v1.5-Q`, 768-dim)
- **Storage**: SQLite (WAL mode) + [sqlite-vec](https://github.com/asg017/sqlite-vec) — documents, metadata, and vectors in a single file
- **Summarization**: Anthropic API (Claude Haiku) or Ollama, index-time only
- **MCP transport**: stdio by default; streamable-HTTP via `--http`
````

- [ ] **Step 8: Verify, then commit**

Run: `grep -n "code-rag serve\|make serve\b" README.md`
Expected: no matches.

```bash
git add README.md
git commit -m "Update README for HTTP-based MCP architecture"
```

---

## Task 3: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the project description (line 9)**

Replace:

> The primary interface is a **REST API** served by the web UI mode. `skill.md` documents the API for LLM consumption. An MCP stdio server is also available as a secondary interface.

With:

> The primary interface is a **REST API** served by the `webui` subcommand. `skill.md` documents the API for LLM consumption. `code-rag-mcp.py` (at repo root) is a standalone MCP server that proxies tool calls to the REST API; `code-rag-cli.py` is a parallel stdlib CLI client for humans/scripts.

- [ ] **Step 2: Update the Commands block (lines 28-35)**

Replace the last two `# Start ...` lines with:

```bash
# Start REST API + web UI (primary interface)
uv run code-rag webui --db ./index.db --port 8081

# Start MCP server (stdio) — talks to the running webui
python code-rag-mcp.py --base-url http://localhost:8081
```

- [ ] **Step 3: Rewrite the Architecture section (lines 37-51)**

Replace the entire "Three modes" block with:

````markdown
## Architecture

Two CLI subcommands dispatched from `mcp_rag/__main__.py`:

**INDEX mode** (`indexer.py`) — offline pipeline:
1. **Discovery** (`discovery.py`): find git repos, enumerate files via `git ls-files`
2. **Parse** (`parsers.py`): extract semantic units per language (Python via `ast`, C/C++/JS/TS/Java via tree-sitter, Go via subprocess, markdown/SQL/Terraform via regex)
3. **Summarize** (`summarizer.py`): call Claude Haiku (or Ollama) to write a natural-language summary per unit; skips unchanged units by `content_md5`
4. **Embed** (`embedder.py`): in-process fastembed (ONNX Runtime, nomic 768-dim) on the summaries
5. **Reconcile** (`reconcile.py`): diff units vs. DB state; only re-summarize changed/new units
6. **Write** (`db.py`): SQLite with WAL mode + sqlite-vec virtual table for ANN

**WEBUI mode** (`webui.py`) — FastAPI ASGI app serving the REST API and browser UI. OpenAPI spec auto-generated at `/openapi.json`; interactive docs at `/docs`. Indexing runs in a daemon thread decoupled from any client connection. **All SQLite access happens here** — `code-rag-mcp.py` and `code-rag-cli.py` are HTTP clients only.

### Top-level scripts (HTTP clients to the web UI)

- `code-rag-mcp.py` — MCP server with 10 tools; stdio by default, `--http` for streamable-HTTP
- `code-rag-cli.py` — stdlib-only Python CLI for humans/scripts; supports `--json` raw output

Both are standalone (copyable to another machine; no install needed beyond Python + `fastmcp` for the MCP). HTTP helpers are duplicated between them on purpose — keeps each self-contained.
````

- [ ] **Step 4: Update the "Shared data access layer" subsection (lines 53-55)**

Replace:

> `queries.py` is the single source of truth for all DB reads. Both `server.py` (MCP) and `webui.py` (REST) call into it — no SQL is duplicated between them.

With:

> `queries.py` is the single source of truth for all DB reads, used exclusively by `webui.py`. The MCP and CLI scripts at repo root never touch SQLite directly — they always go through the REST API.

- [ ] **Step 5: Verify and commit**

Run: `grep -n "server\.py\|code-rag serve\|SERVE mode" CLAUDE.md`
Expected: no matches.

```bash
git add CLAUDE.md
git commit -m "Update CLAUDE.md for HTTP-based MCP architecture"
```

---

## Task 4: Update `overview.md`

**Files:**
- Modify: `overview.md`

- [ ] **Step 1: Replace the "runtime design" section**

Replace lines 49-59 (the entire `## runtime design` section) with:

````markdown
## runtime design

- **index mode** (`code-rag index [paths...]`): parses each file into semantic units, generates
  a Claude summary per unit with source-path context, and embeds the summary into the vector
  index. Incremental — only changed files are re-processed. Requires `ANTHROPIC_API_KEY` when
  `--summarizer anthropic` is used; `--summarizer ollama` is the default and is offline.

- **webui mode** (`code-rag webui [options]`): starts the FastAPI server. Exposes the REST API
  (documented in `SKILL.md`) and a browser UI. All read and write access to the index goes
  through this service.

Two standalone HTTP clients sit on top of the web UI:

- **`code-rag-mcp.py`** — an MCP server that proxies 10 tools to the REST API; stdio by
  default, `--http` for streamable-HTTP transport. Embeds each incoming query through the
  web UI's `/api/search` endpoint and returns matched units.

- **`code-rag-cli.py`** — a stdlib-only CLI client for humans and shell scripts; mirrors the
  REST API as named subcommands and supports a top-level `--json` flag for raw passthrough.
````

- [ ] **Step 2: Commit**

```bash
git add overview.md
git commit -m "Update overview.md runtime design for HTTP-based MCP"
```

---

## Task 5: Update `design_spec.md`

**Files:**
- Modify: `design_spec.md`

This is targeted: only the sections this branch invalidated, not a wholesale rewrite. The pre-existing `mcp-rag` → `code-rag` rename drift is out of scope (separate work).

- [ ] **Step 1: Update the CLI block (lines 609-613)**

Replace:

```
mcp-rag [paths...]                    Index if absent, then serve (stdio)
mcp-rag index [paths...] [options]    Build or update the index
mcp-rag serve [options]               Start the MCP server
```

With:

```
code-rag index [paths...] [options]   Build or update the index
code-rag webui [options]              Start the REST API + web UI

code-rag-cli.py SUBCOMMAND [args]     Standalone stdlib CLI client to a running web UI
code-rag-mcp.py [options]             Standalone MCP server proxying tools to a running web UI
```

- [ ] **Step 2: Replace the `serve` options table (lines 623-629) with `webui` options**

Replace the entire `**`serve` options**` block with:

````
**`webui` options**

| Flag | Default | Description |
|---|---|---|
| `--db PATH` | `./index.db` | Index file location |
| `--host HOST` | `0.0.0.0` | Bind address |
| `--port N` | `8080` | Listen port |
| `--embed-model MODEL` | (from DB) | Override embedding model (defaults to what the DB was indexed with) |
| `--summarizer {anthropic,ollama}` | `ollama` | Summarization backend used by reindex jobs triggered via the web UI |
````

- [ ] **Step 3: Update the Multi-repository Behaviour intro (line 651)**

Replace:

> Multiple paths may be passed to `mcp-rag index` or `mcp-rag` (combined mode):

With:

> Multiple paths may be passed to `code-rag index`:

- [ ] **Step 4: Delete the entire "Combined-mode Behaviour" section (lines 685-690)**

Remove the section header and its single paragraph. The next section ("First-run Behaviour") follows directly after the "Multi-repository Behaviour" section.

- [ ] **Step 5: Update the environment-variable note (line 635)**

In the environment-variables table, the row for `ANTHROPIC_API_KEY` currently reads:

> Required for `mcp-rag index`; checked at startup before any file I/O

Replace with:

> Required when `code-rag index --summarizer anthropic` is used; checked at startup before any file I/O

- [ ] **Step 6: Add a top-level note about the data path**

Find the "MCP server" row in the tech-stack table (around line 15):

```
| MCP server | `fastmcp`, stdio transport (default); Streamable HTTP optional |
```

Replace with:

```
| MCP server | `code-rag-mcp.py` (root) — `fastmcp`, stdio default, `--http` for streamable-HTTP. Proxies all reads/writes to the web UI; never touches SQLite directly. |
| Web UI | `mcp_rag/webui.py` — FastAPI ASGI app; sole owner of SQLite read/write |
```

- [ ] **Step 7: Verify**

Run: `grep -nE "mcp-rag (serve|combined)|Combined-mode" design_spec.md`
Expected: no matches.

- [ ] **Step 8: Commit**

```bash
git add design_spec.md
git commit -m "Update design_spec for HTTP-only data path; drop combined mode"
```

---

## Task 6: Extend `make lint` to cover top-level scripts

**Files:**
- Modify: `Makefile` (lint and format targets)

- [ ] **Step 1: Update `lint` and `format` to include the root scripts**

In `Makefile`, replace:

```makefile
lint:
	uv run ruff check mcp_rag tests
	uv run ruff format --check mcp_rag tests

format:
	uv run ruff format mcp_rag tests
```

With:

```makefile
lint:
	uv run ruff check mcp_rag tests code-rag-cli.py code-rag-mcp.py scripts
	uv run ruff format --check mcp_rag tests code-rag-cli.py code-rag-mcp.py scripts

format:
	uv run ruff format mcp_rag tests code-rag-cli.py code-rag-mcp.py scripts
```

- [ ] **Step 2: Verify the broader scope is clean**

Run: `make lint`
Expected: PASS.

If any new file fails format check, run `make format` and commit the diff before proceeding.

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "Extend make lint to cover top-level scripts and scripts/"
```

---

## Task 7: Bring `code-rag.sh` to parity with `code-rag-cli.py`

**Files:**
- Modify: `code-rag.sh`

Add `CODE_RAG_URL` env var fallback, `staleness` and `ls` subcommands, `--json` raw passthrough, and `--wait` on `index`. Also fix the duplicated `case "$1"` block at lines 247-252.

- [ ] **Step 1: Read the current `code-rag.sh` to identify its structure**

```bash
wc -l code-rag.sh
```
Expected: ~268 lines.

Read the file to find: the `BASE_URL=` default at the top, the `_get`/`_post` helpers, the `cmd_*` functions, the usage text block, and the entry-point `case` block.

- [ ] **Step 2: Replace the `BASE_URL` default and add JSON_OUTPUT flag**

In `code-rag.sh`, replace line 4 (`BASE_URL="http://localhost:8081"`) with:

```bash
BASE_URL="${CODE_RAG_URL:-http://localhost:8081}"
JSON_OUTPUT=0
```

- [ ] **Step 3: Add an `_emit` helper after `_post`**

After the `_post` function block, add:

```bash
_emit() {
  # Reads JSON from stdin. If --json was set, pretty-prints and returns 0
  # (caller should `return` after calling). Else returns 1 so caller formats.
  if [[ "$JSON_OUTPUT" == "1" ]]; then
    jq '.'
    return 0
  fi
  return 1
}
```

- [ ] **Step 4: Route each `cmd_*` through `_emit`**

For every `cmd_*` function that produces output, capture the API response into a local variable, then check `_emit`:

```bash
cmd_search() {
  local query="" top_k=5 glibs=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --top-k) top_k="$2"; shift 2 ;;
      --glob)  glibs+=("$2"); shift 2 ;;
      *)       query="$1"; shift ;;
    esac
  done
  [[ -z "$query" ]] && _die "search requires a query"
  local qparams=(q "$query" top_k "$top_k")
  for g in "${glibs[@]}"; do qparams+=(globs "$g"); done
  local out
  out="$(_get "/api/search" "${qparams[@]}")"
  echo "$out" | _emit && return
  echo "$out" | jq -r '.[] | "\(.score)\t\(.path)\n  \(.summary)"'
}
```

Apply the same shape (capture into `out`, pipe to `_emit && return`, then pipe to the existing formatter) to: `cmd_unit`, `cmd_fetch`, `cmd_units`, `cmd_files`, `cmd_repos`, `cmd_status`, `cmd_browse`, `cmd_clear_repo`, and (after Step 5) `cmd_staleness`, `cmd_ls`. The job-status commands (`cmd_index`, `cmd_index_status`, `cmd_index_cancel`) already route through `_print_job_status` — update `_print_job_status` instead to call `_emit` first.

`_print_job_status` becomes:

```bash
_print_job_status() {
  local data="$1"
  echo "$data" | _emit && return
  echo "running: $(echo "$data" | jq -r '.running')"
  local lr lf
  lr="$(echo "$data" | jq -r '.last_result // empty')"
  lf="$(echo "$data" | jq -r '.last_finished_at // empty')"
  [[ -n "$lr" ]] && echo "last_result: $lr"
  [[ -n "$lf" ]] && echo "last_finished_at: $lf"
}
```

- [ ] **Step 5: Add `cmd_staleness` and `cmd_ls`**

```bash
cmd_staleness() {
  local out
  out="$(_get "/api/repos/staleness")"
  echo "$out" | _emit && return
  echo "$out" | jq -r '.[] | "\(.repo)\tindexed=\(.last_indexed_at // "never")\thead=\(.last_commit_at // "unknown")\tstale=\(.stale)\t\(.reason)"'
}

cmd_ls() {
  local path="${1:-}"
  local out
  if [[ -n "$path" ]]; then
    out="$(_get "/api/ls" path "$path")"
  else
    out="$(_get "/api/ls")"
  fi
  echo "$out" | _emit && return
  local marker=""
  [[ "$(echo "$out" | jq -r '.is_git')" == "true" ]] && marker=" *"
  echo "$(echo "$out" | jq -r '.path')$marker"
  echo "$out" | jq -r '.dirs[].name'
}
```

- [ ] **Step 6: Add `--wait` to `cmd_index`**

Replace `cmd_index`:

```bash
cmd_index() {
  local reindex="false" paths=() wait_flag=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --reindex) reindex="true"; shift ;;
      --wait)    wait_flag=1; shift ;;
      *) paths+=("$1"); shift ;;
    esac
  done
  [[ ${#paths[@]} -eq 0 ]] && _die "index requires at least one path"
  local body
  body="$(printf '%s\n' "${paths[@]}" | jq -R -s -c --argjson r "$reindex" 'split("\n") | map(select(length > 0)) | {paths: ., reindex: $r}')"
  local status
  status="$(_post "/api/index" "$body")"
  if [[ "$wait_flag" -eq 0 ]]; then
    _print_job_status "$status"
    return
  fi
  while [[ "$(echo "$status" | jq -r '.running')" == "true" ]]; do
    sleep 2
    printf '.' >&2
    status="$(_get "/api/index/status")"
  done
  printf '\n' >&2
  _print_job_status "$status"
  local last_result
  last_result="$(echo "$status" | jq -r '.last_result // empty')"
  [[ "$last_result" != "ok" ]] && exit 1
}
```

- [ ] **Step 7: Register `staleness` and `ls`, add `--json` parsing, fix the duplicated case block**

In the entry-point dispatch block (around line 243), replace the entire post-usage logic:

```bash
[[ $# -eq 0 ]] && { usage; exit 1; }

# Parse top-level flags (--base-url, --json, --help) in any order before the subcommand
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)   usage; exit 0 ;;
    --base-url)  BASE_URL="$2"; shift 2 ;;
    --json)      JSON_OUTPUT=1; shift ;;
    *) break ;;
  esac
done

case "${1:-}" in
  search)       shift; cmd_search "$@" ;;
  unit)         shift; cmd_unit "$@" ;;
  fetch)        shift; cmd_fetch "$@" ;;
  units)        shift; cmd_units "$@" ;;
  files)        shift; cmd_files "$@" ;;
  repos)        shift; cmd_repos "$@" ;;
  status)       shift; cmd_status "$@" ;;
  browse)       shift; cmd_browse "$@" ;;
  index)        shift; cmd_index "$@" ;;
  index-status) shift; cmd_index_status "$@" ;;
  index-cancel) shift; cmd_index_cancel "$@" ;;
  clear-repo)   shift; cmd_clear_repo "$@" ;;
  staleness)    shift; cmd_staleness "$@" ;;
  ls)           shift; cmd_ls "$@" ;;
  *)            _die "unknown command: ${1:-}"; usage >&2; exit 1 ;;
esac
```

Replace the existing duplicated `case "$1" in ... --base-url) BASE_URL="$2"; shift 2 ;; esac` blocks entirely.

- [ ] **Step 8: Update the usage text**

Replace the `usage()` body to add `staleness`, `ls`, and the new flags:

```bash
usage() {
  cat <<'EOF'
Usage: code-rag.sh [--base-url URL] [--json] COMMAND [ARGS...]

Commands:
  search  QUERY                Search indexed code
          --top-k N            (default: 5)
          --glob GLOB          (repeatable)

  unit    PATH                 Get a single unit by qualified path
  fetch   PATH...              Fetch multiple units by qualified path

  units                        List semantic units
          --limit N            (default: 100)
          --glob GLOB          (repeatable)

  files                        List indexed files
          --glob GLOB          (repeatable)

  repos                        List indexed repositories
  status                       Index health check
  staleness                    Per-repo index freshness vs git HEAD
  browse  [PATH]               Browse the index tree
  ls      [PATH]               List filesystem directories (server-side)

  index   PATH...              Start an indexing job
          --reindex
          --wait               Block until job completes; exit non-zero on failure

  index-status                 Poll indexing job state
  index-cancel                 Cancel running indexing job

  clear-repo REPO              Remove indexed data for a repository

Options:
  --base-url URL               Base URL (env: CODE_RAG_URL, default: http://localhost:8081)
  --json                       Emit raw JSON instead of formatted output
  -h, --help                   Show this help
EOF
}
```

- [ ] **Step 9: Verify**

Run: `bash code-rag.sh --help`
Expected: usage shows new commands and flags.

Run: `bash code-rag.sh --json repos 2>&1 | head -3`
Expected: either JSON output (if webui is running) or an error about unreachable server — no shell errors.

Run: `CODE_RAG_URL=http://localhost:8081 bash code-rag.sh repos 2>&1 | head -3`
Expected: same as above; env var was read.

- [ ] **Step 10: Commit**

```bash
git add code-rag.sh
git commit -m "Bring code-rag.sh to parity with code-rag-cli.py"
```

---

## Task 8: Add `staleness` tool to `code-rag-mcp.py`

**Files:**
- Modify: `code-rag-mcp.py` (add one tool)
- Modify: `tests/unit/test_mcp_client.py` (add one test)

- [ ] **Step 1: Add the test first**

In `tests/unit/test_mcp_client.py`, in the `# Read tools` section (after `test_index_status_returns_repos_list`), add:

```python
def test_staleness_returns_list(mcp_mod):
    payload = [{
        "repo": "r", "root": "/r",
        "last_indexed_at": "t1", "last_commit_at": "t2",
        "stale": True, "reason": "older than HEAD",
    }]
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)) as m:
        result = asyncio.run(mcp_mod.staleness())
    assert "/api/repos/staleness" in m.call_args[0][0].full_url
    assert result == payload
```

Run: `~/.local/bin/uv run pytest tests/unit/test_mcp_client.py::test_staleness_returns_list -v`
Expected: FAIL — `AttributeError` because `mcp_mod.staleness` does not exist.

- [ ] **Step 2: Add the tool to `code-rag-mcp.py`**

After the `index_status` tool definition, add:

```python
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
```

- [ ] **Step 3: Run the test — expect PASS**

Run: `~/.local/bin/uv run pytest tests/unit/test_mcp_client.py -v`
Expected: 12 passed (was 11; one new).

- [ ] **Step 4: Commit**

```bash
git add code-rag-mcp.py tests/unit/test_mcp_client.py
git commit -m "Add staleness tool to code-rag-mcp"
```

---

## Task 9: Final verification

- [ ] **Step 1: Lint and test**

Run: `make lint && make test`
Expected: PASS.

- [ ] **Step 2: Verify no `serve` references remain anywhere**

Run: `grep -rnE "code-rag serve|make serve\b|mcp-rag serve|combined.mode|SERVE mode" --include='*.md' --include='Makefile' --include='*.py' --include='*.sh' /home/kali/workspace/py-mcp-code-rag`
Expected: no matches (or only matches inside `docs/superpowers/plans/` and `docs/superpowers/specs/`, which describe historical work).

- [ ] **Step 3: Smoke test the docs flow**

Walk through the README Quickstart literally, in a scratch shell:

```bash
# Assume webui already running on :8081 (or skip)
make mcp BASE_URL=http://localhost:8081 --dry-run
# Expected: shows the new code-rag-mcp.py invocation, no error
```

- [ ] **Step 4: Confirm `code-rag-cli.py` is executable**

Run: `[[ -x code-rag-cli.py ]] && echo OK || echo MISSING`
Expected: `OK`.

---

## Self-Review

**Coverage check:** every item from the holistic review is addressed.
- Critical: Makefile (Task 1, 6), `scripts/add_pi_mcp.py` (Task 1), README (Task 2), exec bit (Task 1)
- Doc drift: CLAUDE.md (Task 3), overview.md (Task 4), design_spec.md (Task 5)
- Polish: lint scope (Task 6), code-rag.sh parity (Task 7)
- Design follow-on: staleness MCP tool (Task 8)

**Naming consistency:** `mcp` is the new Makefile target (replaces `serve`). `code-rag-mcp.py` and `code-rag-cli.py` are referenced with the same paths everywhere.

**Out of scope:**
- The pre-existing `mcp-rag` → `code-rag` rename drift in `design_spec.md` and elsewhere (this branch didn't introduce it; separate cleanup).
- Adding `clear-repo` or `ls` to the MCP — these are admin/filesystem operations, not appropriate for an LLM agent to invoke autonomously.
