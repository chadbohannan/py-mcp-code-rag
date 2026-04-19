# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

**code-rag** is a semantic code search server. The core technique is **Semantic Surrogate Indexing**: instead of embedding raw source code, it generates Claude-written natural-language summaries of each code unit (function, class, method, SQL block, etc.) and embeds *those summaries* into vector space. This makes natural-language queries match effectively against code semantics, not just keywords.

The primary interface is a **REST API** served by the web UI mode. `skill.md` documents the API for LLM consumption. An MCP stdio server is also available as a secondary interface.

## Commands

This is a `uv` project. Use `uv run` to execute and `uv add` to add dependencies.

```bash
make install           # Install all dependencies
make test              # Full test suite
make test-unit         # Unit tests only (fast, no external deps)
make test-integration  # Integration tests
make lint              # ruff check + ruff format --check
make format            # Auto-format
make skill             # Regenerate skill.md from live OpenAPI spec

# Run a single test
uv run pytest tests/unit/test_parsers.py::test_parse_python_function

# Index a directory
uv run code-rag index /path/to/repo --db ./index.db

# Start REST API + web UI (primary interface)
uv run code-rag webui --db ./index.db --port 8081

# Start MCP stdio server (secondary interface)
uv run code-rag serve --db ./index.db
```

## Architecture

Three modes dispatched from `mcp_rag/__main__.py`:

**INDEX mode** (`indexer.py`) — offline pipeline:
1. **Discovery** (`discovery.py`): find git repos, enumerate files via `git ls-files`
2. **Parse** (`parsers.py`): extract semantic units per language (Python via `ast`, C/C++/JS/TS/Java via tree-sitter, Go via subprocess, markdown/SQL/Terraform via regex)
3. **Summarize** (`summarizer.py`): call Claude Haiku (or Ollama) to write a natural-language summary per unit; skips unchanged units by `content_md5`
4. **Embed** (`embedder.py`): in-process fastembed (ONNX Runtime, nomic 768-dim) on the summaries
5. **Reconcile** (`reconcile.py`): diff units vs. DB state; only re-summarize changed/new units
6. **Write** (`db.py`): SQLite with WAL mode + sqlite-vec virtual table for ANN

**SERVE mode** (`server.py`) — read-only MCP stdio server with 6 tools: `search`, `get_unit`, `list_units`, `list_files`, `list_repos`, `index_status`. All tools delegate to `queries.py`.

**WEBUI mode** (`webui.py`) — FastAPI ASGI app serving the REST API and browser UI. OpenAPI spec auto-generated at `/openapi.json`; interactive docs at `/docs`. Indexing runs in a daemon thread decoupled from any client connection.

### Shared data access layer

`queries.py` is the single source of truth for all DB reads. Both `server.py` (MCP) and `webui.py` (REST) call into it — no SQL is duplicated between them.

### Indexing job state

`job.py` holds the singleton indexing job state (`running`, `cancel_event`, `last_result`). Both the WebSocket handler and the REST endpoints (`POST /api/index`, `GET /api/index/status`, `POST /api/index/cancel`) share this module. A job started via REST can be cancelled via WebSocket and vice versa.

### REST API

Endpoints follow a consistent pattern — see `skill.md` for the full reference. Key endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/search` | Vector search by natural-language query |
| `GET` | `/api/unit` | Fetch single unit source by qualified path |
| `POST` | `/api/units/fetch` | Fetch multiple units by qualified path (no URL length limit) |
| `GET` | `/api/units` | List units with optional glob filter |
| `GET` | `/api/status` | Index health check |
| `GET` | `/api/browse` | Tree navigation: repos → dirs → files → units |
| `POST` | `/api/index` | Start indexing job (202, non-blocking) |
| `GET` | `/api/index/status` | Poll job state |
| `POST` | `/api/index/cancel` | Signal cancellation |

Glob filters use SQLite GLOB syntax and are passed as repeated query params: `?globs=*.py&globs=backend/*`. Multiple globs are AND'd.

When the API changes, regenerate `skill.md` with `make skill`.

### Database Schema (SQLite + sqlite-vec)

```
metadata    -- embed_model, embed_dim, schema_version
repos       -- git repositories (id, name, root, added_at)
files       -- indexed files (id, repo_id, path, mtime, md5, indexed_at)
units       -- semantic units (id, repo_id, file_id, path, content, summary, unit_type, unit_name, char_offset)
embeddings  -- sqlite-vec virtual table (unit_id, embedding FLOAT[768])
```

Qualified path format: `repo_name/relative/file.py:Class:method`

## Testing

Two tiers with test fakes (no real API calls needed):
- `tests/unit/` — pure logic: parsing, fingerprinting, reconciliation
- `tests/integration/` — full pipeline with real SQLite and sqlite-vec

Key fixtures in `tests/conftest.py`:
- `FakeEmbedder`: deterministic MD5-seeded vectors
- `FakeSummarizer`: deterministic summaries + call log (assert unchanged units are not re-summarized)
- `git_init()`, `make_git_repo()`: test repo helpers

## Spec-Driven Workflow

`design_spec.md` is the source of truth. `overview.md` explains the core technique.

1. Read `design_spec.md` before starting any task
2. Fix spec error → confirm with tests → update spec
3. Choose simpler approach → update spec with rationale
4. Spec ambiguity → make explicit choice → log it in spec
5. Before committing → verify spec still matches code

Spec deviations without comments explaining the change are rejected in review.

## Gotchas

- **New language support**: Adding a parser in `parsers.py` requires also adding the extension to `_SUPPORTED_EXTENSIONS` in `indexer.py`. The allowlist gates processing before `parse_file` is called — missing it causes silent no-ops.
- **sqlite-vec loading**: Every DB connection must call `sqlite_vec.load(conn)` immediately after opening.
- **Change detection**: Unchanged units (same `content_md5`) are not re-summarized even if the file changed — this saves API calls during incremental indexing.
- **Repo name collisions**: Auto-resolved by prepending parent directory segments with `-` (e.g., `a-backend`, `b-backend`).
- **Pydantic models**: Request/response shapes live in `api_models.py`. Add new models there before adding new endpoints.
- **skill.md is generated**: Do not edit it by hand — run `make skill` after any API change.
