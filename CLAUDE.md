# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

**code-rag** is a semantic code search server. The core technique is **Semantic Surrogate Indexing**: instead of embedding raw source code, it generates an LLM-written natural-language summary of each semantic unit (function, class, method, markdown section, SQL block, directory rollup, etc.) and embeds *those summaries* into vector space. Natural-language queries then match on code semantics rather than keywords.

The summarizer backend is pluggable. The **default is Ollama with `gemma4:latest`** running locally; `claude-haiku-4-5` via the Anthropic API is available as an alternative with `--summarizer anthropic`. Model identity is **not** currently recorded in the DB — if you need to know which engine produced a given index, check how it was invoked. (See Gotchas.)

The primary interface is a **REST API** served by the web UI mode. `SKILL.md` documents the API for LLM consumption. An MCP stdio server is available as a secondary interface.

## Commands

This is a `uv` project. Use `uv run` to execute and `uv add` to add dependencies.

```bash
make install           # Install all dependencies
make test              # Full test suite
make test-unit         # Unit tests only (fast, no external deps)
make test-integration  # Integration tests
make lint              # ruff check + ruff format --check
make format            # Auto-format
make skill             # Regenerate SKILL.md from live OpenAPI spec

# Run a single test
uv run pytest tests/unit/test_parsers.py::test_parse_python_function

# Index a directory (defaults: --summarizer ollama --ollama-model gemma4:latest)
uv run code-rag index /path/to/repo --db ./index.db

# Index with Anthropic Haiku instead
uv run code-rag index /path/to/repo --db ./index.db --summarizer anthropic

# Start REST API + web UI (primary interface)
uv run code-rag webui --db ./index.db --port 8081

# Start MCP stdio server (secondary interface)
uv run code-rag serve --db ./index.db
```

Common `index` flags: `--reindex` (re-embed everything), `--embed-model` (override embedding model), `--ollama-host` (override Ollama endpoint).

## Architecture

Three modes dispatched from `mcp_rag/__main__.py`:

**INDEX mode** (`indexer.py`) — offline pipeline:
1. **Discovery** (`discovery.py`): find git repos, enumerate files via `git ls-files`
2. **Parse** (`parsers.py`): extract semantic units per language — Python via `ast`; C/C++/JS/TS/Java/Go via tree-sitter; markdown/SQL/Terraform via regex
3. **Summarize** (`summarizer.py`): call the configured backend (Ollama `gemma4:latest` by default, Anthropic Haiku optional) to write a natural-language summary per unit; unchanged units skip re-summarization via `content_md5`
4. **Embed** (`embedder.py`): in-process fastembed (ONNX Runtime, nomic 768-dim) over the summaries
5. **Reconcile** (`reconcile.py`): diff units vs. DB state; only re-summarize changed/new units
6. **Rollup** (`indexer.py:_upsert_module_unit`, `_upsert_directory_unit`): build module-level and directory-level summaries by feeding child summaries back through the summarizer. Rollups are summaries-of-summaries — a different prompting task from leaf units.
7. **Write** (`db.py`): SQLite with WAL mode + sqlite-vec virtual table for ANN

**SERVE mode** (`server.py`) — read-only MCP stdio server with 6 tools: `search`, `get_unit`, `list_units`, `list_files`, `list_repos`, `index_status`. All tools delegate to `queries.py`.

**WEBUI mode** (`webui.py`) — FastAPI ASGI app serving the REST API and browser UI. OpenAPI spec auto-generated at `/openapi.json`; interactive docs at `/docs`. Indexing runs in a daemon thread decoupled from any client connection.

### Shared data access layer

`queries.py` is the single source of truth for all DB reads. Both `server.py` (MCP) and `webui.py` (REST) call into it — no SQL is duplicated between them.

### Indexing job state

`job.py` holds the singleton indexing job state (`running`, `cancel_event`, `last_result`). Both the WebSocket handler and the REST endpoints (`POST /api/index`, `GET /api/index/status`, `POST /api/index/cancel`) share this module. A job started via REST can be cancelled via WebSocket and vice versa.

### REST API

Endpoints follow a consistent pattern — see `SKILL.md` for the full reference. Key endpoints:

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

When the API changes, regenerate `SKILL.md` with `make skill`.

### Database Schema (SQLite + sqlite-vec)

```
metadata    -- embed_model, embed_dim, schema_version
repos       -- git repositories (id, name, root, added_at)
files       -- indexed files (id, repo_id, path, mtime, md5, indexed_at)
units       -- semantic units (id, repo_id, file_id, path, content, summary, unit_type, unit_name, char_offset)
embeddings  -- sqlite-vec virtual table (unit_id, embedding FLOAT[768])
```

Qualified path format: `repo_name/relative/file.py:Class:method`

`unit_type` in the current corpus spans: `function`, `method`, `class`, `struct`, `interface`, `enum`, `paragraph`, `h1`–`h4`, `module` (file-level rollup), `directory` (directory-level rollup). The eval harness groups these into three slices (`leaf_code`, `markdown`, `rollup`) because each responds to prompt changes differently — see `eval/harness/slices.py`.

**Rollup `content` is not persisted.** Leaf units store their raw source in `units.content`; module and directory rollups build their input text at index time from child summaries and discard it after summarization. This is a known gap that makes some rollup analyses (e.g., symbol-grounding intrinsic metric) impossible without re-reconstruction.

## Eval harness (`eval/`)

The `eval/` folder contains a prompt-quality harness that ratchets prompt variants toward monotonic improvement. The claim depends on four properties enforced by the tooling: frozen versioned query sets, held-out isolation, everything-but-the-prompt held fixed, and a per-slice promotion rule with explicit guardrails.

### Layout

```
eval/
├── queries/
│   ├── dev.jsonl           # iteration set; schema in eval/harness/label.py
│   ├── heldout.jsonl       # champion-promotion only
│   ├── adversarial.jsonl   # hallucination probes (must_not_include)
│   └── smoke.jsonl         # must-rank-1 regression tests
├── variants/
│   └── v0_baseline.py      # frozen copy of the in-tree prompt as starting champion
├── harness/
│   ├── label.py            # interactive CLI to grow query sets against a live index
│   ├── slices.py           # unit_type → slice mapping; MIN_HELDOUT_SLICE_N threshold
│   ├── intrinsic.py        # label-free per-slice text-quality metrics
│   ├── score.py            # writes a run receipt with extrinsic + intrinsic metrics
│   └── ratchet.py          # per-slice bootstrap promotion vs. champion
├── runs/                   # run receipts, one per (variant, corpus_sha) score
├── CHAMPION                # JSON map {slice: run_receipt_path}
└── CHAMPIONS.log           # one line per bootstrap or promotion
```

### Workflow

1. **Score** a candidate variant's index:
   ```bash
   uv run python -m eval.harness.score --db <path> --variant-id <id>
   ```
   Produces `eval/runs/<id>-<ts>.json` containing per-slice MRR@10, Recall@k, adversarial FPR, smoke pass rate, and intrinsic metrics (banned-preamble rate, name restatement, symbol grounding).

2. **Ratchet** the candidate against the current champion:
   ```bash
   uv run python -m eval.harness.ratchet --candidate <receipt>
   ```
   Per slice, candidate is promoted only when its held-out MRR@10 gain is statistically significant (paired bootstrap, p<0.05) AND no guardrail regresses (adversarial FPR, smoke pass rate, per-slice dev MRR within a 0.02 budget). Each slice can be won independently; `CHAMPION` is a mapping, not a single winner.

3. **Grow the query set** interactively:
   ```bash
   uv run python -m eval.harness.label --db <path> --out eval/queries/dev.jsonl --query "<natural language>"
   ```

### Ratchet rules (constants in `eval/harness/ratchet.py`)

- `PRIMARY_P = 0.05` — bootstrap significance threshold on held-out MRR@10
- `ADV_FPR_BUDGET = 0.00` — hallucination regressions strictly forbidden
- `DEV_REGRESSION_BUDGET = 0.02` — per-slice dev MRR may slip by up to 2 points
- `MIN_HELDOUT_SLICE_N = 20` — slices with fewer held-out queries have their ratchet disarmed

### Intrinsic metrics (`intrinsic.py`)

Computed over all summaries in the DB, per slice. Cheap (seconds), label-free, diagnostic:

- `banned_preamble_rate` — summaries starting with "This function/method/class/…"
- `name_restatement_mean` — overlap between first 10 summary words and unit_name
- `symbol_grounding_mean` — fraction of CamelCase/backticked tokens in the summary that appear in the source (hallucination catch). Returns N/A for rollups in the current schema — see rollup-content gap above.
- Length distribution per slice

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
- **sqlite-vec loading**: Every DB connection must call `conn.enable_load_extension(True)` then `sqlite_vec.load(conn)` immediately after opening. Without enable_load_extension the loader raises "not authorized".
- **Change detection**: Unchanged units (same `content_md5`) are not re-summarized even if the file changed — this saves tokens during incremental indexing.
- **Repo name collisions**: Auto-resolved by prepending parent directory segments with `-` (e.g., `a-backend`, `b-backend`).
- **Summarizer identity is not persisted**: `metadata` records the embedding model but not the summarizer engine/model. An index built with gemma4:latest and one built with claude-haiku-4-5 are indistinguishable on-disk. Any comparison across them (e.g., the eval ratchet) assumes the caller knows what produced each DB.
- **Rollup content is discarded after summarization**: `units.content` is empty for `unit_type IN ('module', 'directory')`. Consumers that need the rollup's input text must reconstruct it from child summaries (same builders in `indexer.py:_build_module_content` / `_build_directory_content`).
- **New `unit_type` values must be mapped in `eval/harness/slices.py`**: The eval scorer refuses to run against a DB that contains an unmapped type. This is intentional — unmapped types would silently drop out of per-slice metrics.
- **Pydantic models**: Request/response shapes live in `api_models.py`. Add new models there before adding new endpoints.
- **SKILL.md is generated**: Do not edit it by hand — run `make skill` after any API change.
