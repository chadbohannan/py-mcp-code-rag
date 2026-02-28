# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

- `overview.md` — design rationale and the Semantic Surrogate Indexing technique
- `design_spec.md` — complete implementation specification (architecture, schema, CLI, testing)

When implementation diverges from `design_spec.md` — whether fixing a spec error, choosing a
simpler approach, or resolving an ambiguity — update the relevant section of `design_spec.md`
with the actual decision and a brief rationale. The spec should reflect what was built, not
just what was planned.

## Status

All modules implemented and fully tested (199/199 tests passing). The project is feature-complete.

## Test commands

```bash
uv run pytest tests/unit tests/integration -x -q   # fast dev loop
uv run pytest                                       # full suite
uv run pytest tests/unit/test_parsers.py -x -q     # single test file
```

No external services required. `FakeEmbedder` and `FakeSummarizer` isolate fastembed and the Anthropic API from all test runs.

## Dev commands

```bash
uv sync                                        # install dependencies
uv run mcp-rag index /path/to/project          # build or update the index
uv run mcp-rag serve /path/to/project          # start MCP server (stdio)
uv run mcp-rag /path/to/project                # index if absent, then serve
uv run mcp-rag serve --http /path/to/project   # HTTP transport (localhost:8000)
```

## Stack

| Component | Technology |
|---|---|
| Runtime | Python ≥ 3.12, managed with `uv` |
| MCP server | `fastmcp`, stdio transport (default) |
| Embeddings | `fastembed` in-process (ONNX Runtime) |
| Default embed model | `nomic-ai/nomic-embed-text-v1.5-Q` (768-dim) |
| Summarization | Anthropic API (`claude-haiku-4-5-20251001`), index-time only |
| Storage | SQLite WAL + `sqlite-vec` virtual table → single `index.db` |
| Go parsing | `go/ast` via bundled `mcp_rag/go_parser/main.go`; `go run` per file |

## Architecture

**Core idea — Semantic Surrogate Indexing:** files are parsed into structural units (functions, classes, CTEs, paragraphs), each unit is summarized by Claude with its file-path context, and the *summary* (not raw source) is embedded. Raw source is stored alongside and returned to callers on a match. See `overview.md`.

**Two modes, one binary:**
- `mcp-rag index` — parse → summarize (Anthropic API) → embed (fastembed) → write to SQLite. Requires `ANTHROPIC_API_KEY`.
- `mcp-rag serve` — embed incoming query → ANN lookup via `sqlite-vec` → return results. Read-only; no API key needed.

**Parsers by extension:**
- `.py` → stdlib `ast` (module-level, class, function, method)
- `.go` → Go helper subprocess emitting JSON units
- `.md` / `.mdx` → heading/paragraph splitter
- `.sql` → whole file (skipped if > 4 KB)
- everything else → silently skipped

**Change detection:** file-level fingerprint (`mtime` + `md5` in `mcp_rag_files`). On change, unit-level reconciliation using `(unit_type, unit_name, char_offset)` + `content_md5` avoids re-summarizing unchanged units.

**File discovery:** `git ls-files --cached --others --exclude-standard`; falls back to `pathlib.Path.walk` with a hardcoded exclusion list for non-git roots.

## Package layout

```
mcp_rag/
  go_parser/main.go   # Go AST helper; invoked with `go run -- <file>` per .go file
  __init__.py
  __main__.py         # CLI entry-point (index / serve / combined mode)
  db.py               # open_db, schema DDL, ModelMismatchError
  discovery.py        # discover_files, DiscoveryError
  embedder.py         # FastEmbedder (fastembed, L2-normalised), DEFAULT_MODEL
  indexer.py          # run_index, IndexAbortError
  models.py           # SemanticUnit, Embedder/Summarizer protocols
  parsers.py          # parse_python, parse_markdown, parse_sql, parse_go, parse_file
  reconcile.py        # StoredUnit, diff_units
  server.py           # MCP server: search + index_status tools, configure()
  summarizer.py       # AnthropicSummarizer (claude-haiku, retry with backoff)
tests/
  conftest.py         # FakeEmbedder(dim=4, model="fake-model"), FakeSummarizer
  unit/               # pure-logic tests, no I/O
  integration/        # full pipeline: real SQLite + real FS, FakeEmbedder + FakeSummarizer
```

## Test seams

```python
class Embedder(Protocol):
    dim: int
    model: str
    def embed(self, text: str) -> list[float]: ...

class Summarizer(Protocol):
    def summarize(self, unit: SemanticUnit) -> str: ...
```

`FakeEmbedder(dim=4)` has `model = "fake-model"` and returns deterministic unit-length vectors
(MD5-seeded). `FakeSummarizer` records every call in `.calls` and returns a deterministic string.
Both live in `tests/conftest.py` and are used in all integration tests.

## Key constraints

- Every DB connection must call `sqlite_vec.load(conn)` immediately after opening.
- `embed_model` and `embed_dim` in `mcp_rag_meta` are validated at startup; mismatch → abort with a message pointing to `--reindex`.
- `ANTHROPIC_API_KEY` is checked at startup for `mcp-rag index`; abort before any file I/O if absent.
- Progress goes to stderr; stdout is reserved for the MCP stdio protocol.
- All deletes + inserts for a single file are wrapped in one transaction.
- Summarization truncates units > 8,000 estimated tokens (`len(source) // 4`) with a stderr warning.
- Retry policy for Anthropic API: 429/529/5xx → exponential backoff ±20% jitter, 3 retries at [1 s, 4 s, 16 s].
