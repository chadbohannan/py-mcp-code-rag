# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

- `overview.md` — design rationale and the Semantic Surrogate Indexing technique
- `design_spec.md` — complete implementation specification (architecture, schema, CLI, testing)

## Status

This is a specification-only repository. No source code exists yet. The task is to implement `mcp-rag` from the specs above.

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
  go_parser/main.go     # Go AST helper; invoked with `go run` per file
  ...                   # Python modules to be created per design_spec.md
tests/
  unit/                 # pure-logic tests, no I/O
  integration/          # full pipeline: real SQLite + real FS, FakeEmbedder + FakeSummarizer
```

## Test seams

```python
class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...

class Summarizer(Protocol):
    def summarize(self, unit: SemanticUnit) -> str: ...
```

`FakeEmbedder` returns deterministic unit-length vectors (MD5-seeded). `FakeSummarizer` returns a deterministic string. Both live in a shared `tests/` fixtures module and are used in all integration tests.

## Key constraints

- Every DB connection must call `sqlite_vec.load(conn)` immediately after opening.
- `embed_model` and `embed_dim` in `mcp_rag_meta` are validated at startup; mismatch → abort with a message pointing to `--reindex`.
- `ANTHROPIC_API_KEY` is checked at startup for `mcp-rag index`; abort before any file I/O if absent.
- Progress goes to stderr; stdout is reserved for the MCP stdio protocol.
- All deletes + inserts for a single file are wrapped in one transaction.
- Summarization truncates units > 8,000 estimated tokens (`len(source) // 4`) with a stderr warning.
- Retry policy for Anthropic API: 429/529/5xx → exponential backoff ±20% jitter, 3 retries at [1 s, 4 s, 16 s].
