# Design Specification

Complete implementation specification for `mcp-rag`. See `overview.md` for design rationale.

---

## Architecture

**Stack**

| Component | Technology |
|---|---|
| Runtime | Python ≥ 3.12, managed with [uv](https://github.com/astral-sh/uv) |
| Packaging | `uvx`-runnable; entry point script `mcp-rag` |
| MCP server | `fastmcp`, stdio transport (default); Streamable HTTP optional |
| Embeddings | `fastembed` in-process via ONNX Runtime |
| Default model | `nomic-ai/nomic-embed-text-v1.5-Q` (quantized, 768-dim, ~130 MB) |
| Summarization | Anthropic API (`claude-haiku-4-5-20251001`), index-time only |
| Storage | SQLite (WAL mode) + `sqlite-vec` virtual table, single file per root |

**Key decisions**

- `fastembed` in-process — single process, no IPC, no entrypoint complexity; Ollama only reconsidered if in-process generation becomes a requirement
- `sqlite-vec` — documents, metadata, and embeddings in one file; WAL mode allows concurrent reads during future watch/serve scenarios
- stdio default transport — canonical for local developer MCP tools; clients spawn and own the process
- Explicit `mcp-rag index` over a background daemon — Claude API cost and latency is visible and controlled; serve path is read-only and cheap
- Project-local `.mcp-rag/` — analogous to `.git/`; one index per root, discoverable, gitignore-able

---

## Data Model

All tables live in `.mcp-rag/index.db` (or the path given by `--db`). SQLite WAL mode is
enabled on creation.

```sql
-- Configuration and versioning
CREATE TABLE mcp_rag_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
    -- keys: schema_version, embed_model, embed_dim, last_indexed_at
);

-- One row per indexed file; basis for fingerprint comparison
CREATE TABLE mcp_rag_files (
    id         INTEGER PRIMARY KEY,
    root       TEXT NOT NULL,       -- absolute path of the indexed root
    path       TEXT NOT NULL,       -- absolute path of the file
    mtime      REAL NOT NULL,       -- float unix timestamp
    sha256     TEXT NOT NULL,
    indexed_at TEXT NOT NULL,       -- ISO-8601
    UNIQUE (root, path)
);

-- One row per semantic unit extracted from a file
CREATE TABLE mcp_rag_units (
    id          INTEGER PRIMARY KEY,
    file_id     INTEGER NOT NULL REFERENCES mcp_rag_files(id) ON DELETE CASCADE,
    unit_type   TEXT NOT NULL,      -- function | class | method | paragraph | cte | ...
    unit_name   TEXT,               -- identifier; NULL if not applicable
    content     TEXT NOT NULL,      -- original source, returned to MCP callers
    summary     TEXT NOT NULL,      -- Claude-generated; raw content copy in --no-summarize mode
    char_offset INTEGER NOT NULL    -- byte offset in source file, for stable ordering
);

-- Vector index (sqlite-vec virtual table)
-- Dimension must match embed_dim stored in mcp_rag_meta
CREATE VIRTUAL TABLE mcp_rag_embeddings USING vec0 (
    unit_id   INTEGER PRIMARY KEY,
    embedding FLOAT[768]
);
```

**Model/dimension coupling** — on startup, `embed_model` and `embed_dim` in `mcp_rag_meta` are
compared against the configured model. A mismatch aborts with:

```
Index was built with nomic-ai/nomic-embed-text-v1.5-Q (dim=768).
Current model is <other-model> (dim=512).
Run: mcp-rag index --reindex <paths...>
```

---

## Indexing Pipeline

### Supported file types

| Extension(s) | Parser | Unit boundaries |
|---|---|---|
| `.py` | stdlib `ast` | module-level, class, function, method |
| `.sql` | custom | CTE, view, function, procedure, top-level statement |
| `.md`, `.mdx`, `.rst`, `.txt` | heading/paragraph splitter | heading sections, paragraphs |
| `.js`, `.ts`, `.jsx`, `.tsx` | `tree-sitter` | function, class, arrow function, method |
| `.go`, `.rs`, `.java`, `.c`, `.cpp`, `.h`, `.rb` | `tree-sitter` | function, struct/impl/class, method |
| `.json`, `.yaml`, `.yml`, `.toml` | document-level | whole file (skip if > 4 KB) |
| binary (null bytes in first 512 B) | — | skip silently |
| everything else | — | skip; log extension at DEBUG |

### Summarization

Called once per semantic unit during `mcp-rag index`. Skipped in `--no-summarize` mode, which
embeds the raw parsed text directly instead.

**Model**: `claude-haiku-4-5-20251001` &nbsp;·&nbsp; **Max output tokens**: 256

```
You are indexing a codebase for semantic search. Write a dense, searchable description
of the following {unit_type} from `{relative_path}`.

Describe: what it does, what problem it solves, key inputs/outputs/parameters, and any
important constraints or design patterns. Use natural language that will match developer
questions about this code.

{source_content}
```

### Large unit handling

Token estimate: `len(source) // 4` (no tokenizer dependency).

| Estimated size | Behaviour |
|---|---|
| ≤ 2,000 tokens | Summarize directly |
| > 2,000 tokens | Split at natural sub-boundaries (methods within a class, paragraphs within a section); summarize each sub-unit independently |
| > 8,000 tokens per sub-unit after splitting | Truncate to 8,000; log warning to stderr |

No summary-of-summaries. Hierarchical summarization is future scope.

### Reliability

**Retry** — 429 / 529 / 5xx: exponential backoff ±20% jitter, up to 3 retries at delays
[1 s, 4 s, 16 s]. Other 4xx: fail immediately with a clear message.

**Resume** — indexing is naturally incremental: files with matching mtime + sha256 fingerprints
are skipped entirely. Re-run after a failure to continue from where it stopped.

**Progress** — written to stderr (stdout is reserved for the MCP stdio protocol):
```
[42/150] src/auth/jwt.py
```

---

## MCP Tools

The server holds a single read-only SQLite connection for the lifetime of the session.

### `search`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | required | Natural language question about the codebase |
| `top_k` | `int` | `5` | Results to return (max 20) |

Returns a list of result objects:

```json
[{
  "path": "src/auth/jwt.py",
  "unit_type": "function",
  "unit_name": "validate_token",
  "content": "def validate_token(token: str) -> Claims:\n    ...",
  "summary": "Validates a JWT token and returns its decoded claims. Checks expiry against a configurable clock skew. Raises AuthError on invalid or expired tokens.",
  "score": 0.12
}]
```

### `index_status`

No parameters. Returns a list of per-root status objects:

```json
[{
  "root": "/home/user/myproject",
  "file_count": 142,
  "unit_count": 891,
  "last_indexed_at": "2026-02-23T14:32:00Z"
}]
```

---

## CLI

```
mcp-rag [paths...]                    Index if absent, then serve (stdio)
mcp-rag index [paths...] [options]    Build or update the index
mcp-rag serve [paths...] [options]    Start the MCP server
```

**`index` options**

| Flag | Default | Description |
|---|---|---|
| `--reindex` | off | Wipe and rebuild the entire index |
| `--no-summarize` | off | Skip Claude summarization; embed raw parsed text |
| `--embed-model MODEL` | `nomic-ai/nomic-embed-text-v1.5-Q` | Embedding model |
| `--db PATH` | `{root}/.mcp-rag/index.db` | Override index file location |

**`serve` options**

| Flag | Default | Description |
|---|---|---|
| `--http` | off | Streamable HTTP instead of stdio; binds `127.0.0.1` only |
| `--port N` | `8000` | Port for `--http` mode |
| `--db PATH` | `{root}/.mcp-rag/index.db` | Override index file location |
| `--reindex` | off | Re-index before serving |

**Environment variables**

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Required for `mcp-rag index` in semantic mode |
| `EMBED_MODEL` | Default embedding model; overridden by `--embed-model` |

---

## First-run Behaviour

On the first `mcp-rag index` run for a given root:

- `.gitignore` exists and does not contain `.mcp-rag` → append `.mcp-rag/` and print `"Added .mcp-rag/ to .gitignore"`
- No `.gitignore` → print `"Created .mcp-rag/ — add it to .gitignore to avoid committing the index and model cache (~130 MB)"`

`.mcp-rag/` contains:
- `index.db` — the SQLite database (documents, summaries, embeddings)
- `models/` — fastembed model cache (controlled by `FASTEMBED_CACHE_PATH`)

---

## Testing

Two tiers. The full run requires no external API calls or running services.

### Tiers

| Tier | Path | Scope |
|---|---|---|
| Unit | `tests/unit/` | Pure logic, no I/O: parsing, fingerprinting, reconciliation diff |
| Integration | `tests/integration/` | Full pipeline with real SQLite + real filesystem: indexer loop, sqlite-vec queries, MCP tool handlers |

### Test seams

Two protocols isolate the external dependencies:

```python
class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...

class Summarizer(Protocol):
    def summarize(self, unit: SemanticUnit) -> str: ...
```

`FakeEmbedder` returns deterministic unit-length vectors (MD5-seeded) of the correct dimension.
`FakeSummarizer` returns a deterministic string derived from the input unit.

Both fakes enable integration tests that exercise the full index pipeline — real SQLite, real
filesystem, real sqlite-vec queries — with no API calls.
