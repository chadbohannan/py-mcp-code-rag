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
| Storage | SQLite (WAL mode) + `sqlite-vec` virtual table, single `index.db` file |
| Go parsing | stdlib `go/ast` via bundled Go helper script; requires `go` in `PATH` |

**Key decisions**

- `fastembed` in-process — single process, no IPC, no entrypoint complexity; Ollama only reconsidered if in-process generation becomes a requirement
- `sqlite-vec` — documents, metadata, and embeddings in one file; WAL mode allows concurrent reads during future watch/serve scenarios
- stdio default transport — canonical for local developer MCP tools; clients spawn and own the process
- Explicit `mcp-rag index` over a background daemon — Claude API cost and latency is visible and controlled; serve path is read-only and cheap
- `go/ast` via `go run` over tree-sitter — zero Python packaging complexity; target audience (Go developers) has `go` in `PATH` by definition; graceful skip if absent

---

## Data Model

All tables live in `index.db` in the working directory (or the path given by `--db`). SQLite WAL
mode is enabled on creation.

**`sqlite-vec` loading** — the `sqlite-vec` Python package (minimum version `0.1.0`) bundles the
native extension and exposes a `load` helper. Every DB connection must call it immediately after
opening:

```python
import sqlite_vec
sqlite_vec.load(conn)
```

This handles `enable_load_extension` internally. If loading fails (extension not found, version
too old, or platform incompatibility), abort immediately with:

```
error: failed to load sqlite-vec extension: <reason>
Ensure sqlite-vec >= 0.1.0 is installed: uv add sqlite-vec
```

```sql
-- Configuration and versioning
CREATE TABLE mcp_rag_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
    -- keys: schema_version, embed_model, embed_dim
);

-- One row per indexed file; basis for fingerprint comparison
CREATE TABLE mcp_rag_files (
    id         INTEGER PRIMARY KEY,
    root       TEXT NOT NULL,       -- absolute path of the indexed root
    path       TEXT NOT NULL,       -- absolute path of the file
    mtime      REAL NOT NULL,       -- float unix timestamp
    md5        TEXT NOT NULL,
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
    content_md5 TEXT NOT NULL,      -- MD5 of content; used for per-unit change detection
    summary     TEXT NOT NULL,      -- Claude-generated summary
    char_offset INTEGER NOT NULL    -- Unicode character offset in source file, for stable ordering
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

### File discovery

Files are enumerated by shelling out to git:

```
git ls-files --cached --others --exclude-standard -- <root>
```

This delegates gitignore handling, default exclusions, and symlink policy entirely to git.

**Non-git roots** — if `root` is not inside a git repository (`git rev-parse --show-toplevel`
fails), fall back to `pathlib.Path.walk(follow_symlinks=False)` with a hardcoded exclusion set:
`.git/`, `__pycache__/`, `*.pyc`, `.venv/`, `venv/`, `node_modules/`, `dist/`, `build/`.

**Symlinks** — git does not follow directory symlinks; file symlinks are included as regular
files. The non-git fallback uses `follow_symlinks=False` to match this behaviour.

**Subprocess failure** — if `git ls-files` exits non-zero (e.g. the root is not inside any git
repository), fall back to the `pathlib.Path.walk` path described above rather than aborting.
Rationale: aborting would break the common case of indexing a plain directory that has never
been `git init`-ed; silent fallback is preferable to a hard failure here.

### Deleted-file reconciliation

Immediately after file discovery, before processing any file, the indexer reconciles the
discovered file set against `mcp_rag_files`:

1. Query all `path` values in `mcp_rag_files` for the current `root`.
2. Compute the set difference: paths in the DB that are absent from the discovered set.
3. `DELETE FROM mcp_rag_files WHERE root = ? AND path = ?` for each orphan. `ON DELETE CASCADE`
   propagates the delete to `mcp_rag_units` and `mcp_rag_embeddings` automatically.
4. Log the count to stderr if any rows were removed:

```
reconciled: removed 3 deleted file(s) from index
```

This runs on every `mcp-rag index` invocation, including incremental runs. No separate
`--prune` flag is needed.

### Changed-file handling

For each discovered file whose `mtime` or `md5` (file-level fingerprint in `mcp_rag_files`)
differs from the stored value, the indexer re-parses the file and reconciles at the unit level
to avoid re-summarizing unchanged units (Claude API calls are expensive):

1. Re-parse the file to produce the new set of units.
2. Load existing units for the file from `mcp_rag_units` (keyed by `(unit_type, unit_name, char_offset)`).
3. For each new unit, compute `content_md5 = md5(content)`.
4. **Unchanged unit** — `(unit_type, unit_name, char_offset)` matches an existing row and `content_md5` is identical: retain the existing row and embedding unchanged.
5. **Changed or new unit** — delete the old row if one matched (cascade removes its embedding), then summarize, embed, and insert a new row.
6. **Removed unit** — existing rows with no corresponding new unit are deleted (cascade removes embeddings).
7. `UPDATE mcp_rag_files SET mtime = ?, md5 = ?, indexed_at = ? WHERE id = ?`.

All deletes and inserts for a single file are wrapped in one transaction so a crash mid-file
leaves the DB fully consistent (old state or new state, never a mix).

### Supported file types

| Extension(s) | Parser | Unit boundaries |
|---|---|---|
| `.py` | stdlib `ast` | module-level, class, function, method |
| `.go` | `go/ast` via subprocess | function, method, struct, interface |
| `.md`, `.mdx` | heading/paragraph splitter | heading sections, paragraphs |
| `.sql` | document-level | whole file (skip if > 4 KB) |
| binary (null bytes in first 512 B) | — | skip silently |
| everything else | — | skip; log extension at DEBUG |

**Go parsing** — a small Go helper script is bundled with the package under `mcp_rag/go_parser/main.go`.
The indexer invokes it with `go run -- <file>` per file (the `--` separates the helper source file
from the target `.go` file so `go run` does not treat the target as an additional source file to
compile); the helper parses it with `go/ast` and writes a JSON array of units to stdout.
No compiled binary is distributed. If `go` is not found in `PATH`, `.go` files are skipped
with a one-time warning:

```
warning: 'go' not found in PATH — .go files will not be indexed
```

**Go helper JSON schema** — each element of the stdout array:

```json
{
  "unit_type":   "function",
  "unit_name":   "HandleRequest",
  "content":     "func HandleRequest(...) { ... }",
  "char_offset": 1024
}
```

| Field | Type | Nullable | Notes |
|---|---|---|---|
| `unit_type` | string | no | `function`, `method`, `struct`, or `interface` |
| `unit_name` | string | yes | `null` for anonymous constructs |
| `content` | string | no | Full source text of the unit |
| `char_offset` | integer | no | Byte offset in the source file (`token.Pos`); Python stores it as-is |

**Error handling** — the helper writes error messages to stderr and exits non-zero on any failure
(file not found, permission error). A syntax error in the `.go` file is treated as a parse
failure: the helper exits non-zero, Python logs a warning and skips the file:

```
warning: skipping src/foo.go — go parser error (see stderr)
```

**Performance** — `go run` compiles the helper on each invocation, adding approximately 200–500 ms
per `.go` file. This is acceptable because indexing is incremental (unchanged files are skipped)
and Go developers are the target audience.

### Summarization

Called once per semantic unit during `mcp-rag index`.

**Model**: `claude-haiku-4-5-20251001` &nbsp;·&nbsp; **Max output tokens**: 256

```
You are indexing a codebase for semantic search. Write a dense, searchable description
of the following {unit_type} named `{unit_name}`.

Describe: what it does, what problem it solves, key inputs/outputs/parameters, and any
important constraints or design patterns. Use natural language that will match developer
questions about this code.

{source_content}
```

Rationale: `SemanticUnit` carries no file path (path lives in `mcp_rag_files`), so
`{relative_path}` cannot be included at summarization time without changing the protocol.
`{unit_name}` provides equivalent context for named units; anonymous units (e.g. SQL) omit
the name clause entirely.

### Large unit handling

Because Claude summarizes each unit into a compact output (≤ 256 tokens), the size of the raw
source does not affect embedding quality or retrieval. Sub-boundary splitting is not performed.

Units are sent to Claude as-is, with one safety truncation to control cost:

| Estimated size | Behaviour |
|---|---|
| ≤ 8,000 tokens | Summarize directly |
| > 8,000 tokens | Truncate to 8,000; log warning to stderr |

Token estimate: `len(source) // 4` (no tokenizer dependency).

### Embedding dimension

The `Embedder` protocol exposes `dim: int` directly:

```python
class Embedder(Protocol):
    dim: int
    model: str
    def embed(self, text: str) -> list[float]: ...
```

`embed_dim` is read from `embedder.dim` at startup and used in the DDL when creating
`mcp_rag_embeddings` and stored in `mcp_rag_meta`. No hardcoded model-to-dimension registry
is maintained.

Rationale: a probe-embed approach (`len(embedder.embed("probe"))`) works but wastes a forward
pass at startup. Real embedders (fastembed) know their output dimension without running
inference; exposing it as a protocol attribute is cheaper and makes the contract explicit.

### Reindexing (`--reindex`)

Used when switching embedding models. Summaries are model-agnostic text and are preserved;
only the vector table is rebuilt.

**DDL sequence:**

```sql
DROP TABLE IF EXISTS mcp_rag_embeddings;
DROP TRIGGER IF EXISTS mcp_rag_units_delete_cascade;
CREATE VIRTUAL TABLE mcp_rag_embeddings USING vec0 (
    unit_id   INTEGER PRIMARY KEY,
    embedding FLOAT[<new_dim>]
);
CREATE TRIGGER mcp_rag_units_delete_cascade
AFTER DELETE ON mcp_rag_units FOR EACH ROW
BEGIN
    DELETE FROM mcp_rag_embeddings WHERE unit_id = OLD.id;
END;
UPDATE mcp_rag_meta SET value = '<new_model>' WHERE key = 'embed_model';
UPDATE mcp_rag_meta SET value = '<new_dim>'   WHERE key = 'embed_dim';
```

Rationale: `vec0` virtual tables cannot carry `FOREIGN KEY` constraints, so the cascade is
implemented via a trigger. Dropping the virtual table without dropping the trigger leaves a
dangling trigger that references a non-existent table; SQLite will error on the next delete
from `mcp_rag_units`. Both must be dropped and recreated together.

**Re-embedding pass** — every row in `mcp_rag_units` is re-embedded from its `summary` using the
new model. No API calls are made. Progress is reported to stderr:

```
re-embedding: [891/891] units
```

**Incremental pass** — the normal indexing pipeline then runs. Files with matching mtime + md5
are skipped entirely (existing summaries are reused). Only changed or new files trigger
summarization API calls.

### Reliability

**Retry** — 429 / 529 / 5xx: exponential backoff ±20% jitter, up to 3 retries at delays
[1 s, 4 s, 16 s]. Other 4xx: fail immediately with a clear message.

**Resume** — indexing is naturally incremental: files with matching mtime + md5 fingerprints
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

Returns a list of result objects, sorted by `score` descending:

```json
[{
  "path": "/home/user/myproject/src/auth/jwt.py",
  "unit_type": "function",
  "unit_name": "validate_token",
  "content": "def validate_token(token: str) -> Claims:\n    ...",
  "summary": "Validates a JWT token and returns its decoded claims. Checks expiry against a configurable clock skew. Raises AuthError on invalid or expired tokens.",
  "score": 0.82
}]
```

**Path** — the absolute path from `mcp_rag_files.path`, returned as-is with no relativization.

**Score** — cosine similarity in `[0.0, 1.0]`; higher is better, 1.0 means identical vectors.
Derived from `sqlite-vec`'s cosine distance: `score = 1.0 - (vec_distance_cosine(embedding, ?) / 2.0)`.
Embeddings are stored as unit-length vectors (the default output of `nomic-embed-text`), so cosine
distance is in `[0.0, 2.0]` and the mapping is exact with no clipping required.

**Query pattern** — `vec0` in sqlite-vec 0.1.6 does not expose a `distance` column via the
`MATCH`/`k =` ANN syntax. Use `vec_distance_cosine()` as a scalar function with a plain `LIMIT`
instead:

```sql
SELECT f.path, u.unit_type, u.unit_name, u.content, u.summary,
       vec_distance_cosine(e.embedding, ?) AS dist
FROM mcp_rag_embeddings e
JOIN mcp_rag_units u ON u.id = e.unit_id
JOIN mcp_rag_files f ON f.id = u.file_id
ORDER BY dist ASC
LIMIT ?
```

Rationale: the `MATCH`/`k` ANN path only exposes the primary key and the raw embedding bytes,
not the computed distance. The scalar-function approach is a full scan but is sufficient for
typical index sizes; revisit if performance becomes a concern.

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

Each field is derived from the DB at query time (not stored in `mcp_rag_meta`):

```sql
SELECT
    f.root,
    COUNT(DISTINCT f.id)  AS file_count,
    COUNT(u.id)           AS unit_count,
    MAX(f.indexed_at)     AS last_indexed_at
FROM mcp_rag_files f
LEFT JOIN mcp_rag_units u ON u.file_id = f.id
GROUP BY f.root;
```

One object is returned per distinct `root` present in `mcp_rag_files`.

---

## CLI

```
mcp-rag [paths...]                    Index if absent, then serve (stdio)
mcp-rag index [paths...] [options]    Build or update the index
mcp-rag serve [options]               Start the MCP server
```

**`index` options**

| Flag | Default | Description |
|---|---|---|
| `--reindex` | off | Rebuild the vector table; preserve summaries for unchanged files |
| `--embed-model MODEL` | `nomic-ai/nomic-embed-text-v1.5-Q` | Embedding model |
| `--db PATH` | `./index.db` | Override index file location |

**`serve` options**

| Flag | Default | Description |
|---|---|---|
| `--http` | off | Streamable HTTP instead of stdio; binds `127.0.0.1` only |
| `--port N` | `8000` | Port for `--http` mode |
| `--db PATH` | `./index.db` | Override index file location |

**Environment variables**

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Required for `mcp-rag index`; checked at startup before any file I/O |
| `EMBED_MODEL` | Default embedding model; overridden by `--embed-model` |

`mcp-rag index` aborts immediately on startup if `ANTHROPIC_API_KEY` is not set:

```
error: ANTHROPIC_API_KEY is not set. An API key is required for indexing.
```

Summarization uses the Anthropic Python SDK directly (`anthropic` package); no CLI binary is
required or used.

---

## Multi-root Behaviour

Multiple paths may be passed to `mcp-rag index` or `mcp-rag` (combined mode):

```
mcp-rag index /path/to/backend /path/to/frontend
```

- **Single DB** — all roots are indexed into the same `--db` file. The `root` column in
  `mcp_rag_files` keeps their data separate; reconciliation and change detection are performed
  independently per root.
- **Default `--db`** — always `./index.db` in the caller's working directory, regardless of how
  many paths are given.
- **Overlapping roots** — if any two resolved absolute paths have an ancestor/descendant
  relationship, abort immediately with:

  ```
  error: roots overlap: /path/to/project contains /path/to/project/sub
  ```

- **First-run warning** — emitted once per DB, not per root:

  ```
  No index found at ./index.db — initializing a new database.
  ```

---

## Combined-mode Behaviour

`mcp-rag [paths...]` runs `index` then `serve` in one invocation. The index step is only
triggered if the DB file does not exist at the resolved `--db` path. If the DB file already
exists (even if empty or partially indexed), the index step is skipped and `serve` starts
immediately.

## First-run Behaviour

If no database is found at the resolved `--db` path, the tool warns before creating it:

```
No index found at ./index.db — initializing a new database.
```

The fastembed model cache location is controlled by `FASTEMBED_CACHE_PATH` (defaults to the
fastembed library default, typically `~/.cache/fastembed`).

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
