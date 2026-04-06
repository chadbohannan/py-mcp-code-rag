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
CREATE TABLE metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
    -- keys: schema_version, embed_model, embed_dim
);

-- Named repository registry; one row per git repository root
CREATE TABLE repos (
    id       INTEGER PRIMARY KEY,
    name     TEXT NOT NULL UNIQUE,  -- basename-derived label (disambiguated on collision)
    root     TEXT NOT NULL UNIQUE,  -- absolute git repo root path
    added_at TEXT NOT NULL          -- ISO-8601
);

-- One row per indexed file; basis for fingerprint comparison
CREATE TABLE files (
    id         INTEGER PRIMARY KEY,
    repo_id    INTEGER NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    path       TEXT NOT NULL,       -- relative to repo root
    mtime      REAL NOT NULL,       -- float unix timestamp
    md5        TEXT NOT NULL,
    indexed_at TEXT NOT NULL,       -- ISO-8601
    UNIQUE (repo_id, path)
);

-- One row per semantic unit extracted from a file
CREATE TABLE units (
    id          INTEGER PRIMARY KEY,
    file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    path        TEXT NOT NULL,      -- repo_name/relative/file.py:Class:method
    content     TEXT NOT NULL,      -- original source, returned to MCP callers
    content_md5 TEXT NOT NULL,      -- MD5 of content; used for per-unit change detection
    summary     TEXT NOT NULL,      -- generated summary
    unit_type   TEXT NOT NULL,      -- function | class | method | paragraph | cte | ...
    unit_name   TEXT NOT NULL,      -- identifier; empty string if not applicable
    char_offset INTEGER NOT NULL    -- Unicode character offset in source file, for stable ordering
);

-- Vector index (sqlite-vec virtual table)
-- Dimension must match embed_dim stored in metadata
CREATE VIRTUAL TABLE embeddings USING vec0 (
    unit_id   INTEGER PRIMARY KEY,
    embedding FLOAT[768]
);

CREATE TRIGGER units_delete_cascade
AFTER DELETE ON units FOR EACH ROW
BEGIN
    DELETE FROM embeddings WHERE unit_id = OLD.id;
END;
```

**Table naming** — tables use short unprefixed names (`repos`, `files`, `units`, `metadata`,
`embeddings`). The database file itself provides namespace isolation.

**Qualified path format** — the repo name is prepended to the stored unit path:
`repo_name/relative/file.py:Class:method`. The first `/` in a qualified path marks the
repo/file boundary. Existing GLOB patterns (e.g. `*.py:*`) continue to work because SQLite
`GLOB *` matches `/`.

**Model/dimension coupling** — on startup, `embed_model` and `embed_dim` in `metadata` are
compared against the configured model. A mismatch aborts with:

```
Index was built with nomic-ai/nomic-embed-text-v1.5-Q (dim=768).
Current model is <other-model> (dim=512).
Run: mcp-rag index --reindex <paths...>
```

---

## Indexing Pipeline

### Two-pass architecture

Indexing uses a two-pass approach:

1. **First pass — repo discovery**: `discover_git_repos(path)` finds all git repositories
   accessible from each input path and upserts them into the `repos` table. If the path is
   inside a git repo, that single repo is returned. Otherwise, the directory tree is walked
   (BFS) for `.git` directories.

2. **Second pass — per-repo indexing**: For each discovered repo, the normal incremental
   indexing pipeline runs, scoped to that repo's git root.

**Name derivation** — repo names default to the git root's basename. Collisions (two repos
with the same basename) are resolved by prepending parent directory segments with `-` as
separator (e.g. `a-backend`, `b-backend`).

**Repo description** — read from `.git/description` and returned by `list_repos`.

### File discovery

Within each repo, files are enumerated by shelling out to git:

```
git ls-files --cached --others --exclude-standard -- <root>
```

This delegates gitignore handling, default exclusions, and symlink policy entirely to git.

**Non-git fallback** — if `git ls-files` exits non-zero, fall back to
`pathlib.Path.walk(follow_symlinks=False)` with a hardcoded exclusion set:
`.git/`, `__pycache__/`, `*.pyc`, `.venv/`, `venv/`, `node_modules/`, `dist/`, `build/`.

**Symlinks** — git does not follow directory symlinks; file symlinks are included as regular
files. The non-git fallback uses `follow_symlinks=False` to match this behaviour.

### Deleted-file reconciliation

Immediately after file discovery, before processing any file, the indexer reconciles the
discovered file set against `files`:

1. Query all `path` values in `files` for the current `repo_id`.
2. Compute the set difference: paths in the DB that are absent from the discovered set.
3. `DELETE FROM files WHERE id = ?` for each orphan. `ON DELETE CASCADE`
   propagates the delete to `units` and `embeddings` automatically.
4. Log the count to stderr if any rows were removed:

```
reconciled: removed 3 deleted file(s) from index
```

This runs on every `mcp-rag index` invocation, including incremental runs. No separate
`--prune` flag is needed.

### Changed-file handling

For each discovered file whose `mtime` or `md5` (file-level fingerprint in `files`)
differs from the stored value, the indexer re-parses the file and reconciles at the unit level
to avoid re-summarizing unchanged units (API calls are expensive):

1. Re-parse the file to produce the new set of units.
2. Load existing units for the file from `units` (keyed by `(qualified_path, char_offset)`).
3. For each new unit, compute `content_md5 = md5(content)`.
4. **Unchanged unit** — key matches and `content_md5` is identical: retain the existing row and embedding unchanged.
5. **Changed or new unit** — delete the old row if one matched (cascade removes its embedding), then summarize, embed, and insert a new row.
6. **Removed unit** — existing rows with no corresponding new unit are deleted (cascade removes embeddings).
7. `UPDATE files SET mtime = ?, md5 = ?, indexed_at = ? WHERE id = ?`.

All deletes and inserts for a single file are wrapped in one transaction so a crash mid-file
leaves the DB fully consistent (old state or new state, never a mix).

### Module-level summary units

After all child units in a file are summarized, the indexer synthesizes a **module unit** — a
file-level summary that answers "what does this file do?" with cross-file context from imports.

**When created** — for files with more than one child unit. Files with a single child unit (e.g.
SQL, `.tfvars`) already serve as their own file-level summary.

**Content assembly** — the module unit's content is a structured text block:

```
File: repo/path/to/file.py
Imports: repo/path/to/dep_a.py, repo/path/to/dep_b.py

Units in this file:
- function foo: <summary>
- class Bar: <summary>

Imported module context:
- repo/path/to/dep_a.py: <that file's module summary>
```

**Import extraction** (`mcp_rag/imports.py`) — per-language regex/AST extraction of import
statements, resolved to file paths within the same repo on a best-effort basis. Unresolvable
imports (external packages, ambiguous paths) are silently dropped. Supported: Python
(`import`/`from`), JS/TS (`import`/`require`), C/C++ (`#include "..."`), Java (`import`),
Go (`import`).

**Depth-first processing** — files are processed in topological order (dependency leaves first)
so that when a file is summarized, its imports' module summaries are already available. This
gives the summarizer real cross-file context in a single indexing pass.

**Cycle handling** — circular imports are detected during topological sort. Files in a cycle
fall back to unit-level summaries of their cyclic imports instead of (not-yet-available)
file-level summaries. This is graceful degradation — the summary is slightly less
contextualized but still useful.

**Schema** — module units are stored as regular rows in `units` with `unit_type = "module"`,
`unit_name = ""`, and `char_offset = -1` (sentinel distinguishing them from parsed units).

### Directory-level summary units

After all files in a repository are processed, the indexer synthesizes **directory units** —
summaries for each directory in the repo, built bottom-up from child file and subdirectory
summaries.

**Processing order** — directories are sorted deepest-first so that subdirectory summaries are
available when their parent directory is summarized. The repo root directory is processed last,
producing a repo-level summary that synthesizes all top-level file and subdirectory summaries.

**Content assembly** — each directory unit's content lists its direct child file module summaries
and direct child subdirectory summaries. Directories with no indexed content (e.g. only binary
files or oversized SQL) are skipped.

**Schema** — directory units use `unit_type = "directory"`, `char_offset = -2`, and
`file_id = NULL` (they have no parent file). The `repo_id` column on `units` provides
ownership and enables cascade deletion when a repo is removed.

**Orphan cleanup** — after each indexing run, directory units whose directories no longer
contain any indexed files are deleted.

### Schema v3 changes

Schema version 3 modifies the `units` table:
- Adds `repo_id INTEGER NOT NULL REFERENCES repos(id) ON DELETE CASCADE`
- Makes `file_id` nullable (was `NOT NULL`) to support directory units

Migration from v2 is automatic: the `units` table is recreated with the new schema and
`repo_id` is populated from `files.repo_id` via a join. Embeddings and the cascade trigger
are preserved.

**Reconciliation** — module units are excluded from the normal `diff_units` reconciliation
and handled separately. The module unit's `content_md5` is derived from the assembled content
(child summaries + import context), so it naturally changes when any child summary or import
set changes.

**Summarizer prompt** — when `unit_type == "module"`, the prompt asks for the file's purpose,
key exports, and role relative to its dependencies, rather than the standard unit-level prompt.

### Supported file types

| Extension(s) | Parser | Unit boundaries |
|---|---|---|
| `.py` | stdlib `ast` | module-level, class, function, method |
| `.go` | `go/ast` via subprocess | function, method, struct, interface |
| `.c`, `.h` | tree-sitter (C grammar) | function, struct, enum |
| `.cc`, `.cpp`, `.cxx`, `.hh`, `.hpp`, `.hxx` | tree-sitter (C++ grammar) | function, method, class, struct, enum |
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

**C/C++ parsing** — uses `tree-sitter` with `tree-sitter-c` and `tree-sitter-cpp` Python packages
(in-process, no subprocess). The parser extracts `function_definition`, `struct_specifier`,
`class_specifier`, and `enum_specifier` nodes that have bodies (forward declarations are skipped).
Methods inside `class_specifier` are extracted with `ClassName:method` naming, consistent with the
Go and Python parser conventions. Declarations inside `namespace_definition`,
`template_declaration`, and `linkage_specification` are traversed recursively.

If `tree-sitter-c` or `tree-sitter-cpp` is not installed, the corresponding file extensions are
skipped with a one-time warning:

```
warning: tree-sitter-c not installed — .c/.h files will not be indexed
warning: tree-sitter-cpp not installed — C++ files will not be indexed
```

**C header ambiguity** — `.h` files are parsed with the C grammar. Projects using C++ headers with
a `.h` extension should rename them to `.hpp` or another C++ extension for correct parsing.

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
`embeddings` and stored in `metadata`. No hardcoded model-to-dimension registry is maintained.

Rationale: a probe-embed approach (`len(embedder.embed("probe"))`) works but wastes a forward
pass at startup. Real embedders (fastembed) know their output dimension without running
inference; exposing it as a protocol attribute is cheaper and makes the contract explicit.

### Reindexing (`--reindex`)

Used when switching embedding models. Summaries are model-agnostic text and are preserved;
only the vector table is rebuilt.

**DDL sequence:**

```sql
DROP TABLE IF EXISTS embeddings;
DROP TRIGGER IF EXISTS units_delete_cascade;
CREATE VIRTUAL TABLE embeddings USING vec0 (
    unit_id   INTEGER PRIMARY KEY,
    embedding FLOAT[<new_dim>]
);
CREATE TRIGGER units_delete_cascade
AFTER DELETE ON units FOR EACH ROW
BEGIN
    DELETE FROM embeddings WHERE unit_id = OLD.id;
END;
INSERT OR REPLACE INTO metadata (key, value) VALUES ('embed_model', '<new_model>');
INSERT OR REPLACE INTO metadata (key, value) VALUES ('embed_dim', '<new_dim>');
```

Rationale: `vec0` virtual tables cannot carry `FOREIGN KEY` constraints, so the cascade is
implemented via a trigger. Dropping the virtual table without dropping the trigger leaves a
dangling trigger that references a non-existent table; SQLite will error on the next delete
from `units`. Both must be dropped and recreated together.

**Re-embedding pass** — every row in `units` is re-embedded from its `summary` using the
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
| `globs` | `list[str] \| None` | `None` | SQLite GLOB filters on the qualified path (AND'd together) |

Returns a list of result objects, sorted by `score` descending.  **Does not include
source content** — use `get_unit` to retrieve full code for specific paths.

```json
[{
  "path": "backend/src/auth/jwt.py:validate_token",
  "summary": "Validates a JWT token and returns decoded claims, checking expiry with configurable clock skew.",
  "score": 0.82
}]
```

**Path** — the qualified path (``repo_name/relative/file.py:Class:method``), using ``:`` to
separate file path from unit hierarchy. The repo name is the first path segment.

**Score** — cosine similarity in `[0.0, 1.0]`; higher is better, 1.0 means identical vectors.
Derived from `sqlite-vec`'s cosine distance: `score = 1.0 - (vec_distance_cosine(embedding, ?) / 2.0)`.

**Multi-glob filtering** — when multiple globs are provided, all must match (AND semantics).
Example: `["backend/*", "*.py:*"]` returns only Python units in the backend repo.

**Query pattern** — `vec0` in sqlite-vec 0.1.6 does not expose a `distance` column via the
`MATCH`/`k =` ANN syntax. Use `vec_distance_cosine()` as a scalar function with a plain `LIMIT`
instead:

```sql
SELECT u.path, u.summary,
       vec_distance_cosine(e.embedding, ?) AS dist
FROM embeddings e
JOIN units u ON u.id = e.unit_id
ORDER BY dist ASC
LIMIT ?
```

### `get_unit`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `paths` | `list[str]` | required | Qualified paths to retrieve (exact match) |

Returns the full source content for one or more units by qualified path.  Use after
`search` or `list_units` to read code for specific results.  Unknown paths are silently
skipped.

```json
[{
  "path": "backend/src/auth/jwt.py:validate_token",
  "content": "def validate_token(token: str) -> Claims:\n    ...",
  "summary": "Validates a JWT token and returns decoded claims, checking expiry with configurable clock skew."
}]
```

### `list_units`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `globs` | `list[str] \| None` | `None` | SQLite GLOB filters on qualified path (AND'd) |
| `limit` | `int` | `100` | Max results (capped at 500) |

Returns qualified paths and summaries for structural browsing, ordered alphabetically
by path.  No semantic ranking — use `search` when relevance matters.

```json
[{
  "path": "backend/src/auth/jwt.py:validate_token",
  "summary": "Validates a JWT token and returns decoded claims, checking expiry with configurable clock skew."
}]
```

The qualified path starts with the repo name then the relative file path:
- `["backend/*"]` — all units in the backend repo
- `["*.py:*"]` — all Python units
- `["*:Router:*"]` — all Router members
- `["backend/*", "*.py:*"]` — Python units in backend only

### `list_files`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `globs` | `list[str] \| None` | `None` | SQLite GLOB filters on `repo_name/path` (AND'd) |

Returns the repo name, relative file path, and last-indexed timestamp for every indexed file.

```json
[{
  "repo": "backend",
  "path": "src/auth/jwt.py",
  "indexed_at": "2026-02-23T14:32:00Z"
}]
```

### `list_repos`

No parameters. Returns a list of all indexed repositories.

```json
[{
  "name": "backend",
  "root": "/home/user/workspace/backend",
  "added_at": "2026-02-23T14:32:00Z",
  "description": "Backend API service"
}]
```

The `description` field is read from `.git/description` at query time.

### `index_status`

No parameters. Returns a list of per-repo status objects:

```json
[{
  "repo": "backend",
  "file_count": 142,
  "unit_count": 891,
  "last_indexed_at": "2026-02-23T14:32:00Z"
}]
```

Each field is derived from the DB at query time (not stored in `metadata`):

```sql
SELECT
    r.name,
    COUNT(DISTINCT f.id)  AS file_count,
    COUNT(u.id)           AS unit_count,
    MAX(f.indexed_at)     AS last_indexed_at
FROM repos r
JOIN files f ON f.repo_id = r.id
LEFT JOIN units u ON u.file_id = f.id
GROUP BY r.id
ORDER BY r.name;
```

One object is returned per repo in the `repos` table.

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

## Multi-repository Behaviour

Multiple paths may be passed to `mcp-rag index` or `mcp-rag` (combined mode):

```
mcp-rag index /path/to/workspace
```

A single path may contain multiple git repositories. The first pass discovers all repos
under each input path and upserts them into the `repos` table. Each repo is then indexed
independently.

- **Single DB** — all repos are indexed into the same `--db` file. The `repo_id` foreign key
  in `files` keeps their data separate; reconciliation and change detection are performed
  independently per repo.
- **Default `--db`** — always `./index.db` in the caller's working directory, regardless of how
  many paths are given.
- **Repo naming** — defaults to the git root's basename. Collisions are resolved by prepending
  parent directory segments (e.g. `a-backend`, `b-backend`).
- **Overlapping roots** — if any two resolved absolute input paths have an ancestor/descendant
  relationship, abort immediately with:

  ```
  error: roots overlap: /path/to/project contains /path/to/project/sub
  ```

- **No git repos found** — if a path contains no git repositories, a warning is logged and
  the path is skipped.
- **First-run warning** — emitted once per DB, not per repo:

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
