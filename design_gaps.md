# Design Gaps

Analysis of open questions and missing specifications in `design_spec.md` and `overview.md`.

---

## Critical (would cause bugs or broken functionality)

### 1. No file-walker specification

The spec defines *what* file types to parse but never specifies *how files are discovered*. Missing:

- Gitignore-aware walking (without it, `node_modules/`, `dist/`, `.venv/` get indexed — potentially millions of files)
- Symlink handling (follow? skip? guard against cycles?)
- Default exclusion list (`.git/`, `__pycache__/`, `*.pyc`, etc.)

### 2. No deleted-file reconciliation

When a file is deleted from the codebase, its row in `mcp_rag_files` persists. The `ON DELETE CASCADE` handles cleanup of units/embeddings, but the spec never describes a reconciliation step that detects and removes orphaned file rows. Incremental indexing will silently return stale results for deleted files.

### 3. `char_offset` naming inconsistency

The column comment says `-- byte offset in source file` but the column is named `char_offset`. For non-ASCII files (Python with Unicode identifiers, SQL with UTF-8 comments), these diverge. The spec never resolves which it actually is.

### 4. Multi-path / multi-root DB relationship unspecified

`mcp-rag index /path/a /path/b` — is that one DB or two? If one, which default `--db` path applies? If two, what does `--db PATH` mean when there are multiple roots? The spec is silent. `index_status` implies a single server can cover multiple roots, but serving multiple separate `index.db` files is never spelled out.

### 5. `--reindex` schema migration undefined

`--reindex` "wipes and rebuilds" — but the embedding virtual table `mcp_rag_embeddings USING vec0(embedding FLOAT[768])` has a hardcoded dimension. Switching models (e.g., to a 512-dim model) requires dropping and recreating the virtual table with a new dimension. The spec mentions the mismatch error and `--reindex` as the remedy, but never specifies the DDL migration steps.

### 6. Tree-sitter grammar distribution is a packaging blocker

Tree-sitter parsers require per-language grammar packages (`tree-sitter-python`, `tree-sitter-javascript`, etc.). The spec lists 10+ supported languages but never addresses whether grammars are bundled dependencies, optional extras, or downloaded at runtime. This is a hard packaging decision that affects `pyproject.toml`, install size, and first-run behavior.

### 7. Score semantics undefined

The example shows `"score": 0.12`. Is this cosine distance (lower = more similar), cosine similarity (higher = better), or L2 distance? `sqlite-vec` supports multiple distance functions. Callers need to know whether to rank ascending or descending, and what a "good" score looks like.

---

## Significant (missing important behaviors)

### 8. No concurrent-indexer protection

Two simultaneous `mcp-rag index` processes on the same root will produce SQLite write conflicts (WAL mode allows only one writer). No advisory lock, PID file, or detection mechanism is specified.

### 9. Summarization concurrency not specified

For 1,000 files × ~10 units each = 10,000 API calls. At ~1 s/call serial, that is ~3 hours. The spec doesn't state whether calls are serial or concurrent, and if concurrent, how concurrency is bounded and how the 429 backoff interacts with a worker pool.

### 10. Serve mode has no staleness detection

The server opens the DB at startup and holds it for the session lifetime. If files change and the user forgets to re-index, there is no warning in search results or in `index_status` that the index is stale. `last_indexed_at` is available but no freshness check is triggered.

### 11. No `--exclude` patterns

There is no way to exclude specific directories or file patterns from indexing (e.g., `--exclude vendor/ --exclude '*.generated.py'`). For monorepos or projects with auto-generated code, this is essential.

### 12. Model download progress and failure handling unspecified

The spec specifies progress output for file indexing (`[42/150] src/auth/jwt.py`) but the first-run fastembed model download (~130 MB) has no progress output or failure/retry behavior specified.

### 13. No persistent per-project configuration

All options (model, exclusions, summarization) require CLI flags on every run. There is no `.mcp-rag/config.toml` or `mcp-rag.toml`, so settings are not reproducible without a wrapper script.

---

## Minor (edge cases and polish)

### 14. No `CREATE INDEX` statements in schema

The data model shows table DDL but no secondary indexes. `mcp_rag_units.file_id` (used in every cascade and join) has no index. For large codebases, lookups by `file_id` will full-scan the units table.

### 15. `--no-summarize` mode produces confusing results

In `--no-summarize` mode, `summary = content`. Search results return both fields; callers receiving identical `content` and `summary` may not understand the mode difference. The spec should clarify the field semantics or consider omitting `summary` from results in this mode.

### 16. No search filtering by path or unit type

`search` accepts only `query` and `top_k`. There is no way to scope results to a subdirectory (`src/auth/`) or unit type (`function`). Useful for focused queries in large codebases.

### 17. No line numbers in search results

Results return `char_offset` (or byte offset — see gap 3) but not line numbers. Developers navigating to a match in an editor work with line numbers, not byte offsets. Line numbers should be stored or derived.

### 18. No `get_file` MCP tool

When `search` returns a matching unit (e.g., one method of a class), there is no tool to retrieve the surrounding context — the rest of the class, or the full file. AI assistants frequently need this escalation path.

### 19. HTTP mode has no authentication

`--http` binds to `127.0.0.1` only, which is good. But on shared machines, any local user can query the server. No token-based auth or even a flag to disable the HTTP mode warning is specified.

### 20. `ANTHROPIC_API_KEY` validation UX

A `401 Unauthorized` from the API will "fail immediately with a clear message," but the message content is not specified. Users with expired or wrong-scope keys need actionable guidance, not a raw HTTP status.

---

## Summary

| Category | Count | Representative items |
|---|---|---|
| Critical (blocking bugs) | 7 | File walker, deleted-file reconciliation, schema migration, tree-sitter packaging |
| Significant (missing behaviors) | 6 | Summarization concurrency, staleness detection, `--exclude` patterns |
| Minor (polish/UX) | 7 | Line numbers, search filtering, `get_file` tool |

The two highest-priority gaps to resolve before implementation begins are **file walker specification** (gitignore-awareness determines index quality for virtually every real project) and **tree-sitter grammar distribution** (determines the entire packaging story for multi-language support).
