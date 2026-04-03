# mcp-rag

Semantic code search over local codebases, exposed as an [MCP](https://modelcontextprotocol.io/) stdio server. Designed for navigating complex, sprawling codebases — surfacing architectural intent rather than matching surface text.

## How it works

Raw source code embeds poorly against natural language queries. A developer asking "how does authentication work?" shares almost no embedding space with the code that implements it.

**mcp-rag** uses Semantic Surrogate Indexing: files are parsed into language-aware units (functions, classes, methods, SQL, markdown sections), each unit is summarized by Claude, and the *summary* — not the raw source — is embedded. The raw source is stored alongside and returned on a match.

```
source file → semantic parser → semantic units
                                      │
                              semantic summary
                              (file path + unit type as context)
                                      │
                              fastembed → vector index
                              raw source stored alongside, returned on match
```

This means queries like "how does token expiry work?" match a summary like *"Validates a JWT and checks clock skew against a configurable tolerance"* — something raw source embedding cannot achieve.

## Requirements

- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) package manager
- `go` in `PATH` (for indexing `.go` files)
- [Ollama](https://ollama.com/) 
- gemma4:latest

## Quickstart

```bash
# Install dependencies
make install

# Index a codebase
make index SRC=../my-project DB=./index.db

# Start the MCP server
make serve DB=./index.db
```

## Installation

```bash
git clone <repo-url>
cd py-mcp-code-rag
make install
```

## Usage

### CLI

```
mcp-rag index [paths...] [options]    Build or update the index
mcp-rag serve [options]               Start the MCP server (stdio)
```

### Indexing

Index one or more directories. Incremental by default — only changed files are re-processed.

```bash
# Index the current directory
mcp-rag index .

# Index specific directories into a single DB
mcp-rag index /path/to/backend /path/to/frontend

# Use a custom database path
mcp-rag index --db myproject.db ../my-project

# Rebuild embeddings after changing the embed model (preserves summaries)
mcp-rag index --reindex .

# Use Ollama for summarization instead of Anthropic API
mcp-rag index --summarizer ollama --ollama-model gemma3 .
```

**Index options:**

| Flag | Default | Description |
|---|---|---|
| `--reindex` | off | Rebuild vector table; preserves summaries for unchanged units |
| `--embed-model MODEL` | `nomic-ai/nomic-embed-text-v1.5-Q` | Embedding model (fastembed) |
| `--db PATH` | `./index.db` | Index file location |
| `--summarizer {anthropic,ollama}` | `ollama` | Summarization backend |
| `--ollama-model MODEL` | — | Ollama model name |
| `--ollama-host HOST` | — | Ollama API host |

### Serving

Start the MCP stdio server. Read-only — no API key required.

```bash
mcp-rag serve --db index.db
```

**Serve options:**

| Flag | Default | Description |
|---|---|---|
| `--db PATH` | `./index.db` | Index file location |

### Claude Code integration

Register mcp-rag as an MCP server in Claude Code:

```bash
make add-claude-mcp
```

This lets Claude Code search your indexed codebase directly. To unregister:

```bash
make remove-claude-mcp
```

## Make targets

All targets that operate on a source directory accept `SRC=` (defaults to `.`). Targets that use the index database accept `DB=` (defaults to `index.db`).

| Target | Description | Example |
|---|---|---|
| `install` | Install all dependencies | `make install` |
| `index` | Index a directory (incremental) | `make index SRC=../repo DB=my.db` |
| `reindex` | Rebuild embeddings from scratch | `make reindex SRC=../repo` |
| `serve` | Start MCP stdio server | `make serve DB=my.db` |
| `test` | Run full test suite | `make test` |
| `test-unit` | Run unit tests only | `make test-unit` |
| `test-integration` | Run integration tests | `make test-integration` |
| `lint` | Check code style | `make lint` |
| `format` | Auto-format code | `make format` |
| `add-claude-mcp` | Register with Claude Code | `make add-claude-mcp` |
| `remove-claude-mcp` | Unregister from Claude Code | `make remove-claude-mcp` |
| `clean` | Remove index.db and WAL files | `make clean` |

## Supported file types

| Extension | Parser | Unit boundaries |
|---|---|---|
| `.py` | stdlib `ast` | module-level, class, function, method |
| `.go` | `go/ast` via subprocess | function, method, struct, interface |
| `.md`, `.mdx` | heading splitter | heading sections, paragraphs |
| `.sql` | document-level | whole file (skipped if > 4 KB) |

Binary files are detected and skipped automatically. Unrecognized extensions are skipped silently.

## MCP tools

### `search`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | required | Natural language question about the codebase |
| `top_k` | `int` | `5` | Number of results (max 20) |

Returns matching source units ranked by cosine similarity, each with file path, unit type/name, original source content, summary, and relevance score.

### `index_status`

No parameters. Returns per-root statistics: file count, unit count, and last indexed timestamp.

## Architecture

- **Embeddings**: [fastembed](https://github.com/qdrant/fastembed) in-process via ONNX Runtime (`nomic-ai/nomic-embed-text-v1.5-Q`, 768-dim)
- **Storage**: SQLite (WAL mode) + [sqlite-vec](https://github.com/asg017/sqlite-vec) — documents, metadata, and vectors in a single file
- **Summarization**: Anthropic API (Claude Haiku) or Ollama, index-time only
- **MCP transport**: stdio (default)

## License

See [LICENSE](LICENSE) for details.
