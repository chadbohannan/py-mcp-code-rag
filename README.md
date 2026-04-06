# code-rag

Semantic code search over local codebases, exposed as an [MCP](https://modelcontextprotocol.io/) stdio server. Designed for navigating complex, sprawling codebases — surfacing architectural intent rather than matching surface text.

## How it works

Raw source code embeds poorly against natural language queries. A developer asking "how does authentication work?" shares almost no embedding space with the code that implements it.

**code-rag** uses Semantic Surrogate Indexing: files are parsed into language-aware units (functions, classes, methods, SQL, markdown sections), each unit is summarized by Claude, and the *summary* — not the raw source — is embedded. The raw source is stored alongside and returned on a match.

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
- [Ollama](https://ollama.com/) with gemma4:e2b

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
code-rag index [paths...] [options]    Build or update the index
code-rag serve [options]               Start the MCP server (stdio)
```

### Indexing

Index one or more directories. Incremental by default — only changed files are re-processed.

```bash
# Index the current directory
code-rag index .

# Index specific directories into a single DB
code-rag index /path/to/backend /path/to/frontend

# Use a custom database path
code-rag index --db myproject.db ../my-project

# Rebuild embeddings after changing the embed model (preserves summaries)
code-rag index --reindex .

# Use Ollama for summarization instead of Anthropic API
code-rag index --summarizer ollama --ollama-model gemma3 .
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
code-rag serve --db index.db
```

**Serve options:**

| Flag | Default | Description |
|---|---|---|
| `--db PATH` | `./index.db` | Index file location |

### Claude Code integration

Register code-rag as an MCP server in Claude Code:

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
| `.py` | stdlib `ast` | module-level function, class, method |
| `.go` | tree-sitter (Go) | function, method, struct, interface |
| `.c`, `.h` | tree-sitter (C) | function, struct, enum |
| `.cc`, `.cpp`, `.cxx`, `.hh`, `.hpp`, `.hxx`, `.ino` | tree-sitter (C++) | function, method, class, struct, enum |
| `.js`, `.jsx`, `.mjs`, `.cjs` | tree-sitter (JavaScript) | function, class, method, arrow function |
| `.ts`, `.tsx`, `.mts`, `.cts` | tree-sitter (TypeScript) | function, class, method, interface, type, enum |
| `.java` | tree-sitter (Java) | class, interface, enum, method, constructor |
| `.tf` | HCL block splitter | resource, variable, output, module, data, locals, … |
| `.tfvars` | document-level | whole file |
| `.md`, `.mdx` | heading splitter | heading sections |
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
