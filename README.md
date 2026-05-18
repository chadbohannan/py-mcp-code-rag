# code-rag

Semantic code search over local codebases, exposed as a REST API and an [MCP](https://modelcontextprotocol.io/) server. Designed for navigating complex, sprawling codebases — surfacing architectural intent rather than matching surface text.

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

# Start the REST API + web UI (primary interface)
make webui DB=./index.db

# In another terminal, start the MCP server pointing at the web UI
make mcp
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
code-rag webui [options]               Start the REST API + web UI
code-rag-cli.py SUBCOMMAND [args]      Stdlib CLI client to a running web UI
code-rag-mcp.py [options]              MCP server proxying tools to the web UI
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

### Web UI (REST API)

The primary interface. Serves the REST API (documented in `SKILL.md`) and the browser UI.

```bash
code-rag webui --db index.db --port 8081
```

**Webui options:**

| Flag | Default | Description |
|---|---|---|
| `--db PATH` | `./index.db` | Index file location |
| `--host HOST` | `0.0.0.0` | Bind address |
| `--port N` | `8080` | Listen port |
| `--embed-model MODEL` | (from DB) | Override the embedding model |
| `--summarizer {anthropic,ollama}` | `ollama` | Summarization backend used for reindex jobs triggered via the web UI |

### MCP server

A standalone script that proxies MCP tool calls to a running web UI over REST. Stdio by default; pass `--http` for streamable-HTTP transport.

```bash
# Stdio MCP server (default — for direct agent stdin/stdout integration)
python code-rag-mcp.py --base-url http://localhost:8081

# Or via the env var
CODE_RAG_URL=http://localhost:8081 python code-rag-mcp.py
```

**MCP server options:**

| Flag | Default | Description |
|---|---|---|
| `--base-url URL` | `http://localhost:8081` | Web UI to talk to (env: `CODE_RAG_URL`) |
| `--http` | off | Run as streamable-HTTP MCP server on `127.0.0.1` |
| `--port N` | `8000` | Port for `--http` mode |

### Standalone CLI client

For shell scripts and human use, `code-rag-cli.py` wraps the REST API in plain-text output:

```bash
python code-rag-cli.py search "how does authentication work?"
python code-rag-cli.py --json repos | jq '.[].name'
python code-rag-cli.py index /path/to/repo --wait
```

Subcommands: `search`, `unit`, `fetch`, `units`, `files`, `repos`, `status`, `browse`, `index`, `index-status`, `index-cancel`, `clear-repo`, `staleness`, `ls`. The top-level `--json` flag emits raw JSON from any subcommand.

### Agent integration

Register `code-rag-mcp.py` with your agent of choice. Both commands accept `BASE_URL=` to point at a non-default web UI.

| Agent | Register | Unregister |
|---|---|---|
| Claude Code | `make add-claude-mcp` | `make remove-claude-mcp` |
| pi-agent | `make add-pi-mcp` | `make remove-pi-mcp` |

The web UI must be running for the MCP server to work — start it once on the host that owns the index, then point agents at it.

#### System prompt

For best results, include the following in your agent's system prompt or persona config:

> code-rag is a RAG server for an index of code repositories indexed to accelerate design, debugging, and discovery of relevant code prior to the use of filesystem tools. Use code-rag for vague or exploratory queries about a codebase; start with `index_status` then discover relevant code using the search tool with natural language topic descriptions.

For agents that support per-project instruction files (e.g. `AGENTS.md` for Claude Code), place this text there so it applies automatically whenever you work in an indexed repo.

## Make targets

All targets that operate on a source directory accept `SRC=` (defaults to `.`). Targets that use the index database accept `DB=` (defaults to `index.db`).

| Target | Description | Example |
|---|---|---|
| `install` | Install all dependencies | `make install` |
| `index` | Index a directory (incremental) | `make index SRC=../repo DB=my.db` |
| `reindex` | Rebuild embeddings from scratch | `make reindex SRC=../repo` |
| `mcp` | Start MCP server (stdio) | `make mcp BASE_URL=http://host:8081` |
| `webui` | Start REST API + web UI | `make webui DB=my.db PORT=8081` |
| `test` | Run full test suite | `make test` |
| `test-unit` | Run unit tests only | `make test-unit` |
| `test-integration` | Run integration tests | `make test-integration` |
| `lint` | Check code style | `make lint` |
| `format` | Auto-format code | `make format` |
| `add-claude-mcp` | Register with Claude Code | `make add-claude-mcp` |
| `remove-claude-mcp` | Unregister from Claude Code | `make remove-claude-mcp` |
| `add-pi-mcp` | Register with pi-agent | `make add-pi-mcp` |
| `remove-pi-mcp` | Unregister from pi-agent | `make remove-pi-mcp` |
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

`code-rag-mcp.py` exposes 10 tools, all proxied to the REST API documented in `SKILL.md`:

| Tool | Purpose |
|---|---|
| `search` | Natural-language vector search; returns ranked unit summaries |
| `get_unit` | Fetch full source for one or more qualified paths |
| `list_units` | List indexed units with optional glob filter |
| `list_files` | List indexed files with optional glob filter |
| `list_repos` | List indexed repositories |
| `index_status` | Per-repo file/unit counts and last-indexed timestamp |
| `staleness` | Per-repo index freshness vs. each repo's git HEAD |
| `index_start` | Enqueue paths for indexing (returns immediately) |
| `index_job_status` | Poll the running indexing job |
| `index_cancel` | Signal the running job to cancel |

Full parameter and response schemas are in `SKILL.md` (generated from the live OpenAPI spec).

## Architecture

- **Web UI** (`mcp_rag/webui.py`): FastAPI ASGI app — REST API + browser UI; single source of truth for SQLite reads/writes
- **MCP server** (`code-rag-mcp.py`): standalone script proxying 10 MCP tools over HTTP to the web UI
- **CLI client** (`code-rag-cli.py`): standalone stdlib-only script for human/script use
- **Embeddings**: [fastembed](https://github.com/qdrant/fastembed) in-process via ONNX Runtime (`nomic-ai/nomic-embed-text-v1.5-Q`, 768-dim)
- **Storage**: SQLite (WAL mode) + [sqlite-vec](https://github.com/asg017/sqlite-vec) — documents, metadata, and vectors in a single file
- **Summarization**: Anthropic API (Claude Haiku) or Ollama, index-time only
- **MCP transport**: stdio by default; streamable-HTTP via `--http`

## License

See [LICENSE](LICENSE) for details.
