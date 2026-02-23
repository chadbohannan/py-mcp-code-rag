# CLAUDE.md

Guidance for Claude Code when working in this repository.

- `overview.md` — design rationale and the Semantic Surrogate Indexing technique
- `design_spec.md` — complete implementation specification (architecture, schema, CLI, testing)

## Test commands

```bash
uv run pytest tests/unit tests/integration -x -q   # fast dev loop
uv run pytest                                       # full suite
```

No external services required. `FakeEmbedder` and `FakeSummarizer` isolate fastembed and the
Anthropic API from all test runs.

## Dev commands

```bash
uv sync                                        # install dependencies
uv run mcp-rag index /path/to/project          # build or update the index
uv run mcp-rag serve /path/to/project          # start MCP server (stdio)
uv run mcp-rag /path/to/project                # index if absent, then serve
uv run mcp-rag serve --http /path/to/project   # HTTP transport (localhost:8000)
```
