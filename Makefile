.PHONY: install test test-unit test-integration lint format index reindex serve clean add-claude-mcp remove-claude-mcp

# Resolve absolute path at make-time so the registered command works from any working directory
DIR := $(shell pwd)

# Install all dependencies including dev group
install:
	uv sync --all-groups

# Run the full test suite
test:
	uv run pytest

# Run only unit tests (fast, no external dependencies)
test-unit:
	uv run pytest tests/unit

# Run only integration tests (may require Ollama or other services)
test-integration:
	uv run pytest tests/integration

# Check code style without modifying files
lint:
	uv run ruff check mcp_rag tests
	uv run ruff format --check mcp_rag tests

# Auto-format source and tests
format:
	uv run ruff format mcp_rag tests

# Index the current directory into index.db (skips unchanged files)
index:
	uv run mcp-rag index .

# Re-embed everything from scratch (use after changing embed model)
reindex:
	uv run mcp-rag index --reindex .

# Start the MCP stdio server against the local index.db
serve:
	uv run mcp-rag serve --db index.db

# Register this server with Claude Code (run once after cloning)
add-claude-mcp:
	claude mcp add --transport stdio mcp-rag -- uv run --directory $(DIR) mcp-rag serve --db $(DIR)/index.db

# Unregister this server from Claude Code
remove-claude-mcp:
	claude mcp remove mcp-rag

# Remove the local index database and any SQLite WAL artifacts
clean:
	rm -f index.db index.db-wal index.db-shm
