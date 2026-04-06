.PHONY: install test test-unit test-integration lint format index reindex serve clean add-claude-mcp remove-claude-mcp add-pi-mcp remove-pi-mcp

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

# Index a directory into index.db (skips unchanged files). Usage: make index SRC=../some-repo DB=my.db
index:
	uv run code-rag index --db $(or $(DB),index.db) $(or $(SRC),.)

# Re-embed everything from scratch (use after changing embed model). Usage: make reindex SRC=../some-repo DB=my.db
reindex:
	uv run code-rag index --reindex --db $(or $(DB),index.db) $(or $(SRC),.)

# Start the MCP stdio server. Usage: make serve DB=my.db
serve:
	uv run code-rag serve --db $(or $(DB),index.db)

# Register this server with Claude Code (run once after cloning). Usage: make add-claude-mcp DB=my.db
add-claude-mcp:
	claude mcp add --transport stdio code-rag -- uv run --directory $(DIR) code-rag serve --db $(abspath $(or $(DB),index.db))

# Unregister this server from Claude Code
remove-claude-mcp:
	claude mcp remove code-rag

# Register this server with the pi agent (run once after cloning). Usage: make add-pi-mcp DB=my.db
add-pi-mcp:
	python3 scripts/add_pi_mcp.py $(DIR) $(abspath $(or $(DB),index.db))

# Unregister this server from the pi agent
remove-pi-mcp:
	python3 scripts/remove_pi_mcp.py

# Remove the local index database and any SQLite WAL artifacts
clean:
	rm -f index.db index.db-wal index.db-shm
