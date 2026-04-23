.PHONY: install test test-unit test-integration lint format index reindex serve webui skill clean add-claude-mcp remove-claude-mcp add-pi-mcp remove-pi-mcp eval-variant

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

# Start the web UI. Usage: make webui DB=my.db PORT=8081
webui:
	uv run code-rag webui --db $(or $(DB),index.db) --port $(or $(PORT),8081)

# Register this server with Claude Code (run once after cloning). Usage: make add-claude-mcp DB=my.db
add-claude-mcp:
	claude mcp add --transport stdio -s user code-rag -- uv run --directory $(DIR) code-rag serve --db $(abspath $(or $(DB),index.db))

# Unregister this server from Claude Code
remove-claude-mcp:
	claude mcp remove code-rag -s user

# Register this server with the pi agent (run once after cloning). Usage: make add-pi-mcp DB=my.db
add-pi-mcp:
	pi install npm:pi-mcp-adapter
	python3 scripts/add_pi_mcp.py $(DIR) $(abspath $(or $(DB),index.db))

# Unregister this server from the pi agent
remove-pi-mcp:
	python3 scripts/remove_pi_mcp.py

# Regenerate SKILL.md from the live OpenAPI spec
skill:
	uv run python scripts/gen_skill.py > SKILL.md

# Evaluate a prompt variant end-to-end: rebuild index with the variant,
# score it against the eval query sets, and ratchet against the current
# champion(s). Usage: make eval-variant VARIANT=<id> [SRC_DB=index.db]
eval-variant:
	@test -n "$(VARIANT)" || (echo "VARIANT=<id> required"; exit 2)
	@mkdir -p eval/dbs
	uv run python -m eval.harness.rebuild --src-db $(or $(SRC_DB),index.db) --variant $(VARIANT) --out-db eval/dbs/$(VARIANT).db --force
	@set -e; \
	RECEIPT=$$(uv run python -m eval.harness.score --db eval/dbs/$(VARIANT).db --variant-id $(VARIANT) --print-path); \
	echo "receipt: $$RECEIPT"; \
	uv run python -m eval.harness.ratchet --candidate "$$RECEIPT"

# Remove the local index database and any SQLite WAL artifacts
clean:
	rm -f index.db index.db-wal index.db-shm
