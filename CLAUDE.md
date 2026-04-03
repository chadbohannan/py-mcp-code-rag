CLAUDE.md — Guidance for Claude Code developers working with this repository

PURPOSE:
This file provides guidance to keep implementation documentation synchronized with actual code. The repo uses a spec-driven development workflow where design_spec.md is the single source of truth during implementation, but must be updated whenever deviations occur.

KEY FILES:
- overview.md: Design rationale and explanation of the Semantic Surrogate Indexing technique (the core tech)
- design_spec.md: Complete implementation specification including microservice architecture, PostgreSQL schema with JSONB columns for metadata/FTS, CLI interface contract, API boundaries, testing strategy, and observability requirements

WORKFLOW:
1. Read design_spec.md before starting a new task
2. If fixing a spec error → confirm fix with tests
3. If choosing simpler approach (e.g., using pgcrypto instead of external KMS) → update spec to reflect decision with rationale
4. If spec ambiguity → make explicit implementation choice and log it in spec
5. Before committing → verify design_spec.md still accurately describes the codebase

WHY THIS MATTERS:
- Prevents documentation rot over time
- Makes onboarding faster (new devs follow spec, not existing code)
- Creates audit trail of design decisions (who chose what tech and why)
- Technical debt in understanding (code doesn't match spec) is treated as critical bug

BUILD:
- This is a uv project. Use `uv run` to execute commands and `uv add` to manage dependencies.

GOTCHAS:
- When adding support for a new language, you MUST update `_SUPPORTED_EXTENSIONS` in `mcp_rag/indexer.py` in addition to adding the parser in `mcp_rag/parsers.py`. The indexer's extension allowlist gates file processing before `parse_file` is ever called. Missing this causes new parsers to silently do nothing.

CONSTRAINTS:
- Spec deviations require comments explaining the change
- The spec must be kept current — it's a living document
- Review process checks for unlogged deviations; these are rejected