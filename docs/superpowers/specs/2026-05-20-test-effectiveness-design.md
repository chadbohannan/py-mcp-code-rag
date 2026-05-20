# Test Effectiveness Resolution — Design

**Date:** 2026-05-20
**Branch:** rework-mcp
**Status:** Draft for review

## Background

A review of the `rework-mcp` branch identified eight gaps in test effectiveness introduced by the MCP-to-HTTP migration. The deletions removed 594 lines of integration coverage (`tests/integration/test_mcp_server.py` and `tests/integration/test_server_search.py`); the additions are 322 lines of mocked-`urlopen` unit tests for the two new top-level scripts (`code-rag-mcp.py`, `code-rag-cli.py`).

The net effect is that the contract between the MCP/CLI clients and the FastAPI webui is no longer verified end-to-end. Both sides are mocked independently, so a divergence between them would not surface until production use. This spec resolves all eight findings.

## Goals

- Restore end-to-end coverage of the public REST API surface (the load-bearing regression).
- Tighten the new mocked-client tests so they catch encoding bugs (lists, None params, wrong methods).
- Cover every CLI subcommand at smoke level.
- Land everything in three commits so the work can be reviewed and parallelized cleanly.

## Non-Goals

- WebSocket (`/ws/index`) coverage — orchestration cost is too high for a fast test tier.
- Index lifecycle endpoints (`POST /api/index`, `GET /api/index/status`, `POST /api/index/cancel`) integration coverage — same reason; these stay mocked at the unit tier.
- Frontend/HTML coverage of the webui index page.
- Refactoring `code-rag-mcp.py` / `code-rag-cli.py` themselves; the duplicated `_url`/`_request` helpers are intentional per CLAUDE.md.

## Architecture

### File layout

```
tests/
  conftest.py                     # MODIFY — add parse_request() helper
  integration/
    conftest.py                   # NEW — indexed_db (module) + client (function) fixtures
    test_webui_api.py             # NEW — replaces deleted test_server_search.py
  unit/
    test_mcp_client.py            # MODIFY — tighten asserts, add gaps
    test_cli_client.py            # MODIFY — cover 11 untested subcommands
```

### Shared fixture (`tests/integration/conftest.py`)

```python
@pytest.fixture(scope="module")
def indexed_db(tmp_path_factory):
    """Build a small real SQLite index once per module."""
    db_path = tmp_path_factory.mktemp("idx") / "test.db"
    repo = make_git_project(
        "testrepo",
        {
            "src/app.py": "def login(user): ...\ndef logout(): ...\nclass Session: ...",
            "src/db.py":  "def connect(): ...\ndef disconnect(): ...",
            "README.md":  "## Overview\nProject docs.",
        },
    )
    run_index([repo], db_path, FakeEmbedder(dim=4), FakeSummarizer())
    return db_path

@pytest.fixture
def client(indexed_db):
    """fastapi.TestClient wired to the indexed DB."""
    from mcp_rag.webui import create_app
    return TestClient(create_app(db_path=indexed_db))
```

Module-scoped DB build is shared across endpoint tests (~1s once vs. ~1s each). `client` is per-test so isolated overrides remain safe.

### Shared assertion helper (`tests/conftest.py`)

```python
def parse_request(mock_call) -> tuple[str, str, dict, dict | None]:
    """Return (method, path, query_dict, json_body) from a urlopen MagicMock call."""
    req = mock_call[0][0]
    parsed = urllib.parse.urlparse(req.full_url)
    query = urllib.parse.parse_qs(parsed.query)
    body = json.loads(req.data.decode()) if req.data else None
    return req.get_method(), parsed.path, query, body
```

This replaces every `m.call_args[0][0].full_url` substring assert in `test_mcp_client.py` and `test_cli_client.py`. ~20 lines, used by every new and rewritten test in the unit tier.

## Per-Issue Resolution

| # | Issue | Resolution | File |
|---|---|---|---|
| 1 | No webui end-to-end coverage | New `test_webui_api.py` with ~12 tests against `TestClient` covering each endpoint family | `tests/integration/test_webui_api.py` |
| 2 | URL substring assertions loose | `parse_request()` helper; rewrite all asserts to compare `path` and `query` dicts | `tests/conftest.py` + both client test files |
| 3 | Globs list-encoding unverified | Replace single-globs test with multi-globs case asserting `query["globs"] == ["*.py", "backend/*"]` | `test_mcp_client.py` |
| 4 | 11 CLI subcommands untested | One smoke test per uncovered subcommand: `unit`, `fetch`, `units`, `files`, `repos`, `status`, `browse`, `index-status`, `index-cancel`, `clear-repo`, plain `index` | `test_cli_client.py` |
| 5 | `index_status` `{}` branch untested | `test_index_status_missing_repos_key_returns_empty` returns `{}`, asserts `result == []` | `test_mcp_client.py` |
| 6 | `index --wait` under-asserted | Replace `assert s.called` with `assert s.call_count == 2`; assert success path returns without `SystemExit`; assert each `_print_job_status` field appears in stdout | `test_cli_client.py` |
| 7 | Module-load inconsistency | Convert `test_cli_client.py` to the same module-scoped fixture pattern `test_mcp_client.py` uses | `test_cli_client.py` |
| 8 | `--http` / `--port` wiring untested | Add `test_mcp_main_stdio_default` and `test_mcp_main_http_uses_port`, monkeypatching `mcp.run` and asserting kwargs | `test_mcp_client.py` |

## Integration Suite (`test_webui_api.py`)

Twelve tests, grouped by endpoint family. All share `indexed_db` and `client` fixtures.

| Endpoint family | Test | Verifies |
|---|---|---|
| `/api/search` | `test_search_returns_results_ordered_by_score` | non-empty; scores monotone non-increasing; `path`/`summary`/`score` present |
| | `test_search_top_k_caps_results` | `top_k=2` returns ≤ 2 |
| | `test_search_globs_filter_narrows_results` | `globs=["*.md"]` returns only README; `globs=["*.py:*"]` excludes README |
| `/api/units` & `/api/unit` | `test_units_listing_and_glob_filter` | `?globs=testrepo/src/app.py:*` returns 3 app.py units |
| | `test_unit_fetch_single_returns_content` | full `content`/`summary`/`path` for `testrepo/src/db.py:connect` |
| | `test_units_fetch_post_returns_multiple` | POST `/api/units/fetch` with 2 paths returns 2; missing paths silently dropped |
| `/api/files` | `test_files_listing_and_glob` | 3 files; `?globs=*.md` returns 1 |
| `/api/repos` & `/api/status` | `test_repos_lists_indexed_repo` | name/root/added_at present |
| | `test_status_returns_counts` | `total_units` matches actual count; `repos[0].file_count == 3` |
| `/api/repos/staleness` | `test_staleness_fresh_after_index` | freshly indexed → `stale=False` |
| `/api/browse` | `test_browse_repo_then_file` | top-level returns repos; drilling into a file returns its units |
| Error shape | `test_unit_not_found_returns_404` | 404 JSON shape matches what `code-rag-mcp.py:_request` extracts `detail` from |

The last test is the load-bearing cross-check that the mocked client tests cannot perform. If the webui's error JSON shape changes, the MCP's `detail = json.loads(body).get("detail", body)` breaks; this test catches it.

## Test Data

One small git repo built by `make_git_project`:

```
testrepo/
  src/app.py       # def login(user), def logout(), class Session
  src/db.py        # def connect(), def disconnect()
  README.md        # one ## section
```

~6 semantic units across 3 files. Enough to verify search ordering, glob filtering, and counts without being slow. Built once per test module via `FakeEmbedder(dim=4)` and `FakeSummarizer`.

## Sequencing

```
Step 1 — Foundation (1 commit)                  [blocking]
  - tests/conftest.py: parse_request() helper
  - tests/integration/conftest.py: indexed_db + client fixtures
  - One smoke test in test_webui_api.py to prove the fixture works

Step 2a — Integration suite (1 commit)          [parallelizable with 2b]
  - Fill in the remaining 11 integration tests

Step 2b — Unit-test polish (1 commit)           [parallelizable with 2a]
  - Issues #2, #3, #5, #6, #7, #8 from the resolution table
  - Add the 11 missing CLI subcommand tests (#4)

Step 3 — Verify (no commit)
  - Full test run, lint, manual review of diff stats
```

Step 1 is the only blocking dependency. 2a and 2b can be done by separate sessions or agents.

## Acceptance Criteria

1. `make test` passes; new test count is roughly +27 (12 integration + 4 new in `test_mcp_client.py` + 11 new in `test_cli_client.py`); several existing tests are rewritten in place to use `parse_request()`.
2. Every `m.call_args[0][0].full_url` substring assert is replaced with `parse_request()` equality.
3. Every CLI subparser in `code-rag-cli.py` has at least one test that calls its `func`.
4. `tests/integration/test_webui_api.py` runs in under 5s on the Pi (target).
5. `git diff master..HEAD --stat -- tests/` shows net non-negative line count vs. master.
6. `make lint` green.

## Risks & Mitigations

- **`create_app(db_path=...)` may not exist with that signature.** `webui.py` exposes `create_app` at line 582; the fixture must match its real signature. Verify during Step 1 and adjust the fixture (use `monkeypatch` on a module-level path if needed).
- **`make_git_project` helper may not support the dict-of-paths form shown above.** Verify in `tests/conftest.py` during Step 1; fall back to writing files manually if the helper is simpler.
- **Integration tests may be slower than 5s on the Pi.** The first-run summarize+embed loop dominates. If too slow, drop README.md and one .py file; ~4 units is still enough to exercise every assertion.
- **`FakeSummarizer` produces deterministic but non-meaningful summaries.** Search-ranking assertions must use score monotonicity and presence of expected paths, not specific score values or summary text.
