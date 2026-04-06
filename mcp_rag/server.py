"""MCP server for mcp-rag."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fastmcp import FastMCP

from mcp_rag.db import list_repos_db, open_db
from mcp_rag.discovery import read_git_description
from mcp_rag.models import Embedder, encode_embedding

mcp = FastMCP("code-rag")

# ---------------------------------------------------------------------------
# Runtime state — injected by configure() before serving or in tests
# ---------------------------------------------------------------------------

_db_path: Path | None = None
_embedder: Embedder | None = None
_conn: sqlite3.Connection | None = None


def configure(db_path: Path | None, embedder: Embedder | None) -> None:
    """Inject the DB path and embedder. Called by the CLI and tests."""
    global _db_path, _embedder, _conn
    if _conn is not None:
        _conn.close()
        _conn = None
    _db_path = db_path
    _embedder = embedder


def _get_conn() -> sqlite3.Connection | None:
    """Return the cached DB connection, opening it lazily on first use."""
    global _conn
    if _conn is None and _db_path is not None and _embedder is not None:
        _conn = open_db(_db_path, embed_dim=_embedder.dim, embed_model=_embedder.model)
    return _conn


# ---------------------------------------------------------------------------
# Glob helpers
# ---------------------------------------------------------------------------


def _glob_where(globs: list[str] | None, column: str = "u.path") -> tuple[str, list]:
    """Build a WHERE clause from GLOB filters.

    Returns ``("", [])`` when *globs* is empty/None, or
    ``("WHERE col GLOB ? AND col GLOB ?", [g1, g2])`` otherwise.
    """
    if not globs:
        return "", []
    clauses = " AND ".join(f"{column} GLOB ?" for _ in globs)
    return f"WHERE {clauses}", list(globs)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool
async def search(query: str, top_k: int = 5, globs: list[str] | None = None) -> list[dict]:
    """Search the indexed codebase using natural language.

    Every indexed unit (function, class, markdown section, etc.) has a
    pre-computed natural-language summary.  Queries are matched against these
    summaries via vector similarity, so **ask questions the way you'd ask a
    colleague** — e.g. "how does authentication work?" rather than keyword
    fragments.

    Results include the qualified path (``repo/file.py:Class:method``), the
    human-readable summary, and a relevance score in [0.0, 1.0] (higher is
    better).  top_k is capped at 20.  Use ``get_unit`` to retrieve full
    source content for specific paths.

    **Recommended workflow:**
    - Start with ``*.md`` to find authored documentation and module overviews.
    - Narrow into code (``*.py``, ``*.h``, ``*.cpp``) only after you have
      the big picture from docs.
    - Use semantic questions, not grep-style keywords — the index understands
      intent, not just tokens.

    Use globs to filter by qualified path with SQLite GLOB syntax.  Multiple
    globs are AND'd together — all must match.
    (e.g. ``["backend/*", "*.py:*"]`` → only Python units in the backend repo).
    """
    if _db_path is None or _embedder is None:
        return []

    k = min(top_k, 20)
    emb = _embedder.embed(query)

    if globs:
        where, glob_params = _glob_where(globs)
        sql = f"""
            SELECT
                u.path,
                u.summary,
                sub.dist
            FROM (
                SELECT e.unit_id, vec_distance_cosine(e.embedding, ?) AS dist
                FROM embeddings e
                ORDER BY dist ASC
                LIMIT ?
            ) sub
            JOIN units u ON u.id = sub.unit_id
            {where}
            ORDER BY sub.dist ASC
        """
        # Fetch more candidates from the vector search so filtering still
        # returns enough results.  Cap at 200 to keep the scan reasonable.
        candidates = min(k * 10, 200)
        rows = _get_conn().execute(sql, [encode_embedding(emb), candidates] + glob_params).fetchall()
        rows = rows[:k]
    else:
        sql = """
            SELECT
                u.path,
                u.summary,
                vec_distance_cosine(e.embedding, ?) AS dist
            FROM embeddings e
            JOIN units u ON u.id = e.unit_id
            ORDER BY dist ASC
            LIMIT ?
        """
        rows = _get_conn().execute(sql, (encode_embedding(emb), k)).fetchall()

    return [
        {
            "path": row[0],
            "summary": row[1],
            "score": round(1.0 - row[2] / 2.0, 6),
        }
        for row in rows
    ]


@mcp.tool
async def get_unit(paths: list[str]) -> list[dict]:
    """Retrieve the full source content of one or more indexed units by
    qualified path.

    Use this after ``search`` or ``list_units`` to read the actual code for
    specific results.  Paths must match exactly (use the ``path`` values
    returned by those tools).

    Returns the qualified path, source content, and summary for each matched
    path.  Paths that do not match any indexed unit are silently skipped.
    """
    if _db_path is None or _embedder is None:
        return []

    if not paths:
        return []

    conn = _get_conn()
    placeholders = ",".join("?" for _ in paths)
    sql = f"""
        SELECT u.path, u.content, u.summary
        FROM units u
        WHERE u.path IN ({placeholders})
        ORDER BY u.path
    """
    rows = conn.execute(sql, paths).fetchall()

    return [
        {"path": row[0], "content": row[1], "summary": row[2]}
        for row in rows
    ]


@mcp.tool
async def list_units(globs: list[str] | None = None, limit: int = 100) -> list[dict]:
    """List semantic units (functions, classes, methods, sections, etc.) in the
    index.

    Returns the qualified path (``repo/file.py:Class:method``) and summary for
    each unit, ordered alphabetically by path.  Use this to understand the
    structure of a file, module, or the entire codebase without fetching
    full source content.

    Use globs to filter by qualified path with SQLite GLOB syntax.  Multiple
    globs are AND'd together.  The qualified path starts with the repo name
    then the relative file path:

    - ``["backend/*"]``          — all units in the backend repo
    - ``["*.js:*"]``             — all JS units
    - ``["*:Router:*"]``         — all Router members across languages
    - ``["backend/*", "*.py:*"]`` — Python units in backend only
    """
    if _db_path is None or _embedder is None:
        return []

    capped = min(limit, 500)
    conn = _get_conn()

    where, params = _glob_where(globs)
    sql = f"SELECT u.path, u.summary FROM units u {where} ORDER BY u.path LIMIT ?"
    params.append(capped)
    rows = conn.execute(sql, params).fetchall()

    return [{"path": row[0], "summary": row[1]} for row in rows]


@mcp.tool
async def list_files(globs: list[str] | None = None) -> list[dict]:
    """List files that have been indexed.

    Returns the repo name, relative file path, and last-indexed timestamp for
    every file in the index.  Call this early to understand what content is
    available before searching.

    Use globs to filter by file path with SQLite GLOB syntax.  Multiple globs
    are AND'd together (e.g. ``["backend/*", "*.py"]``).
    """
    if _db_path is None or _embedder is None:
        return []

    conn = _get_conn()

    # Can't reuse _glob_where here: files table has no single qualified
    # path column, so we GLOB against the concatenated repo_name/path.
    from_clause = (
        "SELECT r.name, f.path, f.indexed_at "
        "FROM files f JOIN repos r ON r.id = f.repo_id"
    )
    params: list = []
    if globs:
        clauses = " AND ".join(
            "(r.name || '/' || f.path) GLOB ?" for _ in globs
        )
        from_clause += f" WHERE {clauses}"
        params = list(globs)

    from_clause += " ORDER BY r.name, f.path"
    rows = conn.execute(from_clause, params).fetchall()

    return [
        {"repo": row[0], "path": row[1], "indexed_at": row[2]}
        for row in rows
    ]


@mcp.tool
async def index_status() -> list[dict]:
    """Return the current state of the index.

    Call this before any other RAG tool to confirm the index is populated
    and fresh.  If ``unit_count`` is 0 or ``last_indexed_at`` is stale,
    search results will be empty or incomplete.

    Reports per-repo file count, semantic unit count, and the timestamp of
    the most recent indexing run.
    """
    if _db_path is None or _embedder is None:
        return []

    rows = (
        _get_conn()
        .execute(
            """
        SELECT
            r.name,
            COUNT(DISTINCT f.id)  AS file_count,
            COUNT(u.id)           AS unit_count,
            MAX(f.indexed_at)     AS last_indexed_at
        FROM repos r
        JOIN files f ON f.repo_id = r.id
        LEFT JOIN units u ON u.file_id = f.id
        GROUP BY r.id
        ORDER BY r.name
        """,
        )
        .fetchall()
    )

    return [
        {
            "repo": row[0],
            "file_count": row[1],
            "unit_count": row[2],
            "last_indexed_at": row[3],
        }
        for row in rows
    ]


@mcp.tool
async def list_repos() -> list[dict]:
    """List all indexed repositories.

    Returns the repo name, absolute root path, and git description for each
    repository in the index.
    """
    if _db_path is None or _embedder is None:
        return []

    conn = _get_conn()
    repos = list_repos_db(conn)
    for repo in repos:
        root = Path(repo["root"])
        repo["description"] = read_git_description(root)
    return repos
