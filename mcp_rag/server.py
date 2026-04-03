"""MCP server for mcp-rag."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fastmcp import FastMCP

from mcp_rag.db import open_db
from mcp_rag.models import Embedder, encode_embedding

mcp = FastMCP("mcp-rag")

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
# Tools
# ---------------------------------------------------------------------------


@mcp.tool
async def search(query: str, top_k: int = 5, path_glob: str | None = None) -> list[dict]:
    """Search the indexed codebase using a natural language question.

    Embeds the query and returns the closest matching semantic units by vector
    similarity. Each result includes the qualified path
    (``relative/file.py:Class:method``), original source content, a
    human-readable summary, and a relevance score in [0.0, 1.0] (higher is
    better). top_k is capped at 20.

    Use path_glob to filter results by qualified path using SQLite GLOB syntax
    (e.g. ``*.py:Router:*``, ``*/go/*``, ``*:Wire Format*``).
    """
    if _db_path is None or _embedder is None:
        return []

    k = min(top_k, 20)
    emb = _embedder.embed(query)

    if path_glob is not None:
        sql = """
            SELECT
                u.path,
                u.content,
                u.summary,
                sub.dist
            FROM (
                SELECT e.unit_id, vec_distance_cosine(e.embedding, ?) AS dist
                FROM mcp_rag_embeddings e
                ORDER BY dist ASC
                LIMIT ?
            ) sub
            JOIN mcp_rag_units u ON u.id = sub.unit_id
            WHERE u.path GLOB ?
            ORDER BY sub.dist ASC
        """
        # Fetch more candidates from the vector search so filtering still
        # returns enough results.  Cap at 200 to keep the scan reasonable.
        candidates = min(k * 10, 200)
        rows = _get_conn().execute(sql, (encode_embedding(emb), candidates, path_glob)).fetchall()
        rows = rows[:k]
    else:
        sql = """
            SELECT
                u.path,
                u.content,
                u.summary,
                vec_distance_cosine(e.embedding, ?) AS dist
            FROM mcp_rag_embeddings e
            JOIN mcp_rag_units u ON u.id = e.unit_id
            ORDER BY dist ASC
            LIMIT ?
        """
        rows = _get_conn().execute(sql, (encode_embedding(emb), k)).fetchall()

    return [
        {
            "path": row[0],
            "content": row[1],
            "summary": row[2],
            "score": round(1.0 - row[3] / 2.0, 6),
        }
        for row in rows
    ]


@mcp.tool
async def list_files(path_glob: str | None = None) -> list[dict]:
    """List files that have been indexed.

    Returns the file path, root, and last-indexed timestamp for every file
    in the index.  Use path_glob to filter by file path using SQLite GLOB
    syntax (e.g. ``*.py``, ``*/tests/*``).
    """
    if _db_path is None or _embedder is None:
        return []

    conn = _get_conn()
    if path_glob is not None:
        rows = conn.execute(
            "SELECT root, path, indexed_at FROM mcp_rag_files WHERE path GLOB ? ORDER BY path",
            (path_glob,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT root, path, indexed_at FROM mcp_rag_files ORDER BY path",
        ).fetchall()

    return [
        {"root": row[0], "path": row[1], "indexed_at": row[2]}
        for row in rows
    ]


@mcp.tool
async def index_status() -> list[dict]:
    """Return the current state of the index.

    Reports per-root file count, semantic unit count, and the timestamp of the
    most recent indexing run. Returns one entry per distinct root path that has
    been indexed into the active database.
    """
    if _db_path is None or _embedder is None:
        return []

    rows = (
        _get_conn()
        .execute(
            """
        SELECT
            f.root,
            COUNT(DISTINCT f.id)  AS file_count,
            COUNT(u.id)           AS unit_count,
            MAX(f.indexed_at)     AS last_indexed_at
        FROM mcp_rag_files f
        LEFT JOIN mcp_rag_units u ON u.file_id = f.id
        GROUP BY f.root
        """,
        )
        .fetchall()
    )

    return [
        {
            "root": row[0],
            "file_count": row[1],
            "unit_count": row[2],
            "last_indexed_at": row[3],
        }
        for row in rows
    ]
