"""MCP server for mcp-rag."""
from __future__ import annotations

import struct
from pathlib import Path

from fastmcp import FastMCP

from mcp_rag.db import open_db
from mcp_rag.models import Embedder

mcp = FastMCP("mcp-rag")

# ---------------------------------------------------------------------------
# Runtime state — injected by configure() before serving or in tests
# ---------------------------------------------------------------------------

_db_path: Path | None = None
_embedder: Embedder | None = None


def configure(db_path: Path | None, embedder: Embedder | None) -> None:
    """Inject the DB path and embedder. Called by the CLI and tests."""
    global _db_path, _embedder
    _db_path = db_path
    _embedder = embedder


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool
async def search(query: str, top_k: int = 5) -> list[dict]:
    """Search the indexed codebase using a natural language question.

    Embeds the query and returns the closest matching semantic units by vector
    similarity. Each result includes the source path, unit type, unit name,
    original source content, a human-readable summary, and a relevance score
    in [0.0, 1.0] (higher is better). top_k is capped at 20.
    """
    if _db_path is None or _embedder is None:
        return []

    k = min(top_k, 20)
    emb = _embedder.embed(query)
    emb_bytes = struct.pack(f"{len(emb)}f", *emb)

    conn = open_db(_db_path, embed_dim=_embedder.dim, embed_model=_embedder.model)
    try:
        rows = conn.execute(
            """
            SELECT
                f.path,
                u.unit_type,
                u.unit_name,
                u.content,
                u.summary,
                vec_distance_cosine(e.embedding, ?) AS dist
            FROM mcp_rag_embeddings e
            JOIN mcp_rag_units u ON u.id = e.unit_id
            JOIN mcp_rag_files f ON f.id = u.file_id
            ORDER BY dist ASC
            LIMIT ?
            """,
            (emb_bytes, k),
        ).fetchall()
    finally:
        conn.close()

    return [
        {
            "path": row[0],
            "unit_type": row[1],
            "unit_name": row[2],
            "content": row[3],
            "summary": row[4],
            "score": round(1.0 - row[5] / 2.0, 6),
        }
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

    conn = open_db(_db_path, embed_dim=_embedder.dim, embed_model=_embedder.model)
    try:
        rows = conn.execute(
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
        ).fetchall()
    finally:
        conn.close()

    return [
        {
            "root": row[0],
            "file_count": row[1],
            "unit_count": row[2],
            "last_indexed_at": row[3],
        }
        for row in rows
    ]
