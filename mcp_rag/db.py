"""Database connection and schema management for mcp-rag."""
import sqlite3
from pathlib import Path

import sqlite_vec

SCHEMA_VERSION = "1"

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL_META = """\
CREATE TABLE mcp_rag_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_DDL_FILES = """\
CREATE TABLE mcp_rag_files (
    id         INTEGER PRIMARY KEY,
    root       TEXT NOT NULL,
    path       TEXT NOT NULL,
    mtime      REAL NOT NULL,
    md5        TEXT NOT NULL,
    indexed_at TEXT NOT NULL,
    UNIQUE (root, path)
);
"""

_DDL_UNITS = """\
CREATE TABLE mcp_rag_units (
    id          INTEGER PRIMARY KEY,
    file_id     INTEGER NOT NULL REFERENCES mcp_rag_files(id) ON DELETE CASCADE,
    unit_type   TEXT NOT NULL,
    unit_name   TEXT,
    content     TEXT NOT NULL,
    content_md5 TEXT NOT NULL,
    summary     TEXT NOT NULL,
    char_offset INTEGER NOT NULL
);
"""

# Dimension is substituted at creation time; stored in mcp_rag_meta.
_DDL_EMBEDDINGS = """\
CREATE VIRTUAL TABLE mcp_rag_embeddings USING vec0 (
    unit_id   INTEGER PRIMARY KEY,
    embedding FLOAT[{dim}]
);
"""

# vec0 virtual tables cannot carry FOREIGN KEY constraints, so cascade the
# delete from mcp_rag_units to mcp_rag_embeddings via a trigger instead.
_DDL_TRIGGER = """\
CREATE TRIGGER mcp_rag_units_delete_cascade
AFTER DELETE ON mcp_rag_units
FOR EACH ROW
BEGIN
    DELETE FROM mcp_rag_embeddings WHERE unit_id = OLD.id;
END;
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ModelMismatchError(Exception):
    """Raised when the stored embed model or dimension does not match."""


def open_db(path: Path, embed_dim: int, embed_model: str) -> sqlite3.Connection:
    """Open or create the index database.

    New database
    ------------
    Creates all tables, enables WAL mode, and stores ``embed_model``,
    ``embed_dim``, and ``schema_version`` in ``mcp_rag_meta``.

    Existing database
    -----------------
    Validates that ``embed_model`` and ``embed_dim`` match the stored values.
    Raises ``ModelMismatchError`` on mismatch so the caller can direct the user
    to ``mcp-rag index --reindex``.

    The sqlite-vec extension is loaded on every call.
    """
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        sqlite_vec.load(conn)
    except Exception as exc:
        conn.close()
        raise RuntimeError(
            f"error: failed to load sqlite-vec extension: {exc}\n"
            "Ensure sqlite-vec >= 0.1.0 is installed: uv add sqlite-vec"
        ) from exc

    conn.execute("PRAGMA journal_mode = WAL")

    # Determine whether this is a first-time open by checking for the meta table.
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='mcp_rag_meta'"
    ).fetchone()

    if exists is None:
        _create_schema(conn, embed_dim, embed_model)
    else:
        _validate_meta(conn, embed_dim, embed_model)

    return conn


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _create_schema(conn: sqlite3.Connection, embed_dim: int, embed_model: str) -> None:
    conn.execute(_DDL_META)
    conn.execute(_DDL_FILES)
    conn.execute(_DDL_UNITS)
    conn.execute(_DDL_EMBEDDINGS.format(dim=embed_dim))
    conn.execute(_DDL_TRIGGER)
    conn.executemany(
        "INSERT INTO mcp_rag_meta (key, value) VALUES (?, ?)",
        [
            ("schema_version", SCHEMA_VERSION),
            ("embed_model", embed_model),
            ("embed_dim", str(embed_dim)),
        ],
    )
    conn.commit()


def _validate_meta(conn: sqlite3.Connection, embed_dim: int, embed_model: str) -> None:
    meta = dict(conn.execute("SELECT key, value FROM mcp_rag_meta").fetchall())
    stored_model = meta.get("embed_model", "")
    stored_dim = meta.get("embed_dim", "")
    if stored_model != embed_model or stored_dim != str(embed_dim):
        conn.close()
        raise ModelMismatchError(
            f"Index was built with {stored_model} (dim={stored_dim}).\n"
            f"Current model is {embed_model} (dim={embed_dim}).\n"
            "Run: mcp-rag index --reindex <paths...>"
        )
