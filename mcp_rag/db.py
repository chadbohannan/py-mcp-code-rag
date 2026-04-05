"""Database connection and schema management for mcp-rag."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import sqlite_vec

SCHEMA_VERSION = "2"

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL_METADATA = """\
CREATE TABLE metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_DDL_REPOS = """\
CREATE TABLE repos (
    id       INTEGER PRIMARY KEY,
    name     TEXT NOT NULL UNIQUE,
    root     TEXT NOT NULL UNIQUE,
    added_at TEXT NOT NULL
);
"""

_DDL_FILES = """\
CREATE TABLE files (
    id         INTEGER PRIMARY KEY,
    repo_id    INTEGER NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    path       TEXT NOT NULL,
    mtime      REAL NOT NULL,
    md5        TEXT NOT NULL,
    indexed_at TEXT NOT NULL,
    UNIQUE (repo_id, path)
);
"""

_DDL_UNITS = """\
CREATE TABLE units (
    id          INTEGER PRIMARY KEY,
    file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    path        TEXT NOT NULL,
    content     TEXT NOT NULL,
    content_md5 TEXT NOT NULL,
    summary     TEXT NOT NULL,
    unit_type   TEXT NOT NULL,
    unit_name   TEXT NOT NULL,
    char_offset INTEGER NOT NULL
);
"""

# Dimension is substituted at creation time; stored in metadata.
_DDL_EMBEDDINGS = """\
CREATE VIRTUAL TABLE embeddings USING vec0 (
    unit_id   INTEGER PRIMARY KEY,
    embedding FLOAT[{dim}]
);
"""

# vec0 virtual tables cannot carry FOREIGN KEY constraints, so cascade the
# delete from units to embeddings via a trigger instead.
_DDL_TRIGGER = """\
CREATE TRIGGER units_delete_cascade
AFTER DELETE ON units
FOR EACH ROW
BEGIN
    DELETE FROM embeddings WHERE unit_id = OLD.id;
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
    ``embed_dim``, and ``schema_version`` in ``metadata``.

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
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception as exc:
        conn.close()
        raise RuntimeError(
            f"error: failed to load sqlite-vec extension: {exc}\n"
            "Ensure sqlite-vec >= 0.1.0 is installed: uv add sqlite-vec"
        ) from exc

    conn.execute("PRAGMA journal_mode = WAL")

    # Determine whether this is a first-time open by checking for the metadata table.
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata'"
    ).fetchone()

    if exists is None:
        _create_schema(conn, embed_dim, embed_model)
    else:
        _validate_meta(conn, embed_dim, embed_model)

    return conn


# ---------------------------------------------------------------------------
# Repo helpers
# ---------------------------------------------------------------------------


def upsert_repo(conn: sqlite3.Connection, name: str, root: str) -> int:
    """Insert a repo or return its id if it already exists.

    Uses INSERT OR IGNORE on root (unique), then SELECT to get the id.
    """
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO repos (name, root, added_at) VALUES (?, ?, ?)",
        (name, root, now),
    )
    row = conn.execute("SELECT id FROM repos WHERE root = ?", (root,)).fetchone()
    if row is None:
        raise RuntimeError(f"Failed to upsert repo with root {root!r}")
    return row[0]


def list_repos_db(conn: sqlite3.Connection) -> list[dict]:
    """Return all repos as ``[{name, root, added_at}]``."""
    rows = conn.execute("SELECT name, root, added_at FROM repos ORDER BY name").fetchall()
    return [{"name": r[0], "root": r[1], "added_at": r[2]} for r in rows]


def remove_repo_db(conn: sqlite3.Connection, name: str) -> int:
    """Delete a repo by name. Returns the number of rows deleted (0 or 1)."""
    cur = conn.execute("DELETE FROM repos WHERE name = ?", (name,))
    conn.commit()
    return cur.rowcount


def rename_repo_db(conn: sqlite3.Connection, old_name: str, new_name: str) -> int:
    """Rename a repo. Returns the number of rows updated (0 or 1)."""
    cur = conn.execute(
        "UPDATE repos SET name = ? WHERE name = ?", (new_name, old_name)
    )
    conn.commit()
    return cur.rowcount


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _create_schema(conn: sqlite3.Connection, embed_dim: int, embed_model: str) -> None:
    conn.execute(_DDL_METADATA)
    conn.execute(_DDL_REPOS)
    conn.execute(_DDL_FILES)
    conn.execute(_DDL_UNITS)
    conn.execute(_DDL_EMBEDDINGS.format(dim=embed_dim))
    conn.execute(_DDL_TRIGGER)
    conn.executemany(
        "INSERT INTO metadata (key, value) VALUES (?, ?)",
        [
            ("schema_version", SCHEMA_VERSION),
            ("embed_model", embed_model),
            ("embed_dim", str(embed_dim)),
        ],
    )
    conn.commit()


def _validate_meta(conn: sqlite3.Connection, embed_dim: int, embed_model: str) -> None:
    meta = dict(conn.execute("SELECT key, value FROM metadata").fetchall())
    stored_model = meta.get("embed_model", "")
    stored_dim = meta.get("embed_dim", "")
    if stored_model != embed_model or stored_dim != str(embed_dim):
        conn.close()
        raise ModelMismatchError(
            f"Index was built with {stored_model} (dim={stored_dim}).\n"
            f"Current model is {embed_model} (dim={embed_dim}).\n"
            "Run: mcp-rag index --reindex <paths...>"
        )
