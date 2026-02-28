"""Indexing pipeline for mcp-rag.

Discovers files, parses them into SemanticUnits, summarises each unit,
embeds the summary, and writes everything to SQLite.
"""
from __future__ import annotations

import hashlib
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

import sqlite_vec

from mcp_rag.db import _DDL_EMBEDDINGS, _DDL_TRIGGER, open_db
from mcp_rag.discovery import discover_files
from mcp_rag.models import Embedder, SemanticUnit, Summarizer
from mcp_rag.parsers import parse_file
from mcp_rag.reconcile import StoredUnit, diff_units

_MAX_TOKENS = 8000
_MAX_CHARS = _MAX_TOKENS * 4

_SUPPORTED_EXTENSIONS = frozenset({".py", ".md", ".mdx", ".sql"})


class IndexAbortError(Exception):
    """Raised when the indexer cannot proceed."""


def run_index(
    roots: list[Path],
    db_path: Path,
    embedder: Embedder,
    summarizer: Summarizer,
    reindex: bool = False,
) -> None:
    """Build or update the index for *roots* into *db_path*."""
    # Guard: roots must not overlap
    resolved = [r.resolve() for r in roots]
    for i, a in enumerate(resolved):
        for b in resolved[i + 1 :]:
            if b.is_relative_to(a) or a.is_relative_to(b):
                raise IndexAbortError(
                    f"Roots overlap: {a} and {b}. Provide non-overlapping directories."
                )

    # Open (or create) the database
    is_new = not db_path.exists()
    if reindex and not is_new:
        conn = _open_for_reindex(db_path, embedder)
    else:
        conn = open_db(db_path, embed_dim=embedder.dim, embed_model=embedder.model)

    if is_new:
        print("No index found — creating a new index.", file=sys.stderr)

    total_deleted = 0
    try:
        for root in resolved:
            total_deleted += _index_root(conn, root, embedder, summarizer)
    finally:
        conn.close()

    if total_deleted:
        print(
            f"reconciled {total_deleted} deleted file(s) from the index.",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Reindex helpers
# ---------------------------------------------------------------------------

def _open_for_reindex(db_path: Path, embedder) -> sqlite3.Connection:
    """Open an existing DB and rebuild the embeddings table with a new dimension."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    sqlite_vec.load(conn)
    conn.execute("PRAGMA journal_mode = WAL")

    new_dim = embedder.dim
    new_model = embedder.model

    with conn:
        # Drop old virtual table and its cascade trigger
        conn.execute("DROP TABLE IF EXISTS mcp_rag_embeddings")
        conn.execute("DROP TRIGGER IF EXISTS mcp_rag_units_delete_cascade")
        # Recreate with new dimension
        conn.execute(_DDL_EMBEDDINGS.format(dim=new_dim))
        conn.execute(_DDL_TRIGGER)
        # Update stored metadata
        conn.execute(
            "INSERT OR REPLACE INTO mcp_rag_meta (key, value) VALUES ('embed_dim', ?)",
            (str(new_dim),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO mcp_rag_meta (key, value) VALUES ('embed_model', ?)",
            (new_model,),
        )
        # Re-embed all existing units from their stored summaries (no API calls)
        rows = conn.execute("SELECT id, summary FROM mcp_rag_units").fetchall()
        for unit_id, summary in rows:
            embedding = embedder.embed(summary)
            emb_bytes = struct.pack(f"{len(embedding)}f", *embedding)
            conn.execute(
                "INSERT INTO mcp_rag_embeddings (unit_id, embedding) VALUES (?, ?)",
                (unit_id, emb_bytes),
            )

    return conn


# ---------------------------------------------------------------------------
# Per-root indexing
# ---------------------------------------------------------------------------

def _index_root(
    conn: sqlite3.Connection,
    root: Path,
    embedder,
    summarizer,
) -> int:
    """Index one root directory. Returns the number of deleted files."""
    disk_files: set[Path] = set(discover_files(root))

    rows = conn.execute(
        "SELECT id, path, mtime, md5 FROM mcp_rag_files WHERE root = ?",
        (str(root),),
    ).fetchall()
    db_map: dict[Path, tuple[int, float, str]] = {
        Path(row[1]): (row[0], row[2], row[3]) for row in rows
    }

    # Remove entries for files that no longer exist on disk
    deleted_count = 0
    for path, (file_id, _, _) in db_map.items():
        if path not in disk_files:
            with conn:
                conn.execute("DELETE FROM mcp_rag_files WHERE id = ?", (file_id,))
            deleted_count += 1

    # Process files present on disk
    for file_path in disk_files:
        _process_file(conn, root, file_path, db_map.get(file_path), embedder, summarizer)

    return deleted_count


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def _process_file(
    conn: sqlite3.Connection,
    root: Path,
    file_path: Path,
    file_info: tuple[int, float, str] | None,
    embedder,
    summarizer,
) -> None:
    """Parse, reconcile, summarise, embed, and store one file."""
    # Read bytes once for binary check + md5
    try:
        raw = file_path.read_bytes()
    except OSError:
        return

    # Skip binary files and unsupported extensions — clean up stale DB rows
    binary = b"\x00" in raw[:512]
    unsupported = file_path.suffix.lower() not in _SUPPORTED_EXTENSIONS
    if binary or unsupported:
        if file_info is not None:
            file_id, _, _ = file_info
            with conn:
                conn.execute("DELETE FROM mcp_rag_files WHERE id = ?", (file_id,))
        return

    # Compute fingerprint and skip if unchanged
    mtime = file_path.stat().st_mtime
    md5 = hashlib.md5(raw).hexdigest()
    if file_info is not None:
        file_id, stored_mtime, stored_md5 = file_info
        if stored_mtime == mtime and stored_md5 == md5:
            return

    # Parse into semantic units (may be [] for oversized SQL, empty files, etc.)
    units = parse_file(file_path)

    # Truncate units that exceed the token estimate threshold
    processed: list[SemanticUnit] = []
    for unit in units:
        if len(unit.content) // 4 > _MAX_TOKENS:
            print(
                f"Warning: {file_path.name}:{unit.unit_name!r} exceeds "
                f"{_MAX_TOKENS} estimated tokens; truncating.",
                file=sys.stderr,
            )
            unit = SemanticUnit(
                unit_type=unit.unit_type,
                unit_name=unit.unit_name,
                content=unit.content[:_MAX_CHARS],
                char_offset=unit.char_offset,
            )
        processed.append(unit)
    units = processed

    # Load existing DB units for unit-level reconciliation
    if file_info is not None:
        file_id, _, _ = file_info
        stored_rows = conn.execute(
            "SELECT id, unit_type, unit_name, content_md5, char_offset "
            "FROM mcp_rag_units WHERE file_id = ?",
            (file_id,),
        ).fetchall()
        existing = [StoredUnit(r[0], r[1], r[2], r[3], r[4]) for r in stored_rows]
    else:
        file_id = None
        existing = []

    _to_keep, to_add, to_delete = diff_units(existing, units)

    indexed_at = datetime.now(timezone.utc).isoformat()

    # All deletes + inserts for this file in one transaction
    with conn:
        if file_info is not None:
            conn.execute(
                "UPDATE mcp_rag_files SET mtime=?, md5=?, indexed_at=? WHERE id=?",
                (mtime, md5, indexed_at, file_id),
            )
        else:
            cur = conn.execute(
                "INSERT INTO mcp_rag_files (root, path, mtime, md5, indexed_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (str(root), str(file_path), mtime, md5, indexed_at),
            )
            file_id = cur.lastrowid

        # Delete stale units (cascade trigger removes their embeddings)
        for stored in to_delete:
            conn.execute("DELETE FROM mcp_rag_units WHERE id=?", (stored.id,))

        # Summarise, embed, and insert new or changed units
        for unit in to_add:
            summary = summarizer.summarize(unit)
            cur = conn.execute(
                "INSERT INTO mcp_rag_units "
                "(file_id, unit_type, unit_name, content, content_md5, summary, char_offset) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    file_id,
                    unit.unit_type,
                    unit.unit_name,
                    unit.content,
                    unit.content_md5,
                    summary,
                    unit.char_offset,
                ),
            )
            unit_id = cur.lastrowid
            embedding = embedder.embed(summary)
            emb_bytes = struct.pack(f"{len(embedding)}f", *embedding)
            conn.execute(
                "INSERT INTO mcp_rag_embeddings (unit_id, embedding) VALUES (?, ?)",
                (unit_id, emb_bytes),
            )
