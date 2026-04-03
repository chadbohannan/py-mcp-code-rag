"""Indexing pipeline for mcp-rag.

Discovers files, parses them into SemanticUnits, summarises each unit,
embeds the summary, and writes everything to SQLite.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from tqdm import tqdm

import sqlite_vec

from mcp_rag.db import _DDL_EMBEDDINGS, _DDL_TRIGGER, open_db
from mcp_rag.discovery import discover_files
from mcp_rag.models import (
    Embedder,
    SemanticUnit,
    Summarizer,
    encode_embedding,
)
from mcp_rag.parsers import parse_file
from mcp_rag.reconcile import StoredUnit, diff_units

_MAX_TOKENS = 8000
_MAX_CHARS = _MAX_TOKENS * 4

_SUPPORTED_EXTENSIONS = frozenset({
    ".py", ".go", ".md", ".mdx", ".sql",
    ".c", ".h",
    ".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx", ".ino",
    ".js", ".jsx", ".mjs", ".cjs",
    ".ts", ".tsx", ".mts", ".cts",
    ".java",
})


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
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
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
        # Re-embed all existing units using the stored qualified path
        rows = conn.execute(
            "SELECT id, path, summary FROM mcp_rag_units"
        ).fetchall()
        for unit_id, path, summary in rows:
            embed_input = f"{path} | {summary}" if path else summary
            embedding = embedder.embed(embed_input)
            conn.execute(
                "INSERT INTO mcp_rag_embeddings (unit_id, embedding) VALUES (?, ?)",
                (unit_id, encode_embedding(embedding)),
            )

    return conn


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def _embed_text(unit: SemanticUnit, summary: str) -> str:
    """Build the string to embed: ``qualified_path | summary``."""
    qp = unit.qualified_path
    return f"{qp} | {summary}" if qp else summary


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

    # Identify files that have units with empty summaries so they are
    # re-processed even when their mtime/md5 are unchanged.
    empty_summary_file_ids: set[int] = {
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT f.id FROM mcp_rag_files f "
            "JOIN mcp_rag_units u ON u.file_id = f.id "
            "WHERE f.root = ? AND length(u.summary) = 0",
            (str(root),),
        ).fetchall()
    }

    # Remove entries for files that no longer exist on disk
    deleted_count = 0
    for path, (file_id, _, _) in db_map.items():
        if path not in disk_files:
            with conn:
                conn.execute("DELETE FROM mcp_rag_files WHERE id = ?", (file_id,))
            deleted_count += 1

    # Filter to parsable files
    parsable_files = sorted(
        f for f in disk_files if f.suffix.lower() in _SUPPORTED_EXTENSIONS
    )

    # Pre-scan: determine which parsable files actually need (re-)indexing
    disable_bars = not sys.stderr.isatty()
    needs_indexing: list[Path] = []
    with tqdm(
        total=len(parsable_files),
        desc="scanning",
        unit="file",
        file=sys.stderr,
        disable=disable_bars,
    ) as scan_bar:
        for file_path in parsable_files:
            try:
                raw = file_path.read_bytes()
            except OSError:
                scan_bar.update(1)
                continue
            if b"\x00" in raw[:512]:
                scan_bar.update(1)
                continue
            file_info = db_map.get(file_path)
            if file_info is not None:
                file_id, stored_mtime, stored_md5 = file_info
                mtime = file_path.stat().st_mtime
                md5 = hashlib.md5(raw).hexdigest()
                if stored_mtime == mtime and stored_md5 == md5 \
                        and file_id not in empty_summary_file_ids:
                    scan_bar.update(1)
                    continue
            needs_indexing.append(file_path)
            scan_bar.update(1)

    print(
        f"{len(needs_indexing)} files to index of "
        f"{len(parsable_files)} parsable ({len(disk_files)} total)",
        file=sys.stderr,
    )

    # Index only the files that need it
    with tqdm(
        total=len(needs_indexing),
        desc="indexing",
        unit="file",
        file=sys.stderr,
        position=0,
        disable=disable_bars,
    ) as file_bar:
        with tqdm(
            total=0,
            desc="units",
            unit="unit",
            file=sys.stderr,
            position=1,
            leave=False,
            disable=disable_bars,
        ) as unit_bar:
            for file_path in needs_indexing:
                file_bar.set_postfix(
                    file=file_path.name, status="scanning", refresh=True
                )
                _process_file(
                    conn,
                    root,
                    file_path,
                    db_map.get(file_path),
                    embedder,
                    summarizer,
                    file_bar,
                    unit_bar,
                )
                file_bar.update(1)

    return deleted_count


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------


def _backfill_empty_summaries(
    conn: sqlite3.Connection,
    file_id: int,
    file_path: Path,
    root: Path,
    embedder,
    summarizer,
    file_bar: tqdm | None,
    unit_bar: tqdm | None,
    _file_status,
) -> None:
    """Re-summarise and re-embed units whose summary is empty."""
    rows = conn.execute(
        "SELECT id, path, content, char_offset "
        "FROM mcp_rag_units WHERE file_id = ? AND length(summary) = 0",
        (file_id,),
    ).fetchall()

    if not rows:
        _file_status("unchanged")
        return

    _file_status(f"backfill {len(rows)} unit(s)")
    if unit_bar is not None:
        unit_bar.reset(total=len(rows))
        unit_bar.set_description(file_path.name)

    for unit_id, qpath, content, char_offset in rows:
        # Reconstruct a SemanticUnit for the summariser prompt
        parts = qpath.rsplit(":", 1)
        unit_name = parts[1] if len(parts) > 1 else None
        unit = SemanticUnit(
            unit_type="unit",
            unit_name=unit_name,
            content=content,
            char_offset=char_offset,
            file_path=file_path,
            root=root,
        )
        unit_label = unit_name or "unit"
        if unit_bar is not None:
            unit_bar.set_postfix(name=unit_label, refresh=True)
        try:
            summary = summarizer.summarize(unit)
        except subprocess.TimeoutExpired:
            tqdm.write(
                f"Warning: {file_path.name}:{unit_label!r} timed out — skipping.",
                file=sys.stderr,
            )
            if unit_bar is not None:
                unit_bar.update(1)
            continue

        embedding = embedder.embed(_embed_text(unit, summary))
        with conn:
            conn.execute(
                "UPDATE mcp_rag_units SET summary = ? WHERE id = ?",
                (summary, unit_id),
            )
            conn.execute(
                "DELETE FROM mcp_rag_embeddings WHERE unit_id = ?",
                (unit_id,),
            )
            conn.execute(
                "INSERT INTO mcp_rag_embeddings (unit_id, embedding) VALUES (?, ?)",
                (unit_id, encode_embedding(embedding)),
            )
        if unit_bar is not None:
            unit_bar.update(1)


def _process_file(
    conn: sqlite3.Connection,
    root: Path,
    file_path: Path,
    file_info: tuple[int, float, str] | None,
    embedder,
    summarizer,
    file_bar: tqdm | None = None,
    unit_bar: tqdm | None = None,
) -> None:
    """Parse, reconcile, summarise, embed, and store one file."""

    def _file_status(status: str) -> None:
        if file_bar is not None:
            file_bar.set_postfix(file=file_path.name, status=status, refresh=True)

    # Read bytes once for binary check + md5
    try:
        raw = file_path.read_bytes()
    except OSError:
        _file_status("unreadable")
        return

    # Skip binary files and unsupported extensions — clean up stale DB rows
    binary = b"\x00" in raw[:512]
    unsupported = file_path.suffix.lower() not in _SUPPORTED_EXTENSIONS
    if binary or unsupported:
        _file_status("skipped")
        if file_info is not None:
            file_id, _, _ = file_info
            with conn:
                conn.execute("DELETE FROM mcp_rag_files WHERE id = ?", (file_id,))
        return

    # Compute fingerprint — check whether file content changed
    mtime = file_path.stat().st_mtime
    md5 = hashlib.md5(raw).hexdigest()
    content_unchanged = False
    if file_info is not None:
        file_id, stored_mtime, stored_md5 = file_info
        if stored_mtime == mtime and stored_md5 == md5:
            content_unchanged = True

    # If content is unchanged, only backfill units with empty summaries
    if content_unchanged:
        file_id = file_info[0]  # type: ignore[index]
        _backfill_empty_summaries(
            conn, file_id, file_path, root, embedder, summarizer,
            file_bar, unit_bar, _file_status,
        )
        return

    # Parse into semantic units (may be [] for oversized SQL, empty files, etc.)
    _file_status("parsing")
    units = parse_file(file_path)
    for unit in units:
        unit.file_path = file_path
        unit.root = root

    # Truncate units that exceed the token estimate threshold
    processed: list[SemanticUnit] = []
    for unit in units:
        if len(unit.content) > _MAX_CHARS:
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
                file_path=file_path,
                root=root,
            )
        processed.append(unit)
    units = processed

    # Load existing DB units for unit-level reconciliation
    if file_info is not None:
        file_id, _, _ = file_info
        stored_rows = conn.execute(
            "SELECT id, path, content_md5, char_offset "
            "FROM mcp_rag_units WHERE file_id = ?",
            (file_id,),
        ).fetchall()
        existing = [StoredUnit(r[0], r[1], r[2], r[3]) for r in stored_rows]
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
        if unit_bar is not None:
            unit_bar.reset(total=len(to_add))
            unit_bar.set_description(file_path.name)
        _file_status(f"indexing {len(to_add)} unit(s)")
        for unit in to_add:
            unit_label = unit.unit_name or unit.unit_type
            if unit_bar is not None:
                unit_bar.set_postfix(type=unit.unit_type, name=unit_label, refresh=True)
            try:
                summary = summarizer.summarize(unit)
            except subprocess.TimeoutExpired:
                tqdm.write(
                    f"Warning: {file_path.name}:{unit_label!r} timed out — skipping unit.",
                    file=sys.stderr,
                )
                if unit_bar is not None:
                    unit_bar.update(1)
                continue
            cur = conn.execute(
                "INSERT INTO mcp_rag_units "
                "(file_id, path, content, content_md5, summary, char_offset) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    file_id,
                    unit.qualified_path,
                    unit.content,
                    unit.content_md5,
                    summary,
                    unit.char_offset,
                ),
            )
            unit_id = cur.lastrowid
            embedding = embedder.embed(_embed_text(unit, summary))
            conn.execute(
                "INSERT INTO mcp_rag_embeddings (unit_id, embedding) VALUES (?, ?)",
                (unit_id, encode_embedding(embedding)),
            )
            if unit_bar is not None:
                unit_bar.update(1)
