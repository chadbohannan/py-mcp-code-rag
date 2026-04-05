"""Indexing pipeline for mcp-rag.

Discovers git repositories, parses files into SemanticUnits, summarises each
unit, embeds the summary, and writes everything to SQLite.

Two-pass pipeline:
  1. Discover git repos from the given path(s) and upsert into ``repos`` table.
  2. For each repo, incrementally index its files.
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

from mcp_rag.db import _DDL_EMBEDDINGS, _DDL_TRIGGER, open_db, upsert_repo
from mcp_rag.discovery import discover_files, discover_git_repos
from mcp_rag.imports import extract_and_resolve_imports
from mcp_rag.models import (
    Embedder,
    SemanticUnit,
    Summarizer,
    encode_embedding,
    relative_path,
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
    ".tf", ".tfvars",
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

    # First pass: discover git repos and upsert
    repos: list[tuple[int, str, Path]] = []  # (repo_id, repo_name, git_root)
    for root in resolved:
        discovered = discover_git_repos(root)
        if not discovered:
            print(
                f"Warning: no git repositories found under {root} — skipping.",
                file=sys.stderr,
            )
            continue
        for name, git_root, _description in discovered:
            repo_id = upsert_repo(conn, name, str(git_root))
            repos.append((repo_id, name, git_root))
    conn.commit()

    # Second pass: index each repo
    total_deleted = 0
    try:
        for repo_id, repo_name, git_root in repos:
            total_deleted += _index_repo(
                conn, repo_id, repo_name, git_root, embedder, summarizer
            )
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
        conn.execute("DROP TABLE IF EXISTS embeddings")
        conn.execute("DROP TRIGGER IF EXISTS units_delete_cascade")
        # Recreate with new dimension
        conn.execute(_DDL_EMBEDDINGS.format(dim=new_dim))
        conn.execute(_DDL_TRIGGER)
        # Update stored metadata
        conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('embed_dim', ?)",
            (str(new_dim),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('embed_model', ?)",
            (new_model,),
        )
        # Re-embed all existing units using the stored qualified path
        rows = conn.execute(
            "SELECT id, path, summary FROM units"
        ).fetchall()
        for unit_id, path, summary in rows:
            embed_input = f"{path} | {summary}" if path else summary
            embedding = embedder.embed(embed_input)
            conn.execute(
                "INSERT INTO embeddings (unit_id, embedding) VALUES (?, ?)",
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
# Import graph and topological sort
# ---------------------------------------------------------------------------


def _build_import_graph(
    parsable_files: list[Path],
    repo_root: Path,
    repo_files_set: set[Path],
) -> dict[Path, list[Path]]:
    """Build a file-level import graph for a set of parsable files.

    Returns a dict mapping each file to the list of repo files it imports.
    """
    graph: dict[Path, list[Path]] = {}
    parsable_set = set(parsable_files)
    for file_path in parsable_files:
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            graph[file_path] = []
            continue
        imports = extract_and_resolve_imports(
            file_path, source, repo_root, repo_files_set,
        )
        graph[file_path] = [p for p in imports if p in parsable_set]
    return graph


def _topological_sort(graph: dict[Path, list[Path]]) -> tuple[list[Path], set[Path]]:
    """Topological sort with cycle detection (iterative).

    Returns ``(sorted_files, cycle_members)`` where cycle_members are files
    involved in import cycles. Cycle members are included in sorted_files
    but their cyclic dependencies should use unit-level summaries instead
    of file-level summaries.

    Uses an explicit stack instead of recursion to avoid hitting Python's
    recursion limit on deep dependency chains.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[Path, int] = {node: WHITE for node in graph}
    order: list[Path] = []
    cycle_members: set[Path] = set()

    for start in graph:
        if color[start] != WHITE:
            continue
        # Stack entries: (node, iterator over deps, entered)
        # ``entered`` is False for the initial push (pre-visit) and True
        # when we've already coloured the node GRAY and are iterating deps.
        stack: list[tuple[Path, int]] = [(start, 0)]
        color[start] = GRAY
        while stack:
            node, dep_idx = stack[-1]
            deps = graph.get(node, [])
            if dep_idx < len(deps):
                stack[-1] = (node, dep_idx + 1)
                dep = deps[dep_idx]
                if dep not in color:
                    continue
                if color[dep] == GRAY:
                    cycle_members.add(node)
                    cycle_members.add(dep)
                elif color[dep] == WHITE:
                    color[dep] = GRAY
                    stack.append((dep, 0))
            else:
                stack.pop()
                color[node] = BLACK
                order.append(node)

    return order, cycle_members


# ---------------------------------------------------------------------------
# Module unit builder
# ---------------------------------------------------------------------------

MODULE_UNIT_OFFSET = -1  # Sentinel char_offset for module units


def _build_module_content(
    file_path: Path,
    repo_name: str,
    git_root: Path,
    child_summaries: list[tuple[str, str, str]],
    import_paths: list[Path],
    conn: sqlite3.Connection,
    cycle_members: set[Path],
) -> str:
    """Build the structured content for a module unit.

    ``child_summaries`` is a list of ``(unit_type, unit_name, summary)``
    for all child units in the file.

    ``import_paths`` is the list of resolved import file paths.
    """
    rel = str(relative_path(file_path, git_root))
    qpath = f"{repo_name}/{rel}" if repo_name else rel

    lines = [f"File: {qpath}"]

    if import_paths:
        import_strs = []
        for ip in import_paths:
            ip_rel = str(relative_path(ip, git_root))
            import_strs.append(f"{repo_name}/{ip_rel}" if repo_name else ip_rel)
        lines.append(f"Imports: {', '.join(import_strs)}")

    lines.append("")
    lines.append("Units in this file:")
    for unit_type, unit_name, summary in child_summaries:
        label = f"{unit_type} {unit_name}" if unit_name else unit_type
        lines.append(f"- {label}: {summary}")

    # Add imported module context
    if import_paths:
        lines.append("")
        lines.append("Imported module context:")
        for ip in import_paths:
            ip_rel = str(relative_path(ip, git_root))
            ip_qpath = f"{repo_name}/{ip_rel}" if repo_name else ip_rel

            if ip in cycle_members:
                # For cyclic imports, use child unit summaries instead
                _append_unit_summaries(conn, ip_qpath, lines)
            else:
                # Look up the module-level summary
                row = conn.execute(
                    "SELECT summary FROM units WHERE path = ? AND char_offset = ?",
                    (ip_qpath, MODULE_UNIT_OFFSET),
                ).fetchone()
                if row and row[0]:
                    lines.append(f"- {ip_qpath}: {row[0]}")
                else:
                    _append_unit_summaries(conn, ip_qpath, lines)

    return "\n".join(lines)


def _append_unit_summaries(
    conn: sqlite3.Connection, file_qpath: str, lines: list[str]
) -> None:
    """Append child unit summaries for a file as fallback context."""
    rows = conn.execute(
        "SELECT unit_type, unit_name, summary FROM units "
        "WHERE path LIKE ? AND char_offset != ?",
        (f"{file_qpath}:%", MODULE_UNIT_OFFSET),
    ).fetchall()
    if rows:
        parts = [f"{r[0]} {r[1]}: {r[2]}" for r in rows if r[2]]
        if parts:
            lines.append(f"- {file_qpath}: [{'; '.join(parts)}]")


# ---------------------------------------------------------------------------
# Per-repo indexing
# ---------------------------------------------------------------------------


def _index_repo(
    conn: sqlite3.Connection,
    repo_id: int,
    repo_name: str,
    git_root: Path,
    embedder,
    summarizer,
) -> int:
    """Index one git repository. Returns the number of deleted files."""
    disk_files: set[Path] = set(discover_files(git_root))

    rows = conn.execute(
        "SELECT id, path, mtime, md5 FROM files WHERE repo_id = ?",
        (repo_id,),
    ).fetchall()
    # DB stores relative paths; reconstruct absolute for comparison
    db_map: dict[Path, tuple[int, float, str]] = {
        (git_root / row[1]).resolve(): (row[0], row[2], row[3]) for row in rows
    }

    # Identify files that have units with empty summaries so they are
    # re-processed even when their mtime/md5 are unchanged.
    empty_summary_file_ids: set[int] = {
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT f.id FROM files f "
            "JOIN units u ON u.file_id = f.id "
            "WHERE f.repo_id = ? AND length(u.summary) = 0",
            (repo_id,),
        ).fetchall()
    }

    # Remove entries for files that no longer exist on disk
    deleted_count = 0
    for path, (file_id, _, _) in db_map.items():
        if path not in disk_files:
            with conn:
                conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
            deleted_count += 1

    # Filter to parsable files
    parsable_files = sorted(
        f for f in disk_files if f.suffix.lower() in _SUPPORTED_EXTENSIONS
    )

    # Build import graph and topological sort for dependency-ordered processing
    import_graph = _build_import_graph(parsable_files, git_root, disk_files)
    topo_order, cycle_members = _topological_sort(import_graph)

    # Pre-scan: determine which parsable files actually need (re-)indexing
    disable_bars = not sys.stderr.isatty()
    needs_indexing_set: set[Path] = set()
    with tqdm(
        total=len(parsable_files),
        desc=f"scanning ({repo_name})",
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
            needs_indexing_set.add(file_path)
            scan_bar.update(1)

    # Order files needing indexing by topological sort (leaves first)
    topo_set = set(topo_order)
    needs_indexing: list[Path] = [f for f in topo_order if f in needs_indexing_set]
    # Include any files not in the topo order (shouldn't happen, but safe)
    for f in needs_indexing_set:
        if f not in topo_set:
            needs_indexing.append(f)

    print(
        f"[{repo_name}] {len(needs_indexing)} files to index of "
        f"{len(parsable_files)} parsable ({len(disk_files)} total)",
        file=sys.stderr,
    )

    # Index only the files that need it, in dependency order
    with tqdm(
        total=len(needs_indexing),
        desc=f"indexing ({repo_name})",
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
                    repo_id,
                    repo_name,
                    git_root,
                    file_path,
                    db_map.get(file_path),
                    embedder,
                    summarizer,
                    file_bar,
                    unit_bar,
                    import_paths=import_graph.get(file_path, []),
                    cycle_members=cycle_members,
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
    repo_name: str,
    git_root: Path,
    embedder,
    summarizer,
    file_bar: tqdm | None,
    unit_bar: tqdm | None,
    _file_status,
) -> None:
    """Re-summarise and re-embed units whose summary is empty."""
    rows = conn.execute(
        "SELECT id, path, content, char_offset "
        "FROM units WHERE file_id = ? AND length(summary) = 0",
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
            root=git_root,
            repo_name=repo_name,
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
                "UPDATE units SET summary = ? WHERE id = ?",
                (summary, unit_id),
            )
            conn.execute(
                "DELETE FROM embeddings WHERE unit_id = ?",
                (unit_id,),
            )
            conn.execute(
                "INSERT INTO embeddings (unit_id, embedding) VALUES (?, ?)",
                (unit_id, encode_embedding(embedding)),
            )
        if unit_bar is not None:
            unit_bar.update(1)


def _upsert_module_unit(
    conn: sqlite3.Connection,
    file_id: int,
    file_path: Path,
    repo_name: str,
    git_root: Path,
    import_paths: list[Path],
    cycle_members: set[Path],
    embedder,
    summarizer,
) -> None:
    """Create or update the module-level summary unit for a file.

    Skipped when the file has only one child unit (SQL, tfvars, etc.)
    since the single unit already serves as the file-level summary.
    """
    # Read all child unit summaries for this file
    child_rows = conn.execute(
        "SELECT unit_type, unit_name, summary FROM units "
        "WHERE file_id = ? AND char_offset != ?",
        (file_id, MODULE_UNIT_OFFSET),
    ).fetchall()

    # Skip if ≤1 child unit — the single unit already is the file summary
    if len(child_rows) <= 1:
        # Clean up any existing module unit
        conn.execute(
            "DELETE FROM units WHERE file_id = ? AND char_offset = ?",
            (file_id, MODULE_UNIT_OFFSET),
        )
        conn.commit()
        return

    child_summaries = [(r[0], r[1], r[2]) for r in child_rows]

    # Build the module unit content
    module_content = _build_module_content(
        file_path, repo_name, git_root,
        child_summaries, import_paths, conn, cycle_members,
    )

    # Build a SemanticUnit for the summarizer prompt (content is used only
    # for summarization and content_md5 change detection, not stored in DB).
    module_unit = SemanticUnit(
        unit_type="module",
        unit_name=None,
        content=module_content,
        char_offset=MODULE_UNIT_OFFSET,
        file_path=file_path,
        root=git_root,
        repo_name=repo_name,
    )

    # Check if an existing module unit has the same content_md5
    existing_module = conn.execute(
        "SELECT id, content_md5 FROM units "
        "WHERE file_id = ? AND char_offset = ?",
        (file_id, MODULE_UNIT_OFFSET),
    ).fetchone()

    if existing_module and existing_module[1] == module_unit.content_md5:
        return  # Module unit unchanged

    # Summarize and embed the module unit
    try:
        summary = summarizer.summarize(module_unit)
    except subprocess.TimeoutExpired:
        tqdm.write(
            f"Warning: {file_path.name} module summary timed out — skipping.",
            file=sys.stderr,
        )
        return

    with conn:
        # Delete old module unit if present
        if existing_module:
            conn.execute("DELETE FROM units WHERE id = ?", (existing_module[0],))

        cur = conn.execute(
            "INSERT INTO units "
            "(file_id, path, content, content_md5, summary, "
            "unit_type, unit_name, char_offset) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                file_id,
                module_unit.qualified_path,
                "",
                module_unit.content_md5,
                summary,
                module_unit.unit_type,
                "",
                MODULE_UNIT_OFFSET,
            ),
        )
        unit_id = cur.lastrowid
        embedding = embedder.embed(_embed_text(module_unit, summary))
        conn.execute(
            "INSERT INTO embeddings (unit_id, embedding) VALUES (?, ?)",
            (unit_id, encode_embedding(embedding)),
        )


def _process_file(
    conn: sqlite3.Connection,
    repo_id: int,
    repo_name: str,
    git_root: Path,
    file_path: Path,
    file_info: tuple[int, float, str] | None,
    embedder,
    summarizer,
    file_bar: tqdm | None = None,
    unit_bar: tqdm | None = None,
    import_paths: list[Path] | None = None,
    cycle_members: set[Path] | None = None,
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
                conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
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
            conn, file_id, file_path, repo_name, git_root, embedder, summarizer,
            file_bar, unit_bar, _file_status,
        )
        return

    # Parse into semantic units (may be [] for oversized SQL, empty files, etc.)
    _file_status("parsing")
    units = parse_file(file_path)
    for unit in units:
        unit.file_path = file_path
        unit.root = git_root
        unit.repo_name = repo_name

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
                root=git_root,
                repo_name=repo_name,
            )
        processed.append(unit)
    units = processed

    # Store file path relative to git root
    rel_path = str(file_path.resolve().relative_to(git_root.resolve()))

    # Load existing DB units for unit-level reconciliation
    # Exclude module units (char_offset = -1) — handled separately
    if file_info is not None:
        file_id, _, _ = file_info
        stored_rows = conn.execute(
            "SELECT id, path, content_md5, char_offset "
            "FROM units WHERE file_id = ? AND char_offset != ?",
            (file_id, MODULE_UNIT_OFFSET),
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
                "UPDATE files SET mtime=?, md5=?, indexed_at=? WHERE id=?",
                (mtime, md5, indexed_at, file_id),
            )
        else:
            cur = conn.execute(
                "INSERT INTO files (repo_id, path, mtime, md5, indexed_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (repo_id, rel_path, mtime, md5, indexed_at),
            )
            file_id = cur.lastrowid

        # Delete stale units (cascade trigger removes their embeddings)
        for stored in to_delete:
            conn.execute("DELETE FROM units WHERE id=?", (stored.id,))

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
                "INSERT INTO units "
                "(file_id, path, content, content_md5, summary, "
                "unit_type, unit_name, char_offset) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    file_id,
                    unit.qualified_path,
                    unit.content,
                    unit.content_md5,
                    summary,
                    unit.unit_type,
                    unit.unit_name or "",
                    unit.char_offset,
                ),
            )
            unit_id = cur.lastrowid
            embedding = embedder.embed(_embed_text(unit, summary))
            conn.execute(
                "INSERT INTO embeddings (unit_id, embedding) VALUES (?, ?)",
                (unit_id, encode_embedding(embedding)),
            )
            if unit_bar is not None:
                unit_bar.update(1)

    # Build and insert module-level summary unit after child units are committed.
    _upsert_module_unit(
        conn, file_id, file_path, repo_name, git_root,
        import_paths or [], cycle_members or set(),
        embedder, summarizer,
    )
