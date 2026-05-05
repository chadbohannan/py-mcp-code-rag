"""Tests for schema migration from v2 to v3."""

import sqlite3
import struct

import sqlite_vec

from mcp_rag.db import open_db


def _create_v2_db(path):
    """Create a v2 schema database manually."""
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode = WAL")

    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute(
        "CREATE TABLE repos (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, root TEXT NOT NULL UNIQUE, added_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE files (id INTEGER PRIMARY KEY, repo_id INTEGER NOT NULL REFERENCES repos(id) ON DELETE CASCADE, path TEXT NOT NULL, mtime REAL NOT NULL, md5 TEXT NOT NULL, indexed_at TEXT NOT NULL, UNIQUE (repo_id, path))"
    )
    conn.execute(
        "CREATE TABLE units (id INTEGER PRIMARY KEY, file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE, path TEXT NOT NULL, content TEXT NOT NULL, content_md5 TEXT NOT NULL, summary TEXT NOT NULL, unit_type TEXT NOT NULL, unit_name TEXT NOT NULL, char_offset INTEGER NOT NULL)"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE embeddings USING vec0 (unit_id INTEGER PRIMARY KEY, embedding FLOAT[4])"
    )
    conn.execute(
        "CREATE TRIGGER units_delete_cascade AFTER DELETE ON units FOR EACH ROW BEGIN DELETE FROM embeddings WHERE unit_id = OLD.id; END"
    )
    conn.executemany(
        "INSERT INTO metadata (key, value) VALUES (?, ?)",
        [("schema_version", "2"), ("embed_model", "fake-model"), ("embed_dim", "4")],
    )

    # Insert test data
    conn.execute(
        "INSERT INTO repos (name, root, added_at) VALUES ('testrepo', '/tmp/testrepo', '2026-01-01')"
    )
    conn.execute(
        "INSERT INTO files (repo_id, path, mtime, md5, indexed_at) VALUES (1, 'a.py', 1.0, 'abc', '2026-01-01')"
    )
    conn.execute(
        "INSERT INTO units (file_id, path, content, content_md5, summary, unit_type, unit_name, char_offset) VALUES (1, 'testrepo/a.py:foo', 'def foo(): pass', 'md5', 'summary', 'function', 'foo', 0)"
    )
    unit_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO embeddings (unit_id, embedding) VALUES (?, ?)",
        (unit_id, struct.pack("4f", 1.0, 0.0, 0.0, 0.0)),
    )
    conn.commit()
    conn.close()


def test_migration_v2_to_v3_adds_repo_id(tmp_path):
    db_path = tmp_path / "index.db"
    _create_v2_db(db_path)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")

    # Schema version should be updated
    version = conn.execute(
        "SELECT value FROM metadata WHERE key = 'schema_version'"
    ).fetchone()[0]
    assert version == "3"

    # Units should have repo_id column populated from files.repo_id
    row = conn.execute("SELECT repo_id, file_id FROM units").fetchone()
    assert row[0] == 1  # repo_id derived from files
    assert row[1] == 1  # file_id preserved
    conn.close()


def test_migration_preserves_embeddings(tmp_path):
    db_path = tmp_path / "index.db"
    _create_v2_db(db_path)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    emb_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    conn.close()
    assert emb_count == 1


def test_migration_preserves_cascade_trigger(tmp_path):
    db_path = tmp_path / "index.db"
    _create_v2_db(db_path)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")

    # Delete the unit — should cascade to embeddings via trigger
    conn.execute("DELETE FROM units WHERE id = 1")
    conn.commit()
    emb_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    conn.close()
    assert emb_count == 0


def test_migration_allows_null_file_id(tmp_path):
    """After migration, file_id should be nullable for directory units."""
    db_path = tmp_path / "index.db"
    _create_v2_db(db_path)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")

    # Insert a directory unit with NULL file_id
    conn.execute(
        "INSERT INTO units (repo_id, file_id, path, content, content_md5, "
        "summary, unit_type, unit_name, char_offset) "
        "VALUES (1, NULL, 'testrepo', '', 'md5dir', 'dir summary', "
        "'directory', '', -2)"
    )
    conn.commit()

    row = conn.execute(
        "SELECT file_id FROM units WHERE unit_type = 'directory'"
    ).fetchone()
    conn.close()
    assert row[0] is None
