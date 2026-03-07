"""Unit tests for mcp_rag.db — database schema and connection helpers."""
import sqlite3
from pathlib import Path

import pytest

from mcp_rag.db import ModelMismatchError, open_db


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------

def test_open_db_creates_db_file(tmp_path):
    db_path = tmp_path / "index.db"
    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    conn.close()
    assert db_path.exists()


def test_open_db_creates_all_tables(tmp_path):
    db_path = tmp_path / "index.db"
    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'shadow') OR type='table'"
        )
    }
    conn.close()
    assert "mcp_rag_meta" in tables
    assert "mcp_rag_files" in tables
    assert "mcp_rag_units" in tables
    # sqlite-vec creates mcp_rag_embeddings as a virtual table
    vtables = {
        row[0]
        for row in sqlite3.connect(str(db_path)).execute(
            "SELECT name FROM sqlite_master"
        )
    }
    assert "mcp_rag_embeddings" in vtables


def test_open_db_sets_wal_mode(tmp_path):
    db_path = tmp_path / "index.db"
    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    conn.close()
    assert mode == "wal"


def test_open_db_stores_embed_model_in_meta(tmp_path):
    db_path = tmp_path / "index.db"
    conn = open_db(db_path, embed_dim=4, embed_model="my-model")
    value = conn.execute(
        "SELECT value FROM mcp_rag_meta WHERE key = 'embed_model'"
    ).fetchone()[0]
    conn.close()
    assert value == "my-model"


def test_open_db_stores_embed_dim_in_meta(tmp_path):
    db_path = tmp_path / "index.db"
    conn = open_db(db_path, embed_dim=8, embed_model="fake-model")
    value = conn.execute(
        "SELECT value FROM mcp_rag_meta WHERE key = 'embed_dim'"
    ).fetchone()[0]
    conn.close()
    assert value == "8"


def test_open_db_stores_schema_version_in_meta(tmp_path):
    db_path = tmp_path / "index.db"
    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    row = conn.execute(
        "SELECT value FROM mcp_rag_meta WHERE key = 'schema_version'"
    ).fetchone()
    conn.close()
    assert row is not None
    assert len(row[0]) > 0


def test_open_db_idempotent_on_existing_db(tmp_path):
    db_path = tmp_path / "index.db"
    conn1 = open_db(db_path, embed_dim=4, embed_model="fake-model")
    conn1.close()
    # Opening again with the same model must not raise and must not duplicate rows
    conn2 = open_db(db_path, embed_dim=4, embed_model="fake-model")
    count = conn2.execute(
        "SELECT COUNT(*) FROM mcp_rag_meta WHERE key = 'embed_model'"
    ).fetchone()[0]
    conn2.close()
    assert count == 1


# ---------------------------------------------------------------------------
# Model mismatch
# ---------------------------------------------------------------------------

def test_open_db_model_mismatch_raises(tmp_path):
    db_path = tmp_path / "index.db"
    conn = open_db(db_path, embed_dim=4, embed_model="model-a")
    conn.close()
    with pytest.raises(ModelMismatchError, match="--reindex"):
        open_db(db_path, embed_dim=4, embed_model="model-b")


def test_open_db_dim_mismatch_raises(tmp_path):
    db_path = tmp_path / "index.db"
    conn = open_db(db_path, embed_dim=4, embed_model="same-model")
    conn.close()
    with pytest.raises(ModelMismatchError, match="--reindex"):
        open_db(db_path, embed_dim=8, embed_model="same-model")


def test_model_mismatch_error_message_names_models(tmp_path):
    db_path = tmp_path / "index.db"
    conn = open_db(db_path, embed_dim=4, embed_model="old-model")
    conn.close()
    with pytest.raises(ModelMismatchError) as exc_info:
        open_db(db_path, embed_dim=4, embed_model="new-model")
    assert "old-model" in str(exc_info.value)
    assert "new-model" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Cascade deletes
# ---------------------------------------------------------------------------

def test_cascade_delete_units_on_file_delete(tmp_path):
    db_path = tmp_path / "index.db"
    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")

    conn.execute(
        "INSERT INTO mcp_rag_files (root, path, mtime, md5, indexed_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("/root", "/root/file.py", 1.0, "abc123", "2026-01-01T00:00:00Z"),
    )
    file_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO mcp_rag_units (file_id, unit_type, unit_name, content, content_md5, summary, char_offset) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (file_id, "function", "foo", "def foo(): pass", "md5", "summary", 0),
    )
    conn.commit()

    conn.execute("DELETE FROM mcp_rag_files WHERE id = ?", (file_id,))
    conn.commit()

    unit_count = conn.execute(
        "SELECT COUNT(*) FROM mcp_rag_units WHERE file_id = ?", (file_id,)
    ).fetchone()[0]
    conn.close()
    assert unit_count == 0


def test_cascade_delete_embeddings_on_unit_delete(tmp_path):
    db_path = tmp_path / "index.db"
    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")

    conn.execute(
        "INSERT INTO mcp_rag_files (root, path, mtime, md5, indexed_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("/root", "/root/file.py", 1.0, "abc123", "2026-01-01T00:00:00Z"),
    )
    file_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO mcp_rag_units (file_id, unit_type, unit_name, content, content_md5, summary, char_offset) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (file_id, "function", "bar", "def bar(): pass", "md5bar", "sumbar", 0),
    )
    unit_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    import struct
    vec_bytes = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)
    conn.execute(
        "INSERT INTO mcp_rag_embeddings (unit_id, embedding) VALUES (?, ?)",
        (unit_id, vec_bytes),
    )
    conn.commit()

    conn.execute("DELETE FROM mcp_rag_units WHERE id = ?", (unit_id,))
    conn.commit()

    emb_count = conn.execute(
        "SELECT COUNT(*) FROM mcp_rag_embeddings WHERE unit_id = ?", (unit_id,)
    ).fetchone()[0]
    conn.close()
    assert emb_count == 0
