"""Unit tests for mcp_rag.db — database schema and connection helpers."""

import sqlite3

import pytest

from mcp_rag.db import ModelMismatchError, open_db, upsert_repo, list_repos_db


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
    assert "metadata" in tables
    assert "repos" in tables
    assert "files" in tables
    assert "units" in tables
    # sqlite-vec creates embeddings as a virtual table
    vtables = {
        row[0]
        for row in sqlite3.connect(str(db_path)).execute(
            "SELECT name FROM sqlite_master"
        )
    }
    assert "embeddings" in vtables


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
        "SELECT value FROM metadata WHERE key = 'embed_model'"
    ).fetchone()[0]
    conn.close()
    assert value == "my-model"


def test_open_db_stores_embed_dim_in_meta(tmp_path):
    db_path = tmp_path / "index.db"
    conn = open_db(db_path, embed_dim=8, embed_model="fake-model")
    value = conn.execute(
        "SELECT value FROM metadata WHERE key = 'embed_dim'"
    ).fetchone()[0]
    conn.close()
    assert value == "8"


def test_open_db_stores_schema_version_in_meta(tmp_path):
    db_path = tmp_path / "index.db"
    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    row = conn.execute(
        "SELECT value FROM metadata WHERE key = 'schema_version'"
    ).fetchone()
    conn.close()
    assert row is not None
    assert row[0] == "2"


def test_open_db_idempotent_on_existing_db(tmp_path):
    db_path = tmp_path / "index.db"
    conn1 = open_db(db_path, embed_dim=4, embed_model="fake-model")
    conn1.close()
    # Opening again with the same model must not raise and must not duplicate rows
    conn2 = open_db(db_path, embed_dim=4, embed_model="fake-model")
    count = conn2.execute(
        "SELECT COUNT(*) FROM metadata WHERE key = 'embed_model'"
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
# Repo helpers
# ---------------------------------------------------------------------------


def test_upsert_repo_returns_id(tmp_path):
    db_path = tmp_path / "index.db"
    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    repo_id = upsert_repo(conn, "myrepo", "/path/to/myrepo")
    conn.close()
    assert isinstance(repo_id, int)
    assert repo_id > 0


def test_upsert_repo_idempotent_on_same_root(tmp_path):
    db_path = tmp_path / "index.db"
    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    id1 = upsert_repo(conn, "myrepo", "/path/to/myrepo")
    id2 = upsert_repo(conn, "myrepo", "/path/to/myrepo")
    conn.close()
    assert id1 == id2


def test_list_repos_db_returns_all(tmp_path):
    db_path = tmp_path / "index.db"
    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    upsert_repo(conn, "alpha", "/alpha")
    upsert_repo(conn, "beta", "/beta")
    conn.commit()
    repos = list_repos_db(conn)
    conn.close()
    names = [r["name"] for r in repos]
    assert "alpha" in names
    assert "beta" in names


# ---------------------------------------------------------------------------
# Cascade deletes
# ---------------------------------------------------------------------------


def _insert_repo(conn, name="test-repo", root="/root"):
    """Helper: insert a repo and return its id."""
    return upsert_repo(conn, name, root)


def test_cascade_delete_units_on_file_delete(tmp_path):
    db_path = tmp_path / "index.db"
    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")

    repo_id = _insert_repo(conn)
    conn.execute(
        "INSERT INTO files (repo_id, path, mtime, md5, indexed_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (repo_id, "file.py", 1.0, "abc123", "2026-01-01T00:00:00Z"),
    )
    file_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO units (file_id, path, content, content_md5, summary, "
        "unit_type, unit_name, char_offset) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (file_id, "file.py:foo", "def foo(): pass", "md5", "summary",
         "function", "foo", 0),
    )
    conn.commit()

    conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
    conn.commit()

    unit_count = conn.execute(
        "SELECT COUNT(*) FROM units WHERE file_id = ?", (file_id,)
    ).fetchone()[0]
    conn.close()
    assert unit_count == 0


def test_cascade_delete_embeddings_on_unit_delete(tmp_path):
    db_path = tmp_path / "index.db"
    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")

    repo_id = _insert_repo(conn)
    conn.execute(
        "INSERT INTO files (repo_id, path, mtime, md5, indexed_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (repo_id, "file.py", 1.0, "abc123", "2026-01-01T00:00:00Z"),
    )
    file_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO units (file_id, path, content, content_md5, summary, "
        "unit_type, unit_name, char_offset) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (file_id, "file.py:bar", "def bar(): pass", "md5bar", "sumbar",
         "function", "bar", 0),
    )
    unit_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    import struct

    vec_bytes = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)
    conn.execute(
        "INSERT INTO embeddings (unit_id, embedding) VALUES (?, ?)",
        (unit_id, vec_bytes),
    )
    conn.commit()

    conn.execute("DELETE FROM units WHERE id = ?", (unit_id,))
    conn.commit()

    emb_count = conn.execute(
        "SELECT COUNT(*) FROM embeddings WHERE unit_id = ?", (unit_id,)
    ).fetchone()[0]
    conn.close()
    assert emb_count == 0
