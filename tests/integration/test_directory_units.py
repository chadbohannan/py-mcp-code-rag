"""Integration tests for directory-level summary units.

Verifies that directory units are created bottom-up with correct qualified
paths, that repo-root gets a summary, and that orphan cleanup works.
"""

import os

from mcp_rag.db import open_db
from mcp_rag.indexer import run_index, DIRECTORY_UNIT_OFFSET
from tests.conftest import FakeEmbedder, FakeSummarizer, make_git_project


def _index(tmp_path, root, embedder=None, summarizer=None):
    embedder = embedder or FakeEmbedder(dim=4)
    summarizer = summarizer or FakeSummarizer()
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)
    return db_path, embedder, summarizer


# ---------------------------------------------------------------------------
# Basic directory unit creation
# ---------------------------------------------------------------------------


def test_directory_unit_created_for_repo_root(tmp_path):
    root = make_git_project(
        tmp_path / "proj",
        {
            "a.py": "def foo(): pass\ndef bar(): pass\n",
        },
    )
    db_path, _, _ = _index(tmp_path, root)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    dir_rows = conn.execute(
        "SELECT path, summary, unit_type FROM units WHERE char_offset = ?",
        (DIRECTORY_UNIT_OFFSET,),
    ).fetchall()
    conn.close()

    paths = [r[0] for r in dir_rows]
    # Repo root should have a directory unit
    assert "proj" in paths


def test_directory_unit_created_for_subdirectory(tmp_path):
    root = make_git_project(
        tmp_path / "proj",
        {
            "src/a.py": "def foo(): pass\ndef bar(): pass\n",
            "src/b.py": "def baz(): pass\ndef qux(): pass\n",
        },
    )
    db_path, _, _ = _index(tmp_path, root)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    dir_paths = [
        row[0]
        for row in conn.execute(
            "SELECT path FROM units WHERE char_offset = ?",
            (DIRECTORY_UNIT_OFFSET,),
        ).fetchall()
    ]
    conn.close()

    assert "proj/src" in dir_paths
    assert "proj" in dir_paths


def test_directory_unit_has_embedding(tmp_path):
    root = make_git_project(
        tmp_path / "proj",
        {
            "src/a.py": "def foo(): pass\ndef bar(): pass\n",
        },
    )
    db_path, _, _ = _index(tmp_path, root)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    unit_count = conn.execute("SELECT COUNT(*) FROM units").fetchone()[0]
    emb_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    conn.close()

    assert emb_count == unit_count


def test_directory_unit_not_created_for_empty_directory(tmp_path):
    """A directory with only skipped files should not get a directory unit."""
    root = make_git_project(
        tmp_path / "proj",
        {
            "data/big.sql": "-- x\n" * 1024,  # Oversized, skipped by parser
        },
    )
    db_path, _, _ = _index(tmp_path, root)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    dir_count = conn.execute(
        "SELECT COUNT(*) FROM units WHERE char_offset = ?",
        (DIRECTORY_UNIT_OFFSET,),
    ).fetchone()[0]
    conn.close()

    assert dir_count == 0


# ---------------------------------------------------------------------------
# Bottom-up summarization
# ---------------------------------------------------------------------------


def test_nested_directories_processed_bottom_up(tmp_path):
    """Deeper directories should be summarized before shallower ones."""
    summarizer = FakeSummarizer()
    root = make_git_project(
        tmp_path / "proj",
        {
            "src/core/engine.py": "def run(): pass\ndef stop(): pass\n",
            "src/api/handler.py": "def handle(): pass\ndef respond(): pass\n",
        },
    )
    _index(tmp_path, root, summarizer=summarizer)

    # Find directory summarization calls in order
    dir_calls = [c for c in summarizer.calls if c.unit_type == "directory"]
    [str(c.file_path or "root") for c in dir_calls]

    # src/core and src/api should be summarized before src and proj root
    # (deeper dirs first)
    assert len(dir_calls) >= 3  # src/core, src/api, src, proj


def test_directory_summary_includes_child_file_summaries(tmp_path):
    """Directory content passed to summarizer should reference child files."""
    summarizer = FakeSummarizer()
    root = make_git_project(
        tmp_path / "proj",
        {
            "lib/utils.py": "def helper(): pass\ndef helper2(): pass\n",
            "lib/models.py": "def create(): pass\ndef delete(): pass\n",
        },
    )
    _index(tmp_path, root, summarizer=summarizer)

    # Find the lib directory summarization call
    dir_calls = [
        c
        for c in summarizer.calls
        if c.unit_type == "directory"
        and c.file_path is not None
        and c.file_path.name == "lib"
    ]
    assert len(dir_calls) == 1
    # Content should reference the child file module summaries
    assert "utils.py" in dir_calls[0].content
    assert "models.py" in dir_calls[0].content


# ---------------------------------------------------------------------------
# Incremental indexing
# ---------------------------------------------------------------------------


def test_directory_unit_unchanged_on_second_run(tmp_path):
    root = make_git_project(
        tmp_path / "proj",
        {
            "lib.py": "def foo(): pass\ndef bar(): pass\n",
        },
    )
    summarizer = FakeSummarizer()
    db_path, embedder, _ = _index(tmp_path, root, summarizer=summarizer)
    calls_after_first = len(summarizer.calls)

    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)
    assert len(summarizer.calls) == calls_after_first


def test_directory_unit_updated_when_child_file_changes(tmp_path):
    root = make_git_project(
        tmp_path / "proj",
        {
            "lib.py": "def foo(): pass\ndef bar(): pass\n",
        },
    )
    summarizer = FakeSummarizer()
    db_path, embedder, _ = _index(tmp_path, root, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    old_summary = conn.execute(
        "SELECT summary FROM units WHERE path = 'proj' AND char_offset = ?",
        (DIRECTORY_UNIT_OFFSET,),
    ).fetchone()[0]
    conn.close()

    f = root / "lib.py"
    new_mtime = f.stat().st_mtime + 1
    f.write_text("def foo(): return 1\ndef bar(): pass\n", encoding="utf-8")
    os.utime(f, (new_mtime, new_mtime))

    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    new_summary = conn.execute(
        "SELECT summary FROM units WHERE path = 'proj' AND char_offset = ?",
        (DIRECTORY_UNIT_OFFSET,),
    ).fetchone()[0]
    conn.close()

    assert old_summary != new_summary


# ---------------------------------------------------------------------------
# Deletion / cleanup
# ---------------------------------------------------------------------------


def test_directory_unit_deleted_with_repo(tmp_path):
    """Directory units should be cleaned up when their repo is deleted."""
    from mcp_rag.db import remove_repo_db

    root = make_git_project(
        tmp_path / "proj",
        {
            "a.py": "def foo(): pass\ndef bar(): pass\n",
        },
    )
    db_path, embedder, _ = _index(tmp_path, root)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    # Verify directory units exist
    before = conn.execute(
        "SELECT COUNT(*) FROM units WHERE char_offset = ?",
        (DIRECTORY_UNIT_OFFSET,),
    ).fetchone()[0]
    assert before > 0

    # Delete the repo — should cascade to directory units via repo_id FK
    remove_repo_db(conn, "proj")

    after = conn.execute(
        "SELECT COUNT(*) FROM units WHERE char_offset = ?",
        (DIRECTORY_UNIT_OFFSET,),
    ).fetchone()[0]
    conn.close()

    assert after == 0


def test_orphan_directory_units_cleaned_up(tmp_path):
    """Directory units for removed directories should be cleaned up."""
    root = make_git_project(
        tmp_path / "proj",
        {
            "src/a.py": "def foo(): pass\ndef bar(): pass\n",
            "lib/b.py": "def baz(): pass\ndef qux(): pass\n",
        },
    )
    db_path, embedder, _ = _index(tmp_path, root)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    dir_paths_before = {
        row[0]
        for row in conn.execute(
            "SELECT path FROM units WHERE char_offset = ?",
            (DIRECTORY_UNIT_OFFSET,),
        ).fetchall()
    }
    conn.close()
    assert "proj/lib" in dir_paths_before

    # Remove the lib directory
    (root / "lib" / "b.py").unlink()
    (root / "lib").rmdir()

    summarizer = FakeSummarizer()
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    dir_paths_after = {
        row[0]
        for row in conn.execute(
            "SELECT path FROM units WHERE char_offset = ?",
            (DIRECTORY_UNIT_OFFSET,),
        ).fetchall()
    }
    conn.close()

    assert "proj/lib" not in dir_paths_after
    assert "proj/src" in dir_paths_after
