"""Integration tests for file/module-level summary units.

Verifies that module units are created with correct qualified paths,
non-empty summaries, and import context when files have multiple child units.
"""

import os
import textwrap
import time

from mcp_rag.db import open_db
from mcp_rag.indexer import run_index, MODULE_UNIT_OFFSET
from tests.conftest import FakeEmbedder, FakeSummarizer, make_git_project


def _index(tmp_path, root, embedder=None, summarizer=None):
    embedder = embedder or FakeEmbedder(dim=4)
    summarizer = summarizer or FakeSummarizer()
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)
    return db_path, embedder, summarizer


# ---------------------------------------------------------------------------
# Basic module unit creation
# ---------------------------------------------------------------------------


def test_module_unit_created_for_multi_unit_file(tmp_path):
    root = make_git_project(tmp_path / "proj", {
        "lib.py": "def foo(): pass\ndef bar(): pass\n",
    })
    db_path, embedder, _ = _index(tmp_path, root)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    module_rows = conn.execute(
        "SELECT path, summary, unit_type FROM units WHERE char_offset = ?",
        (MODULE_UNIT_OFFSET,),
    ).fetchall()
    conn.close()

    assert len(module_rows) == 1
    path, summary, unit_type = module_rows[0]
    assert path == "proj/lib.py"
    assert unit_type == "module"
    assert summary  # non-empty


def test_module_unit_skipped_for_single_unit_file(tmp_path):
    root = make_git_project(tmp_path / "proj", {
        "query.sql": "SELECT 1;",
    })
    db_path, embedder, _ = _index(tmp_path, root)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    module_count = conn.execute(
        "SELECT COUNT(*) FROM units WHERE char_offset = ?",
        (MODULE_UNIT_OFFSET,),
    ).fetchone()[0]
    conn.close()

    assert module_count == 0


def test_module_unit_has_embedding(tmp_path):
    root = make_git_project(tmp_path / "proj", {
        "lib.py": "def foo(): pass\ndef bar(): pass\n",
    })
    db_path, embedder, _ = _index(tmp_path, root)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    # Every unit (including module) should have an embedding
    unit_count = conn.execute("SELECT COUNT(*) FROM units").fetchone()[0]
    emb_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    conn.close()

    assert emb_count == unit_count


def test_module_unit_qualified_path_is_file_path(tmp_path):
    root = make_git_project(tmp_path / "proj", {
        "src/models.py": textwrap.dedent("""\
            class User:
                def __init__(self): pass
                def name(self): pass
        """),
    })
    db_path, embedder, _ = _index(tmp_path, root)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    module_path = conn.execute(
        "SELECT path FROM units WHERE char_offset = ?",
        (MODULE_UNIT_OFFSET,),
    ).fetchone()[0]
    conn.close()

    # Module path should be repo/relative/file.py with no :UnitName suffix
    assert module_path == "proj/src/models.py"
    assert ":" not in module_path


# ---------------------------------------------------------------------------
# Import context
# ---------------------------------------------------------------------------


def test_module_unit_includes_import_context(tmp_path):
    """The summarizer should receive import context in the module unit content."""
    summarizer = FakeSummarizer()
    root = make_git_project(tmp_path / "proj", {
        "utils.py": "def helper(): pass\ndef helper2(): pass\n",
        "main.py": "from utils import helper\ndef run(): pass\ndef start(): pass\n",
    })
    _index(tmp_path, root, summarizer=summarizer)

    # Find the module unit summarizer call for main.py
    module_calls = [c for c in summarizer.calls if c.unit_type == "module"]
    main_module = [c for c in module_calls if c.file_path and c.file_path.name == "main.py"]
    assert len(main_module) == 1
    # The content passed to the summarizer should reference the imported module
    assert "utils" in main_module[0].content


# ---------------------------------------------------------------------------
# Incremental indexing
# ---------------------------------------------------------------------------


def test_module_unit_unchanged_on_second_run(tmp_path):
    root = make_git_project(tmp_path / "proj", {
        "lib.py": "def foo(): pass\ndef bar(): pass\n",
    })
    summarizer = FakeSummarizer()
    db_path, embedder, _ = _index(tmp_path, root, summarizer=summarizer)
    calls_after_first = len(summarizer.calls)

    # Second run — nothing changed
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)
    assert len(summarizer.calls) == calls_after_first


def test_module_unit_updated_when_child_changes(tmp_path):
    root = make_git_project(tmp_path / "proj", {
        "lib.py": "def foo(): pass\ndef bar(): pass\n",
    })
    summarizer = FakeSummarizer()
    db_path, embedder, _ = _index(tmp_path, root, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    old_summary = conn.execute(
        "SELECT summary FROM units WHERE path = 'proj/lib.py' AND char_offset = ?",
        (MODULE_UNIT_OFFSET,),
    ).fetchone()[0]
    conn.close()

    # Modify a child unit
    f = root / "lib.py"
    new_mtime = f.stat().st_mtime + 1
    f.write_text("def foo(): return 1\ndef bar(): pass\n", encoding="utf-8")
    os.utime(f, (new_mtime, new_mtime))

    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    new_summary = conn.execute(
        "SELECT summary FROM units WHERE path = 'proj/lib.py' AND char_offset = ?",
        (MODULE_UNIT_OFFSET,),
    ).fetchone()[0]
    conn.close()

    # Module summary should have changed since child content changed
    assert old_summary != new_summary


# ---------------------------------------------------------------------------
# Cycle handling
# ---------------------------------------------------------------------------


def test_circular_imports_dont_hang(tmp_path):
    """Files with circular imports should both get module summaries."""
    root = make_git_project(tmp_path / "proj", {
        "a.py": "import b\ndef func_a1(): pass\ndef func_a2(): pass\n",
        "b.py": "import a\ndef func_b1(): pass\ndef func_b2(): pass\n",
    })
    db_path, embedder, _ = _index(tmp_path, root)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    module_paths = [
        row[0] for row in conn.execute(
            "SELECT path FROM units WHERE char_offset = ?",
            (MODULE_UNIT_OFFSET,),
        ).fetchall()
    ]
    conn.close()

    assert "proj/a.py" in module_paths
    assert "proj/b.py" in module_paths


# ---------------------------------------------------------------------------
# Deletion cascade
# ---------------------------------------------------------------------------


def test_module_unit_deleted_with_file(tmp_path):
    root = make_git_project(tmp_path / "proj", {
        "lib.py": "def foo(): pass\ndef bar(): pass\n",
    })
    db_path, embedder, _ = _index(tmp_path, root)

    (root / "lib.py").unlink()
    summarizer = FakeSummarizer()
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    count = conn.execute("SELECT COUNT(*) FROM units").fetchone()[0]
    conn.close()

    assert count == 0
