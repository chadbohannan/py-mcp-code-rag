"""
Integration tests for the mcp-rag index pipeline.

Uses real SQLite on disk (tmp_path), real file I/O, FakeEmbedder, and
FakeSummarizer.  No Anthropic API calls; no fastembed.
"""
import os
import textwrap
import time
from pathlib import Path

import pytest

from mcp_rag.db import ModelMismatchError, open_db
from mcp_rag.indexer import IndexAbortError, run_index
from tests.conftest import FakeEmbedder, FakeSummarizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=4)


@pytest.fixture
def summarizer() -> FakeSummarizer:
    return FakeSummarizer()


@pytest.fixture(autouse=True)
def set_api_key(monkeypatch):
    """Satisfy the ANTHROPIC_API_KEY startup check in all integration tests."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-tests")


# ---------------------------------------------------------------------------
# Startup guards
# ---------------------------------------------------------------------------

def test_index_aborts_without_anthropic_api_key(tmp_path, embedder, summarizer, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    root = tmp_path / "proj"
    root.mkdir()
    db_path = tmp_path / "index.db"
    with pytest.raises(IndexAbortError, match="ANTHROPIC_API_KEY"):
        run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)
    # No DB should have been created
    assert not db_path.exists()


def test_index_aborts_on_overlapping_roots(tmp_path, embedder, summarizer):
    parent = tmp_path / "proj"
    child = parent / "sub"
    parent.mkdir()
    child.mkdir()
    db_path = tmp_path / "index.db"
    with pytest.raises(IndexAbortError, match="overlap"):
        run_index([parent, child], db_path=db_path, embedder=embedder, summarizer=summarizer)


# ---------------------------------------------------------------------------
# First run — fresh database
# ---------------------------------------------------------------------------

def test_first_run_creates_db_file(tmp_path, embedder, summarizer):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "hello.py").write_text("def hello(): pass\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)
    assert db_path.exists()


def test_first_run_emits_warning_to_stderr(tmp_path, embedder, summarizer, capsys):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "x.py").write_text("x = 1\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)
    captured = capsys.readouterr()
    assert "No index found" in captured.err


def test_first_run_indexes_python_file(tmp_path, embedder, summarizer):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "service.py").write_text(
        textwrap.dedent("""\
            def process(data):
                return data.strip()
        """),
        encoding="utf-8",
    )
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    units = conn.execute(
        "SELECT unit_type, unit_name FROM mcp_rag_units"
    ).fetchall()
    conn.close()

    assert any(unit_type == "function" and unit_name == "process" for unit_type, unit_name in units)


def test_first_run_indexes_markdown_file(tmp_path, embedder, summarizer):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "README.md").write_text(
        "# Overview\n\nIntro text.\n\n# Usage\n\nUsage text.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    count = conn.execute("SELECT COUNT(*) FROM mcp_rag_units").fetchone()[0]
    conn.close()
    assert count >= 2


def test_first_run_indexes_sql_file_under_4kb(tmp_path, embedder, summarizer):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "query.sql").write_text("SELECT id FROM users;", encoding="utf-8")
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    units = conn.execute(
        "SELECT unit_type FROM mcp_rag_units WHERE unit_type = 'sql'"
    ).fetchall()
    conn.close()
    assert len(units) == 1


def test_first_run_skips_binary_file(tmp_path, embedder, summarizer):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 512)
    (root / "app.py").write_text("def run(): pass\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    # Only app.py units should be present; image.png produces no units
    files = conn.execute("SELECT path FROM mcp_rag_files").fetchall()
    conn.close()
    paths = [row[0] for row in files]
    assert not any("image.png" in p for p in paths)


def test_first_run_skips_oversized_sql(tmp_path, embedder, summarizer):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "big.sql").write_text("-- x\n" * 1024, encoding="utf-8")  # > 4 KB
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    sql_units = conn.execute(
        "SELECT COUNT(*) FROM mcp_rag_units WHERE unit_type = 'sql'"
    ).fetchone()[0]
    conn.close()
    assert sql_units == 0


def test_first_run_calls_summarizer_per_unit(tmp_path, embedder, summarizer):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "mod.py").write_text(
        "def alpha(): pass\ndef beta(): pass\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)
    # At least two units (alpha, beta)
    assert len(summarizer.calls) >= 2


def test_first_run_stores_summary_in_units(tmp_path, embedder, summarizer):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "mod.py").write_text("def compute(): return 1\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    rows = conn.execute("SELECT summary FROM mcp_rag_units WHERE unit_name = 'compute'").fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0][0].startswith("summary:")


def test_first_run_creates_embeddings_per_unit(tmp_path, embedder, summarizer):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "mod.py").write_text("def foo(): pass\ndef bar(): pass\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    unit_count = conn.execute("SELECT COUNT(*) FROM mcp_rag_units").fetchone()[0]
    emb_count = conn.execute("SELECT COUNT(*) FROM mcp_rag_embeddings").fetchone()[0]
    conn.close()
    assert emb_count == unit_count


def test_first_run_writes_file_fingerprint(tmp_path, embedder, summarizer):
    root = tmp_path / "proj"
    root.mkdir()
    f = root / "app.py"
    f.write_text("def run(): pass\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    row = conn.execute(
        "SELECT mtime, md5, indexed_at FROM mcp_rag_files"
    ).fetchone()
    conn.close()
    mtime, md5, indexed_at = row
    assert mtime > 0
    assert len(md5) == 32  # hex MD5
    assert indexed_at  # ISO-8601 string


# ---------------------------------------------------------------------------
# Incremental run — unchanged files
# ---------------------------------------------------------------------------

def test_second_run_unchanged_file_no_resummarize(tmp_path, embedder, summarizer):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "stable.py").write_text("def stable(): pass\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)
    calls_after_first = len(summarizer.calls)

    # Run again without touching the file
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)
    assert len(summarizer.calls) == calls_after_first


def test_second_run_unchanged_file_same_embedding_count(tmp_path, embedder, summarizer):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "stable.py").write_text("def stable(): pass\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    count_after_first = conn.execute("SELECT COUNT(*) FROM mcp_rag_embeddings").fetchone()[0]
    conn.close()

    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    count_after_second = conn.execute("SELECT COUNT(*) FROM mcp_rag_embeddings").fetchone()[0]
    conn.close()

    assert count_after_first == count_after_second


# ---------------------------------------------------------------------------
# Incremental run — changed file
# ---------------------------------------------------------------------------

def test_changed_file_triggers_new_unit(tmp_path, embedder, summarizer):
    root = tmp_path / "proj"
    root.mkdir()
    f = root / "mod.py"
    f.write_text("def alpha(): pass\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    # Add a second function and update mtime
    time.sleep(0.01)
    f.write_text("def alpha(): pass\ndef beta(): pass\n", encoding="utf-8")
    # Force mtime change
    new_mtime = f.stat().st_mtime + 1
    os.utime(f, (new_mtime, new_mtime))

    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    names = [
        row[0]
        for row in conn.execute("SELECT unit_name FROM mcp_rag_units").fetchall()
    ]
    conn.close()
    assert "beta" in names


def test_changed_unit_content_triggers_resummarize(tmp_path, embedder, summarizer):
    root = tmp_path / "proj"
    root.mkdir()
    f = root / "mod.py"
    f.write_text("def greet(): return 'hello'\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)
    calls_after_first = len(summarizer.calls)

    new_mtime = f.stat().st_mtime + 1
    f.write_text("def greet(): return 'hi there'\n", encoding="utf-8")
    os.utime(f, (new_mtime, new_mtime))

    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)
    assert len(summarizer.calls) > calls_after_first


def test_unchanged_unit_in_changed_file_not_resummarized(tmp_path, embedder, summarizer):
    """
    When a file changes, units whose content_md5 is unchanged must NOT be
    re-summarized.
    """
    root = tmp_path / "proj"
    root.mkdir()
    f = root / "mod.py"
    # Two functions; we'll only change the second one
    f.write_text(
        "def stable(): return 1\n\ndef changing(): return 0\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    new_mtime = f.stat().st_mtime + 1
    f.write_text(
        "def stable(): return 1\n\ndef changing(): return 99\n",
        encoding="utf-8",
    )
    os.utime(f, (new_mtime, new_mtime))

    summarizer.calls.clear()
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    resumed_names = [u.unit_name for u in summarizer.calls]
    assert "stable" not in resumed_names
    assert "changing" in resumed_names


def test_changed_file_updates_fingerprint(tmp_path, embedder, summarizer):
    root = tmp_path / "proj"
    root.mkdir()
    f = root / "mod.py"
    f.write_text("x = 1\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    old_md5 = conn.execute("SELECT md5 FROM mcp_rag_files").fetchone()[0]
    conn.close()

    new_mtime = f.stat().st_mtime + 1
    f.write_text("x = 2\n", encoding="utf-8")
    os.utime(f, (new_mtime, new_mtime))

    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    new_md5 = conn.execute("SELECT md5 FROM mcp_rag_files").fetchone()[0]
    conn.close()
    assert old_md5 != new_md5


# ---------------------------------------------------------------------------
# Deleted-file reconciliation
# ---------------------------------------------------------------------------

def test_deleted_file_removed_from_db(tmp_path, embedder, summarizer):
    root = tmp_path / "proj"
    root.mkdir()
    f = root / "ephemeral.py"
    f.write_text("def temp(): pass\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    f.unlink()
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    count = conn.execute("SELECT COUNT(*) FROM mcp_rag_files").fetchone()[0]
    conn.close()
    assert count == 0


def test_deleted_file_cascades_to_units(tmp_path, embedder, summarizer):
    root = tmp_path / "proj"
    root.mkdir()
    f = root / "gone.py"
    f.write_text("def gone(): pass\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    f.unlink()
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    count = conn.execute("SELECT COUNT(*) FROM mcp_rag_units").fetchone()[0]
    conn.close()
    assert count == 0


def test_deleted_file_cascades_to_embeddings(tmp_path, embedder, summarizer):
    root = tmp_path / "proj"
    root.mkdir()
    f = root / "gone.py"
    f.write_text("def gone(): pass\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    f.unlink()
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    count = conn.execute("SELECT COUNT(*) FROM mcp_rag_embeddings").fetchone()[0]
    conn.close()
    assert count == 0


def test_deleted_file_count_logged_to_stderr(tmp_path, embedder, summarizer, capsys):
    root = tmp_path / "proj"
    root.mkdir()
    f = root / "bye.py"
    f.write_text("def bye(): pass\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    f.unlink()
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    captured = capsys.readouterr()
    assert "reconciled" in captured.err
    assert "1" in captured.err


# ---------------------------------------------------------------------------
# Transactional integrity
# ---------------------------------------------------------------------------

def test_single_file_indexing_is_atomic(tmp_path, embedder, monkeypatch):
    """
    If the summarizer raises on the second unit, the DB must remain in its
    pre-run state (no partial inserts for that file).
    """
    root = tmp_path / "proj"
    root.mkdir()
    (root / "mod.py").write_text(
        "def first(): pass\ndef second(): pass\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "index.db"

    # First run succeeds
    good_summarizer = FakeSummarizer()
    run_index([root], db_path=db_path, embedder=embedder, summarizer=good_summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    units_before = conn.execute("SELECT COUNT(*) FROM mcp_rag_units").fetchone()[0]
    conn.close()

    # Modify the file so it gets re-indexed
    f = root / "mod.py"
    new_mtime = f.stat().st_mtime + 1
    f.write_text("def first(): return 1\ndef second(): return 2\n", encoding="utf-8")
    os.utime(f, (new_mtime, new_mtime))

    # Summarizer that raises on the second call
    class BombSummarizer:
        def __init__(self):
            self.calls = []
        def summarize(self, unit):
            self.calls.append(unit)
            if len(self.calls) == 2:
                raise RuntimeError("Simulated API failure")
            return "ok"

    bomb = BombSummarizer()
    with pytest.raises(RuntimeError, match="Simulated API failure"):
        run_index([root], db_path=db_path, embedder=embedder, summarizer=bomb)

    # DB must be in pre-run state
    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    units_after = conn.execute("SELECT COUNT(*) FROM mcp_rag_units").fetchone()[0]
    conn.close()
    assert units_after == units_before


# ---------------------------------------------------------------------------
# Large unit truncation
# ---------------------------------------------------------------------------

def test_large_unit_truncated_and_warned(tmp_path, embedder, summarizer, capsys):
    root = tmp_path / "proj"
    root.mkdir()
    # Create a function whose body > 8000 estimated tokens (len // 4 > 8000)
    big_body = "    x = 1\n" * 3400  # ~34,000 chars → ~8,500 tokens
    source = f"def huge():\n{big_body}"
    (root / "big.py").write_text(source, encoding="utf-8")
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    captured = capsys.readouterr()
    assert "truncat" in captured.err.lower() or "warning" in captured.err.lower()

    # The summarizer should have been called with truncated content
    assert len(summarizer.calls) >= 1
    called_content = summarizer.calls[-1].content
    assert len(called_content) <= 8000 * 4 + len("def huge():\n")  # generous bound


# ---------------------------------------------------------------------------
# Multi-root indexing
# ---------------------------------------------------------------------------

def test_multi_root_both_indexed_into_same_db(tmp_path, embedder, summarizer):
    root_a = tmp_path / "proj_a"
    root_b = tmp_path / "proj_b"
    root_a.mkdir()
    root_b.mkdir()
    (root_a / "a.py").write_text("def fa(): pass\n", encoding="utf-8")
    (root_b / "b.py").write_text("def fb(): pass\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    run_index([root_a, root_b], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    roots = {
        row[0]
        for row in conn.execute("SELECT DISTINCT root FROM mcp_rag_files").fetchall()
    }
    conn.close()
    assert str(root_a) in roots
    assert str(root_b) in roots


def test_multi_root_reconciliation_is_per_root(tmp_path, embedder, summarizer):
    root_a = tmp_path / "proj_a"
    root_b = tmp_path / "proj_b"
    root_a.mkdir()
    root_b.mkdir()
    fa = root_a / "a.py"
    fa.write_text("def fa(): pass\n", encoding="utf-8")
    (root_b / "b.py").write_text("def fb(): pass\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    run_index([root_a, root_b], db_path=db_path, embedder=embedder, summarizer=summarizer)

    # Delete the file in root_a only
    fa.unlink()
    run_index([root_a, root_b], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    remaining_roots = {
        row[0]
        for row in conn.execute("SELECT DISTINCT root FROM mcp_rag_files").fetchall()
    }
    conn.close()
    # root_b still has its file; root_a has none left
    assert str(root_b) in remaining_roots


# ---------------------------------------------------------------------------
# Reindex (--reindex flag)
# ---------------------------------------------------------------------------

def test_reindex_drops_and_recreates_embeddings_table(tmp_path, embedder, summarizer):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "mod.py").write_text("def foo(): pass\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    new_embedder = FakeEmbedder(dim=8)
    run_index(
        [root],
        db_path=db_path,
        embedder=new_embedder,
        summarizer=summarizer,
        reindex=True,
    )

    conn = open_db(db_path, embed_dim=8, embed_model="fake-model")
    dim_val = conn.execute(
        "SELECT value FROM mcp_rag_meta WHERE key = 'embed_dim'"
    ).fetchone()[0]
    conn.close()
    assert dim_val == "8"


def test_reindex_preserves_summaries(tmp_path, embedder, summarizer):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "mod.py").write_text("def foo(): pass\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    summary_before = conn.execute("SELECT summary FROM mcp_rag_units WHERE unit_name = 'foo'").fetchone()[0]
    conn.close()

    fresh_summarizer = FakeSummarizer()
    run_index(
        [root],
        db_path=db_path,
        embedder=embedder,
        summarizer=fresh_summarizer,
        reindex=True,
    )

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    summary_after = conn.execute("SELECT summary FROM mcp_rag_units WHERE unit_name = 'foo'").fetchone()[0]
    conn.close()

    # Summary unchanged; no new API calls for the unchanged file
    assert summary_after == summary_before
    assert len(fresh_summarizer.calls) == 0


def test_reindex_updates_meta_embed_model(tmp_path, summarizer):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "mod.py").write_text("def foo(): pass\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    embedder_a = FakeEmbedder(dim=4)
    run_index([root], db_path=db_path, embedder=embedder_a, summarizer=summarizer)

    embedder_b = FakeEmbedder(dim=8)
    run_index(
        [root],
        db_path=db_path,
        embedder=embedder_b,
        summarizer=summarizer,
        reindex=True,
    )

    conn = open_db(db_path, embed_dim=8, embed_model="fake-model")
    stored_dim = conn.execute(
        "SELECT value FROM mcp_rag_meta WHERE key = 'embed_dim'"
    ).fetchone()[0]
    conn.close()
    assert stored_dim == "8"
