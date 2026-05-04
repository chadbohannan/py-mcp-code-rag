"""
Integration tests for the mcp-rag index pipeline.

Uses real SQLite on disk (tmp_path), real file I/O, FakeEmbedder, and
FakeSummarizer.  No Anthropic API calls; no fastembed.

All test project directories are git-initialized so that discover_git_repos
can find them during the first pass of indexing.
"""

import os
import textwrap
import time

import pytest

from mcp_rag.db import open_db
from mcp_rag.indexer import DEFAULT_EXCLUDE_GLOBS, IndexAbortError, run_index
from tests.conftest import FakeEmbedder, FakeSummarizer, make_git_project


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=4)


@pytest.fixture
def summarizer() -> FakeSummarizer:
    return FakeSummarizer()




# ---------------------------------------------------------------------------
# Startup guards
# ---------------------------------------------------------------------------


def test_index_aborts_on_overlapping_roots(tmp_path, embedder, summarizer):
    parent = make_git_project(tmp_path / "proj")
    child = parent / "sub"
    child.mkdir()
    db_path = tmp_path / "index.db"
    with pytest.raises(IndexAbortError, match="overlap"):
        run_index(
            [parent, child], db_path=db_path, embedder=embedder, summarizer=summarizer
        )


# ---------------------------------------------------------------------------
# First run — fresh database
# ---------------------------------------------------------------------------


def test_first_run_creates_db_file(tmp_path, embedder, summarizer):
    root = make_git_project(tmp_path / "proj", {"hello.py": "def hello(): pass\n"})
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)
    assert db_path.exists()


def test_first_run_logs_new_index_message(tmp_path, embedder, summarizer):
    root = make_git_project(tmp_path / "proj", {"x.py": "x = 1\n"})
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)
    log_path = db_path.with_suffix(".log")
    assert log_path.exists()
    assert "No index found" in log_path.read_text()


def test_first_run_indexes_python_file(tmp_path, embedder, summarizer):
    root = make_git_project(tmp_path / "proj", {
        "service.py": textwrap.dedent("""\
            def process(data):
                return data.strip()
        """),
    })
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    paths = [
        row[0] for row in conn.execute("SELECT path FROM units").fetchall()
    ]
    conn.close()

    assert any("process" in p for p in paths)


def test_first_run_indexes_markdown_file(tmp_path, embedder, summarizer):
    root = make_git_project(tmp_path / "proj", {
        "README.md": "# Overview\n\nIntro text.\n\n# Usage\n\nUsage text.\n",
    })
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    count = conn.execute("SELECT COUNT(*) FROM units").fetchone()[0]
    conn.close()
    assert count >= 2


def test_first_run_indexes_sql_file_under_4kb(tmp_path, embedder, summarizer):
    root = make_git_project(tmp_path / "proj", {
        "query.sql": "SELECT id FROM users;",
    })
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    paths = [
        row[0] for row in conn.execute("SELECT path FROM units").fetchall()
    ]
    conn.close()
    assert any("query.sql" in p for p in paths)


def test_first_run_skips_binary_file(tmp_path, embedder, summarizer):
    root = make_git_project(tmp_path / "proj", {
        "app.py": "def run(): pass\n",
    })
    # Add binary file after init (won't be tracked but discover_files includes untracked)
    (root / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 512)
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    files = conn.execute("SELECT path FROM files").fetchall()
    conn.close()
    paths = [row[0] for row in files]
    assert not any("image.png" in p for p in paths)


def test_first_run_skips_oversized_sql(tmp_path, embedder, summarizer):
    root = make_git_project(tmp_path / "proj", {
        "big.sql": "-- x\n" * 1024,
    })
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    unit_count = conn.execute(
        "SELECT COUNT(*) FROM units"
    ).fetchone()[0]
    conn.close()
    assert unit_count == 0


def test_first_run_calls_summarizer_per_unit(tmp_path, embedder, summarizer):
    root = make_git_project(tmp_path / "proj", {
        "mod.py": "def alpha(): pass\ndef beta(): pass\n",
    })
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)
    # At least two units (alpha, beta)
    assert len(summarizer.calls) >= 2


def test_first_run_stores_summary_in_units(tmp_path, embedder, summarizer):
    root = make_git_project(tmp_path / "proj", {
        "mod.py": "def compute(): return 1\n",
    })
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    rows = conn.execute(
        "SELECT summary FROM units WHERE path LIKE '%compute'"
    ).fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0][0].startswith("summary:")


def test_first_run_creates_embeddings_per_unit(tmp_path, embedder, summarizer):
    root = make_git_project(tmp_path / "proj", {
        "mod.py": "def foo(): pass\ndef bar(): pass\n",
    })
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    unit_count = conn.execute("SELECT COUNT(*) FROM units").fetchone()[0]
    emb_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    conn.close()
    assert emb_count == unit_count


def test_first_run_writes_file_fingerprint(tmp_path, embedder, summarizer):
    root = make_git_project(tmp_path / "proj", {
        "app.py": "def run(): pass\n",
    })
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    row = conn.execute("SELECT mtime, md5, indexed_at FROM files").fetchone()
    conn.close()
    mtime, md5, indexed_at = row
    assert mtime > 0
    assert len(md5) == 32  # hex MD5
    assert indexed_at  # ISO-8601 string


def test_first_run_stores_repo_in_repos_table(tmp_path, embedder, summarizer):
    root = make_git_project(tmp_path / "proj", {
        "app.py": "def run(): pass\n",
    })
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    repos = conn.execute("SELECT name, root FROM repos").fetchall()
    conn.close()
    assert len(repos) == 1
    assert repos[0][0] == "proj"
    assert repos[0][1] == str(root)


def test_first_run_qualified_path_includes_repo_name(tmp_path, embedder, summarizer):
    root = make_git_project(tmp_path / "myrepo", {
        "lib.py": "def helper(): pass\n",
    })
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    paths = [
        row[0] for row in conn.execute("SELECT path FROM units").fetchall()
    ]
    conn.close()
    assert any(p.startswith("myrepo/") for p in paths)


# ---------------------------------------------------------------------------
# Incremental run — unchanged files
# ---------------------------------------------------------------------------


def test_second_run_unchanged_file_no_resummarize(tmp_path, embedder, summarizer):
    root = make_git_project(tmp_path / "proj", {
        "stable.py": "def stable(): pass\n",
    })
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)
    calls_after_first = len(summarizer.calls)

    # Run again without touching the file
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)
    assert len(summarizer.calls) == calls_after_first


def test_second_run_unchanged_file_same_embedding_count(tmp_path, embedder, summarizer):
    root = make_git_project(tmp_path / "proj", {
        "stable.py": "def stable(): pass\n",
    })
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    count_after_first = conn.execute(
        "SELECT COUNT(*) FROM embeddings"
    ).fetchone()[0]
    conn.close()

    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    count_after_second = conn.execute(
        "SELECT COUNT(*) FROM embeddings"
    ).fetchone()[0]
    conn.close()

    assert count_after_first == count_after_second


# ---------------------------------------------------------------------------
# Incremental run — changed file
# ---------------------------------------------------------------------------


def test_changed_file_triggers_new_unit(tmp_path, embedder, summarizer):
    root = make_git_project(tmp_path / "proj", {
        "mod.py": "def alpha(): pass\n",
    })
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    # Add a second function and update mtime
    f = root / "mod.py"
    time.sleep(0.01)
    f.write_text("def alpha(): pass\ndef beta(): pass\n", encoding="utf-8")
    # Force mtime change
    new_mtime = f.stat().st_mtime + 1
    os.utime(f, (new_mtime, new_mtime))

    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    paths = [
        row[0] for row in conn.execute("SELECT path FROM units").fetchall()
    ]
    conn.close()
    assert any("beta" in p for p in paths)


def test_changed_unit_content_triggers_resummarize(tmp_path, embedder, summarizer):
    root = make_git_project(tmp_path / "proj", {
        "mod.py": "def greet(): return 'hello'\n",
    })
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)
    calls_after_first = len(summarizer.calls)

    f = root / "mod.py"
    new_mtime = f.stat().st_mtime + 1
    f.write_text("def greet(): return 'hi there'\n", encoding="utf-8")
    os.utime(f, (new_mtime, new_mtime))

    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)
    assert len(summarizer.calls) > calls_after_first


def test_unchanged_unit_in_changed_file_not_resummarized(
    tmp_path, embedder, summarizer
):
    """
    When a file changes, units whose content_md5 is unchanged must NOT be
    re-summarized.
    """
    root = make_git_project(tmp_path / "proj", {
        "mod.py": "def stable(): return 1\n\ndef changing(): return 0\n",
    })
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    f = root / "mod.py"
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
    root = make_git_project(tmp_path / "proj", {
        "mod.py": "x = 1\n",
    })
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    old_md5 = conn.execute("SELECT md5 FROM files").fetchone()[0]
    conn.close()

    f = root / "mod.py"
    new_mtime = f.stat().st_mtime + 1
    f.write_text("x = 2\n", encoding="utf-8")
    os.utime(f, (new_mtime, new_mtime))

    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    new_md5 = conn.execute("SELECT md5 FROM files").fetchone()[0]
    conn.close()
    assert old_md5 != new_md5


# ---------------------------------------------------------------------------
# Deleted-file reconciliation
# ---------------------------------------------------------------------------


def test_deleted_file_cascades_to_db(tmp_path, embedder, summarizer):
    root = make_git_project(tmp_path / "proj", {
        "gone.py": "def gone(): pass\n",
    })
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    (root / "gone.py").unlink()
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM units").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 0
    conn.close()


def test_deleted_file_count_logged_to_stderr(tmp_path, embedder, summarizer, capsys):
    root = make_git_project(tmp_path / "proj", {
        "bye.py": "def bye(): pass\n",
    })
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    (root / "bye.py").unlink()
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
    root = make_git_project(tmp_path / "proj", {
        "mod.py": "def first(): pass\ndef second(): pass\n",
    })
    db_path = tmp_path / "index.db"

    # First run succeeds
    good_summarizer = FakeSummarizer()
    run_index([root], db_path=db_path, embedder=embedder, summarizer=good_summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    units_before = conn.execute("SELECT COUNT(*) FROM units").fetchone()[0]
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
    units_after = conn.execute("SELECT COUNT(*) FROM units").fetchone()[0]
    conn.close()
    assert units_after == units_before


# ---------------------------------------------------------------------------
# Large unit truncation
# ---------------------------------------------------------------------------


def test_large_unit_truncated_and_warned(tmp_path, embedder, summarizer, capsys):
    big_body = "    x = 1\n" * 3400  # ~34,000 chars → ~8,500 tokens
    source = f"def huge():\n{big_body}"
    root = make_git_project(tmp_path / "proj", {"big.py": source})
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    captured = capsys.readouterr()
    assert "truncat" in captured.err.lower() or "warning" in captured.err.lower()

    # The summarizer should have been called with truncated content
    assert len(summarizer.calls) >= 1
    called_content = summarizer.calls[-1].content
    assert len(called_content) <= 8000 * 4 + len("def huge():\n")  # generous bound


# ---------------------------------------------------------------------------
# Multi-repo indexing
# ---------------------------------------------------------------------------


def test_multi_repo_both_indexed_into_same_db(tmp_path, embedder, summarizer):
    root_a = make_git_project(tmp_path / "proj_a", {"a.py": "def fa(): pass\n"})
    root_b = make_git_project(tmp_path / "proj_b", {"b.py": "def fb(): pass\n"})
    db_path = tmp_path / "index.db"
    run_index(
        [root_a, root_b], db_path=db_path, embedder=embedder, summarizer=summarizer
    )

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    repos = {
        row[0]
        for row in conn.execute("SELECT DISTINCT name FROM repos").fetchall()
    }
    conn.close()
    assert "proj_a" in repos
    assert "proj_b" in repos


def test_multi_repo_reconciliation_is_per_repo(tmp_path, embedder, summarizer):
    root_a = make_git_project(tmp_path / "proj_a", {"a.py": "def fa(): pass\n"})
    root_b = make_git_project(tmp_path / "proj_b", {"b.py": "def fb(): pass\n"})
    db_path = tmp_path / "index.db"
    run_index(
        [root_a, root_b], db_path=db_path, embedder=embedder, summarizer=summarizer
    )

    # Delete the file in root_a only
    (root_a / "a.py").unlink()
    run_index(
        [root_a, root_b], db_path=db_path, embedder=embedder, summarizer=summarizer
    )

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    # proj_b should still have files
    remaining = conn.execute(
        "SELECT r.name FROM files f JOIN repos r ON r.id = f.repo_id"
    ).fetchall()
    conn.close()
    remaining_names = {row[0] for row in remaining}
    assert "proj_b" in remaining_names


# ---------------------------------------------------------------------------
# Reindex (--reindex flag)
# ---------------------------------------------------------------------------


def test_reindex_drops_and_recreates_embeddings_table(tmp_path, embedder, summarizer):
    root = make_git_project(tmp_path / "proj", {"mod.py": "def foo(): pass\n"})
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
        "SELECT value FROM metadata WHERE key = 'embed_dim'"
    ).fetchone()[0]
    conn.close()
    assert dim_val == "8"


def test_reindex_preserves_summaries(tmp_path, embedder, summarizer):
    root = make_git_project(tmp_path / "proj", {"mod.py": "def foo(): pass\n"})
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=4, embed_model="fake-model")
    summary_before = conn.execute(
        "SELECT summary FROM units WHERE path LIKE '%foo'"
    ).fetchone()[0]
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
    summary_after = conn.execute(
        "SELECT summary FROM units WHERE path LIKE '%foo'"
    ).fetchone()[0]
    conn.close()

    # Summary unchanged; no new API calls for the unchanged file
    assert summary_after == summary_before
    assert len(fresh_summarizer.calls) == 0


def test_reindex_updates_meta_embed_model(tmp_path, summarizer):
    root = make_git_project(tmp_path / "proj", {"mod.py": "def foo(): pass\n"})
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
        "SELECT value FROM metadata WHERE key = 'embed_dim'"
    ).fetchone()[0]
    conn.close()
    assert stored_dim == "8"


# ---------------------------------------------------------------------------
# Exclude globs
# ---------------------------------------------------------------------------


def test_exclude_globs_skips_generated_go_files(tmp_path, embedder, summarizer):
    root = make_git_project(tmp_path / "proj", {
        "main.go": "package main\n\nfunc main() {}\n",
        "api.pb.go": "package main\n\nfunc GeneratedStub() {}\n",
    })
    db_path = tmp_path / "index.db"
    run_index([root], db_path=db_path, embedder=embedder, summarizer=summarizer)

    conn = open_db(db_path, embed_dim=embedder.dim, embed_model=embedder.model)
    paths = [r[0] for r in conn.execute("SELECT path FROM units").fetchall()]
    conn.close()

    assert any("main.go" in p for p in paths)
    assert not any("api.pb.go" in p for p in paths)


def test_exclude_globs_empty_includes_generated_files(tmp_path, embedder, summarizer):
    root = make_git_project(tmp_path / "proj", {
        "main.go": "package main\n\nfunc main() {}\n",
        "api.pb.go": "package main\n\nfunc GeneratedStub() {}\n",
    })
    db_path = tmp_path / "index.db"
    run_index(
        [root], db_path=db_path, embedder=embedder, summarizer=summarizer,
        exclude_globs=(),
    )

    conn = open_db(db_path, embed_dim=embedder.dim, embed_model=embedder.model)
    paths = [r[0] for r in conn.execute("SELECT path FROM units").fetchall()]
    conn.close()

    assert any("main.go" in p for p in paths)
    assert any("api.pb.go" in p for p in paths)


def test_exclude_globs_custom_pattern(tmp_path, embedder, summarizer):
    root = make_git_project(tmp_path / "proj", {
        "app.py": "def run(): pass\n",
        "generated_client.py": "def stub(): pass\n",
    })
    db_path = tmp_path / "index.db"
    run_index(
        [root], db_path=db_path, embedder=embedder, summarizer=summarizer,
        exclude_globs=("generated_*.py",),
    )

    conn = open_db(db_path, embed_dim=embedder.dim, embed_model=embedder.model)
    paths = [r[0] for r in conn.execute("SELECT path FROM units").fetchall()]
    conn.close()

    assert any("app.py" in p for p in paths)
    assert not any("generated_client.py" in p for p in paths)
