"""Shared fixtures for the webui integration tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mcp_rag.indexer import run_index
from mcp_rag.webui import create_app
from tests.conftest import FakeEmbedder, FakeSummarizer, make_git_project


REPO_FILES = {
    "src/app.py": (
        "def login(user):\n"
        "    return user\n"
        "\n"
        "def logout():\n"
        "    return None\n"
        "\n"
        "class Session:\n"
        "    def keep_alive(self):\n"
        "        return True\n"
    ),
    "src/db.py": (
        "def connect():\n    return None\n\ndef disconnect():\n    return None\n"
    ),
    "README.md": "# Overview\n\nProject docs for the test fixture repo.\n",
}


@pytest.fixture(scope="module")
def indexed_db(tmp_path_factory):
    """Build a real on-disk SQLite index once per test module."""
    db_path = tmp_path_factory.mktemp("idx") / "test.db"
    repo_root = tmp_path_factory.mktemp("repo") / "testrepo"
    make_git_project(repo_root, REPO_FILES)
    run_index(
        roots=[repo_root],
        db_path=db_path,
        embedder=FakeEmbedder(dim=4),
        summarizer=FakeSummarizer(),
    )
    return db_path


@pytest.fixture
def client(indexed_db):
    """fastapi.TestClient wired to the indexed DB.

    The summarizer_factory is supplied because create_app requires it, but no
    integration test in this file exercises the indexing endpoints.
    """
    app = create_app(
        db_path=indexed_db,
        embedder=FakeEmbedder(dim=4),
        summarizer_factory=lambda: FakeSummarizer(),
    )
    return TestClient(app)
