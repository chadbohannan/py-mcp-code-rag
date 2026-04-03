"""Unit tests for mcp_rag.discovery — file enumeration logic.

These tests use real filesystem I/O (tmp_path) but no database or network.
Git-based tests require `git` to be in PATH.
"""

import subprocess
from pathlib import Path


from mcp_rag.discovery import discover_files


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_init(path: Path) -> None:
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        check=True,
        capture_output=True,
        cwd=str(path),
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        check=True,
        capture_output=True,
        cwd=str(path),
    )


def _git_add_commit(path: Path, file: Path) -> None:
    subprocess.run(
        ["git", "add", str(file)], check=True, capture_output=True, cwd=str(path)
    )
    subprocess.run(
        ["git", "commit", "-m", "init"],
        check=True,
        capture_output=True,
        cwd=str(path),
    )


# ---------------------------------------------------------------------------
# Git-based discovery
# ---------------------------------------------------------------------------


def test_discover_files_git_repo_returns_committed_file(tmp_path):
    _git_init(tmp_path)
    f = tmp_path / "main.py"
    f.write_text("print('hi')\n", encoding="utf-8")
    _git_add_commit(tmp_path, f)

    files = discover_files(tmp_path)
    assert f in files


def test_discover_files_git_excludes_gitignored(tmp_path):
    _git_init(tmp_path)
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("secret.txt\n", encoding="utf-8")
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret", encoding="utf-8")
    main = tmp_path / "main.py"
    main.write_text("pass\n", encoding="utf-8")
    _git_add_commit(tmp_path, gitignore)
    # main.py is untracked but not ignored → should appear via --others
    files = discover_files(tmp_path)
    assert secret not in files


def test_discover_files_returns_absolute_paths(tmp_path):
    _git_init(tmp_path)
    f = tmp_path / "a.py"
    f.write_text("x = 1\n", encoding="utf-8")
    _git_add_commit(tmp_path, f)

    files = discover_files(tmp_path)
    for path in files:
        assert path.is_absolute(), f"Expected absolute path, got: {path}"


def test_discover_files_untracked_not_ignored_included(tmp_path):
    """git ls-files --others includes untracked, non-ignored files."""
    _git_init(tmp_path)
    committed = tmp_path / "committed.py"
    committed.write_text("pass\n", encoding="utf-8")
    _git_add_commit(tmp_path, committed)
    untracked = tmp_path / "untracked.py"
    untracked.write_text("pass\n", encoding="utf-8")

    files = discover_files(tmp_path)
    assert untracked in files


# ---------------------------------------------------------------------------
# Non-git fallback
# ---------------------------------------------------------------------------


def test_discover_files_non_git_returns_files(tmp_path):
    f = tmp_path / "data.py"
    f.write_text("x = 1\n", encoding="utf-8")
    files = discover_files(tmp_path)
    assert f in files


def test_discover_files_non_git_excludes_pycache(tmp_path):
    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    cached = pycache / "module.cpython-312.pyc"
    cached.write_bytes(b"\x00" * 8)
    real = tmp_path / "module.py"
    real.write_text("pass\n", encoding="utf-8")

    files = discover_files(tmp_path)
    assert cached not in files
    assert real in files


def test_discover_files_non_git_excludes_dotgit(tmp_path):
    # Create a .git directory manually (not a real repo — just a directory)
    dotgit = tmp_path / ".git"
    dotgit.mkdir()
    git_file = dotgit / "config"
    git_file.write_text("[core]\n", encoding="utf-8")
    real = tmp_path / "app.py"
    real.write_text("pass\n", encoding="utf-8")

    files = discover_files(tmp_path)
    assert git_file not in files
    assert real in files


def test_discover_files_non_git_excludes_venv(tmp_path):
    venv = tmp_path / ".venv"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
    real = tmp_path / "app.py"
    real.write_text("pass\n", encoding="utf-8")

    files = discover_files(tmp_path)
    assert (venv / "pyvenv.cfg") not in files
    assert real in files


def test_discover_files_non_git_excludes_node_modules(tmp_path):
    nm = tmp_path / "node_modules"
    nm.mkdir()
    (nm / "lib.js").write_text("module.exports = {}\n", encoding="utf-8")
    real = tmp_path / "index.js"
    real.write_text("console.log('hi')\n", encoding="utf-8")

    files = discover_files(tmp_path)
    assert (nm / "lib.js") not in files


def test_discover_files_non_git_no_symlink_follow(tmp_path):
    # Create a dangling symlink; it should not raise and should not be returned
    link = tmp_path / "dangling.py"
    link.symlink_to(tmp_path / "does_not_exist.py")
    real = tmp_path / "real.py"
    real.write_text("pass\n", encoding="utf-8")

    # Should not raise
    files = discover_files(tmp_path)
    assert link not in files


def test_discover_files_non_git_absolute_paths(tmp_path):
    (tmp_path / "x.py").write_text("pass\n", encoding="utf-8")
    files = discover_files(tmp_path)
    for path in files:
        assert path.is_absolute()
