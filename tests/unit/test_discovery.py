"""Unit tests for mcp_rag.discovery — file and repo enumeration logic.

These tests use real filesystem I/O (tmp_path) but no database or network.
Git-based tests require `git` to be in PATH.
"""

import pytest

from mcp_rag.discovery import discover_files, discover_git_repos
from tests.conftest import git_init, git_add_commit


# ---------------------------------------------------------------------------
# Git-based file discovery
# ---------------------------------------------------------------------------


def test_discover_files_git_repo_returns_committed_file(tmp_path):
    git_init(tmp_path)
    f = tmp_path / "main.py"
    f.write_text("print('hi')\n", encoding="utf-8")
    git_add_commit(tmp_path)

    files = discover_files(tmp_path)
    assert f in files


def test_discover_files_git_excludes_gitignored(tmp_path):
    git_init(tmp_path)
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("secret.txt\n", encoding="utf-8")
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret", encoding="utf-8")
    main = tmp_path / "main.py"
    main.write_text("pass\n", encoding="utf-8")
    git_add_commit(tmp_path)
    # main.py is untracked but not ignored → should appear via --others
    files = discover_files(tmp_path)
    assert secret not in files


def test_discover_files_returns_absolute_paths(tmp_path):
    git_init(tmp_path)
    f = tmp_path / "a.py"
    f.write_text("x = 1\n", encoding="utf-8")
    git_add_commit(tmp_path)

    files = discover_files(tmp_path)
    for path in files:
        assert path.is_absolute(), f"Expected absolute path, got: {path}"


def test_discover_files_untracked_not_ignored_included(tmp_path):
    """git ls-files --others includes untracked, non-ignored files."""
    git_init(tmp_path)
    committed = tmp_path / "committed.py"
    committed.write_text("pass\n", encoding="utf-8")
    git_add_commit(tmp_path)
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


@pytest.mark.parametrize(
    "dirname",
    [
        ".terraform",
        ".webpack",
        ".cache",
        ".npm",
        "bower_components",
        "coverage",
        "lib-cov",
        ".Trash",
        ".Trashes",
    ],
)
def test_discover_files_non_git_excludes_vendor_dirs(tmp_path, dirname):
    excluded = tmp_path / dirname
    excluded.mkdir()
    (excluded / "some_file.txt").write_text("data\n", encoding="utf-8")
    real = tmp_path / "main.py"
    real.write_text("pass\n", encoding="utf-8")

    files = discover_files(tmp_path)
    assert (excluded / "some_file.txt") not in files
    assert real in files


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


# ---------------------------------------------------------------------------
# Git repo discovery — discover_git_repos
# ---------------------------------------------------------------------------


def test_discover_git_repos_inside_repo(tmp_path):
    """When path is inside a git repo, returns that single repo."""
    git_init(tmp_path)
    (tmp_path / "main.py").write_text("pass\n", encoding="utf-8")
    git_add_commit(tmp_path)

    repos = discover_git_repos(tmp_path)
    assert len(repos) == 1
    name, root, _desc = repos[0]
    assert root == tmp_path.resolve()


def test_discover_git_repos_name_is_basename(tmp_path):
    proj = tmp_path / "myproject"
    proj.mkdir()
    git_init(proj)
    (proj / "f.py").write_text("pass\n", encoding="utf-8")
    git_add_commit(proj)

    repos = discover_git_repos(proj)
    assert repos[0][0] == "myproject"


def test_discover_git_repos_finds_nested_repos(tmp_path):
    """BFS from parent dir finds child repos."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    for name in ("alpha", "beta"):
        r = workspace / name
        r.mkdir()
        git_init(r)
        (r / "f.py").write_text("pass\n", encoding="utf-8")
        git_add_commit(r)

    repos = discover_git_repos(workspace)
    names = {r[0] for r in repos}
    assert "alpha" in names
    assert "beta" in names


def test_discover_git_repos_name_collision_resolved(tmp_path):
    """Two repos with the same basename get disambiguated."""
    for parent in ("a", "b"):
        r = tmp_path / parent / "shared"
        r.mkdir(parents=True)
        git_init(r)
        (r / "f.py").write_text("pass\n", encoding="utf-8")
        git_add_commit(r)

    repos = discover_git_repos(tmp_path)
    names = [r[0] for r in repos]
    assert len(names) == 2
    assert len(set(names)) == 2  # all unique


def test_discover_git_repos_empty_for_non_repo_dir(tmp_path):
    """Dir with no repos returns empty list."""
    (tmp_path / "f.txt").write_text("hello\n", encoding="utf-8")
    repos = discover_git_repos(tmp_path)
    assert repos == []


def test_discover_git_repos_reads_description(tmp_path):
    proj = tmp_path / "myproj"
    proj.mkdir()
    git_init(proj)
    (proj / "f.py").write_text("pass\n", encoding="utf-8")
    git_add_commit(proj)

    # Write a custom description
    (proj / ".git" / "description").write_text("My cool project\n", encoding="utf-8")

    repos = discover_git_repos(proj)
    assert repos[0][2] == "My cool project"
