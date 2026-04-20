"""File and repository discovery for mcp-rag.

Uses `git ls-files --cached --others --exclude-standard` when the root is
inside a git repo; falls back to pathlib.Path.walk with a hardcoded exclusion
list otherwise.

``discover_git_repos`` finds all git repositories accessible from a filesystem
root, resolving name collisions by prepending parent directory segments.
"""

from __future__ import annotations

import logging
import subprocess
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

_EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".eggs",
    ".terraform",
    ".webpack",
    ".cache",
    ".npm",
    "bower_components",
    "coverage",
    "lib-cov",
    ".Trash",
    ".Trashes",
}


class DiscoveryError(Exception):
    """Raised when file discovery fails unrecoverably."""


def discover_files(root: Path) -> list[Path]:
    """Return a list of absolute Paths to index under *root*.

    Tries git-based discovery first; falls back to filesystem walk.
    """
    root = root.resolve()
    try:
        return _git_discover(root)
    except DiscoveryError:
        return _walk_discover(root)


def discover_git_repos(path: Path) -> list[tuple[str, Path, str]]:
    """Discover git repositories accessible from *path*.

    Returns ``[(name, git_root, description), ...]``.

    If *path* is inside a git repository, returns that single repo.
    Otherwise walks the directory tree for ``.git`` directories.

    Name collisions (two repos with the same basename) are resolved by
    prepending parent directory segments with ``-`` as separator until
    names are unique.
    """
    path = path.resolve()

    # Check if path is inside a git repo.
    enclosing = _git_toplevel(path)
    if enclosing is not None:
        roots = [enclosing]
    else:
        roots = _find_git_roots(path)

    # Build (name, root, description) tuples with unique names.
    return _assign_unique_names(roots)


def _git_toplevel(path: Path) -> Path | None:
    """Return the git toplevel for *path*, or None if not in a repo."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=str(path),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip()).resolve()


def _find_git_roots(path: Path) -> list[Path]:
    """BFS for directories containing ``.git`` under *path*."""
    roots: list[Path] = []
    for dirpath, dirnames, _ in path.walk(follow_symlinks=False):
        if ".git" in dirnames or (dirpath / ".git").is_file():
            roots.append(dirpath.resolve())
            # Don't descend into this repo.
            dirnames[:] = [d for d in dirnames if d != ".git"]
            # Remove subdirs that are inside this repo to avoid nested repos
            # being double-counted — but nested repos (submodules) are valid,
            # so we only prune .git itself.
        else:
            dirnames[:] = [d for d in dirnames if d not in _EXCLUDED_DIRS]
    return roots


def read_git_description(git_root: Path) -> str:
    """Read the ``.git/description`` file, returning its contents stripped."""
    desc_path = git_root / ".git" / "description"
    try:
        return desc_path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _assign_unique_names(
    roots: list[Path],
) -> list[tuple[str, Path, str]]:
    """Assign unique names to repos, disambiguating by parent dirs."""
    if not roots:
        return []

    # Start with basenames.
    names = [r.name for r in roots]

    # Find duplicates and resolve by prepending parent segments.
    dupes = {n for n, c in Counter(names).items() if c > 1}
    if dupes:
        depth = 1
        while dupes:
            for i, root in enumerate(roots):
                if names[i] in dupes:
                    parts = root.parts
                    # Prepend up to `depth` parent segments.
                    prefix_start = max(0, len(parts) - 1 - depth)
                    prefix = "-".join(parts[prefix_start : len(parts) - 1])
                    names[i] = f"{prefix}-{root.name}" if prefix else root.name
            dupes = {n for n, c in Counter(names).items() if c > 1}
            depth += 1
            if depth > 10:
                logger.warning(
                    "Could not fully disambiguate repo names after %d "
                    "levels; remaining duplicates: %s",
                    depth,
                    sorted(dupes),
                )
                break

    return [
        (name, root, read_git_description(root)) for name, root in zip(names, roots)
    ]


def _git_discover(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise DiscoveryError(f"git ls-files failed: {result.stderr.strip()}")

    paths: list[Path] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        p = (root / line).resolve()
        if p.is_file():
            paths.append(p)
    return paths


def _walk_discover(root: Path) -> list[Path]:
    paths: list[Path] = []
    for dirpath, dirnames, filenames in root.walk(follow_symlinks=False):
        # Prune excluded directories in-place so walk doesn't descend into them
        dirnames[:] = [d for d in dirnames if d not in _EXCLUDED_DIRS]
        for name in filenames:
            p = dirpath / name
            if p.is_symlink():
                # Skip symlinks (including dangling ones)
                continue
            paths.append(p.resolve())
    return paths
