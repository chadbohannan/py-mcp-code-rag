"""File discovery for mcp-rag.

Uses `git ls-files --cached --others --exclude-standard` when the root is
inside a git repo; falls back to pathlib.Path.walk with a hardcoded exclusion
list otherwise.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

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
