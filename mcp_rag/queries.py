"""Shared data-access functions used by both the REST API and MCP server."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from mcp_rag.db import list_repos_db
from mcp_rag.discovery import read_git_description
from mcp_rag.models import Embedder, encode_embedding


def _glob_where(globs: list[str] | None, column: str = "u.path") -> tuple[str, list]:
    """Return a WHERE clause and params for optional GLOB filters.

    Multiple globs are AND'd: all must match.
    """
    if not globs:
        return "", []
    clauses = " AND ".join(f"{column} GLOB ?" for _ in globs)
    return f"WHERE {clauses}", list(globs)


def search(
    conn: sqlite3.Connection,
    embedder: Embedder,
    query: str,
    top_k: int = 5,
    globs: list[str] | None = None,
) -> list[dict]:
    k = min(top_k, 20)
    emb = embedder.embed(query)

    if globs:
        where, glob_params = _glob_where(globs)
        sql = f"""
            SELECT u.path, u.summary, sub.dist
            FROM (
                SELECT e.unit_id, vec_distance_cosine(e.embedding, ?) AS dist
                FROM embeddings e ORDER BY dist ASC LIMIT ?
            ) sub
            JOIN units u ON u.id = sub.unit_id
            {where}
            ORDER BY sub.dist ASC
        """
        candidates = min(k * 10, 200)
        rows = conn.execute(
            sql, [encode_embedding(emb), candidates] + glob_params
        ).fetchall()
        rows = rows[:k]
    else:
        sql = """
            SELECT u.path, u.summary,
                   vec_distance_cosine(e.embedding, ?) AS dist
            FROM embeddings e
            JOIN units u ON u.id = e.unit_id
            ORDER BY dist ASC LIMIT ?
        """
        rows = conn.execute(sql, (encode_embedding(emb), k)).fetchall()

    return [
        {"path": r[0], "summary": r[1], "score": round(1.0 - r[2] / 2.0, 6)}
        for r in rows
    ]


def get_units(conn: sqlite3.Connection, paths: list[str]) -> list[dict]:
    if not paths:
        return []
    placeholders = ",".join("?" for _ in paths)
    rows = conn.execute(
        f"SELECT u.path, u.content, u.summary FROM units u"
        f" WHERE u.path IN ({placeholders}) ORDER BY u.path",
        paths,
    ).fetchall()
    return [{"path": r[0], "content": r[1], "summary": r[2]} for r in rows]


def list_units(
    conn: sqlite3.Connection,
    globs: list[str] | None = None,
    limit: int = 100,
) -> list[dict]:
    capped = min(limit, 500)
    where, params = _glob_where(globs)
    sql = f"SELECT u.path, u.summary FROM units u {where} ORDER BY u.path LIMIT ?"
    params.append(capped)
    rows = conn.execute(sql, params).fetchall()
    return [{"path": r[0], "summary": r[1]} for r in rows]


def list_files(conn: sqlite3.Connection, globs: list[str] | None = None) -> list[dict]:
    base = (
        "SELECT r.name, f.path, f.indexed_at "
        "FROM files f JOIN repos r ON r.id = f.repo_id"
    )
    params: list = []
    if globs:
        clauses = " AND ".join("(r.name || '/' || f.path) GLOB ?" for _ in globs)
        base += f" WHERE {clauses}"
        params = list(globs)
    base += " ORDER BY r.name, f.path"
    rows = conn.execute(base, params).fetchall()
    return [{"repo": r[0], "path": r[1], "indexed_at": r[2]} for r in rows]


def list_repos(conn: sqlite3.Connection) -> list[dict]:
    repos = list_repos_db(conn)
    for repo in repos:
        repo["description"] = read_git_description(Path(repo["root"]))
    return repos


def index_status(conn: sqlite3.Connection) -> dict:
    rows = conn.execute("""
        SELECT
            r.name,
            r.root,
            COUNT(DISTINCT f.id) AS file_count,
            COUNT(u.id)          AS unit_count,
            MAX(f.indexed_at)    AS last_indexed_at
        FROM repos r
        JOIN files f ON f.repo_id = r.id
        LEFT JOIN units u ON u.file_id = f.id
        GROUP BY r.id
        ORDER BY r.name
    """).fetchall()
    embed_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

    repo_list = [
        {
            "repo": r[0],
            "root": r[1],
            "file_count": r[2],
            "unit_count": r[3],
            "last_indexed_at": r[4],
        }
        for r in rows
    ]
    return {
        "repos": repo_list,
        "total_units": sum(r["unit_count"] for r in repo_list),
        "embed_count": embed_count,
    }


def browse(conn: sqlite3.Connection, path: str) -> list[dict]:
    return _build_browse_nodes(conn, path)


def _build_browse_nodes(conn: sqlite3.Connection, path: str) -> list[dict]:
    if not path:
        repos = conn.execute("SELECT name FROM repos ORDER BY name").fetchall()
        result = []
        for (name,) in repos:
            row = conn.execute(
                "SELECT summary FROM units WHERE path = ? LIMIT 1", (name,)
            ).fetchone()
            result.append(
                {
                    "type": "repo",
                    "name": name,
                    "path": name,
                    "summary": row[0] if row else "",
                    "has_children": True,
                }
            )
        return result

    is_unit_path = ":" in path
    file_seg = path.split(":")[0].split("/")[-1]
    is_file_path = "." in file_seg

    prefix = path + ":" if (is_unit_path or is_file_path) else path + "/"

    all_rows = conn.execute(
        "SELECT path, summary, unit_type FROM units WHERE path = ? OR path LIKE ? ORDER BY path",
        (path, prefix + "%"),
    ).fetchall()

    all_paths = {r[0] for r in all_rows}
    nodes: list[dict] = []
    seen_dirs: dict[str, dict] = {}

    for row_path, summary, unit_type in all_rows:
        if row_path == path:
            nodes.append(
                {
                    "type": "self",
                    "unit_type": unit_type,
                    "name": "(overview)",
                    "path": path,
                    "summary": summary,
                    "has_children": False,
                }
            )
            continue

        if is_unit_path or is_file_path:
            rest = row_path[len(path) + 1 :]
            if ":" in rest:
                continue
            nodes.append(
                {
                    "type": "unit",
                    "unit_type": unit_type,
                    "name": rest,
                    "path": row_path,
                    "summary": summary,
                    "has_children": False,
                }
            )
        else:
            fp = row_path.split(":")[0]
            if not fp.startswith(prefix):
                continue
            rest = fp[len(prefix) :]
            next_seg = rest.split("/")[0]
            if not next_seg:
                continue
            child_path = prefix + next_seg
            if child_path not in seen_dirs:
                is_f = "." in next_seg
                seen_dirs[child_path] = {
                    "type": "file" if is_f else "dir",
                    "name": next_seg,
                    "path": child_path,
                    "summary": "",
                    "has_children": True,
                }
            entry = seen_dirs[child_path]
            if not entry["summary"]:
                if row_path == child_path:
                    entry["summary"] = summary
                elif (
                    entry["type"] == "file"
                    and row_path.split(":")[0] == child_path
                    and row_path.count(":") == 1
                    and summary
                ):
                    entry["summary"] = summary

    if is_file_path:
        for node in nodes:
            if node["type"] == "unit":
                class_prefix = path + ":" + node["name"] + ":"
                if any(p.startswith(class_prefix) for p in all_paths):
                    node["type"] = "class"
                    node["unit_type"] = node.get("unit_type", "class")
                    node["has_children"] = True

    nodes.extend(seen_dirs.values())

    type_order = {"self": 0, "dir": 1, "file": 2, "class": 3, "unit": 4}
    nodes.sort(key=lambda n: (type_order.get(n["type"], 5), n["name"]))
    return nodes
