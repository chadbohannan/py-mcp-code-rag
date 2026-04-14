"""Web UI for code-rag — Starlette ASGI app with WebSocket indexing."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket

from mcp_rag.db import list_repos_db, open_db
from mcp_rag.discovery import read_git_description
from mcp_rag.indexer import IndexAbortError, run_index
from mcp_rag.models import Embedder, encode_embedding

_STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# Runtime state — set by create_app()
# ---------------------------------------------------------------------------

_db_path: Path | None = None
_embedder: Embedder | None = None
_summarizer_factory: callable | None = None  # () -> Summarizer

# Indexing state
_index_lock = threading.Lock()
_index_running = False
_cancel_event: threading.Event | None = None


def _get_read_conn() -> sqlite3.Connection:
    """Open a read-only connection for API queries."""
    assert _db_path is not None and _embedder is not None
    return open_db(_db_path, embed_dim=_embedder.dim, embed_model=_embedder.model)


# ---------------------------------------------------------------------------
# HTTP API handlers
# ---------------------------------------------------------------------------


async def index_page(request: Request) -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


async def api_search(request: Request) -> JSONResponse:
    q = request.query_params.get("q", "")
    if not q or _embedder is None or _db_path is None:
        return JSONResponse([])

    top_k = min(int(request.query_params.get("top_k", "5")), 20)
    globs_raw = request.query_params.get("globs", "")
    globs = [g.strip() for g in globs_raw.split(",") if g.strip()] or None

    emb = _embedder.embed(q)
    conn = _get_read_conn()
    try:
        if globs:
            glob_clauses = " AND ".join("u.path GLOB ?" for _ in globs)
            sql = f"""
                SELECT u.path, u.summary, sub.dist
                FROM (
                    SELECT e.unit_id, vec_distance_cosine(e.embedding, ?) AS dist
                    FROM embeddings e ORDER BY dist ASC LIMIT ?
                ) sub
                JOIN units u ON u.id = sub.unit_id
                WHERE {glob_clauses}
                ORDER BY sub.dist ASC
            """
            candidates = min(top_k * 10, 200)
            rows = conn.execute(
                sql, [encode_embedding(emb), candidates] + list(globs)
            ).fetchall()
            rows = rows[:top_k]
        else:
            sql = """
                SELECT u.path, u.summary,
                       vec_distance_cosine(e.embedding, ?) AS dist
                FROM embeddings e
                JOIN units u ON u.id = e.unit_id
                ORDER BY dist ASC LIMIT ?
            """
            rows = conn.execute(sql, (encode_embedding(emb), top_k)).fetchall()
    finally:
        conn.close()

    return JSONResponse([
        {"path": r[0], "summary": r[1], "score": round(1.0 - r[2] / 2.0, 6)}
        for r in rows
    ])


async def api_units(request: Request) -> JSONResponse:
    if _db_path is None or _embedder is None:
        return JSONResponse([])
    limit = min(int(request.query_params.get("limit", "100")), 500)
    globs_raw = request.query_params.get("globs", "")
    globs = [g.strip() for g in globs_raw.split(",") if g.strip()]

    conn = _get_read_conn()
    try:
        if globs:
            clauses = " AND ".join("u.path GLOB ?" for _ in globs)
            sql = f"SELECT u.path, u.summary FROM units u WHERE {clauses} ORDER BY u.path LIMIT ?"
            rows = conn.execute(sql, globs + [limit]).fetchall()
        else:
            rows = conn.execute(
                "SELECT u.path, u.summary FROM units u ORDER BY u.path LIMIT ?",
                (limit,),
            ).fetchall()
    finally:
        conn.close()

    return JSONResponse([{"path": r[0], "summary": r[1]} for r in rows])


async def api_unit(request: Request) -> JSONResponse:
    path = request.query_params.get("path", "")
    if not path or _db_path is None or _embedder is None:
        return JSONResponse([])

    conn = _get_read_conn()
    try:
        row = conn.execute(
            "SELECT u.path, u.content, u.summary FROM units u WHERE u.path = ?",
            (path,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return JSONResponse([])
    return JSONResponse({"path": row[0], "content": row[1], "summary": row[2]})


async def api_files(request: Request) -> JSONResponse:
    if _db_path is None or _embedder is None:
        return JSONResponse([])
    globs_raw = request.query_params.get("globs", "")
    globs = [g.strip() for g in globs_raw.split(",") if g.strip()]

    conn = _get_read_conn()
    try:
        base = (
            "SELECT r.name, f.path, f.indexed_at "
            "FROM files f JOIN repos r ON r.id = f.repo_id"
        )
        params: list = []
        if globs:
            clauses = " AND ".join(
                "(r.name || '/' || f.path) GLOB ?" for _ in globs
            )
            base += f" WHERE {clauses}"
            params = list(globs)
        base += " ORDER BY r.name, f.path"
        rows = conn.execute(base, params).fetchall()
    finally:
        conn.close()

    return JSONResponse([
        {"repo": r[0], "path": r[1], "indexed_at": r[2]} for r in rows
    ])


async def api_repos(request: Request) -> JSONResponse:
    if _db_path is None or _embedder is None:
        return JSONResponse([])

    conn = _get_read_conn()
    try:
        repos = list_repos_db(conn)
    finally:
        conn.close()

    for repo in repos:
        root = Path(repo["root"])
        repo["description"] = read_git_description(root)
    return JSONResponse(repos)


async def api_status(request: Request) -> JSONResponse:
    if _db_path is None or _embedder is None:
        return JSONResponse([])

    conn = _get_read_conn()
    try:
        rows = conn.execute("""
            SELECT
                r.name,
                COUNT(DISTINCT f.id) AS file_count,
                COUNT(u.id) AS unit_count,
                MAX(f.indexed_at) AS last_indexed_at
            FROM repos r
            JOIN files f ON f.repo_id = r.id
            LEFT JOIN units u ON u.file_id = f.id
            GROUP BY r.id
            ORDER BY r.name
        """).fetchall()
    finally:
        conn.close()

    return JSONResponse([
        {
            "repo": r[0], "file_count": r[1],
            "unit_count": r[2], "last_indexed_at": r[3],
        }
        for r in rows
    ])


# ---------------------------------------------------------------------------
# Filesystem browsing
# ---------------------------------------------------------------------------

# Directories to hide in the folder picker
_HIDDEN_DIRS = frozenset({
    ".git", ".venv", "venv", "__pycache__", "node_modules", ".tox",
    ".mypy_cache", ".pytest_cache", "dist", "build", ".eggs",
    ".terraform", ".webpack", ".cache", ".npm", "bower_components",
    "coverage", ".ruff_cache",
})


async def api_ls(request: Request) -> JSONResponse:
    """List directory contents for the folder picker.

    Returns dirs and an ``is_git`` flag when a ``.git`` folder is present.
    Only directories are returned (files are irrelevant for path selection).
    """
    raw = request.query_params.get("path", "")
    target = Path(raw) if raw else Path.home()

    try:
        target = target.resolve()
        if not target.is_dir():
            return JSONResponse({"error": "Not a directory"}, status_code=400)

        dirs: list[dict] = []
        is_git = False
        for entry in sorted(target.iterdir()):
            if not entry.is_dir():
                continue
            name = entry.name
            if name == ".git":
                is_git = True
                continue
            if name in _HIDDEN_DIRS or name.startswith("."):
                continue
            dirs.append({"name": name, "path": str(entry)})

        return JSONResponse({
            "path": str(target),
            "parent": str(target.parent) if target != target.parent else None,
            "is_git": is_git,
            "dirs": dirs,
        })
    except PermissionError:
        return JSONResponse({"error": "Permission denied"}, status_code=403)


# ---------------------------------------------------------------------------
# WebSocket — real-time indexing
# ---------------------------------------------------------------------------


async def ws_index(websocket: WebSocket) -> None:
    await websocket.accept()
    global _index_running, _cancel_event

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            action = msg.get("action")

            if action == "cancel":
                if _cancel_event is not None:
                    _cancel_event.set()
                    await websocket.send_json({"type": "status", "phase": "cancelling"})
                continue

            if action == "start":
                with _index_lock:
                    if _index_running:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Indexing is already running.",
                        })
                        continue
                    _index_running = True
                    _cancel_event = threading.Event()

                paths = [Path(p) for p in msg.get("paths", [])]
                reindex = msg.get("reindex", False)

                if not paths:
                    await websocket.send_json({
                        "type": "error", "message": "No paths provided.",
                    })
                    with _index_lock:
                        _index_running = False
                    continue

                # Validate paths
                bad = [str(p) for p in paths if not p.exists()]
                if bad:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Paths do not exist: {', '.join(bad)}",
                    })
                    with _index_lock:
                        _index_running = False
                    continue

                loop = asyncio.get_event_loop()
                cancel_ev = _cancel_event

                def progress_cb(event: dict) -> None:
                    """Send progress from indexer thread to WebSocket."""
                    asyncio.run_coroutine_threadsafe(
                        websocket.send_json(event), loop,
                    )

                def do_index() -> None:
                    global _index_running
                    try:
                        summarizer = _summarizer_factory()
                        if summarizer is None:
                            raise IndexAbortError(
                                "Summarizer not configured. Check --summarizer flag."
                            )
                        from mcp_rag.embedder import FastEmbedder
                        embedder = FastEmbedder(model_name=_embedder.model)
                        run_index(
                            roots=[p.resolve() for p in paths],
                            db_path=_db_path,
                            embedder=embedder,
                            summarizer=summarizer,
                            reindex=reindex,
                            progress_cb=progress_cb,
                            cancel_event=cancel_ev,
                        )
                    except IndexAbortError as exc:
                        progress_cb({"type": "error", "message": str(exc)})
                    except Exception as exc:
                        progress_cb({"type": "error", "message": str(exc)})
                    finally:
                        with _index_lock:
                            _index_running = False

                await asyncio.to_thread(do_index)

    except Exception:
        # WebSocket disconnect or other error
        pass
    finally:
        with _index_lock:
            if _index_running and _cancel_event is not None:
                _cancel_event.set()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    db_path: Path,
    embedder: Embedder,
    summarizer_factory: callable,
) -> Starlette:
    """Create the Starlette ASGI app."""
    global _db_path, _embedder, _summarizer_factory
    _db_path = db_path
    _embedder = embedder
    _summarizer_factory = summarizer_factory

    routes = [
        Route("/", index_page),
        Route("/api/search", api_search),
        Route("/api/units", api_units),
        Route("/api/unit", api_unit),
        Route("/api/files", api_files),
        Route("/api/repos", api_repos),
        Route("/api/status", api_status),
        Route("/api/ls", api_ls),
        WebSocketRoute("/ws/index", ws_index),
    ]

    return Starlette(routes=routes)
