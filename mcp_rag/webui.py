"""Web UI and REST API for code-rag — FastAPI ASGI app with WebSocket indexing."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Query, WebSocket
from fastapi.responses import FileResponse

from mcp_rag.api_models import (
    BrowseNode,
    DirEntry,
    FetchRequest,
    FileEntry,
    IndexRequest,
    IndexStatus,
    IndexStatusRepo,
    JobStatus,
    LsResponse,
    RepoEntry,
    SearchResult,
    UnitDetail,
    UnitSummary,
)
from mcp_rag.db import open_db
from mcp_rag.indexer import DEFAULT_EXCLUDE_GLOBS, IndexAbortError, run_index
from mcp_rag.models import Embedder
from mcp_rag import job, queries

_STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# Runtime state — set by create_app()
# ---------------------------------------------------------------------------

_db_path: Path | None = None
_embedder: Embedder | None = None
_summarizer_factory: Callable | None = None
_exclude_globs: tuple[str, ...] = DEFAULT_EXCLUDE_GLOBS

_last_progress: dict | None = None
_ws_clients: dict[WebSocket, asyncio.AbstractEventLoop] = {}
_ws_clients_lock = threading.Lock()


def _get_read_conn() -> sqlite3.Connection:
    assert _db_path is not None and _embedder is not None
    return open_db(_db_path, embed_dim=_embedder.dim, embed_model=_embedder.model)


# ---------------------------------------------------------------------------
# Shared indexing logic
# ---------------------------------------------------------------------------


def _launch_index_job(
    paths: list[Path],
    reindex: bool,
    cancel_ev: threading.Event,
    progress_cb: Callable[[dict], None],
) -> None:
    """Start an index job in a daemon thread. Caller must have acquired job.start()."""

    def _run() -> None:
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
                exclude_globs=_exclude_globs,
            )
            job.finish("ok")
        except IndexAbortError as exc:
            job.finish(str(exc))
        except Exception as exc:
            progress_cb({"type": "error", "message": str(exc)})
            job.finish(str(exc))

    threading.Thread(target=_run, daemon=True).start()


# ---------------------------------------------------------------------------
# Filesystem browsing helpers
# ---------------------------------------------------------------------------

_HIDDEN_DIRS = frozenset(
    {
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
        ".ruff_cache",
    }
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="code-rag", version="0.1.0")


@app.get("/", include_in_schema=False)
async def index_page() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@app.get(
    "/api/search",
    response_model=list[SearchResult],
    summary="Search indexed code by natural-language query",
)
async def api_search(
    q: str = Query(..., description="Natural-language query"),
    top_k: int = Query(5, ge=1, le=20),
    globs: list[str] = Query(
        default=[], description="SQLite GLOB filters on qualified path"
    ),
) -> list[SearchResult]:
    if _embedder is None or _db_path is None:
        return []
    conn = _get_read_conn()
    try:
        return queries.search(conn, _embedder, q, top_k=top_k, globs=globs or None)
    finally:
        conn.close()




# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


@app.get(
    "/api/units",
    response_model=list[UnitSummary],
    summary="List semantic units (path + summary) with optional glob filter",
)
async def api_units(
    limit: int = Query(100, ge=1, le=500),
    globs: list[str] = Query(default=[]),
) -> list[UnitSummary]:
    if _db_path is None or _embedder is None:
        return []
    conn = _get_read_conn()
    try:
        return queries.list_units(conn, globs=globs or None, limit=limit)
    finally:
        conn.close()


@app.get(
    "/api/unit",
    response_model=UnitDetail,
    summary="Retrieve full source and summary for a single unit by qualified path",
)
async def api_unit(
    path: str = Query(
        ..., description="Qualified path, e.g. repo/file.py:Class:method"
    ),
) -> UnitDetail:
    if _db_path is None or _embedder is None:
        raise HTTPException(status_code=503, detail="Index not ready")
    conn = _get_read_conn()
    try:
        results = queries.get_units(conn, [path])
    finally:
        conn.close()
    if not results:
        raise HTTPException(status_code=404, detail="Unit not found")
    return results[0]


@app.post(
    "/api/units/fetch",
    response_model=list[UnitDetail],
    summary="Retrieve full source and summary for multiple units by qualified path",
)
async def api_units_fetch(body: FetchRequest) -> list[UnitDetail]:
    if _db_path is None or _embedder is None:
        raise HTTPException(status_code=503, detail="Index not ready")
    conn = _get_read_conn()
    try:
        return queries.get_units(conn, body.paths)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Files, repos, status
# ---------------------------------------------------------------------------


@app.get(
    "/api/files",
    response_model=list[FileEntry],
    summary="List indexed files with optional glob filter",
)
async def api_files(
    globs: list[str] = Query(default=[]),
) -> list[FileEntry]:
    if _db_path is None or _embedder is None:
        return []
    conn = _get_read_conn()
    try:
        return queries.list_files(conn, globs=globs or None)
    finally:
        conn.close()


@app.post(
    "/api/clear_repo",
    summary="Remove all indexed data for a repository by name",
)
async def api_clear_repo(
    repo: str = Query(..., description="Repository name to clear"),
) -> dict:
    if _db_path is None or _embedder is None:
        raise HTTPException(status_code=503, detail="Index not ready")
    conn = _get_read_conn()
    try:
        row = conn.execute("SELECT id FROM repos WHERE name = ?", (repo,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Repo not found")
        repo_id = row[0]
        conn.execute("DELETE FROM files WHERE repo_id = ?", (repo_id,))
        conn.execute("DELETE FROM units WHERE repo_id = ?", (repo_id,))
        conn.commit()
    finally:
        conn.close()
    return {"ok": True, "repo": repo}


@app.get(
    "/api/repos",
    response_model=list[RepoEntry],
    summary="List all indexed repositories",
)
async def api_repos() -> list[RepoEntry]:
    if _db_path is None or _embedder is None:
        return []
    conn = _get_read_conn()
    try:
        return queries.list_repos(conn)
    finally:
        conn.close()


@app.get(
    "/api/status",
    response_model=IndexStatus,
    summary="Index health: per-repo file/unit counts and last-indexed timestamp",
)
async def api_status() -> IndexStatus:
    if _db_path is None or _embedder is None:
        return IndexStatus(repos=[], total_units=0, embed_count=0)
    conn = _get_read_conn()
    try:
        raw = queries.index_status(conn)
    finally:
        conn.close()
    return IndexStatus(
        repos=[IndexStatusRepo(**r) for r in raw["repos"]],
        total_units=raw["total_units"],
        embed_count=raw["embed_count"],
    )


# ---------------------------------------------------------------------------
# Browse
# ---------------------------------------------------------------------------


@app.get(
    "/api/browse",
    response_model=list[BrowseNode],
    summary="Browse the index tree: repos → dirs → files → units",
)
async def api_browse(
    path: str = Query(
        default="",
        description="Qualified path prefix, e.g. repo, repo/dir, repo/file.py, repo/file.py:Class",
    ),
) -> list[BrowseNode]:
    if _db_path is None or _embedder is None:
        return []
    conn = _get_read_conn()
    try:
        return queries.browse(conn, path)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Filesystem picker
# ---------------------------------------------------------------------------


@app.get(
    "/api/ls",
    response_model=LsResponse,
    summary="List filesystem directories for the index path picker",
)
async def api_ls(
    path: str = Query(
        default="", description="Absolute filesystem path; defaults to home directory"
    ),
) -> LsResponse:
    target = Path(path) if path else Path.home()
    try:
        target = target.resolve()
        if not target.is_dir():
            raise HTTPException(status_code=400, detail="Not a directory")

        dirs: list[DirEntry] = []
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
            dirs.append(DirEntry(name=name, path=str(entry)))

        return LsResponse(
            path=str(target),
            parent=str(target.parent) if target != target.parent else None,
            is_git=is_git,
            dirs=dirs,
        )
    except HTTPException:
        raise
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")


# ---------------------------------------------------------------------------
# Indexing — REST
# ---------------------------------------------------------------------------


@app.post(
    "/api/index",
    response_model=JobStatus,
    status_code=202,
    summary="Start an indexing job (returns 409 if one is already running)",
)
async def api_index_start(body: IndexRequest) -> JobStatus:
    if _db_path is None or _embedder is None or _summarizer_factory is None:
        raise HTTPException(status_code=503, detail="Server not configured")

    paths = [Path(p) for p in body.paths]
    bad = [str(p) for p in paths if not Path(p).exists()]
    if bad:
        raise HTTPException(
            status_code=400, detail=f"Paths do not exist: {', '.join(bad)}"
        )

    cancel_ev = job.start()
    if cancel_ev is None:
        raise HTTPException(status_code=409, detail="Indexing already running")

    _launch_index_job(paths, body.reindex, cancel_ev, progress_cb=_broadcast)
    return JobStatus(**job.status())


@app.get(
    "/api/index/status",
    response_model=JobStatus,
    summary="Poll the current indexing job state",
)
async def api_index_status() -> JobStatus:
    return JobStatus(**job.status())


@app.post(
    "/api/index/cancel",
    response_model=JobStatus,
    summary="Signal the running indexing job to cancel",
)
async def api_index_cancel() -> JobStatus:
    job.cancel()
    return JobStatus(**job.status())


# ---------------------------------------------------------------------------
# WebSocket — real-time indexing with progress streaming
# ---------------------------------------------------------------------------


def _broadcast(event: dict) -> None:
    global _last_progress
    _last_progress = event
    with _ws_clients_lock:
        clients = list(_ws_clients.items())
    for ws, loop in clients:
        asyncio.run_coroutine_threadsafe(ws.send_json(event), loop)


@app.websocket("/ws/index")

async def ws_index(websocket: WebSocket) -> None:
    await websocket.accept()

    loop = asyncio.get_event_loop()
    with _ws_clients_lock:
        _ws_clients[websocket] = loop

    # Replay last known progress so a reconnecting client catches up.
    if job.status()["running"] and _last_progress is not None:
        await websocket.send_json(_last_progress)

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            action = msg.get("action")

            if action == "cancel":
                job.cancel()
                _broadcast({"type": "status", "phase": "cancelling"})
                continue

            if action == "start":
                paths = [Path(p) for p in msg.get("paths", [])]
                reindex = msg.get("reindex", False)

                if not paths:
                    await websocket.send_json(
                        {"type": "error", "message": "No paths provided."}
                    )
                    continue

                bad = [str(p) for p in paths if not p.exists()]
                if bad:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": f"Paths do not exist: {', '.join(bad)}",
                        }
                    )
                    continue

                cancel_ev = job.start()
                if cancel_ev is None:
                    await websocket.send_json(
                        {"type": "error", "message": "Indexing already running."}
                    )
                    continue

                _launch_index_job(paths, reindex, cancel_ev, _broadcast)

    except Exception:
        pass
    finally:
        with _ws_clients_lock:
            _ws_clients.pop(websocket, None)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    db_path: Path,
    embedder: Embedder,
    summarizer_factory: Callable,
    exclude_globs: tuple[str, ...] = DEFAULT_EXCLUDE_GLOBS,
) -> FastAPI:
    global _db_path, _embedder, _summarizer_factory, _exclude_globs
    _db_path = db_path
    _embedder = embedder
    _summarizer_factory = summarizer_factory
    _exclude_globs = exclude_globs
    return app
