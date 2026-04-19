"""Singleton indexing job state — shared by WebSocket and REST handlers."""

from __future__ import annotations

import threading
from datetime import datetime, timezone

_lock = threading.Lock()
_running = False
_cancel: threading.Event | None = None
_last_result: str | None = None
_last_finished_at: str | None = None


def start() -> threading.Event | None:
    """Claim the job slot.

    Returns a cancel Event the caller must pass to run_index, or None if a job
    is already running.
    """
    global _running, _cancel
    with _lock:
        if _running:
            return None
        _cancel = threading.Event()
        _running = True
        return _cancel


def finish(result: str) -> None:
    global _running, _cancel, _last_result, _last_finished_at
    with _lock:
        _running = False
        _cancel = None
        _last_result = result
        _last_finished_at = datetime.now(timezone.utc).isoformat()


def cancel() -> bool:
    """Signal cancellation. Returns True if a job was running."""
    with _lock:
        if _cancel is not None:
            _cancel.set()
            return True
        return False


def status() -> dict:
    with _lock:
        return {
            "running": _running,
            "last_result": _last_result,
            "last_finished_at": _last_finished_at,
        }
