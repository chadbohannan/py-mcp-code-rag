"""Singleton indexing job state — shared by WebSocket and REST handlers.

Supports a dynamic queue: paths can be enqueued while the worker is running.
The worker drains the queue one path at a time, picking up new entries as they
arrive.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

_lock = threading.Lock()
_running = False
_cancel: threading.Event | None = None
_last_result: str | None = None
_last_finished_at: str | None = None
_queue: list[Path] = []


def enqueue(paths: list[Path]) -> bool:
    """Add paths to the queue, deduplicating. Returns True if any were new."""
    added = False
    with _lock:
        existing = set(_queue)
        for p in paths:
            rp = p.resolve()
            if rp not in existing:
                _queue.append(rp)
                existing.add(rp)
                added = True
    return added


def dequeue() -> Path | None:
    """Pop the next path from the queue, or None if empty."""
    with _lock:
        return _queue.pop(0) if _queue else None


def pending() -> list[str]:
    """Return the current queue contents as strings."""
    with _lock:
        return [str(p) for p in _queue]


def remove_pending(path: str) -> bool:
    """Remove a path from the queue. Returns True if it was present."""
    rp = Path(path).resolve()
    with _lock:
        try:
            _queue.remove(rp)
            return True
        except ValueError:
            return False


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
    """Signal cancellation and clear the queue. Returns True if a job was running."""
    with _lock:
        _queue.clear()
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
            "queue": [str(p) for p in _queue],
        }
