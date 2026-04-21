"""CLI client for the code-rag REST API.

Talks to a running webui server over HTTP, producing plain-text output
suitable for agentic LLM consumption. Uses only stdlib — no extra deps.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request


def _url(base: str, path: str, params: dict | None = None) -> str:
    url = base.rstrip("/") + path
    if params:
        filtered = {}
        for k, v in params.items():
            if isinstance(v, list):
                for item in v:
                    filtered.setdefault(k, []).append(str(item))
            elif v is not None:
                filtered[k] = [str(v)]
        parts = []
        for k, vs in filtered.items():
            for v in vs:
                parts.append(f"{urllib.parse.quote(k)}={urllib.parse.quote(v)}")
        if parts:
            url += "?" + "&".join(parts)
    return url


def _get(base: str, path: str, params: dict | None = None) -> object:
    url = _url(base, path, params)
    try:
        with urllib.request.urlopen(url) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            detail = json.loads(body).get("detail", body)
        except Exception:
            detail = body
        print(f"error: {e.code} {detail}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"error: cannot reach server: {e.reason}", file=sys.stderr)
        sys.exit(1)


def _post(base: str, path: str, body: dict | None = None, params: dict | None = None) -> object:
    url = _url(base, path, params)
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        resp_body = e.read().decode(errors="replace")
        try:
            detail = json.loads(resp_body).get("detail", resp_body)
        except Exception:
            detail = resp_body
        print(f"error: {e.code} {detail}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"error: cannot reach server: {e.reason}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_search(base: str, query: str, top_k: int, globs: list[str]) -> None:
    params: dict = {"q": query, "top_k": top_k}
    if globs:
        params["globs"] = globs
    results = _get(base, "/api/search", params)
    for r in results:
        print(f"{r['score']:.4f}\t{r['path']}")
        print(f"  {r['summary']}")


def cmd_unit(base: str, path: str) -> None:
    result = _get(base, "/api/unit", {"path": path})
    print(f"# {result['path']}")
    if result.get("summary"):
        print(f"# {result['summary']}")
    print()
    print(result["content"])


def cmd_fetch(base: str, paths: list[str]) -> None:
    results = _post(base, "/api/units/fetch", {"paths": paths})
    for i, r in enumerate(results):
        if i > 0:
            print("\n---\n")
        print(f"# {r['path']}")
        if r.get("summary"):
            print(f"# {r['summary']}")
        print()
        print(r["content"])


def cmd_units(base: str, limit: int, globs: list[str]) -> None:
    params: dict = {"limit": limit}
    if globs:
        params["globs"] = globs
    results = _get(base, "/api/units", params)
    for r in results:
        print(f"{r['path']}\t{r['summary']}")


def cmd_files(base: str, globs: list[str]) -> None:
    params: dict = {}
    if globs:
        params["globs"] = globs
    results = _get(base, "/api/files", params)
    for f in results:
        print(f"{f['repo']}/{f['path']}\t{f['indexed_at']}")


def cmd_repos(base: str) -> None:
    results = _get(base, "/api/repos")
    for r in results:
        print(f"{r['name']}\t{r['root']}\t{r['added_at']}")


def cmd_status(base: str) -> None:
    data = _get(base, "/api/status")
    print(f"total_units: {data['total_units']}")
    print(f"embed_count: {data['embed_count']}")
    for r in data["repos"]:
        ts = r.get("last_indexed_at") or "never"
        print(f"  {r['repo']}\tfiles={r['file_count']}\tunits={r['unit_count']}\tlast_indexed={ts}")


def cmd_browse(base: str, path: str) -> None:
    results = _get(base, "/api/browse", {"path": path})
    for n in results:
        parts = [n["type"], n["name"], n["path"]]
        if n.get("unit_type"):
            parts.append(n["unit_type"])
        if n.get("summary"):
            parts.append(n["summary"])
        print("\t".join(parts))


def cmd_index_start(base: str, paths: list[str], reindex: bool) -> None:
    data = _post(base, "/api/index", {"paths": paths, "reindex": reindex})
    _print_job_status(data)


def cmd_index_status(base: str) -> None:
    data = _get(base, "/api/index/status")
    _print_job_status(data)


def cmd_index_cancel(base: str) -> None:
    data = _post(base, "/api/index/cancel")
    _print_job_status(data)


def cmd_clear_repo(base: str, repo: str) -> None:
    data = _post(base, "/api/clear_repo", params={"repo": repo})
    if data.get("ok"):
        print(f"cleared: {data.get('repo')}")
    else:
        print(json.dumps(data))


def _print_job_status(data: dict) -> None:
    print(f"running: {data['running']}")
    if data.get("last_result"):
        print(f"last_result: {data['last_result']}")
    if data.get("last_finished_at"):
        print(f"last_finished_at: {data['last_finished_at']}")
