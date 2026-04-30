#!/usr/bin/env python3
"""Standalone CLI client for the code-rag REST API.

Talks to a running webui server over HTTP, producing plain-text output
suitable for agentic LLM consumption. Uses only stdlib — no extra deps.

Usage:
    python code-rag-cli.py search "database connection"
    python code-rag-cli.py unit repo/file.py:Class:method
    python code-rag-cli.py --base-url http://host:9090 repos
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


def _url(base: str, path: str, params: dict | None = None) -> str:
    url = base.rstrip("/") + path
    if params:
        flat: list[tuple[str, str]] = []
        for k, v in params.items():
            if isinstance(v, list):
                flat.extend((k, str(item)) for item in v)
            elif v is not None:
                flat.append((k, str(v)))
        if flat:
            url += "?" + urllib.parse.urlencode(flat)
    return url


def _request(url: str, data: bytes | None = None) -> Any:
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
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


def _get(base: str, path: str, params: dict | None = None) -> Any:
    return _request(_url(base, path, params))


def _post(
    base: str, path: str, body: dict | None = None, params: dict | None = None
) -> Any:
    return _request(_url(base, path, params), json.dumps(body or {}).encode())


def _cmd_search(args: argparse.Namespace, base: str) -> None:
    params: dict = {"q": args.query, "top_k": args.top_k}
    if args.globs:
        params["globs"] = args.globs
    for r in _get(base, "/api/search", params):
        print(f"{r['score']:.4f}\t{r['path']}")
        print(f"  {r['summary']}")


def _print_unit(r: dict) -> None:
    print(f"# {r['path']}")
    if r.get("summary"):
        print(f"# {r['summary']}")
    print()
    print(r["content"])


def _cmd_unit(args: argparse.Namespace, base: str) -> None:
    _print_unit(_get(base, "/api/unit", {"path": args.path}))


def _cmd_fetch(args: argparse.Namespace, base: str) -> None:
    for i, r in enumerate(_post(base, "/api/units/fetch", {"paths": args.paths})):
        if i > 0:
            print("\n---\n")
        _print_unit(r)


def _cmd_units(args: argparse.Namespace, base: str) -> None:
    params: dict = {"limit": args.limit}
    if args.globs:
        params["globs"] = args.globs
    for r in _get(base, "/api/units", params):
        print(f"{r['path']}\t{r['summary']}")


def _cmd_files(args: argparse.Namespace, base: str) -> None:
    params: dict = {}
    if args.globs:
        params["globs"] = args.globs
    for f in _get(base, "/api/files", params):
        print(f"{f['repo']}/{f['path']}\t{f['indexed_at']}")


def _cmd_repos(_args: argparse.Namespace, base: str) -> None:
    for r in _get(base, "/api/repos"):
        print(f"{r['name']}\t{r['root']}\t{r['added_at']}")


def _cmd_status(_args: argparse.Namespace, base: str) -> None:
    data = _get(base, "/api/status")
    print(f"total_units: {data['total_units']}")
    print(f"embed_count: {data['embed_count']}")
    for r in data["repos"]:
        ts = r.get("last_indexed_at") or "never"
        print(
            f"  {r['repo']}\tfiles={r['file_count']}\tunits={r['unit_count']}\tlast_indexed={ts}"
        )


def _cmd_browse(args: argparse.Namespace, base: str) -> None:
    for n in _get(base, "/api/browse", {"path": args.path}):
        parts = [n["type"], n["name"], n["path"]]
        if n.get("unit_type"):
            parts.append(n["unit_type"])
        if n.get("summary"):
            parts.append(n["summary"])
        print("\t".join(parts))


def _cmd_index_start(args: argparse.Namespace, base: str) -> None:
    _print_job_status(
        _post(base, "/api/index", {"paths": args.paths, "reindex": args.reindex})
    )


def _cmd_index_status(_args: argparse.Namespace, base: str) -> None:
    _print_job_status(_get(base, "/api/index/status"))


def _cmd_index_cancel(_args: argparse.Namespace, base: str) -> None:
    _print_job_status(_post(base, "/api/index/cancel"))


def _cmd_clear_repo(args: argparse.Namespace, base: str) -> None:
    data = _post(base, "/api/clear_repo", params={"repo": args.repo})
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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="code-rag-cli",
        description="CLI client for the code-rag REST API",
    )
    p.add_argument(
        "--base-url",
        default="http://localhost:8081",
        dest="base_url",
        help="Base URL of the code-rag webui server",
    )
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("search", help="Search indexed code")
    s.add_argument("query", help="Natural-language search query")
    s.add_argument("--top-k", type=int, default=5, dest="top_k")
    s.add_argument("--glob", action="append", default=[], dest="globs")
    s.set_defaults(func=_cmd_search)

    s = sub.add_parser("unit", help="Get a single unit by qualified path")
    s.add_argument("path", help="Qualified path, e.g. repo/file.py:Class:method")
    s.set_defaults(func=_cmd_unit)

    s = sub.add_parser("fetch", help="Fetch multiple units by qualified path")
    s.add_argument("paths", nargs="+", help="Qualified paths")
    s.set_defaults(func=_cmd_fetch)

    s = sub.add_parser("units", help="List semantic units")
    s.add_argument("--limit", type=int, default=100)
    s.add_argument("--glob", action="append", default=[], dest="globs")
    s.set_defaults(func=_cmd_units)

    s = sub.add_parser("files", help="List indexed files")
    s.add_argument("--glob", action="append", default=[], dest="globs")
    s.set_defaults(func=_cmd_files)

    s = sub.add_parser("repos", help="List indexed repositories")
    s.set_defaults(func=_cmd_repos)

    s = sub.add_parser("status", help="Index health check")
    s.set_defaults(func=_cmd_status)

    s = sub.add_parser("browse", help="Browse the index tree")
    s.add_argument("path", nargs="?", default="", help="Path prefix to browse")
    s.set_defaults(func=_cmd_browse)

    s = sub.add_parser("index", help="Start an indexing job")
    s.add_argument("paths", nargs="+", help="Directory paths to index")
    s.add_argument("--reindex", action="store_true")
    s.set_defaults(func=_cmd_index_start)

    s = sub.add_parser("index-status", help="Poll indexing job state")
    s.set_defaults(func=_cmd_index_status)

    s = sub.add_parser("index-cancel", help="Cancel running indexing job")
    s.set_defaults(func=_cmd_index_cancel)

    s = sub.add_parser("clear-repo", help="Remove indexed data for a repository")
    s.add_argument("repo", help="Repository name")
    s.set_defaults(func=_cmd_clear_repo)

    return p


def main() -> None:
    args = _build_parser().parse_args()
    args.func(args, args.base_url)


if __name__ == "__main__":
    main()
