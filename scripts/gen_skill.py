"""Generate skill.md from the FastAPI OpenAPI spec.

Usage:
    uv run python scripts/gen_skill.py > skill.md
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_rag.webui import app  # noqa: E402

BASE_URL = "http://localhost:8081"

# Curl examples per operation (keyed by operationId).
# Anything not listed here gets a generic example generated automatically.
_CURL_OVERRIDES: dict[str, str] = {
    "api_search_api_search_get": (
        f'curl "{BASE_URL}/api/search?q=how+does+auth+work&top_k=5"\n\n'
        f"# With glob filters\n"
        f'curl "{BASE_URL}/api/search?q=parser+entry+point&globs=*.py&globs=backend/*"'
    ),
    "api_units_fetch_api_units_fetch_post": (
        f"curl -X POST {BASE_URL}/api/units/fetch \\\n"
        '  -H "Content-Type: application/json" \\\n'
        '  -d \'{"paths": ["myrepo/src/auth.py:AuthHandler:validate"]}\''
    ),
    "api_index_start_api_index_post": (
        f"curl -X POST {BASE_URL}/api/index \\\n"
        '  -H "Content-Type: application/json" \\\n'
        '  -d \'{"paths": ["/path/to/repo"], "reindex": false}\''
    ),
    "api_index_cancel_api_index_cancel_post": (
        f'curl -X POST "{BASE_URL}/api/index/cancel"'
    ),
}


def _resolve_ref(spec: dict, ref: str) -> dict:
    parts = ref.lstrip("#/").split("/")
    node = spec
    for p in parts:
        node = node[p]
    return node


def _schema_to_shape(spec: dict, schema: dict, depth: int = 0) -> str:
    if "$ref" in schema:
        schema = _resolve_ref(spec, schema["$ref"])

    typ = schema.get("type")
    indent = "  " * depth

    if typ == "array":
        items = schema.get("items", {})
        return f"[{_schema_to_shape(spec, items, depth)}]"

    if typ == "object" or "properties" in schema:
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        lines = ["{"]
        for k, v in props.items():
            opt = "" if k in required else "?"
            lines.append(f"{indent}  {k}{opt}: {_schema_to_shape(spec, v, depth + 1)},")
        lines.append(f"{indent}}}")
        return "\n".join(lines)

    if "anyOf" in schema:
        types = [_schema_to_shape(spec, s, depth) for s in schema["anyOf"]]
        return " | ".join(types)

    return typ or "any"


def _params_table(parameters: list[dict]) -> str:
    if not parameters:
        return ""
    lines = [
        "| Parameter | Type | Required | Description |",
        "|-----------|------|----------|-------------|",
    ]
    for p in parameters:
        schema = p.get("schema", {})
        typ = schema.get("type", "string")
        if schema.get("items"):
            typ = "string[] (repeat param)"
        req = "yes" if p.get("required") else "no"
        desc = p.get("description", "")
        lines.append(f"| `{p['name']}` | {typ} | {req} | {desc} |")
    return "\n".join(lines)


def _body_table(spec: dict, request_body: dict) -> str:
    content = request_body.get("content", {})
    json_content = content.get("application/json", {})
    schema = json_content.get("schema", {})
    if "$ref" in schema:
        schema = _resolve_ref(spec, schema["$ref"])
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    if not props:
        return ""
    lines = [
        "| Field | Type | Required | Default |",
        "|-------|------|----------|---------|",
    ]
    for k, v in props.items():
        typ = v.get("type", "any")
        req = "yes" if k in required else "no"
        default = str(v.get("default", "—"))
        lines.append(f"| `{k}` | {typ} | {req} | {default} |")
    return "\n".join(lines)


def _auto_curl(method: str, path: str, parameters: list[dict], has_body: bool) -> str:
    url = f"{BASE_URL}{path}"
    # Add representative query params
    qp = []
    for p in parameters:
        if p.get("in") == "query" and p.get("required"):
            qp.append(f"{p['name']}=<{p['name']}>")
    if qp:
        url += "?" + "&".join(qp)

    if method == "get":
        return f'curl "{url}"'
    if has_body:
        return (
            f"curl -X {method.upper()} {url} \\\n"
            '  -H "Content-Type: application/json" \\\n'
            "  -d '{...}'"
        )
    return f'curl -X {method.upper()} "{url}"'


def render(spec: dict) -> str:
    sections = [
        "# code-rag skill\n",
        "code-rag is a semantic code search server. It indexes source code by generating "
        "natural-language summaries of each function, class, and section, then embeds those "
        "summaries for vector search. Queries should be asked as natural-language questions, "
        "not keyword fragments.\n",
        f"**Base URL:** `{BASE_URL}` (set with `--port` when starting `code-rag webui`)\n",
        "## Recommended workflow\n",
        "1. `GET /api/status` — confirm the index is populated and fresh\n"
        "2. `GET /api/search?q=...` — find relevant units by natural-language topic\n"
        "3. `POST /api/units/fetch` — retrieve full source for specific paths from search results\n"
        "4. `GET /api/browse?path=repo/file.py` — explore structure of a file or directory\n",
    ]

    paths = spec.get("paths", {})
    # Group by tag or just sort
    for path, path_item in sorted(paths.items()):
        for method, op in path_item.items():
            if method not in ("get", "post", "put", "delete", "patch"):
                continue

            op_id = op.get("operationId", "")
            summary = op.get("summary", op.get("description", ""))
            parameters = op.get("parameters", [])
            request_body = op.get("requestBody")

            # Response shape
            responses = op.get("responses", {})
            success = responses.get("200") or responses.get("202") or {}
            resp_schema = (
                success.get("content", {}).get("application/json", {}).get("schema", {})
            )

            sections.append(f"---\n\n## `{method.upper()} {path}`\n")
            if summary:
                sections.append(f"{summary}\n")

            params_md = _params_table(parameters)
            if params_md:
                sections.append(f"\n**Query parameters:**\n\n{params_md}\n")

            if request_body:
                body_md = _body_table(spec, request_body)
                if body_md:
                    sections.append(f"\n**Request body (JSON):**\n\n{body_md}\n")

            curl = _CURL_OVERRIDES.get(op_id) or _auto_curl(
                method, path, parameters, bool(request_body)
            )
            sections.append(f"\n```sh\n{curl}\n```\n")

            if resp_schema:
                shape = _schema_to_shape(spec, resp_schema)
                sections.append(f"\n**Response:**\n```\n{shape}\n```\n")

    return "\n".join(sections)


if __name__ == "__main__":
    spec = app.openapi()
    print(render(spec))
