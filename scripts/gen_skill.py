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

# Paths excluded from the skill doc — UI helpers and admin/management endpoints
# that agents have no reason to call.
_EXCLUDE_PATHS = {
    "/api/ls",           # browser path-picker UI only
    "/api/index",        # indexing management
    "/api/index/status",
    "/api/index/cancel",
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


def render(spec: dict) -> str:
    sections = [
        "---\n",
        "name: code-rag\n",
        "description: provides preprocessed summaries of code repositories and markdown archives.\n",
        "---\n",
        "# code-rag skill\n",
        "code-rag is a semantic code search server. It indexes source code by generating "
        "natural-language summaries of each function, class, and section, then embeds those "
        "summaries for vector search. Queries should be asked as natural-language questions, "
        "not keyword fragments.\n",
        f"**Base URL:** `{BASE_URL}`\n",
        "## Recommended workflow\n",
        "1. `GET /api/status` — confirm the index is populated and fresh\n"
        "2. `GET /api/search?q=...` — find relevant units by natural-language topic\n"
        "3. `POST /api/units/fetch` — retrieve full source for specific paths from search results\n"
        "4. `GET /api/browse?path=repo/file.py` — explore structure of a file or directory\n",
    ]

    paths = spec.get("paths", {})
    for path, path_item in sorted(paths.items()):
        if path in _EXCLUDE_PATHS:
            continue
        for method, op in path_item.items():
            if method not in ("get", "post", "put", "delete", "patch"):
                continue

            summary = op.get("summary", op.get("description", ""))
            parameters = op.get("parameters", [])
            request_body = op.get("requestBody")

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

            if resp_schema:
                shape = _schema_to_shape(spec, resp_schema)
                sections.append(f"\n**Response:**\n```\n{shape}\n```\n")

    return "\n".join(sections)


if __name__ == "__main__":
    spec = app.openapi()
    print(render(spec))
