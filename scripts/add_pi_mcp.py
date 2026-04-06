#!/usr/bin/env python3
"""Add the mcp-rag server entry to ~/.pi/agent/mcp.json."""

import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 3:
        print("usage: add_pi_mcp.py <directory> <db-path>", file=sys.stderr)
        sys.exit(1)

    directory, db_path = sys.argv[1], sys.argv[2]
    mcp_json = Path.home() / ".pi" / "agent" / "mcp.json"

    cfg = json.loads(mcp_json.read_text())
    cfg.setdefault("mcpServers", {})["code-rag"] = {
        "command": "uv",
        "args": ["run", "--directory", directory, "code-rag", "serve", "--db", db_path],
    }
    mcp_json.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"added code-rag to {mcp_json}")


if __name__ == "__main__":
    main()
