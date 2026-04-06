#!/usr/bin/env python3
"""Remove the mcp-rag server entry from ~/.pi/agent/mcp.json."""

import json
from pathlib import Path


def main():
    mcp_json = Path.home() / ".pi" / "agent" / "mcp.json"

    cfg = json.loads(mcp_json.read_text())
    cfg.get("mcpServers", {}).pop("code-rag", None)
    mcp_json.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"removed mcp-rag from {mcp_json}")


if __name__ == "__main__":
    main()
