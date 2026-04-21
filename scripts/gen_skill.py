"""Generate SKILL.md — CLI-first reference for code-rag-cli.py.

The command reference is derived from the actual CLI output formats defined
in mcp_rag/client.py rather than the OpenAPI spec, so agents get documentation
that matches what they'll see when running the tool.

Usage:
    uv run python scripts/gen_skill.py > SKILL.md
"""

from __future__ import annotations

CLI = "python3 code-rag-cli.py"

# Each command: (name, summary, usage_lines, flags_table_or_None, output_example)
_COMMANDS: list[tuple[str, str, list[str], str | None, str]] = [
    (
        "status",
        "Check index health. Always run this first to confirm the repo you care about is indexed.",
        [f"{CLI} status"],
        None,
        (
            "total_units: 1234\n"
            "embed_count: 1234\n"
            "  myrepo\tfiles=42\tunits=350\tlast_indexed=2025-01-15T10:30:00"
        ),
    ),
    (
        "search",
        "Semantic search across all indexed code. Describe what you're looking for in "
        "plain English — the index matches against natural-language summaries, not keywords."
        "The index currently has 41,136 piece-wise summaries of code, configuration, and markdown files in 36 repositories.",
        [
            f'{CLI} search "retry logic for HTTP requests"',
            f'{CLI} search --top-k 10 --glob "*.py" "error handling"',
        ],
        (
            "| Flag | Default | Description |\n"
            "|------|---------|-------------|\n"
            "| `--top-k N` | 5 | Number of results (1–20) |\n"
            "| `--glob PATTERN` | — | SQLite GLOB filter on qualified path (repeatable) |"
        ),
        (
            "0.9500\trepo/file.py:retry_request\n"
            "  Retries HTTP requests with exponential backoff\n"
            "0.8200\trepo/client.py:HttpClient:_do_request\n"
            "  Sends HTTP request and handles transient failures"
        ),
    ),
    (
        "unit",
        "Retrieve full source code and summary for a single unit by its qualified path.",
        [f"{CLI} unit repo/file.py:Class:method"],
        None,
        (
            "# repo/file.py:Class:method\n"
            "# Summary of what this unit does\n"
            "\n"
            "def method(self):\n"
            "    ..."
        ),
    ),
    (
        "fetch",
        "Retrieve full source for multiple units at once. Use this after `search` to "
        "pull the top results in a single call.",
        [f"{CLI} fetch repo/a.py:func repo/b.py:Class:method"],
        None,
        (
            "# repo/a.py:func\n"
            "# Summary of func\n"
            "\n"
            "def func(): ...\n"
            "\n"
            "---\n"
            "\n"
            "# repo/b.py:Class:method\n"
            "# Summary of method\n"
            "\n"
            "def method(self): ..."
        ),
    ),
    (
        "browse",
        "Navigate the index tree hierarchically: repos → directories → files → units. "
        "Useful for exploring code neighboring a search result.",
        [
            f"{CLI} browse",
            f"{CLI} browse myrepo/src",
            f"{CLI} browse myrepo/src/main.py",
        ],
        None,
        (
            "repo\tmyrepo\tmyrepo\n"
            "dir\tsrc\tmyrepo/src\n"
            "file\tmain.py\tmyrepo/src/main.py\n"
            "unit\tMyClass\tmyrepo/src/main.py:MyClass\tclass\tTop-level application class"
        ),
    ),
    (
        "units",
        "List all semantic units (path + summary) with optional filtering.",
        [
            f"{CLI} units",
            f'{CLI} units --limit 50 --glob "backend/*.py"',
        ],
        (
            "| Flag | Default | Description |\n"
            "|------|---------|-------------|\n"
            "| `--limit N` | 100 | Max results (1–500) |\n"
            "| `--glob PATTERN` | — | SQLite GLOB filter (repeatable) |"
        ),
        "repo/file.py:func\tSummary of what func does",
    ),
    (
        "files",
        "List indexed files with optional glob filtering.",
        [
            f"{CLI} files",
            f'{CLI} files --glob "*.ts"',
        ],
        (
            "| Flag | Default | Description |\n"
            "|------|---------|-------------|\n"
            "| `--glob PATTERN` | — | SQLite GLOB filter (repeatable) |"
        ),
        "myrepo/src/main.py\t2025-01-15T10:30:00",
    ),
    (
        "repos",
        "List all indexed repositories.",
        [f"{CLI} repos"],
        None,
        "myrepo\t/home/user/code/myrepo\t2025-01-15",
    ),
    (
        "clear-repo",
        "Remove all indexed data for a repository.",
        [f"{CLI} clear-repo myrepo"],
        None,
        "cleared: myrepo",
    ),
]


def render() -> str:
    sections = [
        "---\n",
        "name: code-rag\n",
        "description: Semantic code search over pre-indexed repositories via "
        "code-rag-cli.py. Use for exploratory or cross-repo questions about code — "
        "architecture, patterns, usage examples — when you don't already know which "
        "file to open.\n",
        "---\n",

        "# code-rag skill\n",
        "code-rag is a semantic code search server. It indexes source code by generating "
        "natural-language summaries of each function, class, and section, then embeds "
        "those summaries for vector search. `code-rag-cli.py` is the CLI interface — "
        "it produces plain-text, tab-delimited output designed for agentic LLM use.\n",

        "## When to use\n",
        "Use code-rag when you have a **vague or exploratory** question about a codebase: "
        "\"how does authentication work?\", \"where are database migrations defined?\", "
        "\"what calls the billing API?\". It finds semantically relevant code across all "
        "indexed repositories without requiring you to know file paths or symbol names.\n",
        "**Do NOT use** when you already know the exact file or symbol — use Read, Grep, "
        "or Glob directly. code-rag only covers pre-indexed repositories; if a repo is "
        "not indexed, results will be empty.\n",

        "## Setup\n",
        "The CLI requires a running code-rag webui server (default: `http://localhost:8080`). "
        "Override with `--base-url`:\n",
        "```bash\n"
        f"{CLI} --base-url http://host:9090 status\n"
        "```\n",

        "## Workflow\n",
        f"1. **`{CLI} status`** — confirm the repo is indexed; if missing, stop\n"
        f"2. **`{CLI} search \"<question>\"`** — describe what you need in plain English\n"
        f"3. **`{CLI} fetch <path1> <path2> ...`** — pull full source for the top hits\n"
        f"4. **`{CLI} browse <repo/dir>`** — explore neighboring code\n",

        "## Tips\n",
        "- **Write full sentences**, not keywords. \"function that parses CSV uploads\" "
        "outperforms \"csv parse\".\n"
        "- **Use `--glob` to narrow scope**: "
        f"`{CLI} search --glob \"backend/*.py\" \"error handling\"`.\n"
        "- Read the summaries from `search` before fetching source — skip irrelevant hits.\n"
        "- Output is tab-delimited, one record per line (except `unit`/`fetch` which "
        "print full source blocks separated by `---`).\n",

        "---\n",
        "# Command reference\n",
    ]

    for name, summary, usage, flags, output in _COMMANDS:
        sections.append(f"\n## `{name}`\n")
        sections.append(f"{summary}\n")
        sections.append("```bash\n" + "\n".join(usage) + "\n```\n")
        if flags:
            sections.append(f"\n{flags}\n")
        sections.append(f"\n**Output:**\n```\n{output}\n```\n")

    return "\n".join(sections)


if __name__ == "__main__":
    print(render())
