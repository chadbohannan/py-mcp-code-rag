---

name: code-rag

description: Semantic code search over pre-indexed repositories via code-rag-cli.py. Use for exploratory or cross-repo questions about code — architecture, patterns, usage examples — when you don't already know which file to open.

---

# code-rag skill

code-rag is a semantic code search server. It indexes source code by generating natural-language summaries of each function, class, and section, then embeds those summaries for vector search. `code-rag-cli.py` is the CLI interface — it produces plain-text, tab-delimited output designed for agentic LLM use.

## When to use

Use code-rag when you have a **vague or exploratory** question about a codebase: "how does authentication work?", "where are database migrations defined?", "what calls the billing API?". It finds semantically relevant code across all indexed repositories without requiring you to know file paths or symbol names.

**Do NOT use** when you already know the exact file or symbol — use Read, Grep, or Glob directly. code-rag only covers pre-indexed repositories; if a repo is not indexed, results will be empty.

## Setup

The CLI requires a running code-rag webui server (default: `http://localhost:8081`). Override with `--base-url`:

```bash
python code-rag-cli.py status
```

## Workflow

1. **`python code-rag-cli.py status`** — confirm the repo is indexed; if missing, stop
2. **`python code-rag-cli.py search "<question>"`** — describe what you need in plain English
3. **`python code-rag-cli.py fetch <path1> <path2> ...`** — pull full source for the top hits
4. **`python code-rag-cli.py browse <repo/dir>`** — explore neighboring code

## Tips

- **Write full sentences**, not keywords. "function that parses CSV uploads" outperforms "csv parse".
- **Use `--glob` to narrow scope**: `python code-rag-cli.py search --glob "backend/*.py" "error handling"`.
- Read the summaries from `search` before fetching source — skip irrelevant hits.
- Output is tab-delimited, one record per line (except `unit`/`fetch` which print full source blocks separated by `---`).

---

# Command reference


## `status`

Check index health. Always run this first to confirm the repo you care about is indexed.

```bash
python code-rag-cli.py status
```


**Output:**
```
total_units: 1234
embed_count: 1234
  myrepo	files=42	units=350	last_indexed=2025-01-15T10:30:00
```


## `search`

Semantic search across all indexed code. Describe what you're looking for in plain English — the index matches against natural-language summaries, not keywords.

```bash
python code-rag-cli.py search "retry logic for HTTP requests"
python code-rag-cli.py search --top-k 10 --glob "*.py" "error handling"
```


| Flag | Default | Description |
|------|---------|-------------|
| `--top-k N` | 5 | Number of results (1–20) |
| `--glob PATTERN` | — | SQLite GLOB filter on qualified path (repeatable) |


**Output:**
```
0.9500	repo/file.py:retry_request
  Retries HTTP requests with exponential backoff
0.8200	repo/client.py:HttpClient:_do_request
  Sends HTTP request and handles transient failures
```


## `unit`

Retrieve full source code and summary for a single unit by its qualified path.

```bash
python code-rag-cli.py unit repo/file.py:Class:method
```


**Output:**
```
# repo/file.py:Class:method
# Summary of what this unit does

def method(self):
    ...
```


## `fetch`

Retrieve full source for multiple units at once. Use this after `search` to pull the top results in a single call.

```bash
python code-rag-cli.py fetch repo/a.py:func repo/b.py:Class:method
```


**Output:**
```
# repo/a.py:func
# Summary of func

def func(): ...

---

# repo/b.py:Class:method
# Summary of method

def method(self): ...
```


## `browse`

Navigate the index tree hierarchically: repos → directories → files → units. Useful for exploring code neighboring a search result.

```bash
python code-rag-cli.py browse
python code-rag-cli.py browse myrepo/src
python code-rag-cli.py browse myrepo/src/main.py
```


**Output:**
```
repo	myrepo	myrepo
dir	src	myrepo/src
file	main.py	myrepo/src/main.py
unit	MyClass	myrepo/src/main.py:MyClass	class	Top-level application class
```


## `units`

List all semantic units (path + summary) with optional filtering.

```bash
python code-rag-cli.py units
python code-rag-cli.py units --limit 50 --glob "backend/*.py"
```


| Flag | Default | Description |
|------|---------|-------------|
| `--limit N` | 100 | Max results (1–500) |
| `--glob PATTERN` | — | SQLite GLOB filter (repeatable) |


**Output:**
```
repo/file.py:func	Summary of what func does
```


## `files`

List indexed files with optional glob filtering.

```bash
python code-rag-cli.py files
python code-rag-cli.py files --glob "*.ts"
```


| Flag | Default | Description |
|------|---------|-------------|
| `--glob PATTERN` | — | SQLite GLOB filter (repeatable) |


**Output:**
```
myrepo/src/main.py	2025-01-15T10:30:00
```


## `repos`

List all indexed repositories.

```bash
python code-rag-cli.py repos
```


**Output:**
```
myrepo	/home/user/code/myrepo	2025-01-15
```


## `clear-repo`

Remove all indexed data for a repository.

```bash
python code-rag-cli.py clear-repo myrepo
```


**Output:**
```
cleared: myrepo
```

