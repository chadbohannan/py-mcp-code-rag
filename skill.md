# code-rag skill

code-rag is a semantic code search server. It indexes source code by generating natural-language summaries of each function, class, and section, then embeds those summaries for vector search. Queries should be asked as natural-language questions, not keyword fragments.

**Base URL:** `http://localhost:8081` (set with `--port` when starting `code-rag webui`)

## Recommended workflow

1. `GET /api/status` — confirm the index is populated and fresh
2. `GET /api/search?q=...` — find relevant units by natural-language topic
3. `POST /api/units/fetch` — retrieve full source for specific paths from search results
4. `GET /api/browse?path=repo/file.py` — explore structure of a file or directory

---

## `GET /api/browse`

Browse the index tree: repos → dirs → files → units


**Query parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | no | Qualified path prefix, e.g. repo, repo/dir, repo/file.py, repo/file.py:Class |


```sh
curl "http://localhost:8081/api/browse"
```


**Response:**
```
[{
  type: string,
  name: string,
  path: string,
  summary: string,
  has_children: boolean,
  unit_type?: string | null,
}]
```

---

## `GET /api/files`

List indexed files with optional glob filter


**Query parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `globs` | string[] (repeat param) | no |  |


```sh
curl "http://localhost:8081/api/files"
```


**Response:**
```
[{
  repo: string,
  path: string,
  indexed_at: string,
}]
```

---

## `POST /api/index`

Start an indexing job (returns 409 if one is already running)


**Request body (JSON):**

| Field | Type | Required | Default |
|-------|------|----------|---------|
| `paths` | array | yes | — |
| `reindex` | boolean | no | False |


```sh
curl -X POST http://localhost:8081/api/index \
  -H "Content-Type: application/json" \
  -d '{"paths": ["/path/to/repo"], "reindex": false}'
```


**Response:**
```
{
  running: boolean,
  last_result?: string | null,
  last_finished_at?: string | null,
}
```

---

## `POST /api/index/cancel`

Signal the running indexing job to cancel


```sh
curl -X POST "http://localhost:8081/api/index/cancel"
```


**Response:**
```
{
  running: boolean,
  last_result?: string | null,
  last_finished_at?: string | null,
}
```

---

## `GET /api/index/status`

Poll the current indexing job state


```sh
curl "http://localhost:8081/api/index/status"
```


**Response:**
```
{
  running: boolean,
  last_result?: string | null,
  last_finished_at?: string | null,
}
```

---

## `GET /api/ls`

List filesystem directories for the index path picker


**Query parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | no | Absolute filesystem path; defaults to home directory |


```sh
curl "http://localhost:8081/api/ls"
```


**Response:**
```
{
  path: string,
  parent: string | null,
  is_git: boolean,
  dirs: [{
    name: string,
    path: string,
  }],
}
```

---

## `GET /api/repos`

List all indexed repositories


```sh
curl "http://localhost:8081/api/repos"
```


**Response:**
```
[{
  name: string,
  root: string,
  added_at: string,
  description: string,
}]
```

---

## `GET /api/search`

Search indexed code by natural-language query


**Query parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `q` | string | yes | Natural-language query |
| `top_k` | integer | no |  |
| `globs` | string[] (repeat param) | no | SQLite GLOB filters on qualified path |


```sh
curl "http://localhost:8081/api/search?q=how+does+auth+work&top_k=5"

# With glob filters
curl "http://localhost:8081/api/search?q=parser+entry+point&globs=*.py&globs=backend/*"
```


**Response:**
```
[{
  path: string,
  summary: string,
  score: number,
}]
```

---

## `GET /api/status`

Index health: per-repo file/unit counts and last-indexed timestamp


```sh
curl "http://localhost:8081/api/status"
```


**Response:**
```
{
  repos: [{
    repo: string,
    root: string,
    file_count: integer,
    unit_count: integer,
    last_indexed_at?: string | null,
  }],
  total_units: integer,
  embed_count: integer,
}
```

---

## `GET /api/unit`

Retrieve full source and summary for a single unit by qualified path


**Query parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | yes | Qualified path, e.g. repo/file.py:Class:method |


```sh
curl "http://localhost:8081/api/unit?path=<path>"
```


**Response:**
```
{
  path: string,
  content: string,
  summary: string,
}
```

---

## `GET /api/units`

List semantic units (path + summary) with optional glob filter


**Query parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `limit` | integer | no |  |
| `globs` | string[] (repeat param) | no |  |


```sh
curl "http://localhost:8081/api/units"
```


**Response:**
```
[{
  path: string,
  summary: string,
}]
```

---

## `POST /api/units/fetch`

Retrieve full source and summary for multiple units by qualified path


**Request body (JSON):**

| Field | Type | Required | Default |
|-------|------|----------|---------|
| `paths` | array | yes | — |


```sh
curl -X POST http://localhost:8081/api/units/fetch \
  -H "Content-Type: application/json" \
  -d '{"paths": ["myrepo/src/auth.py:AuthHandler:validate"]}'
```


**Response:**
```
[{
  path: string,
  content: string,
  summary: string,
}]
```

