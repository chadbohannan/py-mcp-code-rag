---

name: code-rag

description: provides preprocessed summaries of code repositories and markdown archives.

---

# code-rag skill

code-rag is a semantic code search server. It indexes source code by generating natural-language summaries of each function, class, and section, then embeds those summaries for vector search. Queries should be asked as natural-language questions, not keyword fragments.

**Base URL:** `http://localhost:8081`

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


**Response:**
```
[{
  repo: string,
  path: string,
  indexed_at: string,
}]
```

---

## `GET /api/repos`

List all indexed repositories


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


**Response:**
```
[{
  path: string,
  content: string,
  summary: string,
}]
```

