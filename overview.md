# py-mcp-rag

A developer utility for RAG over local codebases, exposed as an MCP stdio server. Designed
for navigating complex, sprawling codebases — surfacing architectural intent rather than
matching surface text.

Embeddings run in-process via `fastembed` (ONNX Runtime). The index — content, metadata, and
vectors — lives in a single SQLite file at `.mcp-rag/index.db` within each indexed root.

## Semantic Surrogate Indexing

Raw source code embeds poorly against natural language queries. A developer asking "how does
authentication work?" shares almost no embedding space with the code that implements it.

This tool uses **Semantic Surrogate Indexing**: files are parsed into language-aware semantic
units (functions, classes, CTEs, paragraphs), each unit is summarized by Claude with its
source-path context, and the *summary* — not the raw source — is embedded. The raw source is
stored alongside and returned to callers on a match.

```
source file → semantic parser → semantic units
                                      │
                              Claude summary
                              (file path + unit type as context)
                                      │
                              fastembed → vector index
                              raw source stored alongside, returned on match
```

Three properties make this well-suited to complex codebases:

**Meaningful boundaries** — unit boundaries are structural, not arbitrary. A function, class,
CTE, or section heading is the natural unit of codebase knowledge. Fixed-window chunking splits
functions in half and merges unrelated paragraphs.

**Query-aligned targets** — summaries are written in the language developers use to ask
questions. *"Validates a JWT and checks clock skew against a configurable tolerance"* sits far
closer in embedding space to *"how does token expiry work?"* than the raw source does.

**Context propagation** — passing file path and unit type to Claude lets summaries encode
architectural position ("this is the rate limiter in the ingestion pipeline") that no amount
of raw-text embedding can recover.

This extends Anthropic's *Contextual Retrieval* — which prepends a generated context statement
to each chunk before embedding — by replacing the raw content as the embedding target entirely
and using structural rather than fixed-window boundaries.

## runtime design

- **index mode** (`mcp-rag index [paths...]`): parses each file into semantic units, generates
  a Claude summary per unit with source-path context, and embeds the summary into the vector
  index. Incremental — only changed files are re-processed. Requires `ANTHROPIC_API_KEY`.

- **serve mode** (`mcp-rag serve [paths...]`): starts the MCP stdio server. Embeds each
  incoming query and returns the closest matching source units by vector similarity, with path
  and relevance score. Read-only; no API key required.

- **combined mode** (`mcp-rag [paths...]`): indexes if the index is absent, then serves.
