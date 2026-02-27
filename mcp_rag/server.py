from fastmcp import FastMCP

mcp = FastMCP("mcp-rag")


@mcp.tool
async def search(query: str, top_k: int = 5) -> list[dict]:
    """Search the indexed codebase using a natural language question.

    Embeds the query and returns the closest matching semantic units by vector
    similarity. Each result includes the source path, unit type, unit name,
    original source content, a human-readable summary, and a relevance score
    in [0.0, 1.0] (higher is better). top_k is capped at 20.
    """
    raise NotImplementedError


@mcp.tool
async def index_status() -> list[dict]:
    """Return the current state of the index.

    Reports per-root file count, semantic unit count, and the timestamp of the
    most recent indexing run. Returns one entry per distinct root path that has
    been indexed into the active database.
    """
    raise NotImplementedError
