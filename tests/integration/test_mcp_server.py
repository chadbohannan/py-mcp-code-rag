"""
Integration tests for the MCP server.

These tests use fastmcp's in-process Client to connect to the server instance
directly — the same MCP protocol used by stdio transport, without spawning a
subprocess.
"""

import pytest
from fastmcp import Client

from mcp_rag.server import mcp


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
async def tools_by_name() -> dict:
    """Fetch the tool list once via the MCP protocol and key it by name."""
    async with Client(mcp) as client:
        tools = await client.list_tools()
    return {t.name: t for t in tools}


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


async def test_server_connection():
    """Client can connect and the server responds to a ping."""
    async with Client(mcp) as client:
        await client.ping()


# ---------------------------------------------------------------------------
# Tool presence
# ---------------------------------------------------------------------------


async def test_server_exposes_search_tool(tools_by_name):
    assert "search" in tools_by_name, "search tool not registered"


async def test_server_exposes_index_status_tool(tools_by_name):
    assert "index_status" in tools_by_name, "index_status tool not registered"


async def test_server_exposes_get_unit_tool(tools_by_name):
    assert "get_unit" in tools_by_name, "get_unit tool not registered"


async def test_server_exposes_list_units_tool(tools_by_name):
    assert "list_units" in tools_by_name, "list_units tool not registered"


async def test_server_exposes_list_repos_tool(tools_by_name):
    assert "list_repos" in tools_by_name, "list_repos tool not registered"


# ---------------------------------------------------------------------------
# search — schema discovery
# ---------------------------------------------------------------------------


async def test_search_has_description(tools_by_name):
    """LLM needs a description to decide when to call this tool."""
    tool = tools_by_name["search"]
    assert tool.description and len(tool.description) > 0


async def test_search_query_param_is_required_string(tools_by_name):
    """'query' must be a required string — it is the natural language question."""
    schema = tools_by_name["search"].inputSchema
    props = schema.get("properties", {})
    assert "query" in props, "search schema missing 'query' property"
    assert props["query"].get("type") == "string"
    assert "query" in schema.get("required", [])


async def test_search_top_k_param_is_optional_integer_defaulting_to_5(tools_by_name):
    """'top_k' must be an optional integer with a default of 5."""
    schema = tools_by_name["search"].inputSchema
    props = schema.get("properties", {})
    assert "top_k" in props, "search schema missing 'top_k' property"
    assert props["top_k"].get("type") == "integer"
    assert props["top_k"].get("default") == 5
    assert "top_k" not in schema.get("required", [])


# ---------------------------------------------------------------------------
# index_status — schema discovery
# ---------------------------------------------------------------------------


async def test_index_status_has_description(tools_by_name):
    """LLM needs a description to decide when to call this tool."""
    tool = tools_by_name["index_status"]
    assert tool.description and len(tool.description) > 0


async def test_index_status_has_no_required_params(tools_by_name):
    """index_status takes no arguments."""
    schema = tools_by_name["index_status"].inputSchema
    assert schema.get("required", []) == []
