"""Integration tests for the webui REST API.

Drives the FastAPI app in-process via TestClient against a real SQLite index
built with FakeEmbedder + FakeSummarizer. Replaces the deleted
test_server_search.py from before the MCP-HTTP migration.
"""

from __future__ import annotations


def test_status_endpoint_returns_populated_index(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["total_units"] > 0
    # embed_count includes embeddings for directory units, which are not
    # joined into total_units (which counts via files); see queries.index_status.
    assert body["embed_count"] >= body["total_units"]
    assert len(body["repos"]) == 1
    assert body["repos"][0]["repo"] == "testrepo"
