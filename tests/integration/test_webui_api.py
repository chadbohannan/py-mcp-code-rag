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


def test_search_returns_results_with_required_fields(client):
    r = client.get("/api/search", params={"q": "user login", "top_k": 5})
    assert r.status_code == 200
    results = r.json()
    assert len(results) > 0
    for entry in results:
        assert set(entry.keys()) >= {"path", "summary", "score"}
        assert isinstance(entry["score"], float)


def test_search_results_are_ordered_by_score_descending(client):
    r = client.get("/api/search", params={"q": "database connection", "top_k": 5})
    scores = [e["score"] for e in r.json()]
    assert scores == sorted(scores, reverse=True)


def test_search_globs_filter_narrows_to_markdown(client):
    r = client.get(
        "/api/search",
        params=[("q", "overview"), ("top_k", 10), ("globs", "*.md*")],
    )
    assert r.status_code == 200
    paths = [e["path"] for e in r.json()]
    assert paths, "globs filter should still return at least one match"
    assert all(".md" in p for p in paths)


def test_search_top_k_caps_returned_count(client):
    r = client.get("/api/search", params={"q": "anything", "top_k": 2})
    assert r.status_code == 200
    assert len(r.json()) <= 2
