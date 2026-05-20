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
    assert r.status_code == 200
    scores = [e["score"] for e in r.json()]
    assert len(scores) >= 2
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
    results = r.json()
    assert len(results) == 2


def test_units_listing_returns_paths_and_summaries(client):
    r = client.get("/api/units", params={"limit": 100})
    assert r.status_code == 200
    units = r.json()
    assert len(units) > 0
    assert all(set(u.keys()) >= {"path", "summary"} for u in units)
    paths = {u["path"] for u in units}
    assert any(p.endswith(":login") for p in paths)
    assert any(p.endswith(":connect") for p in paths)


def test_unit_fetch_single_returns_full_content(client):
    listing = client.get("/api/units").json()
    target = next(u["path"] for u in listing if u["path"].endswith(":login"))
    r = client.get("/api/unit", params={"path": target})
    assert r.status_code == 200
    body = r.json()
    assert body["path"] == target
    assert "def login" in body["content"]
    assert body["summary"]


def test_units_fetch_post_returns_only_matching_paths(client):
    listing = client.get("/api/units").json()
    real_path = listing[0]["path"]
    r = client.post(
        "/api/units/fetch",
        json={"paths": [real_path, "testrepo/does/not/exist:nope"]},
    )
    assert r.status_code == 200
    results = r.json()
    assert len(results) == 1
    assert results[0]["path"] == real_path


def test_files_listing_returns_three_files(client):
    r = client.get("/api/files")
    assert r.status_code == 200
    files = r.json()
    paths = {f["path"] for f in files}
    assert paths == {"src/app.py", "src/db.py", "README.md"}


def test_files_glob_filter_narrows_to_markdown(client):
    r = client.get("/api/files", params={"globs": "*.md"})
    assert r.status_code == 200
    paths = [f["path"] for f in r.json()]
    assert paths == ["README.md"]


def test_repos_and_status_consistent(client):
    repos = client.get("/api/repos").json()
    assert len(repos) == 1
    assert repos[0]["name"] == "testrepo"

    status = client.get("/api/status").json()
    assert status["repos"][0]["repo"] == "testrepo"
    assert status["repos"][0]["file_count"] == 3
