"""Unit tests for the YAML parser (tree-sitter implementation) in mcp_rag.parsers."""

import pytest

from mcp_rag.parsers import parse_yaml


# ---------------------------------------------------------------------------
# Empty / no-op inputs
# ---------------------------------------------------------------------------


def test_parse_yaml_empty_string_returns_empty():
    assert parse_yaml("") == []


# ---------------------------------------------------------------------------
# Missing tree-sitter-yaml
# ---------------------------------------------------------------------------


def test_parse_yaml_missing_package_warns(monkeypatch):
    monkeypatch.setattr("mcp_rag.parsers._get_ts_yaml_language", lambda: None)
    with pytest.warns(UserWarning, match="tree-sitter-yaml"):
        result = parse_yaml("name: CI\n")
    assert result == []


# ---------------------------------------------------------------------------
# Top-level mappings
# ---------------------------------------------------------------------------


def test_parse_yaml_top_level_keys():
    src = "name: CI\nversion: 1\n"
    units = parse_yaml(src)
    names = [u.unit_name for u in units]
    assert "name" in names
    assert "version" in names
    assert all(u.unit_type == "key" for u in units)


def test_parse_yaml_char_offsets_are_correct():
    src = "name: CI\nversion: 1\n"
    units = parse_yaml(src)
    by_name = {u.unit_name: u for u in units}
    assert src[by_name["name"].char_offset :].startswith("name:")
    assert src[by_name["version"].char_offset :].startswith("version:")


# ---------------------------------------------------------------------------
# Nested mappings recurse one level
# ---------------------------------------------------------------------------


def test_parse_yaml_nested_mapping_emits_dotted_names():
    src = (
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "  test:\n"
        "    runs-on: ubuntu-latest\n"
    )
    units = parse_yaml(src)
    names = [u.unit_name for u in units]
    assert "jobs" in names
    assert "jobs.build" in names
    assert "jobs.test" in names


def test_parse_yaml_does_not_recurse_past_two_levels():
    src = (
        "a:\n"
        "  b:\n"
        "    c:\n"
        "      d: 1\n"
    )
    units = parse_yaml(src)
    names = [u.unit_name for u in units]
    assert "a" in names
    assert "a.b" in names
    # c and d should not appear as their own units
    assert "a.b.c" not in names
    assert "a.b.c.d" not in names


# ---------------------------------------------------------------------------
# Sequences of mappings (e.g. GitHub Actions steps)
# ---------------------------------------------------------------------------


def test_parse_yaml_sequence_items_use_name_field():
    src = (
        "steps:\n"
        "  - name: checkout\n"
        "    uses: actions/checkout@v4\n"
        "  - name: test\n"
        "    run: make test\n"
    )
    units = parse_yaml(src)
    names = [u.unit_name for u in units]
    assert "steps[checkout]" in names
    assert "steps[test]" in names


def test_parse_yaml_sequence_items_fall_back_to_index():
    src = (
        "steps:\n"
        "  - run: echo hi\n"
        "  - run: echo bye\n"
    )
    units = parse_yaml(src)
    names = [u.unit_name for u in units]
    assert "steps[0]" in names
    assert "steps[1]" in names


# ---------------------------------------------------------------------------
# Multi-document YAML
# ---------------------------------------------------------------------------


def test_parse_yaml_multi_document_prefixes_doc_index():
    src = (
        "kind: Pod\n"
        "metadata:\n"
        "  name: a\n"
        "---\n"
        "kind: Service\n"
        "metadata:\n"
        "  name: b\n"
    )
    units = parse_yaml(src)
    names = [u.unit_name for u in units]
    assert "doc0.kind" in names
    assert "doc1.kind" in names
    assert "doc0.metadata.name" in names
    assert "doc1.metadata.name" in names


def test_parse_yaml_single_document_no_prefix():
    src = "kind: Pod\n"
    units = parse_yaml(src)
    assert [u.unit_name for u in units] == ["kind"]


# ---------------------------------------------------------------------------
# Realistic GitHub Actions
# ---------------------------------------------------------------------------


def test_parse_yaml_github_actions_workflow():
    src = (
        "name: CI\n"
        "on: [push]\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/checkout@v4\n"
        "      - run: make test\n"
    )
    units = parse_yaml(src)
    names = {u.unit_name for u in units}
    assert {"name", "on", "jobs", "jobs.build"}.issubset(names)
    # The job content includes the steps
    job = next(u for u in units if u.unit_name == "jobs.build")
    assert "runs-on: ubuntu-latest" in job.content
    assert "actions/checkout@v4" in job.content


# ---------------------------------------------------------------------------
# Top-level sequence
# ---------------------------------------------------------------------------


def test_parse_yaml_top_level_sequence():
    src = "- a\n- b\n- c\n"
    units = parse_yaml(src)
    names = [u.unit_name for u in units]
    assert "[0]" in names
    assert "[1]" in names
    assert "[2]" in names
