"""Unit tests for the slice-scoped variant authoring helpers."""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

from eval.harness.slices import SLICES
from eval.variants.make_candidate import RESERVED_IDS, make_candidate
from mcp_rag.models import SemanticUnit


def _unit(unit_type: str, content: str = "def f(): pass") -> SemanticUnit:
    return SemanticUnit(
        unit_type=unit_type,
        unit_name="f",
        content=content,
        char_offset=0,
    )


def _leaf_prompts() -> dict[str, str]:
    return {t: f"Summarize this {t} tersely." for t in SLICES["leaf_code"]}


def _cleanup(cid: str, path):
    sys.modules.pop(f"eval.variants.{cid}", None)
    if path.exists():
        path.unlink()


def test_make_candidate_writes_loadable_variant():
    cid = "t_leaf_candidate"
    path = make_candidate(
        candidate_id=cid,
        slice_name="leaf_code",
        prompts=_leaf_prompts(),
        description="test candidate",
        parent_receipt="eval/runs/v0_baseline-123.json",
        overwrite=True,
    )
    try:
        sys.modules.pop(f"eval.variants.{cid}", None)
        mod = importlib.import_module(f"eval.variants.{cid}")
        assert mod.ID == cid
        assert mod.SLICE == "leaf_code"
        assert mod.PARENT_RECEIPT == "eval/runs/v0_baseline-123.json"
        assert mod.GENERATED_AT  # ISO timestamp set
        out = mod.build_prompt(_unit("function"))
        assert "Summarize this function tersely." in out
        assert "def f(): pass" in out

        with pytest.raises(RuntimeError, match="no prompt for unit_type"):
            mod.build_prompt(_unit("paragraph", "hello"))
    finally:
        _cleanup(cid, path)


def test_make_candidate_without_parent_emits_none():
    cid = "t_no_parent"
    path = make_candidate(
        candidate_id=cid,
        slice_name="leaf_code",
        prompts=_leaf_prompts(),
        overwrite=True,
    )
    try:
        sys.modules.pop(f"eval.variants.{cid}", None)
        mod = importlib.import_module(f"eval.variants.{cid}")
        assert mod.PARENT_RECEIPT is None
    finally:
        _cleanup(cid, path)


def test_generated_file_is_ruff_clean():
    cid = "t_ruff_clean"
    # Include ugly strings that could tempt the formatter to complain.
    prompts = {
        t: f'''Quote "inside" and 'also' with \\ backslash for {t}.'''
        for t in SLICES["leaf_code"]
    }
    path = make_candidate(
        candidate_id=cid,
        slice_name="leaf_code",
        prompts=prompts,
        description="ruff format check",
        overwrite=True,
    )
    try:
        check = subprocess.run(
            ["uv", "run", "ruff", "check", str(path)],
            capture_output=True,
            text=True,
        )
        assert check.returncode == 0, check.stdout + check.stderr
        fmt = subprocess.run(
            ["uv", "run", "ruff", "format", "--check", str(path)],
            capture_output=True,
            text=True,
        )
        assert fmt.returncode == 0, fmt.stdout + fmt.stderr
    finally:
        _cleanup(cid, path)


def test_make_candidate_rejects_mismatched_prompt_keys():
    with pytest.raises(ValueError, match="missing"):
        make_candidate(
            candidate_id="t_incomplete",
            slice_name="leaf_code",
            prompts={"function": "x"},
            overwrite=True,
        )


def test_make_candidate_rejects_bad_identifier():
    with pytest.raises(ValueError, match="identifier"):
        make_candidate(
            candidate_id="9-bad",
            slice_name="leaf_code",
            prompts=_leaf_prompts(),
        )


def test_make_candidate_rejects_unknown_slice():
    with pytest.raises(ValueError, match="unknown slice"):
        make_candidate(
            candidate_id="t_unknown",
            slice_name="not_a_slice",
            prompts={},
        )


@pytest.mark.parametrize("reserved", sorted(RESERVED_IDS))
def test_make_candidate_rejects_reserved_ids(reserved):
    with pytest.raises(ValueError, match="reserved"):
        make_candidate(
            candidate_id=reserved,
            slice_name="leaf_code",
            prompts=_leaf_prompts(),
        )
