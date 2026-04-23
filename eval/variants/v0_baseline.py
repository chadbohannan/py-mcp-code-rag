"""v0 baseline: captures the current in-tree prompt as the starting champion.

Every variant module exposes:
    ID:          stable identifier used in run filenames and CHAMPIONS.log
    DESCRIPTION: one-line human summary
    build_prompt(unit): returns the user-message string sent to the LLM

Future variants should NOT import this one; copy the function and modify.
Keeping variants self-contained prevents silent drift when the baseline
source moves.
"""

from __future__ import annotations

from mcp_rag.models import SemanticUnit

ID = "v0_baseline"
DESCRIPTION = "Frozen copy of mcp_rag/summarizer.py:_build_prompt as of the first eval run."


def build_prompt(unit: SemanticUnit) -> str:
    if unit.unit_type == "directory":
        return (
            "Summarize this directory's purpose and what it contains based on its "
            "files and subdirectories below. 2-3 sentences, terse and dense. "
            "No preamble, no headings, no bullet points.\n\n"
            f"{unit.content}"
        )
    if unit.unit_type == "module":
        return (
            "Summarize this file's purpose, key exports, and role relative to the "
            "modules it depends on. 2-3 sentences, terse and dense. "
            "No preamble, no headings, no bullet points.\n\n"
            f"{unit.content}"
        )
    return (
        f"Summarize this {unit.unit_type} in 2-3 sentences. "
        f"Say what it does and why using terse, dense natural language a developer would "
        f"search for. No preamble, no headings, no bullet points.\n\n"
        f"{unit.content}"
    )
