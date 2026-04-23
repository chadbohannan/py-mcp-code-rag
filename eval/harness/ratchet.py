"""Per-slice promotion decision.

The champion is a JSON mapping ``{slice: run_receipt_path}``. A
candidate can win any subset of slices in a single ratchet invocation:
each slice is evaluated independently and promoted only when its
held-out gain is statistically significant AND none of the (global)
guardrails regress.

Usage:
    uv run python -m eval.harness.ratchet \\
        --candidate eval/runs/v1_banned_preamble-<ts>.json

    # first-ever run: bootstrap every slice to the candidate
    uv run python -m eval.harness.ratchet \\
        --candidate eval/runs/v0_baseline-<ts>.json --bootstrap

Per-slice promotion rule:
    heldout.by_slice[S].mrr_at_10(candidate) > champion(S) with p < PRIMARY_P
    AND heldout.by_slice[S].n >= MIN_HELDOUT_SLICE_N
    AND adversarial.fpr(candidate) <= adversarial.fpr(champion(S)) + ADV_FPR_BUDGET
    AND smoke.pass_rate(candidate) >= smoke.pass_rate(champion(S))
    AND dev.by_slice[S].mrr_at_10(candidate) >= champion(S) - DEV_REGRESSION_BUDGET
    AND intrinsic.by_slice[S].banned_preamble_rate(cand)
            <= champion(S) + BANNED_PREAMBLE_BUDGET
    AND intrinsic.by_slice[S].name_restatement_mean(cand)
            <= champion(S) + NAME_RESTATEMENT_BUDGET
    AND intrinsic.by_slice[S].symbol_grounding_mean(cand)
            >= champion(S) - SYMBOL_GROUNDING_BUDGET   (skipped if N/A)

Guardrails are checked against the CURRENT CHAMPION OF THAT SLICE, not
against a global champion. This matters because different slices may have
different champions. Intrinsic guardrails catch reward-hacking variants
that win MRR by producing fluent-but-ungrounded or name-restating text.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from eval.harness.slices import MIN_HELDOUT_SLICE_N, SLICES

EVAL_DIR = Path(__file__).resolve().parent.parent
CHAMPION_FILE = EVAL_DIR / "CHAMPION"
CHAMPIONS_LOG = EVAL_DIR / "CHAMPIONS.log"

PRIMARY_P = 0.05
ADV_FPR_BUDGET = 0.00
DEV_REGRESSION_BUDGET = 0.02
BOOTSTRAP_ITERS = 10_000
BOOTSTRAP_SEED = 0

# Intrinsic guardrails. These catch reward-hacking: a variant that raises
# MRR but tanks text quality (hallucinated identifiers, banned preambles,
# name restatement) should not be promoted. Budgets are slack, not slope —
# small drift is tolerated so noise on small slices doesn't block gains.
BANNED_PREAMBLE_BUDGET = 0.02      # candidate rate <= champion rate + budget
NAME_RESTATEMENT_BUDGET = 0.02     # same direction
SYMBOL_GROUNDING_BUDGET = 0.02     # candidate >= champion - budget (drops are bad)


def _load_receipt(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_champion() -> dict[str, str]:
    if not CHAMPION_FILE.exists():
        return {}
    text = CHAMPION_FILE.read_text().strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    # legacy single-path format: treat as champion of every slice
    return {slice_name: text for slice_name in SLICES}


def _save_champion(mapping: dict[str, str]) -> None:
    CHAMPION_FILE.write_text(json.dumps(mapping, indent=2) + "\n")


def _slice_reciprocal_ranks(
    receipt: dict[str, Any], split: str, slice_name: str
) -> list[float]:
    """Extract per-query reciprocal ranks for queries that contribute to this slice."""
    pq = receipt["splits"][split]["per_query"]
    return [
        q["slice_ranks"][slice_name]["reciprocal_rank"]
        for q in pq
        if slice_name in q.get("slice_ranks", {})
    ]


def _query_ids_in_slice(
    receipt: dict[str, Any], split: str, slice_name: str
) -> list[str]:
    pq = receipt["splits"][split]["per_query"]
    return [q["id"] for q in pq if slice_name in q.get("slice_ranks", {})]


def _bootstrap_diff(
    champ_rrs: list[float],
    cand_rrs: list[float],
    iters: int = BOOTSTRAP_ITERS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float, float]:
    if len(champ_rrs) != len(cand_rrs):
        raise ValueError(
            f"paired bootstrap requires same query set: "
            f"{len(champ_rrs)} vs {len(cand_rrs)}"
        )
    n = len(champ_rrs)
    if n == 0:
        return 0.0, 0.0, 0.0, 1.0

    rng = random.Random(seed)
    diffs = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        c_mean = sum(cand_rrs[i] for i in idx) / n
        p_mean = sum(champ_rrs[i] for i in idx) / n
        diffs.append(c_mean - p_mean)
    diffs.sort()
    mean_diff = sum(diffs) / iters
    ci_low = diffs[int(0.025 * iters)]
    ci_high = diffs[int(0.975 * iters)]
    p_value = sum(1 for d in diffs if d <= 0) / iters
    return mean_diff, ci_low, ci_high, p_value


def decide_slice(
    slice_name: str,
    champion: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[bool, list[str], dict[str, Any]]:
    reasons: list[str] = []
    diag: dict[str, Any] = {"slice": slice_name}

    # Same queries must contribute to this slice in both receipts, else
    # the paired bootstrap is undefined. This is guaranteed when both
    # receipts scored the same query files at the same index schema.
    champ_ids = _query_ids_in_slice(champion, "heldout", slice_name)
    cand_ids = _query_ids_in_slice(candidate, "heldout", slice_name)
    if champ_ids != cand_ids:
        reasons.append(
            f"heldout query set for slice differs between receipts "
            f"(champ={len(champ_ids)}, cand={len(cand_ids)}); "
            f"re-score against the current query set."
        )
        return False, reasons, diag

    n = len(cand_ids)
    diag["heldout_n"] = n
    if n < MIN_HELDOUT_SLICE_N:
        reasons.append(
            f"heldout n={n} < MIN_HELDOUT_SLICE_N={MIN_HELDOUT_SLICE_N}; "
            "slice ratchet disarmed"
        )
        return False, reasons, diag

    champ_rrs = _slice_reciprocal_ranks(champion, "heldout", slice_name)
    cand_rrs = _slice_reciprocal_ranks(candidate, "heldout", slice_name)
    mean_diff, ci_low, ci_high, p = _bootstrap_diff(champ_rrs, cand_rrs)
    diag["heldout_mrr_diff"] = {
        "mean": mean_diff,
        "ci95": [ci_low, ci_high],
        "p_value": p,
        "champion": champion["splits"]["heldout"]["by_slice"][slice_name]["mrr_at_10"],
        "candidate": candidate["splits"]["heldout"]["by_slice"][slice_name]["mrr_at_10"],
    }
    if p >= PRIMARY_P:
        reasons.append(
            f"heldout MRR@10 gain not significant (p={p:.3f} >= {PRIMARY_P})"
        )

    # Global guardrails — per-slice promotion still must not regress them.
    adv_champ = champion["splits"]["adversarial"]["adversarial_fpr"]
    adv_cand = candidate["splits"]["adversarial"]["adversarial_fpr"]
    diag["adversarial_fpr"] = {"champion": adv_champ, "candidate": adv_cand}
    if adv_cand > adv_champ + ADV_FPR_BUDGET:
        reasons.append(
            f"adversarial FPR regressed: {adv_cand:.3f} > "
            f"{adv_champ:.3f} + {ADV_FPR_BUDGET}"
        )

    smoke_champ = champion["splits"]["smoke"]["pass_rate"]
    smoke_cand = candidate["splits"]["smoke"]["pass_rate"]
    diag["smoke_pass_rate"] = {"champion": smoke_champ, "candidate": smoke_cand}
    if smoke_cand < smoke_champ:
        reasons.append(
            f"smoke pass_rate regressed: {smoke_cand:.3f} < {smoke_champ:.3f}"
        )

    dev_champ = champion["splits"]["dev"]["by_slice"][slice_name]["mrr_at_10"]
    dev_cand = candidate["splits"]["dev"]["by_slice"][slice_name]["mrr_at_10"]
    diag["dev_slice_mrr"] = {"champion": dev_champ, "candidate": dev_cand}
    if dev_cand < dev_champ - DEV_REGRESSION_BUDGET:
        reasons.append(
            f"dev MRR@10 for slice regressed beyond budget: "
            f"{dev_cand:.3f} < {dev_champ:.3f} - {DEV_REGRESSION_BUDGET}"
        )

    # Intrinsic guardrails — per-slice. Missing intrinsic blocks (older
    # receipts predating intrinsic.py) are treated as "no data, skip".
    champ_intr = champion.get("intrinsic", {}).get("by_slice", {}).get(slice_name)
    cand_intr = candidate.get("intrinsic", {}).get("by_slice", {}).get(slice_name)
    if champ_intr and cand_intr:
        diag["intrinsic"] = {"champion": champ_intr, "candidate": cand_intr}

        bp_c, bp_k = champ_intr["banned_preamble_rate"], cand_intr["banned_preamble_rate"]
        if bp_k > bp_c + BANNED_PREAMBLE_BUDGET:
            reasons.append(
                f"banned_preamble_rate regressed: "
                f"{bp_k:.3f} > {bp_c:.3f} + {BANNED_PREAMBLE_BUDGET}"
            )

        nr_c, nr_k = champ_intr["name_restatement_mean"], cand_intr["name_restatement_mean"]
        if nr_k > nr_c + NAME_RESTATEMENT_BUDGET:
            reasons.append(
                f"name_restatement_mean regressed: "
                f"{nr_k:.3f} > {nr_c:.3f} + {NAME_RESTATEMENT_BUDGET}"
            )

        sg_c = champ_intr.get("symbol_grounding_mean")
        sg_k = cand_intr.get("symbol_grounding_mean")
        if sg_c is not None and sg_k is not None:
            if sg_k < sg_c - SYMBOL_GROUNDING_BUDGET:
                reasons.append(
                    f"symbol_grounding_mean regressed: "
                    f"{sg_k:.3f} < {sg_c:.3f} - {SYMBOL_GROUNDING_BUDGET}"
                )

    return (len(reasons) == 0), reasons, diag


def _bootstrap_all_slices(candidate_path: Path, candidate: dict[str, Any]) -> None:
    mapping = {slice_name: str(candidate_path) for slice_name in SLICES}
    _save_champion(mapping)
    line = (
        f"{candidate['timestamp']}\tBOOTSTRAP\t{candidate['variant_id']}\t"
        f"slices={','.join(SLICES)}\t"
        f"run={candidate_path}\n"
    )
    with CHAMPIONS_LOG.open("a") as f:
        f.write(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Install candidate as champion of every slice (first-run only).",
    )
    args = parser.parse_args(argv)

    candidate = _load_receipt(args.candidate)
    champion_map = _load_champion()

    if not champion_map:
        if not args.bootstrap:
            print(
                "No champion yet. Rerun with --bootstrap to install this "
                "receipt as the starting champion for every slice.",
            )
            return 2
        _bootstrap_all_slices(args.candidate, candidate)
        print(f"bootstrapped all slices to: {args.candidate}")
        return 0

    if args.bootstrap:
        print("--bootstrap refused: champion already exists.")
        return 2

    print(f"candidate: {candidate['variant_id']}  ({args.candidate.name})")
    print()

    new_map = dict(champion_map)
    promoted: list[str] = []
    rejected: list[tuple[str, list[str]]] = []

    for slice_name in SLICES:
        champ_path = Path(champion_map.get(slice_name, ""))
        if not champ_path or not champ_path.exists():
            rejected.append((slice_name, ["missing champion receipt for slice"]))
            continue
        champion = _load_receipt(champ_path)
        promote, reasons, diag = decide_slice(slice_name, champion, candidate)
        print(f"[{slice_name}] champion={champion['variant_id']}")
        print(json.dumps(diag, indent=2))
        if promote:
            new_map[slice_name] = str(args.candidate)
            promoted.append(slice_name)
            with CHAMPIONS_LOG.open("a") as f:
                f.write(
                    f"{candidate['timestamp']}\tPROMOTE\t{slice_name}\t"
                    f"{candidate['variant_id']}\t"
                    f"from={champion['variant_id']}\t"
                    f"heldout_diff={diag['heldout_mrr_diff']['mean']:+.4f}\t"
                    f"p={diag['heldout_mrr_diff']['p_value']:.4f}\t"
                    f"run={args.candidate}\n"
                )
            print(f"  -> PROMOTE slice={slice_name}")
        else:
            rejected.append((slice_name, reasons))
            print("  -> reject:")
            for r in reasons:
                print(f"     - {r}")
        print()

    if promoted:
        _save_champion(new_map)

    print(f"promoted slices: {promoted or '(none)'}")
    if rejected:
        print(f"rejected slices: {[s for s, _ in rejected]}")
    return 0 if promoted else 1


if __name__ == "__main__":
    raise SystemExit(main())
