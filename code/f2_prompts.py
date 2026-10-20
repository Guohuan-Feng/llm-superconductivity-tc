"""Frozen identity-ablation prompts; no calls or writes on import."""
from __future__ import annotations

import json
from e2_prompts import STATIC_PREFIX as E2_PREFIX
from e2_packets import round_prompt_numbers


STATIC_PREFIX = E2_PREFIX.replace(
    "Use only the supplied scientific evidence and your prior chemical knowledge.",
    "Use only the supplied data to make the prediction.",
)
assert "prior chemical knowledge" not in STATIC_PREFIX

_INSTRUCTIONS = {
    "A3-M1": (
        "Arm A3: data only. You receive only an anonymized material's composition ratios, "
        "coarse elemental-property bins and available structural summary; no rules, neighbors, measured Tc, or labels. "
        "E1, E2, ... are identifiers local to this material, randomly assigned, not chemical symbols. "
        "For each supplied elemental property, bins1-5 increase from low to high; unknown means unavailable. "
        "The valence count follows the supplied table's occupied s+p+d+f convention. "
        "Missing space group is genuinely unavailable. Directly output p(Tc>10K), abstain, and a scientific reason "
        "of at most80 characters. selected_rule_ids=[], action=decide; do not request evidence."
    ),
    "A3-M2": (
        "Arm A3: data only. You receive only the material's X127 feature vector, with columns anonymized as f1-f127 "
        "and standardized using StandardScaler fitted exclusively on this outer fold TRAIN. "
        "No feature names, rules, neighbors, measured Tc, labels or additional identity are supplied. "
        "Directly output p(Tc>10K), abstain, and a scientific reason of at most80 characters. "
        "selected_rule_ids=[], action=decide; do not request evidence."
    ),
    "A2-M2": (
        "Arm A2: direct prediction with numeric features and neighbors. You receive the material's standardized "
        "X127 feature vector and five nearest TRAIN materials under Euclidean distance after X127 standardization "
        "fitted exclusively on this outer fold TRAIN. The columns are anonymized as f1-f127; each neighbor uses "
        "the same feature order and scaling and includes only its features and Tc>10K label. "
        "Weigh the supplied data and uncertainty, then directly output p(Tc>10K), abstain, and a reason "
        "of at most80 characters. selected_rule_ids=[]; do not request more evidence; action=decide."
    ),
}


def make_prompt(arm, payload, *, round_number=1, repair=False):
    assert arm in _INSTRUCTIONS and round_number in (1, 2)
    instruction = _INSTRUCTIONS[arm]
    if repair:
        instruction += (
            " This is the FINAL allowed round, a format/transport repair of the same scientific input. "
            "Return a valid decision JSON: finite numeric p in[0,1] even if abstain=true, "
            "action=decide, selected_rule_ids=[], request_rule_ids=[], request_counterexamples=false."
        )
    scientific = json.dumps(round_prompt_numbers(payload), ensure_ascii=False,
                            separators=(",", ":"), allow_nan=False)
    return STATIC_PREFIX + "\n" + instruction + "\nRound " + str(round_number) + "\nScientific inputs:\n" + scientific
