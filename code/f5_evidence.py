"""Read compact, hash-attributed F0--F3 evidence without refitting or bootstrap.

extract_evidence(root=None) returns JSON-safe builtins. Estimates and intervals
are copied from the completed metrics files, in proportions (multiply by 100
for percentages / percentage-point differences). Reversed paired comparisons
only negate the stored point and reverse/negate its stored interval.
This module never writes files or invokes a model, f_stats, or f3_stats.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

SUBSETS = ("all", "exclude_93", "only_93")
WEIGHTS = ("group_weighted", "unweighted")
METRICS = ("balanced_accuracy", "precision", "recall", "accuracy")
SOURCE_DIRS = {"F0": "F0_reanalysis", "F1": "F1_recall", "F2": "F2_mask",
               "F3": "F3_full3764"}
F3_METHODS = ("A2_forced", "A3_forced", "HGB127", "StrongML", "5NN")
F3_PAIRS = (("A2_forced", "A3_forced"),
            *((arm, baseline) for arm in ("A2_forced", "A3_forced")
              for baseline in ("HGB127", "StrongML", "5NN")))
STRATUM_PAIRS = tuple((arm, baseline) for arm in ("A2_forced", "A3_forced")
                      for baseline in ("HGB127", "StrongML"))
F2_PAIRS = (("A3-M1", "A3_E2"), ("A3-M2", "A3-M1"),
            ("A3-M2", "A3_E2"), ("A3-M2", "HGB127"),
            ("A3-M2", "5NN"), ("A2-M2", "A3-M2"),
            ("A2-M2", "HGB127"), ("A2-M2", "5NN"),
            ("A2-M2", "A2_E2"))


def _read_source(root, stage):
    path = root / SOURCE_DIRS[stage] / "metrics.json"
    raw = path.read_bytes()
    data = json.loads(raw)
    if stage != "F0" and data.get("status") != "complete":
        raise ValueError(f"{stage} must be complete before summarizing")
    source = {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
              "bytes": len(raw)}
    return data, source


def _stat(value):
    result = {"point": value["point"], "ci95": list(value["ci95"])}
    for number in [result["point"], *result["ci95"]]:
        if number is not None and not math.isfinite(number):
            raise ValueError("Source contains a nonfinite estimate or interval")
    # Keep sparse-bootstrap evidence without repeating zero counts everywhere.
    if value.get("n_undefined_boot", 0):
        result.update(n_valid_boot=value["n_valid_boot"],
                      n_undefined_boot=value["n_undefined_boot"])
    return result


def _pair(table, weighting, a, b, metric):
    pairs = table[weighting]["paired_differences"]
    if f"{a} - {b}" in pairs:
        return _stat(pairs[f"{a} - {b}"][metric])
    value = _stat(pairs[f"{b} - {a}"][metric])
    point, (lo, hi) = value["point"], value["ci95"]
    value["point"] = -point if point is not None else None
    value["ci95"] = [-hi if hi is not None else None,
                     -lo if lo is not None else None]
    return value


def _denominator(table):
    keys = ("n", "n_groups", "n_positive", "n_negative", "n_requested",
            "n_requested_groups", "n_missing", "n_missing_groups", "n_boot",
            "seed", "threshold", "single_group_warning")
    return {key: table[key] for key in keys if key in table}


def _summary(table, *, methods=None, pairs=(), metrics=METRICS):
    if table["n_boot"] != 5000:
        raise ValueError("F5 requires the frozen 5000-draw group bootstrap")
    methods = methods or tuple(table[WEIGHTS[0]]["metrics"])
    result = _denominator(table)
    for weighting in WEIGHTS:
        result[weighting] = {
            "weight_sum": table[weighting]["weight_sum"],
            "metrics": {arm: {metric: _stat(table[weighting]["metrics"][arm][metric])
                              for metric in metrics} for arm in methods},
            "paired_differences": {
                f"{a} - {b}": {metric: _pair(table, weighting, a, b, metric)
                                for metric in metrics} for a, b in pairs}}
    return result


def _subsets(data, **kwargs):
    return {subset: _summary(data[subset], **kwargs) for subset in SUBSETS}


def _fraction(table):
    result = _denominator(table)
    result.update({weight: _stat(table[weight]) for weight in WEIGHTS})
    return result


def _fraction_tree(tree):
    if "group_weighted" in tree and "point" in tree["group_weighted"]:
        return _fraction(tree)
    return {key: _fraction_tree(value) for key, value in tree.items()}


def _focus(data, pairs, metric):
    return {subset: {**_denominator(table),
                     **{weight: {f"{a} - {b}": _pair(table, weight, a, b, metric)
                                  for a, b in pairs} for weight in WEIGHTS}}
            for subset, table in data.items()}


def extract_evidence(root=None):
    """Return source hashes, frozen estimates/CIs and denominators for F5.

    Main subsets retain all four metrics under both weights. F3 family and
    original-group-size strata retain BA/P and A2/A3 comparisons against HGB
    and StrongML. F2 paired comparisons use the explicitly named direction;
    identity_BA_drops use earlier representation minus later representation.
    All subgroup weights retain full-cohort original component sizes.
    No output artifact is created. F4 external-evaluation status is left to
    the caller, which owns its separate collection evidence.
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    stages, sources = {}, {}
    for stage in SOURCE_DIRS:
        stages[stage], sources[stage] = _read_source(root, stage)
    f0, f1, f2, f3 = (stages[key] for key in SOURCE_DIRS)
    f0_pairs = (("A2_forced", "HGB127"), ("A3_forced", "HGB127"),
                ("A2_forced", "A3_forced"), ("A1_forced", "R-info"))
    f1_pairs = (("R-recall", "HGB127"), ("A2_forced", "HGB127"),
                ("A3_forced", "HGB127"), ("A2_forced", "A3_forced"))
    drops = (("A3_E2", "A3-M1"), ("A3-M1", "A3-M2"),
             ("A3_E2", "A3-M2"))
    evidence = {
        "schema_version": 1,
        "sources": sources,
        "units": {"estimates": "proportion", "paired_differences": "proportion; multiply by 100 for percentage points",
                  "CI": "stored percentile 95% group-bootstrap intervals; never recomputed"},
        "protocol": {"development_only": True, "n_boot": 5000,
                     "bootstrap_unit": "component_id, for both weightings",
                     "group_weights": "1/full original 3764-cohort group size, retained after all masks",
                     "unweighted": "1 per material", "forced_positive_rule": "p >= 0.5, including valid explicit abstentions",
                     "source_intervals_recomputed": False},
        "F0": {"subsets": _subsets(f0["subsets"], pairs=f0_pairs)},
        "F1": {
            "observed_counts": copy.deepcopy(f1["observed_counts"]),
            "recognition": _fraction_tree(f1["recognition"]),
            "recall_correctness": _fraction_tree(f1["recall_correctness"]),
            "classification": _subsets(f1["classification"], pairs=f1_pairs),
            "by_recognition": {label: _subsets(tables, pairs=f1_pairs,
                                              metrics=("balanced_accuracy",))
                               for label, tables in f1["by_recognition"].items()}},
        "F2": {
            "case_counts": copy.deepcopy(f2["case_counts"]),
            "subsets": _subsets(f2["subsets"], pairs=F2_PAIRS),
            "identity_BA_drops": _focus(f2["subsets"], drops, "balanced_accuracy"),
            "paired_comparisons": _focus(f2["subsets"], F2_PAIRS, "balanced_accuracy"),
            "rates": _fraction_tree(f2["rates"]),
            "limitations": copy.deepcopy(f2["limitations"])},
        "F3": {
            "subsets": _subsets(f3["subsets"], methods=F3_METHODS, pairs=F3_PAIRS),
            "precision_focus": _focus(f3["subsets"], F3_PAIRS, "precision"),
            "families": {label: _subsets(tables, methods=F3_METHODS,
                                         pairs=STRATUM_PAIRS,
                                         metrics=("balanced_accuracy", "precision"))
                         for label, tables in f3["families"].items()},
            "group_size_bins": {label: _subsets(tables, methods=F3_METHODS,
                                                pairs=STRATUM_PAIRS,
                                                metrics=("balanced_accuracy", "precision"))
                                for label, tables in f3["group_size_bins"].items()},
            "a15_component93_overlap": copy.deepcopy(f3["a15_component93_overlap"]),
            "family_definition": copy.deepcopy(f3["family_definition"]),
            "case_counts": copy.deepcopy(f3["case_counts"]),
            "rates": _fraction_tree(f3["rates"]),
            "fixed300_reuse": copy.deepcopy(f3["fixed300_reuse"])},
        "interpretation_limits": [
            "Self-reported recognition is not verified pretraining exposure; only three real materials self-report no recognition.",
            "F2 jointly changes identity, quantization, representation and prompt: performance drops are not a causal percentage due to memory.",
            "A3-M2 gives p=0.5 and abstains in all 300 cases; threshold-forced all-positive results do not express confident positive predictions.",
            "Equal input features do not equalize learned supervision: HGB uses all outer-TRAIN labels, whereas A2-M2 sees five neighbour labels.",
            "A2-A3 CIs containing zero mean no detected additional overall benefit, not proof of zero contribution or equivalence.",
            "Family/size comparisons are descriptive without multiple-comparison correction; sparse/single-group intervals cannot establish generalization.",
            "The strict SG223 binary-A3B A15 stratum is not the broad doped A15 family; only 2 of component 93's 114 rows meet it.",
            "F0/F1/F2 fixed-300 and F3 full-3764 results are development evidence, not independent external tests."]}
    # Validate the public interface without modifying any source dictionary.
    json.dumps(evidence, ensure_ascii=False, allow_nan=False)
    return evidence


if __name__ == "__main__":
    print(json.dumps(extract_evidence(), ensure_ascii=False, allow_nan=False))
