"""Stage 3 read-only cohort statistics; no fitting, scientific LLM calls, or writes.

The entry point returns JSON-safe objects for the finite E3/E4 runner. Families
use the frozen E2 occupancy composition, in the user-specified precedence.
Every subset keeps the full 3764-row association-group denominators.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter

from common import ROOT, load_data, np
from stats import summarize

FAMILIES = ("cuprate", "iron_based", "hydride", "other", "unknown")
FAMILY_DEFINITION = {
    "priority": list(FAMILIES),
    "cuprate": "positive occupancy of both Cu and O",
    "iron_based": "positive Fe occupancy and at least one of As/P/Se/Te; not cuprate",
    "hydride": "H atomic fraction >=0.5; neither of the preceding families",
    "other": "all remaining valid supplied compositions",
    "unknown": "composition absent or invalid, never silently assigned other",
    "source": "E2/config.json cases[id].material.composition; proxy-structure occupancy composition",
}


def material_family(material):
    """Return fixed family and any missing-composition diagnostic."""
    composition = material.get("composition") if isinstance(material, dict) else None
    if not isinstance(composition, list) or not composition:
        return "unknown", "missing/non-list/empty composition"
    fractions = {}
    for entry in composition:
        try:
            symbol = entry["element"]
            amount = entry["atomic_fraction"]
            if (not isinstance(symbol, str) or not symbol
                    or isinstance(amount, bool)):
                raise ValueError("invalid symbol or Boolean fraction")
            amount = float(amount)
            if not math.isfinite(amount) or amount < 0:
                raise ValueError("nonfinite/negative fraction")
            if symbol in fractions:
                raise ValueError("duplicate element")
            fractions[symbol] = amount
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            return "unknown", f"invalid composition entry: {exc}"
    if not math.isclose(math.fsum(fractions.values()), 1, abs_tol=2e-8):
        return "unknown", "atomic fractions do not sum to one"
    elements = {symbol for symbol, amount in fractions.items() if amount > 0}
    if {"Cu", "O"}.issubset(elements):
        return "cuprate", None
    if "Fe" in elements and elements.intersection({"As", "P", "Se", "Te"}):
        return "iron_based", None
    if fractions.get("H", 0) >= .5:
        return "hydride", None
    return "other", None


def _read_predictions(path, data):
    lookup = {str(key): j for j, key in enumerate(data["id"])}
    arms, counts = {}, Counter()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key, arm = str(row["id"]), str(row["arm"])
            if key not in lookup:
                raise ValueError(f"{path.name}: unexpected material {key}")
            j = lookup[key]
            if (int(row["fold"]) != int(data["fold"][j])
                    or str(row["group"]) != str(data["group"][j])):
                raise ValueError(f"{path.name}: identity mismatch {arm}/{key}")
            scores = arms.setdefault(arm, {})
            if key in scores:
                raise ValueError(f"{path.name}: duplicate prediction {arm}/{key}")
            score = float(row["score"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"{path.name}: nonfinite or invalid probability {arm}/{key}")
            if int(row["y_pred"]) != int(score >= .5):
                raise ValueError(f"{path.name}: threshold mismatch {arm}/{key}")
            scores[key] = score
            counts[arm] += 1
    return arms, dict(counts)


def _independent_point_check(summary, y, scores, groups, full_sizes, mask):
    """Recompute points with scalar math rather than stats.py's vector code."""
    selected = [j for j, keep in enumerate(mask) if keep]
    evidence = {}
    for arm, probabilities in scores.items():
        cells = {name: [] for name in ("tn", "fp", "fn", "tp")}
        for j in selected:
            truth = bool(y[j])
            prediction = bool(probabilities[j] >= .5)
            name = ("tp" if prediction else "fn") if truth else ("fp" if prediction else "tn")
            cells[name].append(1 / full_sizes[str(groups[j])])
        tn, fp, fn, tp = (math.fsum(cells[name]) for name in ("tn", "fp", "fn", "tp"))
        recall = tp / (tp + fn) if tp + fn else None
        specificity = tn / (tn + fp) if tn + fp else None
        points = {
            "balanced_accuracy": (recall + specificity) / 2
                if recall is not None and specificity is not None else None,
            "precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": recall,
            "accuracy": (tp + tn) / (tn + fp + fn + tp),
        }
        for metric, point in points.items():
            observed = summary["metrics"][arm][metric]["point"]
            if not ((point is None and observed is None)
                    or (point is not None and observed is not None
                        and math.isclose(point, observed, abs_tol=2e-13))):
                raise ValueError(f"Independent scalar check failed: {arm}/{metric}")
        evidence[arm] = {"weighted_confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
                         "points": points}
    return evidence


def evaluate_stage3(d=None, config=None, *, n_boot=5000, seed=20261016):
    """Return family tables and E4 matched-cohort comparisons, all JSON-safe.

    ``d`` may be the existing ``common.load_data`` result and ``config`` the
    parsed frozen E2 configuration. Predictions are always read from the saved
    CSV files; deterministic models are never refitted.
    """
    d = load_data() if d is None else d
    config = json.loads((ROOT / "E2/config.json").read_text(encoding="utf-8")) if config is None else config
    indices = np.asarray(config["indices"], dtype=int)
    if len(indices) != 300 or len(np.unique(indices)) != 300:
        raise ValueError("Stage 3 requires the fixed 300 unique E2 materials")
    ids = [str(d["id"][j]) for j in indices]
    y = d["y"][indices]
    groups = d["group"][indices].astype(str)
    full_sizes = {str(g): int(n) for g, n in zip(d["group"], d["group_size"])}
    if len(np.unique(groups)) != 81 or set(config["cases"]) != set(ids):
        raise ValueError("Frozen E2 cohort identity or association-group count differs")
    family_by_id, warnings = {}, {}
    for key, j in zip(ids, indices):
        case = config["cases"][key]
        if (case["id"] != key or int(case["i"]) != int(j)
                or str(case["group"]) != str(d["group"][j])
                or int(case["fold"]) != int(d["fold"][j])):
            raise ValueError(f"Frozen E2 identity mismatch {key}")
        family_by_id[key], diagnostic = material_family(case.get("material"))
        if diagnostic:
            warnings[key] = diagnostic
    families = np.asarray([family_by_id[key] for key in ids])
    e1, e1_counts = _read_predictions(ROOT / "E1/predictions.csv", d)
    e2, e2_counts = _read_predictions(ROOT / "E2/predictions.csv", d)
    all_data_ids = set(map(str, d["id"]))
    if any(set(values) != all_data_ids for values in e1.values()):
        raise ValueError("E1 saved prediction arm does not cover full original cohort")
    if any(set(values) != set(ids) for values in e2.values()):
        raise ValueError("E2 saved prediction arm does not cover matched 300")
    scores = {name: np.asarray([e2[name][key] for key in ids]) for name in
              ("A2_forced", "A2_HGBfallback", "A1_forced", "A1_HGBfallback", "HGB127", "R-info")}
    for name in ("Tree3", "Tree4"):
        scores[name] = np.asarray([e1[name][key] for key in ids])
    if (not np.array_equal(scores["HGB127"], d["hgb"][indices])
            or not np.array_equal(scores["HGB127"], np.asarray([e1["HGB127"][key] for key in ids]))
            or not np.array_equal(scores["R-info"], np.asarray([e1["R-info"][key] for key in ids]))
            or not np.array_equal(scores["R-info"], np.asarray([config["cases"][key]["rinfo_score"] for key in ids]))):
        raise ValueError("Frozen control predictions differ across stages or input data")
    all_mask = np.ones(len(ids), dtype=bool)
    same_cohort = summarize(y, scores, groups, full_sizes, n_boot=n_boot, seed=seed)
    independent = {"all300": _independent_point_check(same_cohort, y, scores, groups, full_sizes, all_mask)}
    family_metrics = {}
    for family in FAMILIES:
        mask = families == family
        n = int(mask.sum())
        n_groups = int(len(np.unique(groups[mask])))
        counts = {"positive": int(np.sum(y[mask] == 1)), "negative": int(np.sum(y[mask] == 0))}
        if n == 0:
            family_metrics[family] = {"n": 0, "n_groups": 0, "class_counts": counts,
                "status": "absent_family", "metrics": {}, "paired_differences": {},
                "note": "No cohort rows; no metrics or CI are defined."}
            continue
        keep = {name: scores[name] for name in
                ("A2_forced", "A2_HGBfallback", "A1_forced", "HGB127")}
        result = summarize(y, keep, groups, full_sizes, mask=mask, n_boot=n_boot, seed=seed)
        result.update(class_counts=counts, status="single_true_class" if min(counts.values()) == 0 else "both_true_classes",
            note="Balanced accuracy is undefined unless both true classes occur; finite bootstrap draws only are used for CI.")
        family_metrics[family] = result
        independent[family] = _independent_point_check(result, y, keep, groups, full_sizes, mask)
    errors = {}
    for name in ("A1_forced", "A2_forced", "A2_HGBfallback", "HGB127"):
        bad = (scores[name] >= .5) != y
        errors[name] = {"n": len(ids), "n_groups": int(len(np.unique(groups))),
            "n_errors": int(bad.sum()), "n_error_groups": int(len(np.unique(groups[bad]))),
            "error_ids": [key for key, mismatch in zip(ids, bad) if mismatch],
            "by_family": {family: {"n": int(np.sum(families == family)),
                "n_groups": int(len(np.unique(groups[families == family]))),
                "n_errors": int(np.sum(bad & (families == family))),
                "n_error_groups": int(len(np.unique(groups[bad & (families == family)])))}
                for family in FAMILIES}}
    # Verify the Stage 2 headline points from independently loaded saved CSVs.
    saved = json.loads((ROOT / "E2/metrics.json").read_text(encoding="utf-8"))
    for name, reference in {
        "A2_forced": saved["arms"]["A2"]["forced_valid_cases"],
        "A2_HGBfallback": saved["arms"]["A2"]["abstention_or_failure_HGB_fallback_all_cases"],
        "A1_forced": saved["arms"]["A1"]["forced_valid_cases"],
        "A1_HGBfallback": saved["arms"]["A1"]["abstention_or_failure_HGB_fallback_all_cases"],
        "HGB127": saved["baselines"], "R-info": saved["baselines"],
    }.items():
        for metric, estimate in same_cohort["metrics"][name].items():
            prior = reference["metrics"][name][metric]["point"]
            if not math.isclose(estimate["point"], prior, abs_tol=2e-13):
                raise ValueError(f"Stage 2 saved point differs: {name}/{metric}")
    result = {
        "stage": 3, "scope": "adaptive development data; fixed E2 cohort only",
        "n": len(ids), "n_groups": int(len(np.unique(groups))),
        "primary_llm_arm": "A2_forced", "family_definition": FAMILY_DEFINITION,
        "family_by_id": family_by_id, "composition_warnings": warnings,
        "same300_comparison": same_cohort, "family_metrics": family_metrics,
        "errors": errors,
        "validation": {"csv_arm_row_counts": {"E1": e1_counts, "E2": e2_counts},
            "identity_threshold_control_point_checks": "passed",
            "independent_scalar_points": independent,
            "new_model_fits": 0, "new_scientific_llm_calls": 0},
        "interpretation_limits": [
            "Family strata may contain a single true class, making balanced accuracy undefined.",
            "Group counts can overlap across family strata if original groups span families; do not sum them.",
            "Precision uses zero_division=0, consistent with E0/E1/E2.",
            "Only the same 300 materials are comparable to A2; all3764 E1 values are a different cohort.",
            "Original association-group row denominators are preserved inside all family subsets.",
        ],
    }
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    return result


if __name__ == "__main__":
    result = evaluate_stage3()
    compact = {"n": result["n"], "n_groups": result["n_groups"],
               "same300": result["same300_comparison"]["metrics"],
               "families": result["family_metrics"], "errors": result["errors"],
               "validation": result["validation"]["identity_threshold_control_point_checks"]}
    print(json.dumps(compact, ensure_ascii=False, allow_nan=False))
