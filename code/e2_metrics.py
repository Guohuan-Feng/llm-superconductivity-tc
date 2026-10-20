"""Finite E2 evaluation; no LLM calls, training, or file writes.

Missing or invalid LLM probabilities are never replaced for forced decisions.
They remain failures, with matched-control forced metrics on valid cases and a
separately named full-cohort HGB fallback system. All uncertainty resamples
association groups, retaining Stage 1's original-group row weights.
"""

from __future__ import annotations

from collections.abc import Mapping
import math

from common import np
from stats import summarize


ARMS = ("A1", "A2", "A3")
_SUCCESS = {"ok", "success", "complete", "completed", "valid"}


def _finite_number(value):
    if isinstance(value, (bool, np.bool_)) or value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _record_index(records):
    if isinstance(records, Mapping):
        indexed = {}
        for key, record in records.items():
            if not isinstance(record, Mapping):
                raise TypeError("each arm result must be a mapping")
            normalized = str(key)
            if normalized in indexed:
                raise ValueError("duplicate result id after string normalization")
            if "id" in record and str(record["id"]) != normalized:
                raise ValueError("result mapping key and record id disagree")
            indexed[normalized] = record
        return indexed
    indexed = {}
    for record in records:
        key = str(record["id"])
        if key in indexed:
            raise ValueError(f"duplicate final result for {key!r}")
        indexed[key] = record
    return indexed


def _interval(point, draws):
    draws = np.asarray(draws, dtype=float)
    valid = draws[np.isfinite(draws)]
    return {
        "point": float(point) if np.isfinite(point) else None,
        "ci95": np.percentile(valid, [2.5, 97.5]).tolist() if valid.size else [None, None],
        "n_valid_boot": int(valid.size),
        "n_undefined_boot": int(draws.size - valid.size),
    }


def mean_summary(values, groups, original_sizes, *, n_boot=5000, seed=20261016):
    """Material and group-weighted means; NaN measurements remain unavailable.

    Means and CIs use only cases with finite measurements. Original association
    group denominators are preserved when only a subset has measurements.
    """
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups).astype(str)
    original_sizes = np.asarray(original_sizes, dtype=float)
    if values.ndim != 1 or groups.shape != values.shape or original_sizes.shape != values.shape:
        raise ValueError("mean inputs must be aligned one-dimensional arrays")
    if not isinstance(n_boot, (int, np.integer)) or n_boot < 2000:
        raise ValueError("n_boot must be >= 2000")
    if not np.isfinite(original_sizes).all() or np.any(original_sizes <= 0):
        raise ValueError("original group sizes must be finite and positive")
    finite = np.isfinite(values)
    result = {
        "n": int(finite.sum()),
        "n_groups": int(np.unique(groups[finite]).size),
        "n_requested": int(values.size),
        "n_missing": int((~finite).sum()),
        "n_boot": int(n_boot),
        "bootstrap_unit": "observed association groups, uniform with replacement",
        "confidence_level": 0.95,
    }
    if not finite.any():
        empty = _interval(np.nan, np.full(n_boot, np.nan))
        return {**result, "material_mean": dict(empty), "group_weighted_mean": dict(empty)}
    values, groups, original_sizes = values[finite], groups[finite], original_sizes[finite]
    unique, inverse = np.unique(groups, return_inverse=True)
    count = np.bincount(inverse).astype(float)
    total = np.bincount(inverse, weights=values)
    weights = 1 / original_sizes
    weighted_count = np.bincount(inverse, weights=weights)
    weighted_total = np.bincount(inverse, weights=weights * values)
    rng = np.random.default_rng(seed)
    raw_draws, weighted_draws = np.empty(n_boot), np.empty(n_boot)
    for start in range(0, n_boot, 256):
        stop = min(start + 256, n_boot)
        m = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=stop - start)
        raw_draws[start:stop] = (m @ total) / (m @ count)
        weighted_draws[start:stop] = (m @ weighted_total) / (m @ weighted_count)
    return {
        **result,
        "material_mean": _interval(values.mean(), raw_draws),
        "group_weighted_mean": _interval(weighted_total.sum() / weighted_count.sum(), weighted_draws),
    }


def _resources(calls, arm, ids, groups, sizes, *, n_boot, seed):
    per_case = {key: [] for key in ids}
    arm_calls = []
    for call in calls:
        if call.get("arm") != arm:
            continue
        key = str(call.get("id"))
        if key not in per_case:
            raise ValueError(f"call for unselected id {key!r}")
        per_case[key].append(call)
        arm_calls.append(call)
    result = {"n": len(ids), "n_groups": int(np.unique(groups).size), "calls_total": len(arm_calls)}
    field_values = {}
    for field in ("input_tokens", "output_tokens", "cached_input_tokens", "elapsed_seconds"):
        values = []
        observed_values = []
        for key in ids:
            records = per_case[key]
            observed = [call.get(field) for call in records]
            complete = all(_finite_number(v) and float(v) >= 0 for v in observed)
            values.append(sum(float(v) for v in observed) if complete else np.nan)
            observed_values.append(sum(float(v) for v in observed if _finite_number(v) and float(v) >= 0))
        field_values[field] = np.asarray(values)
        stats = mean_summary(values, groups, sizes, n_boot=n_boot, seed=seed)
        stats["observed_call_total"] = float(sum(float(call[field]) for call in arm_calls
            if _finite_number(call.get(field)) and float(call[field]) >= 0))
        stats["n_calls_with_measurement"] = sum(_finite_number(call.get(field))
            and float(call[field]) >= 0 for call in arm_calls)
        stats["all_calls_measured"] = stats["n_calls_with_measurement"] == len(arm_calls)
        stats["observed_total_divided_all_cases"] = stats["observed_call_total"] / len(ids)
        stats["observed_is_lower_bound"] = not stats["all_calls_measured"]
        stats["observed_per_case_all_cases"] = mean_summary(observed_values, groups, sizes,
            n_boot=n_boot, seed=seed)
        result[f"{field}_per_case"] = stats
    result["total_tokens_per_case"] = mean_summary(
        field_values["input_tokens"] + field_values["output_tokens"], groups, sizes,
        n_boot=n_boot, seed=seed)
    result["calls_per_case"] = mean_summary([len(per_case[key]) for key in ids], groups, sizes,
        n_boot=n_boot, seed=seed)
    status_counts = {}
    for call in arm_calls:
        status = str(call.get("status", "missing_status"))
        status_counts[status] = status_counts.get(status, 0) + 1
    result["call_status_counts"] = status_counts
    result["n_failed_calls"] = sum(call.get("status") not in _SUCCESS for call in arm_calls)
    if arm_calls:
        case_lookup = {key: j for j, key in enumerate(ids)}
        call_indices = [case_lookup[str(call["id"])] for call in arm_calls]
        result["call_failure_rate"] = mean_summary(
            [float(call.get("status") not in _SUCCESS) for call in arm_calls],
            groups[call_indices], sizes[call_indices], n_boot=n_boot, seed=seed)
        result["call_failure_rate"]["denominator"] = "actual attempted calls, including retries"
    result["time_definition"] = "sum of individual call elapsed seconds per case, including failed attempts; not wall-clock throughput"
    result["token_definition"] = "sum over every attempted call for each case; case is unavailable if any attempt lacks usage"
    result["observed_usage_definition"] = "observed totals divided by all requested cases; a lower bound if any attempted call lacks measurement; complete-measurement means also reported"
    return result


def evaluate_e2(d, indices, results_by_arm, calls, controls, *, n_boot=5000, seed=20261016):
    """Return JSON-serializable E2 metrics and required paired comparisons.

    ``d``: Stage 1 full-data dict with y/group/group_size/id/hgb arrays.
    ``indices``: fixed E0 cohort indices, without duplicates.
    ``results_by_arm``: A1/A2/A3 -> final records (iterable or id->mapping).
        Required record fields: id, p (finite [0,1]), abstain (Boolean).
        Optional: status (ok/success/complete/completed/valid), selected_rules.
        Missing/invalid records remain failures. Reasons remain in call logs.
    ``calls``: all attempted calls, each {arm,id,input_tokens,output_tokens,
        elapsed_seconds}; missing resource values are reported as unavailable.
    ``controls``: mapping with R-info and 5NN arrays, either full-data or cohort
        aligned. HGB127 is taken directly from d['hgb'].
    """
    indices = np.asarray(indices, dtype=int)
    if indices.ndim != 1 or np.unique(indices).size != indices.size or not indices.size:
        raise ValueError("indices must be a nonempty one-dimensional unique cohort")
    if np.any(indices < 0) or np.any(indices >= len(d["y"])):
        raise ValueError("cohort index outside full data")
    ids = [str(value) for value in np.asarray(d["id"])[indices]]
    if len(set(ids)) != len(ids):
        raise ValueError("cohort ids must be unique")
    id_set = set(ids)
    y = np.asarray(d["y"])[indices]
    groups = np.asarray(d["group"])[indices].astype(str)
    sizes = np.asarray(d["group_size"])[indices]
    hgb = np.asarray(d["hgb"], dtype=float)[indices]
    full_sizes = {str(group): int(size) for group, size in zip(d["group"], d["group_size"])}
    control_scores = {}
    for name, values in controls.items():
        score = np.asarray(values, dtype=float)
        if score.shape == np.asarray(d["y"]).shape:
            score = score[indices]
        if score.shape != y.shape or not np.isfinite(score).all():
            raise ValueError(f"invalid control shape or score: {name}")
        control_scores[name] = score
    if "HGB127" in control_scores and not np.array_equal(control_scores["HGB127"], hgb):
        raise ValueError("provided HGB127 differs from frozen original OOF")
    control_scores["HGB127"] = hgb
    if not {"R-info", "5NN"}.issubset(control_scores):
        raise ValueError("controls must include R-info and 5NN")
    unknown_arms = set(results_by_arm) - set(ARMS)
    if unknown_arms:
        raise ValueError(f"unknown LLM arms: {unknown_arms}")
    calls = list(calls)
    if any(call.get("arm") not in ARMS for call in calls):
        raise ValueError("all call records must have arm A1/A2/A3")
    baselines = summarize(y, control_scores, groups, full_sizes, n_boot=n_boot, seed=seed)
    output = {
        "stage": 2,
        "experiment": "E2",
        "scope": "adaptive development estimates; no independent test set",
        "n": len(ids),
        "n_groups": int(np.unique(groups).size),
        "bootstrap": {"draws": int(n_boot), "seed": int(seed), "unit": "association groups", "confidence_level": 0.95},
        "baselines": baselines,
        "arms": {},
        "required_comparisons": {},
        "chance_reference": {
            "n": len(ids), "n_groups": int(np.unique(groups).size),
            "balanced_accuracy": {"point": 0.5, "ci95": [0.5, 0.5]},
            "definition": "analytical expected BA of independent fair-coin classification, not one sampled realization",
        },
        "failure_policy": "forced metrics exclude invalid or missing probabilities and use identically matched controls; full-cohort system falls back to HGB for explicit abstention or any failure",
    }
    for arm in ARMS:
        indexed = _record_index(results_by_arm.get(arm, []))
        if set(indexed) - id_set:
            raise ValueError(f"{arm}: result for unselected ids")
        p = np.full(len(ids), np.nan)
        valid = np.zeros(len(ids), dtype=bool)
        abstain = np.zeros(len(ids), dtype=bool)
        failure_reasons = {}
        for j, key in enumerate(ids):
            record = indexed.get(key)
            reason = None
            if record is None:
                reason = "missing_result"
            elif record.get("status") not in (None, *_SUCCESS):
                reason = "unsuccessful_status:" + str(record.get("status"))
            elif not _finite_number(record.get("p")) or not 0 <= float(record["p"]) <= 1:
                reason = "missing_or_invalid_probability"
            elif not isinstance(record.get("abstain"), (bool, np.bool_)):
                reason = "missing_or_invalid_abstention"
            if reason is not None:
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                continue
            valid[j] = True
            p[j] = float(record["p"])
            abstain[j] = bool(record["abstain"])
        forced_name = arm + "_forced"
        fallback_name = arm + "_HGBfallback"
        forced = summarize(y, {forced_name: p, **control_scores}, groups, full_sizes,
            mask=valid, n_boot=n_boot, seed=seed) if valid.any() else None
        fallback_mask = ~valid | abstain
        system_scores = np.where(fallback_mask, hgb, p)
        system = summarize(y, {fallback_name: system_scores, **control_scores}, groups, full_sizes,
            n_boot=n_boot, seed=seed)
        behavior = {
            "n": len(ids), "n_groups": int(np.unique(groups).size),
            "n_valid": int(valid.sum()), "n_valid_groups": int(np.unique(groups[valid]).size),
            "n_failure": int((~valid).sum()), "failure_reasons": failure_reasons,
            "n_explicit_abstention": int(abstain.sum()), "n_total_HGB_fallback": int(fallback_mask.sum()),
            "explicit_abstention_rate_all_cases": mean_summary(abstain.astype(float), groups, sizes, n_boot=n_boot, seed=seed),
            "abstention_rate_valid_cases": mean_summary(np.where(valid, abstain.astype(float), np.nan), groups, sizes, n_boot=n_boot, seed=seed),
            "failure_rate": mean_summary((~valid).astype(float), groups, sizes, n_boot=n_boot, seed=seed),
            "total_fallback_rate": mean_summary(fallback_mask.astype(float), groups, sizes, n_boot=n_boot, seed=seed),
        }
        output["arms"][arm] = {
            "forced_valid_cases": forced,
            "abstention_or_failure_HGB_fallback_all_cases": system,
            "behavior": behavior,
            "resources": _resources(calls, arm, ids, groups, sizes, n_boot=n_boot, seed=seed),
        }
        required_forced = {"A1": ("R-info",), "A2": ("5NN", "HGB127"), "A3": ()}[arm]
        if forced is not None:
            for baseline in required_forced:
                key = f"{forced_name} - {baseline}"
                output["required_comparisons"][key] = {
                    "n": forced["n"], "n_groups": forced["n_groups"],
                    **forced["paired_differences"][key],
                }
        for baseline in (("R-info", "HGB127") if arm == "A1" else ("HGB127",)):
            key = f"{fallback_name} - {baseline}"
            output["required_comparisons"][key] = {
                "n": system["n"], "n_groups": system["n_groups"],
                **system["paired_differences"][key],
            }
        if arm == "A3" and forced is not None:
            ba = forced["metrics"][forced_name]["balanced_accuracy"]
            output["required_comparisons"][f"{forced_name} - chanceBA0.5"] = {
                "n": forced["n"], "n_groups": forced["n_groups"],
                "arm_a": forced_name, "arm_b": "chanceBA0.5",
                "balanced_accuracy": {
                    **ba,
                    "point": ba["point"] - 0.5 if ba["point"] is not None else None,
                    "ci95": [value - 0.5 if value is not None else None for value in ba["ci95"]],
                },
            }
    return output
