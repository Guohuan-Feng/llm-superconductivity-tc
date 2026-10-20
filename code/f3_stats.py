"""Finite full-cohort F3 stratified statistics; no I/O, fits, or model calls.

All outcome estimates delegate to the frozen F-series statistics. This module
only constructs the predeclared masks and exposes requested/valid denominators.
"""
from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations

from common import np
from f_stats import summarize

FAMILIES = ("copper", "iron", "hydride", "A15", "other")
SIZE_BINS = ("1", "2-5", "6-20", ">20")
SUBSETS = ("all", "exclude_93", "only_93")


def _boolean_mask(value, n, label):
    value = np.asarray(value)
    if value.shape != (n,) or value.dtype != np.bool_:
        raise ValueError(f"{label} must be an aligned Boolean array")
    return value


def _counts(mask, y, groups):
    n = int(mask.sum())
    positives = int(y[mask].sum())
    return {"n": n, "n_groups": int(len(np.unique(groups[mask]))),
            "n_positive": positives, "n_negative": n - positives}


def build_f3_statistics(y, scores_by_arm: Mapping[str, object], groups,
                        full_group_sizes, *, family_masks,
                        group_size_values=None, valid_mask=None, a15_mask=None,
                        n_boot=5000, seed=20261016):
    """Return complete F3 statistics in memory, leaving arguments unchanged.

    Arrays align to all 3764 source-order rows (smaller arrays are supported for
    deterministic checks). Scores may be NaN where final responses are missing;
    every comparison uses the same finite rows across all supplied methods.
    valid_mask optionally adds protocol-validity requirements; it never imputes
    a missing forced decision or removes an explicit valid abstention.

    family_masks must contain exactly the five exclusive, exhaustive families.
    The caller derives them from frozen material metadata, never from outcomes.
    Optional a15_mask is the independent raw structure criterion before family
    precedence; its overlap with component 93 is reported separately. Original
    group sizes determine weights and size strata, including after filtering.

    Output subsets/families/group_size_bins hold f_stats.summarize results for
    all/exclude_93/only_93, augmented with requested and missing row/group counts.
    Every unordered method pair receives paired CIs for all four metrics.
    """
    y = np.asarray(y)
    if y.ndim != 1 or not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must be a one-dimensional binary array")
    n = len(y)
    groups = np.asarray(groups).astype(str)
    if groups.shape != (n,):
        raise ValueError("groups must align to y")
    if not isinstance(scores_by_arm, Mapping) or not scores_by_arm:
        raise ValueError("named method scores are required")
    names, scores = list(scores_by_arm), {}
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("method names must be nonempty strings")
    for name, values in scores_by_arm.items():
        values = np.asarray(values, dtype=float)
        if values.shape != (n,):
            raise ValueError(f"scores must align to y: {name}")
        if np.any(np.isinf(values)) or np.any((values[np.isfinite(values)] < 0) | (values[np.isfinite(values)] > 1)):
            raise ValueError(f"scores must be probabilities or missing NaN: {name}")
        scores[name] = values
    protocol_valid = np.ones(n, dtype=bool) if valid_mask is None else _boolean_mask(valid_mask, n, "valid_mask")
    common = protocol_valid & np.logical_and.reduce([np.isfinite(scores[name]) for name in names])
    # summarize performs the authoritative integer/coverage validation of sizes.
    sizes = {str(group): value for group, value in full_group_sizes.items()}
    if len(sizes) != len(full_group_sizes) or any(group not in sizes for group in groups):
        raise ValueError("full_group_sizes must uniquely cover all cohort groups")
    original_sizes = np.asarray([sizes[group] for group in groups])
    if group_size_values is not None:
        provided = np.asarray(group_size_values)
        if provided.shape != (n,) or not np.array_equal(provided, original_sizes):
            raise ValueError("group_size_values must equal the full original group sizes")
    if not isinstance(family_masks, Mapping) or set(family_masks) != set(FAMILIES):
        raise ValueError("family_masks must contain exactly copper,iron,hydride,A15,other")
    families = {name: _boolean_mask(family_masks[name], n, f"family_masks[{name}]") for name in FAMILIES}
    if not np.all(np.sum(np.stack(list(families.values())), axis=0) == 1):
        raise ValueError("family masks must be mutually exclusive and exhaustive")
    raw_a15 = families["A15"] if a15_mask is None else _boolean_mask(a15_mask, n, "a15_mask")
    subsets = {"all": np.ones(n, dtype=bool), "exclude_93": groups != "93", "only_93": groups == "93"}
    size_bins = {"1": original_sizes == 1,
                 "2-5": (original_sizes >= 2) & (original_sizes <= 5),
                 "6-20": (original_sizes >= 6) & (original_sizes <= 20),
                 ">20": original_sizes > 20}
    if not np.all(np.sum(np.stack(list(size_bins.values())), axis=0) == 1):
        raise ValueError("full original group sizes must be positive integers")
    pairs = list(combinations(names, 2))

    def summarize_mask(requested):
        selected = requested & common
        result = summarize(y, scores, groups, full_group_sizes, selected,
                           n_boot=n_boot, seed=seed, pairs=pairs)
        result.update(n_requested=int(requested.sum()),
                      n_requested_groups=int(len(np.unique(groups[requested]))),
                      n_missing=int((requested & ~common).sum()),
                      n_missing_groups=int(len(np.unique(groups[requested & ~common]))))
        return result

    result = {"method_order": names, "report_pairs": [list(pair) for pair in pairs],
              "subsets": {name: summarize_mask(mask) for name, mask in subsets.items()},
              "families": {family: {subset: summarize_mask(mask & subsets[subset]) for subset in SUBSETS}
                           for family, mask in families.items()},
              "group_size_bins": {label: {subset: summarize_mask(mask & subsets[subset]) for subset in SUBSETS}
                                  for label, mask in size_bins.items()},
              "coverage": {"requested": _counts(subsets["all"], y, groups),
                           "common_valid": _counts(common, y, groups),
                           "missing_common": _counts(~common, y, groups),
                           "method_finite": {name: _counts(np.isfinite(scores[name]), y, groups) for name in names},
                           "protocol_valid": _counts(protocol_valid, y, groups)},
              "statistical_protocol": {
                  "development_only": True, "n_boot": int(n_boot), "seed": int(seed),
                  "threshold": 0.5, "positive_rule": "p >= 0.5, even for valid explicit abstentions",
                  "weights": "group_weighted=1/full original group size; unweighted=1 per material",
                  "bootstrap": "component-level resampling; shared draws across methods, both weights and paired differences within each stratum",
                  "denominator": "all supplied methods must have finite, protocol-valid forced probabilities; missing outcomes are never imputed",
                  "family_partition": "caller-frozen, mutually exclusive and exhaustive; group counts across families can overlap",
                  "size_bin_definition": "full original component size; fixed before all masks",
                  "sparse_strata": "empty=all NA; single-class BA undefined; one-group CI necessarily degenerate"}}
    intersection_masks = {"raw_A15": raw_a15,
                          "component_93": subsets["only_93"],
                          "raw_A15_and_component_93": raw_a15 & subsets["only_93"],
                          "raw_A15_without_component_93": raw_a15 & subsets["exclude_93"],
                          "component_93_without_raw_A15": ~raw_a15 & subsets["only_93"],
                          "exclusive_A15_family": families["A15"]}
    result["a15_component93_overlap"] = {
        "criterion_source": "independent raw a15_mask" if a15_mask is not None else "exclusive family mask (raw criterion not separately provided)",
        "requested": {name: _counts(mask, y, groups) for name, mask in intersection_masks.items()},
        "common_valid": {name: _counts(mask & common, y, groups) for name, mask in intersection_masks.items()}}
    return result
