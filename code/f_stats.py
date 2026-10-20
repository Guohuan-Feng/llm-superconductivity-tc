"""Finite F-series statistics; no reads, writes, fits, or LLM calls.

Both weighting schemes use the SAME 5000 cluster draws. The group-weighted
scheme retains 1/full original group size after subsetting. Point estimates
pool confusion counts; no fold average is used. A one-group bootstrap is
necessarily degenerate and cannot measure between-group generalization.
"""
from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
from common import np

METRICS = ("balanced_accuracy", "precision", "recall", "accuracy")
WEIGHTINGS = ("group_weighted", "unweighted")


def _ratio(a, b, zero=np.nan):
    return np.divide(a, b, out=np.full(np.shape(a), zero, dtype=float), where=b > 0)


def _metric(confusion):
    tn, fp, fn, tp = np.moveaxis(confusion, -1, 0)
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    return {"balanced_accuracy": (recall + specificity) / 2,
            "precision": _ratio(tp, tp + fp, zero=0.0), "recall": recall,
            "accuracy": _ratio(tp + tn, tp + tn + fp + fn)}


def _interval(point, draws):
    finite = np.asarray(draws)[np.isfinite(draws)]
    return {"point": float(point) if np.isfinite(point) else None,
            "ci95": np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) else [None, None],
            "n_valid_boot": int(len(finite)),
            "n_undefined_boot": int(np.size(draws) - len(finite))}


def _validate_groups(groups, full_group_sizes, n):
    groups = np.asarray(groups).astype(str)
    if groups.ndim != 1 or len(groups) != n:
        raise ValueError("groups must be one-dimensional and aligned")
    sizes = {}
    for group, value in full_group_sizes.items():
        key = str(group)
        if key in sizes or isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or int(value) != value or value <= 0:
            raise ValueError("full group sizes must be unique finite positive integers")
        sizes[key] = int(value)
    for group, count in zip(*np.unique(groups, return_counts=True)):
        if group not in sizes or sizes[group] < count:
            raise ValueError(f"missing or too-small full original group size: {group}")
    return groups, sizes


def _mask(mask, n):
    if mask is None:
        return np.ones(n, dtype=bool)
    mask = np.asarray(mask)
    if mask.shape != (n,) or mask.dtype != np.bool_:
        raise ValueError("mask must be an aligned Boolean array")
    return mask


def _settings(n_boot, seed):
    if not isinstance(n_boot, (int, np.integer)) or n_boot < 5000:
        raise ValueError("F-series requires at least 5000 bootstrap draws")
    return {"n_boot": int(n_boot), "seed": int(seed), "confidence_level": 0.95,
            "bootstrap_unit": "observed component_id groups, uniformly with replacement",
            "interval": "percentile 95%; all methods, weightings and paired differences share group draws",
            "group_weighting": "1 / full original group size before any mask",
            "unweighted_weighting": "1 per material; still group-bootstrap uncertainty",
            "undefined_metrics": "precision=0 if no predicted positive; BA undefined without both true classes; recall undefined without true positives; empty sample all undefined; nonfinite draws excluded and counted"}


def summarize(y, scores_by_arm: Mapping[str, object], groups, full_group_sizes,
              mask=None, *, n_boot=5000, seed=20261016, pairs=None, threshold=0.5):
    """Report BA/P/R/accuracy under both schemes and paired differences.

    y/groups/scores are aligned full-cohort arrays; mask can select any subset.
    Score >= threshold is positive. pairs=None reports all unordered pairs in
    input order; [("A", "B")] reports A minus B. Empty masks are supported.
    Returns metrics and paired_differences within each weighting key. Values
    are proportions (multiply differences by 100 for percentage points).
    """
    settings = _settings(n_boot, seed)
    y = np.asarray(y)
    if y.ndim != 1 or not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must be a one-dimensional binary array")
    groups, sizes = _validate_groups(groups, full_group_sizes, len(y))
    selected = _mask(mask, len(y))
    names = list(scores_by_arm)
    if not names or any(not isinstance(k, str) for k in names):
        raise ValueError("at least one named method is required")
    scores = []
    for name in names:
        values = np.asarray(scores_by_arm[name], dtype=float)
        if values.shape != y.shape or not np.isfinite(values[selected]).all():
            raise ValueError(f"invalid selected scores: {name}")
        scores.append(values[selected])
    pairs = list(combinations(names, 2)) if pairs is None else list(pairs)
    if any(a not in names or b not in names or a == b for a, b in pairs):
        raise ValueError("paired comparisons must name two different existing methods")
    observed, inverse = np.unique(groups[selected], return_inverse=True)
    n_groups, n_arms = len(observed), len(names)
    n = int(selected.sum())
    result = {"n": n, "n_groups": n_groups, "n_positive": int(y[selected].sum()),
              "n_negative": int(n - y[selected].sum()), **settings,
              "threshold": float(threshold),
              "single_group_warning": "Only one group: all bootstrap CIs are degenerate; no between-group uncertainty is estimable." if n_groups == 1 else None}
    if n == 0:
        empty = _interval(np.nan, np.full(n_boot, np.nan))
        for weighting in WEIGHTINGS:
            result[weighting] = {"weight_sum": 0.0,
                "metrics": {name: {metric: dict(empty) for metric in METRICS} for name in names},
                "paired_differences": {f"{a} - {b}": {"arm_a": a, "arm_b": b,
                    **{metric: dict(empty) for metric in METRICS}} for a, b in pairs}}
        return result
    weights = np.stack(([1 / sizes[g] for g in groups[selected]], np.ones(n)), axis=1)
    confusion = np.zeros((n_groups, n_arms, 2, 4), dtype=float)
    for index, score in enumerate(scores):
        cell = 2 * y[selected].astype(int) + (score >= threshold).astype(int)
        for scheme in range(2):
            np.add.at(confusion[:, index, scheme, :], (inverse, cell), weights[:, scheme])
    point = _metric(confusion.sum(axis=0))
    bootstrap_confusion = np.empty((n_boot, n_arms, 2, 4), dtype=float)
    rng = np.random.default_rng(seed)
    for start in range(0, n_boot, 256):
        stop = min(start + 256, n_boot)
        draws = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=stop - start)
        bootstrap_confusion[start:stop] = (draws @ confusion.reshape(n_groups, -1)).reshape(-1, n_arms, 2, 4)
    bootstrap = _metric(bootstrap_confusion)
    for scheme, weighting in enumerate(WEIGHTINGS):
        paired = {}
        for a, b in pairs:
            i, j = names.index(a), names.index(b)
            paired[f"{a} - {b}"] = {"arm_a": a, "arm_b": b,
                **{metric: _interval(point[metric][i, scheme] - point[metric][j, scheme],
                    bootstrap[metric][:, i, scheme] - bootstrap[metric][:, j, scheme]) for metric in METRICS}}
        result[weighting] = {"weight_sum": float(weights[:, scheme].sum()),
            "metrics": {name: {metric: _interval(point[metric][i, scheme], bootstrap[metric][:, i, scheme])
                for metric in METRICS} for i, name in enumerate(names)},
            "paired_differences": paired}
    return result


def fraction_summary(values, groups, full_group_sizes, mask=None, *, n_boot=5000, seed=20261016):
    """Binary fraction under both weightings with 5000 shared cluster draws.

    Aligned NaN values are unavailable and excluded, with counts reported.
    Use masks for recognized-only fractions; keep a separate missing-Tc count
    so model null outputs are never silently presented as correct recalls.
    """
    settings = _settings(n_boot, seed)
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("fraction values must be one-dimensional")
    groups, sizes = _validate_groups(groups, full_group_sizes, len(values))
    requested = _mask(mask, len(values))
    finite = np.isfinite(values)
    selected = requested & finite
    if np.any(~np.isin(values[selected], [0.0, 1.0])):
        raise ValueError("fraction values must be binary or NaN")
    observed, inverse = np.unique(groups[selected], return_inverse=True)
    n, n_groups = int(selected.sum()), len(observed)
    result = {"n": n, "n_groups": n_groups, "n_requested": int(requested.sum()),
              "n_missing": int((requested & ~finite).sum()), **settings,
              "single_group_warning": "Only one group; bootstrap cannot estimate between-group uncertainty." if n_groups == 1 else None}
    if not n:
        return {**result, **{scheme: _interval(np.nan, np.full(n_boot, np.nan)) for scheme in WEIGHTINGS}}
    weights = np.stack(([1 / sizes[g] for g in groups[selected]], np.ones(n)), axis=1)
    denominator = np.stack([np.bincount(inverse, weights=weights[:, k], minlength=n_groups) for k in range(2)], axis=1)
    numerator = np.stack([np.bincount(inverse, weights=weights[:, k] * values[selected], minlength=n_groups) for k in range(2)], axis=1)
    point = numerator.sum(axis=0) / denominator.sum(axis=0)
    boot = np.empty((n_boot, 2))
    rng = np.random.default_rng(seed)
    for start in range(0, n_boot, 256):
        stop = min(start + 256, n_boot)
        draws = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=stop - start)
        boot[start:stop] = (draws @ numerator) / (draws @ denominator)
    return {**result, **{scheme: _interval(point[k], boot[:, k]) for k, scheme in enumerate(WEIGHTINGS)}}
