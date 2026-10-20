"""Fixed-weight pooled metrics and paired cluster bootstrap for Stage 1.

Each row's weight is 1 / its ORIGINAL association-group size, including when
``mask`` selects a subset of rows. Balanced accuracy is the mean of the pooled,
weighted sensitivity and specificity; it is not a mean of fold-level metrics.
Bootstrap draws sample observed groups uniformly with replacement and multiply
their original row weights by the sampled multiplicity. Every arm and every
paired difference uses exactly the same draws.

Positive precision uses zero_division=0 in points and draws. Recall without any
positive true labels, balanced accuracy without both true classes, and accuracy on an
empty sample are undefined. Undefined points become None; nonfinite draws are
excluded from percentile intervals, with valid/undefined counts reported.
No files are read or written by this module.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


_METRICS = ("balanced_accuracy", "precision", "recall", "accuracy")


def _ratio(numerator, denominator, *, zero=np.nan):
    result = np.full(np.shape(numerator), zero, dtype=float)
    return np.divide(numerator, denominator, out=result, where=denominator > 0)


def _metrics(confusion):
    """Return arrays from weighted confusion counts (..., TN/FP/FN/TP)."""
    tn, fp, fn, tp = np.moveaxis(confusion, -1, 0)
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    return {
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": _ratio(tp, tp + fp, zero=0.0),
        "recall": recall,
        "accuracy": _ratio(tp + tn, tp + tn + fp + fn),
    }


def _interval(point, draws):
    valid = draws[np.isfinite(draws)]
    bounds = np.percentile(valid, [2.5, 97.5]).tolist() if valid.size else [None, None]
    return {
        "point": float(point) if np.isfinite(point) else None,
        "ci95": bounds,
        "n_valid_boot": int(valid.size),
        "n_undefined_boot": int(draws.size - valid.size),
    }


def summarize(
    y,
    scores_by_arm: Mapping[str, object],
    groups,
    full_group_sizes: Mapping[str, int],
    mask=None,
    n_boot: int = 5000,
    seed: int = 20261016,
):
    """Summarize score arrays and all paired arm differences.

    Parameters
    ----------
    y : one-dimensional array of 0/1 true labels
    scores_by_arm : ordered mapping from arm name to one score per input row
        Scores must be finite. Prediction is ``score >= 0.5``. 0/1 prediction
        arrays are also accepted. All arms must cover exactly the same rows.
    groups : one-dimensional array of association-group IDs
        IDs and ``full_group_sizes`` keys are normalized with ``str``.
    full_group_sizes : mapping from group ID to full original row count
        Do not recompute sizes after applying a subset mask. Counts must be at
        least as large as the number of input rows belonging to each group.
    mask : optional Boolean array aligned with the input arrays
    n_boot : at least 2000; default 5000 independent cluster draws
    seed : integer NumPy random seed

    Returns
    -------
    JSON-serializable dict containing ``n``, ``n_groups``, ``metrics[arm]``, and
    ``paired_differences['A - B']`` for each pair in mapping order. A difference
    is always A minus B; values are proportions, not percentage points.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    if y.ndim != 1 or groups.ndim != 1 or y.size != groups.size:
        raise ValueError("y and groups must be aligned one-dimensional arrays")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must contain only binary labels 0 and 1")
    if not isinstance(n_boot, (int, np.integer)) or n_boot < 2000:
        raise ValueError("n_boot must be an integer >= 2000")
    if not scores_by_arm:
        raise ValueError("at least one arm is required")

    group_strings = np.asarray([str(group) for group in groups])
    all_groups, all_counts = np.unique(group_strings, return_counts=True)
    normalized_sizes = {}
    for group, size in full_group_sizes.items():
        key = str(group)
        if key in normalized_sizes:
            raise ValueError("full_group_sizes has duplicate normalized group IDs")
        if isinstance(size, (bool, np.bool_)) or not np.isfinite(size):
            raise ValueError("full group sizes must be finite positive integers")
        if int(size) != size or size <= 0:
            raise ValueError("full group sizes must be finite positive integers")
        normalized_sizes[key] = int(size)
    for group, count in zip(all_groups, all_counts):
        if group not in normalized_sizes or normalized_sizes[group] < count:
            raise ValueError(f"missing/too-small original size for group {group!r}")

    if mask is None:
        selected = np.ones(y.size, dtype=bool)
    else:
        selected = np.asarray(mask)
        if selected.dtype != np.bool_ or selected.shape != y.shape:
            raise ValueError("mask must be a Boolean array aligned with y")
    if not selected.any():
        raise ValueError("the selected subset must contain at least one row")

    names = list(scores_by_arm)
    if any(not isinstance(name, str) for name in names):
        raise ValueError("arm names must be strings")
    predictions = []
    for name in names:
        scores = np.asarray(scores_by_arm[name], dtype=float)
        if scores.shape != y.shape:
            raise ValueError(f"scores for {name!r} must align with y")
        if not np.isfinite(scores[selected]).all():
            raise ValueError(f"selected scores for {name!r} must be finite")
        predictions.append(scores[selected] >= 0.5)

    selected_y = y[selected].astype(bool)
    selected_groups = group_strings[selected]
    observed_groups, group_index = np.unique(selected_groups, return_inverse=True)
    n_groups = observed_groups.size
    weights = np.asarray([1.0 / normalized_sizes[group] for group in selected_groups])

    # Every row contributes to exactly one weighted confusion cell per arm.
    group_confusion = np.zeros((n_groups, len(names), 4), dtype=float)
    for arm_index, predicted in enumerate(predictions):
        cell = 2 * selected_y.astype(int) + predicted.astype(int)
        np.add.at(group_confusion[:, arm_index, :], (group_index, cell), weights)
    point = _metrics(group_confusion.sum(axis=0))

    # Batching bounds memory without changing resampling units or arm pairing.
    rng = np.random.default_rng(seed)
    boot_confusion = np.empty((n_boot, len(names), 4), dtype=float)
    group_probabilities = np.full(n_groups, 1.0 / n_groups)
    for start in range(0, n_boot, 256):
        stop = min(start + 256, n_boot)
        multiplicities = rng.multinomial(n_groups, group_probabilities, size=stop - start)
        boot_confusion[start:stop] = (
            multiplicities @ group_confusion.reshape(n_groups, -1)
        ).reshape(stop - start, len(names), 4)
    bootstrap = _metrics(boot_confusion)

    arm_metrics = {
        name: {
            metric: _interval(point[metric][i], bootstrap[metric][:, i])
            for metric in _METRICS
        }
        for i, name in enumerate(names)
    }
    paired = {}
    for i, name_a in enumerate(names):
        for j in range(i + 1, len(names)):
            name_b = names[j]
            paired[f"{name_a} - {name_b}"] = {
                "arm_a": name_a,
                "arm_b": name_b,
                **{
                    metric: _interval(
                        point[metric][i] - point[metric][j],
                        bootstrap[metric][:, i] - bootstrap[metric][:, j],
                    )
                    for metric in _METRICS
                },
            }

    return {
        "n": int(selected.sum()),
        "n_groups": int(n_groups),
        "weight_sum": float(weights.sum()),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "confidence_level": 0.95,
        "weighting": "row weight = 1 / full original association-group size",
        "aggregation": "pooled OOF weighted confusion counts; no fold macro-average",
        "bootstrap_unit": "observed association groups, uniform with replacement",
        "interval": "percentile, paired arm draws",
        "threshold": 0.5,
        "undefined_precision": "zero_division=0 for points and bootstrap draws",
        "other_undefined_metrics": "None point; exclude nonfinite bootstrap draws",
        "metrics": arm_metrics,
        "paired_differences": paired,
    }
