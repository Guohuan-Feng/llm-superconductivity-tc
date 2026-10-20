"""Stage 1 E1: fixed outer-fold rule-bank diagnostics; no bank mining or LLM.

This module only reads frozen rule-bank JSON. Output persistence belongs to the
stage runner. Rule scores are class-balanced decision scores, not probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier


DEFAULT_BANK_ROOT = Path(
    "C:/Users/28908/Codex/fung-research/workflow-review-2026-10-10/"
    "actual-train-rulebank-run-001"
)


@dataclass(frozen=True)
class Rule:
    rule_id: str
    predicates: tuple[tuple[int, str, float], ...]
    score: float
    support_groups: int
    coverage_rows: int
    support_indices: tuple[int, ...]


def load_rules(path):
    """Read the literal fixed cards and recompute their documented score."""
    bank = json.loads(Path(path).read_text(encoding="utf-8"))["finalized_bank"]
    partition = bank["partition"]
    train_indices = {int(r["source_index"]) for r in partition["TRAIN_rows"]}
    c0, c1 = map(float, partition["TRAIN_class_ws_mass"])
    total = math.fsum((c0, c1))
    if min(c0, c1) <= 0:
        raise ValueError("Both TRAIN classes need positive group mass")
    rules = []
    for envelope in bank["rule_envelopes"]:
        card, statistics = envelope["card"], envelope["statistics"]
        effect = card["prediction_effect"]
        if effect["executor_id"] != "balanced-local-rule-score/v1":
            raise ValueError("Unexpected frozen rule executor")
        if float(effect["smoothing_per_class"]) != 1.0:
            raise ValueError("Unexpected smoothing; this stage freezes lambda=1")
        u0, u1 = map(float, statistics["matched_class_ws_mass"])
        v0, v1 = u0 * total / (2 * c0), u1 * total / (2 * c1)
        score = (v1 + 1.0) / math.fsum((v0, v1, 2.0))
        application = card["applicability"]
        if application["dsl_version"] != "raw-X127-AND/v1":
            raise ValueError("Unexpected fixed predicate DSL")
        predicates = tuple(
            (int(p["feature_index"]), p["operator"], float(p["threshold"]))
            for p in application["all"]
        )
        if not predicates or any(op not in ("<=", ">") for _, op, _ in predicates):
            raise ValueError("Unsupported fixed predicate")
        support_indices = tuple(card["provenance"]["source_indices"])
        if not set(support_indices).issubset(train_indices):
            raise ValueError("Frozen rule support contains a non-TRAIN source index")
        support = card["training_group_support"]
        if int(support["n_rows"]) != len(support_indices):
            raise ValueError("Frozen coverage count disagrees with card provenance")
        rules.append(Rule(
            rule_id=card["rule_id"], predicates=predicates, score=score,
            support_groups=int(support["n_groups"]),
            coverage_rows=int(support["n_rows"]), support_indices=support_indices,
        ))
    rules.sort(key=lambda r: r.rule_id)
    if len({r.rule_id for r in rules}) != len(rules):
        raise ValueError("Duplicate fixed rule IDs")
    info = {
        "path": str(Path(path)), "n_rules": len(rules),
        "train_indices": [int(r["source_index"]) for r in partition["TRAIN_rows"]],
        "train_class_ws_mass": [c0, c1],
        "score_range": [min(r.score for r in rules), max(r.score for r in rules)],
        "n_support_groups_ge_20": sum(r.support_groups >= 20 for r in rules),
        "score_formula": "(U1*W/(2*C1)+1)/(U0*W/(2*C0)+U1*W/(2*C1)+2)",
    }
    return rules, info


def applicability_matrix(X, rules):
    """Evaluate literal raw-X127 AND thresholds, retaining equality direction."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 127 or not np.isfinite(X).all():
        raise ValueError("E1 requires finite original X127 in original feature order")
    matrix = np.ones((len(X), len(rules)), dtype=bool)
    for j, rule in enumerate(rules):
        for feature, operator, threshold in rule.predicates:
            if not 0 <= feature < X.shape[1]:
                raise ValueError("Rule feature index outside X127")
            column = X[:, feature]
            matrix[:, j] &= column <= threshold if operator == "<=" else column > threshold
    return matrix


def select_rule_scores(matrix, rules, *, mode, k=3, min_support_groups=20):
    """Select using frozen TRAIN statistics only; mean in sorted rule-ID order."""
    if mode == "R-cov":
        order = sorted(range(len(rules)), key=lambda j: (-rules[j].coverage_rows, rules[j].rule_id))
    elif mode == "R-info":
        order = sorted(
            (j for j, r in enumerate(rules) if r.support_groups >= min_support_groups),
            key=lambda j: (-abs(rules[j].score - 0.5), rules[j].rule_id),
        )
    else:
        raise ValueError(mode)
    scores = np.full(matrix.shape[0], 0.5, dtype=float)
    statuses = np.full(matrix.shape[0], "no_applicable_rules", dtype=object)
    selected = []
    for i in range(matrix.shape[0]):
        chosen = [j for j in order if matrix[i, j]][:k]
        chosen.sort(key=lambda j: rules[j].rule_id)
        selected.append([rules[j].rule_id for j in chosen])
        if chosen:
            scores[i] = math.fsum(rules[j].score for j in chosen) / len(chosen)
            statuses[i] = "ok"
    return scores, statuses, selected


def group_weighted_ba(y, prediction, ws):
    y, prediction, ws = map(np.asarray, (y, prediction, ws))
    masses = [float(ws[y == c].sum()) for c in (0, 1)]
    if min(masses) <= 0:
        raise ValueError("A BA validation pool is missing a class")
    return float(np.mean([
        ws[(y == c) & (prediction == c)].sum() / masses[c] for c in (0, 1)
    ]))


def group_class_weights(y, ws):
    """Class balance inverse-original-group-size masses, then mean-normalize."""
    y, ws = np.asarray(y), np.asarray(ws, dtype=float)
    mass = np.array([ws[y == c].sum() for c in (0, 1)])
    if np.any(mass <= 0):
        raise ValueError("Training split is missing a class")
    weight = ws / mass[y]
    return weight / weight.mean()


def run_e1(data, *, bank_root=DEFAULT_BANK_ROOT,
           c_grid=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0), seed=20261016,
           progress=None, log=None):
    """Produce original three-fold OOF scores from the frozen rule banks.

    data keys: X127, y, fold, group, group_size, source_index. The callback receives
    progress(arm, global_indices, scores, statuses) once per completed fold/arm;
    the runner writes each chunk in batches of ten for interruption tolerance.
    C is selected by pooled inner OOF group-weighted BA; exact ties use lower C.
    The fixed outer TRAIN rule representation remains fixed during inner CV.
    """
    log = log or (lambda message: None)
    X, y = np.asarray(data["X127"]), np.asarray(data["y"], dtype=int)
    folds = np.asarray(data["fold"], dtype=int)
    groups = np.asarray(data["group"]).astype(str)
    sizes = np.asarray(data["group_size"], dtype=float)
    ids = np.asarray(data["source_index"], dtype=int)
    if len(set(map(len, (X, y, folds, groups, sizes, ids)))) != 1:
        raise ValueError("E1 arrays do not share original row order")
    if np.any(sizes <= 0):
        raise ValueError("Original group sizes must be positive")
    ws = 1.0 / sizes
    arms = ("R-cov", "R-info", "R-all", "Tree3", "Tree4")
    scores = {arm: np.full(len(y), np.nan) for arm in arms}
    statuses = {arm: np.full(len(y), "pending", dtype=object) for arm in arms}
    diagnostics = {
        "folds": {}, "max_abs_score_minus_half": [None] * len(y),
        "selected_rules": {arm: [None] * len(y) for arm in ("R-cov", "R-info")},
        "no_rule_behavior": "score=0.5, y_pred=1, no HGB fallback",
        "rule_aggregation": "equal-weight mean of at most three selected scores; sorted ID math.fsum",
        "lr_weighting": "inverse-original-group-size, class-balanced by weighted mass; mean weight=1",
        "tree_weighting": "explicit inverse-original-group-size and weighted-class-mass balance; equal total effective weight per true class",
        "inner_cv_limitation": "Frozen rule predicates were mined using all outer TRAIN labels; C CV tunes a fixed supervised representation, not a nested rule-mining pipeline. Outer HELD is excluded throughout.",
    }

    def record(arm, held, values, state):
        scores[arm][held] = values
        statuses[arm][held] = state
        if progress is not None:
            progress(arm, held, np.asarray(values), np.asarray(state))

    for outer in (0, 1, 2):
        train, held = np.flatnonzero(folds != outer), np.flatnonzero(folds == outer)
        if set(groups[train]) & set(groups[held]):
            raise ValueError("Outer folds split an association group")
        rules, metadata = load_rules(Path(bank_root) / f"fold{outer}" / "ROOT_REVIEWABLE_BANK.json")
        if set(metadata["train_indices"]) != set(ids[train].tolist()):
            raise ValueError("Fixed bank TRAIN source IDs disagree with data fold assignment")
        metadata.pop("train_indices")
        matrix = applicability_matrix(X, rules)
        test_matrix = matrix[held]
        log(f"E1 outer fold {outer}: {len(train)} TRAIN, {len(held)} HELD, {len(rules)} fixed rules")
        for arm in ("R-cov", "R-info"):
            values, state, chosen = select_rule_scores(test_matrix, rules, mode=arm)
            for i, selected in zip(held, chosen):
                diagnostics["selected_rules"][arm][int(i)] = selected
            record(arm, held, values, state)
        deviations = np.abs(np.array([r.score for r in rules]) - 0.5)
        max_deviation = np.where(test_matrix, deviations[None, :], 0.0).max(axis=1)
        for i, value in zip(held, max_deviation):
            diagnostics["max_abs_score_minus_half"][int(i)] = float(value)
        metadata["held_no_applicable_rules"] = int((~test_matrix.any(axis=1)).sum())
        metadata["held_n_ge_0.15"] = int((max_deviation >= 0.15).sum())

        # L2 logistic regression uses the whole fixed bank applicability matrix.
        design = matrix.astype(np.float64)
        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed + outer)
        splits = list(cv.split(design[train], y[train], groups[train]))
        candidate = []
        for c_value in sorted(set(map(float, c_grid))):
            inner_prediction = np.full(len(train), -1, dtype=int)
            for inner_train, inner_valid in splits:
                fit_rows, valid_rows = train[inner_train], train[inner_valid]
                if set(groups[fit_rows]) & set(groups[valid_rows]):
                    raise ValueError("Inner CV split an association group")
                model = LogisticRegression(C=c_value, solver="liblinear", max_iter=3000,
                                           tol=1e-7, random_state=seed)
                model.fit(design[fit_rows], y[fit_rows],
                          sample_weight=group_class_weights(y[fit_rows], ws[fit_rows]))
                if model.n_iter_[0] >= model.max_iter:
                    raise RuntimeError("R-all inner logistic regression failed to converge")
                inner_prediction[inner_valid] = (model.predict_proba(design[valid_rows])[:, 1] >= 0.5).astype(int)
            if np.any(inner_prediction < 0):
                raise ValueError("Incomplete pooled inner OOF predictions")
            cv_ba = group_weighted_ba(y[train], inner_prediction, ws[train])
            candidate.append({"C": c_value, "pooled_inner_group_weighted_ba": cv_ba})
        best = max(candidate, key=lambda row: (row["pooled_inner_group_weighted_ba"], -row["C"]))
        model = LogisticRegression(C=best["C"], solver="liblinear", max_iter=3000,
                                   tol=1e-7, random_state=seed)
        model.fit(design[train], y[train], sample_weight=group_class_weights(y[train], ws[train]))
        if model.n_iter_[0] >= model.max_iter:
            raise RuntimeError("R-all final logistic regression failed to converge")
        record("R-all", held, model.predict_proba(design[held])[:, 1], np.full(len(held), "ok", object))
        metadata["R_all_C_grid_inner_scores"] = candidate
        metadata["R_all_selected_C"] = best["C"]
        metadata["inner_cv_group_counts"] = [
            {"train": len(set(groups[train[a]])), "validation": len(set(groups[train[b]]))}
            for a, b in splits
        ]
        for depth in (3, 4):
            model = DecisionTreeClassifier(max_depth=depth, class_weight=None, random_state=seed)
            tree_wc = ws[train] * ws[train].sum() / (2*np.array([ws[train][y[train]==c].sum() for c in (0,1)])[y[train]])
            model.fit(X[train], y[train], sample_weight=tree_wc)
            record(f"Tree{depth}", held, model.predict_proba(X[held])[:, 1], np.full(len(held), "ok", object))
        diagnostics["folds"][str(outer)] = metadata
        log(f"E1 outer fold {outer} complete; R-all selected C={best['C']}")
    if any(not np.isfinite(value).all() for value in scores.values()):
        raise ValueError("Incomplete E1 OOF scores")
    return {"scores": scores, "statuses": statuses, "diagnostics": diagnostics}
