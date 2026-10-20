"""Standalone pure-RAM numeric rule executor v1; SOURCE, not integrated/scored.

No broker, scientific runner, file, model, network or held outcomes are used.
Only a trusted owner may issue immutable TRAIN pools/queries. Digest/HMAC bind
declarations; they do not prove actual TRAIN statistics, source bytes or baseline
score alignment. Hostile same-process reflection is not OS-isolated.

raw X127 AND(1..8) predicates -> TRAIN ws sufficient statistics -> ONE global
class-balance conversion -> fixed Laplace smoothing1/class -> sorted max3 equal
contributions. Scores describe curated-class associations, not calibrated
prevalence, independent evidence, chemical mechanism or laboratory measurement.
Full case agent, exporter, cohort collector and matched four arms remain future.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import re
import secrets
from typing import Any, Mapping, Sequence

PLAN_SHA256 = "b22da10b14442863de5313aeb7e425534caf120f20890a31def6f36dd73f5a18"
CARD_SCHEMA_SHA256 = "6495a86f847589e5b70bfe3a13b75002588696d9301f92214bc86f0b359cd76a"
EXECUTOR_ID = "balanced-local-rule-score/v1"
DSL_VERSION = "raw-X127-AND/v1"
AGGREGATION = "sorted-rule-ID-equal-contributions-fsum-max3/v1"
SMOOTHING_PER_CLASS = 1.0
MASS_ATOL = 1e-10
MASS_RTOL = 2e-12
MAX_TERMS_PER_RULE = 8
MAX_SELECTED_RULES = 3
BASELINE_OOF_SHA256 = "1d5b924a7c80330b485e99b85b7c53427269ae939513d0753d3a112484ce1390"
BASELINE_AUDIT_SHA256 = "c8b98b059dbb29f3fec77638284e3e46af2bf9a814f3edf28b29b7bb969f9917"
BASELINE_REFERENCE = {"policy": "fixed_original_HGB127", "score_artifact_sha256": BASELINE_OOF_SHA256,
                      "audit_sha256": BASELINE_AUDIT_SHA256}
RULE_FIELDS = ("rule_id", "version", "executable_code_sha256", "theoretical_basis", "applicability",
               "known_failures", "provenance", "empirical_scope", "fold_training_partition_id",
               "training_group_support", "counterexamples", "missing_input_behavior", "prediction_effect")
PARTITION_FIELDS = ("partition_id", "held_fold", "source_container_sha256", "feature_schema_sha256",
                    "TRAIN_rows", "held_component_ids", "TRAIN_class_ws_mass", "TRAIN_class_row_counts",
                    "statistics_audit_sha256", "rule_factory_recipe_sha256")
STAT_FIELDS = ("partition_id", "predicate_sha256", "support_source_indices", "matched_class_row_counts",
               "matched_class_ws_mass", "statistics_audit_sha256")
EFFECT_FIELDS = ("executor_id", "statistics_sha256", "smoothing_per_class", "aggregation", "score_semantics")
SCORE_SEMANTICS = "class_balanced_decision_score_not_natural_prevalence_probability"
FALLBACK_REASONS = ("no_selected_applicable_rules", "invalid_selection", "numeric_validation_failed",
                    "required_input_missing", "budget_exhausted", "tool_unavailable")


class RuleExecutionError(ValueError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuleExecutionError(message)


def canonical(v: Any) -> str:
    try:
        return json.dumps(v, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise RuleExecutionError("Noncanonical/nonfinite declaration") from exc


def digest(v: Any) -> str:
    return hashlib.sha256(canonical(v).encode("utf-8")).hexdigest()


def sha(v: Any, name: str) -> str:
    need(type(v) is str and re.fullmatch(r"[0-9a-f]{64}", v) is not None, "Invalid SHA: " + name)
    return v


def text(v: Any, name: str) -> str:
    need(type(v) is str and bool(v), "Missing text: " + name)
    return v


def exact(v: Any, fields: Sequence[str], name: str) -> None:
    need(isinstance(v, Mapping) and set(v) == set(fields), "Unexpected fields: " + name)


def f64(v: Any, name: str) -> float:
    # The exporter must supply Python binary64 floats, not integer thresholds,
    # Decimal, float32 casts, strings, booleans or nonfinite/missing values.
    need(type(v) is float and math.isfinite(v), "Expected finite binary64 float: " + name)
    return v


def close_mass(a: float, b: float) -> bool:
    # Metadata consistency ONLY. Predicate tests never use a tolerance.
    return math.isclose(a, b, rel_tol=MASS_RTOL, abs_tol=MASS_ATOL)


def mass_le(a: float, b: float) -> bool:
    return a <= b or close_mass(a, b)


def pair(v: Any, name: str, *, positive: bool = False) -> tuple[float, float]:
    need(type(v) is list and len(v) == 2, "Two classes required: " + name)
    values = tuple(f64(x, name) for x in v)
    need(all(x > 0.0 if positive else x >= 0.0 for x in values), "Invalid class masses: " + name)
    return values


def counts(v: Any, name: str, *, positive: bool = False) -> tuple[int, int]:
    need(type(v) is list and len(v) == 2 and all(type(x) is int for x in v), "Two raw class counts: " + name)
    need(all(x > 0 if positive else x >= 0 for x in v), "Invalid raw class counts: " + name)
    return tuple(v)


def normalize_predicate(dsl: Mapping) -> dict:
    """Factory-time semantic normalization BEFORE approval, never query scaling.

    Sort atoms by index/operator/threshold.hex, collapse duplicate atoms, and
    represent either zero threshold as +0.0 (identical <=/> comparisons). All
    nonzero threshold bits survive. No redundant-bound simplification is done.
    """
    exact(dsl, ("dsl_version", "all"), "AND DSL")
    terms = dsl["all"]
    need(dsl["dsl_version"] == DSL_VERSION and type(terms) is list
         and 1 <= len(terms) <= MAX_TERMS_PER_RULE, "Fixed1..8 AND terms required")
    atoms = {}
    for term in terms:
        exact(term, ("feature_index", "operator", "threshold"), "predicate atom")
        need(type(term["feature_index"]) is int and 0 <= term["feature_index"] < 127,
             "Predicate feature index outside original127")
        need(term["operator"] in ("<=", ">"), "Only <= or > operators permitted")
        threshold = f64(term["threshold"], "predicate threshold")
        if threshold == 0.0:
            threshold = 0.0
        atom = {"feature_index": term["feature_index"], "operator": term["operator"], "threshold": threshold}
        key = (atom["feature_index"], atom["operator"], threshold.hex())
        atoms[key] = atom
    return {"dsl_version": DSL_VERSION, "all": [atoms[key] for key in sorted(atoms)]}


def validate_predicate(dsl: Mapping) -> None:
    need(canonical(dsl) == canonical(normalize_predicate(dsl)),
         "Predicate must be factory-normalized before approval")


def evaluate_predicate(dsl: Mapping, X127: Sequence[float]) -> tuple[bool, list[dict]]:
    validate_predicate(dsl)
    need(type(X127) in (list, tuple) and len(X127) == 127, "Original X127 query required")
    xs = [f64(v, "query X127") for v in X127]
    witness = []
    for term in dsl["all"]:
        x, threshold = xs[term["feature_index"]], term["threshold"]
        matched = x <= threshold if term["operator"] == "<=" else x > threshold
        witness.append({"feature_index": term["feature_index"], "operator": term["operator"],
            "input_hex": x.hex(), "threshold_hex": threshold.hex(), "matched": bool(matched)})
    return all(atom["matched"] for atom in witness), witness


def validate_partition(p: Mapping) -> tuple[dict[int, dict], dict[str, int], float]:
    exact(p, PARTITION_FIELDS, "TRAIN partition")
    text(p["partition_id"], "partition ID")
    need(type(p["held_fold"]) is int and p["held_fold"] in (0, 1, 2), "Original3fold held ID required")
    for field in ("source_container_sha256", "feature_schema_sha256", "statistics_audit_sha256",
                  "rule_factory_recipe_sha256"):
        sha(p[field], field)
    rows = p["TRAIN_rows"]
    need(type(rows) is list and bool(rows), "Complete nonempty outer TRAIN identity map required")
    by_index, group_sizes, group_folds = {}, {}, {}
    for row in rows:
        exact(row, ("source_index", "component_id", "role", "fold"), "TRAIN identity")
        i, g, fold = row["source_index"], text(row["component_id"], "TRAIN group"), row["fold"]
        need(type(i) is int and i >= 0 and i not in by_index, "Invalid/duplicate TRAIN source index")
        need(row["role"] == "TRAIN" and type(fold) is int and fold in (0, 1, 2)
             and fold != p["held_fold"], "Foreign role or held row in TRAIN partition")
        need(g not in group_folds or group_folds[g] == fold, "TRAIN group crosses original folds")
        by_index[i], group_folds[g] = row, fold
        group_sizes[g] = group_sizes.get(g, 0) + 1
    held = p["held_component_ids"]
    need(type(held) is list and bool(held) and all(type(g) is str and bool(g) for g in held)
         and len(set(held)) == len(held) and set(held).isdisjoint(group_sizes), "TRAIN/held group overlap")
    # ws uses FULL outer-TRAIN group size, never the number of rule hits.
    W = math.fsum(1.0 / group_sizes[row["component_id"]] for row in rows)
    C0, C1 = pair(p["TRAIN_class_ws_mass"], "TRAIN ws class masses", positive=True)
    N0, N1 = counts(p["TRAIN_class_row_counts"], "TRAIN raw class counts", positive=True)
    need(N0 + N1 == len(rows) and close_mass(math.fsum((C0, C1)), W)
         and close_mass(W, float(len(group_sizes))), "TRAIN global rows/ws/group mass mismatch")
    return by_index, group_sizes, W


def local_score(partition: Mapping, statistics: Mapping) -> tuple[float, dict]:
    by_index, group_sizes, W = validate_partition(partition)
    exact(statistics, STAT_FIELDS, "TRAIN sufficient statistics")
    need(statistics["partition_id"] == partition["partition_id"]
         and statistics["statistics_audit_sha256"] == partition["statistics_audit_sha256"],
         "Sufficient statistics belong to another TRAIN partition/audit")
    sha(statistics["predicate_sha256"], "predicate hash")
    support = statistics["support_source_indices"]
    need(type(support) is list and bool(support) and all(type(i) is int for i in support)
         and len(set(support)) == len(support) and set(support).issubset(by_index), "Non-TRAIN support rows")
    U0, U1 = pair(statistics["matched_class_ws_mass"], "matched ws class masses")
    n0, n1 = counts(statistics["matched_class_row_counts"], "matched raw class counts")
    C0, C1 = partition["TRAIN_class_ws_mass"]
    N0, N1 = partition["TRAIN_class_row_counts"]
    expected_hit_ws = math.fsum(1.0 / group_sizes[by_index[i]["component_id"]] for i in support)
    need(n0 + n1 == len(support) and n0 <= N0 and n1 <= N1
         and (U0 == 0.0) == (n0 == 0) and (U1 == 0.0) == (n1 == 0)
         and mass_le(U0, C0) and mass_le(U1, C1)
         and close_mass(math.fsum((U0, U1)), expected_hit_ws),
         "Matched class counts/ws masses are inconsistent with full TRAIN weights")
    factors = [W / (2.0 * C0), W / (2.0 * C1)]
    balanced = [U0 * factors[0], U1 * factors[1]]  # Exactly ONE ws->wc conversion.
    need(all(math.isfinite(v) for v in factors + balanced), "Class-balance numeric overflow")
    numerator = balanced[1] + SMOOTHING_PER_CLASS
    denominator = (balanced[0] + balanced[1]) + 2.0 * SMOOTHING_PER_CLASS
    need(math.isfinite(numerator) and math.isfinite(denominator) and denominator > 0.0,
         "Nonfinite smoothed score")
    score = numerator / denominator
    need(math.isfinite(score) and 0.0 <= score <= 1.0, "Invalid decision score")
    groups = sorted({by_index[i]["component_id"] for i in support})
    witness = {"TRAIN_raw_rows": len(by_index), "TRAIN_original_groups": len(group_sizes),
        "TRAIN_total_ws_mass_W": W, "TRAIN_class_ws_mass_C": [C0, C1],
        "TRAIN_class_raw_row_counts": [N0, N1], "matched_raw_rows": len(support),
        "matched_original_groups": len(groups), "matched_class_raw_row_counts": [n0, n1],
        "matched_class_ws_mass_U": [U0, U1], "matched_ws_mass_from_full_group_sizes": expected_hit_ws,
        "class_balance_factors_W_over_2C": factors, "matched_class_wc_mass_A": balanced,
        "ws_to_wc_conversions": 1, "rule_hit_group_renormalization_used": False,
        "smoothing_per_class": SMOOTHING_PER_CLASS, "numerator": numerator,
        "denominator": denominator, "local_score": score, "local_score_hex": score.hex(),
        "metadata_mass_atol": MASS_ATOL, "metadata_mass_rtol": MASS_RTOL,
        "statistics_audit_sha256": statistics["statistics_audit_sha256"]}
    return score, witness


def validate_rule(partition: Mapping, card: Mapping, stats: Mapping, approved: Mapping,
                  executor_source_sha256: str) -> tuple[float, dict]:
    exact(card, RULE_FIELDS, "full13 rule card")
    exact(approved, ("card_sha256", "statistics_sha256", "envelope_sha256"), "owner approval")
    for k in approved:
        sha(approved[k], k)
    need(approved["card_sha256"] == digest(card) and approved["statistics_sha256"] == digest(stats)
         and approved["envelope_sha256"] == digest({"card": card, "statistics": stats}),
         "Rule differs from independently supplied approved card/statistics digest")
    text(card["rule_id"], "rule ID")
    text(card["version"], "rule version")
    need(card["executable_code_sha256"] == executor_source_sha256
         and card["fold_training_partition_id"] == partition["partition_id"]
         and card["empirical_scope"] == "outer_TRAIN", "Wrong executor/TRAIN scope")
    validate_predicate(card["applicability"])
    need(stats["predicate_sha256"] == digest(card["applicability"]), "Statistics not bound to this predicate")
    score, witness = local_score(partition, stats)
    exact(card["training_group_support"], ("n_groups", "n_rows"), "TRAIN support counts")
    need(type(card["training_group_support"]["n_groups"]) is int
         and type(card["training_group_support"]["n_rows"]) is int
         and card["training_group_support"]["n_groups"] == witness["matched_original_groups"]
         and card["training_group_support"]["n_rows"] == witness["matched_raw_rows"], "Support counts mismatch")
    by_index, _, _ = validate_partition(partition)
    support_groups = sorted({by_index[i]["component_id"] for i in stats["support_source_indices"]})
    provenance = card["provenance"]
    exact(provenance, ("origin", "partition_id", "source_indices", "component_ids"), "TRAIN provenance")
    need(provenance["origin"] == "TRAIN" and provenance["partition_id"] == partition["partition_id"]
         and provenance["source_indices"] == stats["support_source_indices"]
         and provenance["component_ids"] == support_groups, "Card support provenance mismatch")
    counterexamples = card["counterexamples"]
    need(type(counterexamples) is list and all(isinstance(c, Mapping) for c in counterexamples),
         "Explicit TRAIN counterexample list required")
    counter_ids = [c.get("source_index") for c in counterexamples]
    need(all(type(i) is int and i in by_index for i in counter_ids) and len(set(counter_ids)) == len(counter_ids)
         and all(c.get("component_id") == by_index[c["source_index"]]["component_id"] for c in counterexamples),
         "Non-TRAIN/duplicate counterexamples")
    need(type(card["theoretical_basis"]) in (str, dict) and bool(card["theoretical_basis"])
         and type(card["known_failures"]) is list
         and card["missing_input_behavior"] == "fallback_original_HGB127", "Missing theory/failure/input policy")
    effect = card["prediction_effect"]
    exact(effect, EFFECT_FIELDS, "fixed numeric prediction effect")
    need(effect["executor_id"] == EXECUTOR_ID and effect["statistics_sha256"] == digest(stats)
         and f64(effect["smoothing_per_class"], "fixed smoothing") == SMOOTHING_PER_CLASS
         and effect["aggregation"] == AGGREGATION and effect["score_semantics"] == SCORE_SEMANTICS,
         "Free/tuned numeric effect is forbidden")
    canonical(card)
    return score, witness


@dataclass(frozen=True)
class IssuedHandle:
    kind: str
    partition_id: str
    nonce: str
    payload_sha256: str
    authentication_tag: str


class TrustedNumericExecutor:
    """Owner factory/registry/key are not agent tools; reflection is not confined.

    Future exporter MUST verify actual source/labels/predicate hits/statistics/
    baseline bytes under a separate ROOT freeze. Current checks authenticate and
    numerically validate declared inputs, without pretending hashes prove truth.
    """
    def __init__(self, *, executor_source_sha256: str):
        self.source_sha = sha(executor_source_sha256, "ROOT-frozen executor SOURCE")
        self._key = secrets.token_bytes(32)
        self._issued: dict[str, tuple[str, str, str, str]] = {}

    def _tag(self, kind: str, pid: str, nonce: str, blob: str) -> str:
        return hmac.new(self._key, canonical([kind, pid, nonce, blob]).encode("utf-8"), hashlib.sha256).hexdigest()

    def _issue(self, kind: str, pid: str, payload: Mapping) -> IssuedHandle:
        blob, nonce = canonical(payload), secrets.token_hex(16)
        tag = self._tag(kind, pid, nonce, blob)
        self._issued[nonce] = (kind, pid, blob, tag)
        return IssuedHandle(kind, pid, nonce, hashlib.sha256(blob.encode("utf-8")).hexdigest(), tag)

    def _open(self, handle: IssuedHandle, kind: str, pid: str | None = None) -> dict:
        need(type(handle) is IssuedHandle and type(handle.kind) is str and handle.kind == kind,
             "Wrong/unissued handle kind")
        text(handle.partition_id, "handle partition ID")
        need(type(handle.nonce) is str and re.fullmatch(r"[0-9a-f]{32}", handle.nonce) is not None,
             "Malformed handle nonce")
        sha(handle.payload_sha256, "handle payload hash")
        sha(handle.authentication_tag, "handle authentication tag")
        entry = self._issued.get(handle.nonce)
        need(entry is not None, "Handle not issued by this executor")
        k, p, blob, tag = entry
        need(k == kind and p == handle.partition_id and (pid is None or p == pid)
             and handle.payload_sha256 == hashlib.sha256(blob.encode("utf-8")).hexdigest()
             and hmac.compare_digest(handle.authentication_tag, tag)
             and hmac.compare_digest(tag, self._tag(k, p, handle.nonce, blob)), "Handle authentication failed")
        return json.loads(blob)

    def issue_pool(self, partition_manifest: Mapping, rules: Sequence[Mapping], *,
                   approved_rule_digests: Mapping[str, Mapping]) -> IssuedHandle:
        validate_partition(partition_manifest)
        need(type(rules) in (list, tuple), "Rule envelope list required")
        rule_map = {}
        predicates = set()
        for envelope in rules:
            exact(envelope, ("card", "statistics"), "numeric rule envelope")
            rid = envelope["card"]["rule_id"]
            need(type(rid) is str and rid not in rule_map and rid in approved_rule_digests,
                 "Duplicate/unapproved rule ID")
            score, witness = validate_rule(partition_manifest, envelope["card"], envelope["statistics"],
                                           approved_rule_digests[rid], self.source_sha)
            predicate = digest(envelope["card"]["applicability"])
            need(predicate not in predicates, "Duplicate numeric predicate would double-count the same rule")
            predicates.add(predicate)
            rule_map[rid] = {"envelope": envelope, "approved": approved_rule_digests[rid],
                             "local_score": score, "numeric_witness": witness}
        need(set(rule_map) == set(approved_rule_digests), "Rule/approval map ID mismatch")
        payload = {"partition": partition_manifest, "rules": rule_map, "executor_source_sha256": self.source_sha,
            "implementation_plan_sha256": PLAN_SHA256, "rule_card_schema_sha256": CARD_SCHEMA_SHA256,
            "runtime_new_rules": 0, "cross_query_memory_enabled": False}
        return self._issue("numeric_pool", partition_manifest["partition_id"], payload)

    def _pool(self, pool: IssuedHandle) -> dict:
        p = self._open(pool, "numeric_pool")
        exact(p, ("partition", "rules", "executor_source_sha256", "implementation_plan_sha256",
                  "rule_card_schema_sha256", "runtime_new_rules", "cross_query_memory_enabled"), "sealed pool")
        need(p["executor_source_sha256"] == self.source_sha and p["implementation_plan_sha256"] == PLAN_SHA256
             and p["rule_card_schema_sha256"] == CARD_SCHEMA_SHA256 and type(p["runtime_new_rules"]) is int
             and p["runtime_new_rules"] == 0 and p["cross_query_memory_enabled"] is False, "Pool scope changed")
        validate_partition(p["partition"])
        for rid, row in p["rules"].items():
            need(rid == row["envelope"]["card"]["rule_id"], "Pool rule key changed")
            score, witness = validate_rule(p["partition"], row["envelope"]["card"], row["envelope"]["statistics"],
                                           row["approved"], self.source_sha)
            need(canonical(score) == canonical(row["local_score"])
                 and canonical(witness) == canonical(row["numeric_witness"]), "Pool numeric witness changed")
        return p

    def issue_query(self, pool: IssuedHandle, *, opaque_query_id: str, X127: Sequence[float],
                    held_component_id: str, baseline_score: float, baseline_reference: Mapping) -> IssuedHandle:
        p = self._pool(pool)
        text(opaque_query_id, "opaque query ID")
        need(held_component_id in p["partition"]["held_component_ids"], "Query is outside declared held groups")
        need(type(X127) in (list, tuple) and len(X127) == 127, "Original query X127 required")
        xs = [f64(v, "query X127") for v in X127]
        score = f64(baseline_score, "trusted original HGB127 score")
        need(0.0 <= score <= 1.0 and canonical(baseline_reference) == canonical(BASELINE_REFERENCE),
             "Fixed original HGB127 fallback reference required")
        return self._issue("numeric_query", pool.partition_id,
            {"pool_sha256": pool.payload_sha256, "opaque_query_id": opaque_query_id, "X127": xs,
             "held_component_id": held_component_id, "baseline_score": score, "baseline_reference": baseline_reference})

    def _bound(self, pool: IssuedHandle, query: IssuedHandle) -> tuple[dict, dict]:
        p, q = self._pool(pool), self._open(query, "numeric_query", pool.partition_id)
        exact(q, ("pool_sha256", "opaque_query_id", "X127", "held_component_id", "baseline_score",
                  "baseline_reference"), "sealed numeric query")
        need(q["pool_sha256"] == pool.payload_sha256
             and q["held_component_id"] in p["partition"]["held_component_ids"]
             and canonical(q["baseline_reference"]) == canonical(BASELINE_REFERENCE), "Query/pool/fallback mismatch")
        text(q["opaque_query_id"], "opaque query ID")
        need(type(q["X127"]) is list and len(q["X127"]) == 127, "Original127 query required")
        for v in q["X127"]:
            f64(v, "sealed query X127")
        need(0.0 <= f64(q["baseline_score"], "sealed baseline score") <= 1.0, "Bad sealed fallback score")
        return p, q

    def applicable_rules(self, pool: IssuedHandle, query: IssuedHandle) -> tuple[IssuedHandle, ...]:
        p, q = self._bound(pool, query)
        result = []
        for rid in sorted(p["rules"]):
            row = p["rules"][rid]
            matched, terms = evaluate_predicate(row["envelope"]["card"]["applicability"], q["X127"])
            if matched:
                result.append(self._issue("applicable_rule", pool.partition_id,
                    {"pool_sha256": pool.payload_sha256, "query_sha256": query.payload_sha256,
                     "rule_id": rid, "envelope_sha256": row["approved"]["envelope_sha256"], "predicate_witness": terms}))
        return tuple(result)

    def inspect_applicable(self, pool: IssuedHandle, query: IssuedHandle, rule: IssuedHandle) -> dict:
        p, q = self._bound(pool, query)
        selected = self._open(rule, "applicable_rule", pool.partition_id)
        exact(selected, ("pool_sha256", "query_sha256", "rule_id", "envelope_sha256", "predicate_witness"),
              "applicable rule handle payload")
        need(selected["pool_sha256"] == pool.payload_sha256 and selected["query_sha256"] == query.payload_sha256
             and selected["rule_id"] in p["rules"], "Rule capability is not bound to this query/pool")
        row = p["rules"][selected["rule_id"]]
        matched, terms = evaluate_predicate(row["envelope"]["card"]["applicability"], q["X127"])
        need(matched and selected["envelope_sha256"] == row["approved"]["envelope_sha256"]
             and canonical(terms) == canonical(selected["predicate_witness"]), "Rule is not applicable/authenticated")
        return {"rule_id": selected["rule_id"], "card": row["envelope"]["card"],
            "statistics_sha256": row["approved"]["statistics_sha256"], "predicate_witness": terms,
            "local_score": row["local_score"], "numeric_witness": row["numeric_witness"]}

    @staticmethod
    def _fallback(pool: IssuedHandle, q: dict, reason: str, error: str | None = None) -> dict:
        need(reason in FALLBACK_REASONS, "Undeclared/outcome-based fallback trigger")
        score = q["baseline_score"]
        return {"partition_id": pool.partition_id, "opaque_query_id": q["opaque_query_id"],
            "pool_sha256": pool.payload_sha256, "score": score, "score_hex": score.hex(),
            "prediction": int(score >= 0.5), "fallback_used": True, "fallback_reason": reason,
            "validation_error": error, "selected_rule_ids": [], "rule_contributions": [],
            "numeric_rule_prediction_performed": False, "baseline_reference": q["baseline_reference"],
            "score_semantics": SCORE_SEMANTICS, "evidence_scope": "adaptive_development_only",
            "fresh_blind_certification": False}

    def fallback(self, pool: IssuedHandle, query: IssuedHandle, *, reason: str) -> dict:
        _, q = self._bound(pool, query)
        return self._fallback(pool, q, reason)

    def execute_selected(self, pool: IssuedHandle, query: IssuedHandle,
                         selected_rules: Sequence[IssuedHandle]) -> dict:
        _, q = self._bound(pool, query)  # Authentic query required before any fallback.
        try:
            need(type(selected_rules) in (list, tuple) and len(selected_rules) <= MAX_SELECTED_RULES,
                 "At most3 issued rule handles, no free score/weights/action fields")
            if not selected_rules:
                return self._fallback(pool, q, "no_selected_applicable_rules")
            inspected = [self.inspect_applicable(pool, query, h) for h in selected_rules]
            ids = [v["rule_id"] for v in inspected]
            need(len(set(ids)) == len(ids), "Duplicate selected rule IDs")
        except RuleExecutionError as exc:
            return self._fallback(pool, q, "invalid_selection", str(exc))
        inspected.sort(key=lambda row: row["rule_id"])
        m = len(inspected)
        contributions = []
        for row in inspected:
            c = row["local_score"] / m
            contributions.append({**row, "equal_weight_denominator": m, "contribution": c,
                                  "contribution_hex": c.hex()})
        score = math.fsum(row["contribution"] for row in contributions)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            return self._fallback(pool, q, "numeric_validation_failed")
        return {"partition_id": pool.partition_id, "opaque_query_id": q["opaque_query_id"],
            "pool_sha256": pool.payload_sha256, "query_sha256": query.payload_sha256,
            "score": score, "score_hex": score.hex(), "prediction": int(score >= 0.5),
            "fallback_used": False, "fallback_reason": None,
            "selected_rule_ids": [row["rule_id"] for row in contributions],
            "rule_contributions": contributions, "numeric_rule_prediction_performed": True,
            "aggregation": AGGREGATION, "smoothing_per_class": SMOOTHING_PER_CLASS,
            "baseline_reference": q["baseline_reference"], "baseline_score_used_in_aggregation": False,
            "score_semantics": SCORE_SEMANTICS, "evidence_scope": "adaptive_development_only",
            "fresh_blind_certification": False, "new_physical_measurements": 0,
            "atomic_coordinates_available": False}
