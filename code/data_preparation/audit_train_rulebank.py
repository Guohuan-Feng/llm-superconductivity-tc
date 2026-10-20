"""Independent pure TRAIN-rule-work-product audit; SOURCE, not executed.

No producer/executor import, I/O, native estimator, geometry/property or query.
ROOT supplies independently source-authenticated exact7 primitive TRAIN rows,
full original cohort group counts, the fixed8-field scope and the frozen11-field
binding. Curated assigned labels are checked as binary declarations; no measured
Tc/unknown-origin authentication is claimed by this pure interface.

Every work-product field is reconstructed independently. The shared recipe/DSL
is the scientific contract; no producer numerical/statistical helper is called.
Only a future ROOT_actual execution after all comparisons may return the8-field
finalize certificate. Invented mode returns certificate=None. The returned
certificate itself is declaration consistency, not an OS/cryptographic ROOT seal.
"""
import hashlib
import json
import math

FACTORY_SHA256 = "54c3e8587c9e273fa95ad39b8ce5409624bdadb7c925cade5437dfe5959fcbe6"
EXECUTOR_SHA256 = "5ffe51ef90244ce87ef1b0a03f78466a9ff4573226b8d95d68d563d25fef8c01"
DSL = "raw-X127-AND/v1"
ROW_FIELDS = ("source_index", "original_row_index", "component_id", "fold", "role", "X127", "label")
SCOPE_FIELDS = ("partition_id", "held_fold", "source_container_sha256", "feature_schema_sha256", "TRAIN_rows",
                "held_component_ids", "TRAIN_original_row_indices", "rule_factory_recipe_sha256")
BINDING_FIELDS = ("execution_kind", "ROOT_authorization_sha256", "auditor_source_sha256", "factory_source_sha256",
                  "executor_source_sha256", "source_container_sha256", "feature_schema_sha256", "scope_sha256",
                  "primitive_rows_sha256", "groupcounts_sha256", "work_product_sha256")
EXPECTED_RECIPE = {'factory_version': 'TRAIN-X127-weighted-quantile-overlap/v1', 'task_id': 'curated_Tc_gt_10K', 'features': 'original_X127_only', 'feature_count': 127, 'empirical_scope': 'outer_TRAIN_assigned_curated_binary_labels_only', 'seed': 20261010, 'uses_randomness': False, 'quantile_weight': '1/full_outer_TRAIN_original_group_row_count', 'quantile_definition': 'first sorted k with math.fsum(ws[:k+1]) >= q*math.fsum(all_ws)', 'quantile_sort': ['raw_binary64_value', 'source_index', 'original_row_index'], 'median_halfspaces': {'feature_indices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126], 'q': 0.5, 'both_sides_must_pass': True}, 'geometry_extra_halfspaces': {'feature_indices': [118, 119, 120, 121, 122, 123, 124, 125, 126], 'q': [0.25, 0.75]}, 'geometry_AND': 'all unordered distinct geometry-column pairs, four median-direction combinations', 'maximum_generated_candidates': 434, 'minimum_raw_rows': 40, 'minimum_original_groups': 10, 'support_gates_use_labels': False, 'predicate_normalization': 'sorted(index,operator,threshold.hex); duplicate atoms collapse; signed-zero threshold +0; no epsilon or bound simplification', 'predicate_dedup': 'one normalized predicate digest per pool, not TRAIN-mask equivalence', 'numeric_score': '(U1*W/(2*C1)+1)/(U0*W/(2*C0)+U1*W/(2*C1)+2)', 'smoothing_per_class': 1.0, 'maximum_selected_distinct_applicable_rules': 3, 'aggregation': 'sorted-rule-ID-equal-contributions-fsum-max3/v1', 'score_semantics': 'class_balanced_decision_score_not_natural_prevalence_probability', 'degenerate_gate': 'fewer_than_2_eligible_paired_median_columns', 'constant_numeric_score_diagnostic': 'report_only_no_pool_change', 'label_performance_rule_selection': False, 'held_data_used_to_change_factory': False, 'inference_new_rules': 0, 'shared_pool': 'one bank per fold shared by A1/A2/A3; A0 is a separate comparator', 'comparison_limit': 'source factory is not an implemented full comparison or evidence of advantage over tuned traditional ML'}


class RuleBankAuditError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise RuleBankAuditError(message)


def canonical(v):
    try:
        return json.dumps(v, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise RuleBankAuditError("Nonfinite/non-JSON declaration") from exc


def digest(v):
    return hashlib.sha256(canonical(v).encode("utf-8")).hexdigest()


def exact(v, fields, name):
    require(type(v) is dict and set(v) == set(fields), "Unexpected " + name + " fields")


def hash_value(v):
    require(type(v) is str and len(v) == 64 and all(c in "0123456789abcdef" for c in v), "Invalid SHA256")


def compare_tree(actual, expected, path="work_product"):
    # Labels/counts/bools and binary64 bits may not pass through Python's loose
    # equality (e.g. True==1, -0.0==+0.0). No tolerance in scientific outputs.
    require(type(actual) is type(expected), "Type mismatch: " + path)
    if type(expected) is dict:
        require(set(actual) == set(expected), "Key mismatch: " + path)
        for key in expected:
            compare_tree(actual[key], expected[key], path + "." + key)
    elif type(expected) is list:
        require(len(actual) == len(expected), "Length mismatch: " + path)
        for i, v in enumerate(expected):
            compare_tree(actual[i], v, path + "[" + str(i) + "]")
    elif type(expected) is float:
        require(math.isfinite(actual) and actual.hex() == expected.hex(), "Binary64 mismatch: " + path)
    else:
        require(actual == expected, "Value mismatch: " + path)


def predicate(atoms):
    # Grammar/schema shared, normalization independently implemented as records
    # sorted by scalar semantic key. Signed-zero threshold alone canonicalizes.
    unique = {}
    for j, op, value in atoms:
        require(type(j) is int and 0 <= j < 127 and op in ("<=", ">"), "Predicate domain")
        require(type(value) is float and math.isfinite(value), "Predicate binary64 threshold")
        value = 0.0 if value == 0.0 else value
        key = (j, op, value.hex())
        unique[key] = {"feature_index": j, "operator": op, "threshold": value}
    require(1 <= len(unique) <= 8, "Predicate atom budget")
    return {"dsl_version": DSL, "all": [unique[k] for k in sorted(unique)]}


def members(rows, condition):
    result = []
    for i, r in enumerate(rows):
        hit = True
        for term in condition["all"]:
            x = r["X127"][term["feature_index"]]
            hit = x <= term["threshold"] if term["operator"] == "<=" else x > term["threshold"]
            if not hit:
                break
        if hit:
            result.append(i)
    return result


def quantile(rows, ws, j, q):
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["X127"][j], rows[i]["source_index"], rows[i]["original_row_index"]))
    total = math.fsum(ws)
    target = q * total
    # Independent search over prefix lengths, followed by exact rank-crossing
    # reconstruction. No producer quantile helper or approximate cumsum.
    insufficient, enough = 0, len(order)
    while enough - insufficient > 1:
        count = (insufficient + enough) // 2
        prefix = math.fsum(ws[order[i]] for i in range(count))
        if prefix >= target:
            enough = count
        else:
            insufficient = count
    rank = enough - 1
    before = math.fsum(ws[order[i]] for i in range(rank))
    through = math.fsum(ws[order[i]] for i in range(rank + 1))
    require(before < target <= through, "Quantile first-prefix crossing")
    row = rows[order[rank]]
    raw = row["X127"][j]
    threshold = 0.0 if raw == 0.0 else raw
    witness = {"feature_index": j, "q": q, "total_ws": total, "target_ws": target,
        "selected_sorted_rank": rank, "prefix_ws_before": before, "prefix_ws_through": through,
        "selected_source_index": row["source_index"], "selected_original_row_index": row["original_row_index"],
        "selected_raw_binary64_hex": raw.hex(), "threshold_binary64_hex": threshold.hex(),
        "sorted_source_index_digest": digest([rows[i]["source_index"] for i in order])}
    return threshold, witness


def local_statistics(rows, ws, hits, C, W):
    by_class = [[], []]
    class_groups = [set(), set()]
    hit_groups = set()
    for i in hits:
        y = rows[i]["label"]
        by_class[y].append(i)
        class_groups[y].add(rows[i]["component_id"])
        hit_groups.add(rows[i]["component_id"])
    U = [math.fsum(ws[i] for i in by_class[y]) for y in (0, 1)]
    # FullTRAIN factor once, no matched-group denominator or second conversion.
    factors = [W / (2 * C[y]) for y in (0, 1)]
    A = [U[y] * factors[y] for y in (0, 1)]
    score = (A[1] + 1.0) / (A[0] + A[1] + 2.0)
    prediction = int(score >= 0.5)
    counter = []
    for i in hits:
        r = rows[i]
        if r["label"] != prediction:
            counter.append({"source_index": r["source_index"], "component_id": r["component_id"],
                "curated_threshold_label": r["label"], "rule_threshold_prediction": prediction,
                "claim_scope": "TRAIN_curated_label_disagreement_not_physical_counterexample"})
    counter.sort(key=lambda r: r["source_index"])
    return {"support_source_indices": sorted(rows[i]["source_index"] for i in hits),
        "matched_class_row_counts": [len(v) for v in by_class], "matched_class_ws_mass": U,
        "balanced_class_mass": A, "numeric_rule_score": score, "matched_raw_rows": len(hits),
        "matched_original_groups": len(hit_groups), "component_ids": sorted(hit_groups),
        "matched_class_original_group_counts": [len(v) for v in class_groups],
        "counterexamples": counter, "counterexample_raw_rows": len(counter),
        "counterexample_original_groups": len({r["component_id"] for r in counter}),
        "class_group_counts_may_overlap_for_mixed_label_groups": True}


def check_inputs(rows, groupcounts, scope, work, binding):
    exact(binding, BINDING_FIELDS, "ROOT binding")
    require(binding["execution_kind"] in ("invented_RAM", "ROOT_actual"), "Declared execution kind")
    for key in BINDING_FIELDS[1:]:
        hash_value(binding[key])
    require(binding["factory_source_sha256"] == FACTORY_SHA256 and binding["executor_source_sha256"] == EXECUTOR_SHA256,
            "Fixed component SOURCE pins changed")
    exact(scope, SCOPE_FIELDS, "TRAIN scope")
    require(type(scope["partition_id"]) is str and bool(scope["partition_id"]), "Partition identity")
    require(type(scope["held_fold"]) is int and scope["held_fold"] in (0, 1, 2), "Held fold")
    for key in ("source_container_sha256", "feature_schema_sha256", "rule_factory_recipe_sha256"):
        hash_value(scope[key])
    require(scope["source_container_sha256"] == binding["source_container_sha256"]
            and scope["feature_schema_sha256"] == binding["feature_schema_sha256"], "Source/schema closure")
    require(scope["rule_factory_recipe_sha256"] == digest(EXPECTED_RECIPE), "Frozen recipe differs")
    require(type(rows) is list and 1 <= len(rows) <= 3764, "TRAIN row budget")
    require(type(groupcounts) is dict and groupcounts, "Fullcohort group counts")
    for group, count in groupcounts.items():
        require(type(group) is str and bool(group) and type(count) is int and count > 0, "Full original group denominator")
    if binding["execution_kind"] == "ROOT_actual":
        require(len(groupcounts) == 1117 and sum(groupcounts.values()) == 3764, "Original complete cohort counts")
    require(type(scope["held_component_ids"]) is list and scope["held_component_ids"]
            and all(type(g) is str and g for g in scope["held_component_ids"])
            and len(set(scope["held_component_ids"])) == len(scope["held_component_ids"]), "Explicit held groups")
    source_ids, original_ids, folds, counts = [], [], {}, {}
    projected = []
    for row in rows:
        exact(row, ROW_FIELDS, "Primitive TRAIN row")
        for key in ("source_index", "original_row_index"):
            require(type(row[key]) is int and row[key] >= 0, "Original row/source identity")
        require(type(row["component_id"]) is str and bool(row["component_id"]), "Original group identity")
        require(row["role"] == "TRAIN" and type(row["fold"]) is int and row["fold"] in (0, 1, 2)
                and row["fold"] != scope["held_fold"], "Held/nonTRAIN row in audit")
        require(type(row["label"]) is int and row["label"] in (0, 1), "Assigned curated binary labels only")
        require(type(row["X127"]) is list and len(row["X127"]) == 127
                and all(type(v) is float and math.isfinite(v) for v in row["X127"]), "Uncast finite raw X127")
        source_ids.append(row["source_index"]); original_ids.append(row["original_row_index"])
        group = row["component_id"]
        require(group not in folds or folds[group] == row["fold"], "Group split between folds")
        folds[group] = row["fold"]; counts[group] = counts.get(group, 0) + 1
        projected.append({key: row[key] for key in ("source_index", "component_id", "role", "fold")})
    require(len(set(source_ids)) == len(rows) and len(set(original_ids)) == len(rows), "Repeated original rows")
    require(original_ids == sorted(original_ids), "Primitive rows must follow original array order")
    compare_tree(scope["TRAIN_rows"], projected, "scope.TRAIN_rows")
    compare_tree(scope["TRAIN_original_row_indices"], original_ids, "scope.TRAIN_original_row_indices")
    require(set(counts).isdisjoint(scope["held_component_ids"]), "TRAIN/held group overlap")
    require(set(scope["held_component_ids"]) == set(groupcounts) - set(counts), "Held groups are not exact cohort complement")
    for group, count in counts.items():
        require(groupcounts.get(group) == count, "Partial group or hit-count renormalized denominator")
    for key, value in (("scope_sha256", scope), ("primitive_rows_sha256", rows), ("groupcounts_sha256", groupcounts),
                       ("work_product_sha256", work)):
        require(binding[key] == digest(value), "ROOT-bound declaration changed: " + key)
    return counts


def reconstruct(rows, groupcounts, scope):
    ws = [1.0 / groupcounts[r["component_id"]] for r in rows]
    W = math.fsum(ws)
    C = [math.fsum(ws[i] for i, r in enumerate(rows) if r["label"] == y) for y in (0, 1)]
    require(all(v > 0.0 and math.isfinite(v) for v in C), "Both TRAIN classes required")
    raw_counts = [sum(r["label"] == y for r in rows) for y in (0, 1)]
    thresholds, witnesses = {}, []
    for j in range(127):
        for q in ((0.5,) if j < 118 else (0.25, 0.5, 0.75)):
            thresholds[j, q], witness = quantile(rows, ws, j, q)
            witnesses.append(witness)
    candidates = {}; generated = 0
    def add(atoms, origin):
        nonlocal generated
        generated += 1
        p = predicate(atoms); key = digest(p)
        if key not in candidates:
            candidates[key] = {"predicate": p, "origins": []}
        candidates[key]["origins"].append(origin)
    for j in range(127):
        for op in ("<=", ">"):
            add([(j, op, thresholds[j, 0.5])], "median:" + str(j) + ":" + op)
    for j in range(118, 127):
        for q in (0.25, 0.75):
            for op in ("<=", ">"):
                add([(j, op, thresholds[j, q])], "geometry-quantile:" + str(j) + ":" + str(q) + ":" + op)
    for j in range(118, 127):
        for k in range(j + 1, 127):
            for op_j in ("<=", ">"):
                for op_k in ("<=", ">"):
                    add([(j, op_j, thresholds[j, 0.5]), (k, op_k, thresholds[k, 0.5])],
                        "geometry-AND:" + str(j) + ":" + op_j + ":" + str(k) + ":" + op_k)
    require(generated == 434 and len(candidates) <= 434, "Fixed candidate budget")
    statistics = {}
    for key, candidate in candidates.items():
        hits = members(rows, candidate["predicate"])
        statistics[key] = local_statistics(rows, ws, hits, C, W)
    def sufficiently_supported(key):
        r = statistics[key]
        return r["matched_raw_rows"] >= 40 and r["matched_original_groups"] >= 10
    eligible = []
    for j in range(127):
        low = digest(predicate([(j, "<=", thresholds[j, 0.5])]))
        high = digest(predicate([(j, ">", thresholds[j, 0.5])]))
        if sufficiently_supported(low) and sufficiently_supported(high):
            eligible.append(j)
    retained, rejected = [], []
    for key in sorted(candidates):
        candidate = candidates[key]
        origins = sorted(set(candidate["origins"]))
        only_median = all(v.startswith("median:") for v in origins)
        pair_gate = not only_median or candidate["predicate"]["all"][0]["feature_index"] in eligible
        r = statistics[key]
        if not sufficiently_supported(key) or not pair_gate:
            rejected.append({"predicate_sha256": key, "origins": origins,
                "reason": "minimum_support" if not sufficiently_supported(key) else "median_pair_other_side_failed",
                "matched_raw_rows": r["matched_raw_rows"], "matched_original_groups": r["matched_original_groups"]})
        else:
            rid = "R_" + digest({"partition_id": scope["partition_id"], "feature_schema_sha256": scope["feature_schema_sha256"], "predicate_sha256": key})
            retained.append({"rule_id": rid, "predicate": candidate["predicate"], "predicate_sha256": key,
                             "factory_origins": origins, **r})
    retained.sort(key=lambda r: r["rule_id"])
    scores = [r["numeric_rule_score"] for r in retained]
    distinct = len({s.hex() for s in scores}); degenerate = len(eligible) < 2
    fullcounts = {}
    for row in rows:
        group = row["component_id"]; fullcounts[group] = fullcounts.get(group, 0) + 1
    rebuilt = {"schema_version": "unapproved-TRAIN-rule-work-product/v1", "factory_version": EXPECTED_RECIPE["factory_version"],
        "recipe": json.loads(canonical(EXPECTED_RECIPE)), "recipe_sha256": digest(EXPECTED_RECIPE),
        "partition_id": scope["partition_id"], "scope": json.loads(canonical(scope)), "trusted_scope_sha256": digest(scope),
        "global_statistics": {"TRAIN_class_ws_mass": C, "TRAIN_class_row_counts": raw_counts, "W": W,
            "TRAIN_raw_rows": len(rows), "TRAIN_original_groups": len(fullcounts),
            "TRAIN_class_original_group_counts": [len({r["component_id"] for r in rows if r["label"] == y}) for y in (0, 1)],
            "class_group_counts_may_overlap_for_mixed_label_groups": True,
            "group_row_counts": [{"component_id": g, "n_rows": fullcounts[g]} for g in sorted(fullcounts)]},
        "quantile_witnesses": witnesses, "generated_candidates": generated, "canonical_candidates": len(candidates),
        "eligible_paired_median_columns": eligible, "rules": retained, "rejected_candidates": rejected,
        "diagnostics": {"degenerate_overlap_bank": degenerate,
            "degenerate_reason": "fewer_than_2_eligible_paired_median_columns" if degenerate else None,
            "numeric_rule_use_allowed": not degenerate, "all_finite_queries_have_at_least_two_median_rules_when_nondegenerate": not degenerate,
            "distinct_numeric_rule_scores": distinct, "constant_numeric_score_bank": distinct <= 1,
            "numeric_score_range": [min(scores), max(scores)] if scores else None,
            "constant_scores_diagnostic_does_not_select_or_change_pool": True, "correlated_rules_are_not_independent_evidence": True},
        "resource_ledger": {"shared_outer_TRAIN_rule_bank_builds": 1, "TRAIN_quantile_arithmetic_builds": 1,
            "supervised_reference_builds": 1, "native_classifier_fits": 0, "native_scaler_fits": 0, "native_model_queries": 0,
            "model_deserializations": 0, "new_geometry_or_property_reads": 0, "new_descriptor_columns": 0,
            "inner_validation_or_parameter_searches": 0, "threshold_columns": 127, "threshold_quantiles": len(witnesses),
            "candidate_row_predicate_evaluations": len(rows) * len(candidates), "maximum_candidate_row_predicate_evaluations": 3764 * 434,
            "retained_rule_support_memberships": sum(len(r["support_source_indices"]) for r in retained),
            "shared_bank_rebuilds_for_other_arms": 0, "inference_or_candidate_score_passes_performed_by_factory": 0},
        "claims": {"audited_or_owner_approved": False, "actual_scientific_evaluation_performed": False,
            "causal_physical_mechanism_established": False, "blind_generalization_established": False,
            "full_atomistic_or_measured_structure_available": False, "sufficient_tuned_ML_comparison_implemented": False,
            "external_source_and_manifest_authentication_required": True}}
    return rebuilt


def audit_training_work_product(TRAIN_rows, full_original_group_counts, scope, work_product, *, root_binding):
    """Future ROOT call; no run or certificate exists merely by creating SOURCE.

    The caller authenticates actual input/container/SOURCE/authorization bytes
    before calling; this pure function validates declared SHA bindings and all
    numeric fields, but cannot itself independently read/authenticate files.
    """
    counts = check_inputs(TRAIN_rows, full_original_group_counts, scope, work_product, root_binding)
    independent = reconstruct(TRAIN_rows, full_original_group_counts, scope)
    compare_tree(work_product, independent)
    require(digest(independent) == root_binding["work_product_sha256"], "Complete work-product digest differs")
    actual = root_binding["execution_kind"] == "ROOT_actual"
    certificate = None
    if actual:
        certificate = {"schema_version": "TRAIN-rule-statistics-audit/v1", "checks_passed": True,
            "work_product_sha256": digest(independent), "partition_id": scope["partition_id"],
            "source_container_sha256": scope["source_container_sha256"], "feature_schema_sha256": scope["feature_schema_sha256"],
            "recipe_sha256": digest(EXPECTED_RECIPE), "auditor_source_sha256": root_binding["auditor_source_sha256"]}
    return {"schema_version": "independent-pure-TRAIN-rule-work-product-audit/v1", "checks_passed": True,
        "execution_kind": root_binding["execution_kind"], "certificate": certificate,
        "scope_sha256": digest(scope), "primitive_rows_sha256": digest(TRAIN_rows),
        "groupcounts_sha256": digest(full_original_group_counts), "work_product_sha256": digest(independent),
        "TRAIN_raw_rows": len(TRAIN_rows), "TRAIN_original_groups": len(counts), "quantile_witnesses_verified": 145,
        "candidate_generations_verified": 434, "canonical_candidates_verified": independent["canonical_candidates"],
        "retained_rules_verified": len(independent["rules"]), "rejected_candidates_verified": len(independent["rejected_candidates"]),
        "full_original_group_denominators_verified": True, "all_work_product_fields_type_and_binary64_verified": True,
        "mixed_label_class_group_overlap_preserved": True, "complete_TRAIN_counterexample_sets_verified": True,
        "global_class_balance_applied_once_per_local_score": True, "fixed_Laplace_per_class_one_verified": True,
        "labels_are_assigned_curated_not_newly_measured_outcomes": True,
        "actual_source_authentication_performed_by_pure_auditor": False,
        "owner_approval_map_issued": False, "held_features_or_labels_consumed": 0,
        "new_producer_bank_builds_performed_by_auditor": 0, "independent_statistical_reconstructions": 1,
        "native_fits_queries_deserializations": 0, "new_geometry_property_reads": 0,
        "candidate_scientific_evaluation_score_records": 0, "physical_or_blind_or_agent_superiority_verified": False}
