"""Fixed outer-TRAIN X127 empirical rule factory, SOURCE only until ROOT approval.

No I/O, numpy, sklearn, model import, fit, query, calibration, or randomness.
Trusted ROOT supplies a source-authenticated TRAIN manifest and assigned curated
binary labels. JSON declarations alone do not authenticate the source/cohort.
The factory is not a case agent, complete experiment, or sufficient strong-ML
comparison. X127 is an occupancy/computed-geometry proxy, not measured full atomic
structure. Source-only tests are invented RAM and have not been executed.

Two stages avoid an audit-reference cycle: mine once -> independently audit the
unapproved work product -> ROOT authenticates that certificate and finalizes cards.
Owner approval maps must be issued separately by the trusted broker/ROOT; this
module does not self-approve its own statistical output.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import itertools
import json
import math
import re

FACTORY_VERSION = 'TRAIN-X127-weighted-quantile-overlap/v1'
DSL_VERSION = 'raw-X127-AND/v1'
EXECUTOR_ID = 'balanced-local-rule-score/v1'
AGGREGATION = 'sorted-rule-ID-equal-contributions-fsum-max3/v1'
SCORE_SEMANTICS = 'class_balanced_decision_score_not_natural_prevalence_probability'
MAX_CANDIDATES = 434
MIN_ROWS = 40
MIN_GROUPS = 10
MAX_TRAIN_ROWS = 3764
GEOMETRY_COLUMNS = tuple(range(118, 127))
FEATURE_NAMES = tuple(f'occupancy_fraction_Z{j+1:03d}' for j in range(118)) + (
    'volume_per_occupied_atom_A3',
    'nearest_positive_PBC_distance_site_occ_mean_A',
    'nearest_positive_PBC_distance_site_occ_std_A',
    'nearest_positive_PBC_distance_min_A', 'nearest_positive_PBC_distance_max_A',
    'coordination6A_site_occ_mean', 'coordination6A_site_occ_std',
    'PBC_edge_distance_pair_occ_mean_A', 'PBC_edge_distance_pair_occ_std_A')
GEOMETRY_UNITS = ('A^3', 'A', 'A', 'A', 'A', 'dimensionless contact count',
                  'dimensionless contact count', 'A', 'A')
MASS_ATOL = 1e-10
MASS_RTOL = 2e-12

class FactoryContractError(ValueError):
    pass

def need(ok, message):
    if not ok:
        raise FactoryContractError(message)

def exact(v, keys, name):
    need(type(v) is dict and set(v) == set(keys), 'Invalid fields: ' + name)

def text(v, name):
    need(type(v) is str and bool(v.strip()), 'Nonempty text required: ' + name)
    return v

def sha(v, name):
    need(type(v) is str and re.fullmatch(r'[0-9a-f]{64}', v) is not None, 'Invalid SHA: ' + name)
    return v

def canonical(v):
    try:
        return json.dumps(v, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    except (TypeError, ValueError) as exc:
        raise FactoryContractError('Noncanonical/nonfinite value') from exc

def digest(v):
    return hashlib.sha256(canonical(v).encode('utf8')).hexdigest()

def close(a, b):
    return math.isfinite(a) and math.isfinite(b) and abs(a-b) <= MASS_ATOL + MASS_RTOL*abs(b)

def recipe():
    return {'factory_version': FACTORY_VERSION, 'task_id': 'curated_Tc_gt_10K',
        'features': 'original_X127_only', 'feature_count': 127,
        'empirical_scope': 'outer_TRAIN_assigned_curated_binary_labels_only',
        'seed': 20261010, 'uses_randomness': False,
        'quantile_weight': '1/full_outer_TRAIN_original_group_row_count',
        'quantile_definition': 'first sorted k with math.fsum(ws[:k+1]) >= q*math.fsum(all_ws)',
        'quantile_sort': ['raw_binary64_value', 'source_index', 'original_row_index'],
        'median_halfspaces': {'feature_indices': list(range(127)), 'q': 0.5, 'both_sides_must_pass': True},
        'geometry_extra_halfspaces': {'feature_indices': list(GEOMETRY_COLUMNS), 'q': [0.25, 0.75]},
        'geometry_AND': 'all unordered distinct geometry-column pairs, four median-direction combinations',
        'maximum_generated_candidates': MAX_CANDIDATES, 'minimum_raw_rows': MIN_ROWS,
        'minimum_original_groups': MIN_GROUPS, 'support_gates_use_labels': False,
        'predicate_normalization': 'sorted(index,operator,threshold.hex); duplicate atoms collapse; signed-zero threshold +0; no epsilon or bound simplification',
        'predicate_dedup': 'one normalized predicate digest per pool, not TRAIN-mask equivalence',
        'numeric_score': '(U1*W/(2*C1)+1)/(U0*W/(2*C0)+U1*W/(2*C1)+2)',
        'smoothing_per_class': 1.0, 'maximum_selected_distinct_applicable_rules': 3,
        'aggregation': AGGREGATION, 'score_semantics': SCORE_SEMANTICS,
        'degenerate_gate': 'fewer_than_2_eligible_paired_median_columns',
        'constant_numeric_score_diagnostic': 'report_only_no_pool_change',
        'label_performance_rule_selection': False, 'held_data_used_to_change_factory': False,
        'inference_new_rules': 0, 'shared_pool': 'one bank per fold shared by A1/A2/A3; A0 is a separate comparator',
        'comparison_limit': 'source factory is not an implemented full comparison or evidence of advantage over tuned traditional ML'}

def normalize_predicate(dsl):
    exact(dsl, ('dsl_version', 'all'), 'AND predicate')
    need(dsl['dsl_version'] == DSL_VERSION and type(dsl['all']) is list and 1 <= len(dsl['all']) <= 8,
         'Invalid AND version/atom count')
    atoms = {}
    for term in dsl['all']:
        exact(term, ('feature_index', 'operator', 'threshold'), 'predicate atom')
        j, op, t = term['feature_index'], term['operator'], term['threshold']
        need(type(j) is int and 0 <= j < 127 and op in ('<=', '>'), 'Forbidden feature/operator')
        need(type(t) is float and math.isfinite(t), 'Threshold must be finite binary64 float')
        t = 0.0 if t == 0.0 else t
        atoms[(j, op, t.hex())] = {'feature_index': j, 'operator': op, 'threshold': t}
    return {'dsl_version': DSL_VERSION, 'all': [atoms[key] for key in sorted(atoms)]}

def matches(row, predicate):
    need(type(row) in (tuple, list) and len(row) == 127
         and all(type(x) is float and math.isfinite(x) for x in row), 'Finite raw X127 required')
    normalized = normalize_predicate(predicate)
    need(canonical(normalized) == canonical(predicate), 'Predicate must already be normalized')
    return _raw_match(row, predicate)

def _raw_match(row, predicate):
    # Caller has already checked every X127 float and normalized the predicate.
    return all(row[a['feature_index']] <= a['threshold'] if a['operator'] == '<='
               else row[a['feature_index']] > a['threshold'] for a in predicate['all'])

def validate_scope(X127, labels, group_ids, source_indices, partition_id, scope, trusted_scope_sha256):
    exact(scope, ('partition_id', 'held_fold', 'source_container_sha256', 'feature_schema_sha256',
                 'TRAIN_rows', 'held_component_ids', 'TRAIN_original_row_indices', 'rule_factory_recipe_sha256'),
          'trusted TRAIN scope')
    sha(trusted_scope_sha256, 'ROOT-frozen scope'); need(digest(scope) == trusted_scope_sha256, 'Scope differs from supplied ROOT binding')
    text(partition_id, 'partition'); need(scope['partition_id'] == partition_id, 'Wrong partition')
    need(type(scope['held_fold']) is int and scope['held_fold'] in (0, 1, 2), 'Invalid held fold')
    for key in ('source_container_sha256', 'feature_schema_sha256', 'rule_factory_recipe_sha256'):
        sha(scope[key], key)
    need(scope['rule_factory_recipe_sha256'] == digest(recipe()), 'Fixed recipe was changed')
    n = len(X127)
    need(1 <= n <= MAX_TRAIN_ROWS and all(len(v) == n for v in (labels, group_ids, source_indices,
         scope['TRAIN_rows'], scope['TRAIN_original_row_indices'])), 'Input/scope lengths')
    rows = []
    for row in X127:
        need(type(row) in (tuple, list) and len(row) == 127 and all(type(x) is float and math.isfinite(x) for x in row), 'No cast/missing/nonfinite X')
        rows.append(tuple(row))
    need(all(type(y) is int and y in (0, 1) for y in labels), 'Only assigned curated binary labels, never null/unknown')
    need(all(type(i) is int and i >= 0 for i in source_indices) and len(set(source_indices)) == n, 'Unique original source IDs')
    original_rows = scope['TRAIN_original_row_indices']
    need(all(type(i) is int and i >= 0 for i in original_rows) and len(set(original_rows)) == n, 'Unique original row indices')
    need(type(scope['held_component_ids']) is list and len(scope['held_component_ids']) > 0
         and all(type(g) is str and g for g in scope['held_component_ids'])
         and len(set(scope['held_component_ids'])) == len(scope['held_component_ids']), 'Explicit held group exclusion set')
    group_folds = {}
    for k, r in enumerate(scope['TRAIN_rows']):
        exact(r, ('source_index', 'component_id', 'role', 'fold'), 'TRAIN identity row')
        need(type(r['source_index']) is int and r['source_index'] == source_indices[k]
             and type(r['component_id']) is str and r['component_id'] == group_ids[k], 'Input identities differ from bound TRAIN scope')
        text(group_ids[k], 'original group')
        need(r['role'] == 'TRAIN' and type(r['fold']) is int and r['fold'] in (0, 1, 2)
             and r['fold'] != scope['held_fold'], 'Non-TRAIN/held row used by rule builder')
        if group_ids[k] in group_folds:
            need(group_folds[group_ids[k]] == r['fold'], 'Association group split across folds')
        group_folds[group_ids[k]] = r['fold']
    need(set(group_ids).isdisjoint(scope['held_component_ids']), 'TRAIN/held group intersection')
    return tuple(rows), tuple(labels), tuple(group_ids), tuple(source_indices), tuple(original_rows)

def weighted_quantile(values, weights, source_indices, original_rows, q):
    need(type(q) is float and q in (0.25, 0.5, 0.75), 'Only frozen quantiles')
    n = len(values)
    need(n > 0 and len(weights) == len(source_indices) == len(original_rows) == n, 'Quantile lengths')
    need(all(type(v) is float and math.isfinite(v) for v in values)
         and all(type(w) is float and math.isfinite(w) and w > 0 for w in weights), 'Quantile finite values/positive ws')
    order = sorted(range(n), key=lambda i: (values[i], source_indices[i], original_rows[i]))
    total = math.fsum(weights); target = q * total
    # Exact specified prefix fsum via logarithmically many probes, not a cumsum.
    lo, hi = 0, n-1
    while lo < hi:
        mid = (lo+hi)//2
        if math.fsum(weights[order[j]] for j in range(mid+1)) >= target:
            hi = mid
        else:
            lo = mid+1
    k = lo; picked = order[k]; value = values[picked]
    before = math.fsum(weights[order[j]] for j in range(k))
    after = math.fsum(weights[order[j]] for j in range(k+1))
    need((k == 0 or before < target) and after >= target, 'Weighted quantile rank')
    threshold = 0.0 if value == 0.0 else value
    return threshold, {'q': q, 'total_ws': total, 'target_ws': target, 'selected_sorted_rank': k,
        'prefix_ws_before': before, 'prefix_ws_through': after, 'selected_source_index': source_indices[picked],
        'selected_original_row_index': original_rows[picked], 'selected_raw_binary64_hex': value.hex(),
        'threshold_binary64_hex': threshold.hex(), 'sorted_source_index_digest': digest([source_indices[i] for i in order])}

def local_statistics(rows, labels, groups, indices, ws, predicate, global_class_mass, W):
    normalized = normalize_predicate(predicate)
    need(canonical(normalized) == canonical(predicate), 'Local-stat predicate must be canonical')
    hit = [i for i, row in enumerate(rows) if _raw_match(row, predicate)]
    counts = [sum(labels[i] == y for i in hit) for y in (0, 1)]
    U = [math.fsum(ws[i] for i in hit if labels[i] == y) for y in (0, 1)]
    need(close(math.fsum(U), math.fsum(ws[i] for i in hit)), 'Local ws mass mismatch')
    factors = [W/(2*global_class_mass[y]) for y in (0, 1)]
    A = [U[y]*factors[y] for y in (0, 1)]
    score = (A[1]+1.0)/(A[0]+A[1]+2.0)
    support = sorted(indices[i] for i in hit)
    all_groups = sorted({groups[i] for i in hit})
    counter = sorted(({'source_index': indices[i], 'component_id': groups[i],
                       'curated_threshold_label': labels[i], 'rule_threshold_prediction': int(score >= 0.5),
                       'claim_scope': 'TRAIN_curated_label_disagreement_not_physical_counterexample'}
                      for i in hit if labels[i] != int(score >= 0.5)), key=lambda r: r['source_index'])
    return {'support_source_indices': support, 'matched_class_row_counts': counts,
        'matched_class_ws_mass': U, 'balanced_class_mass': A, 'numeric_rule_score': score,
        'matched_raw_rows': len(hit), 'matched_original_groups': len(all_groups), 'component_ids': all_groups,
        'matched_class_original_group_counts': [len({groups[i] for i in hit if labels[i] == y}) for y in (0, 1)],
        'counterexamples': counter, 'counterexample_raw_rows': len(counter),
        'counterexample_original_groups': len({r['component_id'] for r in counter}),
        'class_group_counts_may_overlap_for_mixed_label_groups': True}


def mine_training_pool(X127, labels, group_ids, source_indices, partition_id, *,
                       trusted_partition_manifest, trusted_scope_sha256):
    rows, y, g, ids, original_rows = validate_scope(X127, labels, group_ids, source_indices,
        partition_id, trusted_partition_manifest, trusted_scope_sha256)
    n = len(rows); group_counts = Counter(g); ws = tuple(1.0/group_counts[group] for group in g)
    W = math.fsum(ws)
    C = [math.fsum(ws[i] for i in range(n) if y[i] == c) for c in (0, 1)]
    need(all(v > 0 and math.isfinite(v) for v in C) and close(math.fsum(C), W), 'Both TRAIN classes/full group denominator required')
    counts = [sum(v == c for v in y) for c in (0, 1)]
    thresholds = {}; quantile_witnesses = []
    for j in range(127):
        qs = (0.25, 0.5, 0.75) if j in GEOMETRY_COLUMNS else (0.5,)
        for q in qs:
            t, witness = weighted_quantile(tuple(row[j] for row in rows), ws, ids, original_rows, q)
            thresholds[j, q] = t
            quantile_witnesses.append({'feature_index': j, **witness})
    candidates = {}; generated = 0
    def emit(atoms, origin):
        nonlocal generated
        generated += 1
        predicate = normalize_predicate({'dsl_version': DSL_VERSION, 'all': atoms})
        key = digest(predicate)
        if key not in candidates:
            candidates[key] = {'predicate': predicate, 'origins': []}
        else:
            need(canonical(candidates[key]['predicate']) == canonical(predicate), 'Predicate hash collision')
        candidates[key]['origins'].append(origin)
    def atom(j, q, op):
        return {'feature_index': j, 'operator': op, 'threshold': thresholds[j, q]}
    for j in range(127):
        for op in ('<=', '>'):
            emit([atom(j, 0.5, op)], f'median:{j}:{op}')
    for j in GEOMETRY_COLUMNS:
        for q in (0.25, 0.75):
            for op in ('<=', '>'):
                emit([atom(j, q, op)], f'geometry-quantile:{j}:{q}:{op}')
    for j, k in itertools.combinations(GEOMETRY_COLUMNS, 2):
        for a, z in itertools.product(('<=', '>'), repeat=2):
            emit([atom(j, 0.5, a), atom(k, 0.5, z)], f'geometry-AND:{j}:{a}:{k}:{z}')
    need(generated == MAX_CANDIDATES and len(candidates) <= MAX_CANDIDATES, 'Fixed candidate ceiling')
    statistics = {key: local_statistics(rows, y, g, ids, ws, c['predicate'], C, W)
                  for key, c in candidates.items()}
    def support_pass(key):
        a = statistics[key]
        return a['matched_raw_rows'] >= MIN_ROWS and a['matched_original_groups'] >= MIN_GROUPS
    eligible_median_columns = []
    for j in range(127):
        keys = [digest(normalize_predicate({'dsl_version': DSL_VERSION, 'all': [atom(j, 0.5, op)]}))
                for op in ('<=', '>')]
        if all(support_pass(key) for key in keys):
            eligible_median_columns.append(j)
    retained = []; rejected = []
    for key in sorted(candidates):
        c = candidates[key]; origins = sorted(set(c['origins']))
        only_median = all(origin.startswith('median:') for origin in origins)
        paired = not only_median or c['predicate']['all'][0]['feature_index'] in eligible_median_columns
        if not support_pass(key) or not paired:
            rejected.append({'predicate_sha256': key, 'origins': origins,
                'reason': 'minimum_support' if not support_pass(key) else 'median_pair_other_side_failed',
                'matched_raw_rows': statistics[key]['matched_raw_rows'],
                'matched_original_groups': statistics[key]['matched_original_groups']})
            continue
        rid = 'R_' + digest({'partition_id': partition_id,
            'feature_schema_sha256': trusted_partition_manifest['feature_schema_sha256'], 'predicate_sha256': key})
        retained.append({'rule_id': rid, 'predicate': c['predicate'], 'predicate_sha256': key,
                         'factory_origins': origins, **statistics[key]})
    retained.sort(key=lambda r: r['rule_id'])
    need(len({r['predicate_sha256'] for r in retained}) == len(retained), 'Same predicate cannot occupy multiple selections')
    degenerate = len(eligible_median_columns) < 2
    scores = [r['numeric_rule_score'] for r in retained]
    score_hex = {s.hex() for s in scores}
    support_totals = sum(len(r['support_source_indices']) for r in retained)
    return {'schema_version': 'unapproved-TRAIN-rule-work-product/v1', 'factory_version': FACTORY_VERSION,
        'recipe': recipe(), 'recipe_sha256': digest(recipe()), 'partition_id': partition_id,
        'scope': json.loads(canonical(trusted_partition_manifest)), 'trusted_scope_sha256': trusted_scope_sha256,
        'global_statistics': {'TRAIN_class_ws_mass': C, 'TRAIN_class_row_counts': counts,
            'W': W, 'TRAIN_raw_rows': n, 'TRAIN_original_groups': len(group_counts),
            'TRAIN_class_original_group_counts': [len({g[i] for i in range(n) if y[i] == c}) for c in (0, 1)],
            'class_group_counts_may_overlap_for_mixed_label_groups': True,
            'group_row_counts': [{'component_id': group, 'n_rows': group_counts[group]} for group in sorted(group_counts)]},
        'quantile_witnesses': quantile_witnesses, 'generated_candidates': generated,
        'canonical_candidates': len(candidates), 'eligible_paired_median_columns': eligible_median_columns,
        'rules': retained, 'rejected_candidates': rejected,
        'diagnostics': {'degenerate_overlap_bank': degenerate,
            'degenerate_reason': 'fewer_than_2_eligible_paired_median_columns' if degenerate else None,
            'numeric_rule_use_allowed': not degenerate,
            'all_finite_queries_have_at_least_two_median_rules_when_nondegenerate': not degenerate,
            'distinct_numeric_rule_scores': len(score_hex),
            'constant_numeric_score_bank': len(score_hex) <= 1,
            'numeric_score_range': [min(scores), max(scores)] if scores else None,
            'constant_scores_diagnostic_does_not_select_or_change_pool': True,
            'correlated_rules_are_not_independent_evidence': True},
        'resource_ledger': {'shared_outer_TRAIN_rule_bank_builds': 1, 'TRAIN_quantile_arithmetic_builds': 1,
            'supervised_reference_builds': 1, 'native_classifier_fits': 0, 'native_scaler_fits': 0,
            'native_model_queries': 0, 'model_deserializations': 0, 'new_geometry_or_property_reads': 0,
            'new_descriptor_columns': 0, 'inner_validation_or_parameter_searches': 0,
            'threshold_columns': 127, 'threshold_quantiles': len(quantile_witnesses),
            'candidate_row_predicate_evaluations': n*len(candidates),
            'maximum_candidate_row_predicate_evaluations': MAX_TRAIN_ROWS*MAX_CANDIDATES,
            'retained_rule_support_memberships': support_totals,
            'shared_bank_rebuilds_for_other_arms': 0,
            'inference_or_candidate_score_passes_performed_by_factory': 0},
        'claims': {'audited_or_owner_approved': False, 'actual_scientific_evaluation_performed': False,
            'causal_physical_mechanism_established': False, 'blind_generalization_established': False,
            'full_atomistic_or_measured_structure_available': False,
            'sufficient_tuned_ML_comparison_implemented': False,
            'external_source_and_manifest_authentication_required': True}}

def _variable_description(j):
    return {'feature_index': j, 'name': FEATURE_NAMES[j],
        'unit': 'dimensionless occupancy fraction' if j < 118 else GEOMETRY_UNITS[j-118],
        'information_scope': '3DSC_occupancy_and_computed_geometry_proxy_not_measured_complete_structure'}

def finalize_rule_bank(work_product, *, independent_statistics_certificate,
                       statistics_audit_sha256, executor_source_sha256):
    """Metadata-only finalization after ROOT authenticates the actual certificate.

    This function verifies declaration consistency, not the bytes/authenticity of
    an external audit file. It does not read TRAIN again or issue owner approval.
    The trusted exporter must bind the certificate/ref to its independently
    audited work product and only then register these envelopes in the executor.
    """
    cert = independent_statistics_certificate
    exact(cert, ('schema_version', 'checks_passed', 'work_product_sha256', 'partition_id',
                 'source_container_sha256', 'feature_schema_sha256', 'recipe_sha256', 'auditor_source_sha256'),
          'independent statistics certificate')
    sha(statistics_audit_sha256, 'external statistics audit ref'); sha(executor_source_sha256, 'frozen executor')
    sha(cert['auditor_source_sha256'], 'independent auditor source')
    need(cert['schema_version'] == 'TRAIN-rule-statistics-audit/v1' and cert['checks_passed'] is True
         and cert['work_product_sha256'] == digest(work_product), 'Unapproved/modified work product')
    need(work_product['schema_version'] == 'unapproved-TRAIN-rule-work-product/v1'
         and work_product['recipe'] == recipe() and work_product['recipe_sha256'] == digest(recipe()), 'Different factory recipe')
    scope = work_product['scope']; global_stats = work_product['global_statistics']
    for key in ('partition_id', 'source_container_sha256', 'feature_schema_sha256'):
        need(cert[key] == scope[key], 'Certificate scope mismatch: ' + key)
    need(cert['recipe_sha256'] == work_product['recipe_sha256'], 'Certificate recipe mismatch')
    partition = {'partition_id': scope['partition_id'], 'held_fold': scope['held_fold'],
        'source_container_sha256': scope['source_container_sha256'], 'feature_schema_sha256': scope['feature_schema_sha256'],
        'TRAIN_rows': json.loads(canonical(scope['TRAIN_rows'])), 'held_component_ids': list(scope['held_component_ids']),
        'TRAIN_class_ws_mass': list(global_stats['TRAIN_class_ws_mass']),
        'TRAIN_class_row_counts': list(global_stats['TRAIN_class_row_counts']),
        'statistics_audit_sha256': statistics_audit_sha256, 'rule_factory_recipe_sha256': work_product['recipe_sha256']}
    envelopes = []
    # A degenerate pool cannot silently become a numeric arm; all queries fallback.
    rules = work_product['rules'] if work_product['diagnostics']['numeric_rule_use_allowed'] else []
    for r in rules:
        predicate = normalize_predicate(r['predicate'])
        need(digest(predicate) == r['predicate_sha256'], 'Canonical predicate changed after audit')
        stats = {'partition_id': scope['partition_id'], 'predicate_sha256': r['predicate_sha256'],
            'support_source_indices': list(r['support_source_indices']),
            'matched_class_row_counts': list(r['matched_class_row_counts']),
            'matched_class_ws_mass': list(r['matched_class_ws_mass']), 'statistics_audit_sha256': statistics_audit_sha256}
        card = {'rule_id': r['rule_id'], 'version': FACTORY_VERSION,
            'executable_code_sha256': executor_source_sha256,
            'theoretical_basis': {'claim_scope': 'empirical_TRAIN_association_with_assigned_curated_Tc_gt_10K_label',
                'statement': 'A fixed proxy predicate defines a local TRAIN subset; class-balanced sufficient statistics provide a decision score. No chemical causal or universal Tc-optimal-distance claim.',
                'variables': [_variable_description(a['feature_index']) for a in predicate['all']],
                'numeric_formula': '(U1*W/(2*C1)+1)/(U0*W/(2*C0)+U1*W/(2*C1)+2)'},
            'applicability': predicate,
            'known_failures': ['Training association may be spurious or shift outside the TRAIN population.',
                'Nested/correlated rules reuse evidence and are not independent confirmations.',
                'Computed occupancy/geometry summaries omit measured phase, pressure and full atomic structure.',
                'Legacy curated class0 is not automatically a measured nonsuperconducting result.',
                'Sparse or weak evidence can give nearly equal scores; more applicable rules do not prove reasoning utility.'],
            'provenance': {'origin': 'TRAIN', 'partition_id': scope['partition_id'],
                'source_indices': list(r['support_source_indices']), 'component_ids': list(r['component_ids'])},
            'empirical_scope': 'outer_TRAIN', 'fold_training_partition_id': scope['partition_id'],
            'training_group_support': {'n_groups': r['matched_original_groups'], 'n_rows': r['matched_raw_rows']},
            'counterexamples': json.loads(canonical(r['counterexamples'])),
            'missing_input_behavior': 'fallback_original_HGB127',
            'prediction_effect': {'executor_id': EXECUTOR_ID, 'statistics_sha256': digest(stats),
                'smoothing_per_class': 1.0, 'aggregation': AGGREGATION, 'score_semantics': SCORE_SEMANTICS}}
        envelopes.append({'card': card, 'statistics': stats})
    need(len({e['statistics']['predicate_sha256'] for e in envelopes}) == len(envelopes), 'Duplicate predicate in finalized bank')
    return {'schema_version': 'ROOT-reviewable-TRAIN-rule-bank/v1', 'partition': partition,
        'rule_envelopes': envelopes, 'work_product_sha256': digest(work_product),
        'statistics_audit_sha256': statistics_audit_sha256, 'independent_statistics_certificate': json.loads(canonical(cert)),
        'recipe_sha256': work_product['recipe_sha256'], 'diagnostics': dict(work_product['diagnostics']),
        'owner_approval_map_issued': False, 'ROOT_external_audit_authentication_still_required': True,
        'factory_numeric_input_rereads': 0, 'factory_bank_rebuilds': 0,
        'all_arms_must_use_identical_approved_pool_and_evidence': True,
        'strong_tuned_ML_and_same_information_comparison_not_established_by_this_factory': True}
