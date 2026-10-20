"""Finite E4 input preparation and statistics; no model calls or file writes.

Rule use means an executed R-info selection or an A2 reported selected_rule_id.
An A2 direct prediction may report zero rules; its probability is never silently
replaced by a rule mean.  Rule identities are scoped to their outer TRAIN fold.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math

from common import ROOT, load_data, np
from e2_metrics import mean_summary
from stats import summarize


ANNOTATION_STANDARD = {
    "physical_meaning": (
        "The literal condition is a recognizable materials-physics descriptor "
        "or a specific, qualified mechanism proxy (e.g. Cu and O chemistry, "
        "light-element/phonon proxy). Explain the mechanism and what the "
        "condition cannot establish. Composition alone does not establish "
        "Cu-O layers, pairing mechanism, pressure, phase, or measured Tc."
    ),
    "proxy_only": (
        "A population/dataset association without a specific supported "
        "mechanistic interpretation: rare-element absence, generic occupancy "
        "or coordinate/geometry thresholds, or cell-size summaries. Generic "
        "statistical card text and predictive strength alone are insufficient "
        "to confer physical meaning. When evidence is insufficient choose "
        "proxy_only and state that limitation."
    ),
    "scope": (
        "Binary interpretation of this exact predicate, not a causal or "
        "universal superconductivity claim. Local Codex supplies provisional "
        "labels, rationale and confidence; every row remains human-review-pending."
    ),
}

_ELEMENTS = ('H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn '
             'Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La '
             'Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po '
             'At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og').split()


def _semantic_condition(condition):
    """Decode elemental Z names without changing the underlying predicate."""
    result = dict(condition)
    feature = condition['feature']
    if feature.startswith('occupancy_fraction_Z'):
        number = int(feature.split('_Z')[1])
        result['element'] = _ELEMENTS[number - 1]
        result['atomic_number'] = number
        descriptor = result['element'] + ' occupancy-weighted atomic fraction'
    else:
        descriptor = feature.replace('_', ' ')
    result['human_predicate'] = f"{descriptor} {condition['operator']} {condition['threshold']:.10g} [{condition['unit']}]"
    return result


def family(case):
    """User's ordered, fixed family definitions; use supplied atomic fractions."""
    amounts = {entry['element']: float(entry['atomic_fraction'])
               for entry in case['material']['composition']}
    if 'Cu' in amounts and 'O' in amounts:
        return 'cuprate'
    if 'Fe' in amounts and any(symbol in amounts for symbol in ('As', 'P', 'Se', 'Te')):
        return 'iron_based'
    if amounts.get('H', 0.0) >= .5:
        return 'hydride'
    return 'other'


def load_e2_results(path=None):
    """Only read persisted completed decisions; never import the E2 runner."""
    results = {}
    path = ROOT / 'E2/llm_calls.jsonl' if path is None else path
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            final = record.get('case_result')
            if not final or final.get('status') != 'ok':
                continue
            key = (record['arm'], record['id'])
            if (final['arm'], final['id']) != key:
                raise ValueError('E2 call/result identity mismatch')
            if key in results and results[key] != final:
                raise ValueError('Conflicting persisted E2 final decisions')
            results[key] = final
    return results


def _csv_scores(path, data, arm):
    lookup = {str(identity): i for i, identity in enumerate(data['id'])}
    values = np.full(len(lookup), np.nan)
    with path.open(newline='', encoding='utf-8') as handle:
        for record in csv.DictReader(handle):
            if record['arm'] != arm:
                continue
            i = lookup[record['id']]
            if int(record['fold']) != int(data['fold'][i]) or record['group'] != str(data['group'][i]):
                raise ValueError('Prediction source identity mismatch')
            if np.isfinite(values[i]):
                raise ValueError('Duplicate prediction source arm/id')
            values[i] = float(record['score'])
    return values


def _fingerprint(identity):
    return hashlib.sha256(('E4-example|' + identity).encode()).hexdigest()


def prepare_e4(data=None, *, config=None, results=None, n_boot=5000, seed=20261016):
    """Return JSON-safe stats, 30 unlabeled rule packets, and five exact chains.

    Callers persist this content inside the six permitted experiment files.
    No scientific annotation is inferred by this module.
    """
    data = load_data() if data is None else data
    config = json.loads((ROOT / 'E2/config.json').read_text(encoding='utf-8')) if config is None else config
    results = load_e2_results() if results is None else results
    e1 = json.loads((ROOT / 'E1/metrics.json').read_text(encoding='utf-8'))
    cases = config['cases']
    indices = np.asarray(config['indices'], dtype=int)
    if len(indices) != 300 or len(set(indices.tolist())) != 300:
        raise ValueError('E4 common cohort must be the original E2 fixed300')
    ids = [str(data['id'][i]) for i in indices]
    if set(ids) != set(cases):
        raise ValueError('E2 packet/indices differ')
    all_selected = e1['diagnostics']['selected_rules']['R-info']
    if len(all_selected) != len(data['y']):
        raise ValueError('Incomplete E1 R-info executed selections')
    selections = {
        'R-info': [all_selected[int(i)] for i in indices],
        'A2': [results[('A2', identity)]['selected_rules'] for identity in ids],
    }
    abstain = np.asarray([results[('A2', identity)]['abstain'] for identity in ids], dtype=bool)
    group = data['group'][indices]
    sizes = data['group_size'][indices]
    counts = {arm: np.asarray([len(chosen) for chosen in selected], dtype=float)
              for arm, selected in selections.items()}
    descriptions = {
        'scope': 'development only; fixed original E2 300 cases for paired comparisons',
        'n': int(len(indices)), 'n_groups': int(len(set(group.tolist()))),
        'n_boot': n_boot, 'seed': seed,
        'R_info_definition': 'Executed deterministic top-three applicable TRAIN rule selection; not a model self-report.',
        'A2_definition': 'Number of distinct rule IDs in persisted A2 selected_rules; self-reported use, not a causal faithfulness test.',
        'zero_rule_definition': 'A2 direct probability with selected_rules=[] is counted as zero, including forced abstention backup.',
        'fallback_definition': 'A2 abstention sends execution to HGB; its rule-count backup does not describe a rule-executed fallback judgment.',
        'means': {}, 'paired_count_differences': {}, 'count_distribution': {},
    }
    for arm, values in counts.items():
        descriptions['means'][arm + '_forced_all300'] = mean_summary(values, group, sizes, n_boot=n_boot, seed=seed)
        descriptions['means'][arm + '_A2_nonabstaining_subset'] = mean_summary(
            values[~abstain], group[~abstain], sizes[~abstain], n_boot=n_boot, seed=seed)
        descriptions['count_distribution'][arm] = {
            str(k): {'count': int((values == k).sum()), 'fraction': mean_summary(
                (values == k).astype(float), group, sizes, n_boot=n_boot, seed=seed)}
            for k in (0, 1, 2, 3)
        }
    descriptions['means']['A2_abstaining_forced_backup'] = mean_summary(
        counts['A2'][abstain], group[abstain], sizes[abstain], n_boot=n_boot, seed=seed)
    descriptions['means']['R_info_full3764_descriptive'] = mean_summary(
        [len(value) for value in all_selected], data['group'], data['group_size'], n_boot=n_boot, seed=seed)
    descriptions['paired_count_differences']['A2_minus_R_info_forced_all300'] = mean_summary(
        counts['A2'] - counts['R-info'], group, sizes, n_boot=n_boot, seed=seed)
    descriptions['paired_count_differences']['A2_minus_R_info_nonabstaining'] = mean_summary(
        counts['A2'][~abstain] - counts['R-info'][~abstain], group[~abstain], sizes[~abstain], n_boot=n_boot, seed=seed)

    # Count citations on the common cohort; folds define distinct rule objects.
    usage = {arm: Counter() for arm in selections}
    users = {arm: defaultdict(list) for arm in selections}
    cards = {}
    for position, identity in enumerate(ids):
        case = cases[identity]
        if case['i'] != int(indices[position]) or case['group'] != str(group[position]):
            raise ValueError('E2 frozen case identity mismatch')
        for arm in selections:
            chosen = selections[arm][position]
            if len(chosen) != len(set(chosen)):
                raise ValueError('Duplicate selected rule within one decision')
            for rule_id in chosen:
                if rule_id not in case['rule_cards']:
                    raise ValueError('Selected R-info/A2 rule absent from E2 frozen top-six cards')
                key = (int(case['fold']), rule_id)
                usage[arm][key] += 1
                users[arm][key].append(position)
                card = {k: v for k, v in case['rule_cards'][rule_id].items()
                        if k not in ('counterexamples', 'conditions')}
                card['conditions'] = [_semantic_condition({k: v for k, v in condition.items() if k != 'target_value'})
                                      for condition in case['rule_cards'][rule_id]['conditions']]
                if key in cards and cards[key] != card:
                    raise ValueError('Fold-scoped frozen rule card changes between targets')
                cards[key] = card
    combined = usage['R-info'] + usage['A2']
    rank_key = lambda key: (-combined[key], key[0], key[1])
    ordered = sorted(combined, key=rank_key)
    packets = []
    for rank, key in enumerate(ordered[:30], start=1):
        fold, rule_id = key
        packet = {
            'marker': f'R{rank:02d}', 'rank': rank, 'rule_key': f'fold{fold}:{rule_id}', 'fold': fold,
            'rule_id': rule_id, 'combined_citations': combined[key],
            'TRAIN_n': int((data['fold'] != fold).sum()),
            'TRAIN_n_groups': int(len(set(data['group'][data['fold'] != fold].tolist()))),
            'card': cards[key], 'usage': {},
            'annotation': {'label': None, 'rationale': None, 'confidence': None, 'human_review': 'pending'},
        }
        for arm in usage:
            selected_positions = users[arm][key]
            indicator = np.zeros(len(indices), dtype=float)
            indicator[selected_positions] = 1
            packet['usage'][arm] = {
                'n_decisions_using_rule': len(selected_positions),
                'n_groups_using_rule': len(set(group[selected_positions].tolist())),
                'cohort_n': len(indices), 'cohort_groups': len(set(group.tolist())),
                'citation_fraction': mean_summary(indicator, group, sizes, n_boot=n_boot, seed=seed),
            }
        packets.append(packet)
    ranked_by_arm = {
        arm: [{'fold': fold, 'rule_id': rule_id, 'citations': usage[arm][(fold, rule_id)]}
              for fold, rule_id in sorted(usage[arm], key=lambda key: (-usage[arm][key], key[0], key[1]))[:30]]
        for arm in usage
    }
    ranking = {
        'pool': 'Union of R-info and A2 reported/executed rule citations on the SAME fixed300; abstaining forced backups included.',
        'sort': 'Combined raw material citation count descending, then fold ascending, then rule_id ascending.',
        'fold_scope': 'Same textual rule_id in different outer TRAIN folds is a distinct card.',
        'n_unique_rules': len(ordered),
        'n_unique_by_arm': {arm: len(value) for arm, value in usage.items()},
        'total_citations_by_arm': {arm: int(sum(value.values())) for arm, value in usage.items()},
        'top30_total_citations': int(sum(combined[key] for key in ordered[:30])),
        'total_citations': int(sum(combined.values())),
        'alternative_top30_by_arm': ranked_by_arm,
        'annotation_standard': ANNOTATION_STANDARD,
        'rule_annotation_packets': packets,
    }

    rinfo_scores = _csv_scores(ROOT / 'E1/predictions.csv', data, 'R-info')
    tree3 = _csv_scores(ROOT / 'E1/predictions.csv', data, 'Tree3')
    tree4 = _csv_scores(ROOT / 'E1/predictions.csv', data, 'Tree4')
    a2_forced = np.asarray([results[('A2', identity)]['p'] for identity in ids])
    a2_fallback = np.where(abstain, data['hgb'][indices], a2_forced)
    full_sizes = {str(g): int(s) for g, s in zip(data['group'], data['group_size'])}
    comparison = summarize(data['y'][indices], {
        'R-info': rinfo_scores[indices], 'A2_forced': a2_forced,
        'A2_HGB_fallback': a2_fallback, 'Tree3': tree3[indices],
        'Tree4': tree4[indices], 'HGB127': data['hgb'][indices],
    }, group, full_sizes, n_boot=n_boot, seed=seed)

    # Five complete paired examples require at least one A2-reported rule each.
    # Show zero-ID behavior in one separate supplement, never as a rule chain.
    # Do not select by correctness or label.
    chosen_ids = []
    available_family = Counter(family(case) for case in cases.values())
    for name in ('cuprate', 'iron_based', 'hydride', 'other'):
        pool = [identity for identity in ids if family(cases[identity]) == name
                and results[('A2', identity)]['selected_rules']]
        if pool:
            chosen_ids.append(min(pool, key=_fingerprint))
    remaining = sorted((identity for identity in ids if identity not in chosen_ids
                        and results[('A2', identity)]['selected_rules']), key=_fingerprint)
    chosen_ids.extend(remaining[:5 - len(chosen_ids)])
    if len(chosen_ids) != 5:
        raise ValueError('Fewer than five A2 nonzero-rule examples are available')
    zero_pool = [identity for identity in ids if not results[('A2', identity)]['selected_rules']]
    zero_identity = min(zero_pool, key=_fingerprint) if zero_pool else None
    example_ids = chosen_ids + ([zero_identity] if zero_identity is not None else [])
    chains = []
    for identity in example_ids:
        case = cases[identity]
        i = int(case['i'])
        result = results[('A2', identity)]
        rinfo_rules = all_selected[i]

        def exact_rules(rule_ids):
            records = []
            for rule_id in rule_ids:
                card = case['rule_cards'][rule_id]
                conditions = []
                for condition in card['conditions']:
                    value = condition['target_value']
                    satisfied = value <= condition['threshold'] if condition['operator'] == '<=' else value > condition['threshold']
                    if not satisfied:
                        raise ValueError('An explanation references a nonapplicable rule')
                    conditions.append({**condition, 'satisfied': bool(satisfied)})
                records.append({'rule_key': f"fold{case['fold']}:{rule_id}", 'rule_id': rule_id,
                                'conditions': conditions, 'score': card['class_balanced_score'],
                                'TRAIN_support_groups': card['TRAIN_support_groups'],
                                'TRAIN_coverage_rows': card['TRAIN_coverage_rows']})
            return records

        rmean = math.fsum(case['rule_cards'][rule]['class_balanced_score'] for rule in rinfo_rules) / len(rinfo_rules)
        if rmean != float(rinfo_scores[i]):
            raise ValueError('R-info chain is inconsistent with persisted prediction')
        final_score = float(data['hgb'][i]) if result['abstain'] else result['p']
        chains.append({
            'id': identity, 'fold': int(case['fold']), 'group': case['group'], 'family': family(case),
            'material': case['material'], 'true_Tc_K': float(data['tc'][i]), 'y_true': int(data['y'][i]),
            'R-info': {'executed_rules': exact_rules(rinfo_rules), 'equal_weight_score': rmean,
                       'threshold': .5, 'y_pred': int(rmean >= .5),
                       'correct': bool((rmean >= .5) == data['y'][i])},
            'A2': {'reported_selected_rules': exact_rules(result['selected_rules']),
                   'reported_p': float(result['p']), 'threshold': .5,
                   'forced_y_pred': int(result['p'] >= .5), 'abstain': bool(result['abstain']),
                   'fallback_score': final_score, 'fallback_y_pred': int(final_score >= .5),
                   'reason_verbatim': result['reason'],
                   'forced_correct': bool((result['p'] >= .5) == data['y'][i]),
                   'interpretation': ('No rule IDs reported: a direct chemical/neighbor prediction; no rule chain can be reconstructed.'
                                      if not result['selected_rules'] else
                                      'Reported rule references plus direct probability; not an equal-weight rule executor and not proof of a faithful causal explanation.')},
            'five_nearest_TRAIN_materials': case['neighbours'],
        })
    zero_example = chains.pop() if zero_identity is not None else None
    if zero_example is not None:
        zero_example['example_role'] = 'Supplementary zero-rule A2 prediction; explicitly not an A2 rule chain.'
    return {
        'rule_counts': descriptions, 'top30_rules': ranking,
        'raw_counts': {'row_ids': ids, 'row_indices': indices.tolist(),
                       'R-info': counts['R-info'].astype(int).tolist(),
                       'A2': counts['A2'].astype(int).tolist(),
                       'A2_abstain': abstain.tolist()},
        'accuracy_comparison_fixed300': comparison,
        'examples': {'selection': 'SHA256(E4-example|id), one nonzero-rule A2 case per available fixed family, then nonzero-rule hash-fill to five; separate minimum-hash zero-rule supplement; no labels used.',
                     'family_counts_in_fixed300': dict(available_family),
                     'n': len(chains), 'n_groups': len(set(c['group'] for c in chains)), 'chains': chains,
                     'zero_rule_example': zero_example,
                     'supplementary_zero_rule_n': int(zero_example is not None),
                     'total_examples_including_supplement': len(example_ids)},
    }


def build_e4_inputs(data=None, *, config=None, results=None, n_boot=5000, seed=20261016):
    """Explicit orchestration entry point; alias retaining the complete return."""
    return prepare_e4(data, config=config, results=results, n_boot=n_boot, seed=seed)
