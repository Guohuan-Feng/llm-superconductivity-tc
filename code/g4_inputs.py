"""Label-blind G4 selection and frozen E/F/P scientific payloads; no calls.

The selection function has no measured-label/Tc argument. P changes only the
formula string and its user-authorized display-semantics sentence.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from decimal import Decimal, localcontext
from fractions import Fraction
import csv
import hashlib
import json
import re

from common import ROOT, DATA, load_data, np
from e2_packets import fixed_indices
from e2_prompts import make_prompt, SCHEMA
from f3_inputs import validate_output as original_validate
from g3_inputs import FAMILIES, sha

OUT = ROOT / 'G4_input_ablation'
SCHEMA_PATH = ROOT / 'e2_response.schema.json'
INSTRUCTIONS = ROOT / 'e2_instructions.md'
CATALOG = ROOT / 'e2_model_catalog.json'
SEED = 20261018
ARMS = ('E', 'F', 'P')
SIZE_BINS = ('1', '2-5', '6-20', '>20')
P_SEMANTICS = 'equivalent formula-unit notation; composition occupancies and structure refer to the original proxy cell'
FORMULA_PATTERN = re.compile(r'([A-Z][a-z]?)([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)')


def hashed(value): return hashlib.sha256(value.encode('utf-8')).hexdigest()


def size_bin(n):
    return '1' if n == 1 else '2-5' if n <= 5 else '6-20' if n <= 20 else '>20'


def select_new(ids, full_group_sizes, predicted_families, fixed_ids, *, n_new=500):
    """Minimum-one + Hamilton allocation; fixed deterministic label-blind ranks.

    Inputs are identities, original sizes and previous formula-only predicted
    family labels. Measured outcomes are intentionally unavailable here.
    """
    fixed = set(map(str, fixed_ids))
    strata = {(family, size): [] for family in FAMILIES for size in SIZE_BINS}
    for identity, count, family in zip(ids, full_group_sizes, predicted_families):
        key = str(identity)
        if family not in FAMILIES: raise ValueError('Unfrozen G3 family')
        if key not in fixed: strata[(family, size_bin(int(count)))].append(key)
    active = [key for key, values in strata.items() if values]
    if n_new < len(active) or n_new > sum(len(values) for values in strata.values()):
        raise ValueError('Requested stratified sample is infeasible')
    remaining = n_new - len(active)
    capacity = {key: len(strata[key]) - 1 for key in active}
    denominator = sum(capacity.values())
    quota = {key: 1 + remaining * capacity[key] // denominator for key in active}
    left = n_new - sum(quota.values())
    rank = sorted(active, key=lambda key: (
        -(remaining * capacity[key] % denominator),
        hashed(f'G4|{SEED}|allocation|{key[0]}|{key[1]}')))
    for key in rank:
        if left and quota[key] < len(strata[key]): quota[key] += 1; left -= 1
    assert left == 0 and sum(quota.values()) == n_new
    selected, audit = [], []
    for key, values in strata.items():
        ordered = sorted(values, key=lambda identity: hashed(f'G4|{SEED}|within|{key[0]}|{key[1]}|{identity}'))
        chosen = ordered[:quota.get(key, 0)]
        selected.extend(chosen)
        audit.append({'G3_family': key[0], 'original_group_size_bin': key[1],
            'remaining_population_n': len(values), 'allocated_n': len(chosen),
            'selected_ids': chosen,
            'candidate_ids_hash': hashed(json.dumps(sorted(values), separators=(',', ':'), ensure_ascii=False))})
    assert len(selected) == len(set(selected)) == n_new and not set(selected) & fixed
    return selected, {'seed': SEED, 'n_new': n_new, 'population_excluding_fixed_n': sum(map(len, strata.values())),
        'definition': 'Cartesian G3 formula-only predicted family(12) x full original component-size bin(4). Every nonempty stratum first receives1; remaining places use Hamilton largest-remainder allocation proportional to remaining capacity(c_s-1). Remainder ties and within-stratum selection use fixed SHA256 ranks.',
        'allocation_uses_measured_label_or_Tc': False,
        'selection_function_inputs': ['material IDs', 'original full group sizes', 'frozen G3 predicted family', 'fixed300 IDs'],
        'empty_strata_retained': True, 'strata': audit,
        'selected_id_set_hash': hashed(json.dumps(sorted(selected), separators=(',', ':')))}


def formula_terms(formula):
    matches = list(FORMULA_PATTERN.finditer(formula))
    if not matches or ''.join(m.group(0) for m in matches) != formula:
        raise ValueError('Formula is not the frozen explicit-amount notation: ' + formula)
    terms = [(m.group(1), m.group(2)) for m in matches]
    if len({element for element, amount in terms}) != len(terms):
        raise ValueError('Duplicated element in proxy formula')
    return terms


def rewritten_formula(formula, identity):
    terms = formula_terms(formula)
    elements = [element for element, _ in terms]
    shuffled = sorted(elements, key=lambda element: hashed(f'G4|{SEED}|P_order|{identity}|{element}'))
    if len(elements) > 1 and shuffled == elements: shuffled = shuffled[1:] + shuffled[:1]
    factor = Fraction(1, 2) if int(hashed(f'G4|{SEED}|P_scale|{identity}'), 16) % 2 == 0 else Fraction(2)
    text = dict(terms); transformed = {}
    with localcontext() as context:
        context.prec = 100
        scale = Decimal(factor.numerator) / Decimal(factor.denominator)
        for element, amount in terms:
            # Integer trailing zeroes are significant; trim only fractional zeroes.
            raw = format(Decimal(amount) * scale, 'f')
            transformed[element] = raw.rstrip('0').rstrip('.') if '.' in raw else raw
            assert Fraction(transformed[element]) == Fraction(amount) * factor
    rewritten = ''.join(element + transformed[element] for element in shuffled)
    assert rewritten != formula
    reparsed = dict(formula_terms(rewritten))
    assert set(reparsed) == set(text)
    assert all(Fraction(reparsed[element]) / Fraction(text[element]) == factor for element in text)
    return rewritten, {'factor_numerator': factor.numerator, 'factor_denominator': factor.denominator,
        'original_element_order': elements, 'rewritten_element_order': shuffled,
        'single_element_order_not_applicable': len(elements) == 1,
        'all_element_ratios_exactly_preserved': True}


def payloads(material, identity):
    elements = [element for element, amount in formula_terms(material['formula'])]
    E = {'element_set': elements}
    F = {'formula': material['formula'], 'formula_semantics': material['formula_semantics']}
    P = deepcopy(material)
    P['formula'], audit = rewritten_formula(material['formula'], identity)
    P['formula_semantics'] = P_SEMANTICS
    assert P['composition'] == material['composition'] and P['structure'] == material['structure']
    assert set(P) == set(material)
    assert all(P[key] == material[key] for key in material if key not in ('formula', 'formula_semantics'))
    assert set(E) == {'element_set'} and set(F) == {'formula', 'formula_semantics'}
    return {'E': E, 'F': F, 'P': P}, audit


def prompt(case, arm, repair=False):
    if arm not in ARMS: raise ValueError(arm)
    history = [] if not repair else [{'operational_error': 'Previous attempt failed the decision JSON contract. Return a decision with numeric p, even if abstaining.'}]
    return make_prompt('A3', case['payloads'][arm], [], {}, [], history,
                       round_number=2 if repair else 1, force_decision=False)


def validate(response, case, arm):
    normalized = original_validate(response, {'id': case['id'], 'rule_cards': {}}, 'A3')
    normalized['arm'] = arm
    return normalized


def frozen_inputs(d=None):
    d = load_data() if d is None else d
    f3_path = ROOT / 'F3_full3764/config.json'
    g3_path = ROOT / 'G3_complementarity/config.json'
    f3 = json.loads(f3_path.read_text(encoding='utf-8'))
    g3 = json.loads(g3_path.read_text(encoding='utf-8'))
    assert g3['status'] == 'complete' and len(g3['family_results']) == 3764
    ids = [str(v) for v in d['id']]; lookup = {identity: i for i, identity in enumerate(ids)}
    fixed = fixed_indices(d).tolist(); fixed_ids = [ids[i] for i in fixed]
    labels = [g3['family_results'][identity] for identity in ids]
    new_ids, selection = select_new(ids, d['group_size'].tolist(), labels, fixed_ids)
    new = sorted(lookup[identity] for identity in new_ids)
    selected = fixed + new
    assert len(selected) == len(set(selected)) == 800
    cases = []
    for i in selected:
        identity = ids[i]; source = f3['cases'][identity]
        assert source['id'] == identity and source['fold'] == int(d['fold'][i]) and str(source['group']) == str(d['group'][i])
        packets, transformation = payloads(source['material'], identity)
        c = {'i': i, 'id': identity, 'fold': int(d['fold'][i]), 'group': str(d['group'][i]),
            'cohort': 'fixed300' if i in fixed else 'new500', 'G3_family': labels[i],
            'original_group_size': int(d['group_size'][i]), 'size_bin': size_bin(int(d['group_size'][i])),
            'payloads': packets, 'formula_transformation': transformation}
        c['prompt_hashes'] = {arm: hashed(prompt(c, arm)) for arm in ARMS}
        original_prompt = make_prompt('A3', source['material'], [], {}, [], [], round_number=1, force_decision=False)
        c['original_A3_prompt_hash'] = hashed(original_prompt)
        assert c['original_A3_prompt_hash'] == source['prompt_hashes']['A3']
        source_display = json.loads(original_prompt.split('Scientific inputs:\n', 1)[1])['material']
        P_display = json.loads(prompt(c, 'P').split('Scientific inputs:\n', 1)[1])['material']
        assert P_display['composition'] == source_display['composition']
        assert P_display['structure'] == source_display['structure']
        assert all(P_display[key] == source_display[key] for key in source_display if key not in ('formula', 'formula_semantics'))
        cases.append(c)
    baseline = {}
    selected_id_set = {ids[i] for i in selected}
    with (ROOT / 'F3_full3764/predictions.csv').open(encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            if row['arm'] == 'A3_forced' and row['id'] in selected_id_set:
                i = lookup[row['id']]
                assert int(row['fold']) == int(d['fold'][i]) and row['group'] == str(d['group'][i])
                if row['id'] in baseline: raise ValueError('Duplicate original A3')
                baseline[row['id']] = row
    assert len(baseline) == 800
    sources = [DATA, f3_path, ROOT / 'F3_full3764/predictions.csv', g3_path,
               ROOT / 'G3_complementarity/predictions.csv', ROOT / 'E0/predictions.csv',
               SCHEMA_PATH, INSTRUCTIONS, CATALOG]
    return {'cases': cases, 'selected_indices': selected, 'fixed_indices': fixed, 'new_indices': new,
        'selection': selection, 'baseline_A3_rows': baseline,
        'family_masks': {key: [bool(values[i]) for i in selected] for key, values in f3['family_masks'].items()},
        'raw_a15_mask': [bool(f3['raw_a15_mask'][i]) for i in selected],
        'family_definitions': f3['metadata']['family_definitions'],
        'G3_family_vocabulary': list(FAMILIES),
        'scientific_sources': {str(path): sha(path) for path in sources},
        'P_user_authorization': 'User explicitly permitted correcting formula_semantics together with equivalent formula-string reordering/rescaling. All composition/structure numeric values remain unchanged; no additional note field.',
        'P_identity_boundary': 'Original composition table and structure remain supplied and can reconstruct the original cell formula. P tests representation stability; it is not identity masking.',
        'sampling_interpretation': 'Fixed300 plus label-blind stratified new500 development sample; minimum-one stratification is not a uniform full-cohort estimate and no inverse-inclusion correction is applied.'}
