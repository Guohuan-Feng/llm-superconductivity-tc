"""G3 frozen formula-only family task. Importing performs no writes/calls."""
from common import ROOT, DATA, HGB, load_data, np
from pathlib import Path
import csv
import hashlib
import json

OUT = ROOT / 'G3_complementarity'
FAMILIES = ('铜基', '铁基', 'A15', 'MgB₂ 类', '重费米子', 'Chevrel', '硼碳化物', '富勒烯', '镍基', '单质或合金', '其他', '不是超导体')
SCHEMA = ROOT / 'g3_family.schema.json'
INSTRUCTIONS = ROOT / 'g3_instructions.md'
CATALOG = ROOT / 'e2_model_catalog.json'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()

def prompt(formula, repair=None):
    text = ('Choose the single most suitable superconducting-material family for the supplied proxy-cell formula. '
            'The formula is the only material information supplied. Use your existing chemical knowledge. '
            'Do not estimate Tc or a superconductivity probability. Select exactly one of these fixed labels:\n'
            + json.dumps(list(FAMILIES), ensure_ascii=False) + '\nproxy cell formula: '
            + json.dumps(formula, ensure_ascii=False) + '\nReturn only {"family": "one exact label"}.')
    if repair is not None:
        text += '\nThe preceding attempt did not satisfy the JSON contract. Return one allowed family string in the required JSON object using exactly the same formula.'
    return text

def validate(response):
    if not isinstance(response, dict) or set(response) != {'family'} or response['family'] not in FAMILIES:
        raise ValueError('Expected only family with an exact fixed-list value')
    return {'family': response['family']}

def frozen_inputs(d=None):
    d = load_data() if d is None else d
    f3path = ROOT / 'F3_full3764/config.json'
    f3 = json.loads(f3path.read_text(encoding='utf8'))
    cases = []
    for i, key in enumerate(f3['case_order']):
        case = f3['cases'][key]
        assert str(d['id'][i]) == key and int(d['fold'][i]) == case['fold'] and str(d['group'][i]) == case['group']
        cases.append({'i': i, 'id': key, 'formula': case['material']['formula'],
                      'prompt_hash': hashlib.sha256(prompt(case['material']['formula']).encode('utf8')).hexdigest()})
    source_scores = {'A2': np.full(3764, np.nan), 'A3': np.full(3764, np.nan), 'HGB127': d['hgb'].copy(), 'StrongML': np.full(3764, np.nan)}
    ids = {str(v): i for i, v in enumerate(d['id'])}
    with (ROOT / 'F3_full3764/predictions.csv').open(newline='', encoding='utf8') as handle:
        for row in csv.DictReader(handle):
            arm = row['arm'].replace('_forced', '')
            if arm in ('A2', 'A3'):
                source_scores[arm][ids[row['id']]] = float(row['score'])
    with (ROOT / 'E0/predictions.csv').open(newline='', encoding='utf8') as handle:
        for row in csv.DictReader(handle):
            if row['arm'] == 'StrongML': source_scores['StrongML'][ids[row['id']]] = float(row['score'])
    assert all(np.isfinite(v).all() for v in source_scores.values())
    sources = [DATA, HGB, f3path, ROOT / 'F3_full3764/predictions.csv', ROOT / 'F3_full3764/metrics.json',
               ROOT / 'E0/config.json', ROOT / 'E0/predictions.csv', ROOT / 'E0/metrics.json',
               ROOT.parent / '2026-10-10-structure-nonlinear-classification/nonlinear_classification.py',
               ROOT.parent / '2026-10-10-structure-nonlinear-classification/run/protocol.json']
    return {'cases': cases, 'scores': {k: v.tolist() for k, v in source_scores.items()},
            'family_masks': f3['family_masks'], 'raw_a15_mask': f3['raw_a15_mask'],
            'group_size_values': d['group_size'].tolist(), 'scientific_sources': {str(p): sha(p) for p in sources},
            'old_A3_input': 'Original A3 used proxy formula plus structural summary; not formula only.'}
