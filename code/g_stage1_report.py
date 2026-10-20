"""Shared finite G1/G3 classification reporting, never imported old runners."""
from __future__ import annotations

import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path

from common import ROOT, load_data, np
from f3_stats import build_f3_statistics
from f_report import _classification_section

SEED = 20261018
SUBSETS = (('all', '全部'), ('exclude_93', '去掉component 93'), ('only_93', '仅component 93'))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=1)
def cohort_masks():
    """Reuse the label-free, predeclared F3 family/size masks exactly."""
    path = ROOT / 'F3_full3764/config.json'
    cfg = json.loads(path.read_text(encoding='utf-8'))
    masks = {k: np.asarray(v, dtype=bool) for k, v in cfg['family_masks'].items()}
    raw_a15 = np.asarray(cfg['raw_a15_mask'], dtype=bool)
    sizes = np.asarray(cfg['group_size_values'], dtype=int)
    assert all(v.shape == (3764,) for v in masks.values())
    assert np.all(np.sum(np.stack(list(masks.values())), axis=0) == 1)
    assert {k: int(v.sum()) for k, v in masks.items()} == {
        'copper': 712, 'iron': 281, 'hydride': 4, 'A15': 33, 'other': 2734}
    return masks, raw_a15, sizes, {'path': str(path), 'sha256': sha256(path),
        'family_definitions': cfg['metadata']['family_definitions']}


def read_scores(path, methods=None, data=None):
    """Align OOF CSV probabilities by exact material/fold/group identities."""
    d = load_data() if data is None else data
    lookup = {str(v): i for i, v in enumerate(d['id'])}
    wanted = None if methods is None else set(methods)
    result, seen = {}, set()
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            arm = row['arm']
            if wanted is not None and arm not in wanted:
                continue
            identity = row['id']
            if (arm, identity) in seen:
                raise ValueError(f'Duplicate OOF outcome: {arm}/{identity}')
            seen.add((arm, identity))
            i = lookup[identity]
            if int(row['fold']) != int(d['fold'][i]) or str(row['group']) != str(d['group'][i]):
                raise ValueError(f'OOF identity mismatch: {arm}/{identity}')
            a = result.setdefault(arm, np.full(3764, np.nan))
            if row.get('score', '') != '':
                value = float(row['score'])
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f'Invalid probability: {arm}/{identity}')
                if row.get('y_pred', '') != '' and int(row['y_pred']) != int(value >= .5):
                    raise ValueError('Prediction is inconsistent with the frozen p>=0.5 threshold')
                a[i] = value
    if wanted is not None and set(result) != wanted:
        raise ValueError(f'Missing requested methods: {wanted - set(result)}')
    return result


def frozen_baselines(data=None, include_a2=False, include_knn=False):
    d = load_data() if data is None else data
    names = ['HGB127', 'StrongML', 'A3_forced']
    if include_a2:
        names.append('A2_forced')
    if include_knn:
        names.append('5NN')
    return read_scores(ROOT / 'F3_full3764/predictions.csv', names, d)


def classification_bundle(scores, title, *, data=None, n_boot=5000, seed=SEED):
    """Both weights/four metrics/three masks and all paired comparisons."""
    if n_boot != 5000:
        raise ValueError('Publication intervals require the requested 5000 group draws')
    d = load_data() if data is None else data
    families, a15, sizes, provenance = cohort_masks()
    assert np.array_equal(sizes, d['group_size'])
    original = {str(g): int(s) for g, s in zip(d['group'], d['group_size'])}
    output = build_f3_statistics(d['y'], scores, d['group'], original,
        family_masks=families, a15_mask=a15, group_size_values=sizes,
        n_boot=n_boot, seed=seed)
    output['scope'] = 'Adaptive development comparisons on the frozen 3764 materials; not an external test.'
    output['frozen_stratum_provenance'] = provenance
    output['report_markdown'] = render_classification(output, title)
    return output


def render_classification(stats, title):
    methods = stats['method_order']
    pairs = [tuple(p) for p in stats['report_pairs']]
    lines = ['# ' + title, '',
        '已有3764条材料的开发结果。BA/P/R/Acc均按p>=0.5计算；单元为百分比点估计[95% CI]，配对差为百分点。',
        '组权重=1/原始component大小，所有子集保留该权重；不加权每材料=1。两种统计均按组bootstrap 5000次，层内所有方法及配对共享draw。',
        '表内采用所有所列方法共同有效分母；缺失不填成p=0.5或负类。每层保留请求/有效n与组数；单组CI退化、稀疏层未定义次数单列。',
        '原A3输入包含化学式与结构摘要；不能把它称为只有化学式的基线。G1预训练/无序投影及G3训练/标签语义限制见各config。', '']
    collections = [('主比较', stats['subsets'])]
    collections += [('家族·' + k, v) for k, v in stats['families'].items()]
    collections += [('原始组大小·' + k, v) for k, v in stats['group_size_bins'].items()]
    for name, collection in collections:
        for subset, label in SUBSETS:
            table = collection[subset]
            lines += [f"请求n={table['n_requested']} / {table['n_requested_groups']}组；缺失n={table['n_missing']}。", '']
            _classification_section(lines, name + '：' + label, table, methods, pairs)
    return '\n'.join(lines).rstrip() + '\n'


def choose_g1_best(scores, stats):
    """Requested development comparator selection, frozen before G1 outcomes."""
    if stats['subsets']['all']['n'] != 3764:
        raise ValueError('Select full-cohort G1 best only from a complete-methods-only report')
    complete = [name for name, a in scores.items() if np.isfinite(a).all()]
    if not complete:
        raise ValueError('No full-cohort G1 method; partial-coverage selection requires an explicit coverage report')
    estimates = stats['subsets']['all']['group_weighted']['metrics']
    return min(complete, key=lambda name: (-estimates[name]['balanced_accuracy']['point'], name))
