"""Small input/file helpers for the finite stage-1 diagnosis."""
import os
import sys
from pathlib import Path

ROOT = Path('C:/Users/28908/Codex/fung-research/v2-diagnosis').resolve()
BASE = ROOT.parent
SITE = Path('C:/Users/28908/Documents/Codex/2026-10-05/tc-performance-push/env/Lib/site-packages').resolve()
assert not any('onedrive' in p.lower() for p in [str(ROOT), str(SITE)])
sys.dont_write_bytecode = True
sys.path.insert(0, str(SITE))
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')

import csv
import json
import numpy as np

DATA = BASE / '2026-10-09-structure-classification/inputs/accepted_train.npz'
HGB = BASE / '2026-10-10-structure-nonlinear-classification/run/oof_predictions.npz'
COMMON = BASE / 'workflow-review-2026-10-10/common-facts-cohort-actual-001/COMMON_FACTS_FEATURES_ONLY.npz'
BANKS = BASE / 'workflow-review-2026-10-10/actual-train-rulebank-run-001'
DIAGNOSTIC = BASE / 'workflow-review-2026-10-15/local-codex-final-case-001/ROOT_CORRECTED_LABEL_FREE_WHOLE300_MECHANISM_DIAGNOSTIC_002.json'
PROGRAMME = BASE / 'workflow-review-2026-10-15/strong-ml-priority-campaign-005/FROZEN_PRIORITY_EXECUTION_PROGRAMME.json'

def safe_path(p):
    p = Path(p).resolve()
    if 'onedrive' in str(p).lower() or not p.is_relative_to(ROOT):
        raise ValueError(f'Write destination outside authorized root: {p}')
    return p

def write_json(path, obj):
    with safe_path(path).open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def load_data():
    with np.load(DATA, allow_pickle=False) as z:
        raw = {k: z[k] for k in z.files}
    d = dict(X127=raw['X127'], y=raw['label'], fold=raw['fold'],
             group=raw['component_id'].astype(str), id=raw['material_id'],
             source_index=raw['source_index'], tc=raw['target_K'])
    assert np.array_equal(d['y'], d['tc'] > 10)
    assert len(d['y']) == 3764
    groups, inverse, sizes = np.unique(d['group'], return_inverse=True, return_counts=True)
    assert len(groups) == 1117
    d['group_size'] = sizes[inverse]
    for group in groups:
        assert len(np.unique(d['fold'][d['group'] == group])) == 1
    for path, key, out in [(HGB, 'probability', 'hgb'), (COMMON, 'X969', 'X969')]:
        with np.load(path, allow_pickle=False) as z:
            for k in ('source_index', 'material_id', 'component_id', 'fold'):
                assert np.array_equal(z[k], raw[k]), (path, k)
            if 'label' in z.files:
                assert np.array_equal(z['label'], d['y'])
            d[out] = z[key]
    assert np.all(np.isfinite(d['X127'])) and np.all(np.isfinite(d['X969']))
    return d

FIELDS = ['id', 'fold', 'group', 'arm', 'status', 'score', 'y_pred']

class PredictionWriter:
    """Append ten completed cases at a time; skip existing arm/id pairs."""
    def __init__(self, path, d):
        self.path, self.d = safe_path(path), d
        self.done = set()
        if self.path.exists():
            with self.path.open(newline='', encoding='utf-8') as f:
                self.done = {(r['arm'], r['id']) for r in csv.DictReader(f)}
        else:
            with self.path.open('w', newline='', encoding='utf-8') as f:
                csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def append(self, arm, indices, scores, statuses):
        pending = []
        for i, score, status in zip(indices, scores, statuses):
            key = (arm, str(self.d['id'][i]))
            if key in self.done:
                continue
            pending.append(dict(id=key[1], fold=int(self.d['fold'][i]),
                group=str(self.d['group'][i]), arm=arm, status=str(status),
                score=float(score), y_pred=int(score >= .5)))
            self.done.add(key)
            if len(pending) == 10:
                self._flush(pending)
                pending = []
        if pending:
            self._flush(pending)

    def _flush(self, rows):
        with self.path.open('a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerows(rows)
            f.flush()
            os.fsync(f.fileno())
