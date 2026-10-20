"""Root-only TRAIN preparation; no model fits, no old validation labels."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REVIEW = ROOT.parent / '2026-10-09-superconductivity-structure-review'
SOURCE = Path('C:/Users/28908/OneDrive/文档/ChatGPT/fung教授科研/work')


def checked_output(path):
    p = Path(path).absolute()
    for q in (p, *p.parents):
        assert 'onedrive' not in str(q).lower(), str(q)
        if q.exists():
            assert 'onedrive' not in str(q.resolve(strict=True)).lower(), str(q)
    return p


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    p = checked_output(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with checked_output(p).open('x', encoding='utf-8', newline='\n') as h:
        json.dump(value, h, ensure_ascii=False, indent=2, allow_nan=False)
        h.write('\n')
        h.flush()
        os.fsync(h.fileno())


def main():
    assert sha(ROOT / 'SCIENTIFIC_PLAN.json') == '90721e9ce7252c7c35c2cb59886810ad097a4fb37b3ac446a9d7362381915106'
    plan = json.loads((ROOT / 'SCIENTIFIC_PLAN.json').read_text(encoding='utf-8'))
    assert plan['primary_tau_K'] == 10.0
    dest = checked_output(ROOT / 'inputs/accepted_train.npz')
    assert not dest.exists() and not (ROOT / 'PREFLIGHT.json').exists()
    pins = {
        SOURCE / 'scientific_tc_2026-10-02/data/train.csv.gz': plan['input_pins']['train_csv_gz'],
        SOURCE / 'scientific_tc_2026-10-02/data/fold_assignments.csv': plan['input_pins']['fold_assignments'],
        SOURCE / 'scientific_tc_2026-10-01/data/split_assignments.csv': plan['input_pins']['split_assignments'],
        REVIEW / 'TRAIN_ONLY_STRUCTURE_DESCRIPTORS.jsonl.gz': plan['input_pins']['TRAIN_only_coordinate_projection'],
        REVIEW / 'TRAIN_ONLY_STRUCTURE_IDENTITY.json': plan['input_pins']['TRAIN_only_identity'],
        REVIEW / 'TRAIN_ONLY_GEOMETRY_REVIEW.json': plan['input_pins']['geometry_review'],
    }
    for p, digest in pins.items():
        assert sha(p) == digest, str(p)
    feature_receipt_path = ROOT / 'FEATURE_EXTRACTION.json'
    feature_receipt = json.loads(feature_receipt_path.read_text(encoding='utf-8'))
    assert feature_receipt['checks_passed'] and feature_receipt.get('actual_target_rows_read', 0) == 0
    feature_path = ROOT / 'inputs/FEATURES_ONLY.npz'
    assert sha(feature_path) == feature_receipt['features_npz_sha256']
    feature_code_sha = feature_receipt['feature_code_sha256']
    for p, digest in feature_code_sha.items():
        assert sha(p) == digest, p
    identity = json.loads((REVIEW / 'TRAIN_ONLY_STRUCTURE_IDENTITY.json').read_text(encoding='utf-8'))['records']
    assert len(identity) == 3764 and len({r['material_id'] for r in identity}) == 3764
    fold_frame = pd.read_csv(SOURCE / 'scientific_tc_2026-10-02/data/fold_assignments.csv', dtype={'material_id': str, 'group': str})
    split_frame = pd.read_csv(SOURCE / 'scientific_tc_2026-10-01/data/split_assignments.csv', dtype={'material_id': str, 'group': str})
    split_map = split_frame.set_index('material_id')
    for r in identity:
        s = split_map.loc[r['material_id']]
        assert s['split'] == r['split'] == 'train'
        assert s['role'] == r['role'] == 'inner_train'
        assert str(s['group']) == str(r['group'])
    with np.load(feature_path, allow_pickle=False) as z:
        assert set(z.files) == {'X127', 'material_id', 'feature_names'}
        feat = {k: z[k] for k in z.files}
    assert feat['X127'].shape == (3764, 127) and np.isfinite(feat['X127']).all()
    assert len(feat['feature_names']) == 127 and len(set(feat['feature_names'].tolist())) == 127
    assert len(set(feat['material_id'].tolist())) == 3764
    assert set(feat['material_id'].tolist()) == {r['material_id'] for r in identity}
    index = {str(mid): i for i, mid in enumerate(feat['material_id'])}
    # This is the sole old TRAIN target-table decode in this preparation.
    cols = ['material_id', 'row_id', 'split', 'tc', 'group', 'chemical_system', 'parent_id', 'canonical_composition']
    frame = pd.read_csv(SOURCE / 'scientific_tc_2026-10-02/data/train.csv.gz', usecols=cols,
                        dtype={'material_id': str, 'group': str}, float_precision='round_trip')
    assert len(frame) == 3764 and frame.material_id.is_unique and set(frame.split) == {'train'}
    assert frame.material_id.tolist() == fold_frame.material_id.tolist()
    assert frame.group.tolist() == fold_frame.group.tolist()
    assert set(frame.material_id) == set(index)
    identity_map = {r['material_id']: r for r in identity}
    for _, r in frame.iterrows():
        known = identity_map[r.material_id]
        assert int(r.row_id) == known['row_id'] and r.parent_id == known['parent_id']
        assert str(r.group) == known['group']
    groups = frame.group.to_numpy(str)
    folds = fold_frame.fold.to_numpy(np.int64)
    assert set(folds.tolist()) == {0, 1, 2}
    assert fold_frame.groupby('group').fold.nunique().max() == 1
    overlaps = {}
    for key in ['chemical_system', 'parent_id', 'canonical_composition']:
        assert frame.groupby(key, dropna=False).group.nunique().max() == 1
        overlaps[key] = []
        for f in range(3):
            n = len(set(frame.loc[folds != f, key]) & set(frame.loc[folds == f, key]))
            assert n == 0
            overlaps[key].append(n)
    target = frame.tc.to_numpy(np.float64)
    assert np.isfinite(target).all() and (target >= 0).all()
    label = (target > plan['primary_tau_K']).astype(np.int64)
    support = []
    for f in range(3):
        train, held = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        train_counts = np.bincount(label[train], minlength=2).tolist()
        held_counts = np.bincount(label[held], minlength=2).tolist()
        assert min(train_counts + held_counts) > 0
        assert not (set(groups[train]) & set(groups[held]))
        support.append(dict(fold=f, train_records=len(train), held_records=len(held),
                            train_components=len(set(groups[train])), held_components=len(set(groups[held])),
                            train_class_counts=train_counts, held_class_counts=held_counts))
    X = feat['X127'][[index[mid] for mid in frame.material_id]].astype(np.float64)
    assert np.allclose(X[:, :118].sum(axis=1), 1.0, rtol=0, atol=2e-12)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with checked_output(dest).open('xb') as h:
        np.savez_compressed(h, X127=X, target_K=target, label=label, component_id=groups, fold=folds,
                            material_id=frame.material_id.to_numpy(str), source_index=np.arange(3764, dtype=np.int64),
                            role=np.full(3764, 'TRAIN'))
        h.flush()
        os.fsync(h.fileno())
    receipt = dict(checks_passed=True, errors=[], created_utc=datetime.now(timezone.utc).isoformat(),
                   records=3764, components=len(set(groups)), feature_count=127,
                   branch_feature_counts=[118, 127], primary_tau_K=10.0, probability_cutoff=0.5,
                   scientific_plan_sha256=sha(ROOT / 'SCIENTIFIC_PLAN.json'),
                   input_npz=dict(path=str(dest), sha256=sha(dest)), dataset_pins={str(k): v for k, v in pins.items()},
                   feature_code_sha256=feature_code_sha, feature_receipt_sha256=sha(feature_receipt_path),
                   feature_npz_sha256=sha(feature_path), preparation_code_sha256={str(Path(__file__).resolve()): sha(__file__)},
                   identity_material_id_and_row_id_and_parent_and_group_checked=True,
                   all_train_roles_authenticated=True, original_groups_folds_retained=True,
                   linked_identity_train_held_overlap=overlaps,
                   fold_records=[int((folds == f).sum()) for f in range(3)],
                   fold_components=[len(set(groups[folds == f])) for f in range(3)],
                   support=support, all_train_held_have_both_classes=True,
                   class_counts=np.bincount(label, minlength=2).tolist(), equals_primary_tau_records=int((target == 10).sum()),
                   target_semantics=plan['target_semantics'], label_definition='int(original TRAIN Tc_K >10)',
                   target_rows_read=3764, target_table_decodes=1, original_training_only=True,
                   old_validation_target_reads=0, old_retrospective_target_reads=0, model_fits=0,
                   model_queries=0, new_regression_fits=0, OneDrive_writes=0)
    save(ROOT / 'PREFLIGHT.json', receipt)
    print(json.dumps({k: receipt[k] for k in ['checks_passed', 'records', 'components', 'class_counts', 'fold_records', 'support', 'input_npz']}))


if __name__ == '__main__':
    main()
