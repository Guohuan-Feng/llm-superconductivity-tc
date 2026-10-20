"""Finite G1: official scratch CGCNN, frozen CHGNet embeddings, and Magpie RF.

No LLM calls. Three original outer folds; no outer labels select parameters.
"""
from __future__ import annotations
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from g1_data import (ROOT, CACHE, OUT, DATA, FACTS, SEED, CGCNN_COMMIT, safe,
    environment, prevent_sleep, digest, write_json, load_data, load_structures,
    prepare_official_cgcnn, cgcnn_class, structure, magpie, cgcnn_graphs, frozen_splits)

FIELDS = ['id', 'fold', 'group', 'arm', 'status', 'score', 'y_pred']
MODELS = ['CGCNN_scratch', 'CHGNet_dominant_HGB', 'CHGNet_dominant_LR', 'Magpie_RF']
PARAMS = {
 'CGCNN_scratch': {'atom_fea_len': 64, 'n_conv': 3, 'h_fea_len': 128, 'n_h': 1,
     'classification': True, 'batch_size': 32, 'optimizer': 'Adam', 'lr': 0.001,
     'weight_decay': 0.00001, 'epochs_max': 200, 'patience': 25,
     'early_stopping': 'maximum inner-validation original-group-weighted BA; first epoch wins ties',
     'refit': 'fresh initialization on all outer TRAIN for selected inner epoch count',
     'cumulative_training_wall_limit_seconds': 10800, 'radius_A': 8.0,
     'max_neighbors': 12, 'gaussian_step_A': 0.2},
 'CHGNet_dominant_HGB': {'learning_rate': 0.05, 'max_iter': 300, 'max_leaf_nodes': 15,
     'min_samples_leaf': 20, 'l2_regularization': 1.0, 'early_stopping': False,
     'random_state': SEED},
 'CHGNet_dominant_LR': {'C': 1.0, 'max_iter': 3000, 'solver': 'lbfgs',
     'random_state': SEED, 'n_jobs': 1},
 'Magpie_RF': {'n_estimators': 500, 'min_samples_leaf': 2, 'max_features': 'sqrt',
     'n_jobs': 6, 'random_state': SEED, 'class_weight': None},
}


def utc(): return datetime.now(timezone.utc).isoformat()


def log(message):
    with safe(OUT / 'run.log').open('a', encoding='utf-8') as f:
        f.write(utc() + ' ' + str(message).replace('\n', ' | ') + '\n'); f.flush()


class Writer:
    def __init__(self, d): self.d = d
    def append(self, arm, indices, scores, statuses=None):
        import msvcrt
        rows = []
        statuses = ['ok'] * len(indices) if statuses is None else statuses
        for i, value, status in zip(indices, scores, statuses):
            finite = value is not None and math.isfinite(float(value))
            rows.append({'id': str(self.d['id'][i]), 'fold': int(self.d['fold'][i]),
                'group': str(self.d['group'][i]), 'arm': arm, 'status': status,
                'score': float(value) if finite else '',
                'y_pred': int(float(value) >= .5) if finite else ''})
            if len(rows) == 10: self.flush(rows); rows = []
        if rows: self.flush(rows)
    def flush(self, rows):
        import msvcrt
        with safe(CACHE / 'prediction.lock').open('a+b') as lock:
            lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            try:
                with safe(OUT / 'predictions.csv').open('a', newline='', encoding='utf-8') as f:
                    csv.DictWriter(f, fieldnames=FIELDS).writerows(rows); f.flush(); os.fsync(f.fileno())
            finally:
                lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def weights(d, rows):
    import numpy as np
    y = d['y'][rows]; w = 1.0 / d['group_size'][rows]
    counts = np.bincount(y, weights=w, minlength=2)
    if np.any(counts == 0): raise ValueError('One TRAIN class is absent')
    return w * (w.sum() / (2 * counts))[y]


def prepare(task_name):
    environment()
    if OUT.exists(): raise RuntimeError('G1 directory already exists; use --run or diagnosed --recompute')
    safe(OUT).mkdir()
    d = load_data()
    prepare_official_cgcnn()
    cfg = {'experiment': 'G1', 'status': 'prepared', 'created_utc': utc(),
        'task_name': task_name, 'LLM_calls': 0, 'model_names': MODELS,
        'seed': SEED, 'parameters_frozen_before_fit': PARAMS,
        'classification': 'Tc > 10 K; score >= 0.5 predicts positive',
        'training_weight': '1/original-full-cohort-group-size times TRAIN-only class balance; balance computed separately in inner-fit and outer-TRAIN',
        'early_stopping': 'CGCNN inner SGKF5 first split; group-isolated weighted BA; no outer TEST use',
        'hyperparameter_tuning': 'none; fixed architecture/parameters selected before fitting',
        'limits': {'each_model_3_outer_folds_cumulative_fit_wall_seconds': 10800,
                   'checkpoint_rows': 10, 'automatic_task_restarts': 0},
        'G2_candidate_selection_rule': 'Among full-3764 valid candidates maximize pooled group-weighted OOF BA; exact ties by method lexicographic order. If none complete, use maximum common-valid coverage before BA and report changed n. This is adaptive development selection, not independent test.',
        'input_paths': {'accepted_train': str(DATA), 'raw_structures': str(FACTS)},
        'input_hashes': {'accepted_train': digest(DATA), 'raw_structures': digest(FACTS)},
        'source_hashes': {name: digest(ROOT / name) for name in ('g1_data.py', 'g1_run.py', 'g1_launch.ps1')},
        'outer_inner_splits': frozen_splits(d),
        'implementations': {'CGCNN': {'official_repo': 'https://github.com/txie-93/cgcnn',
            'commit': CGCNN_COMMIT, 'frozen_downloads': prepare_official_cgcnn(),
            'adaptation': 'Unmodified official CrystalGraphConvNet; custom full-occupancy weighted atom initialization and original-geometry neighbor loader; random initialized trainable weights'},
          'CHGNet': {'official_docs': 'https://chgnet.lbl.gov/api', 'checkpoint': '0.3.0',
            'crystal_embedding': 'native mean pooled final atom embeddings, return_crystal_feas=True, no relaxation or fit',
            'disorder_projection': 'one fully occupied atom per supplied site: maximum positive element occupancy; exact ties smallest atomic number; geometry unchanged. Minority occupancies discarded and vacancies filled; limited baseline only.',
            'pretraining_boundary': 'Universal potential trained on MP structures/energies/forces/magmoms, not these Tc labels. Structural/compositional overlap with cohort is possible and has not been ruled out.'},
          'Magpie': {'official_docs': 'https://hackingmaterials.lbl.gov/matminer/matminer.featurizers.composition.html',
            'features': 'ElementProperty.from_preset(magpie, impute_nan=False), 132 property statistics on complete occupancy composition; not X127 element fractions'}},
        'structure_boundary': 'All geometry originates frozen proxy coordinates; no claim these are measured specimen structures.',
        'statistical_protocol': {'threshold': .5, 'group_bootstrap': 5000, 'seed': 20261016,
            'weightings': ['1/original-group-size', 'one per material'],
            'subsets': ['all', 'exclude93', 'only93']},
        'git_commit_at_prepare': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    write_json(OUT / 'config.json', cfg)
    write_json(OUT / 'metrics.json', {'status': 'prepared', 'LLM_calls': 0})
    with safe(OUT / 'predictions.csv').open('x', encoding='utf-8', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    safe(OUT / 'README.md').write_text(
        '# G1 structure and pretrained baselines\n'
        'Finite Stage 1 development experiment; no LLM calls.\n'
        'Original 3764 rows, association groups and three outer folds.\n'
        'Official CGCNN is trained from random initialization on CUDA.\n'
        'CGCNN occupancy-weighted element vectors preserve supplied composition.\n'
        'Inner TRAIN group split selects epochs; outer TEST is never early stopping data.\n'
        'Selected epoch count is refit on all outer TRAIN from fresh initialization.\n'
        'Training loss/ML sample weights use inverse original-group size and TRAIN class balance.\n'
        'CHGNet0.3.0 frozen embeddings use dominant-site ordered projections.\n'
        'Projection removes minority occupancies and fills vacancies; see config coverage.\n'
        'Both geometry and CHGNet pretraining overlap limit interpretation.\n'
        'Magpie132 property statistics use complete occupancy compositions.\n'
        'Imputer and logistic scaling are fitted on each outer TRAIN only.\n'
        'Each model has a cumulative three-fold training wall limit of three hours.\n'
        'One-shot scheduled task has no runtime limit; SetThreadExecutionState prevents sleep.\n'
        'Predictions append/fsync every ten rows; artifacts/cache live outside OneDrive.\n'
        'Full metrics and paired group-bootstrap intervals are consolidated after predictions.\n'
        'All findings are development results; no external validation.\n', encoding='utf-8')
    log('PREPARED finite G1; frozen configuration before any model fitting')


def cpu_worker(arm):
    environment(); prevent_sleep()
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    d = load_data(); cfg = json.loads((OUT / 'config.json').read_text(encoding='utf-8'))
    with np.load(CACHE / ('magpie.npz' if arm == 'Magpie_RF' else 'chgnet_embeddings.npz'), allow_pickle=False) as z:
        X = z['features']
    writer = Writer(d); metadata = {'arm': arm, 'folds': [], 'cumulative_fit_wall_seconds': 0,
                                    'status': 'running'}
    timer = 0.0
    for sp in cfg['outer_inner_splits']:
        tr, te = np.asarray(sp['train']), np.asarray(sp['test'])
        valid = np.isfinite(X).any(axis=1)
        tr = tr[valid[tr]]; valid_te = te[valid[te]]
        if len(tr) == 0 or len(np.unique(d['y'][tr])) < 2:
            writer.append(arm, te, [None] * len(te), ['unavailable_training_features'] * len(te)); continue
        imp = SimpleImputer(strategy='median', keep_empty_features=True)
        A = imp.fit_transform(X[tr]); B = imp.transform(X[valid_te])
        scale = None
        if arm == 'CHGNet_dominant_LR':
            scale = StandardScaler(); A = scale.fit_transform(A); B = scale.transform(B)
        Model = {'Magpie_RF': RandomForestClassifier, 'CHGNet_dominant_HGB': HistGradientBoostingClassifier,
                 'CHGNet_dominant_LR': LogisticRegression}[arm]
        model = Model(**PARAMS[arm]); fit_started = time.monotonic()
        model.fit(A, d['y'][tr], sample_weight=weights(d, tr))
        duration = time.monotonic() - fit_started; timer += duration
        if timer > 10800:
            metadata['status'] = 'partial_training_timeout'; break
        p = model.predict_proba(B)[:, 1]
        writer.append(arm, valid_te, p)
        missing_te = te[~valid[te]]
        writer.append(arm, missing_te, [None] * len(missing_te), ['unavailable_structure_embedding'] * len(missing_te))
        # Model artifacts are cache-only, for an authorized later CIF subset comparison.
        import joblib
        joblib.dump({'imputer': imp, 'scaler': scale, 'model': model}, safe(CACHE / f'{arm}_fold{sp["fold"]}.joblib'))
        metadata['folds'].append({'fold': sp['fold'], 'fit_n': len(tr), 'test_n': len(te),
            'valid_test_n': len(valid_te), 'groups_train': int(len(set(d['group'][tr]))),
            'groups_test': int(len(set(d['group'][te]))), 'group_overlap': 0,
            'imputer_fit': 'outer TRAIN only', 'scaler_fit': 'outer TRAIN only' if scale else None,
            'fit_wall_seconds': duration})
        metadata['cumulative_fit_wall_seconds'] = timer
        write_json(CACHE / f'{arm}_metadata.json', metadata)
        log(f'{arm} fold={sp["fold"]} fit_n={len(tr)} test={len(valid_te)}/{len(te)} fit_seconds={duration:.2f}')
    if len(metadata['folds']) == 3: metadata['status'] = 'complete'
    write_json(CACHE / f'{arm}_metadata.json', metadata)


def collate(graphs, selected, device):
    import torch
    atoms, bonds, nbrs, maps = [], [], [], []
    offset = 0
    for i in selected:
        atom, bond, nbr = graphs[int(i)]
        atoms.append(atom); bonds.append(bond); nbrs.append(nbr + offset)
        maps.append(torch.arange(len(atom), dtype=torch.long, device=device) + offset)
        offset += len(atom)
    return (torch.cat(atoms).to(device), torch.cat(bonds).to(device),
            torch.cat(nbrs).to(device), maps)


def gpu_cgcnn(d, graphs, cfg):
    import numpy as np
    import torch
    from sklearn.metrics import balanced_accuracy_score
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    cls = cgcnn_class(); params = PARAMS['CGCNN_scratch']
    writer = Writer(d); total_training = 0.0
    meta = {'arm': 'CGCNN_scratch', 'folds': [], 'status': 'running',
            'cumulative_training_wall_seconds': 0.0}
    def initialize(seed):
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        return cls(graphs[0][0].shape[1], graphs[0][1].shape[2],
                   atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                   classification=True).to('cuda')
    def fit_epoch(model, opt, index, sample_w, rng):
        nonlocal total_training
        started = time.monotonic(); model.train()
        order = rng.permutation(index); losses = []
        for start in range(0, len(order), 32):
            if total_training + time.monotonic() - started >= 10800:
                total_training += time.monotonic() - started
                raise TimeoutError('CGCNN cumulative training wall limit 10800s reached')
            ix = order[start:start+32]
            opt.zero_grad(set_to_none=True)
            output = model(*collate(graphs, ix, 'cuda'))
            losses_ = torch.nn.functional.nll_loss(output, torch.as_tensor(d['y'][ix], device='cuda'), reduction='none')
            w = torch.tensor(sample_w[ix], dtype=torch.float32, device='cuda')
            loss = torch.mean(losses_ * w)
            if not bool(torch.isfinite(loss)): raise FloatingPointError('CGCNN loss nonfinite')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step(); losses.append(float(loss.detach()))
        torch.cuda.synchronize(); total_training += time.monotonic() - started
        return float(np.mean(losses))
    def predict(model, index):
        model.eval(); result = []
        with torch.inference_mode():
            for start in range(0, len(index), 32):
                out = model(*collate(graphs, index[start:start+32], 'cuda'))
                result.extend(out[:, 1].exp().cpu().numpy().tolist())
        return np.asarray(result)
    try:
        for sp in cfg['outer_inner_splits']:
            inner, val = np.asarray(sp['inner_fit']), np.asarray(sp['inner_val'])
            tr, te = np.asarray(sp['train']), np.asarray(sp['test'])
            model = initialize(SEED + sp['fold']); opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)
            w = np.zeros(len(d['y'])); w[inner] = weights(d, inner); w[inner] /= w[inner].mean()
            best, best_epoch, stale, history = -math.inf, 0, 0, []
            rng = np.random.default_rng(SEED + sp['fold'])
            for epoch in range(1, 201):
                loss = fit_epoch(model, opt, inner, w, rng)
                vp = predict(model, val)
                ba = float(balanced_accuracy_score(d['y'][val], vp >= .5,
                    sample_weight=1 / d['group_size'][val]))
                history.append({'epoch': epoch, 'training_loss': loss, 'inner_validation_BA': ba})
                if ba > best + 1e-5: best, best_epoch, stale = ba, epoch, 0
                else: stale += 1
                if epoch % 5 == 0 or epoch == 1:
                    log(f'CGCNN inner fold={sp["fold"]} epoch={epoch} BA={ba:.6f} best_epoch={best_epoch} train_seconds={total_training:.1f}')
                if stale >= 25: break
            del opt, model; torch.cuda.empty_cache()
            model = initialize(SEED + 100 + sp['fold']); opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)
            w = np.zeros(len(d['y'])); w[tr] = weights(d, tr); w[tr] /= w[tr].mean()
            rng = np.random.default_rng(SEED + 100 + sp['fold'])
            for epoch in range(1, best_epoch + 1):
                fit_epoch(model, opt, tr, w, rng)
                if epoch % 5 == 0 or epoch == best_epoch:
                    log(f'CGCNN outer-TRAIN refit fold={sp["fold"]} epoch={epoch}/{best_epoch}')
            p = predict(model, te); writer.append('CGCNN_scratch', te, p)
            torch.save(model.state_dict(), safe(CACHE / f'CGCNN_scratch_fold{sp["fold"]}.pt'))
            meta['folds'].append({'fold': sp['fold'], 'inner_fit_n': len(inner), 'inner_val_n': len(val),
                'outer_train_refit_n': len(tr), 'test_n': len(te), 'selected_epochs': best_epoch,
                'best_inner_weighted_BA': best, 'history': history, 'group_overlap': 0,
                'test_used_for_selection': False})
            meta['cumulative_training_wall_seconds'] = total_training
            write_json(CACHE / 'CGCNN_scratch_metadata.json', meta)
            log(f'CGCNN fold={sp["fold"]} complete test_n={len(te)} selected_epochs={best_epoch}')
            del model, opt; torch.cuda.empty_cache()
        meta['status'] = 'complete'
    except TimeoutError as e:
        meta['status'] = 'partial_training_timeout'; meta['reason'] = str(e)
        log(str(e))
    meta['cumulative_training_wall_seconds'] = total_training
    write_json(CACHE / 'CGCNN_scratch_metadata.json', meta)


def chgnet_embeddings(rows, log):
    import numpy as np
    import torch
    from chgnet.model.model import CHGNet
    path = CACHE / 'chgnet_embeddings.npz'
    if path.exists():
        return json.loads((CACHE / 'chgnet_embedding_metadata.json').read_text(encoding='utf-8'))
    model = CHGNet.load(model_name='0.3.0', use_device='cuda'); model.eval()
    features = np.full((len(rows), 64), np.nan)
    failures = []; wall = time.monotonic()
    for i, row in enumerate(rows):
        try:
            s = structure(row, ordered=True)
            with torch.no_grad():
                result = model.predict_structure(s, task='e', return_crystal_feas=True)
            value = np.asarray(result['crystal_fea'], dtype=float).reshape(-1)
            if value.shape != (64,) or not np.isfinite(value).all():
                raise ValueError(f'Unexpected CHGNet embedding {value.shape}')
            features[i] = value
        except Exception as exc:
            failures.append({'id': row['id'], 'error_type': type(exc).__name__, 'error': str(exc)[:500]})
            log(f'CHGNet embedding unavailable {row["id"]}: {type(exc).__name__}: {exc}')
        if (i+1) % 10 == 0:
            np.savez_compressed(safe(path), features=features)
        if (i+1) % 100 == 0: log(f'CHGNet embeddings {i+1}/3764 failures={len(failures)}')
    np.savez_compressed(safe(path), features=features)
    import chgnet
    checkpoint = Path(chgnet.__file__).parent / 'pretrained' / '0.3.0' / 'chgnet_0.3.0_e29f68s314m37.pth.tar'
    meta = {'checkpoint_name': '0.3.0', 'checkpoint_path': str(checkpoint),
        'checkpoint_sha256': digest(checkpoint), 'n': len(rows),
        'valid': int(np.isfinite(features).all(axis=1).sum()), 'n_features': 64,
        'failures': failures, 'inference_wall_seconds': time.monotonic() - wall,
        'projection_by_id': {r['id']: {'n_disordered_sites': r['n_disordered_sites'],
            'minority_occupancy_lost': r['minority_occupancy_lost'],
            'vacancy_occupancy_filled': r['vacancy_occupancy_filled'],
            'n_sites': r['n_sites']} for r in rows}}
    write_json(CACHE / 'chgnet_embedding_metadata.json', meta)
    return meta


def run():
    environment(); prevent_sleep()
    import numpy as np
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no silent CPU substitution')
    cfg = json.loads((OUT / 'config.json').read_text(encoding='utf-8'))
    for name, expected in cfg['source_hashes'].items():
        if digest(ROOT / name) != expected: raise RuntimeError('Frozen source changed: ' + name)
    d = load_data()
    cfg.update(status='running', started_utc=utc(), runtime={'python': sys.executable,
        'torch': torch.__version__, 'CUDA': torch.version.cuda, 'device': torch.cuda.get_device_name(),
        'packages': {p: importlib.metadata.version(p) for p in ('chgnet', 'matminer', 'pymatgen', 'scikit-learn', 'numpy')}})
    write_json(OUT / 'config.json', cfg)
    log('RUN finite G1 started; GPU exclusive; no LLM or external evaluation')
    rows, coverage = load_structures(d, log)
    atom_map = json.loads((CACHE / 'official_cgcnn/atom_init.json').read_text())
    all_Z = sorted({int(z) for r in rows for z in r['species_Z']})
    unsupported_cgcnn = [z for z in all_Z if str(z) not in atom_map]
    unsupported_chgnet = [z for z in all_Z if z < 1 or z > 94]
    cfg['structure_coverage'] = coverage
    cfg['element_coverage'] = {'atomic_numbers': all_Z, 'unsupported_CGCNN': unsupported_cgcnn,
                              'unsupported_CHGNet': unsupported_chgnet}
    write_json(OUT / 'config.json', cfg)
    if unsupported_cgcnn or unsupported_chgnet:
        raise RuntimeError('Unsupported element; do not replace or manufacture predictions')
    X, labels = magpie(rows, log)
    cfg['magpie_feature_names'] = labels
    cfg['magpie_missing_cells'] = int((~np.isfinite(X)).sum())
    write_json(OUT / 'config.json', cfg)
    childlog = safe(CACHE / 'Magpie_RF_process.log').open('w', encoding='utf-8')
    cpu_started = time.monotonic()
    rf = subprocess.Popen([sys.executable, '-X', 'utf8', '-B', str(ROOT / 'g1_run.py'), '--cpu-worker', 'Magpie_RF'],
                          cwd=ROOT, stdout=childlog, stderr=subprocess.STDOUT, creationflags=0x08000000)
    log(f'Magpie RF parallel finite worker pid={rf.pid}')
    graph_payload = cgcnn_graphs(rows, log)
    cfg['CGCNN_graph_coverage'] = {k: v for k, v in graph_payload.items() if k != 'graphs'}
    write_json(OUT / 'config.json', cfg)
    gpu_cgcnn(d, graph_payload['graphs'], cfg)
    del graph_payload; torch.cuda.empty_cache()
    emb = chgnet_embeddings(rows, log)
    cfg['CHGNet_embedding_metadata'] = emb
    write_json(OUT / 'config.json', cfg)
    for arm in ('CHGNet_dominant_HGB', 'CHGNet_dominant_LR'):
        with safe(CACHE / f'{arm}_process.log').open('w', encoding='utf-8') as dest:
            try:
                result = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(ROOT / 'g1_run.py'), '--cpu-worker', arm],
                    cwd=ROOT, stdout=dest, stderr=subprocess.STDOUT, timeout=10800, creationflags=0x08000000)
                if result.returncode: log(f'CPU worker {arm} failed exit={result.returncode}')
            except subprocess.TimeoutExpired:
                log(f'CPU worker {arm} exceeded cumulative 3h process limit; results partial')
    try:
        rf.wait(timeout=max(1, 10800 - (time.monotonic() - cpu_started)))
    except subprocess.TimeoutExpired:
        rf.kill(); rf.wait(); log('Magpie RF worker exceeded 3h process limit; results partial')
    childlog.close()
    meta = {}
    for arm in MODELS:
        file = CACHE / f'{arm}_metadata.json'
        meta[arm] = json.loads(file.read_text(encoding='utf-8')) if file.exists() else {'status': 'unavailable'}
    with (OUT / 'predictions.csv').open(encoding='utf-8', newline='') as f: pred = list(csv.DictReader(f))
    assert len({(r['id'], r['arm']) for r in pred}) == len(pred)
    counts = {arm: sum(r['arm'] == arm and r['status'] == 'ok' for r in pred) for arm in MODELS}
    complete = all(value == len(d['id']) for value in counts.values())
    cfg.update(status='complete' if complete else 'partial', completed_utc=utc(), model_metadata=meta,
               valid_by_arm=counts, LLM_calls=0)
    write_json(OUT / 'config.json', cfg)
    write_json(OUT / 'metrics.json', {'status': cfg['status'], 'valid_by_arm': counts,
        'n_total': len(d['id']), 'groups': 1117, 'LLM_calls': 0, 'model_metadata': meta,
        'statistical_report_status': 'Predictions complete; root coordinator consolidates 5000-group-bootstrap comparisons with saved baselines and A3.'})
    log(f'COMPLETE status={cfg["status"]} valid_by_arm={counts}; no next stage launched')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--task-name', default='Codex.V2Diagnosis.G1.20261018')
    parser.add_argument('--cpu-worker', choices=MODELS)
    args = parser.parse_args()
    try:
        if args.prepare: prepare(args.task_name)
        elif args.cpu_worker: cpu_worker(args.cpu_worker)
        elif args.run: run()
        else: parser.error('Specify --prepare, --run, or --cpu-worker')
    except Exception:
        if OUT.exists(): log('FATAL ' + traceback.format_exc())
        raise
