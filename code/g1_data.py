"""Label-blind structure/graph adapters for the finite G1 baselines.

Old experiment directories are read only. All new caches are under work/g1.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import ctypes
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import urllib.request

ROOT = Path('C:/Users/28908/Codex/fung-research/v2-diagnosis').resolve()
CACHE = ROOT / 'work' / 'g1'
OUT = ROOT / 'G1_structure_baselines'
DATA = ROOT.parent / '2026-10-09-structure-classification/inputs/accepted_train.npz'
FACTS = ROOT.parent / 'workflow-review-2026-10-10/common-facts-cohort-actual-001/common_facts_fullTRAIN.jsonl'
CGCNN_COMMIT = 'f42ab233c4ee0c416879d6bc2d22a264418413ad'
SEED = 20261018


def safe(path):
    p = Path(path).resolve()
    if 'onedrive' in str(p).casefold() or not p.is_relative_to(ROOT):
        raise ValueError(f'Forbidden destination: {p}')
    for ancestor in (p, *p.parents):
        if ancestor.exists() and (ancestor.is_symlink() or getattr(ancestor, 'is_junction', lambda: False)()):
            raise ValueError(f'Linked write ancestor: {ancestor}')
    return p


def environment():
    sys.dont_write_bytecode = True
    safe(CACHE).mkdir(parents=True, exist_ok=True)
    for key, child in [('TORCH_HOME', 'torch'), ('MPLCONFIGDIR', 'matplotlib'),
                       ('XDG_CACHE_HOME', 'xdg'), ('TMP', 'tmp'), ('TEMP', 'tmp')]:
        dest = safe(CACHE / child)
        dest.mkdir(exist_ok=True)
        os.environ[key] = str(dest)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ.setdefault('OMP_NUM_THREADS', '8')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')


def prevent_sleep():
    value = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    if not value:
        raise RuntimeError('SetThreadExecutionState failed')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    with safe(path).open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def load_data():
    import numpy as np
    with np.load(DATA, allow_pickle=False) as z:
        raw = {k: z[k] for k in z.files}
    d = dict(X127=raw['X127'], y=raw['label'].astype(int), fold=raw['fold'],
             group=raw['component_id'].astype(str), id=raw['material_id'],
             source_index=raw['source_index'], tc=raw['target_K'])
    assert len(d['id']) == 3764 and np.array_equal(d['y'], d['tc'] > 10)
    groups, inverse, sizes = np.unique(d['group'], return_inverse=True, return_counts=True)
    assert len(groups) == 1117 and len(np.unique(d['fold'])) == 3
    d['group_size'] = sizes[inverse]
    for group in groups:
        assert len(np.unique(d['fold'][d['group'] == group])) == 1
    return d


def load_structures(d, log):
    """Retain frozen geometry and complete site occupancies, never label fields."""
    path = CACHE / 'structures.json.gz'
    if path.exists():
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            payload = json.load(f)
        if payload['data_sha256'] != digest(DATA) or payload['facts_sha256'] != digest(FACTS):
            raise ValueError('Structure cache scientific source mismatch')
        return payload['rows'], payload['coverage']
    by_source = {int(v): i for i, v in enumerate(d['source_index'])}
    rows = [None] * len(d['id'])
    flags, sizes, disorder, vacancy, minority = Counter(), [], [], [], []
    with FACTS.open(encoding='utf-8') as f:
        for count, line in enumerate(f, 1):
            cf = json.loads(line)['coordinator_record']['common_facts']
            identity = cf['identity']; i = by_source[int(identity['source_index'])]
            assert str(identity['material_id']) == str(d['id'][i])
            assert str(identity['component_id']) == str(d['group'][i])
            assert int(identity['outer_fold']) == int(d['fold'][i])
            import numpy as np
            assert np.array_equal(np.asarray(cf['values'][:127]), d['X127'][i])
            raw = cf['raw_graph']['raw_structure']
            retained = {k: raw[k] for k in ('lattice', 'frac_coords', 'species_symbols',
                'species_occupancy', 'species_site', 'species_Z', 'n_sites', 'site_occupancy')}
            site = [defaultdict(float) for _ in range(raw['n_sites'])]
            for symbol, amount, j in zip(raw['species_symbols'], raw['species_occupancy'], raw['species_site']):
                if float(amount) > 0:
                    site[int(j)][str(symbol)] += float(amount)
            if len(site) != len(raw['frac_coords']) or any(not s for s in site):
                raise ValueError(f'Empty/misaligned occupied site: {d["id"][i]}')
            sums = [sum(s.values()) for s in site]
            if any(v > 1 + 1e-5 or v <= 0 for v in sums):
                raise ValueError(f'Invalid site occupancy: {d["id"][i]}')
            lost = sum(sum(s.values()) - max(s.values()) for s in site)
            missing = sum(max(0, 1 - v) for v in sums)
            n_disorder = sum(len(s) > 1 for s in site)
            retained.update(id=str(d['id'][i]), source_index=int(d['source_index'][i]),
                            sites=[dict(s) for s in site], proxy_flags=cf['proxy_flags'],
                            n_disordered_sites=n_disorder, minority_occupancy_lost=lost,
                            vacancy_occupancy_filled=missing)
            rows[i] = retained
            sizes.append(raw['n_sites']); disorder.append(n_disorder)
            vacancy.append(missing); minority.append(lost)
            flags.update([str(cf['proxy_flags']['coordinate_source_proxy'])])
            if count % 500 == 0: log(f'Structure alignment {count}/3764')
    assert all(r is not None for r in rows)
    coverage = {'n': len(rows), 'source_alignment_failures': 0,
                'proxy_coordinate_flags': dict(flags), 'n_sites_min': min(sizes),
                'n_sites_max': max(sizes), 'n_disordered_structures': sum(v > 0 for v in disorder),
                'n_partially_vacant_structures': sum(v > 1e-8 for v in vacancy),
                'total_minority_site_occupancy_lost_in_CHGNet_projection': sum(minority),
                'total_vacancy_site_occupancy_filled_in_CHGNet_projection': sum(vacancy)}
    payload = {'data_sha256': digest(DATA), 'facts_sha256': digest(FACTS),
               'rows': rows, 'coverage': coverage}
    with gzip.open(safe(path), 'wt', encoding='utf-8') as f: json.dump(payload, f)
    return rows, coverage


def prepare_official_cgcnn():
    directory = safe(CACHE / 'official_cgcnn'); directory.mkdir(exist_ok=True)
    outputs = {}
    for local, remote in [('model.py', 'cgcnn/model.py'),
                          ('atom_init.json', 'data/sample-regression/atom_init.json'),
                          ('LICENSE', 'LICENSE')]:
        path = directory / local
        url = f'https://raw.githubusercontent.com/txie-93/cgcnn/{CGCNN_COMMIT}/{remote}'
        if not path.exists():
            payload = urllib.request.urlopen(url, timeout=60).read()
            safe(path).write_bytes(payload)
        outputs[local] = {'url': url, 'sha256': digest(path)}
    return outputs


def cgcnn_class():
    spec = importlib.util.spec_from_file_location('g1_official_cgcnn', CACHE / 'official_cgcnn/model.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.CrystalGraphConvNet


def structure(row, ordered=False):
    from pymatgen.core import Structure, Element
    species = row['sites']
    if ordered:
        species = [sorted(s, key=lambda e: (-s[e], Element(e).Z))[0] for s in species]
    return Structure(row['lattice'], species, row['frac_coords'], coords_are_cartesian=False)


def magpie(rows, log):
    import numpy as np
    from matminer.featurizers.composition import ElementProperty
    path = CACHE / 'magpie.npz'
    feat = ElementProperty.from_preset('magpie', impute_nan=False)
    labels = feat.feature_labels()
    if path.exists():
        with np.load(path, allow_pickle=False) as z: return z['features'], labels
    from pymatgen.core import Composition
    values, cached = [], {}
    for i, row in enumerate(rows):
        amounts = defaultdict(float)
        for s in row['sites']:
            for e, v in s.items(): amounts[e] += v
        comp = Composition(dict(amounts))
        key = tuple(sorted(comp.fractional_composition.get_el_amt_dict().items()))
        if key not in cached: cached[key] = feat.featurize(comp)
        values.append(cached[key])
        if (i + 1) % 500 == 0: log(f'Magpie {i+1}/3764')
    X = np.asarray(values, dtype=float); X[~np.isfinite(X)] = np.nan
    np.savez_compressed(safe(path), features=X)
    return X, labels


def cgcnn_graphs(rows, log):
    import numpy as np
    import torch
    path = CACHE / 'cgcnn_graphs.pt'
    if path.exists(): return torch.load(path, weights_only=False)
    atom_map = {int(k): np.asarray(v, dtype=float) for k, v in
                json.loads((CACHE / 'official_cgcnn/atom_init.json').read_text()).items()}
    from pymatgen.core import Element
    centers = np.arange(0, 8.0 + 0.2, 0.2)
    result, low_neighbors = [], []
    for i, row in enumerate(rows):
        s = structure(row)
        atoms = np.stack([sum(amount * atom_map[Element(e).Z] for e, amount in occ.items())
                          for occ in row['sites']])
        nbr = s.get_all_neighbors(8.0, include_index=True)
        indices, distances = [], []
        missing_sites = 0
        for neighbors in nbr:
            ordered = sorted(neighbors, key=lambda t: (float(t.nn_distance), int(t.index), tuple(t.image)))[:12]
            if len(ordered) < 12: missing_sites += 1
            indices.append([int(t.index) for t in ordered] + [0] * (12 - len(ordered)))
            distances.append([float(t.nn_distance) for t in ordered] + [9.0] * (12 - len(ordered)))
        D = np.asarray(distances)
        features = np.exp(-((D[:, :, None] - centers[None, None, :]) / 0.2) ** 2)
        result.append((torch.tensor(atoms, dtype=torch.float32),
                       torch.tensor(features, dtype=torch.float32),
                       torch.tensor(indices, dtype=torch.long)))
        if missing_sites: low_neighbors.append({'id': row['id'], 'sites_padded': missing_sites})
        if (i + 1) % 100 == 0: log(f'CGCNN graph {i+1}/3764')
    payload = {'graphs': result, 'padded_neighbor_structures': low_neighbors,
               'radius_A': 8.0, 'max_neighbors': 12, 'n_bond_features': len(centers)}
    torch.save(payload, safe(path))
    return payload


def frozen_splits(d):
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold
    result = []
    for fold in sorted(np.unique(d['fold'])):
        tr = np.flatnonzero(d['fold'] != fold); te = np.flatnonzero(d['fold'] == fold)
        inner = StratifiedGroupKFold(5, shuffle=True, random_state=SEED + int(fold))
        fit, val = next(inner.split(tr, d['y'][tr], d['group'][tr]))
        fit, val = tr[fit], tr[val]
        groups = [set(d['group'][p]) for p in (fit, val, te)]
        assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        result.append({'fold': int(fold), 'train': tr.tolist(), 'inner_fit': fit.tolist(),
                       'inner_val': val.tolist(), 'test': te.tolist(), 'group_overlap': 0})
    return result
