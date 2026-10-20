"""Separate, no-fit G1 statistics process (keeps training environment isolated)."""
from __future__ import annotations
import csv
import gzip
import hashlib
import json
from pathlib import Path

# Intentionally import the coordinator statistics environment in this process only.
from g_stage1_report import (ROOT, classification_bundle, frozen_baselines,
                            choose_g1_best, load_data, np, read_scores)
from f_stats import summarize
from g1_data import safe, digest, write_json


def finalize():
    out = ROOT / 'G1_structure_baselines'; cache = ROOT / 'work/g1'
    cfg = json.loads((out / 'config.json').read_text(encoding='utf-8'))
    if cfg['status'] not in ('complete', 'partial'):
        raise RuntimeError('G1 model generation has not finished')
    d = load_data(); candidates = read_scores(out / 'predictions.csv', data=d)
    for name in cfg['model_names']:
        candidates.setdefault(name, np.full(3764, np.nan))
    scores = dict(candidates); scores.update(frozen_baselines(d))
    metrics = json.loads((out / 'metrics.json').read_text(encoding='utf-8'))
    metrics['classification'] = classification_bundle(scores,
        'G1：结构/预训练/成分基线与已有 HGB、强 ML、A3 的开发比较', data=d, seed=20261016)
    complete = {k: p for k, p in candidates.items() if np.isfinite(p).all()}
    if complete:
        if len(complete) == len(candidates):
            selection_stats = metrics['classification']
        else:
            selection_stats = classification_bundle(complete, 'G1全覆盖候选选择', data=d, seed=20261016)
        metrics['G2_best_method'] = choose_g1_best(complete, selection_stats)
        metrics['G2_selection_note'] = cfg['G2_candidate_selection_rule']
    else:
        metrics['G2_best_method'] = None
        metrics['G2_selection_note'] = 'No candidate covers all 3764; requires explicitly reported common-valid selection.'
    projection = cfg['CHGNet_embedding_metadata']['projection_by_id']
    ordered = np.asarray([projection[str(i)]['n_disordered_sites'] == 0 and
        projection[str(i)]['vacancy_occupancy_filled'] <= 1e-8 for i in d['id']], dtype=bool)
    loss = np.asarray([projection[str(i)]['minority_occupancy_lost'] for i in d['id']], float)
    vacancy = np.asarray([projection[str(i)]['vacancy_occupancy_filled'] for i in d['id']], float)
    n_sites = np.asarray([projection[str(i)]['n_sites'] for i in d['id']], float)
    with gzip.open(cache / 'structures.json.gz', 'rt', encoding='utf-8') as f:
        source_rows = json.load(f)['rows']
    occupancies = np.asarray([sum(sum(s.values()) for s in r['sites']) for r in source_rows])
    quantiles = [0, .25, .5, .75, .9, .95, .99, 1]
    cfg['CHGNet_projection_summary'] = {
        'fully_ordered_full_occupancy_n': int(ordered.sum()),
        'projection_changed_n': int((~ordered).sum()),
        'projection_changed_fraction': float((~ordered).mean()),
        'quantiles': quantiles,
        'minority_occupancy_lost_per_material': np.quantile(loss, quantiles).tolist(),
        'minority_fraction_of_original_occupied_composition': np.quantile(loss / occupancies, quantiles).tolist(),
        'vacancy_occupancy_filled_per_material': np.quantile(vacancy, quantiles).tolist(),
        'vacancy_fraction_per_site': np.quantile(vacancy / n_sites, quantiles).tolist(),
        'ordered_subset_is_not_external': 'The unaltered ordered subset uses the same frozen outer-fold models trained on the whole eligible TRAIN; no retraining or selection on this subset.'}
    original = {str(g): int(s) for g, s in zip(d['group'], d['group_size'])}
    commonvalid = np.logical_and.reduce([np.isfinite(p) for p in scores.values()])
    metrics['CHGNet_projection_subsets'] = {}
    for name, mask in [('fully_ordered_full_occupancy', ordered), ('projection_changed', ~ordered)]:
        stats = {}
        for subset, selection in [('all', np.ones(3764, bool)),
                                  ('exclude_93', d['group'] != '93'), ('only_93', d['group'] == '93')]:
            request = mask & selection
            table = summarize(d['y'], scores, d['group'], original, mask=request & commonvalid,
                              n_boot=5000, seed=20261016)
            table['n_requested'] = int(request.sum())
            table['n_requested_groups'] = int(len(np.unique(d['group'][request])))
            table['n_missing'] = int((request & ~commonvalid).sum())
            stats[subset] = table
        metrics['CHGNet_projection_subsets'][name] = stats
    metrics['report_markdown'] = metrics['classification']['report_markdown']
    metrics['audit'] = {'source_hashes_unchanged': {name: digest(ROOT / name) == value
                            for name, value in cfg['source_hashes'].items()},
        'scientific_input_hashes_unchanged': {key: digest(cfg['input_paths'][key]) == value
                            for key, value in cfg['input_hashes'].items()},
        'prediction_rows': sum(1 for _ in csv.DictReader((out / 'predictions.csv').open(encoding='utf-8', newline=''))),
        'valid_by_arm': {name: int(np.isfinite(p).sum()) for name, p in candidates.items()},
        'LLM_calls': 0, 'finalizer_source_sha256': digest(Path(__file__)),
        'statistics_sources_sha256': {name: digest(ROOT / name) for name in
            ('g_stage1_report.py', 'f3_stats.py', 'f_stats.py', 'f_report.py', 'common.py')},
        'bootstrap_seed': 20261016}
    assert all(metrics['audit']['source_hashes_unchanged'].values())
    assert all(metrics['audit']['scientific_input_hashes_unchanged'].values())
    cfg['statistics_status'] = 'complete'; cfg['G2_best_method'] = metrics['G2_best_method']
    write_json(out / 'config.json', cfg); write_json(out / 'metrics.json', metrics)
    with safe(out / 'run.log').open('a', encoding='utf-8') as f:
        f.write('G1 no-fit finalization complete: both weights, all three subsets, families, original-group sizes, projection subsets, 5000 group bootstrap; G2 best=' + str(metrics['G2_best_method']) + '\n')
    print(json.dumps({'status': cfg['status'], 'G2_best_method': metrics['G2_best_method'],
                      'valid_by_arm': metrics['audit']['valid_by_arm']}, ensure_ascii=False))


if __name__ == '__main__': finalize()
