"""Finite, frozen G2 analysis; read old outputs only, never fit/call a model."""
from __future__ import annotations

import argparse
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from common import ROOT, DATA, np, safe_path, write_json, PredictionWriter
from g2_stats import summarize, self_check, KS, COVERAGES, WEIGHTINGS

OUT = ROOT / 'G2_ranking_calibration'
SOURCE = ROOT / 'F3_full3764'
METHODS = ('A3_forced', 'A2_forced', 'HGB127', 'StrongML')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def log(message):
    line = f'{now()} {message}'
    with safe_path(OUT/'run.log').open('a', encoding='utf-8') as stream:
        stream.write(line+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(line, flush=True)


def read_scores(path, ids, requested):
    indices = {str(mid): i for i, mid in enumerate(ids)}
    scores = {method: np.full(len(ids), np.nan) for method in requested}
    with Path(path).open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            arm = row['arm']
            if arm in scores:
                i = indices[row['id']]
                if np.isfinite(scores[arm][i]):
                    raise ValueError(f'duplicate prediction {arm}/{row["id"]}')
                scores[arm][i] = float(row['score'])
    if any(not np.isfinite(values).all() for values in scores.values()):
        raise ValueError('G2 requires complete common-cohort predictions')
    return scores


def _fmt(interval, multiplier=100):
    if interval['point'] is None:
        return 'NA'
    point = interval['point']*multiplier
    lo, hi = interval['ci95']
    return f'{point:.1f} [{lo*multiplier:.1f},{hi*multiplier:.1f}]' if lo is not None else f'{point:.1f} [NA]'


def report(metrics):
    lines = ['# G2 排序、校准和选择性判断（开发结果）',
             '所有区间为5000次按component分组重采样95% percentile CI；两种权重共享组draw；所有paired差值在metrics.json逐项保留。',
             '排序并列采用边界桶分数选择，不依赖标签或ID。加权precision为分数选择计数之比，不是随机打破并列后precision比值的期望。',
             'k为材料数；coverage网格为材料覆盖率，同时报告其组加权实际覆盖。k大于原子集n时全部NA。ECE为预固定10等宽概率bin。',
             'G1best若接入，按G1完整覆盖OOF组加权BA选择，属于自适应开发比较，不是独立检验。']
    for subset, result in metrics['subsets'].items():
        lines += [f'\n## {subset}：n={result["n"]}，groups={result["n_groups"]}，positive={result["n_positive"]}']
        if result['single_group_warning']:
            lines.append('只有1组，CI退化，不能反映跨组不确定性。')
        for weighting in WEIGHTINGS:
            lines += [f'\n### {weighting}', '', '|方法|P@50|P@100|P@200|P@500|ECE|Brier|', '|---|---|---|---|---|---|---|']
            for method, value in result['methods'].items():
                scheme = value[weighting]
                cells = [_fmt(scheme['precision_at_k'][str(k)]['precision']) for k in KS]
                cells += [_fmt(scheme['calibration'][m]) for m in ('ece', 'brier')]
                lines.append('|'+ '|'.join([method]+cells)+'|')
            lines += ['', '|方法|材料覆盖|组权重实际覆盖|保留材料准确率|', '|---|---|---|---|']
            for method, value in result['methods'].items():
                for q in COVERAGES:
                    row = value[weighting]['selective'][f'{q:.1f}']
                    lines.append('|'+ '|'.join([method, f'{q:.0%}', _fmt(row['weighted_coverage']), _fmt(row['accuracy'])])+'|')
        lines += ['', '|方法|p取值数|处于重复p桶的材料数|最大并列桶|', '|---|---|---|---|']
        for method, value in result['methods'].items():
            lines.append(f'|{method}|{value["n_unique_probabilities"]}|{value["n_in_repeated_probability_buckets"]}|{value["largest_probability_tie_n"]}|')
    lines += ['\n所有家族、原组大小分档及三个子集的同样统计在families/group_size_bins；边界p、并列数和分数在methods/top_k_boundaries、selective_boundaries。',
              'ECE/Brier越低越好；paired字段始终arm_a减arm_b，故正ECE差值表示arm_a更差。']
    return '\n'.join(lines)+'\n'


def run(extra_path=None, extra_arm=None, extra_name='G1best'):
    started = time.perf_counter()
    for ancestor in (ROOT, *ROOT.parents):
        if 'onedrive' in str(ancestor.resolve()).lower():
            raise ValueError('OneDrive destination forbidden')
    safe_path(OUT).mkdir(exist_ok=True)
    cfg = json.loads((SOURCE/'config.json').read_text(encoding='utf-8'))
    ids = cfg['case_order']
    cohort = [cfg['cases'][mid] for mid in ids]
    y = np.array([row['evaluation']['y'] for row in cohort])
    groups = np.array([row['group'] for row in cohort]).astype(str)
    folds = np.array([row['fold'] for row in cohort])
    sizes = np.array([row['evaluation']['original_group_size'] for row in cohort])
    families = {family: np.array(mask, dtype=bool) for family, mask in cfg['family_masks'].items()}
    family_definition = cfg['metadata']['family_definitions']
    full_sizes = {str(group): int(size) for group, size in zip(groups, sizes)}
    del cohort, cfg
    scores = read_scores(SOURCE/'predictions.csv', ids, METHODS)
    with np.load(DATA, allow_pickle=False) as data:
        assert np.array_equal(ids, data['material_id'])
        assert np.array_equal(y, data['label'])
        assert np.array_equal(groups, data['component_id'].astype(str))
        assert np.array_equal(folds, data['fold'])
    paths = [SOURCE/'predictions.csv', SOURCE/'config.json', DATA, ROOT/'g2_analysis.py', ROOT/'g2_stats.py']
    extra = None
    if extra_path:
        if not extra_arm:
            raise ValueError('extra-arm must identify the frozen complete G1best arm')
        scores[extra_name] = read_scores(extra_path, ids, [extra_arm])[extra_arm]
        paths.append(Path(extra_path))
        extra = {'name': extra_name, 'arm': extra_arm, 'path': str(Path(extra_path).resolve()),
                 'selection': 'Highest full-cohort group-weighted OOF BA among complete G1 candidates; exact tie by method lexicographic name. Adaptive development comparison, not selection-independent inference.'}
    config = {'experiment': 'G2_ranking_calibration', 'status': 'frozen_before_computation',
        'created_utc': now(), 'method_order': list(scores), 'extra_G1best': extra,
        'n': len(y), 'n_groups': len(full_sizes), 'n_boot': 5000, 'seed': 20261018,
        'ks': list(KS), 'coverage_material_grid': list(COVERAGES), 'calibration_bin_edges': np.linspace(0,1,11).tolist(),
        'calibration_bins': 'left-inclusive, right-exclusive; final bin includes1; np.minimum(floor(10*p),9)',
        'confidence': 'float(max(Decimal(str(float(p))),1-Decimal(str(float(p))))), decimal precision50; original p unchanged',
        'confidence_numerical_fix': 'Binary subtraction can split mathematically equal p=.82/.18 confidence into .82/.8200000000000001. Exact decimal complement removes only arithmetic artifacts, without rounding different input p or consulting labels/performance. This complete recomputation supersedes all preliminary G2 statistics.',
        'classification_threshold': .5,
        'ranking_tie_policy': 'Fractional boundary bucket; every tied record selected by same fraction; no label or ID tie-break. Unweighted precision is expected uniform tie precision; weighted precision is ratio of fractional weighted positive and selected counts, not expectation of a random ratio.',
        'bootstrap': '5000 observed-component multinomial draws, uniform groups with replacement; same draws across methods, both weights and paired differences. In every draw rebuild material counts and recalculate top-k/coverage boundary. Paired CIs use the finite intersection of the same draw; never refill invalid draws.',
        'weighting': 'group_weighted=1/full original component size before subsetting, repeated for each sampled cluster copy; unweighted=1/material copy',
        'k_unavailable': 'If k>original subset n: point/CI all NA. Otherwise each bootstrap draw with resampled n<k is NA and counted.',
        'coverage': 'q targets q*resampled material n, with a fractional boundary tie; actual inverse-group-weighted coverage is separately reported and generally differs from q.',
        'calibration': 'Unmodified probabilities. No fitting or probability recalibration. ECE=sum_bin(abs(weighted positives-weighted p))/total weight; Brier=sum(w*(p-y)^2)/sum(w).',
        'family_definitions': family_definition, 'group_bins': ['1','2-5','6-20','>20'],
        'probabilities_status': 'Frozen original forced decisions, including explicit abstentions; no outcome imputation or model refitting.',
        'development_only': True, 'source_sha256': {str(p.resolve()): sha(p) for p in paths}}
    write_json(OUT/'config.json', config)
    safe_path(OUT/'run.log').write_text('', encoding='utf-8')
    log('frozen plan; no LLM calls, no training, no network; source identities validated')
    integrity_check = self_check()
    log(f'expanded-cluster oracle: {integrity_check}')
    ppath = safe_path(OUT/'predictions.csv')
    ppath.write_text('id,fold,group,arm,status,score,y_pred\n', encoding='utf-8')
    writer = PredictionWriter(ppath, {'id': ids, 'fold': folds, 'group': groups})
    for method, probabilities in scores.items():
        writer.append(method, range(len(y)), probabilities, ['frozen_existing_probability']*len(y))
    subsets = {'all': np.ones(len(y), dtype=bool), 'exclude_93': groups != '93', 'only_93': groups == '93'}
    bins = {'1': sizes == 1, '2-5': (sizes >= 2) & (sizes <= 5), '6-20': (sizes >= 6) & (sizes <= 20), '>20': sizes > 20}
    cache = {}
    def analyze(mask, label):
        key = mask.tobytes()
        if key not in cache:
            t = time.perf_counter()
            cache[key] = summarize(y, scores, groups, full_sizes, mask)
            log(f'{label}: n={int(mask.sum())}, groups={cache[key]["n_groups"]}; {time.perf_counter()-t:.3f}s')
        return cache[key]
    metrics = {'status': 'running', 'development_only': True, 'integrity_check': integrity_check,
        'n_boot': 5000, 'seed': 20261018, 'method_order': list(scores), 'extra_G1best': extra,
        'subsets': {name: analyze(mask, f'subset/{name}') for name, mask in subsets.items()}}
    write_json(OUT/'metrics.json', metrics)
    metrics['families'] = {family: {subset: analyze(mask & subsets[subset], f'family/{family}/{subset}')
        for subset in subsets} for family, mask in families.items()}
    metrics['group_size_bins'] = {label: {subset: analyze(mask & subsets[subset], f'size/{label}/{subset}')
        for subset in subsets} for label, mask in bins.items()}
    metrics['status'] = 'complete'
    metrics['elapsed_seconds'] = time.perf_counter()-started
    metrics['finished_utc'] = now()
    metrics['report_markdown'] = report(metrics)
    write_json(OUT/'metrics.json', metrics)
    config.update(status='complete', finished_utc=metrics['finished_utc'], elapsed_seconds=metrics['elapsed_seconds'])
    write_json(OUT/'config.json', config)
    readme = ['# G2 ranking, calibration and selective prediction',
        'Development reanalysis of frozen F3 probabilities; no model calls or fitting.',
        'Methods: '+', '.join(scores)+'.',
        'Complete common3764 source-order records; source hashes in config.json.',
        'Both inverse-original-group and material weighting; all/exclude93/only93.',
        'Five frozen families and four original-group-size bins; n/groups/positives explicit.',
        '5000 shared group bootstrap draws; finite paired-draw intersection; undefined draws counted.',
        'Recompute top-k and coverage on each resampled cohort, including repeated groups.',
        'Precision@k=50,100,200,500; k always material count, k>subset n allNA.',
        'Every boundary tie is selected fractionally without outcomes or IDs.',
        'Weighted precision is ratio of fractional counts, not expected random weighted-ratio precision.',
        'Probability histograms and boundary score/tie size/selection fraction retained.',
        'ECE uses fixed10 equal-width bins; Brier and calibration-bin curves retained.',
        'Confidence=max(p,1-p) via exact decimal complement; material coverage0.1..1; weighted coverage retained.',
        'No probability recalibration; threshold0.5 retains original explicit abstentions.',
        'Only93 is one group: its bootstrap cannot estimate between-group uncertainty.',
        'G1best selection uses complete-coverage G1 OOF weightedBA; adaptive development comparison.',
        'Full curve data and paired differences in metrics.json; main tables in report_markdown.',
        'Expanded-cluster oracle72 checks and decimal-confidence tie oracle4 checks; CSVfsync every10.',
        'Old experiment directories read-only; finite analysis stops on completion.']
    safe_path(OUT/'README.md').write_text('\n'.join(readme)+'\n', encoding='utf-8')
    log(f'complete: {len(scores)} methods, {len(cache)} distinct strata, elapsed={metrics["elapsed_seconds"]:.3f}s')
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--extra-predictions')
    parser.add_argument('--extra-arm')
    parser.add_argument('--extra-name', default='G1best')
    args = parser.parse_args()
    kernel = ctypes.windll.kernel32
    kernel.SetThreadExecutionState(0x80000001)
    try:
        run(args.extra_predictions, args.extra_arm, args.extra_name)
    finally:
        kernel.SetThreadExecutionState(0x80000000)
