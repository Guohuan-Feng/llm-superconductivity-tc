"""Finite no-LLM G4 family controls, using frozen G3/F3 outputs only."""
from __future__ import annotations

from datetime import datetime, timezone
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import time

from common import ROOT, DATA, np, safe_path, write_json, PredictionWriter
from f_stats import summarize
from f_report import _classification_section
from g_stage2_family_stats import (G3_FAMILY_VALUES, FIXED_POSITIVE_FAMILIES,
    INTERNAL_FAMILIES, g3_family_masks, load_g3_family_labels, fixed_family_rule)

OUT = ROOT/'G4_family_controls'
RULE = 'G3_family_rule_fixed'
RULE_METHODS = (RULE, 'HGB127', 'A3_forced')
INTERNAL_METHODS = ('A3_forced', 'A2_forced', 'HGB127')
N_BOOT, SEED = 5000, 20261018


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def log(message):
    with safe_path(OUT/'run.log').open('a', encoding='utf-8') as stream:
        stream.write(now()+' '+message+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(message, flush=True)


def load_inputs():
    with np.load(DATA, allow_pickle=False) as source:
        ids = source['material_id'].astype(str)
        y = source['label'].copy()
        groups = source['component_id'].astype(str)
        fold = source['fold'].copy()
        assert np.array_equal(y, source['target_K'] > 10)
    unique, inverse, sizes = np.unique(groups, return_inverse=True, return_counts=True)
    assert len(ids) == 3764 and len(unique) == 1117
    full_sizes = dict(zip(unique, sizes.astype(int)))
    g3config = json.loads((ROOT/'G3_complementarity/config.json').read_text(encoding='utf-8'))
    assert g3config['status'] == 'complete'
    assert np.array_equal(sizes[inverse], g3config['group_size_values'])
    ids_to_index = {mid: i for i, mid in enumerate(ids)}
    scores = {method: np.full(len(ids), np.nan) for method in INTERNAL_METHODS}
    for method in scores:
        assert len(g3config['scores'][method.replace('_forced', '')]) == len(ids)
    with (ROOT/'F3_full3764/predictions.csv').open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            method = row['arm']
            if method not in scores:
                continue
            i = ids_to_index[row['id']]
            assert not np.isfinite(scores[method][i])
            assert row['group'] == groups[i] and int(row['fold']) == fold[i]
            scores[method][i] = float(row['score'])
    assert all(np.isfinite(values).all() for values in scores.values())
    assert all(np.array_equal(values, g3config['scores'][method.replace('_forced', '')]) for method, values in scores.items())
    labels = load_g3_family_labels(ids)
    assert all(g3config['family_results'][mid] == label for mid, label in zip(ids, labels))
    masks = g3_family_masks(ids, labels=labels)
    original_masks = {label: np.asarray(mask, dtype=bool) for label, mask in g3config['family_masks'].items()}
    assert np.all(np.sum(np.stack(list(original_masks.values())), axis=0) == 1)
    scores[RULE] = fixed_family_rule(labels)
    return {'id': ids, 'y': y, 'group': groups, 'fold': fold, 'group_size': sizes[inverse],
            'full_group_sizes': full_sizes, 'labels': labels, 'G3_masks': masks,
            'F3_masks': original_masks, 'scores': scores}


def build_tables(metrics):
    lines = ['# G4家族标签补充对照（阶段2，开发结果）', '',
        '仅复用原F3强制概率和G3只给化学式的家族预测；没有新模型调用、训练、调参或联网。',
        '原A3/A2使用化学式加结构摘要；G3标签不是实验家族真值，其中“不是超导体”可携带LLM先验。',
        '固定规则：G3标签属于{铜基、铁基、A15、镍基、富勒烯}判正，其余判负。名单事前冻结，未按结果增删。',
        '每项统计两种权重，BA/精确率/召回率/准确率；所有CI为5000次共享component组重采样95% percentile区间。',
        '组权重始终为1/原始完整组大小；子集后不重新归一组大小。不加权也按组bootstrap。',
        '只有一个组时CI退化，不能估计组间不确定性；无正/负类时BA未定义。没有预测正类时精确率按冻结F系列约定记0。',
        '家族标签内的条件统计不能证明标签真实、材料新颖、记忆贡献或因果机制；全部仍是开发数据。', '']
    sections = [('固定家族规则：主比较', metrics['fixed_rule']['subsets'], RULE_METHODS)]
    sections += [('固定规则·原F3划分·'+label, table, RULE_METHODS)
                 for label, table in metrics['fixed_rule']['original_F3_families'].items()]
    sections += [('固定规则·G3预测家族·'+label, table, RULE_METHODS)
                 for label, table in metrics['fixed_rule']['G3_predicted_families'].items()]
    sections += [('固定规则·原组大小·'+label, table, RULE_METHODS)
                 for label, table in metrics['fixed_rule']['group_size_bins'].items()]
    sections += [('G3预测标签内部·'+label, table, INTERNAL_METHODS)
                 for label, table in metrics['label_internal'].items()]
    for title, tables, methods in sections:
        for subset, label in (('all', '全部'), ('exclude_93', '去掉93'), ('only_93', '只看93')):
            value = tables[subset]
            lines.append(f'\n正类n={value["n_positive"]}（分布于{value["n_positive_groups"]}组），负类n={value["n_negative"]}（分布于{value["n_negative_groups"]}组）；同一组可同时含正负类。')
            if value['sparse_warning']:
                lines.append(value['sparse_warning'])
            _classification_section(lines, title+'：'+label, value, list(methods), [tuple(p) for p in value['report_pairs']])
    return '\n'.join(lines)+'\n'


def run():
    started = time.perf_counter()
    for parent in (OUT, *OUT.parents):
        if parent.exists() and (parent.is_symlink() or parent.is_junction()):
            raise ValueError('Linked output destination rejected: '+str(parent))
    safe_path(OUT).mkdir(exist_ok=True)
    paths = [DATA, ROOT/'F3_full3764/predictions.csv', ROOT/'G3_complementarity/config.json',
             ROOT/'G3_complementarity/predictions.csv', ROOT/'g_stage2_family_analysis.py',
             ROOT/'g_stage2_family_stats.py', ROOT/'f_stats.py', ROOT/'f_report.py', ROOT/'common.py']
    config = {'experiment': 'G4_family_controls', 'stage': 2, 'status': 'frozen_before_statistics',
        'created_utc': now(), 'fixed_positive_families': list(FIXED_POSITIVE_FAMILIES),
        'fixed_rule': 'G3 predicted label in exactly the five predeclared positive families ->1; every other allowed label->0',
        'whitelist_policy': 'Frozen from explicit user request before calculating any outcomes; no data-dependent editing or threshold tuning.',
        'G3_family_vocabulary': list(G3_FAMILY_VALUES), 'internal_family_labels': list(INTERNAL_FAMILIES),
        'rule_methods': list(RULE_METHODS), 'internal_methods': list(INTERNAL_METHODS),
        'subsets': ['all','exclude_93','only_93'], 'n_boot': N_BOOT, 'seed': SEED,
        'weights': '1/full original group size; unweighted1/material; retained after all masks',
        'bootstrap': 'uniform observed groups with replacement; same draws for methods/weights/paired differences within each stratum; undefined draws excluded and counted; never refill draws',
        'positive_target': 'experimental label target_K>10K from accepted_train.npz',
        'model_input_boundary': 'F3 A2/A3 original inputs included proxy formula and structure. G3 family labels were generated with formula only and no Tc question; these are model predictions, not experimental family ground truth.',
        'sparse_warning_rule': 'n_groups<10 or either true class has<10 materials or<10 observed groups containing that class; descriptive warning only, never exclusion or model/rule selection',
        'statistical_conventions': 'thresholdp>=0.5; no predicted positives precision0; true-class absence BA undefined; no positives recall undefined; empty allNA; one group CI degenerate',
        'G4_G5_mask_API': {'module': 'g_stage2_family_stats', 'load_g3_family_labels': 'load_g3_family_labels(ids, source=None) -> aligned label array',
            'g3_family_masks': 'g3_family_masks(ids, labels=None, source=None) -> dict of all12 Boolean masks; with labels supplied pure in-memory; otherwise read-only G3 CSV',
            'import_side_effects': 'no experiment reads/writes, fitting, model calls or network'},
        'ML_fits': 0, 'LLM_calls': 0, 'network_requests': 0, 'development_only': True,
        'source_sha256': {str(path.resolve()): sha(path) for path in paths}}
    write_json(OUT/'config.json', config)
    safe_path(OUT/'run.log').write_text('', encoding='utf-8')
    log('Rule whitelist/protocol/source hashes frozen before loading outcomes for statistics.')
    d = load_inputs()
    writer_path = safe_path(OUT/'predictions.csv')
    writer_path.write_text('id,fold,group,arm,status,score,y_pred\n', encoding='utf-8')
    writer = PredictionWriter(writer_path, d)
    for method in (*RULE_METHODS, 'A2_forced'):
        writer.append(method, range(len(d['id'])), d['scores'][method],
                      ['fixed_five_G3_labels']*len(d['id']) if method == RULE else ['frozen_F3_forced_probability']*len(d['id']))
    subsets = {'all': np.ones(len(d['id']), dtype=bool), 'exclude_93': d['group'] != '93', 'only_93': d['group'] == '93'}
    s = d['group_size']
    bins = {'1':s==1, '2-5':(s>=2)&(s<=5), '6-20':(s>=6)&(s<=20), '>20':s>20}
    cache = {}
    def analyze(mask, methods, label):
        key = (tuple(methods), mask.tobytes())
        if key not in cache:
            result = summarize(d['y'], {method:d['scores'][method] for method in methods},
                               d['group'], d['full_group_sizes'], mask, n_boot=N_BOOT, seed=SEED)
            result['report_pairs'] = [[methods[i],methods[j]] for i in range(len(methods)) for j in range(i+1,len(methods))]
            result['n_positive_groups'] = int(len(np.unique(d['group'][mask & (d['y'] == 1)])))
            result['n_negative_groups'] = int(len(np.unique(d['group'][mask & (d['y'] == 0)])))
            sparse = min(result['n_groups'], result['n_positive'], result['n_negative'], result['n_positive_groups'], result['n_negative_groups']) < 10
            result['sparse_warning'] = '稀疏或单类子集：总组数、某真实类别材料数或包含该类别的组数少于10，CI可能很宽、未定义或退化；不能据此下稳定的家族泛化结论。' if sparse else None
            cache[key] = result
            log(f'{label}: n={result["n"]}, groups={result["n_groups"]}, positives={result["n_positive"]}')
        return cache[key]
    def by_subset(mask, methods, label):
        return {subset:analyze(mask & smask, methods, label+'/'+subset) for subset,smask in subsets.items()}
    metrics = {'status':'running', 'experiment':'G4_family_controls', 'development_only':True,
        'fixed_positive_families':list(FIXED_POSITIVE_FAMILIES), 'n_boot':N_BOOT,'seed':SEED,
        'n':len(d['id']), 'n_groups':len(d['full_group_sizes']), 'n_positive':int(d['y'].sum()),
        'ML_fits':0,'LLM_calls':0,'network_requests':0,
        'input_boundary':config['model_input_boundary'], 'fixed_rule': {
            'method_order':list(RULE_METHODS),
            'subsets':by_subset(subsets['all'],RULE_METHODS,'rule/main'),
            'original_F3_families':{label:by_subset(mask,RULE_METHODS,'rule/F3/'+label) for label,mask in d['F3_masks'].items()},
            'G3_predicted_families':{label:by_subset(mask,RULE_METHODS,'rule/G3/'+label) for label,mask in d['G3_masks'].items()},
            'group_size_bins':{label:by_subset(mask,RULE_METHODS,'rule/size/'+label) for label,mask in bins.items()}},
        'label_internal':{label:by_subset(d['G3_masks'][label],INTERNAL_METHODS,'internal/'+label) for label in INTERNAL_FAMILIES},
        'family_membership_counts': {label:{'n':int(mask.sum()),'n_groups':int(len(np.unique(d['group'][mask]))),
            'n_positive':int(d['y'][mask].sum()), 'n_positive_groups':int(len(np.unique(d['group'][mask & (d['y']==1)]))),
            'n_negative_groups':int(len(np.unique(d['group'][mask & (d['y']==0)]))),
            'rule_prediction':int(label in FIXED_POSITIVE_FAMILIES)} for label,mask in d['G3_masks'].items()}}
    metrics['report_markdown'] = build_tables(metrics)
    metrics.update(status='complete',finished_utc=now(),elapsed_seconds=time.perf_counter()-started,
                   distinct_cached_statistics=len(cache),logical_strata=78)
    write_json(OUT/'metrics.json', metrics)
    config.update(status='complete',finished_utc=metrics['finished_utc'],elapsed_seconds=metrics['elapsed_seconds'])
    write_json(OUT/'config.json',config)
    readme = ['# G4 family controls: stage2 development reanalysis',
        'No new LLM calls, ML fits, tuning or network requests; old experiments read-only.',
        'Fixed positive whitelist: 铜基,铁基,A15,镍基,富勒烯; all other G3 labels negative.',
        'Whitelist, protocol and sourceSHA frozen in config before outcome statistics.',
        'Original3764 records/1117groups, target experimentalTc>10K; original folds unchanged.',
        'F3 A3/A2 were formula+structure; frozen forced probabilities include explicit abstentions.',
        'G3 labels are formula-only LLM predictions, never verified family ground truth.',
        'Rule compared with HGB/A3; originalF3 families, all12G3 labels and group sizes retained.',
        'Within predicted铜基/铁基/A15/硼碳化物: A3,A2,HGB four-metric comparisons.',
        'Both inverse-full-original-group and material weights; all/exclude93/only93.',
        'BA/precision/recall/accuracy and every paired difference with5000shared group draws.',
        'Point estimates pool weighted confusion counts; no fold averaging or postmask reweighting.',
        'No predicted positives precision0; no true class BAundefined; no positives recallundefined.',
        'Empty allNA; one-group CIdegenerate; finite/undefined bootstrap draws counted.',
        'Sparse warnings if groups, either class materials or class-containing groups<10; no exclusion.',
        'Family subgroup statistics are conditional development evidence, not family/novelty validation.',
        'Reusable read-only G3 label/mask API: g_stage2_family_stats.py; arbitrary uniqueID subsets.',
        'predictions.csv copies frozen probabilities plus the fixed rule; flush/fsync every10.',
        'Complete78logical strata and paired tables: metrics.json/report_markdown.',
        'Finite analysis stops at completion; no route selection or later-stage execution.']
    safe_path(OUT/'README.md').write_text('\n'.join(readme)+'\n',encoding='utf-8')
    log(f'complete: {metrics["logical_strata"]} logical strata/{len(cache)} distinct; {metrics["elapsed_seconds"]:.3f}s')
    return metrics


if __name__ == '__main__':
    kernel = ctypes.windll.kernel32
    assert kernel.SetThreadExecutionState(0x80000001)
    try:
        run()
    finally:
        kernel.SetThreadExecutionState(0x80000000)
