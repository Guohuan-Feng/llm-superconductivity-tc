"""G4 paired group statistics on800/fixed300/new500; no model fits or calls."""
from __future__ import annotations
from common import np
from f3_stats import build_f3_statistics
from f_stats import summarize, fraction_summary
from f_report import _classification_section
from g4_inputs import SEED, ARMS, FAMILIES

SUBSETS = (('all', '全部'), ('exclude_93', '去掉93'), ('only_93', '仅93'))


def statistics(cfg, d, results):
    indices = np.asarray(cfg['selected_indices'], int)
    y, groups, sizes = d['y'][indices], d['group'][indices], d['group_size'][indices]
    n = len(indices)
    original = {str(g): int(s) for g, s in zip(d['group'], d['group_size'])}
    scores = {name: np.full(n, np.nan) for name in ARMS}
    scores['A3_original'] = np.asarray([float(cfg['baseline_A3_rows'][case['id']]['score']) for case in cfg['cases']])
    position = {case['id']: j for j, case in enumerate(cfg['cases'])}
    for arm in ARMS:
        for identity, result in results.get(arm, {}).items():
            if result.get('response') is not None:
                scores[arm][position[identity]] = result['response']['p']
    families = {k: np.asarray(v, bool) for k, v in cfg['family_masks'].items()}
    a15 = np.asarray(cfg['raw_a15_mask'], bool)
    labels = np.asarray([case['G3_family'] for case in cfg['cases']])
    cohorts = {'all800': np.ones(n, bool),
               'fixed300': np.asarray([case['cohort'] == 'fixed300' for case in cfg['cases']]),
               'new500': np.asarray([case['cohort'] == 'new500' for case in cfg['cases']])}
    output = {'cohorts': {}, 'protocol': {'n_boot': 5000, 'seed': SEED,
        'weightings': '1/full-original3764 component size; unit-material',
        'bootstrap': 'Observed component groups; same sorted groups/seed/draw blocks within each stratum share all classification methods, paired differences, both weights and agreement fractions.',
        'development_only': True,
        'sampling_boundary': cfg['sampling_interpretation'],
        'P_boundary': cfg['P_identity_boundary'],
        'forced_decision': 'p>=0.5 even when model abstains; no HGB fallback is inserted.',
        'missing': 'Only common valid E/F/P/originalA3 rows enter comparisons; absent or failed results are never filled.'}}
    lines = ['# G4 输入消融：开发结果', '',
        '固定300与按G3家族×原组大小、不看真标签抽出的500材料，分别及合并报告。',
        'E仅元素集合；F仅原proxy公式；P保留原组成/结构数值，仅乱序/同比例改写formula并按用户授权更正formula_semantics。',
        'P的组成表仍能重构原式，因此不是隐去身份试验。原A3包含公式、组成和结构摘要。',
        'BA/P/R/Acc用强制p>=0.5；组权重=1/原3764的完整component大小；不加权也按组bootstrap5000次。',
        '每个层内所有方法、两权重、配对差值以及一致率共享相同组重采样；单组CI退化，空/单类BA未定义。',
        '总体800为固定样本加最低配额分层开发样本，未做抽样概率反权重，不能直接当全3764无偏总体估计。', '']
    for cohort, cohort_mask in cohorts.items():
        selected = np.flatnonzero(cohort_mask)
        local_scores = {name: p[selected] for name, p in scores.items()}
        local_groups, local_y = groups[selected], y[selected]
        stats = build_f3_statistics(local_y, local_scores, local_groups, original,
            family_masks={k: v[selected] for k, v in families.items()},
            a15_mask=a15[selected], group_size_values=sizes[selected], n_boot=5000, seed=SEED)
        common = np.logical_and.reduce([np.isfinite(p) for p in local_scores.values()])
        base_masks = {'all': np.ones(len(selected), bool), 'exclude_93': local_groups != '93', 'only_93': local_groups == '93'}
        stats['G3_predicted_family'] = {}
        for label in FAMILIES:
            label_mask = labels[selected] == label
            tables = {}
            for subset, _ in SUBSETS:
                requested = label_mask & base_masks[subset]
                table = summarize(local_y, local_scores, local_groups, original,
                    mask=requested & common, n_boot=5000, seed=SEED)
                table.update(n_requested=int(requested.sum()),
                    n_requested_groups=int(len(np.unique(local_groups[requested]))),
                    n_missing=int((requested & ~common).sum()))
                tables[subset] = table
            stats['G3_predicted_family'][label] = tables
        agreement_values = {}
        for arm in ARMS:
            values = np.full(len(selected), np.nan)
            values[common] = ((local_scores[arm][common] >= .5) == (local_scores['A3_original'][common] >= .5)).astype(float)
            agreement_values[arm] = values
        local_sizes = sizes[selected]
        size_masks = {'1': local_sizes == 1, '2-5': (local_sizes >= 2) & (local_sizes <= 5),
                      '6-20': (local_sizes >= 6) & (local_sizes <= 20), '>20': local_sizes > 20}
        decorator_sets = [(stats['subsets'], np.ones(len(selected), bool))]
        decorator_sets += [(tables, families[name][selected]) for name, tables in stats['families'].items()]
        decorator_sets += [(tables, size_masks[name]) for name, tables in stats['group_size_bins'].items()]
        decorator_sets += [(tables, labels[selected] == label) for label, tables in stats['G3_predicted_family'].items()]
        for tables, stratum in decorator_sets:
            for subset, _ in SUBSETS:
                requested = stratum & base_masks[subset]
                table = tables[subset]
                table.update(n_positive_groups_requested=int(len(np.unique(local_groups[requested & (local_y == 1)]))),
                    n_negative_groups_requested=int(len(np.unique(local_groups[requested & (local_y == 0)]))),
                    n_positive_groups=int(len(np.unique(local_groups[requested & common & (local_y == 1)]))),
                    n_negative_groups=int(len(np.unique(local_groups[requested & common & (local_y == 0)]))))
                table['agreement_with_original_A3'] = {arm: fraction_summary(values, local_groups, original,
                    mask=requested, n_boot=5000, seed=SEED) for arm, values in agreement_values.items()}
        stats['agreement_with_original_A3'] = {subset: table['agreement_with_original_A3'] for subset, table in stats['subsets'].items()}
        collections = [('主比较', stats['subsets'])]
        collections += [('F3家族·' + k, v) for k, v in stats['families'].items()]
        collections += [('原组大小·' + k, v) for k, v in stats['group_size_bins'].items()]
        collections += [('G3预测家族·' + k, v) for k, v in stats['G3_predicted_family'].items()]
        for title, collection in collections:
            for subset, subset_label in SUBSETS:
                table = collection[subset]
                lines += [f'请求n={table["n_requested"]}/{table["n_requested_groups"]}组；共同有效n={table["n"]}/{table["n_groups"]}组；缺失n={table["n_missing"]}。', '']
                lines += [f'正类支持组={table["n_positive_groups"]}，负类支持组={table["n_negative_groups"]}；同组可同时含两类，支持组数不可简单相加。', '']
                _classification_section(lines, cohort + '·' + title + '：' + subset_label,
                    table, list(scores), [tuple(pair) for pair in stats['report_pairs']])
        lines += ['### ' + cohort + '·与原A3二元判断一致率', '',
                  '|子集|方法|n/组|组加权一致率[95% CI]|不加权一致率[95% CI]|',
                  '|---|---|---:|---:|---:|']
        for subset, subset_label in SUBSETS:
            for arm in ARMS:
                row = stats['agreement_with_original_A3'][subset][arm]
                def fmt(v):
                    p, ci = v['point'], v['ci95']
                    return 'NA' if p is None else f'{100*p:.2f} [{100*ci[0]:.2f},{100*ci[1]:.2f}]'
                lines.append(f'|{subset_label}|{arm}|{row["n"]}/{row["n_groups"]}|{fmt(row["group_weighted"])}|{fmt(row["unweighted"])}|')
        lines.append('')
        lines += ['### ' + cohort + '·G3预测家族分层的一致率', '',
                  '|G3家族|子集|方法|n/组|组加权一致率[95% CI]|不加权一致率[95% CI]|',
                  '|---|---|---|---:|---:|---:|']
        for label, tables in stats['G3_predicted_family'].items():
            for subset, subset_label in SUBSETS:
                for arm, row in tables[subset]['agreement_with_original_A3'].items():
                    lines.append(f'|{label}|{subset_label}|{arm}|{row["n"]}/{row["n_groups"]}|{fmt(row["group_weighted"])}|{fmt(row["unweighted"])}|')
        lines.append('')
        output['cohorts'][cohort] = stats
    output['report_markdown'] = '\n'.join(lines).rstrip() + '\n'
    return output
