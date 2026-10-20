"""Offline stage-2 tables from existing metrics only; never fit/call a model."""
from __future__ import annotations

import argparse
import hashlib
import json

from g_stage1_export import ROOT, safe, inline, table, interval, markdown_tables

SUBSETS = (('all', '全部'), ('exclude_93', '去掉 component 93'), ('only_93', '仅 component 93'))
DIRECTORIES = ('G4_family_controls', 'G4_input_ablation', 'G5_tc_multithreshold')
ACTIVE_STATUSES = {'running', 'running_A3_reg', 'preparing_ML', 'statistical_finalization'}


def validate_statuses(folder, config, metrics, allow_partial):
    states = (config['status'], metrics['status'])
    assert not any(s in ACTIVE_STATUSES or s.startswith('running') for s in states), (folder, states, 'experiment active')
    if not allow_partial:
        assert states == ('complete', 'complete'), (folder, states)
    elif metrics.get('report_markdown'):
        assert states[0] == states[1], (folder, states, 'final status mismatch')


def coverage_banner(folder, metrics, config):
    if config['status'] == metrics['status'] == 'complete':
        return ''
    counts = []
    if 'cohorts' in metrics:
        s = metrics['cohorts']['all800']['subsets']['all']
        counts.append(f'共同有效材料 {s["n"]}/{s["n_requested"]}，有效组 {s["n_groups"]}/{s["n_requested_groups"]}')
    if 'coverage' in metrics:
        c = metrics['coverage']
        counts.append(f'有效 LLM 材料 {c["valid_LLM"]}/{c["requested_n"]}')
    detail = '；'.join(counts) or '尚无完整比较'
    return '<div class="note"><strong>本项未全部完成</strong><p>'+inline(folder+'：'+config['status']+'；'+detail)+'。下表仅描述已取得的共同有效结果，不能视为完整实验结论。</p></div>'


def raw_interval(value):
    if value['point'] is None:
        result = 'NA'
    else:
        low, high = value['ci95']
        result = f'{value["point"]:.3f}'
        result += f' [{low:.3f}, {high:.3f}]' if low is not None else ' [NA]'
    if value.get('n_undefined_boot', 0):
        result += f' (有效 {value["n_valid_boot"]}/5000)'
    return result


def classification_tables(subsets, title, opened=False):
    output = ['<details'+(' open' if opened else '')+'><summary>'+inline(title)+'</summary>']
    for key, label in SUBSETS:
        stats = subsets[key]
        output.append(f'<h3>{label} · n={stats["n"]} / {stats["n_groups"]} 组 / 正类 {stats["n_positive"]}</h3>')
        if 'n_requested' in stats:
            output.append(f'<p>请求 n={stats["n_requested"]} / {stats["n_requested_groups"]} 组；共同有效 n={stats["n"]} / {stats["n_groups"]} 组；缺失 {stats["n_missing"]}。</p>')
        output.append('<p>正类支持组 '+str(stats.get('n_positive_groups', '见附表'))+'；负类支持组 '+str(stats.get('n_negative_groups', '见附表'))+'。同组可同时含两类。</p>')
        if stats.get('single_group_warning'):
            output.append('<p>'+inline(stats['single_group_warning'])+'</p>')
        for scheme, scheme_label in (('group_weighted', '组加权'), ('unweighted', '不加权')):
            rows = [[name]+[interval(values[k]) for k in ('balanced_accuracy', 'precision', 'recall', 'accuracy')]
                    for name, values in stats[scheme]['metrics'].items()]
            output.append('<h4>'+scheme_label+'：% [95% CI]</h4>')
            output.append(table(['方法', 'BA', '精确率', '召回率', '准确率'], rows))
    return '\n'.join(output)+'</details>'


def supplement_overview(metrics):
    output = ['<h2>两项固定家族补充对照</h2>',
              '<p>正类名单固定为 {铜基、铁基、A15、镍基、富勒烯}，未按结果调整。标签是 G3 的模型输出，不作为实验家族真值。A15 与硼碳化物各只有 4 个包含正类的组，需连同宽区间及有效抽样数阅读。</p>',
              classification_tables(metrics['fixed_rule']['subsets'], '固定规则 vs HGB / A3', True)]
    for label, stats in metrics['label_internal'].items():
        output.append(classification_tables(stats, 'G3 预测标签内部：'+label))
    return '\n'.join(output)


def g4_overview(metrics):
    output = ['<h2>G4 输入消融</h2>',
              '<p>E 仅元素集合；F 仅原公式；P 保留组成/结构数值，只改等价公式字符串及获准同步更正的语义句。组成表可重构原式，P 不是隐去身份。原 A3 复用历史调用，因此差值还包含生成波动及无法锁定的后端变化，没有同输入的新 A3 重复调用来分离这些因素。</p>',
              '<p>固定 300 与分层新抽 500 都来自已有数据；合并 800 的分层最低配额未作抽样概率校正，不能作为全 3764 的无偏估计。</p>']
    for cohort, stats in metrics['cohorts'].items():
        output.append(classification_tables(stats['subsets'], cohort, cohort == 'all800'))
        rows = []
        for subset, label in SUBSETS:
            for arm, value in stats['agreement_with_original_A3'][subset].items():
                rows.append([label, arm, value['n'], value['n_groups'], interval(value['group_weighted']), interval(value['unweighted'])])
        output.append('<details><summary>'+cohort+'：与历史 A3 的判断一致率</summary>'+table(['子集', '方法', 'n', '组数', '组加权 % [CI]', '不加权 % [CI]'], rows)+'</details>')
    return '\n'.join(output)


def g5_overview(metrics):
    output = ['<h2>G5 Tc 估计与多阈值</h2>',
              '<p>实测 Tc 严格大于 5/10/15 K 才为相应正类。直接概率用 p≥0.5；点估计用 Tc_est&gt;阈值，两种输出单列。输入的科学信息及数值精度与历史 A3 相同，任务与输出契约改变。</p>']
    for threshold, stats in metrics['classification'].items():
        output.append(classification_tables(stats['subsets'], threshold, threshold == 'Tc_gt_10K'))
    output.append('<h3>回归主表：MAE 单位 K；Spearman 为相关系数</h3>')
    for subset, label in SUBSETS:
        for population in ('all_Tc', 'Tc_gt_10K'):
            stats = metrics['regression']['subsets']['cohort'][subset][population]
            rows = []
            for method in stats['group_weighted']['metrics']:
                row = [method]
                for scheme in ('group_weighted', 'unweighted'):
                    row += [raw_interval(stats[scheme]['metrics'][method][key]) for key in ('MAE_K', 'Spearman')]
                rows.append(row)
            output.append(f'<details><summary>{label} / {population} · n={stats["n"]} / {stats["n_groups"]} 组</summary>'+table(['方法', '组加权 MAE K [CI]', '组加权 Spearman [CI]', '不加权 MAE K [CI]', '不加权 Spearman [CI]'], rows)+'</details>')
    return '\n'.join(output)


def export(allow_partial=False):
    loaded = {}
    for folder in DIRECTORIES:
        path = ROOT/folder/'metrics.json'
        metrics = json.loads(path.read_text(encoding='utf-8'))
        config = json.loads((ROOT/folder/'config.json').read_text(encoding='utf-8'))
        validate_statuses(folder, config, metrics, allow_partial)
        loaded[folder] = (metrics, config, hashlib.sha256(path.read_bytes()).hexdigest())
    body = []
    for folder, title, overview in [('G4_family_controls', '补充家族对照', supplement_overview),
                                    ('G4_input_ablation', 'G4 输入消融', g4_overview),
                                    ('G5_tc_multithreshold', 'G5 多阈值与回归', g5_overview)]:
        metrics, config, sha = loaded[folder]
        body.append(coverage_banner(folder, metrics, config))
        if metrics.get('report_markdown'):
            body.append(overview(metrics))
            body.append('<details><summary>'+title+'：完整分层、配对差值及一致性附表</summary><p class="source">'+folder+'/metrics.json · SHA256 '+sha+'</p>'+markdown_tables(metrics['report_markdown'])+'</details>')
        else:
            body.append('<h2>'+title+'</h2><p>未完成比较，状态：'+inline(config['status'])+'；不得据此作该实验的性能结论。</p>')
    statuses = '; '.join(folder+': '+cfg['status'] for folder, (_, cfg, _) in loaded.items())
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>阶段 2：G4、G5 与家族补充对照</title>
<style>body{max-width:1500px;margin:30px auto;padding:0 24px;font:15px/1.6 "Segoe UI","Microsoft YaHei",sans-serif;color:#172b3a}h1{font-size:28px}h2{font-size:23px}h3{font-size:19px}h4,h5,h6{font-size:16px}details{margin:20px 0;border-top:1px solid #b7cbd8;padding-top:10px}summary{font-size:19px;font-weight:650;cursor:pointer}.note{background:#eef5fb;border-left:4px solid #2378ad;padding:14px 20px}.source{font:12px/1.5 monospace;overflow-wrap:anywhere;color:#546772}.scroll{overflow-x:auto;margin:12px 0 24px}table{border-collapse:collapse;white-space:nowrap;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;border-bottom:1px solid #dbe3e8;padding:7px 10px}th{background:#e9f1f6}tbody tr:nth-child(even){background:#f7fafc}code{background:#f0f2f4;padding:2px 4px}</style>
<h1>阶段 2：G4、G5 与家族补充对照</h1><div class="note"><p>全部为已有材料上的开发结果，未作截止后新材料外部检验。旧实验只读；固定规则名单未调整。</p><p>两种统计使用 1/原始完整组大小及每材料权重 1；所有区间按组重采样 5000 次，配对比较使用共享抽样。空集全部 NA；缺任一真实类别时 BA 为 NA；无正类时召回为 NA；无预测正类时精确率按约定为 0。单组区间退化。G3 家族标签是模型预测，分层属于条件分析。</p></div>
'''+'<p class="source">'+inline(statuses)+'</p>'+'\n'.join(body)+'</html>'
    path = safe(ROOT/'outputs/G_stage2_results.html')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding='utf-8')
    print(json.dumps({'path': str(path), 'bytes': path.stat().st_size,
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-partial', action='store_true')
    export(parser.parse_args().allow_partial)
