"""Offline stage-3 report; no model calls, training, or new scientific choices."""
from __future__ import annotations
import hashlib
import json
import csv
from g_stage1_export import ROOT, safe, inline, table, interval, markdown_tables
from g_stage2_export import classification_tables, raw_interval

FOLDERS = ('G5b_repeatability', 'G6_local_cutoff', 'G7_postcutoff_materials')
WEIGHTS = (('group_weighted', '组加权'), ('unweighted', '不加权'))


def paired_key_rows(subsets, comparisons):
    rows = []
    for subset, stat in subsets.items():
        for a, b in comparisons:
            values = {}
            for scheme, _ in WEIGHTS:
                for pair in stat[scheme]['paired_differences'].values():
                    if {pair['arm_a'], pair['arm_b']} != {a, b}:
                        continue
                    sign = 1 if pair['arm_a'] == a else -1
                    v = dict(pair['balanced_accuracy'])
                    if sign == -1:
                        v['point'] = None if v['point'] is None else -v['point']
                        v['ci95'] = [None, None] if v['ci95'][0] is None else [-v['ci95'][1], -v['ci95'][0]]
                    values[scheme] = v
            if len(values) == 2:
                rows.append([subset, a+' − '+b, stat['n'], stat['n_groups']]
                            + [interval(values[w]) for w, _ in WEIGHTS])
    return table(['子集', '配对 ΔBA', 'n', '组数', '组加权 pp [CI]', '不加权 pp [CI]'], rows)


def local_overview(m):
    main = m['classification']['Tc_gt_10K']['subsets']
    result = ['<h2>G6 本地早截止模型</h2>',
              '<p>Meta Llama 3.1 8B Instruct，Q4_K_M；<a href="https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md">Meta 官方模型卡</a>记载预训练数据截止 2023 年 12 月。模型卡未对所有对齐数据逐项保证同一截止日期；模型容量、训练、对齐和量化也与 GPT 不同。GPT 的全量 A3/A3-reg 复用历史 F3/G5 调用，并非与本地模型同时重跑；G5b 的重复波动应一并考虑。</p>',
              '<p><strong>本地 A3 在本次固定协议下退化：3764 例 p 均为 0，全部判负；A3-reg 同模型同科学输入仍有正类预测。50% BA 及其退化区间只描述这一全负决策，不能代表该模型一般能力，也不能作为知识截止或记忆量的独立测量。原始输出和全部调用均保留，未通过改提示词或选参数来改变结果。</strong></p>',
              classification_tables(main, '10 K 主比较：四项分类指标', True),
              '<h3>10 K 主要配对比较</h3>',
              paired_key_rows(main, [('local_A3', 'gpt_A3'), ('local_A3', 'HGB127'),
                                     ('local_A3', 'StrongML'), ('local_A3_reg_direct', 'gpt_A3_reg_direct'),
                                     ('local_A3', 'local_A3_reg_direct')]),
              classification_tables(m['classification']['fixed300_M1_10K']['subsets'],
                                    '固定 300：隐去身份 M1', False),
              paired_key_rows(m['classification']['fixed300_M1_10K']['subsets'],
                              [('local_A3_M1', 'local_A3'), ('local_A3_M1', 'gpt_A3_M1')])]
    rows = []
    for subset, populations in m['regression']['main']['cohort'].items():
        for pop, stat in populations.items():
            for method in stat['group_weighted']['metrics']:
                rows.append([subset, pop, method, stat['n'], stat['n_groups']]
                            + [raw_interval(stat[w]['metrics'][method][metric])
                               for w, _ in WEIGHTS for metric in ('MAE_K', 'Spearman')])
    result.append('<details><summary>回归主表：MAE 与 Spearman</summary>'
                  + table(['子集', '群体', '方法', 'n', '组数', '组加权 MAE K',
                           '组加权 Spearman', '不加权 MAE K', '不加权 Spearman'], rows)+'</details>')
    return '\n'.join(result)


def external_overview(m):
    if m['status'] != 'complete':
        return ''
    stats = m['primary_external']
    stat = stats['new_all']
    result = ['<h2>G7 截止后材料：严格主分析</h2>',
              '<p>最初冻结并评测 89 例；独立文献复核在未查看预测的情况下，按预定资格规则剔除 16 个截止前已报道的 FCC 样品、2 个公式在理论论文中已出现的样品及 2 个超导宿主归属不确定的样品。完整 89 例仅作审计和敏感性分析，不作为新材料主表。</p>',
              f'<p>主表 n={stat["n"]}，正类 {stat["n_positive"]}，母体组 {stat["n_groups"]}。正类仅来自 2 个镍氧化物母体组，未达到 100 材料的目标；达到 60 材料／15 正类的最低门槛。组加权使用 1/严格分析集内母体组大小。</p>',
              '<p>所有材料均无实际取得的 CIF，因此输入只给化学式；5 项已知结构存档因访问限制未取得，结构基线 n=0。部分精确配比首次日只有可核验区间，整个区间均晚于 2024-02-01。日期门槛基于 Llama 官方预训练截止，并非已证实晚于 GPT 的知识截止；新配比可有模型已知的母体家族，不能据此保证身份绝未出现在全部训练或对齐数据中。</p>']
    for scheme, label in WEIGHTS:
        rows = [[method]+[interval(v[k]) for k in ('balanced_accuracy', 'precision', 'recall', 'accuracy')]
                for method, v in stat[scheme]['metrics'].items()]
        result.append('<h3>'+label+'：% [95% CI]</h3>'
                      +table(['方法', 'BA', '精确率', '召回率', '准确率'], rows))
    result.append(paired_key_rows({'new_all': stat}, [('Local_A3', 'GPT_A3'),
                                                     ('Local_A3', 'HGB_composition118'),
                                                     ('GPT_A3', 'HGB_composition118')]))
    ctl = stats['cohort_change_controls']
    rows = []
    for scheme, label in WEIGHTS:
        for pair, values in ctl['difference_in_differences'][scheme].items():
            rows.append([label, pair]+[interval(values[k]) for k in ('balanced_accuracy', 'precision', 'recall', 'accuracy')])
    result.append('<h3>新旧队列的差中之差：百分点 [CI]</h3>'
                  +f'<p>旧 n={ctl["old_n"]}／{ctl["old_groups"]} 组／正类 {ctl["old_positive"]}；新 n={ctl["new_n"]}／{ctl["new_groups"]} 组／正类 {ctl["new_positive"]}。统计量为 (LLM 新−旧)−(成分 HGB 新−旧)。新旧各自独立按组抽样，队列内部所有方法共享抽样。</p>'
                  +table(['权重', '比较', 'ΔΔBA', 'ΔΔ精确率', 'ΔΔ召回率', 'ΔΔ准确率'], rows))
    result.append('<p><strong>本地 A3 在旧 3764 例及新冻结 89 例均为 p=0，因而新旧 BA 都为 50%。若相对 HGB 的 BA 差中之差为正，它来自常数分类器的 BA 不变及 HGB 下降，不能解释为本地模型具有泛化优势。</strong></p>')
    result.append('<p>旧材料并未全部核实早于截止时间；旧 A3 有结构而新输入仅公式；HGB 旧成绩为每次训练约 2/3 数据的 OOF，新预测用全 3764 条训练。新标签限定常压零外场，但按原方案复用的 A3 提示词并未显式限定压力或磁场；同一公式仍可能对应不同相或测量条件，旧队列条件也未统一复核。家族分布、信息量及模型容量也不同。差中之差是描述性对照，不能完整扣除新材料难度，更不能单独识别记忆的因果贡献。</p>')
    return '\n'.join(result)


def execution_notes(loaded):
    c, m = loaded['G5b_repeatability']
    audit = m.get('terminal_scientific_audit', {})
    calls = m.get('calls', {})
    rows = [['G5b', '有效科学结果', m.get('valid_arm_cases')],
            ['G5b', '实际 CLI 调用', calls.get('completed')],
            ['G5b', '全部初始提示词与原文一致', audit.get('all2400_initial_prompt_bytes_match_original')],
            ['G5b', '工具调用', calls.get('tools_used')],
            ['G5b', '实际最大并发', audit.get('max_concurrency')]]
    c6, m6 = loaded['G6_local_cutoff']
    a6 = c6['final_transcript_audit']
    rows += [['G6', '有效任务 / 实际请求', str(a6['observed_unique_tasks'])+' / '+str(a6['request_records'])],
             ['G6', '格式修复 / 最终失败', str(a6['repair_tasks'])+' / '+str(len(a6['final_failures']))],
             ['G6', '全部输入、输出、CSV、运行时校验', a6['status']],
             ['G7', 'GPT / 本地有效科学结果', '89 / 89']]
    text = '<p>G5b 实际以无时长上限的直接 Python 进程运行，未按要求注册 Windows 计划任务；运行时已防睡眠。科学调用完成后的统计步骤因缺少家族掩码元数据而报错，随后从只读原始配置补齐并重算统计，未重跑有效科学调用。完整代码哈希未在首次启动前冻结，因此提示词一致性依据持久化原文及历史冻结哈希做了事后审计。</p>'
    text += '<p>G6 在产生任何科学输出前有一次连接预检中断，查阅系统日志及任务退出码后修复连接复用，完整重算一次；没有逐条续接旧输出。TaskScheduler Operational 日志当时未启用，不能据此声称该通道无错误。G6、G6 统计及 G7 的有限计划任务均退出 0，并已注销；本地服务已退出。</p>'
    text += '<p>G6 的 GPT 回归对照采用历史 G5 config.results 中逐条验证的 3764 个有效输出。其中 3dsc_mp_row_000186 的旧 CSV 仍是早前 cli_error，配置及调用记录中已有有效修复输出；本轮未修改旧文件。G7 指令文件字节哈希与本地传输文本哈希分开记录，差异核定仅为 CRLF→LF；科学用户提示词逐条相同。</p>'
    return '<details><summary>执行审计与偏差</summary>'+table(['实验', '项目', '值'], rows)+text+'</details>'


def material_audit(c, m):
    label_path = safe(m['evaluation_labels_storage']['path'])
    assert hashlib.sha256(label_path.read_bytes()).hexdigest() == m['evaluation_labels_storage']['sha256']
    labels = json.loads(label_path.read_text(encoding='utf8'))
    with (ROOT/'G7_postcutoff_materials/predictions.csv').open(newline='', encoding='utf8') as f:
        records = list(csv.DictReader(f))
    primary = [r for r in records if r['primary_external_eligible'] == 'true']
    assert len(primary) == m['primary_n'] == 69
    rows = []
    for r in primary:
        v = labels[r['material_id']]
        value = v.get('tc_k')
        if value is None:
            value = '≤'+str(v.get('tc_upper_bound_k'))+'（检测限）'
        dates = v.get('first_material_date_interval') or v.get('first_material_report_date_interval')
        date = str(dates) if dates else r['first_report_date']
        rows.append([r['material_id'], r['formula'], value, v['pressure_gpa'],
                     r['parent_family_key'], date, r['doi_or_arxiv']])
    out = '<details class="materials"><summary>G7 严格 69 例：材料、标签、日期及一手来源</summary><p>以下标签只在评测结束后用于此报告；所有科学提示词均未包含这些标签。日期区间表示首次材料报告与具体配比数值证据的可核验边界，不把区间下界当作精确首次日。</p>'
    out += table(['ID', '化学式', 'Tc K', '压力 GPa', '母体组', '首次日或证据区间', 'DOI / arXiv'], rows)
    cor = m['eligibility_correction']['predeclared_corrections']
    out += '<h3>冻结后独立盲审更正：20 项</h3>'+table(['ID', '化学式', '原因', '更早日期或归属证据', '一手来源'],
        [[v['material_id'], v.get('formula', ''), v['predeclared_reason'],
          v.get('first_true_primary_report_date', v.get('primary_evidence_location', '见审计记录')),
          v.get('first_true_primary_source', '')] for v in cor])+'</details>'
    return out


def repeatability(m):
    result = ['<h2>G5b 同提示词重复与跨提示词比较</h2>',
              '<p>相同 G4 的 800 材料；两次新 A3 与一次新 A3-reg。原提示词及科学输入逐条匹配。原 A3 包含公式、组成和结构。以下判断均用 p≥0.5。800 例为预定分层样本，未作抽样概率校正，不能把其绝对成绩直接外推为全 3764 条成绩。</p>',
              classification_tables(m['subsets'], '三次新调用：四项分类指标', True)]
    rows = []
    for subset, pairs in m['agreement'].items():
        for name, v in pairs.items():
            rows.append([subset, name, v['n'], v['n_groups'], interval(v['group_weighted']), interval(v['unweighted'])])
    result.append('<h3>判断一致率：% [95% CI]</h3>'+table(['子集', '比较', 'n', '组数', '组加权', '不加权'], rows))
    rows = []
    for subset, references in m['exceedance'].items():
        for reference, schemes in references.items():
            for scheme, v in schemes.items():
                if v is None:
                    continue
                value = {'point': v['point_abs_cross_minus_abs_within_BA'],
                         'ci95': v['ci95_abs_cross_minus_abs_within_BA'],
                         'n_valid_boot': v['n_valid_boot'],
                         'n_undefined_boot': v['n_undefined_boot']}
                rows.append([subset, reference, dict(WEIGHTS)[scheme], interval(value),
                             f'{100*v["bootstrap_fraction_abs_A3_vs_reg_exceeds_repeat"]:.2f}',
                             v['n_valid_boot']])
    result.append('<h3>跨提示词差是否超过本轮 A3 重复差</h3><p>统计量为 |BA(参照 A3)−BA(A3-reg)|−|BA(A3 重复 1)−BA(A3 重复 2)|，单位为百分点；用共享组抽样直接求 CI。超越比例是描述性 bootstrap 比例，不是 p 值。仅 93 是单组，CI 退化。</p>'
                  +table(['子集', '参照', '权重', '超出幅度 pp [CI]', 'bootstrap 超越比例 %', '有效/5000'], rows))
    result.append('<p>只有两次本轮 A3 重复，无法估计完整的模型运行方差分布。历史 A3、历史 G5 与本轮调用的后端版本和运行时间无法锁定；历史比较及差中之差是实现的运行差异，不能单独归因为提示词或知识记忆。</p>')
    return '\n'.join(result)


def export():
    loaded = {}
    for folder in FOLDERS:
        p = ROOT/folder
        c = json.loads((p/'config.json').read_text(encoding='utf8'))
        m = json.loads((p/'metrics.json').read_text(encoding='utf8'))
        if folder == 'G7_postcutoff_materials':
            assert c['status'] == m['status'], (folder, c['status'], m['status'])
            assert m['status'] in ('complete', 'stopped_insufficient_pool'), (folder, m['status'])
        else:
            assert c['status'] == m['status'] == 'complete', (folder, c['status'], m['status'])
            assert m.get('report_markdown'), (folder, 'full stratified report missing')
        loaded[folder] = (c, m)
    body = [repeatability(loaded['G5b_repeatability'][1]),
            local_overview(loaded['G6_local_cutoff'][1]),
            external_overview(loaded['G7_postcutoff_materials'][1])]
    for folder, title in [('G5b_repeatability', 'G5b 完整分层、配对差值与历史对照'),
                          ('G6_local_cutoff', 'G6 本地早截止模型：分类、回归及 M1 对照'),
                          ('G7_postcutoff_materials', 'G7 截止后新材料')]:
        c, m = loaded[folder]
        if folder == 'G7_postcutoff_materials' and m['status'] == 'stopped_insufficient_pool':
            body.append('<h2>G7 新材料检验</h2><p>材料收集未达到预先规定的最小样本门槛，因此未进行模型评测，不能报告新材料性能或计算差中之差。</p>')
            counts = [[k, v] for k, v in m.items() if isinstance(v, (str, int, float)) and 'sha' not in k]
            body.append(table(['审计字段', '值'], counts))
        text = m.get('report_markdown', '')
        if text:
            body.append('<details><summary>'+inline(title)+'</summary>'+markdown_tables(text)+'</details>')
        source_rows = []
        for name in ('config.json', 'metrics.json'):
            p = ROOT/folder/name
            source_rows.append([folder+'/'+name, hashlib.sha256(p.read_bytes()).hexdigest()])
        body.append('<details><summary>'+inline(title)+'：来源与审计哈希</summary>'+table(['文件', 'SHA256'], source_rows)+'</details>')
    body.append(execution_notes(loaded))
    body.append(material_audit(*loaded['G7_postcutoff_materials']))
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>阶段 3：G5b、G6、G7</title>
<style>body{max-width:1500px;margin:28px auto;padding:0 24px;font:15px/1.65 "Segoe UI","Microsoft YaHei",sans-serif;color:#172b3a}h1{font-size:28px}h2{font-size:23px}h3{font-size:19px}details{margin:22px 0;border-top:1px solid #b7cbd8;padding-top:10px}summary{font-size:19px;font-weight:650;cursor:pointer}.note{background:#eef5fb;border-left:4px solid #2378ad;padding:14px 20px}.scroll{overflow-x:auto;margin:12px 0 24px}table{border-collapse:collapse;white-space:nowrap;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;border-bottom:1px solid #dbe3e8;padding:7px 10px}th{background:#e9f1f6}tbody tr:nth-child(even){background:#f7fafc}code{background:#f0f2f4;padding:2px 4px}.materials td:nth-child(n+3){white-space:normal;min-width:170px;max-width:600px;overflow-wrap:anywhere}</style>
<h1>阶段 3：G5b、G6、G7</h1><div class="note"><p>G5b 与 G6 在原数据上均为开发结果；只有满足新材料门槛并完成评测的 G7 才构成外部检验。原数据同时报告 1/原完整组大小与每材料权重 1；两种统计均按组 bootstrap 5000 次。所有配对差值使用共享组抽样。区间反映固定预测的组抽样变动，不包含重新训练或重新运行模型的全部变动。</p><p>缺少任一真实类别时 BA 为 NA；无正类时召回为 NA；无预测正类时精确率按约定为 0；空集全部 NA；单组区间退化。有效抽样不足 5000 时明确列数值，区间仅由该指标有定义的抽样计算。G3 家族标签是模型预测。早截止模型与 GPT 的比较同时改变模型容量、训练、对齐和量化，不能把全部差异只归因为知识截止时间。</p></div>
'''+'\n'.join(body)+'</html>'
    path = safe(ROOT/'outputs/G_stage3_results.html')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding='utf8')
    print(json.dumps({'path': str(path), 'bytes': path.stat().st_size,
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == '__main__':
    export()
