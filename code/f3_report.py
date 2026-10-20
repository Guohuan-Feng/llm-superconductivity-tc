"""Render finite full-cohort F3 results in memory; no I/O or recomputation."""
from __future__ import annotations

from itertools import combinations

from f_report import _SUBSETS, _classification_section, _fraction_row, _table

FAMILY_LABELS = {"copper": "铜基", "iron": "铁基", "hydride": "氢化物",
                 "A15": "A15结构判据", "other": "其他"}
SIZE_LABELS = {"1": "组大小1", "2-5": "组大小2–5", "6-20": "组大小6–20", ">20": "组大小>20"}
OVERLAP_LABELS = {"raw_A15": "结构判据A15", "component_93": "component 93",
                  "raw_A15_and_component_93": "A15∩93",
                  "raw_A15_without_component_93": "A15且非93",
                  "component_93_without_raw_A15": "93且不满足A15结构判据",
                  "exclusive_A15_family": "互斥拆分中的A15家族"}


def _counts_note(data):
    return (f"请求n={data.get('n_requested', data.get('n', 0))}，"
            f"请求组数={data.get('n_requested_groups', data.get('n_groups', 0))}；"
            f"缺失共同有效判断n={data.get('n_missing', 0)}，"
            f"涉及组数={data.get('n_missing_groups', 0)}。")


def _methods(metrics):
    available = metrics.get("subsets", {}).get("all", {}).get("group_weighted", {}).get("metrics", {})
    names = list(metrics.get("method_order", available))
    if len(names) != len(set(names)) or any(name not in available for name in names):
        raise ValueError("method_order must list unique available methods")
    if set(names) != set(available):
        raise ValueError("method_order must include every supplied method")
    return names


def _three_subsets(lines, title, data_by_subset, methods, pairs):
    for subset, label in _SUBSETS:
        data = data_by_subset.get(subset)
        if data is None:
            lines.extend([f"### {title}：{label}", "", "尚无统计结果；不推算缺失指标。", ""])
            continue
        lines.extend([_counts_note(data), ""])
        _classification_section(lines, title + "：" + label, data, methods, pairs)


def build_f3_report(metrics):
    """Return all F3 classification/paired tables without mutating inputs.

    Required statistics match build_f3_statistics: subsets, families,
    group_size_bins, method_order and report_pairs. Runner metadata may add
    status, calls, fixed300_reuse, case_counts, rates and family_definition.
    Optional A2_priority (or a2_priority) is a second build_f3_statistics output
    for A2 plus baselines only, shown if it has more rows than the main cohort.
    The runner embeds this text in metrics.json; no extra report file is made.
    """
    if not isinstance(metrics, dict):
        raise TypeError("F3 metrics must be a dictionary")
    methods = _methods(metrics)
    pairs = [tuple(pair) for pair in metrics.get("report_pairs", combinations(methods, 2))]
    expected = {frozenset(pair) for pair in combinations(methods, 2)}
    actual = {frozenset(pair) for pair in pairs if len(pair) == 2 and pair[0] != pair[1]}
    if len(pairs) != len(expected) or actual != expected:
        raise ValueError("F3 requires every distinct method pair exactly once")
    lines = ["# 阶段3 F3全部3764条开发结果", "",
             "全部为已有3764条、1117个component组的开发结果，尚非新材料外部检验；不执行全量回忆测试。",
             "A2、A3只评估强制判断；固定300例原样复用E2预测，其余材料用本机codex exec、gpt-6.1-sol/high、冻结的E2科学提示词与输入接口生成。被评测调用禁止工具和联网。",
             "BA=平衡准确率，P=精确率，R=召回率，Acc=准确率；单元为百分比点估计 [95% CI]，配对差为百分点、前者减后者。每张表均列有效n和组数，并另外列请求和缺失数。",
             "组加权采用1/完整3764条中原始component组大小；所有子集、家族和组大小拆分继续使用该原始权重。不加权每材料权重为1。两种权重均按component组重采样5000次，同一层的方法、权重和所有配对差共享抽样。",
             "所有主表在全部方法共同有效的材料上配对；无有效p的材料不伪造为负类。显式弃权但有效的p继续用于强制判断，冻结p≥0.5为正类；p=0.5不表示模型认可正类。",
             "无预测正类时P取0；无真实正类时R未定义，缺少任一真实类别时BA未定义。未定义bootstrap抽样排除并列出数量；空层全部记NA。单组区间必然退化，不能估计组间泛化不确定性。", "",
             f"运行状态：{metrics.get('status', 'unknown')}。", ""]
    reuse = metrics.get("fixed300_reuse")
    if reuse is not None:
        lines.extend(["固定300例复用记录：" + str(reuse), ""])
    lines.extend(["## 全体及明确去掉93组的主比较", ""])
    _three_subsets(lines, "主比较", metrics.get("subsets", {}), methods, pairs)

    priority = metrics.get("A2_priority", metrics.get("a2_priority", {}))
    main_n = metrics.get("subsets", {}).get("all", {}).get("n", 0)
    if priority and priority.get("subsets", {}).get("all", {}).get("n", 0) > main_n:
        priority_methods = _methods(priority)
        priority_pairs = [tuple(pair) for pair in priority.get("report_pairs", combinations(priority_methods, 2))]
        lines.extend(["## A2优先完成的独立共同分母比较", "",
                      "A2先于A3运行；本节只要求A2与冻结基线共同有效，不要求A3有效。其分母可能大于上述A2、A3共同分母，各子集分别注明n与组数；所有配对只在本节相同材料内计算，不与主表跨分母相减。A3覆盖追平后不再重复显示本节。", ""])
        _three_subsets(lines, "A2优先比较", priority.get("subsets", {}), priority_methods, priority_pairs)

    overlap = metrics.get("a15_component93_overlap", {})
    if overlap:
        lines.extend(["## A15结构判据与component 93重叠", "",
                      "A15按冻结结构判据定义：声明空间群223且严格二元A₃B配比。互斥家族拆分还遵循冻结优先顺序；独立结构判据与component 93的交集在此另列，不能把93整组自动等同于A15结构判据。",
                      "交集判据来源：" + str(overlap.get("criterion_source", "未记录")) + "。", ""])
        rows = []
        for scope, scope_label in (("requested", "请求材料"), ("common_valid", "全部方法共同有效")):
            for key, value in overlap.get(scope, {}).items():
                rows.append([scope_label, OVERLAP_LABELS.get(key, key), value.get("n", 0),
                             value.get("n_groups", 0), value.get("n_positive", 0), value.get("n_negative", 0)])
        lines.extend([_table(["范围", "集合", "n", "组数", "正类", "负类"], rows), ""])

    lines.extend(["## 家族拆分", "",
                  "铜基、铁基、氢化物、A15、其他为冻结的互斥材料划分；元素或结构定义及优先顺序记录于config.json。组可跨家族，家族组数不能相加当作总独立组数。每个家族继续报告全部、去掉93、仅93三个子集；空层保留NA。", ""])
    for family, label in FAMILY_LABELS.items():
        _three_subsets(lines, "家族·" + label, metrics.get("families", {}).get(family, {}), methods, pairs)

    lines.extend(["## 原始组大小拆分", "",
                  "分档依据完整3764条中的原始component大小：1、2–5、6–20、>20；不会因移除93、失败过滤或家族子集而重定义。每档继续报告三个子集及全部方法的配对差。", ""])
    for size, label in SIZE_LABELS.items():
        _three_subsets(lines, label, metrics.get("group_size_bins", {}).get(size, {}), methods, pairs)

    rates = metrics.get("rates", {})
    if rates:
        lines.extend(["## 显式弃权、最终失败与未派发", "",
                      "弃权是有效回应中的事件，最终失败与未派发单列；它们不等同于分类错误。率同时使用两种权重和5000次组bootstrap，事件数为材料计数。", ""])
        rows = []
        labels = {"abstain": "显式弃权", "failure": "最终失败", "not_attempted": "未派发"}
        for subset, label in _SUBSETS:
            for method, events in rates.get(subset, {}).items():
                for event, data in events.items():
                    row = _fraction_row(label + "·" + method, labels.get(event, event), data)
                    rows.append(row[:2] + [data.get("count", "NA")] + row[2:])
        lines.extend([_table(["子集与方法", "事件", "事件例数", "有效n", "组数", "请求n", "缺失n",
                              "组加权比例 (%)", "不加权比例 (%)"], rows), ""])

    lines.extend(["## 解释边界", "",
                  "1. 本阶段只增加同一开发数据内的覆盖范围，固定300例复用而非重新采样；不构成独立外部检验，也不能直接估计预训练记忆占优势的百分比。",
                  "2. A2同时含规则、化学身份与近邻，而A3含化学身份与结构；两者的性能差不能孤立解释为规则推理的因果贡献。HGB和强ML使用全部外层TRAIN标签，LLM的提示词训练信息不同。",
                  "3. 家族、组大小及去掉93的比较反映特定开发分布；多组拆分为描述性诊断，未进行多重比较校正。单组或单类层的NA与退化CI不能当作高确定性。",
                  "4. 冻结模型名、推理档位、提示词与输入来源，但不可变服务端模型权重快照不可核验；复用E2与新调用可能来自不同运行时段。",
                  "5. 本阶段结束后停止，等待用户确认；不自动进入新材料测试，不选择研究路线。", ""])
    return "\n".join(lines)
