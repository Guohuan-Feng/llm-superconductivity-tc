"""Render complete finite Stage-1 results as Markdown in memory only.

No file reads/writes, model calls, fits, or recomputation of statistics. Input
objects are the completed F0/F1 metrics dictionaries. The returned string can
be embedded as metrics['report_markdown'] without adding an output artifact.
"""
from __future__ import annotations

import math

_METRICS = ("balanced_accuracy", "precision", "recall", "accuracy")
_SUBSETS = (("all", "全部"), ("exclude_93", "去掉 component_id=93"),
            ("only_93", "仅 component_id=93"))
_F0_ARMS = ("R-info", "5NN", "HGB127", "A1_forced", "A1_HGBfallback",
            "A2_forced", "A2_HGBfallback", "A3_forced", "A3_HGBfallback")
_F1_ARMS = ("R-recall", "A2_forced", "A3_forced", "HGB127")
_F0_PAIRS = (("A2_forced", "HGB127"), ("A3_forced", "HGB127"),
             ("A2_forced", "A3_forced"), ("A1_forced", "R-info"))
_F1_PAIRS = (("A2_forced", "HGB127"), ("A3_forced", "HGB127"),
             ("A2_forced", "A3_forced"), ("R-recall", "HGB127"))
_WEIGHTINGS = ("group_weighted", "unweighted")


def _finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _cell(stat, *, signed=False):
    """Proportions to percentages; differences to percentage points."""
    stat = stat or {}
    point, interval = stat.get("point"), stat.get("ci95", [None, None])
    if not _finite(point):
        return "NA"
    if not isinstance(interval, (list, tuple)) or len(interval) != 2 or not all(_finite(x) for x in interval):
        return f"{100 * point:+.1f} [NA]" if signed else f"{100 * point:.1f} [NA]"
    if signed:
        return f"{100 * point:+.1f} [{100 * interval[0]:+.1f}, {100 * interval[1]:+.1f}]"
    return f"{100 * point:.1f} [{100 * interval[0]:.1f}, {100 * interval[1]:.1f}]"


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |"
                 for row in rows)
    return "\n".join(lines)


def _sample_note(data):
    return f"n={data.get('n', 0)}，组数={data.get('n_groups', 0)}，正类={data.get('n_positive', 0)}，负类={data.get('n_negative', 0)}。"


def _classification_table(data, arms):
    rows = []
    for arm in arms:
        row = [arm]
        for weighting in _WEIGHTINGS:
            metrics = data.get(weighting, {}).get("metrics", {}).get(arm, {})
            row.extend(_cell(metrics.get(metric)) for metric in _METRICS)
        rows.append(row)
    return _table(["方法", "组加权 BA", "组加权 P", "组加权 R", "组加权 Acc",
                   "不加权 BA", "不加权 P", "不加权 R", "不加权 Acc"], rows)


def _difference(data, weighting, a, b, metric):
    pairs = data.get(weighting, {}).get("paired_differences", {})
    direct = pairs.get(f"{a} - {b}")
    if direct is not None:
        return direct.get(metric)
    reverse = pairs.get(f"{b} - {a}", {}).get(metric)
    if reverse is None:
        return None
    point, interval = reverse.get("point"), reverse.get("ci95", [None, None])
    return {"point": -point if _finite(point) else None,
            "ci95": [-interval[1], -interval[0]] if len(interval) == 2 and all(_finite(x) for x in interval) else [None, None]}


def _paired_table(data, pairs):
    rows = []
    for a, b in pairs:
        row = [f"{a} − {b}"]
        for weighting in _WEIGHTINGS:
            row.extend(_cell(_difference(data, weighting, a, b, metric), signed=True) for metric in _METRICS)
        rows.append(row)
    return _table(["配对差（百分点）", "组加权 ΔBA", "组加权 ΔP", "组加权 ΔR", "组加权 ΔAcc",
                   "不加权 ΔBA", "不加权 ΔP", "不加权 ΔR", "不加权 ΔAcc"], rows)


def _undefined_note(data, arms):
    """Expose omitted bootstrap draws when class-sparse strata produce them."""
    items = []
    for weighting, label in (("group_weighted", "组加权"), ("unweighted", "不加权")):
        measurements = data.get(weighting, {}).get("metrics", {})
        # Identical denominators across methods usually give identical counts;
        # group them to keep a large report readable without hiding omissions.
        counts = {}
        for arm in arms:
            for metric in _METRICS:
                count = measurements.get(arm, {}).get(metric, {}).get("n_undefined_boot", 0)
                if count:
                    counts.setdefault((metric, count), []).append(arm)
        for (metric, count), names in counts.items():
            whom = "所有方法" if len(names) == len(arms) else "/".join(names)
            items.append(f"{label} {whom} {metric}: {count}/{data.get('n_boot', 5000)}")
    return "未定义的 bootstrap 次数（CI仅使用其余抽样）：" + "；".join(items) + "。" if items else ""


def _classification_section(lines, title, data, arms, pairs):
    lines.extend([f"### {title}", "", _sample_note(data), "", _classification_table(data, arms), "",
                  "主要配对比较；差值为前者减后者，使用同一组抽样。", "", _paired_table(data, pairs), ""])
    if data.get("single_group_warning") or data.get("n_groups") == 1:
        lines.extend(["**此子集只有1个组；bootstrap区间退化，不能估计组间不确定性或支持跨组泛化。**", ""])
    if data.get("n", 0) == 0:
        lines.extend(["此层为空集；全部指标及区间记为NA。", ""])
    note = _undefined_note(data, arms)
    if note:
        lines.extend([note, ""])


def _fraction_row(subject, quantity, data):
    return [subject, quantity, data.get("n", 0), data.get("n_groups", 0),
            data.get("n_requested", data.get("n", 0)), data.get("n_missing", 0),
            _cell(data.get("group_weighted")), _cell(data.get("unweighted"))]


def build_stage1_report(f0, f1):
    """Return all Stage-1 result tables as one complete Markdown string.

    F0 schema: metrics['subsets'][all/exclude_93/only_93].
    F1 schema: classification[subset], by_recognition[state][subset],
    recognition[real_<subset>/controls_all][rate], and
    recall_correctness[subset][yes/yes_or_partial][rate].
    Both dictionaries remain unchanged. No files are opened by this helper.
    """
    lines = ["# 阶段1 F0复算与F1回忆测试完整开发结果", "",
             "全部为固定300例的开发结果，尚非外部检验。BA=平衡准确率，P=精确率，R=召回率，Acc=准确率。",
             "所有指标为百分比，表格单元为点估计 [95% CI]；配对差为百分点。组加权保留完整3764例的原始组大小倒数，不加权每材料权重为1。",
             "两种权重均按component_id进行5000次组bootstrap；同一子集所有方法、两种权重及配对差共享抽样。",
             "无预测正类时P取0；无正类时R未定义；缺少任一真实类别时BA未定义。未定义抽样排除并在表后报告数量；空集为NA。",
             "识别状态是模型自报，不能直接测定训练暴露或将优势因果拆分为记忆与推理。", "",
             "## F0 既有E2全部9臂复算", ""]
    for key, label in _SUBSETS:
        data = f0.get("subsets", {}).get(key)
        if data is not None:
            _classification_section(lines, f"F0：{label}", data, _F0_ARMS, _F0_PAIRS)
    checks = f0.get("reproduction_checks", {})
    if checks:
        lines.extend(["F0复现检查：旧E2组加权点估计和区间最大绝对差为"
                      f"{checks.get('saved_E2_max_abs_difference', 'NA')}。93组"
                      f"{checks.get('group93_n', 'NA')}例、{checks.get('group93_positive', 'NA')}个正类；"
                      f"Tc范围{checks.get('group93_tc_min', 'NA')}–{checks.get('group93_tc_max', 'NA')} K。", ""])
    lines.extend(["## F1 只给proxy cell formula的回忆测试", "",
                  f"运行状态：{f1.get('status', 'unknown')}；有效真实材料调用 {f1.get('n_valid_real', 0)}/300；"
                  f"有效随机配比对照调用 {f1.get('n_valid_controls', 0)}/20。", "",
                  "recognized=yes表示声称认识该具体配比；partial只认识家族或母体；no表示不认识。严格认识率只计yes，宽口径计yes+partial。",
                  "R-recall以回忆的数值Tc>10 K为正类；null按负类计，同时单列数值覆盖率。分类比较只使用回忆调用有效的共同材料。", ""])
    for key, label in _SUBSETS:
        data = f1.get("classification", {}).get(key)
        if data is not None:
            _classification_section(lines, f"F1分类：{label}", data, _F1_ARMS, _F1_PAIRS)
    lines.extend(["## F1 认识率与数值Tc覆盖率", "",
                  "随机配比对照只确认不在3764例中，不能据此证明它们未在文献或预训练中出现；对照的自报认识率是误认风险提示。", ""])
    recognition_rows = []
    rate_labels = {"yes": "严格认识率 yes", "partial": "部分认识 partial", "no": "不认识 no",
                   "yes_or_partial": "宽口径认识率 yes+partial", "numeric_Tc": "数值Tc覆盖率"}
    for key, label in [*(("real_" + key, "真实材料·" + label) for key, label in _SUBSETS),
                       ("controls_all", "随机配比对照·全部")]:
        for rate, data in f1.get("recognition", {}).get(key, {}).items():
            recognition_rows.append(_fraction_row(label, rate_labels.get(rate, rate), data))
    if recognition_rows:
        lines.extend([_table(["集合", "比例", "有效n", "组数", "请求n", "缺失n",
                              "组加权比例 (%)", "不加权比例 (%)"], recognition_rows), ""])
    else:
        lines.extend(["尚无可用认识率结果。", ""])
    lines.extend(["## F1 回忆正确性", "",
                  "主结果以全部自报认识材料为分母：未给数值Tc视为未正确回忆。numeric-only为仅在给出数值Tc者中的敏感性分析；两种分母均明确列出。", ""])
    correctness_rows = []
    correctness_labels = {"numeric_coverage": "数值Tc覆盖率",
                          "within_2K_all_recognized": "Tc误差≤2 K／全部认识者",
                          "same_10K_side_all_recognized": "10 K同侧／全部认识者",
                          "within_2K_numeric_only": "Tc误差≤2 K／数值Tc者",
                          "same_10K_side_numeric_only": "10 K同侧／数值Tc者"}
    for key, label in _SUBSETS:
        for state, state_label in (("yes", "严格yes"), ("yes_or_partial", "yes+partial")):
            for rate, data in f1.get("recall_correctness", {}).get(key, {}).get(state, {}).items():
                correctness_rows.append(_fraction_row(f"{label}·{state_label}", correctness_labels.get(rate, rate), data))
    if correctness_rows:
        lines.extend([_table(["子集与认识定义", "回忆正确性指标", "分母n", "组数", "请求n", "缺失n",
                              "组加权比例 (%)", "不加权比例 (%)"], correctness_rows), ""])
    else:
        lines.extend(["尚无可用回忆正确性结果。", ""])
    lines.extend(["## F1 按认识状态分层的分类结果", "",
                  "每层都报告R-recall、A2、A3及HGB的共同样本；分层为描述性比较，不能将层间性能差直接解释为记忆的因果贡献。", ""])
    for state, state_label in (("yes", "认识yes"), ("partial", "部分认识partial"), ("no", "不认识no")):
        for key, label in _SUBSETS:
            data = f1.get("by_recognition", {}).get(state, {}).get(key)
            if data is not None:
                _classification_section(lines, f"F1分层：{state_label}；{label}", data, _F1_ARMS, _F1_PAIRS)
    lines.extend(["## 解释边界", "",
                  "1. 删除93组会改变类别和材料构成；性能变化本身不能区分先验化学知识、具体记忆与推理。",
                  "2. 精确回忆或Tc误差≤2 K也可能受到家族知识、推断、相态和数据来源差异影响；F1不能识别一个因果‘记忆占比’。",
                  "3. 仅93只有1组，所有组bootstrap区间必然退化，不能视为高确定性。",
                  "4. 随机配比对照没有真实实验Tc标签，只报告认识率，不计算其超导分类成绩。",
                  "5. F2隐去身份、F3全量和F4新材料外检尚未执行；本阶段完成后停止，等待用户继续。", ""])
    return "\n".join(lines)
