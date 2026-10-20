"""Render the finite F2 masking diagnostic as Markdown in memory only.

No file reads/writes, LLM calls, fits, or recomputation of statistics. The
returned string belongs in metrics.json['report_markdown']; this helper does
not create a seventh experiment artifact or modify earlier-stage sources.
"""
from __future__ import annotations

from f_report import (_SUBSETS, _classification_section, _fraction_row,
                      _table)

METHODS = ("A3_E2", "A3-M1", "A3-M2", "A2_E2", "A2-M2", "HGB127", "5NN")
PAIRS = (("A3-M1", "A3_E2"), ("A3-M2", "A3-M1"),
         ("A3-M2", "A3_E2"), ("A3-M2", "HGB127"),
         ("A3-M2", "5NN"), ("A2-M2", "A3-M2"),
         ("A2-M2", "HGB127"), ("A2-M2", "5NN"),
         ("A2-M2", "A2_E2"))
NUMERIC_METHODS = ("A3_E2", "A3-M2", "A2_E2", "A2-M2", "HGB127", "5NN")
NUMERIC_PAIRS = (("A3-M2", "HGB127"), ("A3-M2", "5NN"),
                 ("A2-M2", "A3-M2"), ("A2-M2", "HGB127"),
                 ("A2-M2", "5NN"), ("A2-M2", "A2_E2"))
_RATE_LABELS = {"abstain": "显式弃权", "abstention": "显式弃权",
                "explicit_abstention": "显式弃权", "failure": "最终调用失败",
                "failed": "最终调用失败", "case_failure": "最终调用失败",
                "not_attempted": "未派发",
                "missing": "缺失最终判断", "invalid": "无效最终判断"}


def _present_methods(data):
    """Keep optional fallback arm names without silently inventing results."""
    available = data.get("group_weighted", {}).get("metrics", {})
    return list(available)


def _count(data):
    """Fraction summaries keep the raw numerator separate from weighted rates."""
    for key in ("n_success", "n_true", "n_events", "numerator", "count"):
        if key in data:
            return data[key]
    return "NA"


def _resume_section(metrics):
    """Describe transport provenance without treating requests as new cases.

    The finite resume runner supplies transport_resume. This renderer neither
    validates/replays calls nor changes their scientific or statistical weight.
    Missing timing fields are left explicitly unavailable, never inferred from
    the order in which calls appear in a log.
    """
    resume = metrics.get("transport_resume")
    if not isinstance(resume, dict):
        return []
    retained = resume.get("retained_valid_count", 5)
    historical = resume.get("historical_call_count", 15)
    previous_valid = resume.get("historical_valid_by_arm", {
        "A3-M1": 2, "A3-M2": 1, "A2-M2": 2})
    arm_order = resume.get("arm_order", ["A3-M2", "A2-M2", "A3-M1"])
    if isinstance(arm_order, (list, tuple)):
        arm_order = "、".join(str(arm) for arm in arm_order)
    lines = ["## 恢复运行与传输边界", "",
             f"先前{historical}次已派发请求中的{retained}个有效判断原样保留；其余没有有效回应的请求重新运行，取消或传输报错不作为模型分类错误。保留的判断不重复计数，每个材料每臂只贡献一个最终有效p。",
             "历史有效判断数：" + "；".join(f"{arm}={count}" for arm, count in previous_valid.items()) + "。",
             f"先用1次全新合成探测请求确认服务状态；恢复并发为{resume.get('concurrency', 4)}，按{arm_order}的优先顺序派发。探测不含真实材料，不计入科学预测或成绩。",
             "CLI的Reconnecting信息仅表示其内部重试，等待CLI终结后才判定该次传输结果；最终429对同一材料、同一轮次、同一提示词最多额外重试3次，分别等待1、2、4分钟。该上限先于8、15分钟两档生效。",
             "明确usage limit或try again at等用量上限提示立即保存并停止；否则只有连续3个材料案例用尽传输重试仍失败时才停止。服务端如给出恢复时间则原样记录，不推算额度恢复时间。",
             "科学格式轮次与传输请求分开计数。传输重试不新增材料、提示词信息或独立科学证据；格式修复仍限原定2轮。", ""]
    timestamps = (
        ("historical_started_utc", "原运行开始（UTC）"),
        ("historical_stopped_utc", "原运行停止（UTC）"),
        ("started_utc", "本次恢复开始（UTC）"),
        ("completed_utc", "本次恢复结束或保存（UTC）"),
    )
    rows = [[label, resume.get(key) or "未记录"] for key, label in timestamps]
    rows.extend([["停止原因", resume.get("stop_reason") or "无或运行尚未结束"],
                 ["服务端恢复时间提示", resume.get("recovery_time_hint") or "未提供"]])
    lines.extend([_table(["运行边界", "记录"], rows), "",
                  "最终统计合并先前保留的5个有效判断和本次新获得的判断，因此不是同一连续时段完成的模型快照。模型名、推理档位和科学输入固定，但服务端不可变权重版本不可核验；时间差异与传输条件是解释边界。", ""])
    return lines


def build_f2_report(metrics):
    """Return complete F2 tables; leave the input dictionary unchanged.

    Required classification schema: metrics['subsets'][subset] is the result
    of f_stats.summarize. Optional fallback_subsets uses the same structure.
    Optional rates[subset][arm][rate_name] entries are fraction_summary results.
    Abstention is among valid responses; failure is among completed attempts;
    not_attempted is among all requested cases. The runner writes event counts.
    Optional report_pairs overrides the nine predefined diagnostic contrasts.
    Optional transport_resume records the retained historical responses and the
    independently revised retry policy; it does not alter any result table.
    Optional numeric_priority_subsets uses the common valid rows of the two
    numeric arms only. Its separate denominator is shown while M1 is incomplete.
    """
    if not isinstance(metrics, dict):
        raise TypeError("F2 metrics must be a dictionary")
    pairs = metrics.get("report_pairs", PAIRS)
    pairs = [tuple(pair) for pair in pairs]
    if any(len(pair) != 2 or pair[0] == pair[1] for pair in pairs):
        raise ValueError("report_pairs must contain distinct method pairs")
    lines = ["# 阶段2 F2隐去身份完整开发结果", "",
             "全部为固定300例、81个原始component组的开发结果；尚非外部检验。",
             "BA=平衡准确率，P=精确率，R=召回率，Acc=准确率。单元为百分比点估计 [95% CI]；配对差为百分点，前者减后者。",
             "组加权采用1/完整3764例中原始组大小，子集筛选后仍保留该权重；不加权每材料权重为1。两种权重均按component_id进行5000次组bootstrap，同一子集共享配对抽样。",
             "主表比较强制判断，即使模型显式弃权也保留其有效p；无有效p的失败例不伪造为负类。若有失败，主表只使用全部方法共同有效的材料，并单列缺失数和完整请求集的失败率。",
             "A3_E2、A2_E2、HGB127、5NN复用冻结E2结果；F2三臂均使用gpt-6.1-sol/high，禁止工具和联网，只根据给定数据判断。",
             "无预测正类时P取0；无真实正类时R未定义，缺少任一真实类别时BA未定义。未定义的bootstrap抽样排除并在表后列出；空集记NA。", "",
             "## 实验接口", "",
             "A3-M1：元素替换为材料内部随机排序的E1、E2等，仅给电负性、共价半径、价电子数的5档性质，并保留配比、晶格与可用空间群。",
             "M1价电子数采用Magpie占据s+p+d+f电子计数约定，含占据d/f电子；并非氧化态或仅最外层电子数。分档来自全元素表，与标签和表现无关。",
             "A3-M2：仅给各折TRAIN拟合StandardScaler后的X127，列名替换为f1–f127；不给化学式、元素符号、空间群或规则。",
             "A2-M2：目标表示与A3-M2相同，再给冻结E2的5个TRAIN近邻的同尺度特征和标签；不给规则或材料身份。",
             "三臂科学格式轮次最多2轮，第2轮仅用冻结的格式修复指令和相同科学载荷；有效首轮即结束。网络传输重试独立记录，重发同一轮的同一提示词，不视为新增科学轮次或独立材料。", "",
             f"运行状态：{metrics.get('status', 'unknown')}。", ""]
    lines.extend(_resume_section(metrics))
    lines.extend(["## 强制判断与主要配对比较", ""])
    for key, label in _SUBSETS:
        data = metrics.get("subsets", {}).get(key)
        if data is not None:
            _classification_section(lines, label, data, METHODS, pairs)
        else:
            lines.extend([f"### {label}", "", "尚无共同有效样本的统计结果。", ""])

    numeric = metrics.get("numeric_priority_subsets", {})
    m1_complete = metrics.get("case_counts", {}).get("A3-M1", {}).get("valid", 0) == metrics.get("planned_n", 300)
    if numeric and not m1_complete:
        lines.extend(["## 数字表示优先比较", "",
                      "A3-M1尚未全部完成。本节单独使用A3-M2与A2-M2共同有效的材料，及同一批材料的冻结E2对照；不要求M1有效。因此本节分母可能大于上方七方法主表，各子集均单独注明n和组数。两种权重、四项指标及5000次配对组bootstrap规则保持一致，不混用不同分母的差值。", ""])
        for key, label in _SUBSETS:
            data = numeric.get(key)
            if data is not None:
                _classification_section(lines, "数字表示优先：" + label, data, NUMERIC_METHODS, NUMERIC_PAIRS)

    rates = metrics.get("rates", {})
    if rates:
        lines.extend(["## 显式弃权与最终失败", "",
                      "弃权率分母为有效回应者，失败率分母为完成尝试者；未派发比例单列，以全部请求材料为分母。事件例数为不加权计数，比例同时报告两种权重与组bootstrap区间。表中有效n指该比例的有效分母，缺失n指未进入该比例分母的请求例数。失败和弃权为不同事件，不能合并解释为模型判断错误。", ""])
        rows = []
        for key, label in _SUBSETS:
            for arm, measurements in rates.get(key, {}).items():
                for rate, data in measurements.items():
                    row = _fraction_row(label + "·" + arm, _RATE_LABELS.get(rate, rate), data)
                    rows.append(row[:2] + [_count(data)] + row[2:])
        if rows:
            lines.extend([_table(["子集与方法", "事件", "事件例数", "有效n", "组数", "请求n", "缺失n",
                                  "组加权比例 (%)", "不加权比例 (%)"], rows), ""])
        else:
            lines.extend(["尚无可用的弃权或失败比例统计。", ""])

    fallback = metrics.get("fallback_subsets", {})
    if fallback:
        lines.extend(["## 附录 HGB回退结果", "",
                      "此附录评估弃权或失败后的系统行为。主分析仍使用强制判断，不能将HGB回退的提升归因于LLM推理。", ""])
        for key, label in _SUBSETS:
            data = fallback.get(key)
            if data is not None:
                methods = _present_methods(data)
                # The full paired values remain in metrics.json; this appendix
                # only displays contrasts explicitly relevant to its arm names.
                available = set(methods)
                fallback_pairs = [pair for pair in metrics.get("fallback_report_pairs", ())
                                  if len(pair) == 2 and all(arm in available for arm in pair)]
                _classification_section(lines, "HGB回退：" + label, data, methods, fallback_pairs)

    lines.extend(["## 解释边界", "",
                  "1. A3→A3-M1同时改变元素身份、物性精度和提示词；A3-M1→A3-M2又改变表示方式。每一步的性能差是联合变化的结果，不能直接换算成预训练记忆所占百分比。",
                  "2. M1的配比、空间群和粗分档性质仍可能指向某些材料；M2的匿名列也可能隐含可恢复的成分或元素索引。隐去名称不等于证明身份信息完全消失。",
                  "3. 对HGB或5NN的比较使用相同X127特征来源；HGB在全TRAIN标签上拟合，LLM只有提示词给定的数据，A2-M2至多有5个近邻标签。因此不能声称训练信息完全相同。",
                  "4. 仅93子集只有1个组，其bootstrap区间必然退化，不能估计组间泛化不确定性；去掉93也会改变类别与家族构成。",
                  "5. 结果仅诊断当前开发集、模型与提示词条件；阶段3及真正外部检验尚未执行。本阶段完成后停止，等待用户确认。", ""])
    return "\n".join(lines)
