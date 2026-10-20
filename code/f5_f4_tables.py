"""Read-only F4 review tables; translate frozen decisions without reassessment.

No network, model calls, experiment writes, or imports of experiment runners.
Paper status is a historical source-screening snapshot; material eligibility is
the frozen final classification. Dates and Tc values retain their source scope.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent

COLUMN_TITLES = {
    "classification_zh": "分类",
    "eligibility_original": "原始材料状态",
    "reason_group_zh": "原因分组",
    "reason_zh": "中文原因",
    "reason_original_en": "原始原因与限定（完整原文）",
    "material_id": "材料ID",
    "formula_original": "化学式或原样标识",
    "Tc_K_recorded": "原记录Tc（K）",
    "Tc_note_zh": "Tc口径说明",
    "Tc_details_original": "Tc定义与备注原文",
    "measurement_pressure_GPa_recorded": "原记录测量压力（GPa）",
    "ambient_measurement_verified": "常压测量已核实",
    "pressure_details_original": "压力口径原文",
    "parent_family_original": "具体家族原样标识",
    "space_group_original": "空间群原记录",
    "paper_id": "论文ID",
    "paper_title_original": "论文标题原文",
    "paper_status_snapshot_original": "论文历史筛查状态",
    "DOI_recorded": "DOI（由已有字段提取）",
    "arxiv_recorded": "arXiv（由已有字段提取）",
    "primary_url": "主文链接",
    "evidence_urls_recorded": "已有证据链接",
    "material_report_date_recorded": "材料条目日期原字段",
    "paper_earliest_date_recorded": "论文条目最早日期原字段",
    "date_details_original": "日期口径原文",
    "first_report_verified": "原记录首报已核实标记",
    "items_to_verify_zh": "待核实项或日期边界",
    "exact_matching_accepted_ids": "精确约化配比匹配行",
    "parent_matching_id_formula_pairs": "具体母体匹配行及配比",
    "dedup_status_original": "去重状态原文",
    "dedup_evidence_original": "去重依据原文",
    "source_json_locations": "原始JSON定位",
    "material_original_json": "完整材料原记录JSON",
    "label_original_json": "完整标签原记录JSON",
    "paper_original_json": "完整论文原记录JSON",
    "parent_check_original_json": "完整母体核查原记录JSON",
}

_PENDING = {
    "qcarbon": ("首报与结构家族待核实", "33 at% Q-carbon 的首报与结构家族资格未同时建立；该配比是名义比例，不能视为精确晶体占位。"),
    "bismuth_oxybromide": ("早期材料报道与家族待核实", "各材料的早期报道与具体家族仍待核实；原记录提及 BaBiO2Br 已见于2014年合成文献。"),
    "tihea_sc_al": ("常压方法、首报与家族待核实", "完整方法中的常压测量、最早材料报道与具体家族尚未核实；摘要不足以接纳该负类候选。"),
    "pbbipd": ("分相母体、各配比Tc与首报待核实", "SG139分支须单独核查具体母体、各配比Tc与首报；弱超导体积分数不能直接改判为已证实非超导。"),
    "bilayer_films": ("各配比首报与具体母体待核实", "各Ln配比的首次材料报道与具体母体核查未完成；不能仅凭宽泛镍酸盐类别判为重复。"),
    "trilayer_nickelate": ("相归属、各配比首报与家族待核实", "低体积分数、非贯通信号及可能的无限层夹层使相归属仍不确定；各配比首报与家族核查未完成。"),
    "organic_fese": ("各配比Tc、首报与间隔层家族待核实", "文献只报最高44K，两个配比各自Tc未确定；首报及有机间隔层的具体家族边界仍待核实。"),
}

_OLD = {
    "qcarbon": "50 at%硼Q-carbon在2019年已有原始报道；2026年更高Tc不构成新配比。",
    "ag2pd3s": "旧超导材料的新表征；2017年单晶论文还引用了1973年的超导报道。",
    "nb2pd3te5": "2021年已有相同材料的块体超导报道；2026年单层或双层样品不构成新化学式。",
    "zr2ru3si4": "2013年已有块体多晶报道；2026年单晶Tc表征不是首次材料报道。",
    "scau2in": "重新表征已知ScAu2In；原记录同时说明旧约3K转变曾归因于In。",
    "sigma_re": "原文明确这些二元Re-X相及既往超导已知，并引用1976年及更早资料；不是新材料身份。",
    "crhea": "2023年腐蚀研究已制备相同CrNbTaTiV合金；2026年超导与μSR表征不是首次材料身份报道。",
    "nd_eu": "2025年预印本已列相同Nd0.7Eu0.3NiO2配比；2026年PLD改变生长工艺，不构成新材料身份。",
    "w_tase2": "2015年已报道相同3R-Ta0.9W0.1Se2；新的CVT单晶制备方法不构成新配比。",
    "rehea10": "2019年原文图1、图2已明确报道同一系列x=0.10的精确配比；不是2026年首报材料。",
    "rehea20": "2019年原文图1、图2已明确报道同一系列x=0.20的精确配比；不是2026年首报材料。",
    "rehea56": "2019年原文图1、图2已明确报道同一系列x=0.56的精确配比；不是2026年首报材料。",
    "ybbi2": "2022年原始论文已生长YbBi2单晶；2026年超导表征不构成新材料身份。",
}

_FAMILY = {
    "lan": "与TRAIN中CeN0.87所代表的岩盐型稀土单氮化物具体母体系列重叠。",
    "cozrh": "低H样品保留C16/Al2Cu型TrZr2框架，与TRAIN中FeZr2、IrZr2、RhZr2的SG140具体母体重叠；不同处理时间与重叠H误差不用于扩充独立材料数。",
    "pbbipd": "x<1的SG140分支与TRAIN中Pb2Pd（Pb4Pd2）的相同四方母体重叠。",
    "hfzrte3": "原文明示在ZrTe3上作Hf替位，与TRAIN的ZrTe3具体母体重叠；不是声称该掺杂精确式已命中。",
    "nbses": "原文明示NbSe2-xSx的x=1，与TRAIN的NbS2及NbSe2具体母体系列重叠；不把proxy NbSe2的SG187误写为2H。",
}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _pointer_part(text):
    return str(text).replace("~", "~0").replace("/", "~1")


def _reason(record):
    state, paper = record["eligibility"], record["paper_id"]
    if state == "pending":
        group, reason = _PENDING[paper]
        if record["id"] == "F4_candidate_bilayer_sm":
            reason += " Sm条目的起始/零电阻定义跨越10K（原记录41K、6K，另列10%电阻28K），不可静默统一标签。"
        if record.get("earlier_Pr_material_reports_pending"):
            reason += " 原记录另注明Pr材料的早期报道仍待核实。"
        return group, reason
    if state == "excluded_preexisting_material":
        return "旧材料的新表征或新工艺", _OLD[paper]
    if state == "excluded_specific_parent_family":
        return "与3764中具体母体家族重复", _FAMILY[paper]
    if state == "excluded_exact_reduced_composition":
        return "约化化学式已在3764中", "约化化学式已命中冻结3764条中的材料；此项排除依据是精确配比重合。"
    if state == "excluded_before_window":
        return "首次报道早于合格窗口", "已有2025年预印本，2026年期刊或新闻日期不是首次报道日期，早于本轮合格窗口。"
    if state == "excluded_identity_public_before_cutoff":
        return "材料身份在知识截止前已公开", "2023年已公开该精确La6Ni5O12身份、结构与超导预测；2026年块体实验没有使材料身份首次出现。"
    raise ValueError(f"Unmapped frozen eligibility: {state}")


def _text_fields(prefix, obj):
    """Keep source-language strings whole, including their original spelling."""
    return [f"{prefix}.{key}: {value}" for key, value in obj.items()
            if isinstance(value, str) and any(part in key for part in
                ("evidence", "note", "status", "boundary", "criterion"))]


def _urls(value):
    if isinstance(value, dict):
        return [url for item in value.values() for url in _urls(item)]
    if isinstance(value, list):
        return [url for item in value for url in _urls(item)]
    if isinstance(value, str):
        return [url.rstrip(".,;)") for url in re.findall(r"https?://[^\s\"<>]+", value)]
    return []


def extract_f4_review_tables(root=None):
    """Return {pending:19 rows, excluded:31 rows, columns, summary, provenance}.

    This reads only F4 config/metrics. It never changes eligibility, completes
    missing Tc values, reinterprets current-paper dates as verified first reports,
    or treats a pending paper snapshot as a final material exclusion.
    """
    folder = (Path(root).resolve() if root is not None else ROOT) / "F4_novel"
    if "onedrive" in str(folder.resolve()).lower():
        raise ValueError("F4 source must stay outside OneDrive")
    paths = {name: folder / name for name in ("config.json", "metrics.json")}
    raw = {name: path.read_bytes() for name, path in paths.items()}
    config, metrics = (json.loads(raw[name].decode("utf-8")) for name in paths)
    if config["status"] != "collection_complete_insufficient":
        raise ValueError("Expected the frozen insufficient-collection F4 snapshot")
    papers = {p["paper_id"]: (index, p) for index, p in enumerate(config["candidate_papers_not_for_prompt"])}
    labels = config["evaluation_labels_not_for_prompt"]
    checks = config["specific_parent_family_checks_not_for_prompt"]
    result = {"pending": [], "excluded": []}
    for index, material in enumerate(config["candidate_materials_not_for_prompt"]):
        state = material["eligibility"]
        if state == "accepted":
            continue
        category = "pending" if state == "pending" else "excluded"
        paper_index, paper = papers[material["paper_id"]]
        label = labels.get(material["id"], {})
        check_key = material.get("parent_family_check_key")
        check = checks.get(check_key, {})
        group, reason = _reason(material)
        source_text = "\n".join([f"material.eligibility: {state}"] + _text_fields("material", material) + _text_fields("paper", paper) + _text_fields("parent_check", check))
        all_text = _json({"material": material, "paper": paper, "check": check})
        urls = list(dict.fromkeys(_urls({"material": material, "paper": paper, "check": check})))
        dois = list(dict.fromkeys(value.rstrip(".,") for value in re.findall(r"10\.\d{4,9}/[A-Za-z0-9.()_:/-]+", all_text)))
        arxiv = list(dict.fromkeys(match.group(1) for match in re.finditer(
            r"(?:arxiv(?:\.org)?/(?:abs|html|pdf)/|arxiv[:/\s]*)(\d{4}\.\d{4,5}(?:v\d+)?)", all_text, re.I)))
        locations = [f"{paths['config.json']}#/candidate_materials_not_for_prompt/{index}",
                     f"{paths['config.json']}#/candidate_papers_not_for_prompt/{paper_index}",
                     f"{paths['config.json']}#/evaluation_labels_not_for_prompt/{_pointer_part(material['id'])}"]
        if check_key:
            locations.append(f"{paths['config.json']}#/specific_parent_family_checks_not_for_prompt/{_pointer_part(check_key)}")
        tc_details = {k: material[k] for k in ("label_criterion", "Tc_criterion", "Tc_note", "alternate_Tc_definitions", "same_material_later_measurements") if k in material}
        tc_details["label_record"] = label
        date_details = {k: material[k] for k in ("first_report_evidence", "public_artifact_date_boundary") if k in material}
        date_details.update({"paper_date_status": paper.get("date_status"), "paper_known_prior_report_year": paper.get("known_prior_report_year")})
        dedup_evidence = {k: material[k] for k in ("parent_family_evidence", "exclusion_evidence", "composition_resolution_note") if k in material}
        if check:
            dedup_evidence["specific_parent_family_check"] = check
        verify = reason if category == "pending" else "按冻结状态已排除；原记录日期可能是本轮论文日期，不能据此重新认定首次材料报道。"
        row = {
            "classification_zh": "待定，未纳入且未排除" if category == "pending" else "已排除",
            "eligibility_original": state, "reason_group_zh": group, "reason_zh": reason,
            "reason_original_en": source_text, "material_id": material["id"],
            "formula_original": material.get("formula", material["id"]),
            "Tc_K_recorded": label.get("Tc_K"),
            "Tc_note_zh": "仅转录原记录，不等于已合格或已评测；未给定值保持为空，不补作0K。",
            "Tc_details_original": _json(tc_details),
            "measurement_pressure_GPa_recorded": label.get("measurement_pressure_GPa"),
            "ambient_measurement_verified": material.get("ambient_measurement_verified"),
            "pressure_details_original": _json({k: label[k] for k in label if "pressure" in k} | {k: material[k] for k in ("ambient_note",) if k in material}),
            "parent_family_original": material.get("parent_family_key"),
            "space_group_original": material.get("space_group") or material.get("phase_space_group"),
            "paper_id": material["paper_id"], "paper_title_original": paper.get("title"),
            "paper_status_snapshot_original": paper.get("status"),
            "DOI_recorded": "; ".join(dois), "arxiv_recorded": "; ".join(arxiv),
            "primary_url": paper.get("primary_url"), "evidence_urls_recorded": "\n".join(urls),
            "material_report_date_recorded": material.get("first_public_report_date"),
            "paper_earliest_date_recorded": paper.get("earliest_public_report_date"),
            "date_details_original": _json(date_details),
            "first_report_verified": material.get("first_report_verified"),
            "items_to_verify_zh": verify,
            "exact_matching_accepted_ids": _json(material.get("matching_accepted_ids", [])),
            "parent_matching_id_formula_pairs": _json(check.get("matching_accepted_id_formula_pairs", [])),
            "dedup_status_original": material.get("parent_family_dedup_status"),
            "dedup_evidence_original": _json(dedup_evidence),
            "source_json_locations": "\n".join(locations),
            "material_original_json": _json(material), "label_original_json": _json(label),
            "paper_original_json": _json(paper), "parent_check_original_json": _json(check),
        }
        assert list(row) == list(COLUMN_TITLES)
        result[category].append(row)
    assert len(result["pending"]) == 19 and len(result["excluded"]) == 31
    counts = Counter(row["eligibility_original"] for rows in result.values() for row in rows)
    expected = {key: value for key, value in metrics["material_status_counts"].items() if key != "accepted"}
    assert counts == expected
    result.update(columns=list(COLUMN_TITLES), column_titles=dict(COLUMN_TITLES),
        summary={name: {"n": len(result[name]), "reason_groups": dict(Counter(row["reason_group_zh"] for row in result[name]))} for name in ("pending", "excluded")},
        source_sha256={str(paths[name]): hashlib.sha256(value).hexdigest() for name, value in raw.items()},
        source_status=config["status"],
        semantics="Only frozen material eligibility is final; paper statuses are historical snapshots. No F4 research resumed.",
        ambiguities=[
            "日期两列原样分开：旧材料条目有时保存2026年当前论文日期，并非最早身份公开日期。",
            "Ti合金标签区的pressure=0不覆盖ambient_measurement_verified=false；完整常压方法仍待核实。",
            "Q-carbon33的first_report_verified=true与论文最终备注中的首报/家族未同时建立并存，保持原记录不改判。",
            "Sm双层镍酸盐不同Tc定义跨越10K；有机FeSe两个配比只报最高44K，原Tc空值不补齐。",
            "被排除Pb/Bi/Pd配比的Tc原字段为空；只据已冻结具体母体排除，不从同论文其他配比推填Tc。",
        ])
    return result
