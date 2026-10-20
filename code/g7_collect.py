import csv, datetime, hashlib, json, re
from collections import Counter
from pathlib import Path

ROOT = Path(r"C:\Users\28908\Codex\fung-research\v2-diagnosis")
OUT = ROOT / "G7_postcutoff_materials"
OUT.mkdir(exist_ok=True)
f4 = json.loads((ROOT / "F4_novel/config.json").read_text(encoding="utf8"))
cutoff = "2023-12-31"  # official Llama 3.1 model card reports Dec 2023, conservative month end
papers = {x.get("paper_id"): x for x in f4["candidate_papers_not_for_prompt"]}
inferred = {
    "lan":"2026-09-19", "ag2pd3s":"2026-06-29", "nb2pd3te5":"2026-08-26", "bismuth_oxybromide":"2026-07-01",
    "zr2ru3si4":"2026-07-01", "la3in":"2026-07-01", "scau2in":"2026-07-01", "yru3b2_luru3b2":"2025-12-16",
    "cozrh":"2026-06-27", "pbbipd":"2026-09-17", "bilayer_films":"2026-08-18", "trilayer_nickelate":"2026-09-17",
    "nd_eu":"2026-07-11", "la6ni5o12":"2026-10-14", "hfzrte3":"2026-06-10", "nbses":"2026-06-17",
    "w_tase2":"2026-06-24", "rehea10":"2026-07-25", "ybbi2":"2026-09-17", "organic_fese":"2026-09-07",
    "qcarbon":"2026-06-08", "mpbbi":"2026-06-24", "sigma_re":"2026-09-24", "ynbtizrhf":"2026-08-11",
    "tihea_sc_al":"2026-09-14", "crhea":"2026-06-04", "rehea20":"2026-07-25", "rehea56":"2026-07-25",
}
labels, rows = {}, []
for m in f4["candidate_materials_not_for_prompt"]:
    mid = m["id"]
    lab = f4["evaluation_labels_not_for_prompt"].get(mid, {})
    status = m.get("eligibility", "pending")
    if m.get("matching_accepted_ids"):
        status = "excluded_exact_3764_reduced_formula"
    date = m.get("first_public_report_date") or inferred.get(m.get("paper_id"))
    p = papers.get(m.get("paper_id"), {}) or {}
    rows.append(dict(material_id=mid, formula=m.get("formula"), tc_k=lab.get("Tc_K"), pressure_gpa=lab.get("measurement_pressure_GPa"),
                     first_report_date=date, doi_or_arxiv=p.get("primary_url"), cif_url=m.get("cif_url"), space_group=m.get("space_group"),
                     status=status, source="F4_novel/config.json", reason=m.get("parent_family_dedup_status") or "F4 status retained"))
    labels[mid] = {"tc_k":lab.get("Tc_K"), "y":lab.get("y"), "pressure_gpa":lab.get("measurement_pressure_GPa"), "status":status,
                   "label_source":"F4 evaluation_labels_not_for_prompt"}

new = [
    ("G7_lapr_bilayer_285", "La2.85Pr0.15Ni2O7", 40.0, "2024-12-20", "https://arxiv.org/abs/2412.16622", "experimental epitaxial film; ambient-pressure onset >40 K; distinct reduced composition"),
    ("G7_hybrid_1212", "(La,Pr)5Ni3O11", 50.0, "2025-09-03", "https://arxiv.org/abs/2509.03502", "experimental hybrid 1212 thin-film superstructure; ambient-pressure onset"),
    ("G7_hybrid_2323", "(La,Pr)7Ni5O17", 46.0, "2025-09-03", "https://arxiv.org/abs/2509.03502", "experimental hybrid 2323 thin-film superstructure; ambient-pressure onset"),
    ("G7_hybrid_1313", "(La,Pr)3Ni2O7", 0.0, "2025-09-03", "https://arxiv.org/abs/2509.03502", "same study explicit non-superconducting 1313 control at ambient pressure"),
    ("G7_bst_pressure_quenched", "Bi0.5Sb1.5Te3", 10.2, "2025-02-03", "https://arxiv.org/abs/2502.01881", "pressure-quenched phase measured after decompression at ambient pressure"),
    ("G7_tinbtan3", "TiNbTaN3", 10.0, "2025-05-21", "https://doi.org/10.1002/advs.202506089", "bulk experimental medium-entropy nitride; Tc onset 10 K at ambient pressure"),
    ("G7_ga_ge_mbe", "Ge0.821Ga0.179", 3.5, "2025-10-30", "https://doi.org/10.1038/s41565-025-02042-8", "experimental but Ga-doped Ge family predates cutoff; screened only"),
]
for mid, formula, tc, date, url, reason in new:
    # Nickelate rows stay pending until a strict parent-family check against the
    # 3764-entry reference is completed; F4 used the same conservative rule.
    if mid == "G7_ga_ge_mbe":
        status = "excluded_preexisting_family"
    elif mid in {"G7_lapr_bilayer_285", "G7_hybrid_1212", "G7_hybrid_2323", "G7_hybrid_1313"}:
        status = "pending_specific_parent_family"
    else:
        status = "accepted"
    rows.append(dict(material_id=mid, formula=formula, tc_k=tc, pressure_gpa=0.0, first_report_date=date, doi_or_arxiv=url,
                     cif_url="", space_group="", status=status, source="primary literature URL", reason=reason))
    labels[mid] = {"tc_k":tc, "y":int(tc > 10), "pressure_gpa":0.0, "status":status, "label_source":"primary abstract/full record"}

qual = [r for r in rows if r["status"] == "accepted" and (r["first_report_date"] or "") > cutoff]
OUT.joinpath("predictions.csv").write_text("", encoding="utf8")
with (OUT / "predictions.csv").open("w", newline="", encoding="utf8") as f:
    # Keep evaluation labels (Tc, pressure, y) in labels.json only.
    public_fields = ["material_id", "formula", "first_report_date", "doi_or_arxiv", "cif_url", "space_group", "status", "source", "reason"]
    w = csv.DictWriter(f, fieldnames=public_fields); w.writeheader(); w.writerows({k:r.get(k) for k in public_fields} for r in rows)
(OUT / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf8")
manifest = {
    "experiment":"G7_postcutoff_material_collection", "status":"collection_in_progress", "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "model_cutoff_date":cutoff, "cutoff_source":"https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct (official model card, December 2023 pretraining cutoff)",
    "eligibility_window_start":"2024-02-01", "as_of_date":"2026-10-19", "target_materials":100, "target_positive":25,
    "stop_rule":"stop if eligible materials <60 or eligible positives <15", "candidate_rows_screened":len(rows),
    "qualified_rows_after_dedup_and_primary_evidence":len(qual), "qualified_positive_rows":sum(labels[r["material_id"]]["y"] == 1 for r in qual),
    "old_f4_candidates_reused":len(f4["candidate_materials_not_for_prompt"]), "new_primary_records":len(new), "evaluation_started":False,
    "no_llm_calls":True, "no_baseline_fits":True, "source_policy":"primary paper/arXiv/DOI links recorded per row; F4 records retained read-only; labels stored separately",
    "dedup_policy":"exact reduced formula/parent-family exclusions retained from F4; ambiguous parent-family cases remain pending/excluded",
    "source_urls":["https://arxiv.org/abs/2412.16622", "https://arxiv.org/abs/2509.03502", "https://arxiv.org/abs/2502.01881", "https://doi.org/10.1002/advs.202506089", "https://doi.org/10.1038/s41565-025-02042-8", "F4_novel/config.json"],
    "reason_for_stop":None, "previous_pass_corrected":"The first short pass used an incorrect provisional cutoff and is not a completed G7 search. Collection resumed over 2024-02-01 through today.",
}
(OUT / "config.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
metrics = {"candidate_rows_screened":len(rows), "qualified_n":len(qual), "qualified_positive_n":sum(labels[r["material_id"]]["y"] == 1 for r in qual),
           "qualified_negative_n":sum(labels[r["material_id"]]["y"] == 0 for r in qual), "status":"collection_in_progress", "evaluation":None,
           "status_counts":dict(Counter(r["status"] for r in rows)),
           "label_file_sha256":hashlib.sha256((OUT / "labels.json").read_bytes()).hexdigest()}
(OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf8")
(OUT / "run.log").write_text(f"G7 collection started {manifest['created_utc']}\nscreened={len(rows)} qualified={len(qual)} positives={metrics['qualified_positive_n']}\nSTOP: eligible <60 or positives <15; no LLM calls; no evaluation.\n", encoding="utf8")
(OUT / "README.md").write_text("G7 post-cutoff material collection in progress.\nLlama 3.1 cutoff Dec 2023; conservative eligibility starts 2024-02-01.\nCandidate rows are in predictions.csv; labels are separate in labels.json.\nCurrent numbers are provisional, not a completed search.\nNo evaluation started; original short-pass stopping decision withdrawn.\nF4 source data were read-only and reused for candidate provenance.\n", encoding="utf8")
print(f"rows={len(rows)} qualified={len(qual)} positives={metrics['qualified_positive_n']}")
