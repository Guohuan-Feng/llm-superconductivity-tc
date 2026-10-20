"""Finite F4 literature collection; no model calls, fits, or import-time writes.

All candidate decisions require primary experimental evidence, earliest public
report dates, and composition/parent-family checks against the frozen 3764.
Literature search is allowed; the evaluated models never receive this ledger.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timezone
import csv
import json
from pathlib import Path

from common import DATA, FIELDS, ROOT, load_data, safe_path, write_json
from f1_inputs import _cohort_formula_index, formula_amounts, reduced_composition_key, sha256_file


OUT = ROOT / "F4_novel"
CUTOFF = date(2026, 4, 30)
EARLIEST = date(2026, 5, 30)
LATEST = date(2026, 10, 17)
OFFICIAL_MODEL_URL = "https://developers.openai.com/api/docs/models/gpt-6.1-sol"


def now():
    return datetime.now(timezone.utc).isoformat()


def checked_destination(path):
    resolved = safe_path(path)
    for ancestor in (resolved, *resolved.parents):
        if ancestor.exists() and (ancestor.is_symlink() or ancestor.is_junction()):
            raise ValueError(f"Linked destination requires explicit resolution review: {ancestor}")
    return resolved


def paper(key, title, url, first_date=None, **extra):
    return dict(paper_id=key, title=title, primary_url=url,
                earliest_public_report_date=first_date, **extra)


def initial_papers():
    """Human-reviewed search inventory, never an automatically accepted cohort."""
    return [
        paper("qcarbon", "Record BCS high temperature superconductivity in B-doped Q-carbon",
              "https://doi.org/10.1080/21663831.2026.2676634", "2026-06-08",
              date_status="publisher/author repository date seen; earlier manuscript dated March2026 needs checking",
              experimental_ambient=True, status="pending_first_report_and_family",
              evidence="QB33 and QB50 report ambient-pressure transitions112/164K; nanostructuring120/187K is not a second composition.",
              auxiliary_urls=["https://doi.org/10.6084/m9.figshare.32608354.v1"]),
        paper("moire", "Discovery of Superconductivity in a Bulk Moire Superlattice Material",
              "https://arxiv.org/abs/2609.06622", "2026-09-06", date_status="arXiv v1 history opened",
              experimental_ambient=True, status="pending_exact_composition_and_earlier_material_report",
              evidence="Bulk (Sr6TaS8)1+x(TaS2)8 reportsTc2.5K at ambient pressure; x not yet extracted."),
        paper("trilayer_nickelate", "Superconductivity at the metal-insulator phase boundary in a bulk nickelate at ambient pressure",
              "https://arxiv.org/abs/2609.19686", "2026-09-17", date_status="arXiv v1 history opened",
              experimental_ambient=True, status="pending_compositions_Tc_family",
              evidence="(La,Pr)4Ni3O8 and (La,Y)4Ni3O8; low-volume nonpercolative signals and possible infinite-layer intergrowth prevent unqualified bulk labels."),
        paper("lan", "Bulk Superconductivity in Rocksalt LaN1-x",
              "https://arxiv.org/abs/2609.23213", "2026-09-19", date_status="primary manuscript date opened",
              experimental_ambient=True, status="pending_earlier_material_and_family",
              evidence="LaN0.94 bulk transport, heat capacity and magnetizationTc5.95K; high nitrogen pressure is synthesis, not measurement pressure."),
        paper("mg2rhh6", "Superconducting Hydride Mg2RhH6 Experimentally Achieved at Lower Pressure",
              "https://arxiv.org/abs/2608.15745", "2026-08-16", date_status="arXiv v1 history opened",
              experimental_ambient=False, status="excluded_high_pressure",
              evidence="Tc24K at30GPa and29K at53GPa; no ambient-pressure experimentalTc."),
        paper("nb2pd3te5", "Realization of Air-Stable Two-Dimensional Superconductor Nb2Pd3Te5 With Quasi-One-Dimensional Pair Density Modulation",
              "https://arxiv.org/abs/2608.25331", "2021-05-05", date_status="older arXiv v1 located",
              experimental_ambient=True, status="excluded_preexisting_material",
              evidence="2026 monolayer/bilayerTc0.6K does not make this a new formula; bulkNb2Pd3Te5 superconductivity reported2021.",
              older_primary_urls=["https://arxiv.org/abs/2105.01886", "https://doi.org/10.7566/jpsj.90.063705"]),
        paper("mpbbi", "New Superconductors in the PtPb3Bi Structure Type",
              "https://arxiv.org/abs/2606.26258", "2026-06-24", date_status="arXiv first-date metadata; earlier parent report must be checked",
              experimental_ambient=True, status="pending_parent_family",
              evidence="ICP-OES AuPb2.78Bi1.35/PdPb2.08Bi1.92/RhPb1.56Bi2.68; Tc4.9/4.2/3.4K; failed synthesis attempts cannot serve as nonSC materials.",
              journal_doi="10.1021/acs.chemmater.6c01826"),
        paper("ag2pd3s", "Unconventional Superconductivity in the Chiral Topological Semimetal Ag2Pd3S",
              "https://arxiv.org/abs/2606.29767", None, date_status="year1973only; no month/day invented", known_prior_report_year=1973,
              experimental_ambient=True, status="excluded_preexisting_material",
              evidence="2026Tc1.1K is a new characterization of an old superconductor;2017 single-crystal primary paperTc2.25K cites1973Tc1.13K.",
              older_primary_urls=["https://doi.org/10.1103/PhysRevB.95.184514"]),
        paper("bismuth_oxybromide", "Two new superconducting bismuth oxybromides: BaBiO2Br and YBi2O4Br",
              "https://doi.org/10.1088/1361-6668/aea819", "2026-09-01", date_status="publisher date; earlier material reports remain to check",
              experimental_ambient=True, status="pending_earlier_material_report",
              evidence="AmbientTcBaBiO2Br4.6K/YBi2O4Br3.6K;BiOBr controls nonSC; BaBiO2Br synthesis is already cited in2014 photocatalysis literature."),
        paper("zr2ru3si4", "Superconductivity in the transition metal silicide Zr2Ru3Si4 single crystals",
              "https://doi.org/10.1103/56tp-5xy3", "2013-10-22", date_status="older arXiv v1 opened in search",
              experimental_ambient=True, status="excluded_preexisting_material",
              evidence="2026 single-crystalTc5.0K follows2013 bulk-polycrystalTc5.5K.",
              older_primary_urls=["https://arxiv.org/abs/1310.5830", "https://doi.org/10.1088/0953-2048/27/2/025012"]),
        paper("la3in", "Free-Anionic Duality of Interstitial Electrons Drives Superconductivity in Electride La3In",
              "https://doi.org/10.1021/jacs.6c05978", "2026-06-23", date_status="publisher date; earliest material report pending",
              experimental_ambient=True, status="pending_earlier_material_report",
              evidence="ReportedTc9.4K and oxygen-treated suppression; exact oxygen control compositions not specified in viewed abstract."),
        paper("scau2in", "Superconductivity in the full-Heusler compound ScAu2In with high-order topological invariants",
              "https://doi.org/10.1088/1361-6668/ae7bcb", "2026-06-10", date_status="publisher date; known prior material in abstract",
              experimental_ambient=True, status="excluded_preexisting_material",
              evidence="NewTc1.3K characterization revisits knownScAu2In; older apparent3K transition was attributed toIn."),
        paper("yru3b2_luru3b2", "Machine Learning-Guided Discovery of Kagome Superconductors YRu3B2 and LuRu3B2",
              "https://arxiv.org/abs/2512.16945", None, date_status="arXiv2512identifier establishes2025December first version; exact day pending",
              experimental_ambient=True, status="excluded_before_window",
              evidence="2026June journal/news date is not first report; YRu3B2Tc0.81K/LuRu3B2Tc0.95K have2025preprint."),
        paper("reo3", "Discovery of record-breaking high-Tc superconductivity among5d transition-metal-oxides up to17.8K inReO3",
              "https://doi.org/10.1093/nsr/nwag425", "2026-07-10", date_status="publisher date",
              experimental_ambient=False, status="excluded_high_pressure",
              evidence="Superconducting dome12-39GPa;17.8K maximum is not ambient-pressureTc."),
        paper("lab_hydride", "Pm-3n(La,Ce)4H23 Phase: A Low-Pressure Hydride Superconductor near Liquid-Nitrogen Temperature",
              "https://doi.org/10.1021/jacs.6c04117", "2026-06-03", date_status="publisher date",
              experimental_ambient=False, status="excluded_high_pressure",
              evidence="Low-pressure in this title refers to reduced multigigapascal pressure, not ambient."),
        paper("tellurium_hydride", "Superconductivity of Tellurium Polyhydride withTc above90K",
              "https://arxiv.org/abs/2608.30139", "2026-08-31", date_status="arXiv v1 metadata",
              experimental_ambient=False, status="pending_pressure_confirmation",
              evidence="Polyhydride high-pressure candidate; no verified ambient experimentalTc found."),
        paper("borophene", "Bilayer Borophenes Establish a New Upper Limit for Elemental Superconducting Transition Temperatures",
              "https://doi.org/10.1103/8l19-rdn2", "2026-08-04", date_status="publisher date",
              experimental_ambient=False, status="pending_experimental_confirmation",
              evidence="Must distinguish computed pairing temperatures from an experimental material; not accepted from title."),
        paper("boron_superatoms", "Toward ambient-pressure superconductivity from boron icosahedral superatoms",
              "https://doi.org/10.1103/yvcr-2yj8", "2026-09-11", date_status="publisher date",
              experimental_ambient=False, status="excluded_theoretical",
              evidence="First-principles structure prediction and predictedTc up to42K, not experimentalTc."),
        paper("b13se", "AI-accelerated metallized sigma-bonding screening for superconductor discovery",
              "https://arxiv.org/abs/2606.21251", None, date_status="arXiv2606identifier; exact day pending",
              experimental_ambient=False, status="excluded_theoretical",
              evidence="B13Se andB13X predictedTc above40K; no measuredTc in viewed source."),
        paper("multilayer_nickelate", "Superconducting phase diagram of multilayer square-planar nickelates",
              "https://doi.org/10.1126/science.adp4440", "2026-06-25", date_status="publisher date; earlier preprint pending",
              experimental_ambient=True, status="pending_earliest_report_and_family",
              evidence="Nd(n+1)Ni(n)O(2n+2),n4-8 show superconducting signatures; exactTc and earlier public report needed."),
        paper("bilayer_films", "Bridging ambient- and high-pressure superconductivity inLa2LnNi2O7 films",
              "https://www.nature.com/articles/s41563-026-02695-3", "2026-08-18", date_status="publisher date and arXiv2608.17745; earlier material reports pending",
              experimental_ambient=True, status="pending_compositions_first_report_and_family",
              evidence="Lanthanide-substituted bilayer films; material novelty must be checked beyond new transport characterization."),
        paper("cuprate_singlelayer", "Superconducting2D cuprate with a singleCuO2 plane",
              "https://www.nature.com/articles/s41586-026-10857-1", "2026-08-12", date_status="publisher date",
              experimental_ambient=True, status="pending_specific_parent_family_and_first_report",
              evidence="Need the exact parent compound and first material report; broadCu-O membership is insufficient to prove a duplicate family."),
        paper("hg_quench", "Ambient-pressure151-K superconductivity inHgBa2Ca2Cu3O8+delta via pressure quench",
              "https://arxiv.org/abs/2603.12437", "2026-03-12", date_status="arXiv v1 history opened",
              experimental_ambient=True, status="excluded_before_window",
              evidence="March first report precedes theMay30eligibility threshold; laterJune news is ineligible."),
        paper("scu_unconfirmed", "SCU Superconductor La3Nb2B2",
              "https://doi.org/10.5281/zenodo.22240200", "2026-09-02", date_status="repository date",
              experimental_ambient=False, status="excluded_unconfirmed_computational_claim",
              evidence="Author describes a prospective computational candidate awaiting independent experimental validation."),
        paper("c2h2b6be", "AI-Guided Discovery of a Predicted Room-Temperature Ambient-Pressure Superconductor:C2H2B6Be",
              "https://figshare.com/articles/preprint/33351291", "2026-08-27", date_status="repository date",
              experimental_ambient=False, status="excluded_theoretical",
              evidence="Predicted300KTc from computational search, no measured superconductivity."),
    ]


def initial_candidates():
    """Exact compositions seen in primary sources; labels are stored separately."""
    values = [
        ("qcarbon33", "B0.33C0.67", "qcarbon", "boron_doped_Q_carbon", 112., True),
        ("qcarbon50", "B0.5C0.5", "qcarbon", "boron_doped_Q_carbon", 164., True),
        ("lan094", "LaN0.94", "lan", "rare_earth_mononitride", 5.95, True),
        ("aupbbi", "AuPb2.78Bi1.35", "mpbbi", "PtPb3Bi_structure_type", 4.9, True),
        ("pdpbbi", "PdPb2.08Bi1.92", "mpbbi", "PtPb3Bi_structure_type", 4.2, True),
        ("rhpbbi", "RhPb1.56Bi2.68", "mpbbi", "PtPb3Bi_structure_type", 3.4, True),
        ("ag2pd3s", "Ag2Pd3S", "ag2pd3s", "Ag2Pd3S_filled_beta_Mn", 1.1, True),
        ("nb2pd3te5", "Nb2Pd3Te5", "nb2pd3te5", "M2Pd3Te5", .6, True),
        ("babio2br", "BaBiO2Br", "bismuth_oxybromide", "Sillen_bismuth_oxybromide", 4.6, True),
        ("ybi2o4br", "YBi2O4Br", "bismuth_oxybromide", "Sillen_bismuth_oxybromide", 3.6, True),
        ("zr2ru3si4", "Zr2Ru3Si4", "zr2ru3si4", "Zr2Ru3Si4", 5., True),
        ("la3in", "La3In", "la3in", "La3In_electride", 9.4, True),
        ("scau2in", "ScAu2In", "scau2in", "full_Heusler_ScAu2In", 1.3, True),
        ("yru3b2", "YRu3B2", "yru3b2_luru3b2", "CeCo3B2_kagome_boride", .81, True),
        ("luru3b2", "LuRu3B2", "yru3b2_luru3b2", "CeCo3B2_kagome_boride", .95, True),
    ]
    candidates, labels = [], {}
    for key, formula, source, family, tc, ambient in values:
        candidates.append(dict(id="F4_candidate_" + key, formula=formula, paper_id=source,
                               parent_family_key=family, eligibility="pending",
                               ambient_measurement_verified=ambient, first_report_verified=False,
                               parent_family_dedup_status="pending_manual_primary_evidence",
                               space_group=None, cif_url=None, structure_summary=None))
        labels["F4_candidate_" + key] = dict(Tc_K=tc, y=int(tc > 10),
                                              measurement_pressure_GPa=0. if ambient else None,
                                              pressure_semantics="ambient as reported, not exact pressure metrology")
    return candidates, labels


def cohort_index():
    data = load_data()
    formulas, composition_keys, facts_sha = _cohort_formula_index(data)
    return formulas, composition_keys, dict(
        n_rows=len(formulas), n_reduced_compositions=len(composition_keys),
        accepted_data_sha256=sha256_file(DATA), common_facts_sha256=facts_sha,
        source="f1_inputs._cohort_formula_index; original common facts and accepted_train read-only")


def attach_exact_dedup(candidates, keys):
    for record in candidates:
        key = reduced_composition_key(record["formula"])
        record["reduced_composition_key"] = list(key)
        record["matching_accepted_ids"] = keys.get(key, [])
        if record["matching_accepted_ids"]:
            record["eligibility"] = "excluded_exact_reduced_composition"


def summary(cfg):
    candidates = cfg["candidate_materials_not_for_prompt"]
    labels = cfg["evaluation_labels_not_for_prompt"]
    accepted = [r for r in candidates if r["eligibility"] == "accepted"]
    classes = Counter(labels[r["id"]]["y"] for r in accepted)
    output = dict(
        status=cfg["status"], updated_utc=now(), stage="F4_literature_collection",
        knowledge_cutoff=cfg["knowledge_cutoff"],
        collection_window=cfg["eligibility_window"],
        candidate_papers=len(cfg["candidate_papers_not_for_prompt"]),
        candidate_materials=len(candidates), accepted_n=len(accepted),
        accepted_groups=len({r["parent_family_key"] for r in accepted}),
        accepted_positive_n=classes.get(1, 0), accepted_negative_n=classes.get(0, 0),
        material_status_counts=dict(Counter(r["eligibility"] for r in candidates)),
        paper_status_counts=dict(Counter(r["status"] for r in cfg["candidate_papers_not_for_prompt"])),
        search_ledger=cfg["search_ledger"],
        gates=dict(target_n=60, target_min_per_class=20, hard_minimum_n=40,
                   permits_evaluation=(len(accepted) >= 40 and classes.get(0, 0) >= 20 and classes.get(1, 0) >= 20)),
        evaluated_LLM_calls=0, fitted_baselines=0, predictions_available=False,
        development_or_external="No external performance result exists until an eligible frozen cohort is evaluated",
        limits=["Declared model knowledge cutoff is not an auditable guarantee of absent post-cutoff information.",
                "An old material with a new2026 characterization is not accepted as a newly reported composition.",
                "Absence from3764 alone does not establish absence from prior literature or model training.",
                "No nonSC label is inferred from a failed synthesis, unreportedTc, or theoretical prediction.",
                "No graph-onlyTc is digitized or exact stoichiometry guessed without source evidence."])
    inventory = cfg.get("arxiv_primary_inventory_not_for_prompt")
    if inventory:
        output["arxiv_bibliographic_coverage"] = {
            key: inventory[key] for key in ("source_url", "total_results", "returned_entries", "month_counts",
                                           "boundary", "discovery_candidate_n")}
        output["arxiv_bibliographic_coverage"]["manual_screening_status_counts"] = inventory.get("manual_screening_status_counts", {})
        output["arxiv_discovery_candidate_reviews_not_predictions"] = inventory["screening_candidates"]
    output["qualified_collection_rows_not_evaluated"] = [
        {"id": row["id"], "formula": row["formula"], "first_public_report_date": row["first_public_report_date"],
         "parent_family": row["parent_family_key"], "primary_url": row["first_report_source"],
         **labels[row["id"]]}
        for row in accepted]
    if cfg.get("collection_completion"):
        output["collection_completion"] = cfg["collection_completion"]
        output["source_helper_hashes"] = cfg["source_helper_hashes"]
        output["evaluation_result_tables"] = cfg["evaluation_result_tables"]
        output["report_markdown"] = cfg["report_markdown"]
    return output


def save(cfg):
    assert len(cfg["accepted_features_for_future_evaluation"]) == sum(
        r["eligibility"] == "accepted" for r in cfg["candidate_materials_not_for_prompt"])
    for feature in cfg["accepted_features_for_future_evaluation"]:
        assert set(feature) <= {"id", "formula", "structure_summary", "space_group", "cif_url"}
    for record in cfg["candidate_materials_not_for_prompt"]:
        if record["eligibility"] != "accepted":
            continue
        first_date = date.fromisoformat(record["first_public_report_date"])
        assert EARLIEST <= first_date <= LATEST and record["first_report_verified"]
        assert not record["matching_accepted_ids"] and record["ambient_measurement_verified"]
        assert record["parent_family_dedup_status"] == "verified_absent_by_necessary_family_composition"
        label = cfg["evaluation_labels_not_for_prompt"][record["id"]]
        assert label["Tc_K"] is not None or label.get("reported_nonSC") is True
    cfg["collection_code_sha256"] = sha256_file(Path(__file__))
    write_json(checked_destination(OUT / "config.json"), cfg)
    metrics = summary(cfg)
    write_json(checked_destination(OUT / "metrics.json"), metrics)
    lines = ["# F4 finite novel-material collection", f"Status: {cfg['status']}; no evaluated calls or fitted baselines.",
             "Official gpt-6.1-sol cutoff2026-04-30; primary source recorded inconfig.json.",
             "Earliest first public report2026-05-30(calendar+1month); latest2026-10-17.",
             "Only ambient-pressure experimentalTc or explicitly measured nonSC; positiveTc>10K.",
             "Target>=60materials and>=20perclass; hardstopifeligibleN<40; maxoneworkday8h.",
             "Candidate paper/material records are not accepted evaluation samples until every gate is verified.",
             "Old formulas with new thickness/process/Tc characterization remain ineligible as newly reported materials.",
             "Reduced-formula and explicit parent-family dedup against frozen3764 are both required.",
             "Evaluation feature whitelist is separate from evaluation_labels_not_for_prompt and literature evidence.",
             "No labels, dates,DOIs,novelty/status fields may enter scientific model prompts.",
             "Failed synthesis, missingTc and theoreticalTc do not become nonSC labels.",
             "Repeated morphologies/pressures of one composition do not inflate the number of novel materials.",
             "Predictions.csv remains header-only and llm_calls.jsonl empty during literature collection.",
             "No E0-E4/P1/F0-F3 source outputs are modified; no deprecated stage5PDF is cited.",
             "No model-evaluation request starts whileF3is running; collection is a separate finite task.",
             "No research route is selected automatically; no ongoinggoal,heartbeat or recurringtask."]
    if cfg["status"] == "collection_complete_insufficient":
        lines[13] = "Frozen with eligible N=5, 3 parent families, 0 positives and 5 negatives; N<40 stops evaluation."
        lines[15] = "A2/A3/recall and all baselines were not run; all performance estimates and CIs are unavailable."
    assert len(lines) <= 20
    checked_destination(OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics


def apply_primary_reviews():
    """Apply source-backed reviews without changing prompts or evaluating models."""
    cfg = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
    assert not cfg["model_evaluation_permitted"]
    if not cfg.get("cohort_formula_lookup_not_for_prompt"):
        formulas, keys, cohort = cohort_index()
        cfg["cohort_formula_lookup_not_for_prompt"] = formulas
        assert cohort == cfg["cohort_dedup_reference"]
    else:
        formulas = cfg["cohort_formula_lookup_not_for_prompt"]
        keys = {}
        for material_id, formula in formulas.items():
            keys.setdefault(reduced_composition_key(formula), []).append(material_id)
    papers = {row["paper_id"]: row for row in cfg["candidate_papers_not_for_prompt"]}
    candidates = {row["id"]: row for row in cfg["candidate_materials_not_for_prompt"]}
    papers["cuprate_singlelayer"].update(status="pending_specific_parent_family_and_first_report",
        evidence="Exact parent material/structure and earlier first reports must be checked; broadCu-O presence in3764is insufficient for a duplicate-family conclusion.")
    papers["multilayer_nickelate"].update(status="excluded_before_window",
        earliest_public_report_date="2026-02-22", date_status="primary arXivv1 history confirmed",
        older_primary_urls=["https://arxiv.org/abs/2602.19093"],
        evidence="ScienceJune25publication followsarXivv1February22; not a post-cutoff first report.")
    papers["qcarbon"].update(date_status="Crossref online andFigsharev1June8timestamps verified; Marchin title is not a public date",
        date_evidence_urls=["https://api.crossref.org/works/10.1080/21663831.2026.2676634",
                            "https://api.figshare.com/v2/articles/32608354/versions/1"],
        prior_material_evidence_urls=["https://www.tandfonline.com/doi/full/10.1080/21663831.2019.1569566",
                                      "https://www.mdpi.com/2311-5629/10/1/14"],
        evidence="QB50(50at%boron)material predates2026, with2019primary synthesis/transport above100K. QB33specific33at%phase still pending earliest-report and structure-family review.")
    candidates["F4_candidate_qcarbon50"].update(eligibility="excluded_preexisting_material",
        exclusion_evidence="2019primary reported50at%boronQ-carbon;2026higherTc does not make new composition.")
    candidates["F4_candidate_qcarbon33"].update(first_public_report_date="2026-06-08",
        first_report_verified=True,
        composition_precision_note="Paper says33at%boron;B0.33C0.67 is reported nominal fraction, not a precise crystallographic occupancy measurement.",
        parent_family_dedup_status="pending_structure_family_specificity",
        parent_family_evidence="TRAIN containsB12C3 but boron carbide and amorphousQ-carbon cannot be declared identical structural parents merely fromB/Celements.")
    # Every duplicate decision remains independent of the current measuredTc.
    for record in candidates.values():
        parent = papers[record["paper_id"]]
        if record["matching_accepted_ids"]:
            record["eligibility"] = "excluded_exact_reduced_composition"
        elif parent["status"].startswith("excluded_"):
            record["eligibility"] = parent["status"]
            record["exclusion_evidence"] = parent["evidence"]
    family_elements = {"Pt", "Au", "Pd", "Rh"}
    matching_family = [(material_id, formula) for material_id, formula in formulas.items()
                       if {"Pb", "Bi"} <= set(formula_amounts(formula))
                       and family_elements & set(formula_amounts(formula))]
    cfg["specific_parent_family_checks_not_for_prompt"] = {
        "PtPb3Bi_structure_type": dict(
            source="https://arxiv.org/html/2606.26258v1",
            source_defined_parent="MPb4-xBix isostructural withPtPb3Bi;M=Pt/Au/Pd/Rh",
            necessary_composition_test="Pb andBi and at least one ofPt/Au/Pd/Rh in every formula among all3764",
            matching_accepted_id_formula_pairs=matching_family,
            interpretation="Absence of these necessary elements together excludes this specific parent series from3764; no broadmetal/oxide classification is used.")}
    assert not matching_family
    for suffix, refined, a, c, deposit in (
        ("aupbbi", "AuPb2.68Bi1.32", 11.5012, 4.297, "2562542"),
        ("pdpbbi", "PdPb2Bi2", 11.4966, 4.09250, "2562544"),
        ("rhpbbi", "RhPb1.48Bi2.52", 11.57429, 3.95324, "2562543"),
    ):
        row = candidates["F4_candidate_" + suffix]
        row.update(eligibility="accepted", first_report_verified=True,
            first_public_report_date="2026-06-24", first_report_source="https://arxiv.org/abs/2606.26258v1",
            first_report_evidence="Primaryv1datedJune24and authors explicitly describe the three compounds as newly synthesized; earlier parentPtPb3Bi is not a3764member.",
            parent_family_dedup_status="verified_absent_by_necessary_family_composition",
            parent_family_check_key="PtPb3Bi_structure_type", space_group="P4_2/mnm",
            cif_deposit_id="CCDC" + deposit, cif_url=None,
            structure_summary=dict(lattice_a_A=a, lattice_b_A=a, lattice_c_A=c,
                                   angles_degrees=[90., 90., 90.], declared_space_group="P4_2/mnm"),
            refined_crystal_formula_not_a_separate_material=refined,
            composition_basis="ICP-OESempiricalbatchratio;SCXRDrefinedoccupanciesdiffer; same parent material counted once",
            structure_composition_alignment="Cell/SGfromsamecompositionfamily; fullCIFnotdownloaded; formula-vs-refinement discrepancy must be resolved before an HGB127 structural evaluation.")
    cfg["candidate_papers_not_for_prompt"] = list(papers.values())
    cfg["candidate_materials_not_for_prompt"] = list(candidates.values())
    cfg["accepted_features_for_future_evaluation"] = [
        {key: row[key] for key in ("id", "formula", "space_group", "structure_summary", "cif_url")}
        for row in candidates.values() if row["eligibility"] == "accepted"]
    cfg["search_ledger"].update(sources_reviewed_requested_result_slots=260,
                              completed_search_queries=23)
    cfg["last_primary_review_utc"] = now()
    metrics = save(cfg)
    with checked_destination(OUT / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(now() + " primary reviews:oldmaterial/date checks; exactand specificparentfamilydedup; accepted3negative/1family; no model calls\n")
    print(json.dumps({k: metrics[k] for k in ("status", "candidate_papers", "candidate_materials", "accepted_n", "accepted_groups", "accepted_positive_n", "accepted_negative_n", "material_status_counts")}, ensure_ascii=False))


def inventory_arxiv():
    """Read the primary bibliography and store metadata, not copied full papers."""
    from f4_arxiv_inventory import retrieve_inventory, discovery_candidates
    inventory = retrieve_inventory()
    selected = discovery_candidates(inventory)
    cfg = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
    assert not cfg["model_evaluation_permitted"]
    cfg["arxiv_primary_inventory_not_for_prompt"] = {
        **{key: value for key, value in inventory.items() if key != "entries"},
        "retrieved_utc": now(),
        "entries": [{key: value for key, value in row.items() if key != "abstract"}
                    for row in inventory["entries"]],
        "heuristic_description": "f4_arxiv_inventory.discovery_candidates; keyword discovery only; omitted rows are not scientifically excluded",
        "screening_candidates": [{**{key: value for key, value in row.items() if key != "abstract"},
                                  "review_status": "pending_primary_Tc_first_report_composition_and_parent_review"}
                                 for row in selected],
        "discovery_candidate_n": len(selected),
    }
    cfg["source_helper_hashes"] = {"f4_arxiv_inventory.py": sha256_file(ROOT / "f4_arxiv_inventory.py")}
    cfg["search_ledger"].update(sources_reviewed_requested_result_slots=325,
                              completed_search_queries=28,
                              root_additional_exa_requested_result_slots=30,
                              root_additional_exa_query_n=2,
                              low_Tc_workstream_exa_requested_result_slots=270,
                              low_Tc_workstream_successful_exa_queries=24,
                              low_Tc_workstream_schema_rejected_calls=3,
                              low_Tc_schema_rejected_requested_slots_not_counted=45)
    metrics = save(cfg)
    with checked_destination(OUT / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(now() + f" primaryarXivbibliographicinventory={inventory['returned_entries']};heuristiccandidates={len(selected)};noheuristicrejectionorLLMcalls\n")
    print(json.dumps({key: value for key, value in inventory.items() if key != "entries"}, ensure_ascii=False))
    print(json.dumps([dict(index=i, **row) for i, row in enumerate(selected)], ensure_ascii=True))


def review_arxiv_candidates():
    """Document manual abstract screening while keeping unresolved gates open."""
    cfg = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
    inventory = cfg["arxiv_primary_inventory_not_for_prompt"]
    by_id = {row["url"].split("/")[-1].split("v")[0]: row
             for row in inventory["screening_candidates"]}
    theory = {
        "2606.04821", "2606.15298", "2606.16195", "2606.28735", "2607.08474",
        "2607.11355", "2607.12213", "2607.15228", "2607.18095", "2607.22029",
        "2608.02362", "2608.04789", "2608.17505", "2609.02295", "2609.16143",
        "2609.17028", "2609.18113", "2609.29263", "2610.03601", "2610.18171", "2610.18286"}
    nonexperimental_data = {"2606.05472", "2606.12520", "2608.30588"}
    pressure = {"2608.03058", "2608.15745", "2608.18865", "2608.30139", "2609.22794"}
    old_characterization = {
        "2606.04559": "KnownSr-Nd infinite-layer cuprate dome; newHall/quantumcriticality characterization is not new material evidence.",
        "2606.20941": "La4Ni3O10strain/thickness study of known trilayer compound; no new exact chemical parent claimed.",
        "2606.24108": "KnownK2Cr3As3newNMR/topologicalphase study; superconducting field phases do not create new compositions.",
        "2606.27040": "KnownLa(n+1)Ni(n)O(3n+1)series revisited for structural transitions; no newTc-composition assignment in abstract.",
        "2607.06819": "KnownLi1-xNbO2with priorTc14K;CaH2reduction/newcrystalprocess not first material report; broadXPScomposition uncertainty.",
        "2607.27607": "KnownLa3Ni2O7structuraldiagram underpressure, not a new ambient-Tc material.",
        "2608.08083": "Paper explicitly selects existingPr1111phase-diagram compositions to optimizeHPsynthesis; knownparent/process modification.",
        "2608.25331": "Nb2Pd3Te5already2021primary;newmonolayer/bilayer thickness does not create new composition.",
        "2609.06622": "BulkSr-Ta-Smoire material alreadyarXiv2510.26880/NatureFeb2026;newSeptemberTc is a characterization of old material.",
        "2609.07510": "Vortexproperties of existingLaPrbilayer thinfilms; no first report of a new composition in abstract.",
        "2609.11686": "KnownCsV3Sb5order-parameter dynamics;not newcomposition.",
        "2609.19321": "NbSe2encapsulation/growth method andmonolayerdevice study; no new materialcomposition.",
        "2609.24843": "OldFeTematerial with new annealing/electronic-reconstruction study; no newparentcomposition claimed.",
        "2609.30677": "KnownBa-K-Bi-Oelectrochemicalgrowth process; no newparent material.",
        "2609.35091": "KnownNbNnewion-sputteringmethod/thinfilm study, not new composition.",
    }
    special = {
        "2606.03710": "Pd7MoTe2mechanochemical growth;pendingexactTc/earlierPd7MoTe2report/structureparent dedup.",
        "2606.03981": "SrEu/CaLiCuO2bipolar doping and>60Konset; pendingexactcompositions, earliestreport and specificinfinite-layerparent overlap.",
        "2606.04956": "Nb0.85Sc0.15Tc6.35K;June3arXivearlier thanAugustjournal;pendingearlierexactalloy and parentdedup.",
        "2606.11495": "TwistedtrilayerWSe2is a newgeometricdevice platform;pendingTc, earliestmaterial/stackingparent and formula-onlyinput adequacy; not multiple compositionrows for gatefillings.",
        "2606.26258": "ThreeacceptednewAu/Pd/Rh-Pb-Bicompositions, firstv1June24, allTc<5K;one specificstructurefamily.",
        "2606.28726": "CoZr2HxhasexactH-dependent series; pendingindividualHcontents/Tc, earliestreport andCoZr2specificparentoverlap.",
        "2607.01302": "Sr0.75ClNbS2withnewAAstacking;pendingexperimentalTc,earlierreport andspecificstructuralparent evidence.",
        "2607.05881": "RhGe4/IrGe4olderprimaryreports;IrSi4abstract-vs-methodssynthesisconflict unresolved. No row accepted merely from abstractTc.",
        "2607.10332": "NdEuNiO2Tc31Kx0.3;earlierchemical/MBEstudies mentioned, so firstcompositionreport must be checked.",
        "2608.01209": "IntercalatedTaSe2bilayerIsing report;pendingexactintercalantcomposition/Tc andspecificparent/firstreport.",
        "2608.25093": "PtBi1.96Te0.04Tc2.4K;partialTe substitutionrestorescentrosymmetry;pendingearliestexactmaterial andPtBi2parentstructure overlap.",
        "2609.00542": "NewLa2SrNi2O7-deltaexplicitlynonSC;pendingnumericoxygenstoichiometry andspecificbilayerparent/firstreport audit.",
        "2609.00574": "NewSm4Ni3O10-deltaexplicitlynonSCatambientandpressure;pendingoxygenstoichiometry andspecifictrilayerparent check.",
        "2609.11441": "NewFe0.68Pd0.80Tespinglasssemiconductor;abstractdoesnotexplicitlyestablishnonSCdown tospecifiedT, so cannot assignnegativeyet.",
        "2609.19686": "La3.56Y0.44Ni3O8~15K andLa1.8Pr2.2Ni3O8~7K signals;pendingearlierPrmaterial,specificparent andminor-intergrowthattribution.",
        "2609.20364": "Pb(2-x)Bi(x)Pdseriesx.2-1.8potentialmanydistinctcompositions;pendingindividualTc,earliestreports andPb2Pd/Bi2Pdparentcheck.",
        "2609.23213": "Specificduplicatefamilyevidence:TRAIN003033CeN0.87declaredSG225Fm-3m;paperLaN0.94sameLnNrocksaltmononitridefamily. Not based onbroadnitrideclass.",
        "2609.28249": "La-Ag-Gedealloyingchangescompositionandstructure;pendingexactpostgrowthstoichiometries,Tc andearlierparentpolymorphreport.",
        "2609.29354": "FourRealloysTc4.46/2.28/1.76/1.64K;threebinaryparentsknown earlier, medium-entropycompositionmightnew;pendingfirstreportpercomposition.",
        "2609.35004": "CeNi2As2CaBe2Ge2isomorph heavyfermionSC;pendingnumericTc andpriorreportofthisspecificstructure, not enough to infernoveltyfromnewSC.",
    }
    for arxiv_id, row in by_id.items():
        if arxiv_id in theory:
            status, reason = "not_an_experimental_material_Tc_source", "Primaryabstractdescribestheory/review/prediction; no newmeasuredmaterialTc or measurednonSC label."
        elif arxiv_id in nonexperimental_data:
            status, reason = "not_a_new_experimental_material_source", "AI/dataanalysis/methodology source, not a first experimentalTc report for a newcomposition."
        elif arxiv_id in pressure:
            status, reason = "not_a_new_ambient_experimental_Tc_source", "Reportednewphysicsrequiresappliedpressure; anyknownambientparents requireseparatefirst-report review."
        elif arxiv_id in old_characterization:
            status, reason = "known_material_characterization_not_accepted", old_characterization[arxiv_id]
        elif arxiv_id in special:
            status, reason = "reviewed_followup_gate_pending", special[arxiv_id]
        else:
            raise AssertionError("Everyheuristiccandidate must have anexplicitmanualreview:" + arxiv_id)
        row.update(review_status=status, review_reason=reason, reviewed_utc=now(),
                   review_basis="PrimaryarXivabstractread;fullprimarymethods/earliest/parent evidence only where independently noted")
    cfg["arxiv_primary_inventory_not_for_prompt"]["manual_screening_status_counts"] = dict(
        Counter(row["review_status"] for row in by_id.values()))
    cfg["search_ledger"].update(root_additional_exa_requested_result_slots=50,root_additional_exa_query_n=4)
    metrics = save(cfg)
    print(json.dumps(inventory["manual_screening_status_counts"],ensure_ascii=True))


def merge_followup_evidence():
    """Retain verified findings and unresolved candidates without inflating N."""
    cfg = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
    papers = {row["paper_id"]: row for row in cfg["candidate_papers_not_for_prompt"]}
    rows = {row["id"]: row for row in cfg["candidate_materials_not_for_prompt"]}
    labels = cfg["evaluation_labels_not_for_prompt"]
    formulas = cfg["cohort_formula_lookup_not_for_prompt"]
    keys = {}
    for material_id, formula in formulas.items():
        keys.setdefault(reduced_composition_key(formula), []).append(material_id)

    def add(key, formula, source, family, tc=None, nonSC=False, **extra):
        material_id = "F4_candidate_" + key
        row = dict(id=material_id, formula=formula, paper_id=source,parent_family_key=family,
            eligibility="pending", ambient_measurement_verified=extra.pop("ambient",True),
            first_report_verified=extra.pop("first_report_verified",False),
            parent_family_dedup_status="pending_specific_parent_and_earlier_report",
            space_group=None,cif_url=None,structure_summary=None,**extra)
        attach_exact_dedup([row],keys)
        rows[material_id] = row
        labels[material_id] = dict(Tc_K=tc,y=int(tc>10) if tc is not None else 0 if nonSC else None,
            reported_nonSC=nonSC,measurement_pressure_GPa=0.,
            pressure_semantics="ambient as reported or conventional no-applied-pressure apparatus; provenance retained")

    for key,title,url,first,status,evidence in (
        ("LiNbO2", "Synthesis of Bulk SuperconductingLiNbO2Crystals throughCaH2Reduction", "https://arxiv.org/abs/2607.06819", "2026-07-07", "excluded_preexisting_material", "Primarymethods/literaturereview statespriorLi1-xNbO2Tc14K;actualXPSLi.96+/-.225NbO1.83+/-.187withH/gradient uncertainty;do not splitprocesssamples."),
        ("trX4", "Chiral-StructuredSuperconductorsTrX4", "https://arxiv.org/abs/2607.05881", "2026-07-07", "reviewed_old_materials_and_unresolved_synthesis_conflict", "RhGe4known1984,IrGe4reported2023;IrSi4abstractTc2.5K conflictswithmethodssayingcrystalscouldnotbesynthesized;noacceptedIrSi4label."),
        ("sigma_re", "Superconductivity in structurallycomplexsigmaRe-Xandmediumentropyalloy", "https://arxiv.org/abs/2609.29354", "2026-09-24", "pending_first_report_per_composition", "BulkTc4.46/2.28/1.76/1.64KforRe.76V.24/Re.56Nb.44/Re.60Ta.40/Re.56Nb.19Ta.19V.06;binaryparentsearlierknown;MEApossiblynew."),
        ("ynbtizrhf", "Superconductivity ofYNbTiZrHfhighentropyalloy", "https://journals.aps.org/prb/pdf/10.1103/k1yr-t59z", "2026-08-11", "pending_first_public_artifact_date_and_purity", "NominalequiatomicTc6.1K,bccIm-3ma3.445A,11.4wt%Y-richhcpimpurity;OSFdatafilecreatedJuly15,notproofitwaspublicthendate;July10wasacceptednotpublished."),
        ("tihea_sc_al", "RobusthighfieldSCinScandAlsubstitutedTi.5(ZrNbTaHf).5", "https://doi.org/10.1063/5.0351820", "2026-09-14", "pending_full_methods_ambient_and_earliest_report", "Ti.475(ZrNbTaHf).475X.05X=Sc/AlTc~5.2K;bcc;publisherSeptember15vsDOISeptember14noted;fullpressuremethodsnotyetverified."),
        ("crhea", "CrbasedequiatomichighentropyalloySC", "https://journals.aps.org/prmaterials/abstract/10.1103/ngx6-vxlq", "2026-06-04", "pending_earliest_report_and_specific_parent", "EarlierthanAugustmuSRpaper2608.21273:June4Crseries(TiVTa).6Nb(.4-x)Crx;equiatomicx.2Tc2.59KvsAugustmagnetic2.33K;singlematerialnotduplicates."),
        ("cozrh", "DistinctRolesofHydrogeninCoZr2Hx", "https://arxiv.org/abs/2606.28726", "2026-06-26", "pending_lowH_earliest_report_and_composition_resolution", "FiveactualHcontents.040/.043/.045/.051/.054withoverlappinguncertainties;Tc5.8-6.1K,SG140I4/mcm,ordinaryPPMS;highH2.786ferrodoesnotbyitselfestablishnonSC."),
        ("pbbipd", "SuperconductivityinPb2-xBixPdSingleCrystals", "https://arxiv.org/abs/2609.20364", "2026-09-17", "pending_phase_specific_parent_and_Tc_per_composition", "Ninex.2-.1.8;x<1SG140samePb2PdTRAINparent;x>=1SG139requiresseparatephasefamilyevidence;weakSVFfor1.2/1.4/1.6notrelabelledasconfirmednonSC."),
        ("eta_carbide", "StructurecompositionandhighfieldSCinmetalrich eta carbide compounds", "https://www.physics.okayama-u.ac.jp/jeschke_homepage/papers/PhysRevMater10p050301.pdf", "2026-05-28", "excluded_before_window", "JournalprimaryPDFpublishedMay28precedesMay30gate,althoughlaterarXiv2606.00894wasinsidethequerywindow;heuristicomissionmanuallychecked."),
        ("organic_fese", "EthylSubstitutedAlkylammoniumIntercalatedbetaFeSe", "https://doi.org/10.1021/acs.inorgchem.6c03261", "2026-09-07", "pending_Tc_per_exact_composition_and_specific_parent", "ETMA.4Fe2Se2andDEDMA.4Fe2Se2reportedupto44K;needindependentTcandearliestpublicmaterialchecks;notexcludedmerelybyironclass."),
        ("nd_eu", "PLDsynthesizedNd1-xEuxNiO2thinfilms", "https://arxiv.org/abs/2607.10332", "2026-07-10", "pending_earlier_material_report", "x.3Tc31K;authorscomparewithpriorMBEandchemicalsynthesisofsamefamily;specificfirstreportmustbeverified."),
        ("bipolar_cuprate", "Bipolardopedsuperconductinginfinite-layercuprates", "https://arxiv.org/abs/2606.03981", "2026-06-02", "pending_exact_composition_and_specific_parent", "SrEuCuO2andCaLiCuO2+deltaonsetabove60K;carrierfilling.07isnotautomaticallyexactLiatomicstoichiometry."),
    ):
        papers[key] = paper(key,title,url,first,status=status,evidence=evidence,experimental_ambient=not status.startswith("excluded_before"),date_status="primarysourceorAPIread; qualifications retained in evidence")
    for suffix,formula,tc in (("re_v","Re0.76V0.24",4.46),("re_nb","Re0.56Nb0.44",2.28),("re_ta","Re0.60Ta0.40",1.76),("re_mea","Re0.56Nb0.19Ta0.19V0.06",1.64)):
        add(suffix,formula,"sigma_re","Re_sigma_phase",tc,first_public_report_date="2026-09-24")
    add("ynbtizrhf","Y0.2Nb0.2Ti0.2Zr0.2Hf0.2","ynbtizrhf","Y_Nb_Ti_Zr_Hf_bcc_HEA",6.1,
        nominal_composition_mixed_phase=True,purity_note="11.4wt%Yrichhcpimpurity;nominalbulkcomposition",first_public_report_date="2026-08-11")
    for element in ("Al","Sc"):
        add("tihea_"+element.lower(),"Ti0.475Zr0.11875Nb0.11875Ta0.11875Hf0.11875"+element+"0.05",
            "tihea_sc_al","Ti_rich_Zr_Nb_Ta_Hf_bcc_HEA",5.2,ambient=False,
            ambient_note="Abstractnotexplicit;fullmethodsneededbeforenegativelabelcouldentercohort.",first_public_report_date="2026-09-14")
    add("cr_equiatomic","Cr0.2V0.2Ti0.2Nb0.2Ta0.2","crhea","Cr_V_Ti_Nb_Ta_bcc_HEA",2.59,
        first_public_report_date="2026-06-04",same_material_later_measurements=[dict(source="https://arxiv.org/abs/2608.21273",Tc_K=2.33,criterion="magnetization"),dict(Tc_K=2.94,criterion="resistivity")])
    for h,tc,dh in ((.040,6.,.001),(.043,6.1,.001),(.045,6.,.003),(.051,6.,.001),(.054,5.8,.001)):
        add("cozrh"+format(h,".3f").replace(".",""),"CoZr2H"+format(h,".3f"),"cozrh","CoZr2_hydrogenated",tc,
            composition_uncertainty_H=dh,composition_resolution_note="OverlappingHuncertaintyanddifferentprocessdurations;notallfivesilentlycountedasindependentnewmaterials.",first_public_report_date="2026-06-26")
    for x in (.2,.4,.6,.8,1.,1.2,1.4,1.6,1.8):
        add("pbbipd"+format(x,".1f").replace(".",""),"Pb"+format(2-x,".1f")+"Bi"+format(x,".1f")+"Pd","pbbipd","Pb2Pd_or_betaBi2Pd_substitution_phase",None,
            phase_space_group="I4/mcm"if x<1 else"I4/mmm",first_public_report_date="2026-09-17")
        if x<1:
            rows["F4_candidate_pbbipd"+format(x,".1f").replace(".","")].update(eligibility="excluded_specific_parent_family",
                parent_family_dedup_status="specific_parent_match",parent_family_evidence="TRAIN3dsc_mp_row_005256Pb4Pd2=Pb2Pd;declaredSG140I4/mcm;paperx<1sametetragonalparentSG140.")
    for ln,tc in (("Pr",47.),("Nd",42.),("Sm",41.),("Eu",10.),("Dy",20.)):
        add("bilayer_"+ln.lower(),"La2"+ln+"Ni2O7","bilayer_films","La2LnNi2O7_bilayer_RP",tc,
            label_criterion="resistiveonset",first_public_report_date="2026-08-18",
            alternate_Tc_definitions=dict(resistive_zero_K=6.,resistive10percent_K=28.)if ln=="Sm"else{},
            eligibility_note="Differentlayer-count/specificparentnotautomaticallyequatedwithTRAINLa2NiO4; SmTccriterioncrosses10K.")
    add("trilayer_y","La3.56Y0.44Ni3O8","trilayer_nickelate","La4Ni3O8_trilayer_infinite_layer",15.,first_public_report_date="2026-09-17",phase_attribution_uncertain=True)
    add("trilayer_pr","La1.8Pr2.2Ni3O8","trilayer_nickelate","La4Ni3O8_trilayer_infinite_layer",7.,first_public_report_date="2026-09-17",phase_attribution_uncertain=True,earlier_Pr_material_reports_pending=True)
    add("nd_eu03","Nd0.7Eu0.3NiO2","nd_eu","Nd_Eu_infinite_layer_nickelate",31.,first_public_report_date="2026-07-10")
    cfg["specific_parent_family_checks_not_for_prompt"]["LnN_rocksalt_mononitride"] = dict(
        candidate="LaN0.94",primary_source="https://arxiv.org/abs/2609.23213",source_defined_parent="RareearthmononitrideLnNrocksaltseries",
        matching_accepted_id_formula_pairs=[["3dsc_mp_row_003033","Ce1N0.87"]],declared_TRAIN_space_group=dict(symbol="Fm-3m",number=225),
        structure_evidence_source="F3_full3764/config.json:cases.3dsc_mp_row_003033.material.structure.space_group_declared_in_source_CIF;frozencommonfacts")
    rows["F4_candidate_lan094"].update(eligibility="excluded_specific_parent_family",parent_family_dedup_status="specific_parent_match",parent_family_check_key="LnN_rocksalt_mononitride")
    papers["cuprate_singlelayer"].update(status="excluded_unchanged_parent_thickness",
        evidence="OpenedNatureabstractexplicitlyexaminessinglelayerBi2Sr2CuO6+delta(Bi2201);newthickness/gatingofknownBi2201,notnewchemicalparent;notexcludedbybroadCu-Oclass.")
    papers["moire"].update(status="excluded_preexisting_material",earliest_public_report_date="2025-10-30",
        older_primary_urls=["https://arxiv.org/abs/2510.26880","https://doi.org/10.1038/s41586-026-10173-8"],
        evidence="BulkSrTaSmoirecompoundfirstpreprintOct2025andNatureFebruary2026;September2.5Kcharacterizationnotnewmaterial.")
    cfg["candidate_papers_not_for_prompt"] = list(papers.values())
    cfg["candidate_materials_not_for_prompt"] = list(rows.values())
    cfg["search_ledger"].update(high_Tc_workstream_web_search_queries=56,high_Tc_workstream_exa_requested_slots=0,
        web_query_count_not_added_to_exa_slots=True,
        definition="SuccessfulExarequestednumResultsresultslots,includingduplicates;notuniquepapers. PrimaryarXivAPI912metadataandwebqueriesareseparate.")
    metrics = save(cfg)
    print(json.dumps({k:metrics[k]for k in("candidate_papers","candidate_materials","accepted_n","accepted_positive_n","accepted_negative_n","material_status_counts")},ensure_ascii=True))


def final_verified_updates():
    """Apply finite primary checks, keeping a new experiment distinct from a new identity."""
    cfg=json.loads((OUT/"config.json").read_text(encoding="utf-8"))
    papers={r["paper_id"]:r for r in cfg["candidate_papers_not_for_prompt"]}
    rows={r["id"]:r for r in cfg["candidate_materials_not_for_prompt"]}
    formulas=cfg["cohort_formula_lookup_not_for_prompt"]
    family_checks=cfg["specific_parent_family_checks_not_for_prompt"]
    # Exact alloy identities can have mechanical/corrosion reports before a
    # later paper first characterizes their superconductivity.
    papers["crhea"].update(status="excluded_preexisting_material",known_prior_report_year=2023,
        earlier_material_source="https://doi.org/10.1016/j.jmrt.2023.11.247",
        evidence="2023primarycorrosionpaperexplicitlyfabricatedCrNbTaTiVbyarcmelting;June2026newSCandAugustmuSRarenotfirstmaterialidentityreports.")
    rows["F4_candidate_cr_equiatomic"].update(eligibility="excluded_preexisting_material",exclusion_evidence=papers["crhea"]["evidence"])
    for suffix in ("re_v","re_nb","re_ta"):
        rows["F4_candidate_"+suffix].update(eligibility="excluded_preexisting_material",
            exclusion_evidence="2026primarypaperstatesknownbinaryphases/priorSC,withRoberts1976andolderRe-Xphase-diagramreferences;notnewmaterialidentities.")
    family_checks["Re_V_Nb_Ta_sigma"] = dict(
        source="https://arxiv.org/html/2609.29354",source_defined_parent="Re-XsigmafamilyX=V/Nb/TaandthederivedReNbTaVMEA",
        necessary_composition_test="ReandoneofV/Nb/Ta,withnoelementsoutsidethisRe-V-Nb-Tafamily",
        matching_accepted_id_formula_pairs=[(mid,f)for mid,f in formulas.items()if"Re"in formula_amounts(f)
            and set(formula_amounts(f)) & {"V","Nb","Ta"}and set(formula_amounts(f)) <= {"Re","V","Nb","Ta"}])
    assert not family_checks["Re_V_Nb_Ta_sigma"]["matching_accepted_id_formula_pairs"]
    rows["F4_candidate_re_mea"].update(eligibility="accepted",first_report_verified=True,
        first_public_report_date="2026-09-24",first_report_source="https://arxiv.org/abs/2609.29354v1",
        first_report_evidence="NewderivedMEAinprimaryv1;exactcompositionearlier-materialsearchdidnotfindapriorreport;binaryparentdateskeptseparate.",
        parent_family_dedup_status="verified_absent_by_necessary_family_composition",parent_family_check_key="Re_V_Nb_Ta_sigma",
        space_group="P4_2/mnm",structure_summary=dict(lattice_a_A=9.7309,lattice_b_A=9.7309,lattice_c_A=5.0650,
            angles_degrees=[90.,90.,90.],declared_space_group="P4_2/mnm"),
        Tc_criterion="Table1heatcapacityTc=1.64K;PPMSand3Heinsert,noexternalpressure")
    family_checks["Y_Nb_Ti_Zr_Hf_bcc_HEA"] = dict(
        source="https://journals.aps.org/prb/pdf/10.1103/k1yr-t59z",source_defined_parent="Y-Nb-Ti-Zr-HfequiatomicrefractoryBCCHEA",
        necessary_composition_test="AtleastfourofY/Nb/Ti/Zr/Hfwithnoelementsoutsidethisspecificsystem",
        matching_accepted_id_formula_pairs=[(mid,f)for mid,f in formulas.items()
            if len(set(formula_amounts(f)) & {"Y", "Nb", "Ti", "Zr", "Hf"}) >= 4
            and set(formula_amounts(f)) <= {"Y", "Nb", "Ti", "Zr", "Hf"}])
    assert not family_checks["Y_Nb_Ti_Zr_Hf_bcc_HEA"]["matching_accepted_id_formula_pairs"]
    rows["F4_candidate_ynbtizrhf"].update(eligibility="accepted",first_report_verified=True,
        first_public_report_date="2026-08-11",first_report_source="https://journals.aps.org/prb/pdf/10.1103/k1yr-t59z",
        first_report_evidence="PrimarypublicationAugust11verified;earlierexactYalloyidentitysearchnotfound;YnotconfusedwithcommonTa-containingalloys.",
        public_artifact_date_boundary="OSFdatafilecreatedJuly15isnotitselfproofitwaspublicthen;bothidentifiedartifactcreationandpublicationpostdateMay30.",
        artifact_metadata_source="https://api.osf.io/v2/files/6a576c0bd8e95dcff47afad3/",
        parent_family_dedup_status="verified_absent_by_necessary_family_composition",parent_family_check_key="Y_Nb_Ti_Zr_Hf_bcc_HEA",
        space_group="Im-3m",structure_summary=dict(lattice_a_A=3.445,lattice_b_A=3.445,lattice_c_A=3.445,
            angles_degrees=[90.,90.,90.],declared_space_group="Im-3m"))
    family_checks["TrZr2_C16"] = dict(source="https://arxiv.org/pdf/2606.28726",
        source_defined_parent="Al2Cu/C16TrZr2framework,I4/mcmSG140;lowHCoZr2Hxremainsinparentframework",
        matching_accepted_id_formula_pairs=[["3dsc_mp_row_003777","Fe2Zr4"],["3dsc_mp_row_004466","Ir2Zr4"],["3dsc_mp_row_005475","Rh2Zr4"]],
        structure_evidence="AllthreefrozenF3materialsdeclareI4/mcmSG140;sourcepaperdescribeslowHasimpuritywithunchangedbands/framework.")
    for row in rows.values():
        if row["paper_id"]=="cozrh":
            row.update(eligibility="excluded_specific_parent_family",parent_family_dedup_status="specific_parent_match",parent_family_check_key="TrZr2_C16",first_public_report_date="2026-06-27")
    papers["cozrh"]["earliest_public_report_date"]="2026-06-27"
    papers["nd_eu"].update(status="excluded_preexisting_exact_composition",earliest_public_report_date="2025-11-27",
        earlier_primary_url="https://arxiv.org/pdf/2511.22026v1",
        evidence="v1alreadygivesNd.7Eu.3NiO2andotherEucompositionswith~36Kchemicalroute;2026PLDchangesgrowthprocess.")
    rows["F4_candidate_nd_eu03"].update(eligibility="excluded_preexisting_material",exclusion_evidence=papers["nd_eu"]["evidence"])
    papers["la6ni5o12"] = paper("la6ni5o12","BulkSCinquintuplelayerLa6Ni5O12","https://arxiv.org/abs/2610.20249","2026-10-14",
        status="excluded_identity_public_before_cutoff",experimental_ambient=True,
        earlier_primary_url="https://arxiv.org/abs/2311.13674",
        evidence="October14firstbulk18Kexperimentisnew,butpreciseLa6Ni5O12identity/structure/SCpredictionwaspublic2023;theOctober15VMCtheoryisnasecondmaterial.")
    rows["F4_candidate_la6ni5o12"] = dict(id="F4_candidate_la6ni5o12",formula="La6Ni5O12",paper_id="la6ni5o12",
        parent_family_key="quintuple_square_planar_La6Ni5O12",eligibility="excluded_identity_public_before_cutoff",matching_accepted_ids=[],
        first_report_verified=False,ambient_measurement_verified=True,parent_family_dedup_status="not_reached_due_prior_identity",
        space_group=None,cif_url=None,structure_summary=None,first_public_report_date="2026-10-14")
    cfg["evaluation_labels_not_for_prompt"]["F4_candidate_la6ni5o12"] = dict(Tc_K=18.,y=1,measurement_pressure_GPa=0.,criterion="onsetandheatcapacity;~25percentsuperconductingvolumefraction")
    cfg["candidate_papers_not_for_prompt"]=list(papers.values())
    cfg["candidate_materials_not_for_prompt"]=list(rows.values())
    cfg["accepted_features_for_future_evaluation"]=[{key:r[key]for key in("id","formula","space_group","structure_summary","cif_url")}for r in rows.values()if r["eligibility"]=="accepted"]
    cfg["search_ledger"].update(sources_reviewed_requested_result_slots=350,completed_search_queries=31,
        low_Tc_workstream_exa_requested_result_slots=285,low_Tc_workstream_successful_exa_queries=26)
    metrics=save(cfg)
    print(json.dumps({k:metrics[k]for k in("candidate_papers","candidate_materials","accepted_n","accepted_groups","accepted_positive_n","accepted_negative_n","material_status_counts")},ensure_ascii=True))


def final_supplement():
    cfg=json.loads((OUT/"config.json").read_text(encoding="utf-8"))
    papers={r["paper_id"]:r for r in cfg["candidate_papers_not_for_prompt"]}
    rows={r["id"]:r for r in cfg["candidate_materials_not_for_prompt"]}
    labels=cfg["evaluation_labels_not_for_prompt"]
    forms=cfg["cohort_formula_lookup_not_for_prompt"]
    keys={}
    for mid,f in forms.items(): keys.setdefault(reduced_composition_key(f),[]).append(mid)
    specs=[
        ("hfzrte3","Hf0.02Zr0.98Te3",3.3,"https://arxiv.org/html/2606.12598","2026-06-10",
         "excluded_specific_parent_family","SourceexplicitHfsubstitutioninZrTe3;TRAIN005718Te6Zr2=ZrTe3declaresSG11P21/m;notanexactnewformulamatch."),
        ("nbses","NbSeS",6.,"https://arxiv.org/html/2606.19107","2026-06-17",
         "excluded_specific_parent_family","SourceexplicitNbSe2-xSxx1;TRAIN004976Nb2S4SG194and004980Nb4Se8proxySG187areconcreteNb(S,Se)2parents;proxyNbSe2notmislabelledas2H."),
        ("w_tase2","Ta0.9W0.1Se2",2.82,"https://arxiv.org/html/2606.25416","2026-06-24",
         "excluded_preexisting_material","Exact3RTa.9W.1Se2alreadyarXiv1505.07336May27,2015andDOI10.1088/0953-8984/27/36/365701;newCVTsinglecrystalmethoddoesnotcreateformula."),
        ("rehea10","Nb0.57Re0.10Ti0.11Zr0.11Hf0.11",5.423,"https://arxiv.org/html/2607.23056","2026-07-25",
         "excluded_preexisting_material","2019arXiv1904.06203v1Figure1+2explicitx.10of[Nb(.67-x)Rex](HfZrTi).33;notfirstmaterialin2026."),
        ("rehea20","Nb0.47Re0.20Ti0.11Zr0.11Hf0.11",5.419,"https://arxiv.org/html/2607.23056","2026-07-25",
         "excluded_preexisting_material","2019arXiv1904.06203v1Figure1+2explicitx.20ofsamepreciseHEAseries;notfirstmaterialin2026."),
        ("rehea56","Nb0.11Re0.56Ti0.11Zr0.11Hf0.11",3.907,"https://arxiv.org/html/2607.23056","2026-07-25",
         "excluded_preexisting_material","2019arXiv1904.06203v1Figure1+2explicitx.56ofsamepreciseHEAseries;notfirstmaterialin2026."),
        ("ybbi2","YbBi2",.9,"https://arxiv.org/abs/2609.20093","2026-09-17",
         "excluded_preexisting_material","Primary2022May10PhysRevB105195114alreadygrewYbBi2singlecrystals;new2026SCcharacterizationdoesnotmakeanewidentity."),
    ]
    for key,f,tc,url,first,status,evidence in specs:
        papers[key]=paper(key,key,url,first,status=status,evidence=evidence,experimental_ambient=True,date_status="currentpaperdate;earlieridentityevidenceseparatelyrecorded")
        mid="F4_candidate_"+key
        row=dict(id=mid,formula=f,paper_id=key,parent_family_key=key,eligibility=status,
            first_report_verified=False,ambient_measurement_verified=True,parent_family_dedup_status="specific_parent_match"if"family"in status else"not_needed_after_prior_identity",
            space_group=None,cif_url=None,structure_summary=None,exclusion_evidence=evidence)
        attach_exact_dedup([row],keys)
        rows[mid]=row
        labels[mid]=dict(Tc_K=tc,y=int(tc>10),measurement_pressure_GPa=0.,criterion="PrimaryreportedTc;excludedrowsnotusedforevaluation")
    for key,f,a,c in(("etma","C2H5.6N0.4Fe2Se2",3.8230,21.069),("dedma","C2.4H6.4N0.4Fe2Se2",3.8119,21.042)):
        mid="F4_candidate_"+key
        row=dict(id=mid,formula=f,paper_id="organic_fese",parent_family_key="alkylammonium_intercalated_Fe2Se2",
            eligibility="pending",first_report_verified=False,ambient_measurement_verified=True,
            first_public_report_date="2026-09-07",parent_family_dedup_status="pending_specific_host_and_structure_family_review",
            space_group="I4/mmm",cif_url=None,structure_summary=dict(lattice_a_A=a,lattice_b_A=a,lattice_c_A=c,
                angles_degrees=[90.,90.,90.],declared_space_group="I4/mmm"),
            structure_source="https://acs.figshare.com/articles/journal_contribution/33456416 TableS3",
            Tc_note="Maintext/SIonlyup-to44KnotindividualnumericTc;notassigned44Ktoboth.",
            parent_family_evidence="PrimarySIstatesbetaFeSehostandThCr2Si2type;TRAIN003768Fe2Se1.96SG129and003769Fe2Se1.98SG129;also003698Fe1.63K.82Se2/003699Fe1.64K.79Se2I4/mmm139intercalatedFe2Se2structure;organicspacerfamilyboundarynotinferredfrombroadironcategory.")
        attach_exact_dedup([row],keys);rows[mid]=row
        labels[mid]=dict(Tc_K=None,y=None,measurement_pressure_GPa=0.,reported_Tc_upper_envelope_K=44.,criterion="IndividualTcnotresolved;notforevaluation")
    cfg["candidate_papers_not_for_prompt"]=list(papers.values())
    cfg["candidate_materials_not_for_prompt"]=list(rows.values())
    cfg["search_ledger"].update(low_Tc_workstream_exa_requested_result_slots=315,low_Tc_workstream_successful_exa_queries=32,
        all_successful_exa_requested_slots_including_duplicates=350+50+315,all_successful_exa_queries=31+4+32,
        broad_manual_title_screen_additional_omitted_entries=251,broad_manual_followup_abstracts=16)
    cfg["late_broad_screen_followups_not_for_prompt"]=[
        dict(arxiv_id="2608.16062",status="not_experimental",reason="YSi220.6KisMigdalEliashberg/DFTpredictionnotmeasuredTc"),
        dict(arxiv_id="2610.20249",status="prior_theoretical_identity",reason="Newbulk18KexperimentbutpreciseLa6Ni5O12identitypredicted2023"),
        dict(arxiv_id="2610.22212",status="theory_same_material",reason="VMC/HubbardstudyofsameLa6Ni5O12notasecondevaluationmaterial"),
        dict(arxiv_id="2608.27349",status="unresolved_formula_and_Tc",reason="Fe-Se-HexperimentalEPRclaimup-to295KhasnodeterminedHstoichiometryorcontinuousTc;notTc=295K"),
    ]
    metrics=save(cfg)
    print(json.dumps({k:metrics[k]for k in("candidate_papers","candidate_materials","accepted_n","accepted_groups","accepted_positive_n","accepted_negative_n","material_status_counts")},ensure_ascii=True))


def initialize():
    if OUT.exists():
        raise RuntimeError("F4 directory exists; never overwrite a prior collection silently")
    checked_destination(OUT).mkdir()
    candidates, labels = initial_candidates()
    formulas, keys, cohort = cohort_index()
    attach_exact_dedup(candidates, keys)
    cfg = dict(experiment="F4_novel", status="collecting_primary_literature",
        created_utc=now(), collection_max_seconds=8 * 60 * 60,
        knowledge_cutoff=dict(model="gpt-6.1-sol", cutoff_date=CUTOFF.isoformat(),
            source_url=OFFICIAL_MODEL_URL, source_section="Model details",
            source_quote="Apr 30, 2026 knowledge cutoff", verified_utc=now()),
        eligibility_window=dict(earliest=EARLIEST.isoformat(), latest=LATEST.isoformat(),
            calculation="2026-04-30 +1calendar month =2026-05-30; inclusive first-public-report date"),
        target=dict(n=60, min_positive=20, min_negative=20, stop_below_n=40),
        cohort_dedup_reference=cohort, scientific_input_sources_read_only=True,
        model_evaluation_permitted=False, no_model_calls_during_F3=True,
        label_separation="Accepted_features whitelist only; labels and literature evidence stay outside prompt serialization.",
        parent_family_policy="Material parent/structural family dedup must be supported by primary evidence; unresolved family matches remain pending rather than silently included.",
        search_ledger=dict(sources_reviewed_requested_result_slots=170,
            definition="Sum of numResults requested across Exa searches; includes duplicates and off-topic results, not170unique sources.",
            completed_search_queries=16,
            workstreams=["Broad new experimental superconductors", "June/July/August/September/October discovery dates", "PositiveTc>10K", "Exact earlier-material report checks", "Composition series and nonSC controls"],
            failed_listing_fetches=["arxiv cond-mat.supr-con monthly lists2606-2610:ExaCRAWL_NOT_FOUND and independentweb404"],
            collection_is_exhaustive=False),
        candidate_papers_not_for_prompt=initial_papers(), candidate_materials_not_for_prompt=candidates,
        evaluation_labels_not_for_prompt=labels, accepted_features_for_future_evaluation=[],
        future_evaluation_plan_not_started=dict(LLM_arms=["A3", "R-recall", "A2_composition_neighbours"],
            nearest_neighbour_reference="full3764 composition-only features; exact feature subset/scaler must be frozen before any prompts",
            baselines=["CompositionHGB_fit3764", "HGB127_only_for_structure_available_subset"],
            approval_gate="root coordinator confirmsF3done and collection meets user criteria before any evaluation call"))
    with checked_destination(OUT / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=FIELDS).writeheader()
    checked_destination(OUT / "llm_calls.jsonl").write_text("", encoding="utf-8")
    checked_destination(OUT / "run.log").write_text(now() + " F4 literature collection initialized; no scientific model calls\n", encoding="utf-8")
    metrics = save(cfg)
    print(json.dumps({k: metrics[k] for k in ("status", "candidate_papers", "candidate_materials", "accepted_n", "material_status_counts")}, ensure_ascii=False))


def freeze_insufficient():
    """Close this finite collection without substituting unverified samples."""
    cfg = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
    if cfg["status"] == "collection_complete_insufficient":
        raise RuntimeError("F4 is already frozen; an independent audit must precede any revision")
    metrics = summary(cfg)
    assert metrics["accepted_n"] < cfg["target"]["stop_below_n"]
    assert not metrics["gates"]["permits_evaluation"]
    assert (OUT / "llm_calls.jsonl").stat().st_size == 0
    with (OUT / "predictions.csv").open(encoding="utf-8", newline="") as handle:
        assert not list(csv.DictReader(handle))
    cfg["status"] = "collection_complete_insufficient"
    cfg["model_evaluation_permitted"] = False
    cfg["search_ledger"].update(high_Tc_workstream_web_search_queries=76,
        high_Tc_workstream_exa_successful_queries=0,
        high_Tc_full_inventory_organic_carbon_abstract_screen=912,
        finite_assigned_followups_complete=True)
    for row in cfg["candidate_papers_not_for_prompt"]:
        if row["paper_id"] == "bipolar_cuprate":
            row["final_review_note"] = (
                "Ca-Li infinite-layer cuprate delta and exact composition are unresolved; "
                "carrier filling is not an exact Li coefficient. The Sr-Eu side has an older "
                "specific parent series (PhysRevMaterials.7.074802, 2023). No exact new sample added.")
        elif row["paper_id"] == "qcarbon":
            row["final_review_note"] = (
                "QB50 is a preexisting identity. QB33 remains pending: no earlier exact 33-at-percent "
                "composition found in the finite search, but first-report and structural-family "
                "eligibility have not both been established. Not counted as a qualified positive.")
    cfg["arxiv_primary_inventory_not_for_prompt"]["review_status_semantics"] = (
        "The 64 discovery-candidate statuses record the first primary-abstract screen. "
        "Later composition-specific resolutions are in candidate_papers_not_for_prompt and "
        "candidate_materials_not_for_prompt; an initial pending status is not an accepted sample.")
    finished = now()
    created = datetime.fromisoformat(cfg["created_utc"])
    elapsed = (datetime.fromisoformat(finished) - created).total_seconds()
    assert 0 <= elapsed < cfg["collection_max_seconds"]
    pending = metrics["material_status_counts"].get("pending", 0)
    cfg["collection_completion"] = dict(
        completed_utc=finished, elapsed_seconds_from_manifest_creation=elapsed,
        first_known_search_anchor_utc="2026-10-17T20:03:10+00:00",
        elapsed_seconds_from_first_known_search_anchor=(datetime.fromisoformat(finished) - datetime.fromisoformat("2026-10-17T20:03:10+00:00")).total_seconds(),
        reason="Eligible N<40 and class targets are unmet after the finite assigned collection.",
        accepted_n=metrics["accepted_n"], accepted_groups=metrics["accepted_groups"],
        accepted_positive_n=metrics["accepted_positive_n"], accepted_negative_n=metrics["accepted_negative_n"],
        pending_exact_composition_records=pending,
        accepted_plus_pending_exact_record_upper_bound=metrics["accepted_n"] + pending,
        bound_semantics="Only an upper bound for the current exact-composition manifest; not an exhaustive bound on the literature.",
        further_unbounded_collection_stopped=True, no_scientific_model_calls=True,
        no_baseline_fits=True, no_performance_or_bootstrap_estimates=True,
        workday_limit_seconds=cfg["collection_max_seconds"],
        followup_decision="No route selected automatically; wait for the user's next instruction.")
    cfg["source_helper_hashes"] = {
        name: sha256_file(ROOT / name)
        for name in ("f4_collect.py", "f4_arxiv_inventory.py", "common.py", "f1_inputs.py", "e2_packets.py")}
    method_names = ("A2", "A3", "R-recall", "CompositionHGB_fit3764", "HGB127_structure_subset")
    cfg["evaluation_result_tables"] = [dict(
        method=method, subset="all", weighting=weighting,
        eligible_collection_n=metrics["accepted_n"], eligible_collection_groups=metrics["accepted_groups"],
        n_evaluated=0, groups_evaluated=0,
        status="not_run_insufficient_eligible_materials",
        BA=None, precision=None, recall=None, accuracy=None,
        BA_CI=None, precision_CI=None, recall_CI=None, accuracy_CI=None,
        bootstrap_group_resamples_planned=5000, bootstrap_group_resamples_performed=0,
        CI_reason="No external evaluation performed; no predictions exist.")
        for method in method_names for weighting in ("group_weighted", "unweighted")]
    cfg["report_markdown"] = (
        "| F4 采集结果 | n | 组数 | 正类 | 负类 |\n"
        "|---|---:|---:|---:|---:|\n"
        f"| 已完成资格核查、未评测 | {metrics['accepted_n']} | {metrics['accepted_groups']} | {metrics['accepted_positive_n']} | {metrics['accepted_negative_n']} |\n"
        f"| 精确配比候选总记录（含排除与待定） | {metrics['candidate_materials']} | — | — | — |\n"
        f"| 精确配比仍待定 | {pending} | — | — | — |\n\n"
        "| 方法 | 组加权 BA / 精确率 / 召回率 / 准确率 | 不加权 BA / 精确率 / 召回率 / 准确率 | 评测 n / 组数 |\n"
        "|---|---|---|---|\n"
        + "".join(f"| {method} | 未运行 | 未运行 | 0 / 0 |\n" for method in method_names)
        + "\n结论：已核实官方知识截止日为 2026-04-30；合格首报窗口为 2026-05-30 至 2026-10-17。\n"
        "完成 arXiv 912 条书目覆盖及有限的原文、首报和具体母体核查，保留全部排除与待定证据。\n"
        "当前合格 5 条、3 个母体家族，全部 Tc≤10 K；未达至少 40 条及每类至少 20 条的门槛，依原规则停止。\n"
        "A2、A3、回忆测试与基线均未运行，无外部性能结论；两种权重指标、配对差值和置信区间均不可计算。\n"
        "有限检索不能证明不存在更多合格材料，也不能把待定、旧材料新表征或理论预测补作外部样本。")
    save(cfg)
    with checked_destination(OUT / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{finished} collection_complete_insufficient: qualified=5, families=3, positive=0, negative=5, exact-pending={pending}; no calls/fits; stop below40\n")
    print(json.dumps({"status": cfg["status"], "accepted_n": metrics["accepted_n"],
        "accepted_groups": metrics["accepted_groups"], "pending": pending,
        "elapsed_seconds": elapsed, "calls": 0, "fits": 0}, ensure_ascii=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("initialize", "refresh", "primary_reviews", "arxiv_inventory", "review_arxiv", "merge_followups", "verified_updates", "final_supplement", "freeze_insufficient"))
    args = parser.parse_args()
    if args.action == "initialize":
        initialize()
    elif args.action == "primary_reviews":
        apply_primary_reviews()
    elif args.action == "arxiv_inventory":
        inventory_arxiv()
    elif args.action == "review_arxiv":
        review_arxiv_candidates()
    elif args.action == "merge_followups":
        merge_followup_evidence()
    elif args.action == "verified_updates":
        final_verified_updates()
    elif args.action == "final_supplement":
        final_supplement()
    elif args.action == "freeze_insufficient":
        freeze_insufficient()
    else:
        cfg = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
        print(json.dumps(save(cfg), ensure_ascii=False))
