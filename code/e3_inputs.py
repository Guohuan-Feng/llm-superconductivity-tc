"""Read-only Stage 3 input reconstruction and predeclared error sampling.

No scientific model is called here.  All classification evidence is reconstructed
from the literal E2 prompts; in-memory cards with ten counterexamples must never
be confused with the maximum five examples actually delivered in Stage 2.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from collections import Counter, defaultdict

from common import ROOT, load_data


FAMILIES = ("copper", "iron", "hydride", "other")
ARMS = ("A1", "A2")
SAMPLE_SALT = "E3-stage3-20261016-v1"
SAMPLE_QUOTAS = {
    "A2_error": {"copper": 1, "other": 14},
    "A1_only": {"copper": 5, "iron": 3, "other": 7},
}


def family(material):
    """Exclusive user predicates, in their written priority order."""
    composition = material["composition"]
    elements = {row["element"] for row in composition if row["atomic_fraction"] > 0}
    hydrogen = next((float(row["atomic_fraction"]) for row in composition
                     if row["element"] == "H"), 0.0)
    if {"Cu", "O"} <= elements:
        return "copper"
    if "Fe" in elements and elements & {"As", "P", "Se", "Te"}:
        return "iron"
    if hydrogen >= 0.5:
        return "hydride"
    return "other"


def _profile(ids, cases, families):
    ids = list(ids)
    return {
        "n": len(ids),
        "n_groups": len({cases[identity]["group"] for identity in ids}),
        "families": {
            name: {
                "n": sum(families[identity] == name for identity in ids),
                "n_groups": len({cases[identity]["group"] for identity in ids
                                 if families[identity] == name}),
            } for name in FAMILIES
        },
    }


def load_inputs(d=None, config=None, call_records=None):
    """Validate E2 identity/result alignment and return only in-memory inputs."""
    data = load_data() if d is None else d
    config = (json.loads((ROOT / "E2/config.json").read_text(encoding="utf-8"))
              if config is None else config)
    cases = config["cases"]
    by_id = {str(identity): i for i, identity in enumerate(data["id"])}
    families = {identity: family(case["material"]) for identity, case in cases.items()}
    calls = {arm: defaultdict(list) for arm in ARMS}
    final_calls = {arm: {} for arm in ARMS}
    results = {arm: {} for arm in ARMS}
    if call_records is None:
        with (ROOT / "E2/llm_calls.jsonl").open(encoding="utf-8") as handle:
            call_records = [json.loads(line) for line in handle]
    if isinstance(call_records, dict):
        call_records = [row for value in call_records.values()
                        for row in (value if isinstance(value, list) else [value])]
    for row in call_records:
        arm, identity = row["arm"], row["id"]
        if arm not in ARMS:
            continue
        if identity not in cases:
            raise ValueError("E2 call identity is not in the frozen cohort")
        calls[arm][identity].append(row)
        if "case_result" in row:
            if identity in results[arm]:
                raise ValueError("Duplicate E2 final case result")
            result = row["case_result"]
            if result["id"] != identity or result["arm"] != arm:
                raise ValueError("E2 final-result identity differs from its call")
            if result["status"] != "ok" or result["p"] is None:
                raise ValueError("E3 requires an effective E2 forced decision")
            results[arm][identity] = result
            final_calls[arm][identity] = row
    with (ROOT / "E2/predictions.csv").open(newline="", encoding="utf-8") as handle:
        predictions = defaultdict(dict)
        for row in csv.DictReader(handle):
            if row["id"] in predictions[row["arm"]]:
                raise ValueError("Duplicate E2 prediction row")
            predictions[row["arm"]][row["id"]] = row
    errors = {arm: set() for arm in ARMS}
    for identity, case in cases.items():
        i = by_id[identity]
        if int(case["i"]) != i or int(case["fold"]) != int(data["fold"][i]):
            raise ValueError("E2 case identity/outer fold changed")
        if str(case["group"]) != str(data["group"][i]):
            raise ValueError("E2 association group changed")
        for arm in ARMS:
            result = results[arm][identity]
            prediction = predictions[arm + "_forced"][identity]
            p = float(result["p"])
            if p != float(prediction["score"]) or int(p >= 0.5) != int(prediction["y_pred"]):
                raise ValueError("E2 CSV differs from its effective case result")
            if (p >= 0.5) != bool(data["y"][i]):
                errors[arm].add(identity)
    for arm in ARMS:
        if set(results[arm]) != set(cases):
            raise ValueError("E2 final decisions are incomplete")
    overlap = errors["A1"] & errors["A2"]
    union = errors["A1"] | errors["A2"]
    summary = {
        "cohort": _profile(cases, cases, families),
        "forced_errors": {arm: _profile(errors[arm], cases, families) for arm in ARMS},
        "union": _profile(union, cases, families),
        "overlap": _profile(overlap, cases, families),
        "error_unit": "forced decisions at p>=0.5, including abstention backup decisions",
    }
    return dict(data=data, config=config, cases=cases, by_id=by_id,
                families=families, calls=calls, final_calls=final_calls,
                results=results, predictions=dict(predictions), errors=errors,
                summary=summary)


def _hash(text):
    return hashlib.sha256((SAMPLE_SALT + "|" + str(text)).encode("utf-8")).hexdigest()


def _take_groups_first(pool, n, cases, *, already=()):
    """Hash-ordered group round robin, avoiding within-group concentration."""
    groups = defaultdict(list)
    for identity in pool:
        groups[str(cases[identity]["group"])].append(identity)
    for identities in groups.values():
        identities.sort(key=_hash)
    picked = []
    prior = Counter(str(cases[identity]["group"]) for identity in already)
    while len(picked) < n:
        available = [group for group, identities in groups.items() if identities]
        if not available:
            raise ValueError("Predeclared error stratum is too small")
        group = min(available, key=lambda value: (prior[value], _hash(value)))
        picked.append(groups[group].pop(0))
        prior[group] += 1
    return picked


def select_error_sample(inputs, *, quotas):
    """Apply frozen disjoint A2-error/A1-only quotas, never inspect reasons.

    Example quotas is {"A2_error":{"copper":1,"other":14},
    "A1_only":{"copper":5,"iron":3,"other":7}}. Quotas are a caller-frozen
    design choice and must be stored in E3/config.json before annotation.
    """
    cases, families, errors = inputs["cases"], inputs["families"], inputs["errors"]
    pools = {"A2_error": errors["A2"], "A1_only": errors["A1"] - errors["A2"]}
    selected = []
    for stratum in ("A2_error", "A1_only"):
        unknown = set(quotas.get(stratum, {})) - set(FAMILIES)
        if unknown:
            raise ValueError("Unknown family quota")
        for name in FAMILIES:
            n = int(quotas.get(stratum, {}).get(name, 0))
            pool = {identity for identity in pools[stratum] if families[identity] == name}
            selected.extend(_take_groups_first(pool, n, cases, already=selected))
    if len(selected) != 30 or len(set(selected)) != 30:
        raise ValueError("E3 sample must contain exactly thirty unique erroneous materials")
    return selected


def sample_summary(inputs, selected):
    cases, families, errors = inputs["cases"], inputs["families"], inputs["errors"]
    return {
        **_profile(selected, cases, families),
        "errored_observations": {
            arm: _profile(set(selected) & errors[arm], cases, families) for arm in ARMS
        },
        "selection_salt": SAMPLE_SALT,
        "selection_order": "A2 errors then disjoint A1-only errors; written family priority; fewest sampled rows per group then SHA256 group/id",
        "population_inference": "stratified diagnostic sample; category fractions do not estimate all-error prevalence",
    }


def scientific_payload(call):
    marker = "Scientific inputs:\n"
    prompt = call["prompt"]
    if marker not in prompt:
        raise ValueError("Final E2 material call has no scientific payload")
    return json.loads(prompt.split(marker, 1)[1])


def decode_delivered_cards(payload, arm):
    """Reverse compact serialization, including the literal first-five cap."""
    summaries = payload.get("applicable_rule_summaries", [])
    field = "returned_rule_evidence" if arm == "A1" else "full_rule_evidence"
    compact = payload.get(field, {})
    if not compact:
        return []
    table = compact["counterexample_table"]
    examples = [dict(zip(table["columns"], row)) for row in table["rows"]]
    encoding = table.get("id_encoding")
    for example in examples:
        if encoding:
            example["id"] = encoding["prefix"] + str(example.pop("id_number")).zfill(encoding["decimal_width"])
    cards = []
    card_table = compact["rule_cards"]
    for row in card_table["rows"]:
        variable = dict(zip(card_table["columns"], row))
        if "initial_rule_index" in variable:
            brief = summaries[int(variable.pop("initial_rule_index"))]
        else:
            brief = next(value for value in summaries if value["rule_id"] == variable["rule_id"])
        card = {**copy.deepcopy(brief), **copy.deepcopy(compact["rule_card_defaults"]), **variable}
        references = card.pop("counterexample_rows", None)
        if references is not None:
            card["counterexamples"] = [copy.deepcopy(examples[int(j)]) for j in references]
        card["delivered_counterexample_cap"] = compact["delivered_counterexamples_per_rule_cap"]
        cards.append(card)
    return cards


def annotation_packet(inputs, identity):
    """One material, both observed arms, and exact actually delivered evidence."""
    case, data = inputs["cases"][identity], inputs["data"]
    i = inputs["by_id"][identity]
    observations = []
    for arm in ARMS:
        result = inputs["results"][arm][identity]
        final_call = inputs["final_calls"][arm][identity]
        payload = scientific_payload(final_call)
        summaries = payload.get("applicable_rule_summaries", [])
        selected = set(result["selected_rules"])
        observations.append({
            "arm": arm,
            "effective_forced_p": result["p"],
            "reported_p": result.get("reported_p"),
            "y_pred": int(float(result["p"]) >= 0.5),
            "forced_error": identity in inputs["errors"][arm],
            "abstain": result["abstain"],
            "reason": result["reason"],
            "selected_rule_ids": result["selected_rules"],
            "selected_rule_summaries": [rule for rule in summaries if rule["rule_id"] in selected],
            "all_six_applicable_rule_summaries": summaries,
            "delivered_evidence": payload.get("returned_rule_evidence" if arm == "A1" else "full_rule_evidence", {}),
            "five_nearest_TRAIN_materials": payload.get("nearest_TRAIN_materials", []) if arm == "A2" else [],
            "round_trace": [{
                "call_id": call["call_id"], "round": call["round"],
                "status": call["status"], "request_hash": call["request_hash"],
                "parsed_response": call.get("parsed_response"),
            } for call in inputs["calls"][arm][identity]],
            "selected_rules_semantics": (
                "A1 forced score is exact equal-weight selected-rule mean, independently of abstention."
                if arm == "A1" else
                "A2 directly chooses p; zero declared selected IDs means no rule formally attributed, not proof of no implicit influence."
            ),
        })
    return {
        "id": identity, "fold": int(case["fold"]), "group": str(case["group"]),
        "family": inputs["families"][identity],
        "material": copy.deepcopy(case["material"]),
        "observed_Tc_K": float(data["tc"][i]),
        "observed_Tc_gt_10K": bool(data["y"][i]),
        "observations": observations,
        "annotation_unit": "Assign a primary error category only to observations with forced_error=true; other arm is contextual evidence.",
        "causal_limitations": [
            "Reasons are short self-reports, not full internal reasoning traces.",
            "Proxy flag alone does not establish label noise or a causal explanation of error.",
            "Curated Tc=0 alone does not prove a measured nonsuperconducting specimen or label corruption.",
            "Category 3 requires explicit evidence of a valid chemical premise being overridden by rules; coexistence is insufficient.",
            "No new web lookups or external target Tc may be introduced by the annotator.",
        ],
    }


def compact_annotation_packet(inputs, identity, primary_arm):
    """Single-primary-arm packet; all effective evidence, no duplicated trace.

    The final prompt already carries cumulative requested evidence. Keeping its
    reversible compact table gives the annotator every actually delivered card
    and example without duplicating six summaries or two full arm transcripts.
    """
    case, data = inputs["cases"][identity], inputs["data"]
    i = inputs["by_id"][identity]
    final_call = inputs["final_calls"][primary_arm][identity]
    payload = scientific_payload(final_call)
    primary = inputs["results"][primary_arm][identity]
    secondary_arm = "A1" if primary_arm == "A2" else "A2"
    secondary = inputs["results"][secondary_arm][identity]
    material = copy.deepcopy(payload["material"])
    # geometry entries already present as named rule target values are retained
    # there. Remaining geometry is still available in the frozen E2 input map;
    # keep global structure context through declared symmetry, lattice, volume,
    # site count and proxy-match fields without 22 redundant scalar summaries.
    material["structure"].pop("existing_global_structure_features", None)
    evidence_key = "returned_rule_evidence" if primary_arm == "A1" else "full_rule_evidence"
    requests = []
    for row in inputs["calls"][primary_arm][identity]:
        response = row.get("parsed_response", {})
        if response.get("action") == "request":
            requests.append({key: response[key] for key in
                             ("request_rule_ids", "request_counterexamples", "reason")})
    return {
        "id": identity, "primary_arm": primary_arm,
        "fold": int(case["fold"]), "group": str(case["group"]),
        "family": inputs["families"][identity],
        "material": material,
        "observed_Tc_K": float(data["tc"][i]),
        "observed_Tc_gt_10K": bool(data["y"][i]),
        "primary_error": {
            "p": primary["p"], "reported_p": primary.get("reported_p"),
            "y_pred": int(float(primary["p"]) >= 0.5),
            "abstain": primary["abstain"], "reason": primary["reason"],
            "selected_rule_ids": primary["selected_rules"],
        },
        "secondary_arm_context": {
            "arm": secondary_arm, "forced_error": identity in inputs["errors"][secondary_arm],
            "p": secondary["p"], "abstain": secondary["abstain"],
            "reason": secondary["reason"], "selected_rule_ids": secondary["selected_rules"],
            "use": "Context only; do not assign a second primary category or double-count this material.",
        },
        "applicable_rule_summaries": payload.get("applicable_rule_summaries", []),
        "delivered_evidence": payload.get(evidence_key, {}),
        "evidence_requests": requests,
        "five_nearest_TRAIN_materials": payload.get("nearest_TRAIN_materials", []) if primary_arm == "A2" else [],
        "score_mechanism": (
            "A1 exact mean of 1-3 selected rule scores, >=0.5 positive; p may differ slightly from reported rounded p."
            if primary_arm == "A1" else
            "A2 direct p independent of rule mean; 0-3 declared selected rules. Zero selected IDs does not prove zero implicit influence."
        ),
        "source_final_call_id": final_call["call_id"],
        "source_request_hash": final_call["request_hash"],
    }


def build_e3_inputs(d=None, config=None, calls=None):
    """Root-runner interface. Pure reads; caller persists the frozen design."""
    inputs = load_inputs(d=d, config=config, call_records=calls)
    identities = select_error_sample(inputs, quotas=SAMPLE_QUOTAS)
    selected = []
    for identity in identities:
        primary_arm = "A2" if identity in inputs["errors"]["A2"] else "A1"
        case = inputs["cases"][identity]
        selected.append({
            "id": identity, "primary_arm": primary_arm,
            "family": inputs["families"][identity], "group": str(case["group"]),
            "fold": int(case["fold"]), "i": int(case["i"]),
            "secondary_arm_also_wrong": identity in inputs["errors"]["A1"] and identity in inputs["errors"]["A2"],
            "annotation_packet": compact_annotation_packet(inputs, identity, primary_arm),
        })
    design = {
        "n_unique_materials": 30,
        "primary_arm_quota": {"A1": 15, "A2": 15},
        "disjoint_quota": SAMPLE_QUOTAS,
        "predeclared_before_reason_annotation": True,
        "selected_profile": sample_summary(inputs, identities),
        "primary_profiles": {
            arm: _profile([row["id"] for row in selected if row["primary_arm"] == arm],
                          inputs["cases"], inputs["families"]) for arm in ARMS
        },
        "secondary_overlap_n": sum(row["secondary_arm_also_wrong"] for row in selected),
        "category_count_unit": "One primary category per selected material; secondary errors only contextual.",
        "selection_salt": SAMPLE_SALT,
    }
    return {**inputs, "selected": selected, "counts": inputs["summary"],
            "sampling_design": design}


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(load_inputs()["summary"], ensure_ascii=False, indent=2))
