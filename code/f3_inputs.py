"""Frozen E2 scientific inputs extended to the original 3764 OOF materials.

No model calls, model fits, experiment writes, or old runner imports.  Scaling
and rule application reproduce the E2 data-only preparation.  The 300 existing
A2/A3 predictions are read for reuse; the remaining 3464 materials are new calls.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from common import BANKS, DATA, HGB, ROOT, load_data, np
from e1_rules import applicability_matrix, load_rules, select_rule_scores
from e2_packets import (FACTS, MAX_RULES, MIN_SUPPORT_GROUPS, NEIGHBOURS,
                        _counterexample_indices, _example, _load_materials,
                        compact_evidence, fixed_indices)
from e2_prompts import SCHEMA, make_prompt
from e2_transport import input_budget
from sklearn.preprocessing import StandardScaler


ARMS = ("A2", "A3")
FAMILIES = ("copper", "iron", "hydride", "A15", "other")
SIZE_BINS = ("1", "2-5", "6-20", ">20")
SOURCE_SCHEMA = ROOT / "e2_response.schema.json"
SOURCE_INSTRUCTIONS = ROOT / "e2_instructions.md"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_for(case, arm, history=None):
    """The literal E2 A2/A3 prompt_for behavior, without importing its runner."""
    if arm not in ARMS:
        raise ValueError(arm)
    history = [] if history is None else history
    evidence = (compact_evidence(case["rule_cards"], rule_summaries=case["rule_summaries"])
                if arm == "A2" else {})
    positions = {r["rule_id"]: j for j, r in enumerate(case["rule_summaries"])}
    short_history = []
    for response in history:
        if response.get("action") == "request":
            short_history.append(dict(action="request",
                request_initial_rule_indices=[positions[r] for r in response["request_rule_ids"]],
                request_counterexamples=response["request_counterexamples"], reason=response["reason"]))
        else:
            short_history.append(response)
    return make_prompt(arm, case["material"], case["rule_summaries"] if arm != "A3" else [],
        evidence, case["neighbours"] if arm == "A2" else [], short_history,
        round_number=len(history) + 1, force_decision=False)


def validate_output(response, case, arm):
    """E2's exact direct-arm contract and normalized case-result fields."""
    if arm not in ARMS:
        raise ValueError(arm)
    if not isinstance(response, dict) or set(response) != set(SCHEMA["required"]):
        raise ValueError("JSON object fields differ from frozen schema")
    if response["action"] not in ("request", "decide") or not isinstance(response["abstain"], bool):
        raise ValueError("Invalid action/abstain")
    if not isinstance(response["reason"], str) or len(response["reason"]) > 80:
        raise ValueError("Reason exceeds80 characters")
    for key in ("selected_rule_ids", "request_rule_ids"):
        value = response[key]
        if (not isinstance(value, list) or len(value) != len(set(value))
                or any(not isinstance(item, str) or item not in case["rule_cards"] for item in value)):
            raise ValueError("Unknown or duplicate rule ID")
    if not isinstance(response["request_counterexamples"], bool):
        raise ValueError("Invalid evidence flag")
    if response["action"] == "request":
        raise ValueError("Invalid evidence request")
    p = response["p"]
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("Missing/invalid forced probability")
    if response["request_rule_ids"] or response["request_counterexamples"]:
        raise ValueError("Decision also requested evidence")
    selected = response["selected_rule_ids"]
    if len(selected) > 3 or (arm == "A3" and selected):
        raise ValueError("Invalid selected rules for direct arm")
    return dict(id=case["id"], arm=arm, p=float(p), abstain=response["abstain"],
        selected_rules=selected, reported_p=response["p"], reason=response["reason"], status="ok")


def declared_sg223(material):
    """Read declared SG223; Pm-3n is its conventional Hermann--Mauguin name.

    This normalizes a declared source label, never infers geometry symmetry.
    A declared numeric SG takes priority; missing numeric SG may use Pm-3n.
    """
    declared = material["structure"]["space_group_declared_in_source_CIF"]
    if declared["number"] is not None:
        return int(declared["number"]) == 223
    symbol = str(declared.get("symbol") or "").replace(" ", "").replace("_", "").replace("−", "-").lower()
    return symbol == "pm-3n"


def strict_a3b(material):
    positive = [float(row["cell_occupancy"]) for row in material["composition"]
                if float(row["cell_occupancy"]) > 0]
    return len(positive) == 2 and math.isclose(max(positive) / min(positive), 3., rel_tol=1e-8, abs_tol=1e-8)


def family_of(material):
    elements = {row["element"] for row in material["composition"] if row["atomic_fraction"] > 0}
    hydrogen = next((float(row["atomic_fraction"]) for row in material["composition"] if row["element"] == "H"), 0.)
    if {"Cu", "O"} <= elements:
        return "copper"
    if "Fe" in elements and elements & {"As", "P", "Se", "Te"}:
        return "iron"
    if hydrogen >= .5:
        return "hydride"
    if declared_sg223(material) and strict_a3b(material):
        return "A15"
    return "other"


def group_size_bin(size):
    return "1" if size == 1 else "2-5" if size <= 5 else "6-20" if size <= 20 else ">20"


def _full_packets(data, log):
    """E2 build_packets algorithm with only its fixed300 restriction removed."""
    indices = np.arange(len(data["id"]), dtype=int)
    log("F3 reading frozen common-facts and declared source-CIF summaries for3764 targets")
    materials, names, units = _load_materials(data, indices)
    log("F3 full3764 material summaries loaded")
    by_source = {int(v): i for i, v in enumerate(data["source_index"])}
    cases = {str(data["id"][i]): {"i": int(i), "id": str(data["id"][i]),
        "fold": int(data["fold"][i]), "group": str(data["group"][i]), "material": materials[int(i)]}
        for i in indices}
    folds = {}
    for outer in (0, 1, 2):
        train = np.flatnonzero(data["fold"] != outer)
        held = np.flatnonzero(data["fold"] == outer)
        if set(map(str, data["group"][train])) & set(map(str, data["group"][held])):
            raise ValueError("Target association group appears in TRAIN")
        path = BANKS / f"fold{outer}" / "ROOT_REVIEWABLE_BANK.json"
        rules, bank_meta = load_rules(path)
        if set(bank_meta["train_indices"]) != set(map(int, data["source_index"][train])):
            raise ValueError("Frozen bank TRAIN differs from original fold TRAIN")
        raw_bank = json.loads(path.read_text(encoding="utf-8"))["finalized_bank"]
        envelopes = {item["card"]["rule_id"]: item for item in raw_bank["rule_envelopes"]}
        matrix = applicability_matrix(data["X127"][held], rules)
        rinfo, _, _ = select_rule_scores(matrix, rules, mode="R-info")
        ranking = sorted((j for j, rule in enumerate(rules) if rule.support_groups >= MIN_SUPPORT_GROUPS),
                         key=lambda j: (-abs(rules[j].score - .5), rules[j].rule_id))
        scaler = StandardScaler().fit(data["X127"][train])
        scaled_train = scaler.transform(data["X127"][train])
        scaled_held = scaler.transform(data["X127"][held])
        card_cache = {}
        for position, i in enumerate(held):
            chosen = [rules[j] for j in ranking if matrix[position, j]][:MAX_RULES]
            briefs, cards = [], {}
            for rule in chosen:
                conditions = [{"feature_index": j, "feature": names[j], "unit": units[j],
                    "operator": operator, "threshold": threshold, "target_value": float(data["X127"][i, j])}
                    for j, operator, threshold in rule.predicates]
                brief = {"rule_id": rule.rule_id, "conditions": conditions,
                    "class_balanced_score": rule.score, "TRAIN_support_groups": rule.support_groups,
                    "TRAIN_coverage_rows": rule.coverage_rows}
                briefs.append(brief)
                if rule.rule_id not in card_cache:
                    envelope = envelopes[rule.rule_id]
                    card = envelope["card"]
                    examples = _counterexample_indices(data, rule, by_source)
                    if any(int(data["fold"][j]) == outer for j in examples):
                        raise ValueError("Counterexample is outside target TRAIN")
                    card_cache[rule.rule_id] = {
                        "score_semantics": card["prediction_effect"]["score_semantics"],
                        "score_formula": card["theoretical_basis"]["numeric_formula"],
                        "theoretical_basis": card["theoretical_basis"]["statement"],
                        "claim_scope": card["theoretical_basis"]["claim_scope"],
                        "TRAIN_class_row_counts": envelope["statistics"]["matched_class_row_counts"],
                        "TRAIN_class_group_weight_mass": envelope["statistics"]["matched_class_ws_mass"],
                        "known_failures": card["known_failures"],
                        "counterexamples": [_example(data, materials, j) for j in examples],
                        "counterexample_count_available_in_TRAIN": sum(
                            int(data["y"][by_source[int(s)]]) != int(rule.score >= .5)
                            for s in rule.support_indices)}
                cards[rule.rule_id] = {**brief, **card_cache[rule.rule_id]}
            distances = np.sqrt(np.sum((scaled_train - scaled_held[position]) ** 2, axis=1))
            nearest_positions = np.lexsort((data["source_index"][train], distances))[:NEIGHBOURS]
            nearest = train[nearest_positions]
            if any(str(data["group"][j]) == str(data["group"][i]) for j in nearest):
                raise ValueError("Target group leaked through neighbour")
            cases[str(data["id"][i])].update(rule_summaries=briefs, rule_cards=cards,
                neighbours=[_example(data, materials, int(j), distance=distances[k]) for j, k in zip(nearest, nearest_positions)],
                knn_score=float(np.mean(data["y"][nearest])), rinfo_score=float(rinfo[position]))
        folds[str(outer)] = {"n_TRAIN": len(train), "n_targets": len(held), "n_rules": len(rules),
                             "bank": str(path), "target_TRAIN_group_overlap": 0}
        log(f"F3 packets fold{outer}: {len(held)} targets; {len(card_cache)} selected rule IDs")
    with (ROOT / "E1/predictions.csv").open(newline="", encoding="utf-8") as handle:
        reference = {row["id"]: float(row["score"]) for row in csv.DictReader(handle) if row["arm"] == "R-info"}
    assert len(reference) == len(cases)
    assert all(case["rinfo_score"] == reference[case["id"]] for case in cases.values())
    return cases, folds


def _read_predictions(path, arms, data, expected_n):
    by_id = {str(value): i for i, value in enumerate(data["id"])}
    result = {arm: {} for arm in arms}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["arm"] not in result:
                continue
            i = by_id[row["id"]]
            if row["id"] in result[row["arm"]]:
                raise ValueError("Duplicate frozen prediction")
            assert row["group"] == str(data["group"][i]) and int(row["fold"]) == int(data["fold"][i])
            score = float(row["score"])
            assert math.isfinite(score) and 0 <= score <= 1 and int(row["y_pred"]) == int(score >= .5)
            result[row["arm"]][row["id"]] = row
    assert all(len(value) == expected_n for value in result.values())
    return result


def build_inputs(data=None, log=None, preflight=True, allow_budget_report=False):
    """Return JSON-safe frozen full-cohort inputs, baselines and reuse manifest."""
    data = load_data() if data is None else data
    log = log or (lambda message: None)
    old = json.loads((ROOT / "E2/config.json").read_text(encoding="utf-8"))
    fixed = fixed_indices(data).tolist()
    assert fixed == old["indices"] and len(fixed) == 300
    cases, folds = _full_packets(data, log)
    # Equality is checked before adding any evaluation-only metadata.
    assert all(cases[key] == value for key, value in old["cases"].items()), "E2 fixed300 packet changed"
    old_call_prompts = {}
    with (ROOT / "E2/llm_calls.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["arm"] in ARMS and row.get("round") == 1:
                key = (row["arm"], row["id"])
                assert key not in old_call_prompts
                old_call_prompts[key] = row["prompt"]
    assert len(old_call_prompts) == 600
    assert all(prompt_for(cases[key], arm) == prompt for (arm, key), prompt in old_call_prompts.items()), "E2 prompt changed"
    reused = _read_predictions(ROOT / "E2/predictions.csv", ("A2_forced", "A3_forced"), data, 300)
    strong = _read_predictions(ROOT / "E0/predictions.csv", ("StrongML",), data, 3764)["StrongML"]
    baseline = {"HGB127": data["hgb"].tolist(),
                "StrongML": [float(strong[str(key)]["score"]) for key in data["id"]],
                "5NN": [cases[str(key)]["knn_score"] for key in data["id"]]}
    fixed_set = set(fixed)
    family_masks = {name: [] for name in FAMILIES}
    sg223, a15, group93 = [], [], []
    bounds = {arm: {"initial": [], "max_repair": []} for arm in ARMS}
    budget_failures = []
    # Only fixed instructions/schema/global-guidance bytes are cached. Runtime
    # call transport still checks every actual complete input independently.
    base_budget = input_budget("", schema_path=SOURCE_SCHEMA, instructions_path=SOURCE_INSTRUCTIONS) if preflight else None
    worst_history = [{"protocol_error": "x" * 240,
                      "instruction": "Repair format/decision using exactly the same evidence."}]
    for i, key in enumerate(data["id"]):
        case = cases[str(key)]
        fam = family_of(case["material"])
        for name in FAMILIES:
            family_masks[name].append(name == fam)
        sg223.append(declared_sg223(case["material"]))
        a15.append(sg223[-1] and strict_a3b(case["material"]))
        group93.append(case["group"] == "93")
        case.update(evaluation={"Tc_K": float(data["tc"][i]), "y": int(data["y"][i]),
                    "original_group_size": int(data["group_size"][i])},
                    family=fam, group_size_bin=group_size_bin(int(data["group_size"][i])), reuse_E2=i in fixed_set)
        case["prompt_hashes"] = {}
        for arm in ARMS:
            p = prompt_for(case, arm)
            case["prompt_hashes"][arm] = hashlib.sha256(p.encode("utf-8")).hexdigest()
            if preflight:
                for name, hist in (("initial", []), ("max_repair", worst_history)):
                    prompt_bytes = len(prompt_for(case, arm, hist).encode("utf-8"))
                    upper_bound = base_budget["input_token_upper_bound"] + prompt_bytes
                    budget = {**base_budget, "prompt_bytes": prompt_bytes,
                        "input_token_upper_bound": upper_bound,
                        "within_budget": upper_bound <= base_budget["input_limit"]}
                    if not budget["within_budget"]:
                        budget_failures.append({"arm": arm, "id": case["id"], "round_kind": name, **budget})
                    bounds[arm][name].append(budget["input_token_upper_bound"])
    def profile(mask):
        return {"n": sum(mask), "n_groups": len({case["group"] for case, take in zip(cases.values(), mask) if take})}
    both = [x and y for x, y in zip(a15, group93)]
    sources = [DATA, HGB, FACTS, ROOT / "E0/predictions.csv", ROOT / "E1/predictions.csv",
               ROOT / "E2/config.json", ROOT / "E2/predictions.csv", ROOT / "E2/llm_calls.jsonl",
               *[BANKS / f"fold{k}" / "ROOT_REVIEWABLE_BANK.json" for k in (0, 1, 2)]]
    metadata = {"n": 3764, "n_groups": len(set(map(str, data["group"]))), "fixed_reused": 300,
        "new_materials": 3464, "new_initial_calls": 6928, "max_calls": 10000,
        "folds": folds, "fixed300_packets_equal_E2": True, "fixed600_initial_prompts_equal_E2": True,
        "R_info_matches_full_E1_exactly": True,
        "prompt_budget_upper_bounds": {arm: {kind: {"min": min(values), "max": max(values)}
            for kind, values in by_kind.items() if values} for arm, by_kind in bounds.items()},
        "prompt_budget_failures": budget_failures,
        "prompt_budget_fixed_overhead_snapshot": base_budget,
        "repair_metadata_policy": "protocol_error is ASCII-escaped and its JSON-encoded content capped at240 UTF8 bytes (excluding enclosing quotes); provider failures contribute bounded status, never free-text provider messages; scientific payload and E2 prompts unchanged",
        "families": {name: profile(mask) for name, mask in family_masks.items()},
        "family_definitions": {"priority": list(FAMILIES), "copper": "Cu and O present",
            "iron": "Fe and at least one As/P/Se/Te present", "hydride": "H atomic fraction>=0.5",
            "A15": "declared SG223 and exactly two positive-occupancy elements in3:1 ratio (rtol=atol1e-8)",
            "other": "remaining materials", "SG223": "declared numeric223, or declared Pm-3n if number missing; no symmetry inference",
            "boundary": "strict binaryA3B excludes substituted multielement A15-family members; proxy structure grouping is not measured-phase validation"},
        "A15_component93_overlap": {"A15": profile(a15), "component93": profile(group93),
            "both": profile(both), "A15_not93": profile([a and not g for a, g in zip(a15, group93)]),
            "component93_notA15": profile([g and not a for a, g in zip(a15, group93)]), "declared_SG223": profile(sg223)},
        "baseline_sources": {"HGB127": str(HGB), "StrongML": str(ROOT / "E0/predictions.csv"),
            "5NN": "unweighted mean of exact five E2-protocol TRAIN labels; original TRAIN-only StandardScaler and source-index ties"},
        "scientific_source_hashes": {str(path): sha256_file(path) for path in sources},
        "scientific_prompt_accessor": "prompt_for(case,arm,history); no evaluation/identity/family/bin/reuse metadata enters prompt"}
    if budget_failures and not allow_budget_report:
        raise ValueError("F3 frozen prompt budget preflight failed: " + json.dumps({
            "n_failures": len(budget_failures), "by_arm_round": dict(Counter(
                item["arm"] + "/" + item["round_kind"] for item in budget_failures)),
            "maximum_upper_bounds": metadata["prompt_budget_upper_bounds"], "first_failures": budget_failures[:5]}))
    log(f"F3 input parity and full3764 baselines verified; budget failures={len(budget_failures)}")
    return {"cases": cases, "indices": list(range(3764)), "fixed_indices": fixed,
        "pending_indices": [i for i in range(3764) if i not in fixed_set],
        "baselines": baseline, "reused_predictions": reused, "family_masks": family_masks,
        "raw_a15_mask": a15, "declared_sg223_mask": sg223,
        "group_size_values": data["group_size"].astype(int).tolist(), "metadata": metadata}
