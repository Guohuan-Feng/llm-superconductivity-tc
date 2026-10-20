"""Frozen scientific inputs for finite E2; never invokes a model or writes data.

Only the current target's formula, occupancy composition and existing structural
summaries enter its material packet. Rule statistics, counterexamples and nearest
neighbours are restricted to that target's original outer TRAIN partition.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shlex
from collections import defaultdict
from pathlib import Path

from common import BANKS, ROOT, load_data, np
from e1_rules import applicability_matrix, load_rules, select_rule_scores
from sklearn.preprocessing import StandardScaler


FACTS = ROOT.parent / "workflow-review-2026-10-10/common-facts-cohort-actual-001/common_facts_fullTRAIN.jsonl"
MIN_SUPPORT_GROUPS = 20
MAX_RULES = 6
MAX_COUNTEREXAMPLES = 10
NEIGHBOURS = 5


def fixed_indices(data):
    """Recover the exact E0 fixed sample from its completed prediction table."""
    by_id = {str(v): i for i, v in enumerate(data["id"])}
    with (ROOT / "E0/predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["arm"] == "Agent"]
    indices = [by_id[r["id"]] for r in rows]
    if len(indices) != 300 or len(set(indices)) != 300:
        raise ValueError("E0 fixed300 is absent or incomplete")
    for row, i in zip(rows, indices):
        if int(row["fold"]) != int(data["fold"][i]) or row["group"] != str(data["group"][i]):
            raise ValueError("E0 sample identity mismatch")
    return np.asarray(indices, dtype=int)


def _cif_space_group(path):
    """Read declared CIF space group only; never infer symmetry or read Tc."""
    keys = {
        "_symmetry_space_group_name_h-m": "symbol",
        "_space_group_name_h-m_alt": "symbol",
        "_symmetry_int_tables_number": "number",
        "_space_group_it_number": "number",
    }
    result = {"symbol": None, "number": None}
    if path and Path(path).is_file():
        with Path(path).open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                key = line.strip().split(maxsplit=1)[0].lower() if line.strip() else ""
                if key in keys:
                    parts = shlex.split(line, comments=True)
                    value = " ".join(parts[1:])
                    if value not in ("", ".", "?"):
                        result[keys[key]] = int(value) if keys[key] == "number" else value
    return result


def _material(facts, packet):
    raw = facts["raw_graph"]["raw_structure"]
    absolute = defaultdict(list)
    for symbol, amount in zip(raw["species_symbols"], raw["species_occupancy"]):
        absolute[str(symbol)].append(float(amount))
    amounts = {s: math.fsum(a) for s, a in absolute.items() if math.fsum(a) > 0}
    total = math.fsum(amounts.values())
    if total <= 0:
        raise ValueError("Target has no occupied composition")
    composition = [{"element": s, "cell_occupancy": float(f"{amounts[s]:.10g}"),
                    "atomic_fraction": float(f"{amounts[s] / total:.10g}")}
                   for s in sorted(amounts)]
    formula = "".join(s + format(amounts[s], ".10g") for s in sorted(amounts))
    lattice = np.asarray(raw["lattice"], dtype=float)
    lengths = np.linalg.norm(lattice, axis=1)
    angles = []
    for a, b in ((1, 2), (0, 2), (0, 1)):
        cosine = np.dot(lattice[a], lattice[b]) / (lengths[a] * lengths[b])
        angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))
    source_cif = packet["source_lineage"]["source_identity"].get("cif")
    # The geometry summaries are the existing global X127 geometry columns and
    # thirteen existing common-facts scalars; no new predictive feature is fitted.
    existing = {
        facts["feature_names"][j]: None if facts["missing_mask"][j] else float(facts["values"][j])
        for j in range(118, 140)
    }
    return {
        "formula": formula,
        "formula_semantics": "occupancy formula of the supplied proxy structure cell; not reduced",
        "composition": composition,
        "structure": {
            "coordinate_source_proxy": bool(facts["proxy_flags"]["coordinate_source_proxy"]),
            "measured_specimen_match": str(facts["proxy_flags"]["measured_specimen_match"]),
            "space_group_declared_in_source_CIF": _cif_space_group(source_cif),
            "lattice_lengths_A": [float(v) for v in lengths],
            "lattice_angles_alpha_beta_gamma_deg": angles,
            "cell_volume_A3": float(raw["volume"]),
            "n_sites": int(raw["n_sites"]),
            "existing_global_structure_features": existing,
        },
    }


def _load_materials(data, target_indices):
    """Stream frozen facts once, retaining compact material summaries only."""
    by_source = {int(v): i for i, v in enumerate(data["source_index"])}
    target_set = set(map(int, target_indices))
    materials, feature_names, feature_units = {}, None, None
    # Formulas for all TRAIN candidates are needed for labelled examples and
    # neighbours, but only target rows need CIF symmetry and geometry summaries.
    with FACTS.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            coordinator = record["coordinator_record"]
            facts = coordinator["common_facts"]
            identity = facts["identity"]
            i = by_source[int(identity["source_index"])]
            if i in materials:
                raise ValueError("Duplicate common-facts row")
            if str(identity["material_id"]) != str(data["id"][i]):
                raise ValueError("Common-facts material identity differs")
            if str(identity["component_id"]) != str(data["group"][i]):
                raise ValueError("Common-facts association group differs")
            if int(identity["outer_fold"]) != int(data["fold"][i]):
                raise ValueError("Common-facts fold differs")
            if not np.array_equal(np.asarray(facts["values"][:127]), data["X127"][i]):
                raise ValueError("Common-facts X127 differs from accepted data")
            if feature_names is None:
                feature_names = list(facts["feature_names"][:127])
                feature_units = list(facts["feature_units"][:127])
            elif facts["feature_names"][:127] != feature_names:
                raise ValueError("Feature order changed within common-facts file")
            if i in target_set:
                materials[i] = _material(facts, coordinator["shared_structure_packet"])
            else:
                raw = facts["raw_graph"]["raw_structure"]
                amounts = defaultdict(list)
                for symbol, amount in zip(raw["species_symbols"], raw["species_occupancy"]):
                    amounts[str(symbol)].append(float(amount))
                materials[i] = {"formula": "".join(
                    s + format(math.fsum(amounts[s]), ".10g") for s in sorted(amounts)
                    if math.fsum(amounts[s]) > 0)}
    if len(materials) != len(data["id"]):
        raise ValueError("Common-facts cohort is incomplete")
    return materials, feature_names, feature_units


def _example(data, materials, i, *, distance=None):
    out = {"id": str(data["id"][i]), "formula": materials[i]["formula"],
           "Tc_K": float(data["tc"][i]), "Tc_gt_10K": bool(data["y"][i])}
    if distance is not None:
        out["standardized_X127_Euclidean_distance"] = float(distance)
    return out


def _counterexample_indices(data, rule, by_source):
    """At most ten opposing-label TRAIN examples, with distinct groups first."""
    predicted = int(rule.score >= .5)
    candidates = [by_source[int(s)] for s in rule.support_indices
                  if int(data["y"][by_source[int(s)]]) != predicted]
    def order(i):
        group = str(data["group"][i])
        return (hashlib.sha256((rule.rule_id + "|" + group).encode()).hexdigest(),
                int(data["source_index"][i]))
    candidates.sort(key=order)
    chosen, seen_groups = [], set()
    for i in candidates:
        group = str(data["group"][i])
        if group not in seen_groups:
            chosen.append(i)
            seen_groups.add(group)
        if len(chosen) == MAX_COUNTEREXAMPLES:
            return chosen
    for i in candidates:
        if i not in chosen:
            chosen.append(i)
        if len(chosen) == MAX_COUNTEREXAMPLES:
            break
    return chosen


def build_packets(data=None, indices=None, log=None):
    """Return JSON-safe per-case inputs and separate deterministic comparisons.

    result['cases'][id] has i,id,fold,group,material,rule_summaries,rule_cards,
    neighbours,knn_score,rinfo_score. Call initial_packet(case, arm) to expose
    only the appropriate scientific fields to an LLM. All cases use the exact
    E0 fixed300 unless explicit indices are provided by the finite-stage runner.
    """
    log = log or (lambda message: None)
    data = load_data() if data is None else data
    indices = fixed_indices(data) if indices is None else np.asarray(indices, dtype=int)
    if len(indices) != len(set(indices.tolist())) or not np.isin(indices, fixed_indices(data)).all():
        raise ValueError("E2 indices must be a unique subset of the E0 fixed300")
    materials, names, units = _load_materials(data, indices)
    by_source = {int(v): i for i, v in enumerate(data["source_index"])}
    cases = {str(data["id"][i]): {
        "i": int(i), "id": str(data["id"][i]), "fold": int(data["fold"][i]),
        "group": str(data["group"][i]), "material": materials[int(i)],
    } for i in indices}
    metadata = {
        "shared_facts": str(FACTS), "n": len(indices),
        "n_groups": len(set(map(str, data["group"][indices]))),
        "min_support_groups": MIN_SUPPORT_GROUPS, "top_rules": MAX_RULES,
        "max_counterexamples_per_rule": MAX_COUNTEREXAMPLES, "neighbours": NEIGHBOURS,
        "delivered_counterexamples_per_rule_cap": 5,
        "prompt_numeric_precision_significant_digits": 8,
        "neighbour_scaling": "sklearn StandardScaler fit on that outer fold TRAIN X127 only; ddof=0; zero-scale->1",
        "neighbour_distance": "Euclidean; exact distance ties broken by ascending original source_index",
        "knn_vote": "unweighted mean of the same five neighbour Tc_gt_10K labels",
        "rule_ranking": "applicable and TRAIN support groups>=20, descending abs(score-0.5), ascending rule_id",
        "counterexamples": "applicable opposing-label TRAIN rows; deterministic SHA256(rule_id|group) order, distinct groups first, then source_index",
        "target_fields": "proxy cell formula, occupancy composition, declared CIF symmetry, lattice and existing global structure summaries; no target Tc/label",
        "folds": {},
    }
    for outer in (0, 1, 2):
        train = np.flatnonzero(data["fold"] != outer)
        held = np.asarray([i for i in indices if int(data["fold"][i]) == outer], dtype=int)
        if set(map(str, data["group"][train])) & set(map(str, data["group"][held])):
            raise ValueError("Target association group appears in TRAIN")
        path = BANKS / f"fold{outer}" / "ROOT_REVIEWABLE_BANK.json"
        rules, bank_meta = load_rules(path)
        if set(bank_meta["train_indices"]) != set(map(int, data["source_index"][train])):
            raise ValueError("Frozen bank TRAIN differs from original fold TRAIN")
        raw_bank = json.loads(path.read_text(encoding="utf-8"))["finalized_bank"]
        envelopes = {e["card"]["rule_id"]: e for e in raw_bank["rule_envelopes"]}
        matrix = applicability_matrix(data["X127"][held], rules)
        rinfo, _, _ = select_rule_scores(matrix, rules, mode="R-info")
        ranking = sorted((j for j, r in enumerate(rules) if r.support_groups >= MIN_SUPPORT_GROUPS),
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
                               "operator": operator, "threshold": threshold,
                               "target_value": float(data["X127"][i, j])}
                              for j, operator, threshold in rule.predicates]
                brief = {"rule_id": rule.rule_id, "conditions": conditions,
                         "class_balanced_score": rule.score,
                         "TRAIN_support_groups": rule.support_groups,
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
                            for s in rule.support_indices),
                    }
                cards[rule.rule_id] = {**brief, **card_cache[rule.rule_id]}
            distances = np.sqrt(np.sum((scaled_train - scaled_held[position]) ** 2, axis=1))
            nearest_positions = np.lexsort((data["source_index"][train], distances))[:NEIGHBOURS]
            nearest = train[nearest_positions]
            case = cases[str(data["id"][i])]
            case.update({
                "rule_summaries": briefs, "rule_cards": cards,
                "neighbours": [_example(data, materials, int(j), distance=distances[k])
                               for j, k in zip(nearest, nearest_positions)],
                "knn_score": float(np.mean(data["y"][nearest])), "rinfo_score": float(rinfo[position]),
            })
        metadata["folds"][str(outer)] = {"n_TRAIN": len(train), "n_targets": len(held),
                                                   "n_rules": len(rules), "bank": str(path)}
        log(f"E2 packets fold {outer}: {len(held)} targets, {len(card_cache)} distinct selected rules")
    # This comparator must be exactly the already-reported E1 R-info, rather
    # than a subtly different implementation using top-six or a new score.
    with (ROOT / "E1/predictions.csv").open(newline="", encoding="utf-8") as handle:
        reference = {r["id"]: float(r["score"]) for r in csv.DictReader(handle) if r["arm"] == "R-info"}
    for case in cases.values():
        if case["rinfo_score"] != reference[case["id"]]:
            raise ValueError("E2 R-info differs from exact E1 comparator")
    metadata["R_info_matches_E1_exactly"] = True
    json.dumps(cases, allow_nan=False)
    return {"cases": cases, "indices": indices.tolist(), "metadata": metadata}


def initial_packet(case, arm):
    """The only accessor the runner should use to construct scientific prompts."""
    packet = {"material": case["material"]}
    if arm == "A1":
        packet["applicable_rule_summaries"] = case["rule_summaries"]
        packet["available_evidence_requests"] = ["rule_card", "counterexamples"]
    elif arm == "A2":
        packet["applicable_rules_full_cards"] = list(case["rule_cards"].values())
        packet["five_nearest_TRAIN_materials"] = case["neighbours"]
    elif arm != "A3":
        raise ValueError(arm)
    return packet


def requested_evidence(case, rule_ids, *, include_cards=True, include_counterexamples=True):
    """Return requested evidence only for this case's frozen top-six rules."""
    if len(rule_ids) != len(set(rule_ids)) or any(r not in case["rule_cards"] for r in rule_ids):
        raise ValueError("Evidence request names unavailable or duplicate rules")
    result = []
    for rule_id in rule_ids:
        card = case["rule_cards"][rule_id]
        item = {k: v for k, v in card.items() if k != "counterexamples"} if include_cards else {"rule_id": rule_id}
        if include_counterexamples:
            item["counterexamples"] = card["counterexamples"]
        result.append(item)
    return result


def round_prompt_numbers(value, significant_digits=8):
    """Round display values only; deterministic scoring remains full precision."""
    if isinstance(value, dict):
        return {key: round_prompt_numbers(item, significant_digits) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [round_prompt_numbers(item, significant_digits) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Scientific prompt contains a non-finite number")
        return float(format(value, f".{significant_digits}g"))
    return value


def compact_evidence(cards, *, briefs_already_supplied=True, significant_digits=8,
                     max_examples=5, rule_summaries=None):
    """Serialize full cards without repeated common text or example objects.

    Pass the same initial rule_summaries alongside this object when leaving
    briefs_already_supplied=True. Every omitted brief field is explicitly named
    in brief_fields_reference and is looked up by rule_id. Defaults apply to all
    supplied cards; per-card fields override them. The counterexample table uses
    zero-based row references and delivers the first max_examples examples in
    the already-frozen deterministic order. The finite E2 run freezes this at
    five for both A1 requests and A2. Stored cards retain up to ten for reference.
    """
    cards = list(cards.values()) if isinstance(cards, dict) else list(cards)
    if not 0 <= int(max_examples) <= MAX_COUNTEREXAMPLES:
        raise ValueError("Compact evidence cap must be between zero and ten")
    cards = [{**card, **({"counterexamples": card["counterexamples"][:int(max_examples)]}
                        if "counterexamples" in card else {})} for card in cards]
    brief_keys = {"conditions", "class_balanced_score", "TRAIN_support_groups", "TRAIN_coverage_rows"}
    if len({card["rule_id"] for card in cards}) != len(cards):
        raise ValueError("Duplicate compact rule cards")
    rule_positions = None
    if rule_summaries is not None:
        rule_positions = {brief["rule_id"]: j for j, brief in enumerate(rule_summaries)}
        if len(rule_positions) != len(rule_summaries) or any(card["rule_id"] not in rule_positions for card in cards):
            raise ValueError("Rule summary lookup is duplicate or incomplete")
    shared = {}
    if cards:
        for key, value in cards[0].items():
            if key in {"rule_id", "counterexamples"} or (briefs_already_supplied and key in brief_keys):
                continue
            if all(key in card and card[key] == value for card in cards):
                shared[key] = value
    # A column/row table keeps all example fields while eliminating repeated
    # JSON property names. IDs remain available for review and cross-rule joins.
    examples, seen = [], {}
    columns = ["id", "formula", "Tc_K", "Tc_gt_10K"]
    compact_cards = []
    for card in cards:
        reduced = {key: value for key, value in card.items()
                   if key not in shared and key != "counterexamples"
                   and not (briefs_already_supplied and key in brief_keys)}
        if rule_positions is not None:
            reduced["initial_rule_index"] = rule_positions[reduced.pop("rule_id")]
        references = []
        for example in card.get("counterexamples", []):
            for key in example:
                if key not in columns:
                    columns.append(key)
            identity = str(example["id"])
            if identity in seen:
                position = seen[identity]
                if examples[position] != example:
                    raise ValueError("Conflicting TRAIN counterexample identity")
            else:
                position = len(examples)
                seen[identity] = position
                examples.append(example)
            references.append(position)
        if "counterexamples" in card:
            reduced["counterexample_rows"] = references
        compact_cards.append(reduced)
    card_columns = []
    for card in compact_cards:
        for key in card:
            if key not in card_columns:
                card_columns.append(key)
    # The six cards share the same small set of varying fields; a table avoids
    # repeating their long property names while keeping every value explicit.
    card_table = {"columns": card_columns,
                  "rows": [[card.get(key) for key in card_columns] for card in compact_cards]}
    table = {"columns": columns,
             "rows": [[example.get(key) for key in columns] for example in examples]}
    # All current material identifiers have a shared prefix and fixed-width
    # decimal suffix. Factoring that prefix is reversible and avoids repeating
    # it up to sixty times in the scientific input budget.
    identities = [re.fullmatch(r"(.*?)([0-9]+)", str(example["id"])) for example in examples]
    if examples and all(match is not None for match in identities):
        prefixes = {match.group(1) for match in identities}
        widths = {len(match.group(2)) for match in identities}
        if len(prefixes) == len(widths) == 1:
            id_column = columns.index("id")
            table["columns"] = ["id_number" if key == "id" else key for key in columns]
            for row, match in zip(table["rows"], identities):
                row[id_column] = int(match.group(2))
            table["id_encoding"] = {"prefix": next(iter(prefixes)), "decimal_width": next(iter(widths)),
                                    "meaning": "id = prefix + id_number padded with leading zeroes to decimal_width"}
    result = {
        "numeric_display_precision": f"{significant_digits} significant digits; executor uses unrounded rule scores",
        "delivered_counterexamples_per_rule_cap": int(max_examples),
        "rule_card_defaults": shared,
        "rule_cards": card_table,
        "counterexample_table": table,
        "counterexample_reference": "counterexample_rows are zero-based rows of counterexample_table; all examples are applicable opposing-label outer TRAIN materials",
    }
    if briefs_already_supplied:
        result["brief_fields_reference"] = "conditions, class_balanced_score, TRAIN_support_groups, TRAIN_coverage_rows: use initial rule summary with the same rule_id"
    if rule_positions is not None:
        result["rule_reference"] = "initial_rule_index is the zero-based position in applicable_rule_summaries; rule_card_defaults apply to every rule_cards table row"
    return round_prompt_numbers(result, significant_digits)
