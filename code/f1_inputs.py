"""Freeze formula-only F1 inputs without calls, fitted models, or file writes.

The exact formula string is recovered from the scientific input JSON of each
completed E2 A3 call. Evaluation metadata is kept in a separate dictionary and
must never be serialized into a recall prompt. Negative controls are absent from
this data set by reduced composition; that is not a claim of real-world novelty.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from fractions import Fraction
from functools import reduce
from pathlib import Path

from common import DATA, ROOT, load_data
from e2_packets import FACTS, fixed_indices


CONTROL_SEED = 20261017
N_CONTROLS = 20
FORMULA_SOURCE = "E2/llm_calls.jsonl: A3 prompt -> Scientific inputs JSON -> material.formula"
_TOKEN = re.compile(r"([A-Z][a-z]?)(\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)?")
_ELEMENTS = frozenset((
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La "
    "Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi "
    "Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs "
    "Mt Ds Rg Cn Nh Fl Mc Lv Ts Og").split())


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def formula_amounts(formula):
    """Parse the flat occupancy notation used in E2; reject stray text."""
    amounts = defaultdict(Fraction)
    position = 0
    for match in _TOKEN.finditer(formula):
        if match.start() != position or match.group(1) not in _ELEMENTS:
            raise ValueError(f"Unrecognized occupancy formula: {formula!r}")
        amount = Fraction(match.group(2) or "1")
        if amount <= 0:
            raise ValueError("Formula occupancies must be positive")
        # Input amounts were rounded to ten significant figures in E2. Restore
        # ordinary rational occupancies within 1e-8 relative tolerance only.
        rational = amount.limit_denominator(1_000_000)
        if rational > 0 and abs(float(rational - amount)) <= 1e-8 * max(1.0, float(amount)):
            amount = rational
        amounts[match.group(1)] += amount
        position = match.end()
    if position != len(formula) or not amounts:
        raise ValueError(f"Unrecognized occupancy formula: {formula!r}")
    return dict(amounts)


def reduced_composition_key(formula):
    """Canonical sorted element/integer ratio key, independent of cell size."""
    amounts = formula_amounts(formula)
    denominator = math.lcm(*(amount.denominator for amount in amounts.values()))
    integers = {element: int(amount * denominator) for element, amount in amounts.items()}
    divisor = reduce(math.gcd, integers.values())
    return tuple((element, integers[element] // divisor) for element in sorted(integers))


def _exact_a3_formulas(data, indices):
    selected_ids = {str(data["id"][i]) for i in indices}
    formulas, request_hashes, counts = {}, {}, Counter()
    with (ROOT / "E2/llm_calls.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            call = json.loads(line)
            material_id = str(call.get("id", ""))
            if call.get("arm") != "A3" or material_id not in selected_ids:
                continue
            prompt = call["prompt"]
            if "Scientific inputs:\n" not in prompt:
                raise ValueError("E2 A3 scientific-input delimiter missing")
            payload = json.loads(prompt.split("Scientific inputs:\n", 1)[1])
            formula = payload["material"]["formula"]
            if not isinstance(formula, str) or not formula:
                raise ValueError("E2 A3 formula is empty or not a string")
            if material_id in formulas and formulas[material_id] != formula:
                raise ValueError("E2 A3 repeated calls disagree on exact formula")
            formulas[material_id] = formula
            request_hashes[material_id] = call.get("request_hash")
            counts[material_id] += 1
    if set(formulas) != selected_ids:
        raise ValueError("The fixed300 E2 A3 formula set is incomplete")
    return formulas, request_hashes, counts


def _cohort_formula_index(data):
    """Read frozen common facts, retaining formula strings and composition keys."""
    by_source = {int(value): i for i, value in enumerate(data["source_index"])}
    formulas, keys, digest = {}, defaultdict(list), hashlib.sha256()
    with FACTS.open("rb") as handle:
        for line in handle:
            digest.update(line)
            record = json.loads(line)["coordinator_record"]
            facts = record["common_facts"]
            identity = facts["identity"]
            i = by_source[int(identity["source_index"])]
            material_id = str(data["id"][i])
            if (str(identity["material_id"]) != material_id
                    or str(identity["component_id"]) != str(data["group"][i])
                    or int(identity["outer_fold"]) != int(data["fold"][i])):
                raise ValueError("Common-facts identity differs from accepted data")
            if material_id in formulas:
                raise ValueError("Duplicate common-facts material")
            raw = facts["raw_graph"]["raw_structure"]
            parts = defaultdict(list)
            for symbol, amount in zip(raw["species_symbols"], raw["species_occupancy"]):
                parts[str(symbol)].append(float(amount))
            formula = "".join(symbol + format(math.fsum(parts[symbol]), ".10g")
                              for symbol in sorted(parts) if math.fsum(parts[symbol]) > 0)
            formulas[material_id] = formula
            keys[reduced_composition_key(formula)].append(material_id)
    if len(formulas) != len(data["id"]):
        raise ValueError("Common-facts formulas do not cover the full3764")
    return formulas, dict(keys), digest.hexdigest()


def _control_formulas(cohort_keys, count, seed):
    """Seeded materials-like ternary/quaternary formulas, checked before use."""
    rng = random.Random(seed)
    transition = "Ti V Cr Mn Fe Co Ni Cu Zn Zr Nb Mo Hf Ta W".split()
    main_group = "Al Ga In Si Ge Sn Pb".split()
    cations = "Mg Ca Sr Ba Y La Ce Pr Nd Sm Gd".split()
    anions = "O S Se Te N P B C".split()
    selected, selected_keys, attempted = [], set(), 0
    while len(selected) < count:
        attempted += 1
        if attempted > count * 1000:
            raise RuntimeError("Failed to construct absent control compositions")
        template = rng.randrange(3)
        if template == 0:
            elements = [rng.choice(cations), rng.choice(transition), rng.choice(anions)]
            coefficients = [rng.randint(1, 3), rng.randint(1, 4), rng.randint(2, 8)]
        elif template == 1:
            elements = rng.sample(transition, 2) + [rng.choice(main_group)]
            coefficients = [rng.randint(1, 5), rng.randint(1, 3), rng.randint(1, 4)]
        else:
            elements = [rng.choice(cations), rng.choice(transition),
                        rng.choice(main_group), rng.choice(anions)]
            coefficients = [rng.randint(1, 3), rng.randint(1, 4),
                            rng.randint(1, 3), rng.randint(3, 9)]
        formula = "".join(element + str(coefficient)
                          for element, coefficient in sorted(zip(elements, coefficients)))
        key = reduced_composition_key(formula)
        if key in cohort_keys or key in selected_keys:
            continue
        selected.append(formula)
        selected_keys.add(key)
    return selected, attempted


def formula_only_input(case):
    """The complete scientific payload allowed into one independent F1 call."""
    return {"formula": case["formula"]}


def build_f1_inputs(data=None, *, n_controls=N_CONTROLS, seed=CONTROL_SEED):
    """Return ordered cases and JSON-safe frozen-input/deduplication evidence.

    Cases use E0 fixed300 order, then seeded controls. Top-level identity fields
    are routing metadata only; pass formula_only_input(case), never the full
    case, into the scientific prompt. All outcomes reside under evaluation.
    """
    data = load_data() if data is None else data
    indices = fixed_indices(data)
    formulas, request_hashes, counts = _exact_a3_formulas(data, indices)
    all_formulas, cohort_keys, facts_hash = _cohort_formula_index(data)
    for material_id, formula in formulas.items():
        if formula != all_formulas[material_id]:
            raise ValueError("A3 formula is not byte-identical to common-facts reconstruction")
    controls, attempted = _control_formulas(cohort_keys, n_controls, seed)
    cases = []
    for i in indices:
        material_id = str(data["id"][i])
        case = {
            "id": material_id, "fold": int(data["fold"][i]),
            "group": str(data["group"][i]), "source_index": int(data["source_index"][i]),
            "is_control": False, "formula": formulas[material_id],
            "evaluation": {
                "Tc_K": float(data["tc"][i]), "y": int(data["y"][i]),
                "original_group_size": int(data["group_size"][i]),
                "accepted_row_index": int(i),
            },
        }
        case["prompt_input"] = formula_only_input(case)
        cases.append(case)
    control_evidence = []
    for j, formula in enumerate(controls, start=1):
        key = reduced_composition_key(formula)
        control_id = f"F1_control_{j:03d}"
        case = {
            "id": control_id, "fold": -1, "group": control_id,
            "source_index": None, "is_control": True, "formula": formula,
            "evaluation": {"Tc_K": None, "y": None,
                           "original_group_size": 1, "accepted_row_index": None},
        }
        case["prompt_input"] = formula_only_input(case)
        cases.append(case)
        control_evidence.append({
            "id": control_id, "formula": formula, "reduced_composition_key": list(key),
            "matching_accepted_material_ids": cohort_keys.get(key, []),
        })
    metadata = {
        "formula_source": FORMULA_SOURCE,
        "scientific_payload_keys": ["formula"],
        "evaluation_fields_excluded_from_prompt": ["id", "fold", "group", "source_index",
                                                    "is_control", "evaluation"],
        "n_real": len(indices), "n_real_groups": len({case["group"] for case in cases if not case["is_control"]}),
        "n_real_unique_formula_strings": len(set(formulas.values())),
        "n_controls": len(controls), "control_seed": seed, "control_candidate_attempts": attempted,
        "n_cohort_rows": len(all_formulas), "n_cohort_reduced_compositions": len(cohort_keys),
        "formula_exact_match_count": len(formulas),
        "e2_a3_call_count_distribution": dict(sorted(Counter(counts.values()).items())),
        "e2_a3_request_hashes": request_hashes,
        "composition_normalization": "sorted element/integer ratios; rational occupancy denominators<=1000000 restored only within 1e-8 relative tolerance",
        "control_interpretation": "seeded plausible-looking formulas absent from accepted3764 by reduced composition; no assertion of absence from literature or pretraining",
        "controls_absent_by_reduced_composition": all(not row["matching_accepted_material_ids"] for row in control_evidence),
        "control_deduplication": control_evidence,
        "input_sha256": {
            str(DATA): sha256_file(DATA), str(FACTS): facts_hash,
            str(ROOT / "E0/predictions.csv"): sha256_file(ROOT / "E0/predictions.csv"),
            str(ROOT / "E2/llm_calls.jsonl"): sha256_file(ROOT / "E2/llm_calls.jsonl"),
        },
    }
    metadata["ordered_formula_manifest_sha256"] = hashlib.sha256(json.dumps(
        [{"id": case["id"], "formula": case["formula"], "is_control": case["is_control"]} for case in cases],
        ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"cases": cases, "metadata": metadata}
