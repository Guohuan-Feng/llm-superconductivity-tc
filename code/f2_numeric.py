"""Prepare anonymous, fold-standardized F2 numeric inputs without model calls.

Only ``numeric_packet(case, arm)`` is suitable for scientific prompts. The outer
case and metadata deliberately retain identities, scaler parameters and original
feature names for an input audit; they must never be sent to the evaluated LLM.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from common import ROOT, load_data, np
from e2_packets import FACTS, fixed_indices
from sklearn.preprocessing import StandardScaler


SIGNIFICANT_DIGITS = 8
NEIGHBOURS = 5


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_vector(values):
    """Eight significant figures, matching the E2 scientific prompt precision."""
    if len(values) != 127 or not np.isfinite(values).all():
        raise ValueError("F2 numeric vector must contain exactly 127 finite values")
    return {f"f{j + 1}": float(format(float(value), f".{SIGNIFICANT_DIGITS}g"))
            for j, value in enumerate(values)}


def numeric_packet(case, arm):
    """Return only the allowed target features and, for A2-M2, TRAIN labels."""
    if arm not in ("A3-M2", "A2-M2"):
        raise ValueError(f"Unknown numeric masking arm: {arm}")
    return case[arm]


def compact_numeric_packet(case, arm):
    """Lossless compact display: one column header and anonymous numeric rows.

    This changes only JSON representation, never precision or feature ordering.
    The material and every TRAIN neighbour share the same 127-column header.
    """
    original = numeric_packet(case, arm)
    feature_order = [f"f{j + 1}" for j in range(127)]
    if list(original["features"]) != feature_order:
        raise ValueError("Anonymous target feature order changed")
    packet = {"feature_order": feature_order,
              "material": {"values": list(original["features"].values())}}
    if arm == "A2-M2":
        packet["neighbours"] = []
        for neighbour in original["neighbours"]:
            if list(neighbour["features"]) != feature_order:
                raise ValueError("Anonymous neighbour feature order changed")
            packet["neighbours"].append({"values": list(neighbour["features"].values()),
                                          "Tc_gt_10K": neighbour["Tc_gt_10K"]})
    return packet


def serialize_numeric_packet(case, arm):
    """Serialize the complete compact scientific payload with no audit fields."""
    return json.dumps(compact_numeric_packet(case, arm), ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def build_numeric_inputs(data=None):
    """Return the fixed300 packets and separate JSON-safe input audit metadata.

    The same five TRAIN neighbours as E2 are independently recomputed from
    unrounded standardized X127, with its exact source-index tie breaker. Both
    targets and neighbours are standardized using the target's outer TRAIN fit.
    No rule cards, identities, Tc values, feature meanings, space-group names or
    target labels enter either packet. Neighbour labels are Tc > 10 K booleans.
    """
    data = load_data() if data is None else data
    indices = fixed_indices(data)
    if data["X127"].shape != (3764, 127):
        raise ValueError("Unexpected accepted X127 shape")
    config_path = ROOT / "E2/config.json"
    previous = json.loads(config_path.read_text(encoding="utf-8"))
    reference_cases = previous["cases"]
    selected_ids = [str(data["id"][i]) for i in indices]
    if set(reference_cases) != set(selected_ids):
        raise ValueError("E2 case manifest does not match the exact fixed300")
    by_id = {str(material_id): i for i, material_id in enumerate(data["id"])}
    if len(by_id) != len(data["id"]):
        raise ValueError("Accepted material IDs are not unique")

    # Names are retained only in metadata, never in numeric_packet().
    with FACTS.open(encoding="utf-8") as handle:
        facts = json.loads(next(handle))["coordinator_record"]["common_facts"]
    feature_names = facts["feature_names"][:127]
    if len(feature_names) != 127 or len(set(feature_names)) != 127:
        raise ValueError("Original X127 feature-name mapping is malformed")
    if feature_names[:118] != [f"occupancy_fraction_Z{j:03d}" for j in range(1, 119)]:
        raise ValueError("Frozen X127 composition feature ordering changed")

    metadata = {
        "n": len(indices),
        "n_groups": len(set(map(str, data["group"][indices]))),
        "numeric_dimension": 127,
        "prompt_numeric_precision_significant_digits": SIGNIFICANT_DIGITS,
        "scaling": "StandardScaler fit on the original target outer fold TRAIN only; ddof=0; zero scale->1",
        "neighbour_selection": "exact E2 five TRAIN neighbours; Euclidean distance on unrounded standardized X127; ties by original source_index",
        "neighbour_label": "binary Tc_gt_10K only; no numeric Tc, formula, identity, fold, group or distance supplied",
        "A3_M2_scientific_fields": ["features: f1-f127"],
        "A2_M2_scientific_fields": ["features: f1-f127", "neighbours: five feature vectors with binary TRAIN labels"],
        "original_feature_name_mapping_not_for_prompt": {
            f"f{j + 1}": name for j, name in enumerate(feature_names)},
        "source_E2_config": str(config_path),
        "source_E2_config_sha256": _sha256(config_path),
        "source_feature_names": str(FACTS),
        "folds": {},
        "comparison_boundaries": [
            "The LLM receives eight-significant-digit text; deterministic comparator fitting and nearest-neighbour selection use the original full-precision numeric data.",
            "Feature renaming and affine standardization do not destroy feature information; the first 118 original columns are ordered atomic-number occupancies, so position or sparsity patterns can still reveal identity if the model infers this mapping.",
            "A3-M2 receives one target vector and no labelled TRAIN set; HGB is fitted on all original outer TRAIN rows. A2-M2 receives only the same five labelled TRAIN neighbours, rather than HGB's entire TRAIN set.",
            "This module follows the predeclared column ordering; no column permutation, feature engineering, model fitting beyond StandardScaler, or hyperparameter search is performed.",
        ],
    }
    cases = {}
    max_rounding_error = 0.0
    for outer in (0, 1, 2):
        train = np.flatnonzero(data["fold"] != outer)
        held = np.asarray([i for i in indices if int(data["fold"][i]) == outer], dtype=int)
        train_groups = set(map(str, data["group"][train]))
        held_groups = set(map(str, data["group"][held]))
        if train_groups & held_groups:
            raise ValueError("Target association group occurs in its outer TRAIN")
        scaler = StandardScaler().fit(data["X127"][train])
        scaled_train = scaler.transform(data["X127"][train])
        scaled_held = scaler.transform(data["X127"][held])
        metadata["folds"][str(outer)] = {
            "n_TRAIN": len(train), "n_TRAIN_groups": len(train_groups),
            "n_targets": len(held), "n_target_groups": len(held_groups),
            "TRAIN_target_group_overlap": 0,
            "mean_not_for_prompt": scaler.mean_.tolist(),
            "scale_not_for_prompt": scaler.scale_.tolist(),
            "variance_not_for_prompt": scaler.var_.tolist(),
        }
        for position, i in enumerate(held):
            material_id = str(data["id"][i])
            reference = reference_cases[material_id]
            if (reference["i"] != int(i) or reference["id"] != material_id
                    or reference["fold"] != outer or reference["group"] != str(data["group"][i])):
                raise ValueError("E2 case identity differs from accepted data")
            distances = np.sqrt(np.sum((scaled_train - scaled_held[position]) ** 2, axis=1))
            nearest_positions = np.lexsort((data["source_index"][train], distances))[:NEIGHBOURS]
            nearest = train[nearest_positions]
            reference_neighbours = reference["neighbours"]
            if len(reference_neighbours) != NEIGHBOURS:
                raise ValueError("E2 case does not have exactly five neighbours")
            if [str(data["id"][j]) for j in nearest] != [r["id"] for r in reference_neighbours]:
                raise ValueError(f"F2 recomputed neighbour order differs from E2: {material_id}")
            target_features = _feature_vector(scaled_held[position])
            rounded_target = np.asarray(list(target_features.values()))
            max_rounding_error = max(max_rounding_error,
                                     float(np.max(np.abs(rounded_target - scaled_held[position]))))
            neighbour_packets = []
            audit_neighbours = []
            for j, neighbour_position, old in zip(nearest, nearest_positions, reference_neighbours):
                j = int(j)
                if by_id[old["id"]] != j or int(data["fold"][j]) == outer:
                    raise ValueError("E2 neighbour is not in the target's outer TRAIN")
                if str(data["group"][j]) == str(data["group"][i]):
                    raise ValueError("Target group leaked through a neighbour")
                if bool(old["Tc_gt_10K"]) != bool(data["y"][j]):
                    raise ValueError("E2 neighbour binary label differs")
                if float(old["Tc_K"]) != float(data["tc"][j]):
                    raise ValueError("E2 neighbour original Tc differs")
                if not np.isclose(float(old["standardized_X127_Euclidean_distance"]),
                                  distances[neighbour_position], rtol=1e-13, atol=1e-13):
                    raise ValueError("Recomputed E2 neighbour distance differs")
                neighbour_features = _feature_vector(scaled_train[neighbour_position])
                rounded_neighbour = np.asarray(list(neighbour_features.values()))
                max_rounding_error = max(max_rounding_error, float(np.max(
                    np.abs(rounded_neighbour - scaled_train[neighbour_position]))))
                neighbour_packets.append({"features": neighbour_features,
                                          "Tc_gt_10K": bool(data["y"][j])})
                audit_neighbours.append({"id": str(data["id"][j]), "i": j,
                    "fold": int(data["fold"][j]), "group": str(data["group"][j]),
                    "distance": float(distances[neighbour_position])})
            knn_score = float(np.mean(data["y"][nearest]))
            if knn_score != float(reference["knn_score"]):
                raise ValueError("F2 five-neighbour baseline score differs from E2")
            cases[material_id] = {
                "i": int(i), "id": material_id, "fold": outer,
                "group": str(data["group"][i]),
                "A3-M2": {"features": target_features},
                "A2-M2": {"features": target_features, "neighbours": neighbour_packets},
                "neighbour_audit_not_for_prompt": audit_neighbours,
                "knn_score_not_for_prompt": knn_score,
            }
    if set(cases) != set(selected_ids):
        raise ValueError("F2 numeric cases are incomplete")
    cases = {material_id: cases[material_id] for material_id in selected_ids}
    metadata["all_300_neighbour_lists_and_scores_match_E2"] = True
    metadata["all_neighbours_outer_TRAIN_and_target_group_excluded"] = True
    metadata["max_absolute_prompt_rounding_error"] = max_rounding_error
    metadata["compact_prompt_representation"] = (
        "feature_order f1-f127 appears once; material and five TRAIN neighbours use values arrays in exactly that column order; lossless relative to eight-significant-digit dictionaries")
    metadata["scientific_payload_utf8_bytes"] = {}
    for arm in ("A3-M2", "A2-M2"):
        lengths = [len(serialize_numeric_packet(case, arm).encode("utf-8"))
                   for case in cases.values()]
        metadata["scientific_payload_utf8_bytes"][arm] = {"min": min(lengths), "max": max(lengths)}
    json.dumps({"cases": cases, "metadata": metadata}, allow_nan=False)
    return {"cases": cases, "indices": indices.tolist(), "metadata": metadata}
