"""Label-blind, deterministic anonymous material packets for finite F2 A3-M1.

Only this module is new. Existing E2 material packets and local elemental tables
are read-only inputs; no experimental/model calls or filesystem writes occur.
"""
from __future__ import annotations

import bisect
import copy
import hashlib
import json
import math
import random
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_SEED = 20261017
MAGPIE_DIR = ROOT / "work/p1-env/Lib/site-packages/matminer/utils/data_files/magpie_elementdata"
SOURCE_README_URL = "https://raw.githubusercontent.com/hackingmaterials/matminer/main/src/matminer/utils/data_files/magpie_elementdata/README.txt"
SOURCE_DOCS_URL = "https://hackingmaterials.lbl.gov/matminer/matminer.utils.html#matminer.utils.data.MagpieData"
ELEMENTS = tuple((
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La "
    "Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po "
    "At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg "
    "Cn Nh Fl Mc Lv Ts Og"
).split())
PROPERTY_FILES = {
    "electronegativity_bin": "Electronegativity.table",
    "covalent_radius_bin": "CovalentRadius.table",
    "valence_electron_count_bin": "NValence.table",
}
GEOMETRY_KEYS = frozenset({
    "volume_per_occupied_atom_A3",
    "nearest_positive_PBC_distance_site_occ_mean_A",
    "nearest_positive_PBC_distance_site_occ_std_A",
    "nearest_positive_PBC_distance_min_A",
    "nearest_positive_PBC_distance_max_A",
    "coordination6A_site_occ_mean", "coordination6A_site_occ_std",
    "PBC_edge_distance_pair_occ_mean_A", "PBC_edge_distance_pair_occ_std_A",
    "cell_site_count", "cell_positive_site_count", "cell_total_absolute_occupancy",
    "cell_volume_A3", "cell_a_length_A", "cell_b_length_A", "cell_c_length_A",
    "cell_alpha_cosine", "cell_beta_cosine", "cell_gamma_cosine",
    "site_absolute_occupancy_unweighted_mean", "site_absolute_occupancy_unweighted_std",
})
REMOVED_STRUCTURE_FIELDS = ["existing_global_structure_features.occupied_mass_on_partial_sites_fraction"]


def _read_table(path: Path) -> list[float | None]:
    """Match MagpieData's atomic-number row order without importing its runtime."""
    lines = path.read_text(encoding="utf-8").splitlines()
    result = []
    for index in range(len(ELEMENTS)):
        try:
            value = float(lines[index].strip())
        except (ValueError, IndexError):
            value = None
        result.append(value if value is not None and math.isfinite(value) else None)
    return result


def _quantile_linear(values: list[float], fraction: float) -> float:
    """Sorted finite-table quantile: NumPy method='linear', no fitted targets."""
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


@lru_cache(maxsize=1)
def load_property_table() -> dict:
    """Return all-element, not cohort-fitted, five-bin properties and metadata."""
    if len(ELEMENTS) != 118:
        raise ValueError("Periodic table symbol order is incomplete")
    raw, file_hashes = {}, {}
    for key, filename in PROPERTY_FILES.items():
        path = MAGPIE_DIR / filename
        raw[key] = _read_table(path)
        file_hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    # NValence must be the sum of the tabulated s/p/d/f valence occupancies,
    # not oxidation state, periodic-table group, or total electron count.
    components = [_read_table(MAGPIE_DIR / f"N{orbital}Valence.table") for orbital in "spdf"]
    checked_valence_elements = 0
    for index, value in enumerate(raw["valence_electron_count_bin"]):
        component_values = [part[index] for part in components]
        if value is not None and all(v is not None for v in component_values):
            if value != sum(component_values):
                raise ValueError(f"NValence differs from occupied s+p+d+f counts at row {index + 1}")
            checked_valence_elements += 1
    for orbital in "spdf":
        path = MAGPIE_DIR / f"N{orbital}Valence.table"
        file_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    cutoffs = {key: [_quantile_linear([v for v in values if v is not None], q)
                     for q in (.2, .4, .6, .8)] for key, values in raw.items()}
    binned = {}
    for index, symbol in enumerate(ELEMENTS):
        binned[symbol] = {
            key: "unknown" if values[index] is None else bisect.bisect_left(cutoffs[key], values[index]) + 1
            for key, values in raw.items()
        }
    readme_path = MAGPIE_DIR / "README.txt"
    file_hashes[readme_path.name] = hashlib.sha256(readme_path.read_bytes()).hexdigest()
    metadata = {
        "source": "Installed matminer 0.10.1 Magpie tables, read directly with no automatic imputation",
        "local_directory": str(MAGPIE_DIR),
        "source_urls": [SOURCE_README_URL, SOURCE_DOCS_URL],
        "sha256": file_hashes,
        "property_definitions": {
            "electronegativity_bin": "Magpie Electronegativity, described as Pauling by source README (which marks the scale with a question). Table already includes source-level Eu/Yb/Tb/Pm estimates.",
            "covalent_radius_bin": "Magpie CovalentRadius, pm, Mathematica provenance; this is NOT AtomicRadius or ionic radius.",
            "valence_electron_count_bin": "Magpie NValence, occupied s+p+d+f valence-electron counts of neutral elements, not oxidation state or periodic-table group. Includes occupied d/f electrons in Magpie convention.",
        },
        "binning": "Five ordered bins, 1 low to 5 high. Four cutoffs are 20/40/60/80 percentiles across finite available Z=1..118 table values with equal element weight and linear interpolation. Exact cutoff ties go to the lower bin; tied properties need not give equal bin counts.",
        "cutoffs": cutoffs,
        "cutoff_units": {"electronegativity_bin": "Magpie electronegativity scale", "covalent_radius_bin": "pm", "valence_electron_count_bin": "electrons"},
        "finite_reference_element_counts": {key: sum(v is not None for v in values) for key, values in raw.items()},
        "valence_s_plus_p_plus_d_plus_f_crosschecked_elements": checked_valence_elements,
        "missing_handling": "Absent/NaN/nonfinite values or unknown element symbols -> literal unknown. No cohort imputation, no exact Z/mass/row/group fallback.",
        "label_independence": "Only fixed elemental tables determine bins; target labels, Tc, X127, performance, and cohort frequencies never enter cutoffs.",
        "anonymization": "Independently permute sorted element symbols within each material using random.Random(int(SHA256('F2-A3-M1|seed|case_id'))). Assign E1..Ek after permutation. Seed and IDs are not emitted in prompt material.",
        "default_seed": DEFAULT_SEED,
        "removed_from_E2_structure": REMOVED_STRUCTURE_FIELDS,
        "identity_mask_limit": "Coarse properties, exact stoichiometry and retained geometry/space group may still permit identity or family inference; masking is not proof of unrecognizability.",
    }
    return {"bins_by_element": binned, "metadata": metadata}


def property_metadata(property_table: dict | None = None) -> dict:
    """Configuration-only provenance; do not append this to model input."""
    table = load_property_table() if property_table is None else property_table
    return copy.deepcopy(table["metadata"])


def build_m1(material: dict, case_id: str, *, seed: int = DEFAULT_SEED,
             property_table: dict | None = None) -> dict:
    """Mask the existing E2 material; return no target ID, formula or symbols."""
    table = load_property_table() if property_table is None else property_table
    rows = list(material["composition"])
    symbols = sorted(str(row["element"]) for row in rows)
    if len(symbols) != len(set(symbols)) or not symbols:
        raise ValueError("E2 material must have unique occupied element rows")
    rng_seed = int.from_bytes(hashlib.sha256(f"F2-A3-M1|{seed}|{case_id}".encode("utf-8")).digest(), "big")
    random.Random(rng_seed).shuffle(symbols)
    by_symbol = {str(row["element"]): row for row in rows}
    composition = []
    for number, symbol in enumerate(symbols, start=1):
        row = by_symbol[symbol]
        composition.append({
            "anonymous_element": f"E{number}",
            "cell_occupancy": row["cell_occupancy"],
            "atomic_fraction": row["atomic_fraction"],
            **table["bins_by_element"].get(symbol, {key: "unknown" for key in PROPERTY_FILES}),
        })
    original = material["structure"]
    existing = original["existing_global_structure_features"]
    extras = set(existing) - GEOMETRY_KEYS - {"occupied_mass_on_partial_sites_fraction"}
    if extras:
        raise ValueError(f"Unreviewed structure summaries: {sorted(extras)}")
    structure = {
        "coordinate_source_proxy": bool(original["coordinate_source_proxy"]),
        "measured_specimen_match": original["measured_specimen_match"],
        "space_group_declared_in_source_CIF": copy.deepcopy(original["space_group_declared_in_source_CIF"]),
        "lattice_lengths_A": copy.deepcopy(original["lattice_lengths_A"]),
        "lattice_angles_alpha_beta_gamma_deg": copy.deepcopy(original["lattice_angles_alpha_beta_gamma_deg"]),
        "cell_volume_A3": original["cell_volume_A3"],
        "n_sites": original["n_sites"],
        "existing_global_structure_features": {key: existing[key] for key in existing if key in GEOMETRY_KEYS},
    }
    out = {
        "composition_semantics": "Occupancy composition of supplied proxy structure cell; not reduced. E1..Ek are independently randomized within this material and have no shared identity across materials.",
        "property_bin_scale": "Each listed elemental property uses five ordered bins: 1 lowest, 5 highest; unknown means unavailable. Only the given data may be used for this judgment.",
        "composition": composition,
        "structure": structure,
    }
    # Serialization proves finite JSON and rejects accidental custom objects.
    json.dumps(out, allow_nan=False)
    return out


def audit_fixed300(e2_config: dict | None = None) -> dict:
    """Pure read-only input audit; returns aggregate checks, writes nothing."""
    config = e2_config
    if config is None:
        config = json.loads((ROOT / "E2/config.json").read_text(encoding="utf-8"))
    cases = config["cases"]
    missing, fingerprints, elements_seen = {}, [], set()
    for case_id, case in cases.items():
        old = case["material"]
        old_serialized = json.dumps(old, sort_keys=True, allow_nan=False)
        packet = build_m1(old, case_id)
        if packet != build_m1(old, case_id):
            raise AssertionError("Anonymization was not reproducible")
        if len(packet["composition"]) != len(old["composition"]):
            raise AssertionError("Anonymization changed element count")
        if sorted((row["cell_occupancy"], row["atomic_fraction"]) for row in packet["composition"]) != sorted((row["cell_occupancy"], row["atomic_fraction"]) for row in old["composition"]):
            raise AssertionError("Anonymization changed occupancy/stoichiometry")
        if "formula" in packet or any("element" in row for row in packet["composition"]):
            raise AssertionError("Raw material identity remained")
        if json.dumps(old, sort_keys=True, allow_nan=False) != old_serialized:
            raise AssertionError("Read-only E2 material was mutated")
        for key in ("coordinate_source_proxy", "measured_specimen_match",
                    "space_group_declared_in_source_CIF", "lattice_lengths_A",
                    "lattice_angles_alpha_beta_gamma_deg", "cell_volume_A3", "n_sites"):
            if packet["structure"][key] != old["structure"][key]:
                raise AssertionError("Retained E2 structure field changed")
        expected_geometry = {key: value for key, value in old["structure"]["existing_global_structure_features"].items()
                             if key != "occupied_mass_on_partial_sites_fraction"}
        if packet["structure"]["existing_global_structure_features"] != expected_geometry:
            raise AssertionError("Retained E2 geometry feature changed")
        for row in old["composition"]:
            elements_seen.add(row["element"])
        for row in packet["composition"]:
            for key in PROPERTY_FILES:
                if row[key] == "unknown":
                    missing[key] = missing.get(key, 0) + 1
                elif row[key] not in (1, 2, 3, 4, 5):
                    raise AssertionError("Non-binned property reached packet")
        fingerprints.append(hashlib.sha256(json.dumps(packet, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest())
    return {
        "n_cases": len(cases),
        "n_distinct_elements_in_read_only_inputs": len(elements_seen),
        "missing_properties_in_emitted_element_rows": missing,
        "all_materials_reproducible": True,
        "all_occupancies_and_fractions_unchanged": True,
        "structure_geometry_unchanged_except_removed_mass_fraction": True,
        "aggregate_packet_sha256": hashlib.sha256("\n".join(fingerprints).encode()).hexdigest(),
    }
