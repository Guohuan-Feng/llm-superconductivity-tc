"""Prepare a retrospective 3DSC-MP Tc cohort with label-free geometry.

This reuses historical strict splits. It is not a new independent holdout.
No model calls, descriptor search, or test-performance computations occur here.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import warnings

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from pymatgen.core import Element, Structure
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
SOURCE_ROOT = Path("C:/Users/28908/Documents/Codex/2026-09-14/x20-hi-guohuan-this-paper-really")
SOURCE = SOURCE_ROOT / "outputs/pris_crosspilot_2026-09-29/superconductivity"
SPLIT = SOURCE_ROOT / "outputs/pris_crosspilot_2026-09-29/sc_rules/results/seed_20260929/split_assignments.csv"
ROLE_MAP = {"inner_train": "train", "inner_validation": "validation", "test": "retrospective_test"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def nullable(value):
    return float(value) if value is not None and np.isfinite(value) else None


def weighted_property(values, normalized_weights):
    if any(value is None for value in values):
        return None, None
    values = np.asarray(values, dtype=float)
    mean = float(np.sum(values * normalized_weights))
    std = float(np.sqrt(np.sum(normalized_weights * (values - mean) ** 2)))
    return mean, std


def make_record(task):
    mid, cif_name, expected_hash = task
    path = SOURCE / "data/cifs" / cif_name
    assert path.resolve().parent == (SOURCE / "data/cifs").resolve()
    actual_hash = sha(path)
    if actual_hash != expected_hash:
        raise RuntimeError(f"CIF hash mismatch: {cif_name}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        structure = Structure.from_file(path, primitive=False)
        n = len(structure)
        assert n > 0
        symbols, atomic_numbers, electronegativities, radii = [], [], [], []
        site_occupancy, z_stds, x_stds, radius_stds = [], [], [], []
        species_site, species_symbols, species_z, species_x, species_radius, species_occupancy = [], [], [], [], [], []
        disordered_sites = 0
        missing_x_species = 0
        missing_radius_species = 0
        for i, site in enumerate(structure):
            entries = []
            for specie, occupancy in site.species.items():
                element = Element(specie.symbol)
                occupancy = float(occupancy)
                assert occupancy > 0
                x = nullable(element.X)
                radius = nullable(CovalentRadius.radius.get(element.symbol))
                entries.append((int(element.Z), element.symbol, occupancy, x, radius))
            entries.sort(key=lambda entry: entry[0])
            total = float(sum(entry[2] for entry in entries))
            assert 0 < total <= 1 + 1e-6
            normalized = np.asarray([entry[2] for entry in entries]) / total
            z_mean, z_std = weighted_property([entry[0] for entry in entries], normalized)
            x_mean, x_std = weighted_property([entry[3] for entry in entries], normalized)
            r_mean, r_std = weighted_property([entry[4] for entry in entries], normalized)
            # A mixed site is explicitly mixed. No dominant-species replacement.
            symbols.append("+".join(entry[1] for entry in entries))
            atomic_numbers.append(z_mean)
            electronegativities.append(x_mean)
            radii.append(r_mean)
            site_occupancy.append(total)
            z_stds.append(z_std)
            x_stds.append(x_std)
            radius_stds.append(r_std)
            disordered_sites += int(not site.is_ordered)
            for z, symbol, occupancy, x, radius in entries:
                species_site.append(i)
                species_symbols.append(symbol)
                species_z.append(z)
                species_x.append(x)
                species_radius.append(radius)
                species_occupancy.append(occupancy)
                missing_x_species += int(x is None)
                missing_radius_species += int(radius is None)

        matrix = np.asarray(structure.distance_matrix, dtype=float)
        _, distances_self, _, _ = structure.lattice.get_points_in_sphere(
            [[0, 0, 0]], [0, 0, 0], min(structure.lattice.abc) + 1e-7, zip_results=False)
        distances_self = np.asarray(distances_self)
        self_distance = float(np.min(distances_self[distances_self > 1e-8]))
        np.fill_diagonal(matrix, self_distance)
        center, other, images, distances = structure.get_neighbor_list(6.0, numerical_tol=1e-8, exclude_self=True)
        order = np.lexsort((images[:, 2], images[:, 1], images[:, 0], other, distances, center))
        center, other, images, distances = [np.asarray(value)[order] for value in (center, other, images, distances)]
        assert matrix.shape == (n, n) and np.isfinite(matrix).all() and (matrix > 0).all()
        assert np.allclose(matrix, matrix.T, atol=1e-8)
        assert np.isfinite(distances).all() and (distances > 0).all() and (distances <= 6.0 + 1e-8).all()
        assert not np.any((center == other) & np.all(images == 0, axis=1))
        cartesian = (structure.frac_coords[other] + images - structure.frac_coords[center]) @ structure.lattice.matrix
        assert np.allclose(np.linalg.norm(cartesian, axis=1), distances, atol=1e-7, rtol=1e-8)

    record = {
        "material_id": mid, "symbols": symbols,
        "frac_coords": structure.frac_coords.tolist(), "lattice": structure.lattice.matrix.tolist(),
        "distance_matrix": matrix.tolist(), "atomic_numbers": atomic_numbers,
        "electronegativities": electronegativities, "covalent_radii": radii,
        "neighbor_center": center.astype(int).tolist(), "neighbor_index": other.astype(int).tolist(),
        "neighbor_distance": distances.tolist(), "neighbor_image": images.astype(int).tolist(),
        "n_sites": n, "volume": float(structure.volume),
        "site_occupancy": site_occupancy, "site_Z_std": z_stds, "site_X_std": x_stds,
        "site_radius_std": radius_stds, "species_site": species_site, "species_symbols": species_symbols,
        "species_Z": species_z, "species_X": species_x, "species_radius": species_radius,
        "species_occupancy": species_occupancy,
    }
    assert all(len(record[key]) == n for key in ("symbols", "atomic_numbers", "electronegativities", "covalent_radii", "site_occupancy", "site_Z_std", "site_X_std", "site_radius_std"))
    assert len({len(record[key]) for key in ("species_site", "species_symbols", "species_Z", "species_X", "species_radius", "species_occupancy")}) == 1
    nearest = matrix.min(axis=1)
    checks = {
        "n_sites": n, "occupancy_sum": float(sum(site_occupancy)), "volume": float(structure.volume),
        "nearest_min": float(nearest.min()), "nearest_mean": float(nearest.mean()),
        "nearest_std": float(nearest.std()), "nearest_max": float(nearest.max()),
        "disordered_sites": disordered_sites, "partial_occupancy_sites": int(np.count_nonzero(np.asarray(site_occupancy) < 1 - 1e-8)),
        "missing_X_species": missing_x_species, "missing_radius_species": missing_radius_species,
        "neighbor_edges": len(distances), "cif_sha256": actual_hash,
    }
    return record, checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "dataset_manifest.json").exists():
        raise RuntimeError("Frozen dataset exists; refusing to overwrite it")

    source_csv = SOURCE / "data/3DSC_MP.csv"
    source = pd.read_csv(source_csv, comment="#")
    cached = pd.read_csv(SOURCE / "results/features.csv")
    old_split = pd.read_csv(SPLIT)
    source_schema = json.loads((SOURCE / "results/feature_schema.json").read_text(encoding="utf-8"))
    upstream_manifest = json.loads((SOURCE / "data/source_manifest.json").read_text(encoding="utf-8"))
    cif_manifest = pd.read_csv(SOURCE / "data/cif_manifest.csv")
    raw_features = source_schema["composition"]
    structure_features = source_schema["structure"]
    assert len(raw_features) == 109 and len(structure_features) == 28
    assert len(set(raw_features + structure_features)) == 137
    assert len(source) == len(cached) == len(old_split) == 5773
    assert cached.row_id.is_unique and old_split.row_id.is_unique
    assert cached.row_id.tolist() == list(range(len(source)))
    assert sha(source_csv) == upstream_manifest["csv_sha256"]
    assert cached.structure_parse_ok.astype(bool).all()
    assert np.isfinite(source.tc.to_numpy(float)).all() and (source.tc >= 0).all()
    assert np.array_equal(source.tc.to_numpy(float), cached.tc.to_numpy(float))
    assert source.formula_sc.tolist() == cached.formula_sc.tolist()
    assert source.material_id_2.tolist() == cached.material_id_2.tolist()
    assert set(old_split.role) == set(ROLE_MAP)
    audit_join = cached[["row_id", "chemical_system", "material_id_2", "canonical_composition"]].merge(
        old_split, on="row_id", suffixes=("_cached", "_split"), validate="one_to_one")
    for key in ("chemical_system", "material_id_2", "canonical_composition"):
        assert audit_join[f"{key}_cached"].tolist() == audit_join[f"{key}_split"].tolist()

    cached = cached.merge(old_split[["row_id", "component", "role"]], on="row_id", validate="one_to_one")
    cached["material_id"] = cached.row_id.map(lambda value: f"3dsc_mp_row_{int(value):06d}")
    cached["split"] = cached.role.map(ROLE_MAP)
    cached["group"] = cached.component.astype(int)
    cached["weight"] = 1.0 / cached.groupby("canonical_composition").canonical_composition.transform("size").to_numpy()
    cached["parent_id"] = cached.material_id_2
    cached["formula"] = cached.formula_sc
    cached["class"] = cached.sc_class
    assert cached.material_id.is_unique
    assert np.isfinite(cached.weight).all() and (cached.weight > 0).all()

    overlap = {}
    for left, right in (("train", "validation"), ("train", "retrospective_test"), ("validation", "retrospective_test")):
        a, b = cached[cached.split.eq(left)], cached[cached.split.eq(right)]
        overlap[f"{left}__{right}"] = {
            key: len(set(a[key].dropna()) & set(b[key].dropna()))
            for key in ("material_id", "group", "chemical_system", "parent_id", "canonical_composition")
        }
    assert all(count == 0 for pair in overlap.values() for count in pair.values())
    # These files deliberately separate predictors from target-bearing metadata.
    cached[["material_id", *raw_features, *structure_features]].to_csv(OUT / "features.csv.gz", index=False)
    label_columns = ["material_id", "row_id", "split", "tc", "weight", "group", "chemical_system", "parent_id", "formula", "class", "canonical_composition", "synth_doped", "upstream_weight", "cif"]
    cached.loc[cached.split.ne("retrospective_test"), label_columns].to_csv(OUT / "development.csv.gz", index=False)
    cached.loc[cached.split.eq("retrospective_test"), label_columns].to_csv(OUT / "retrospective_test.csv.gz", index=False)
    assignment_columns = ["material_id", "row_id", "split", "role", "group", "chemical_system", "parent_id", "canonical_composition", "formula", "cif"]
    cached[assignment_columns].to_csv(OUT / "split_assignments.csv", index=False)

    cif_hashes = dict(zip(cif_manifest.cif, cif_manifest.sha256))
    assert cif_manifest.cif.is_unique
    assert set(cached.cif) == set(cif_hashes)
    tasks = [(row.material_id, row.cif, cif_hashes[row.cif]) for row in cached.itertuples()]
    totals = {key: 0 for key in ("rows", "neighbor_edges", "disordered_sites", "partial_occupancy_sites", "missing_X_species", "missing_radius_species")}
    descriptor_counts = {"development": 0, "retrospective_test": 0}
    maximum_geometry_difference = 0.0
    with gzip.open(OUT / "descriptor_development.jsonl.gz", "wt", encoding="utf-8") as development, gzip.open(OUT / "descriptor_retrospective_test.jsonl.gz", "wt", encoding="utf-8") as retrospective:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for row, (record, checks) in zip(cached.itertuples(), pool.map(make_record, tasks, chunksize=20)):
                assert record["material_id"] == row.material_id
                assert checks["n_sites"] == row.struct_n_sites
                comparisons = ((checks["occupancy_sum"], row.struct_n_atoms_occupancy),
                    (checks["volume"] / checks["occupancy_sum"], row.struct_volume_per_atom),
                    (checks["nearest_min"], row.struct_nearest_distance_min),
                    (checks["nearest_mean"], row.struct_nearest_distance_mean),
                    (checks["nearest_std"], row.struct_nearest_distance_std),
                    (checks["nearest_max"], row.struct_nearest_distance_max))
                for actual, expected in comparisons:
                    assert np.isclose(actual, expected, atol=1e-7, rtol=1e-8), (row.material_id, actual, expected)
                    maximum_geometry_difference = max(maximum_geometry_difference, abs(actual - expected))
                role = "retrospective_test" if row.split == "retrospective_test" else "development"
                target_stream = retrospective if role == "retrospective_test" else development
                target_stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
                descriptor_counts[role] += 1
                totals["rows"] += 1
                for key in totals:
                    if key != "rows":
                        totals[key] += checks[key]
                if totals["rows"] % 500 == 0:
                    print(json.dumps({"prepared_structures": totals["rows"], "requested": len(cached)}), flush=True)

    assert totals["rows"] == len(cached) and sum(descriptor_counts.values()) == len(cached)
    missing = {key: int(cached[key].isna().sum()) for key in raw_features + structure_features if cached[key].isna().any()}
    dump("feature_schema.json", {
        "id": "material_id", "group": "group", "split_column": "split", "target": "tc", "target_unit": "K",
        "raw_features": raw_features, "structure_features": structure_features,
        "raw_feature_count": 109, "structure_feature_count": 28,
        "features": raw_features, "feature_file": "features.csv.gz",
        "development_labels": "development.csv.gz", "retrospective_test_labels": "retrospective_test.csv.gz",
        "missing_policy": "Retain all source rows and original NaNs; fitting must impute on training data only. No target-dependent filtering.",
        "excluded_predictors": ["tc", "class", "formula", "chemical_system", "parent_id", "synth_doped", "weight", "upstream_weight", "canonical_composition", "row_id", "material_id", "cif", "split", "group"],
        "structure_feature_note": source_schema["structure_symmetry_note"],
        "composition_note": "Original 109 numerical composition features are copied without recomputation or changes.",
    })
    fields = {
        "material_id": "Unique row identity; original MP parent IDs are not unique in 3DSC.",
        "symbols": "Length N. Pure-element sites use that symbol; mixed sites join actual species symbols ordered by Z with '+', e.g. La+Sr. A mixed label is not an element and does not assign a dominant species. Use species_* for exact chemical selection.",
        "frac_coords": "N x 3 fractional coordinates in original parsed CIF site order; Cartesian = frac @ lattice.",
        "lattice": "3 x 3 row lattice vectors, angstrom.",
        "distance_matrix": "N x N nearest periodic site-pair distances, angstrom. Diagonal is shortest NONZERO lattice translation. Does not retain neighbor image multiplicities.",
        "atomic_numbers": "Length N; occupancy-normalized mean Z over actual site species, potentially noninteger.",
        "electronegativities": "Length N; occupancy-normalized Pauling X. Null if any occupied species on the site has unknown X; no silent renormalization over a known subset.",
        "covalent_radii": "Length N; occupancy-normalized CovalentRadius.radius mean, angstrom. Null if any occupied species has unknown radius.",
        "site_occupancy": "Length N; sum of absolute species occupancies, may be below one; not normalized to one.",
        "site_Z_std": "Length N; occupancy-normalized within-site population std of Z.",
        "site_X_std": "Length N; occupancy-normalized within-site population std of Pauling X; null if any species X is unknown.",
        "site_radius_std": "Length N; occupancy-normalized within-site population std of covalent radius; null if any species radius is unknown.",
        "species_site": "Flattened actual site-species table; zero-based site index for every occupied species, same length as all species_* fields.",
        "species_symbols": "Flattened actual element symbols, ordered by site then Z.",
        "species_Z": "Flattened actual integer atomic numbers.",
        "species_X": "Flattened actual Pauling electronegativities; null means unknown.",
        "species_radius": "Flattened actual covalent radii, angstrom; null means unknown.",
        "species_occupancy": "Flattened absolute species occupancies, not normalized. Sum within a site equals site_occupancy.",
        "neighbor_center": "Central site i for every directed periodic site-neighbor edge within 6 angstrom.",
        "neighbor_index": "Neighbor site j in original cell for every directed edge. All image multiplicities retained.",
        "neighbor_image": "Integer 3-vector image of site j. Displacement = (frac[j] + image - frac[i]) @ lattice.",
        "neighbor_distance": "Edge distance in angstrom. Nonzero self-images retained; zero-image self edge excluded.",
        "n_sites": "Number of original geometrical sites, not sum of occupancies or count of expanded species.",
        "volume": "Original parsed CIF cell volume, angstrom^3.",
    }
    dump("descriptor_schema.json", {
        "record_id": "material_id", "fields": fields, "source_has_targets": False,
        "development_file": "descriptor_development.jsonl.gz", "retrospective_test_file": "descriptor_retrospective_test.jsonl.gz",
        "cell": "Original final 3DSC-MP CIF parsed with primitive=False. Partial occupancy and synthetic doping are retained; no ordering, primitive reduction, idealization, or relaxation is performed.",
        "neighbor_cutoff_A": 6.0, "neighbor_numerical_tolerance_A": 1e-8,
        "neighbor_limitation": "Finite 6 angstrom geometry; directed image multiplicity retained. Species coexist as occupancy mixtures at each geometrical site; occupancy weighting must be handled explicitly in chemistry-sensitive descriptors.",
        "label_policy": "Tc, class, energies, parent ID, formula and targets are absent from payloads. Retrospective test descriptors must not be used for adaptive selection.",
    })
    split_counts = {role: {"rows": int(part.shape[0]), "groups": int(part.group.nunique()), "chemical_systems": int(part.chemical_system.nunique()), "parents": int(part.parent_id.nunique())} for role, part in cached.groupby("split")}
    audit = {
        "rows": len(cached), "raw_features": 109, "structure_features": 28,
        "unique_material_ids": int(cached.material_id.nunique()), "split_counts": split_counts,
        "overlap": overlap, "descriptor_counts": descriptor_counts, "structural_totals": totals,
        "all_source_CIF_hashes_verified": True, "all_payload_geometry_compared_to_cached_structure_features": True,
        "max_abs_geometry_difference": maximum_geometry_difference,
        "feature_missing_counts": missing, "synth_doped_rows": int(cached.synth_doped.astype(bool).sum()),
        "disordered_structure_rows": int(cached.struct_is_ordered.eq(0).sum()),
        "source_target_compatibility_checks": len(cached),
        "target_summaries": "None computed or printed. Retrospective target file was written without reopening it.",
        "elapsed_seconds": time.perf_counter() - started,
    }
    dump("preparation_audit.json", audit)
    sources = [source_csv, SOURCE / "data/source_manifest.json", SOURCE / "data/cif_manifest.csv", SOURCE / "results/features.csv", SOURCE / "results/feature_schema.json", SOURCE / "run_pilot.py", SPLIT, SPLIT.parents[2] / "run_rules.py"]
    limitations = [
        "Every old 3DSC cohort and previous split has already been analyzed, including these historical test records. Results on retrospective_test are diagnostic reuse, not new independent evidence or a sealed external validation.",
        "Changing a random seed on this dataset does not create a new independent cohort. A new external or genuinely unexamined prospective cohort is required for confirmatory claims.",
        "3DSC links reported experimental Tc values to matched Materials Project geometries; geometry and Tc are not jointly measured on exactly the same physical specimen and conditions.",
        "Artificial/synthetic doping changes species and occupancies on matched parent geometry; it does not constitute a relaxed or measured structure at the reported doping. Partial occupancy and disorder are retained, not ordered away.",
        "Strict connected components join rows sharing a chemical system or MP parent; the split prevents these known links crossing roles but does not establish the absence of all physical, literature, or pretrained-model overlap.",
        "Tc measurement conditions, pressure, phase purity and other experimental context are not modeled here. Predictive improvements do not establish a superconductivity mechanism.",
        "Pauling electronegativity may be missing. Payload fields preserve nulls and require explicit treatment. Radius source is covalent radii, distinct from original composition atomic_radius predictors.",
        "All original 109 composition features and 28 structure features are retained. Class, parent identity, synthetic doping flags, weighting fields and other metadata are excluded as predictors.",
        "Complete periodic neighbor image multiplicities are supplied only within 6 angstrom; the nearest-distance matrix alone cannot define coordination or long-range electronic structure.",
    ]
    manifest = {
        "dataset": "3DSC-MP retrospective scientific-agent Tc pilot", "target": "tc", "target_unit": "K",
        "source_repository": upstream_manifest["repository"], "source_commit": upstream_manifest["commit"], "source_data_license": upstream_manifest["data_license"],
        "split_protocol": "Reuse strict chemical_system_parent_components; connected chemical-system and original-MP-parent grouping; seed 20260929. No new splitting or sampling.",
        "role_mapping": ROLE_MAP, "test_status": "historically analyzed retrospective diagnostic cohort; not new independent evidence",
        "weight_policy": "1 / number of matched rows per canonical_composition across the original full source cohort, matching historical strict sc_rules evaluation. upstream_weight retained separately.",
        "label_and_predictor_separation": "features.csv.gz is numerical predictors plus row ID; development/test files contain labels and metadata; structure JSONL has geometry/species only.",
        "material_id_policy": "3dsc_mp_row_{row_id:06d}; unique source-row identity. Parent MP ID is parent_id metadata and grouping input, not the unique row ID or a predictor.",
        "audit": audit, "limitations": limitations,
        "source_hashes": {str(path.relative_to(SOURCE_ROOT)).replace('\\', '/'): sha(path) for path in sources},
        "verified_CIF_sha256": cif_hashes,
        "preparation_sha256": sha(Path(__file__)),
        "package_versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "pymatgen", "scikit-learn")},
        "files": {path.name: {"sha256": sha(path), "bytes": path.stat().st_size} for path in sorted(OUT.iterdir()) if path.is_file()},
    }
    dump("dataset_manifest.json", manifest)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
