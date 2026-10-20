# Development set

The 3,764 3DSC_MP materials that every experiment in [`code/`](../../code) uses: 990 positives (Tc > 10 K), 1,117 association groups, three group-wise folds.

## Files

| File | Content |
| --- | --- |
| `dev_set.csv` | One row per material: identifiers, link to the 3DSC source row, Tc, label, association group, group weight and fold |
| `features_x127.csv` | The 127 predictors (X127), in the column order all scripts use |
| `hgb127_oof.csv` | Out-of-fold probabilities and predictions of the HGB127 baseline |
| `x969_feature_names.csv` | Names and units of the 969 common features (X969) |
| `accepted_train.npz` | Original NumPy file read by the scripts: `X127`, `target_K`, `label`, `component_id`, `fold`, `material_id`, `source_index`, `role` |
| `FEATURES_ONLY.npz` | `X127` with `feature_names` |
| `oof_predictions_hgb127.npz` | Original HGB127 out-of-fold file; the comparator is `probability` |
| `COMMON_FACTS_FEATURES_ONLY.npz` | `X969`, `missing_mask` and `valid` for the strong ML ensemble |

All files are ordered by `material_id`. `source_index` in the NumPy files and rule banks is the zero-based row position in `dev_set.csv`.

## `dev_set.csv` columns

| Column | Meaning |
| --- | --- |
| `material_id` | `3dsc_mp_row_{row:06d}`, the identifier used in all result files |
| `row_index_3DSC_MP` | Zero-based row of `data/final/MP/3DSC_MP.csv` in 3DSC commit `2471dd51a298a854cb4f365ebd39e72c7cbf3634`, read with `pandas.read_csv(path, comment="#")`. Formula, Tc and Materials Project ID match the source on all 3,764 rows |
| `formula` | 3DSC `formula_sc` (SuperCon formula) |
| `chemical_system` | Elements joined by `-` |
| `mp_parent_id` | Materials Project ID of the matched structure (3DSC `material_id_2`) |
| `sc_class` | 3DSC superconductor class (Other 2,269, Cuprate 710, Ferrite 337, Heavy_fermion 264, Oxide 131, Chevrel 29, Carbon 20, mixed 4) |
| `cif_file` | 3DSC CIF file name under `data/final/MP/cifs/` |
| `tc_K` | Critical temperature in K from 3DSC |
| `label_tc_gt_10K` | 1 if `tc_K` > 10, else 0 |
| `group_id` | Association group: materials connected through a shared chemical system or a shared Materials Project parent structure (`code/data_preparation/run_rules.py`, seed 20260929) |
| `group_size` | Number of materials in the group within this set |
| `group_weight` | `1 / group_size`, the weight for group-weighted metrics |
| `fold` | Outer fold; every group lies in one fold |

| Fold | Materials | Positives | Groups |
| --- | --- | --- | --- |
| 0 | 1,476 | 440 | 373 |
| 1 | 1,140 | 207 | 372 |
| 2 | 1,148 | 343 | 372 |

## X127 features

- Columns 1–118, `occupancy_fraction_Z001` … `occupancy_fraction_Z118`: occupancy-weighted atomic fraction of each element, by atomic number.
- Columns 119–127, nine structure features from the 3DSC CIF with partial occupancy retained: volume per occupied atom (Å³); mean, standard deviation, minimum and maximum nearest-neighbour distance under periodic boundaries (Å); mean and standard deviation of coordination within 6 Å; mean and standard deviation of periodic edge distance (Å).

The structures are those 3DSC matched to each SuperCon entry; many are Materials Project parents with artificial doping, not the measured samples.

## Original locations

| Path in [`code/INPUTS.md`](../../code/INPUTS.md) | File here |
| --- | --- |
| `2026-10-09-structure-classification/inputs/accepted_train.npz` | `accepted_train.npz` |
| `2026-10-09-structure-classification/inputs/FEATURES_ONLY.npz` | `FEATURES_ONLY.npz` |
| `2026-10-10-structure-nonlinear-classification/run/oof_predictions.npz` | `oof_predictions_hgb127.npz` |
| `workflow-review-2026-10-10/common-facts-cohort-actual-001/COMMON_FACTS_FEATURES_ONLY.npz` | `COMMON_FACTS_FEATURES_ONLY.npz` |

Checksums are in [`../SHA256SUMS.txt`](../SHA256SUMS.txt). The scripts that produced these files are in [`code/data_preparation/`](../../code/data_preparation).

## Licence and attribution

Derived from 3DSC (Sommer et al., *Sci. Data* 10, 816 (2023), https://doi.org/10.1038/s41597-023-02721-y), whose Tc values come from the SuperCon subset of Stanev et al. (*npj Comput. Mater.* 4, 29 (2018)) and whose structures come from the Materials Project. Distributed under CC BY 4.0. Changes from the source: selection of 3,764 rows, added association groups and folds, and computed features and predictions.
