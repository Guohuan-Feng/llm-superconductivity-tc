# Data preparation

Scripts from the earlier pipeline that produced the files in [`data/development/`](../../data/development) and [`data/rule_banks/`](../../data/rule_banks). They are published as run, so paths are absolute paths on the original Windows machine.

| Step | Script | Output |
| --- | --- | --- |
| 1 | `run_pilot.py` | Reads 3DSC `3DSC_MP.csv`, assigns `row_id`, parses the CIFs and caches features |
| 2 | `run_rules.py` | Association groups and the development/retrospective-test split (seed 20260929) |
| 3 | `data_prepare.py` | Development tables and label-free structure descriptors |
| 4 | `structural_features.py`, `prepare_features_only.py` | X127 (`FEATURES_ONLY.npz`) |
| 5 | `prepare_root_inputs.py` | `accepted_train.npz`: X127 with labels, groups and the three folds |
| 6 | `nonlinear_classification.py` | HGB127 out-of-fold predictions |
| 7 | `root_export_common_cohort.py` | X969 common features |
| 8 | `train_rule_factory.py`, `actual_train_rulebank_adapter.py`, `numeric_rule_executor.py`, `audit_train_rulebank.py`, `root_build_and_audit.py` | Outer training-fold rule banks |
| 9 | `pure_scope_owner_adapter.py` | Rule selection and scoring used by the rule baselines |

## Limits

- Steps 1–3 need the 3DSC repository at commit `2471dd51a298a854cb4f365ebd39e72c7cbf3634` (https://github.com/aimat-lab/3DSC), including its CIF files.
- Step 5 reads a fold-assignment file that is no longer available; the folds it produced are published in `dev_set.csv` and `accepted_train.npz`.
- The inner rule banks were built with the same rule factory inside earlier campaign runs whose runners are not included; the banks themselves are published.
- Running any experiment only needs the published data files, not these scripts.
