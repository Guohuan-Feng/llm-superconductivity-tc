# Training-fold rule banks

Threshold rules mined only from training folds of the [development set](../development). The rule baselines (R-info, R-all) and the LLM conditions A1 and A2 read these banks; no rules were mined from held-out folds.

[`banks.csv`](banks.csv) lists every bank with its training folds and number of rules.

| Level | Files | Use |
| --- | --- | --- |
| Outer | `outer/outer{0,1,2}_TRAIN_bank.json.gz`, trained on all folds except the one named | Rule cards for A1 and A2; rule baselines on the held-out fold |
| Inner | `inner/outer{k}_innerTRAIN{j}*.json.gz`, trained on one inner fold, some with one subfold removed | Nested rule selection in experiment E0 |

## Format

Each file is gzip-compressed JSON. Rules are at `finalized_bank.rule_envelopes` in outer banks and at `payload.finalized_bank.rule_envelopes` in inner banks. Each envelope has a `card` and `statistics`:

| Field | Meaning |
| --- | --- |
| `card.rule_id` | Stable rule identifier, shared with prediction and citation outputs |
| `card.applicability` | Conjunction of thresholds on X127 (`raw-X127-AND/v1`): `feature_index` is the zero-based column in X127 order (`features_x127.csv` without `material_id`) |
| `card.theoretical_basis.numeric_formula` | Smoothed, class-balanced, group-weighted score for Tc > 10 K among matched training materials |
| `card.training_group_support` | Matched training groups and materials |
| `card.counterexamples` | Matched training materials whose label disagrees with the rule's prediction, as `source_index` and `component_id` |
| `statistics.matched_class_row_counts` | Matched training materials, `[Tc ≤ 10 K, Tc > 10 K]` |
| `statistics.matched_class_ws_mass` | The same counts weighted by `1 / group size` |
| `statistics.support_source_indices` | `source_index` of every matched training material |

`source_index` is the zero-based row position in [`../development/dev_set.csv`](../development/dev_set.csv); `component_id` is its `group_id`.

## Original locations

Outer banks: `workflow-review-2026-10-10/actual-train-rulebank-run-001/fold{0,1,2}/ROOT_REVIEWABLE_BANK.json`. Inner banks: the 18 hash-named files under "E0 frozen inner scientific rule banks" in [`code/INPUTS.md`](../../code/INPUTS.md), renamed here to their bank names. Content is unchanged apart from compression; checksums are in [`../SHA256SUMS.txt`](../SHA256SUMS.txt).

Licence: CC BY 4.0, as derived data of 3DSC (see [`../development/README.md`](../development/README.md)).
