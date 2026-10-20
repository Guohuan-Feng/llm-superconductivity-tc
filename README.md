# LLM versus machine learning for superconducting Tc classification

Code for a diagnostic study of whether a large language model (LLM) predicts if a material's superconducting critical temperature (Tc) exceeds 10 K better than conventional machine learning, and whether any advantage comes from reasoning or from prior literature knowledge.

[中文说明](README.zh-CN.md)

## Data

| Resource | Role in this study | Address | Licence |
| --- | --- | --- | --- |
| 3DSC (3DSC_MP) | Main dataset: superconductors paired with crystal structures. 3,764 3DSC_MP entries form the development set; the remaining entries were held out in an earlier split | Code and data: https://github.com/aimat-lab/3DSC · Data: https://doi.org/10.6084/m9.figshare.c.6914407.v1 · Paper: Sommer et al., *Sci. Data* 10, 816 (2023), https://doi.org/10.1038/s41597-023-02721-y | CC BY 4.0 (data), MIT (code) |
| SuperCon subset (Stanev et al.) | Source of the Tc labels inside 3DSC | Stanev et al., *npj Comput. Mater.* 4, 29 (2018), https://doi.org/10.1038/s41524-018-0085-8 | CC BY 4.0 |
| Materials Project | Source of the 3DSC_MP crystal structures | https://materialsproject.org | CC BY 4.0 |
| Newly reported superconductors (2024–2026) | Temporal extrapolation test (89 frozen candidates, 69 in the strict primary set) | Compiled from primary literature; one row per material with its DOI or arXiv source in [`data/new_materials_sources.csv`](data/new_materials_sources.csv) | See each source |
| Matbench `mp_e_form` | Formation-energy feasibility test for a verifier loop | https://figshare.com/articles/dataset/mp_e_form/9755432 | MIT |
| Development inputs (this repository) | The 3,764-material development set with labels, association groups and three folds; X127 and X969 features; HGB127 out-of-fold predictions; training-fold rule banks | [`data/development/`](data/development) · [`data/rule_banks/`](data/rule_banks) | CC BY 4.0 (derived from 3DSC) |

## Models and software

| Component | Use | Address |
| --- | --- | --- |
| gpt-6.1-sol via Codex CLI (`codex exec`) | LLM conditions A1, A2, A3, M1, M2 and the Tc-estimate prompt | https://github.com/openai/codex |
| CGCNN | Crystal-graph baseline | https://github.com/txie-93/cgcnn |
| CHGNet | Pretrained embeddings and formation-energy verifier | https://github.com/CederGroupHub/chgnet |
| matminer (Magpie features) | Composition-feature baseline | https://github.com/hackingmaterials/matminer |
| scikit-learn | HGB, random forest, logistic regression, k-nearest neighbours | https://scikit-learn.org |

## Repository layout

All scripts are in [`code/`](code). File-name prefixes follow the experiment stages:

| Prefix | Stage | Content |
| --- | --- | --- |
| `e0`–`e4`, `run_stage1.py`, `run_stage2.py` | E | Initial agent diagnosis: ML controls, rule-bank analysis, LLM conditions A1–A3, error analysis, interpretability |
| `f0`–`f5`, `run_f*.py` | F | Recall probe, identity masking (M1, M2), full-data evaluation, first new-material collection, summary |
| `g1`–`g7`, `g5b`, `g5c`, `g7b` | G | Structure and pretrained baselines, ranking and calibration, complementarity, input ablation, multi-threshold and Tc regression, repeatability and majority vote, early-cutoff local model, temporal extrapolation |
| `p1_*`, `run_p1.py` | P1 | Formation-energy rule-iteration feasibility test |
| `common.py`, `stats.py`, `*_stats.py` | Shared | Group-weighted metrics and group-bootstrap confidence intervals |
| `*.ps1` | Shared | Windows scheduled-task launchers for long runs |
| `*_instructions.md`, `*.schema.json`, `*_prompts.py` | Shared | LLM instructions, prompts and output schemas |
| `data_preparation/` | Inputs | Scripts that built the development inputs and rule banks |

Data are in [`data/`](data):

| Path | Content |
| --- | --- |
| [`data/development/`](data/development) | Development set: materials, labels, groups, folds, features and HGB127 out-of-fold predictions |
| [`data/rule_banks/`](data/rule_banks) | Rule banks mined from training folds (3 outer, 18 inner) |
| [`data/new_materials_sources.csv`](data/new_materials_sources.csv) | Sources of the 2024–2026 materials in the temporal test |
| [`data/SHA256SUMS.txt`](data/SHA256SUMS.txt) | Checksums of the data files |

## Reproducing

- All inputs the experiment scripts read are in [`data/`](data). Each data folder's README maps its files to the original paths in [`code/INPUTS.md`](code/INPUTS.md), and every development material links to its row in 3DSC.
- Paths in the scripts and in `INPUTS.md` are absolute paths on the original Windows machine and must be pointed at `data/`.
- Python 3.12 with NumPy and scikit-learn; PyTorch for CGCNN and CHGNet; a logged-in Codex CLI for LLM calls.
- Results are development results on one dataset unless stated otherwise.

## Licence

Data in [`data/development/`](data/development) and [`data/rule_banks/`](data/rule_banks) are derived from 3DSC and distributed under CC BY 4.0 with attribution to the sources above. No licence has been chosen for the code yet; all rights to the code are reserved until one is added.
