# 大语言模型与机器学习在超导 Tc 分类上的对比

本仓库是一项诊断研究的代码：比较大语言模型（LLM）与传统机器学习在判断材料超导临界温度（Tc）是否高于 10 K 上的表现，并分析 LLM 的优势来自推理还是已有文献知识。

[English](README.md)

## 数据

| 资源 | 在本研究中的用途 | 地址 | 许可 |
| --- | --- | --- | --- |
| 3DSC（3DSC_MP） | 主数据集：带晶体结构的超导体数据。本研究以 3DSC_MP 中的 3764 条作为开发集，其余条目在早期划分中留出 | 代码与数据：https://github.com/aimat-lab/3DSC · 数据：https://doi.org/10.6084/m9.figshare.c.6914407.v1 · 论文：Sommer 等，*Sci. Data* 10, 816 (2023)，https://doi.org/10.1038/s41597-023-02721-y | 数据 CC BY 4.0，代码 MIT |
| SuperCon 子集（Stanev 等整理） | 3DSC 中 Tc 标签的来源 | Stanev 等，*npj Comput. Mater.* 4, 29 (2018)，https://doi.org/10.1038/s41524-018-0085-8 | CC BY 4.0 |
| Materials Project | 3DSC_MP 晶体结构的来源 | https://materialsproject.org | CC BY 4.0 |
| 2024–2026 年新报道的超导体 | 时间外推检验（冻结候选 89 个，严格主集 69 个） | 从原始文献整理，每个材料的 DOI 或 arXiv 来源见 [`data/new_materials_sources.csv`](data/new_materials_sources.csv) | 以各来源为准 |
| Matbench `mp_e_form` | 验证器循环的形成能可行性试验 | https://figshare.com/articles/dataset/mp_e_form/9755432 | MIT |

## 模型与软件

| 组件 | 用途 | 地址 |
| --- | --- | --- |
| gpt-6.1-sol，经 Codex CLI（`codex exec`）调用 | LLM 实验条件 A1、A2、A3、M1、M2 及 Tc 估计提示词 | https://github.com/openai/codex |
| CGCNN | 晶体图神经网络基线 | https://github.com/txie-93/cgcnn |
| CHGNet | 预训练嵌入与形成能验证器 | https://github.com/CederGroupHub/chgnet |
| matminer（Magpie 特征） | 成分特征基线 | https://github.com/hackingmaterials/matminer |
| scikit-learn | HGB、随机森林、逻辑回归、k 近邻 | https://scikit-learn.org |

## 目录结构

全部脚本位于 [`code/`](code)，文件名前缀对应实验阶段：

| 前缀 | 阶段 | 内容 |
| --- | --- | --- |
| `e0`–`e4`、`run_stage1.py`、`run_stage2.py` | E | 初版 Agent 诊断：机器学习对照、规则库分析、LLM 条件 A1–A3、错误分析、可解释性 |
| `f0`–`f5`、`run_f*.py` | F | 回忆测试、身份消融（M1、M2）、全量评估、首轮新材料收集、汇总 |
| `g1`–`g7`、`g5b`、`g5c`、`g7b` | G | 结构与预训练基线、排序与校准、互补性、输入消融、多阈值与 Tc 回归、重复性与多数投票、早截止本地模型、时间外推 |
| `p1_*`、`run_p1.py` | P1 | 形成能规则迭代可行性试验 |
| `common.py`、`stats.py`、`*_stats.py` | 公共 | 组加权指标与按组 bootstrap 置信区间 |
| `*.ps1` | 公共 | 长时间运行使用的 Windows 计划任务启动脚本 |
| `*_instructions.md`、`*.schema.json`、`*_prompts.py` | 公共 | LLM 指令、提示词与输出格式 |

## 复现说明

- 开发集输入（`accepted_train.npz`，含 127 维成分与结构特征、标签、关联组与三折划分）以及每折的训练集规则库由早期流程生成，未包含在本仓库中。脚本读取的全部输入文件见 [`code/INPUTS.md`](code/INPUTS.md)。
- 脚本与 `INPUTS.md` 中的路径为原 Windows 机器上的绝对路径，复现时需要修改。
- 运行环境：Python 3.12、NumPy、scikit-learn；CGCNN 与 CHGNet 需要 PyTorch；LLM 调用需要已登录的 Codex CLI。
- 除特别说明外，结果均为同一数据集上的开发结果。

## 许可

尚未选择开源许可；在添加许可之前保留全部权利。
