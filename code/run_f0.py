"""F0 reanalysis of immutable E2 predictions. No model fitting or calls."""
from __future__ import annotations

from common import ROOT, DATA, FIELDS, load_data, np, safe_path, write_json
from f_stats import summarize
from stats import summarize as previous_summarize
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ARMS = ("R-info", "5NN", "HGB127", "A1_forced", "A1_HGBfallback",
        "A2_forced", "A2_HGBfallback", "A3_forced", "A3_HGBfallback")
OUT = safe_path(ROOT / "F0_reanalysis")
N_BOOT, SEED = 5000, 20261016


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    expected = {"config.json", "predictions.csv", "llm_calls.jsonl", "metrics.json", "run.log", "README.md"}
    unexpected = {p.name for p in OUT.iterdir()} - expected
    if unexpected:
        raise ValueError(f"Unexpected files in new F0 directory: {unexpected}")
    # A fresh rerun rewrites complete artifacts; there is no row recovery.
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
    try:
        with safe_path(OUT / "run.log").open("w", encoding="utf-8", buffering=1) as log:
            def note(message):
                text = f"{datetime.now(timezone.utc).isoformat()} {message}"
                print(text, flush=True)
                log.write(text + "\n")
            note("F0 started: recompute from read-only E2 predictions; no LLM calls.")
            d = load_data()
            indices = np.asarray(json.loads((ROOT / "E2/config.json").read_text(encoding="utf-8"))["indices"], dtype=int)
            assert len(indices) == len(np.unique(indices)) == 300
            ids = [str(i) for i in d["id"][indices]]
            row_index = {key: j for j, key in enumerate(ids)}
            scores = {arm: np.full(300, np.nan) for arm in ARMS}
            preserved = {}
            with (ROOT / "E2/predictions.csv").open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    arm, key = row["arm"], row["id"]
                    assert arm in scores and key in row_index
                    assert (arm, key) not in preserved
                    j = row_index[key]
                    assert str(row["group"]) == str(d["group"][indices[j]])
                    assert int(row["fold"]) == int(d["fold"][indices[j]])
                    score = float(row["score"])
                    assert int(row["y_pred"]) == int(score >= 0.5)
                    scores[arm][j] = score
                    preserved[(arm, key)] = row
            assert len(preserved) == 2700 and all(np.isfinite(v).all() for v in scores.values())
            assert np.array_equal(scores["HGB127"], d["hgb"][indices])
            with safe_path(OUT / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                # Append/flush all nine already-frozen arm predictions per ten materials.
                for start in range(0, 300, 10):
                    writer.writerows(preserved[(arm, key)] for key in ids[start:start + 10] for arm in ARMS)
                    handle.flush()
                    os.fsync(handle.fileno())
            safe_path(OUT / "llm_calls.jsonl").write_text("", encoding="utf-8")
            groups, y = d["group"][indices], d["y"][indices]
            full_sizes = {str(g): int(s) for g, s in zip(d["group"], d["group_size"])}
            masks = {"all": np.ones(300, dtype=bool), "exclude_93": groups != "93", "only_93": groups == "93"}
            config = {"experiment": "F0_reanalysis", "stage": 1,
                "scope": "adaptive development reanalysis; not an external test",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "arms": list(ARMS), "subsets": list(masks), "bootstrap_draws": N_BOOT, "bootstrap_seed": SEED,
                "threshold": 0.5, "n": 300, "original_dataset_n": 3764,
                "source_hashes": {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha(p)
                    for p in [DATA, ROOT / "E2/config.json", ROOT / "E2/predictions.csv", ROOT / "E2/metrics.json",
                              ROOT / "common.py", ROOT / "f_stats.py", ROOT / "run_f0.py"]},
                "new_LLM_calls": 0, "prediction_flush_every_materials": 10,
                "weighting": "group-weighted: 1/full original component size; unweighted: 1 per material",
                "resampling": "same 5000 group draws for every arm, both weighting schemes and all paired differences within each subset",
                "one_group_policy": "degenerate CI retained and explicitly flagged; it gives no between-group uncertainty",
                "interruption_policy": "recompute from immutable old predictions; no old-row recovery; no task needed for short no-call reanalysis"}
            write_json(OUT / "config.json", config)
            subsets = {}
            for name, mask in masks.items():
                subsets[name] = summarize(y, scores, groups, full_sizes, mask=mask, n_boot=N_BOOT, seed=SEED)
                note(f"subset={name} n={subsets[name]['n']} groups={subsets[name]['n_groups']}")
            # Check old weighted statistics against the prior independently stored evaluator.
            old_recomputed = previous_summarize(y, scores, groups, full_sizes, n_boot=N_BOOT, seed=SEED)
            max_difference = 0.0
            for arm in ARMS:
                for metric in ("balanced_accuracy", "precision", "recall", "accuracy"):
                    old = old_recomputed["metrics"][arm][metric]
                    new = subsets["all"]["group_weighted"]["metrics"][arm][metric]
                    max_difference = max(max_difference, abs(old["point"] - new["point"]),
                        *[abs(a - b) for a, b in zip(old["ci95"], new["ci95"])])
            assert max_difference < 1e-12
            saved = json.loads((ROOT / "E2/metrics.json").read_text(encoding="utf-8"))
            saved_difference = 0.0
            for arm in ARMS:
                if arm in ("R-info", "5NN", "HGB127"):
                    old = saved["baselines"]["metrics"][arm]
                else:
                    base = arm[:2]
                    key = "forced_valid_cases" if arm.endswith("_forced") else "abstention_or_failure_HGB_fallback_all_cases"
                    old = saved["arms"][base][key]["metrics"][arm]
                for metric in ("balanced_accuracy", "precision", "recall", "accuracy"):
                    new = subsets["all"]["group_weighted"]["metrics"][arm][metric]
                    saved_difference = max(saved_difference, abs(old[metric]["point"] - new["point"]),
                        *[abs(a - b) for a, b in zip(old[metric]["ci95"], new["ci95"])])
            assert saved_difference < 1e-12
            g93 = masks["only_93"]
            checks = {"group93_n": int(g93.sum()), "group93_positive": int(y[g93].sum()),
                "group93_tc_min": float(d["tc"][indices][g93].min()),
                "group93_tc_max": float(d["tc"][indices][g93].max()),
                "group93_negative_controls_all_negative": {arm: bool((scores[arm][g93] < .5).all()) for arm in ("HGB127", "5NN", "R-info")},
                "group93_errors": {arm: int(((scores[arm][g93] >= .5) != y[g93]).sum()) for arm in ARMS},
                "all300_errors": {arm: int(((scores[arm] >= .5) != y).sum()) for arm in ARMS},
                "old_evaluator_max_abs_difference": max_difference,
                "saved_E2_max_abs_difference": saved_difference}
            assert checks["group93_n"] == 114 and checks["group93_positive"] == 74
            result = {"experiment": "F0_reanalysis", "stage": 1,
                "scope": config["scope"], "subsets": subsets, "reproduction_checks": checks,
                "limitations": ["The fixed 300-material cohort was adaptively selected in earlier work, not an external test.",
                    "Removing component 93 changes material and group composition; the change is descriptive, not a causal estimate of memorization.",
                    "Component 93 alone has one group, so bootstrap CIs degenerate and do not support cross-group generalization.",
                    "Group-weighted row weights use all 3764 rows' group sizes and are never recomputed inside subsets.",
                    "All E2 arms are copied verbatim; no new prompts, fits, or material data changes."],
                "completed_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": time.perf_counter() - started}
            write_json(OUT / "metrics.json", result)
            lines = ["# F0 开发集复算", "固定300例；复用E2全部9臂，无新LLM调用、拟合或阈值调整。",
                "完整结果表：metrics.json → subsets → all / exclude_93 / only_93 → group_weighted / unweighted。",
                "各方法同时报告BA、精确率、召回率、准确率及95%CI；全部36对方法差值也报告同样四指标。",
                "不加权也按组bootstrap；5000次；同一子集的所有方法、权重方案和配对差共享抽样。",
                "组加权保留完整3764例中的原始组大小倒数；精确率无预测正类时取0。",
                "BA无两类或召回率无真阳性时未定义；每个指标记录有效/未定义bootstrap次数。",
                "| 子集 | n | 组数 | 正类 |", "|---|---:|---:|---:|"]
            lines.extend(f"| {name} | {obj['n']} | {obj['n_groups']} | {obj['n_positive']} |" for name, obj in subsets.items())
            lines.extend(["仅93有一个组，CI必然退化，不能解释为高确定性或跨组泛化。",
                "复现检查见metrics.json/reproduction_checks；旧E2组加权点估计和区间误差<1e-12。",
                "所有结果为开发结果；删除93的变化不能直接归因于记忆或推理。",
                "predictions.csv每10个材料追加并刷盘；llm_calls.jsonl为空。"])
            assert len(lines) <= 20
            safe_path(OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
            note(f"F0 complete: all six artifacts present; saved E2 max difference={saved_difference:.3g}; elapsed={time.perf_counter() - started:.2f}s")
        assert {p.name for p in OUT.iterdir()} == expected
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
