"""Finite Stage 1 formula-only recall test; F2 and later are not authorized."""
from __future__ import annotations

from common import ROOT, DATA, FIELDS, safe_path, write_json, load_data, np
from f1_inputs import build_f1_inputs, formula_only_input, sha256_file
from f1_transport import call_formula, resolve_runtime
from f_stats import summarize, fraction_summary
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time

OUT = ROOT / "F1_recall"
SCHEMA, INSTRUCTIONS, CATALOG = [ROOT / s for s in (
    "f1_response.schema.json", "f1_instructions.md", "e2_model_catalog.json")]
SOURCE_NAMES = ("run_f1.py", "f1_inputs.py", "f1_transport.py", "f_stats.py",
                "f1_response.schema.json", "f1_instructions.md", "e2_transport.py",
                "e2_model_catalog.json", "common.py")
PROMPT_PREFIX = """请只凭已有材料知识回忆下面这个精确配比，不要根据化学规律推算 Tc，也不要查询信息。
recognized：yes=认识这个具体配比并回忆其报道；partial=只认识材料家族或母体，不能确认这个精确配比；no=不认识。
family：材料家族，最多20字；reported_Tc_K：仅给你能回忆的具体配比的实验 Tc，不能回忆则 null，不要用母体或家族的 Tc 代替；明确记得不超导可填0。
reported_superconductor：是否记得该具体材料被报道为超导，yes/no/unknown；这里不等同于 Tc>10K。
confidence：对这份回忆的信心，0到1；reason：最多60字，说明回忆依据或不确定性。
只输出符合给定 schema 的 JSON。科学输入只有以下一个 proxy cell formula 字符串：
"""
EXPECTED = {"config.json", "predictions.csv", "llm_calls.jsonl", "metrics.json", "run.log", "README.md"}


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_prompt(case):
    payload = formula_only_input(case)
    assert set(payload) == {"formula"}
    return PROMPT_PREFIX + json.dumps(payload["formula"], ensure_ascii=False)


def validate(record):
    if record.get("status") != "ok":
        return None
    try:
        obj = json.loads(record["response_text"])
        assert isinstance(obj, dict) and set(obj) == {
            "recognized", "family", "reported_Tc_K", "reported_superconductor", "confidence", "reason"}
        assert obj["recognized"] in ("yes", "partial", "no")
        assert obj["reported_superconductor"] in ("yes", "no", "unknown")
        assert isinstance(obj["family"], str) and len(obj["family"]) <= 20
        assert isinstance(obj["reason"], str) and len(obj["reason"]) <= 60
        for key in ("confidence", "reported_Tc_K"):
            x = obj[key]
            if key == "reported_Tc_K" and x is None:
                continue
            assert type(x) in (int, float) and math.isfinite(x) and x >= 0
        assert obj["confidence"] <= 1
        return obj
    except (AssertionError, ValueError, TypeError, KeyError):
        record["status"] = "schema_invalid"
        record["error"] = "Response does not satisfy the frozen F1 field contract"
        return None


def write_readme(status, n_valid=0):
    lines = ["# F1 formula-only recall development test", f"Status: {status}; valid responses {n_valid}/320.",
             "300 fixed E2 materials /81 original groups plus20 random composition controls.",
             "Formula strings exactly match E2 A3; all other scientific inputs excluded.",
             "Model gpt-6.1-sol/high; same E2 CLI version; tools disabled; no professor API.",
             "One fresh generation/item;10 concurrent; no scientific or format retry within a run.",
             "An interrupted process may be restarted once after system/task diagnosis; then recompute all items.",
             "Usage/rate limit: save partial work, stop all new dispatch, no automatic restart.",
             "R-recall positive iff recalled numeric Tc>10K; null counts negative and numeric coverage is separate.",
             "recognized=yes and yes+partial rates and recall correctness are reported separately.",
             "Four classifier metrics, both weightings, all/exclude93/only93, and paired differences are in metrics.json.",
             "Group weights retain 1/full3764 original group size;5000 paired group bootstrap draws.",
             "Only93 has one group: intervals are degenerate, not evidence of cross-group precision.",
             "Random controls are absent from3764, not proven absent from literature/pretraining.",
             "Self-reported recognition and accurate Tc do not establish a causal memory contribution.",
             "F2 has not started; await human continue after Stage1 results."]
    safe_path(OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare(task_name):
    if OUT.exists():
        raise RuntimeError("F1 already exists; do not overwrite without checking state")
    inputs = build_f1_inputs()
    runtime = resolve_runtime()
    prompts = [make_prompt(case) for case in inputs["cases"]]
    from e2_transport import input_budget
    bounds = [input_budget(text, schema_path=SCHEMA, instructions_path=INSTRUCTIONS) for text in prompts]
    assert len(prompts) == 320 and all(x["within_budget"] for x in bounds)
    for ancestor in (ROOT, *ROOT.parents):
        assert not ancestor.is_symlink() and not ancestor.is_junction()
    safe_path(OUT).mkdir()
    cfg = {"experiment": "F1_recall", "stage": 1, "scope": "adaptive development diagnostic; F2 not authorized",
           "created_utc": utcnow(), "task_name": task_name, "runtime": runtime,
           "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
           "model_version_note": "Same explicit model slug/effort and CLI version as E2; immutable backend weight version unavailable.",
           "params": {"concurrency": 10, "n_real": 300, "n_controls": 20,
                      "calls_per_run": 320, "max_process_restarts": 1, "max_calls_across_runs": 640,
                      "attempts_per_item_per_run": 1, "bootstrap_draws": 5000, "bootstrap_seed": 20261016,
                      "execution_time_limit": "none", "prediction_flush_every_cases": 10},
           "recognition_definition": {"strict": "yes only", "broad": "yes or partial", "three_strata": ["yes", "partial", "no"]},
           "R_recall_policy": "numeric recalled Tc>10K =>1; all other values including null=>0; numeric coverage separately reported",
           "recall_correctness_policy": "For recognized=yes, and separately yes+partial: null is a failure in the primary all-recognized denominator; numeric-only sensitivity also reported.",
           "group_weighting": "1/full accepted3764 original group size, retained after subset/recognition masks; matches E2",
           "prompt_prefix": PROMPT_PREFIX, "input_metadata": inputs["metadata"], "cases": inputs["cases"],
           "prompt_hashes": {c["id"]: hashlib.sha256(p.encode()).hexdigest() for c, p in zip(inputs["cases"], prompts)},
           "max_input_budget_upper_bound": max(x["input_token_upper_bound"] for x in bounds),
           "source_hashes": {name: sha256_file(ROOT / name) for name in SOURCE_NAMES},
           "scientific_sources": {"accepted_train.npz": sha256_file(DATA),
                                  "E2/predictions.csv": sha256_file(ROOT / "E2/predictions.csv"),
                                  "E2/llm_calls.jsonl": sha256_file(ROOT / "E2/llm_calls.jsonl")},
           "official_CLI_references": ["https://learn.chatgpt.com/docs/developer-commands", "https://learn.chatgpt.com/docs/config-file/config-reference"],
           "runs": []}
    write_json(OUT / "config.json", cfg)
    write_json(OUT / "metrics.json", {"status": "prepared", "n_planned": 320, "n_valid": 0})
    for name in ("llm_calls.jsonl", "run.log"):
        safe_path(OUT / name).write_text("", encoding="utf-8")
    with safe_path(OUT / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=FIELDS).writeheader()
    write_readme("prepared")
    print(json.dumps({"status": "prepared", "n": 320, "runtime": runtime,
                      "max_input_upper_bound": cfg["max_input_budget_upper_bound"]}, ensure_ascii=False))


def baseline_scores(cases):
    real = [case for case in cases if not case["is_control"]]
    positions = {case["id"]: i for i, case in enumerate(real)}
    scores = {name: np.full(len(real), np.nan) for name in ("A2_forced", "A3_forced", "HGB127")}
    with (ROOT / "E2/predictions.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["arm"] in scores:
                scores[row["arm"]][positions[row["id"]]] = float(row["score"])
    assert all(np.isfinite(x).all() for x in scores.values())
    return scores


def finish(cfg, accepted, status, run_index, calls):
    real = [case for case in cfg["cases"] if not case["is_control"]]
    controls = [case for case in cfg["cases"] if case["is_control"]]
    groups = np.asarray([c["group"] for c in real])
    sizes = {c["group"]: c["evaluation"]["original_group_size"] for c in real}
    y = np.asarray([c["evaluation"]["y"] for c in real])
    tc = np.asarray([c["evaluation"]["Tc_K"] for c in real])
    valid = np.asarray([c["id"] in accepted for c in real])
    recognized = np.asarray([accepted.get(c["id"], {}).get("recognized", "missing") for c in real])
    recalled = np.asarray([accepted.get(c["id"], {}).get("reported_Tc_K")
                           if accepted.get(c["id"], {}).get("reported_Tc_K") is not None else np.nan for c in real])
    scores = baseline_scores(cfg["cases"])
    scores = {"R-recall": np.asarray(np.isfinite(recalled) & (recalled > 10), dtype=float), **scores}
    masks = {"all": np.ones(len(real), dtype=bool), "exclude_93": groups != "93", "only_93": groups == "93"}
    result = {"status": status, "scope": "development only; formula-only diagnostic, not a causal attribution estimate",
              "run_index": run_index, "n_planned": len(cfg["cases"]), "n_valid": len(accepted),
              "n_valid_real": int(valid.sum()), "n_valid_controls": sum(c["id"] in accepted for c in controls),
              "missing_ids": [c["id"] for c in cfg["cases"] if c["id"] not in accepted],
              "classification": {}, "by_recognition": {}, "recognition": {}, "recall_correctness": {},
              "calls": {"n": len(calls), "input_tokens": sum(c.get("input_tokens") or 0 for c in calls),
                        "output_tokens": sum(c.get("output_tokens") or 0 for c in calls),
                        "cached_input_tokens": sum(c.get("cached_input_tokens") or 0 for c in calls),
                        "sum_call_seconds": sum(c.get("elapsed_seconds") or 0 for c in calls),
                        "tools_used": sum(len(c.get("tool_activity") or []) for c in calls)},
              "policy": {"null_Tc_prediction": 0, "precision_zero_division": 0,
                         "n_boot": 5000, "seed": 20261016, "group_weights": cfg["group_weighting"]},
              "limitations": ["Recognition is a self-report, not a direct measurement of pretraining exposure or causal memory benefit.",
                              "Random-composition controls are only known absent from accepted3764, not from the literature/pretraining.",
                              "Only93 has one group: degenerate group-bootstrap intervals cannot measure cross-group uncertainty.",
                              "Absolute Tc error<=2K agreement can reflect family knowledge/inference and source/phase differences; no memorization fraction is identified."]}
    for name, mask in masks.items():
        result["classification"][name] = summarize(y, scores, groups, sizes, mask & valid)
    for state in ("yes", "partial", "no"):
        result["by_recognition"][state] = {name: summarize(y, scores, groups, sizes, mask & valid & (recognized == state))
                                           for name, mask in masks.items()}
    for name, mask in masks.items():
        result["recognition"]["real_" + name] = {
            label: fraction_summary(np.where(valid, values.astype(float), np.nan), groups, sizes, mask)
            for label, values in {"yes": recognized == "yes", "partial": recognized == "partial",
                                  "no": recognized == "no", "yes_or_partial": np.isin(recognized, ["yes", "partial"]),
                                  "numeric_Tc": np.isfinite(recalled)}.items()}
        result["recall_correctness"][name] = {}
        for label, recognized_mask in (("yes", recognized == "yes"),
                                       ("yes_or_partial", np.isin(recognized, ["yes", "partial"]))):
            eligible = mask & valid & recognized_mask
            numeric = np.isfinite(recalled)
            within2 = numeric & (np.abs(recalled - tc) <= 2)
            same_side = numeric & ((recalled > 10) == (y == 1))
            result["recall_correctness"][name][label] = {
                "numeric_coverage": fraction_summary(numeric, groups, sizes, eligible),
                "within_2K_all_recognized": fraction_summary(within2, groups, sizes, eligible),
                "same_10K_side_all_recognized": fraction_summary(same_side, groups, sizes, eligible),
                "within_2K_numeric_only": fraction_summary(within2, groups, sizes, eligible & numeric),
                "same_10K_side_numeric_only": fraction_summary(same_side, groups, sizes, eligible & numeric)}
    control_groups = np.asarray([c["group"] for c in controls])
    control_sizes = {g: 1 for g in control_groups}
    control_valid = np.asarray([c["id"] in accepted for c in controls])
    control_states = np.asarray([accepted.get(c["id"], {}).get("recognized", "missing") for c in controls])
    result["recognition"]["controls_all"] = {
        label: fraction_summary(np.where(control_valid, values.astype(float), np.nan), control_groups, control_sizes)
        for label, values in {"yes": control_states == "yes", "partial": control_states == "partial",
                              "no": control_states == "no", "yes_or_partial": np.isin(control_states, ["yes", "partial"])}.items()}
    result["control_responses"] = [{"id": c["id"], "formula": c["formula"], "response": accepted.get(c["id"])} for c in controls]
    result["completed_utc"] = utcnow()
    write_json(OUT / "metrics.json", result)
    write_readme(status, len(accepted))
    return result


def run(task_name, restart_index):
    cfg = read_json(OUT / "config.json")
    # The scheduled-task environment does not inherit the app's CLI PATH entry.
    # Use the already frozen, fingerprinted executable without changing auth.
    os.environ["F1_CODEX_EXE"] = cfg["runtime"]["executable"]
    assert cfg["task_name"] == task_name and 0 <= restart_index <= 1
    assert restart_index == len(cfg["runs"]), "Only one explicitly diagnosed full recomputation is allowed"
    for name, digest in cfg["source_hashes"].items():
        assert sha256_file(ROOT / name) == digest, name
    assert resolve_runtime() == cfg["runtime"], "Frozen runtime changed"
    for c in cfg["cases"]:
        assert hashlib.sha256(make_prompt(c).encode()).hexdigest() == cfg["prompt_hashes"][c["id"]]
    cfg["runs"].append({"index": restart_index, "started_utc": utcnow(), "policy": "full recomputation; no per-row recovery"})
    write_json(OUT / "config.json", cfg)
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    lock, stop = threading.RLock(), threading.Event()
    accepted, calls, pending_rows = {}, [], []
    started = time.monotonic()
    log_handle = safe_path(OUT / "run.log").open("a", encoding="utf-8", buffering=1)

    def log(message):
        with lock:
            log_handle.write(utcnow() + " " + message + "\n")

    def append_call(record):
        with lock, safe_path(OUT / "llm_calls.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def flush_rows(force=False):
        if len(pending_rows) >= 10 or (force and pending_rows):
            with safe_path(OUT / "predictions.csv").open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=FIELDS).writerows(pending_rows)
                handle.flush()
                os.fsync(handle.fileno())
            pending_rows.clear()

    def work(index, case):
        prompt = make_prompt(case)
        metadata = {"experiment": "F1_recall", "run_index": restart_index, "call_id": index,
                    "id": case["id"], "is_control": case["is_control"], "formula": case["formula"],
                    "started_utc": utcnow(), "prompt": prompt}
        append_call({"event": "call_started", **metadata, "request_hash": hashlib.sha256(prompt.encode()).hexdigest()})
        result = call_formula(prompt, schema_path=SCHEMA, instructions_path=INSTRUCTIONS,
                              catalog_path=CATALOG, stop_event=stop, metadata=metadata)
        obj = validate(result)
        result.update(event="call_completed", finished_utc=utcnow(), response_valid=obj is not None, parsed_response=obj)
        append_call(result)
        return case, result, obj

    try:
        log(f"F1 run{restart_index}: full recomputation;320 fresh formula-only calls;10 concurrent; no runtime cap")
        with safe_path(OUT / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=FIELDS).writeheader()
        baseline = baseline_scores(cfg["cases"])
        real = [c for c in cfg["cases"] if not c["is_control"]]
        for j, c in enumerate(real):
            for arm, values in baseline.items():
                score = float(values[j])
                pending_rows.append(dict(id=c["id"], fold=c["fold"], group=c["group"], arm=arm,
                                         status="frozen_E2_reused", score=score, y_pred=int(score >= .5)))
            if (j + 1) % 10 == 0:
                flush_rows(force=True)
        worklist = iter(enumerate(cfg["cases"]))
        finished = 0
        failed_consecutive = 0
        with ThreadPoolExecutor(max_workers=10) as executor:
            running = {}
            for _ in range(10):
                i, case = next(worklist)
                running[executor.submit(work, i, case)] = i
            exhausted = False
            while running:
                done, _ = wait(running, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    running.pop(future)
                    case, result, obj = future.result()
                    calls.append(result)
                    finished += 1
                    if obj is not None:
                        accepted[case["id"]] = obj
                        score = int(obj["reported_Tc_K"] is not None and obj["reported_Tc_K"] > 10)
                        row_status = ("control_" if case["is_control"] else "real_") + obj["recognized"] + ("_null_as_negative" if obj["reported_Tc_K"] is None else "_numeric")
                        failed_consecutive = 0
                    else:
                        score, row_status = "", result["status"]
                        failed_consecutive += 1
                    pending_rows.append(dict(id=case["id"], fold=case["fold"], group=case["group"],
                                             arm="R-recall", status=row_status, score=score, y_pred=score))
                    flush_rows()
                    log(f"{finished}/320 {case['id']} {row_status}; valid={len(accepted)}")
                    if result["status"] in {"tool_activity_rejected", "model_mismatch_rejected", "backend_protocol_rejected", "preflight_rejected", "actual_input_budget_violation"} or failed_consecutive >= 10:
                        stop.set()
                    if finished % 10 == 0:
                        write_json(OUT / "metrics.json", {"status": "running" if not stop.is_set() else "stopping",
                                   "run_index": restart_index, "n_planned": 320, "n_finished": finished, "n_valid": len(accepted)})
                    if not stop.is_set() and not exhausted:
                        try:
                            i, next_case = next(worklist)
                            running[executor.submit(work, i, next_case)] = i
                        except StopIteration:
                            exhausted = True
        flush_rows(force=True)
        status = "complete" if len(accepted) == 320 else "stopped_partial"
        result = finish(cfg, accepted, status, restart_index, calls)
        result["wall_seconds"] = time.monotonic() - started
        write_json(OUT / "metrics.json", result)
        log(f"F1 {status}: valid={len(accepted)}/320; wall={result['wall_seconds']:.2f}s; stop={stop.is_set()}")
        return 0 if status == "complete" else 75
    except BaseException as exc:
        stop.set()
        flush_rows(force=True)
        log(f"FATAL {type(exc).__name__}: {exc}; inspect system/task logs before any one permitted restart")
        raise
    finally:
        log_handle.close()
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--restart-index", type=int, default=0)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.task_name)
    else:
        try:
            raise SystemExit(run(args.task_name, args.restart_index))
        except Exception:
            import traceback
            with safe_path(OUT / "run.log").open("a", encoding="utf-8") as handle:
                handle.write(utcnow() + " startup/runtime failure\n" + traceback.format_exc())
            raise
