"""Finite F2 masking experiment; no Stage3 work and no calls on import."""
from __future__ import annotations

from common import ROOT, DATA, FIELDS, load_data, np, safe_path, write_json
from f1_inputs import sha256_file
from f1_transport import call_formula, resolve_runtime
from e2_transport import input_budget
from f2_m1 import build_m1, property_metadata, audit_fixed300
from f2_numeric import build_numeric_inputs, compact_numeric_packet
from f2_prompts import make_prompt
from f2_report import build_f2_report, PAIRS
from f_stats import summarize, fraction_summary
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
import subprocess
import threading
import time
import traceback

ARMS = ("A3-M1", "A3-M2", "A2-M2")
OUT = ROOT / "F2_mask"
SCHEMA, INSTRUCTIONS, CATALOG = [ROOT / s for s in (
    "e2_response.schema.json", "f2_instructions.md", "e2_model_catalog.json")]
SOURCES = ("run_f2.py", "f2_m1.py", "f2_numeric.py", "f2_prompts.py", "f2_report.py",
           "f2_instructions.md", "launch_f2.ps1", "f_stats.py", "f_report.py",
           "f1_transport.py", "f1_inputs.py", "e2_transport.py", "e2_prompts.py", "e2_packets.py",
           "e2_model_catalog.json", "e2_response.schema.json", "common.py")
INTEGRITY = {"tool_activity_rejected", "model_mismatch_rejected", "backend_protocol_rejected",
             "preflight_rejected", "actual_input_budget_violation"}
REPAIRABLE = {"schema_invalid", "cli_error", "backend_error", "incomplete_response", "transport_error"}


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_readme(status, counts=None):
    lines = ["# F2 identity masking development diagnostic", f"Status: {status}; valid forced decisions: {counts or {}}.",
        "Fixed E2 300materials/81groups; all/exclude93/only93; no Stage3 execution.",
        "A3-M1: per-material shuffled E1..Ek;5bins each for EN,covalent radius,Magpie s+p+d+f valence count.",
        "M1 preserves stoichiometry/geometry/space group; removes mass-derived summary; identity may remain inferable.",
        "A3-M2: TRAIN-standardized X127, f1-f127 only; original column order may still encode element identity.",
        "A2-M2: same target representation plus E2-identical five TRAIN neighbour vectors and binary labels; no rules.",
        "Same gpt-6.1-sol/high and CLI version as E2; local ChatGPT login only; no tools/network/professor API.",
        "First valid forced p retained even if abstain; <=2fresh calls/item, second only format/transport repair.",
        "10concurrent calls; no task time limit; SetThreadExecutionState prevents sleep; predictions fsync every10arm-cases.",
        "Rate/quota immediately saves/stops without restart; process interruption requires logs before at most1full recomputation.",
        "All methods: BA/P/R/accuracy, inverse original-group and material weights;5000 paired group bootstrap;95%CI.",
        "Main tables use all-method common valid rows; failure and abstention coverage are separate; fallback is secondary.",
        "Only93 is one group, so its CI degenerates and cannot quantify cross-group uncertainty.",
        "Full tables and paired changes: metrics.json/report_markdown; source/protocol/scalers/bins: config.json.",
        "Masking/quantization/prompt changes are joint interventions; no causal memorization fraction is identified.",
        "Stage3 revision recorded in INPUTS.md only: full A2+A3 forced, reuse fixed300, <=10000calls, no full recall.",
        "Stop after F2 and await human confirmation; no recurring task or automatic route selection."]
    assert len(lines) <= 20
    safe_path(OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(record):
    if record.get("status") != "ok":
        return None
    try:
        obj = json.loads(record["response_text"])
        assert isinstance(obj, dict) and set(obj) == {"action", "p", "abstain", "selected_rule_ids", "request_rule_ids", "request_counterexamples", "reason"}
        assert obj["action"] == "decide" and type(obj["abstain"]) is bool
        assert type(obj["p"]) in (int, float) and math.isfinite(obj["p"]) and 0 <= obj["p"] <= 1
        assert obj["selected_rule_ids"] == obj["request_rule_ids"] == []
        assert obj["request_counterexamples"] is False
        assert isinstance(obj["reason"], str) and len(obj["reason"]) <= 80
        return obj
    except (AssertionError, ValueError, TypeError, KeyError):
        record["status"] = "schema_invalid"
        record["error"] = "Frozen F2 decision contract violated; only a same-input format repair is allowed"
        return None


def prepare(task_name):
    if OUT.exists():
        raise RuntimeError("F2 directory already exists; inspect state before any recomputation")
    for ancestor in (ROOT, *ROOT.parents):
        assert not ancestor.is_symlink() and not ancestor.is_junction()
    d = load_data()
    e2 = read_json(ROOT / "E2/config.json")
    numeric = build_numeric_inputs(d)
    indices = numeric["indices"]
    assert indices == e2["indices"]
    cases, bounds = [], {arm: [] for arm in ARMS}
    for i in indices:
        key = str(d["id"][i])
        old = e2["cases"][key]
        packets = {"A3-M1": {"material": build_m1(old["material"], key)},
                   **{arm: compact_numeric_packet(numeric["cases"][key], arm) for arm in ("A3-M2", "A2-M2")}}
        case = {"id": key, "i": i, "fold": int(d["fold"][i]), "group": str(d["group"][i]),
                "evaluation": {"Tc_K": float(d["tc"][i]), "y": int(d["y"][i]), "original_group_size": int(d["group_size"][i])},
                "packets": packets, "prompt_hashes": {},
                "neighbour_audit_not_for_prompt": numeric["cases"][key]["neighbour_audit_not_for_prompt"]}
        for arm in ARMS:
            case["prompt_hashes"][arm] = {}
            for attempt in (1, 2):
                prompt = make_prompt(arm, packets[arm], round_number=attempt, repair=attempt == 2)
                bound = input_budget(prompt, schema_path=SCHEMA, instructions_path=INSTRUCTIONS)
                if not bound["within_budget"]:
                    raise ValueError(f"Conservative prompt budget exceeded before experiment creation: {key} {arm} {attempt}: {bound}")
                bounds[arm].append(bound["input_token_upper_bound"])
                case["prompt_hashes"][arm][str(attempt)] = digest(prompt)
        cases.append(case)
    cfg = {"experiment": "F2_mask", "stage": 2, "scope": "development only; stop before revised Stage3",
           "created_utc": now(), "task_name": task_name, "runtime": resolve_runtime(),
           "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
           "model_version_note": "Same explicit model slug/effort and CLI version; immutable backend-weight snapshot unavailable.",
           "params": {"n": 300, "n_groups": 81, "arms": list(ARMS), "concurrency": 10,
                      "max_rounds": 2, "planned_initial_calls": 900, "max_calls_per_run": 1800,
                      "max_process_restarts": 1, "max_calls_across_runs": 3600,
                      "bootstrap_draws": 5000, "seed": 20261016, "execution_time_limit": "none",
                      "prediction_flush_every_arm_cases": 10},
           "cases": cases, "numeric_metadata_not_for_prompt": numeric["metadata"],
           "M1_property_metadata_not_for_prompt": property_metadata(), "M1_input_audit": audit_fixed300(e2),
           "max_prompt_bounds_including_repair": {arm: max(v) for arm, v in bounds.items()},
           "source_hashes": {name: sha256_file(ROOT / name) for name in SOURCES},
           "scientific_sources": {"accepted_train.npz": sha256_file(DATA), **{f"E2/{name}": sha256_file(ROOT / "E2" / name) for name in ("config.json", "predictions.csv", "metrics.json")}},
           "scientific_input_rule": "Only case.packets[arm] enters prompt; all identity, target outcome, property/scaler provenance and audit metadata excluded.",
           "round2_rule": "Fresh context, same scientific payload, fixed format-repair instruction only; no earlier free-text response or guessed identity copied.",
           "interruption_rule": "Read task result and system/task logs before at most one full recomputation; no per-record recovery. Provider limits stop without restart.",
           "revised_Stage3_not_started": {"recall_test": "cancelled", "LLM_arms": ["A2_forced", "A3_forced"], "cohort": 3764,
               "fixed300": "reuse E2 unchanged; no model rerun", "baselines": ["HGB127", "E0_StrongML", "5NN"],
               "max_calls": 10000, "report": "both weightings; all/exclude93/only93; retain original family/size breakdowns where applicable; no full-cohort recognition strata"},
           "runs": []}
    safe_path(OUT).mkdir()
    write_json(OUT / "config.json", cfg)
    write_json(OUT / "metrics.json", {"status": "prepared", "planned_arm_cases": 900})
    for name in ("llm_calls.jsonl", "run.log"):
        safe_path(OUT / name).write_text("", encoding="utf-8")
    with safe_path(OUT / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=FIELDS).writeheader()
    write_readme("prepared")
    print(json.dumps({"status": "prepared", "n": 300, "jobs": 900, "bounds": cfg["max_prompt_bounds_including_repair"], "runtime": cfg["runtime"]}))


def baseline_scores(cases):
    positions = {c["id"]: j for j, c in enumerate(cases)}
    mapping = {"A3_forced": "A3_E2", "A2_forced": "A2_E2", "HGB127": "HGB127", "5NN": "5NN"}
    scores = {name: np.full(len(cases), np.nan) for name in mapping.values()}
    with (ROOT / "E2/predictions.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["arm"] in mapping:
                scores[mapping[row["arm"]]][positions[row["id"]]] = float(row["score"])
    assert all(np.isfinite(v).all() for v in scores.values())
    return scores


def finish(cfg, results, calls, status, run_index, seconds):
    cases = cfg["cases"]
    y = np.asarray([c["evaluation"]["y"] for c in cases])
    groups = np.asarray([c["group"] for c in cases])
    sizes = {c["group"]: c["evaluation"]["original_group_size"] for c in cases}
    scores = baseline_scores(cases)
    fallback = dict(scores)
    valid_by_arm, done_by_arm, abstain_by_arm = {}, {}, {}
    counts, failures = {}, []
    for arm in ARMS:
        done = np.asarray([results.get((arm, c["id"]), {}).get("attempts", 0) > 0 for c in cases])
        objs = [results.get((arm, c["id"]), {}).get("response") for c in cases]
        valid = np.asarray([obj is not None for obj in objs])
        abstain = np.asarray([obj["abstain"] if obj is not None else False for obj in objs])
        p = np.asarray([obj["p"] if obj is not None else np.nan for obj in objs], dtype=float)
        scores[arm] = p
        fallback[arm + "_HGBfallback"] = np.where(done, np.where(valid & ~abstain, p, scores["HGB127"]), np.nan)
        valid_by_arm[arm], done_by_arm[arm], abstain_by_arm[arm] = valid, done, abstain
        counts[arm] = {"completed": int(done.sum()), "valid": int(valid.sum()), "failed": int((done & ~valid).sum()),
                       "not_attempted": int((~done).sum()), "abstain": int((valid & abstain).sum())}
        failures.extend({"arm": arm, "id": c["id"], "status": results.get((arm, c["id"]), {}).get("status", "not_attempted")}
                        for j, c in enumerate(cases) if not valid[j])
    common = np.logical_and.reduce(list(valid_by_arm.values()))
    all_done = np.logical_and.reduce(list(done_by_arm.values()))
    masks = {"all": np.ones(len(cases), dtype=bool), "exclude_93": groups != "93", "only_93": groups == "93"}
    metrics = {"status": status, "scope": "fixed300 development results; no external test or causal memory decomposition",
               "run_index": run_index, "completed_utc": now(), "wall_seconds": seconds, "case_counts": counts,
               "planned_n": 300, "planned_groups": 81, "common_valid_n": int(common.sum()),
               "subsets": {}, "fallback_subsets": {}, "rates": {}, "failures_or_missing": failures,
               "report_pairs": list(PAIRS), "calls": {"n": len(calls),
                   "by_arm": {arm: sum(r["arm"] == arm for r in calls) for arm in ARMS},
                   "second_round_calls": sum(r["attempt"] == 2 for r in calls),
                   "input_tokens": sum(r.get("input_tokens") or 0 for r in calls),
                   "output_tokens": sum(r.get("output_tokens") or 0 for r in calls),
                   "cached_input_tokens": sum(r.get("cached_input_tokens") or 0 for r in calls),
                   "sum_call_seconds": sum(r.get("elapsed_seconds") or 0 for r in calls),
                   "tools_used": sum(len(r.get("tool_activity") or []) for r in calls)},
               "method_notes": {"forced": "valid p is used even when abstain=true", "fallback": "HGB on explicit abstention or exhausted failure; secondary only",
                                "weights": "1/full3764 original group size; retained after all masks", "n_boot": 5000, "seed": 20261016,
                                "M1_valence": "Magpie occupied s+p+d+f convention; not oxidation state or solely outermost electrons"},
               "limitations": ["Joint changes of identity, quantization, representation and prompt cannot identify a causal memorization percentage.",
                   "M1 properties/stoichiometry/geometry may allow identity inference; M2 ordered element-fraction columns may retain implicit identity.",
                   "HGB is trained on all outer TRAIN labels; the LLM sees only its supplied context, at most five neighbour labels.",
                   "Only93 has one group and degenerate bootstrap intervals; no cross-group uncertainty is measurable.",
                   "Model slug/effort and CLI are fixed, but an immutable backend-weight snapshot is not exposed."],
               "Stage3_started": False}
    for label, mask in masks.items():
        metrics["subsets"][label] = summarize(y, scores, groups, sizes, mask & common)
        metrics["fallback_subsets"][label] = summarize(y, fallback, groups, sizes, mask & all_done)
        metrics["rates"][label] = {}
        for arm in ARMS:
            done, valid, abstain = done_by_arm[arm], valid_by_arm[arm], abstain_by_arm[arm]
            values = {"abstain": np.where(valid, abstain.astype(float), np.nan),
                      "failure": np.where(done, (~valid).astype(float), np.nan),
                      "not_attempted": (~done).astype(float)}
            metrics["rates"][label][arm] = {}
            for name, vector in values.items():
                stat = fraction_summary(vector, groups, sizes, mask)
                stat["count"] = int(np.sum(np.isfinite(vector[mask]) & (vector[mask] == 1)))
                metrics["rates"][label][arm][name] = stat
    metrics["report_markdown"] = build_f2_report(metrics)
    write_json(OUT / "metrics.json", metrics)
    write_readme(status, counts)
    return metrics


def run(task_name, restart_index):
    cfg = read_json(OUT / "config.json")
    assert cfg["task_name"] == task_name and 0 <= restart_index <= 1
    assert restart_index == len(cfg["runs"]), "No row recovery; at most one diagnosed full rerun"
    os.environ["F1_CODEX_EXE"] = cfg["runtime"]["executable"]
    assert resolve_runtime() == cfg["runtime"]
    for name, expected in cfg["source_hashes"].items():
        assert sha256_file(ROOT / name) == expected, name
    for name, expected in cfg["scientific_sources"].items():
        source = DATA if name == "accepted_train.npz" else ROOT / name
        assert sha256_file(source) == expected, name
    for c in cfg["cases"]:
        for arm in ARMS:
            for attempt in (1, 2):
                assert digest(make_prompt(arm, c["packets"][arm], round_number=attempt, repair=attempt == 2)) == c["prompt_hashes"][arm][str(attempt)]
    cfg["runs"].append({"index": restart_index, "started_utc": now(), "policy": "full recomputation, no old-record recovery"})
    write_json(OUT / "config.json", cfg)
    if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
        raise RuntimeError("Failed to prevent sleep")
    lock, stop = threading.RLock(), threading.Event()
    results, calls, pending_rows = {}, [], []
    started = time.monotonic()
    log_handle = safe_path(OUT / "run.log").open("a", encoding="utf-8", buffering=1)

    def log(message):
        with lock:
            log_handle.write(now() + " " + message + "\n")

    def save_call(record):
        with lock, safe_path(OUT / "llm_calls.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def flush(force=False):
        if len(pending_rows) >= 20 or (force and pending_rows):
            with safe_path(OUT / "predictions.csv").open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=FIELDS).writerows(pending_rows)
                handle.flush()
                os.fsync(handle.fileno())
            pending_rows.clear()

    def worker(arm, case):
        final, obj = None, None
        for attempt in (1, 2):
            if stop.is_set():
                break
            prompt = make_prompt(arm, case["packets"][arm], round_number=attempt, repair=attempt == 2)
            metadata = {"experiment": "F2_mask", "run_index": restart_index, "arm": arm, "id": case["id"],
                        "attempt": attempt, "call_id": f"{restart_index}|{arm}|{case['id']}|{attempt}",
                        "prompt": prompt, "started_utc": now()}
            save_call({"event": "call_started", **metadata, "request_hash": digest(prompt)})
            record = call_formula(prompt, schema_path=SCHEMA, instructions_path=INSTRUCTIONS,
                                  catalog_path=CATALOG, stop_event=stop, metadata=metadata)
            obj = validate(record)
            record.update(event="call_completed", finished_utc=now(), response_valid=obj is not None, parsed_response=obj)
            save_call(record)
            with lock:
                calls.append(record)
            final = record
            if obj is not None or stop.is_set() or record["status"] not in REPAIRABLE:
                break
        return arm, case, {"response": obj, "status": final["status"] if final else "not_dispatched_due_stop",
                           "attempts": final["attempt"] if final else 0}

    try:
        log(f"F2 run{restart_index} started:900arm-cases;10 concurrent;<=2fresh calls/item;no time cap")
        with safe_path(OUT / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=FIELDS).writeheader()
        baselines = baseline_scores(cfg["cases"])
        positions = {c["id"]: j for j, c in enumerate(cfg["cases"])}
        for j, case in enumerate(cfg["cases"]):
            for arm, values in baselines.items():
                score = float(values[j])
                pending_rows.append(dict(id=case["id"], fold=case["fold"], group=case["group"], arm=arm,
                                         status="frozen_E2_reused", score=score, y_pred=int(score >= .5)))
            if (j + 1) % 10 == 0:
                flush(True)
        jobs = [(arm, c) for c in cfg["cases"] for arm in ARMS]
        jobs.sort(key=lambda job: digest("F2-interleaved-20261017|" + job[0] + "|" + job[1]["id"]))
        iterator = iter(jobs)
        with ThreadPoolExecutor(max_workers=10) as pool:
            active = {}
            for _ in range(10):
                arm, case = next(iterator)
                active[pool.submit(worker, arm, case)] = (arm, case["id"])
            exhausted, consecutive_failed = False, 0
            try:
                while active:
                    ready, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                    for future in ready:
                        active.pop(future)
                        arm, case, result = future.result()
                        results[(arm, case["id"])] = result
                        obj = result["response"]
                        p = float(obj["p"]) if obj is not None else ""
                        dispatched = result["attempts"] > 0
                        fallback = (float(baselines["HGB127"][positions[case["id"]]]) if obj is None or obj["abstain"] else p) if dispatched else ""
                        pending_rows.extend([
                            dict(id=case["id"], fold=case["fold"], group=case["group"], arm=arm,
                                 status=("valid_abstain_forced" if obj["abstain"] else "valid_forced") if obj else result["status"],
                                 score=p, y_pred=int(p >= .5) if p != "" else ""),
                            dict(id=case["id"], fold=case["fold"], group=case["group"], arm=arm + "_HGBfallback",
                                 status=("HGB_on_abstain_or_failure" if obj is None or obj["abstain"] else "LLM_direct") if dispatched else "not_attempted",
                                 score=fallback, y_pred=int(fallback >= .5) if fallback != "" else "")])
                        flush()
                        consecutive_failed = 0 if obj is not None else consecutive_failed + 1
                        if result["status"] in INTEGRITY or consecutive_failed >= 10:
                            stop.set()
                        n = len(results)
                        log(f"{n}/900 {arm} {case['id']} {result['status']} attempts={result['attempts']}")
                        if n % 10 == 0:
                            write_json(OUT / "metrics.json", {"status": "stopping" if stop.is_set() else "running",
                                "run_index": restart_index, "n_finished_arm_cases": n, "n_calls": len(calls),
                                "valid_by_arm": {a: sum(k[0] == a and v["response"] is not None for k, v in results.items()) for a in ARMS}})
                        if not stop.is_set() and not exhausted:
                            try:
                                next_arm, next_case = next(iterator)
                                active[pool.submit(worker, next_arm, next_case)] = (next_arm, next_case["id"])
                            except StopIteration:
                                exhausted = True
            except BaseException:
                stop.set()
                for future in active:
                    future.cancel()
                raise
        flush(True)
        status = "complete" if len(results) == 900 and all(v["response"] is not None for v in results.values()) else "stopped_partial" if stop.is_set() else "complete_with_failures"
        m = finish(cfg, results, calls, status, restart_index, time.monotonic() - started)
        log(f"F2 {status}: {m['case_counts']};calls={len(calls)};wall={m['wall_seconds']:.2f}s; no Stage3 work")
        return 0 if status == "complete" else 75
    except BaseException as exc:
        stop.set()
        flush(True)
        finish(cfg, results, calls, "interrupted_error", restart_index, time.monotonic() - started)
        log(f"FATAL {type(exc).__name__}: {exc}; inspect task/system logs before at most1full restart")
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
            with safe_path(OUT / "run.log").open("a", encoding="utf-8") as handle:
                handle.write(now() + " startup/runtime failure\n" + traceback.format_exc())
            raise
