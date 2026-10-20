"""Human-authorized F2 resume: retain valid results, four workers, bounded retries."""
from __future__ import annotations

from common import ROOT, DATA, FIELDS, safe_path, write_json, np
from f1_inputs import sha256_file
from f2_transport_v2 import call_once, resolve_runtime
from f2_prompts import make_prompt
from f2_report import build_f2_report
from f_stats import summarize
from run_f2 import (ARMS, OUT, SCHEMA, INSTRUCTIONS, CATALOG, INTEGRITY,
                    validate, baseline_scores, finish, read_json, now, digest)
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import argparse
import csv
import json
import os
import ctypes
import subprocess
import threading
import time
import traceback

RUN_ID = "human_resume_v2_1"
DELAYS = (60, 120, 240)
RESUME_SOURCES = ("resume_f2.py", "f2_transport_v2.py", "f2_probe.py", "launch_f2_resume.ps1",
                  "run_f2.py", "f2_report.py", "f2_prompts.py", "f2_instructions.md",
                  "f_stats.py", "f_report.py", "common.py", "f1_inputs.py", "f1_transport.py",
                  "e2_transport.py", "e2_prompts.py", "e2_packets.py",
                  "e2_model_catalog.json", "e2_response.schema.json")


def history(cfg):
    """One light pass over the small call manifest; preserve the FIRST valid answer."""
    cases = {c["id"]: c for c in cfg["cases"]}
    records, results = [], {}
    with (OUT / "llm_calls.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("event") != "call_completed" or record.get("arm") not in ARMS:
                continue
            records.append(record)
            if not record.get("response_valid"):
                continue
            obj = validate(dict(record))
            assert obj is not None
            key = (record["arm"], record["id"])
            case = cases[record["id"]]
            round_number = record.get("scientific_round", record["attempt"])
            assert record["request_hash"] == case["prompt_hashes"][record["arm"]][str(round_number)]
            assert digest(record["prompt"]) == record["request_hash"]
            if key in results:
                raise RuntimeError("A valid material was called again; do not silently select an answer")
            results[key] = {"response": obj, "status": "ok", "attempts": round_number,
                            "retained": True, "source_call_id": record["call_id"]}
    return records, results


def response_hash(value):
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def write_readme(status, counts):
    lines = ["# F2 identity masking development diagnostic", f"Status: {status}; valid forced decisions: {counts}.",
      "Fixed300/81groups; all/exclude93/only93; both weights; BA/P/R/accuracy;5000paired group bootstrap;95%CI.",
      "Scientific packets/prompts unchanged: M1 coarse anonymous properties+geometry; M2 standardized f1-f127; A2-M2 adds same5TRAIN labels.",
      "Magpie valence=s+p+d+f convention; M1/M2 may still retain inferable identity; no causal memorization percentage.",
      "Five historical valid results retained exactly; prior10without responses and all unattempted arm-cases are pending.",
      "A single synthetic probe succeeded before this resume; not a scientific material or an extra prediction.",
      "4concurrent calls; A3-M2/A2-M2 numeric stage first, then A3-M1; no task time cap; sleep prevented.",
      "CLI Reconnecting diagnostics run to final completion; a recovered429 is success when a valid final answer is returned.",
      "Final429: same material/prompt retried at1,2,4minutes, max3additional transport retries shared across its scientific rounds.",
      "The listed8/15minute delays are unused under the human max3retry cap; scientific schema repair remains at most2rounds.",
      "Explicit usage-limit/try-again-at stops immediately; otherwise stop after3consecutive distinct material IDs exhaust all retries.",
      "Prediction CSV drops obsolete failed placeholders, retains valid answers, then appends/fsyncs every10completed arm-cases.",
      "Main forced table uses common valid rows; numeric_priority_subsets preserve six-method numeric comparisons if M1 is incomplete.",
      "HGBfallback is secondary; failed forced values stay blank; cancellation is not a classification error.",
      "Complete tables: metrics.json/report_markdown. Old experiment directories remain read-only.",
      "E2 model slug/effort/CLI retained, no tools/network in evaluated calls; immutable backend weight snapshot unavailable.",
      "Stop after F2 for human confirmation; revised Stage3 is recorded but not launched."]
    assert len(lines) <= 20
    safe_path(OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure(task_name):
    cfg = read_json(OUT / "config.json")
    assert not cfg.get("resume_v2"), "Resume already configured; inspect before repeating"
    probe = cfg["resume_probes"][-1]
    assert probe["status"] == "ok" and not probe["explicit_usage_limit"], "Probe did not establish current availability"
    for name, expected in cfg["scientific_sources"].items():
        assert sha256_file(DATA if name == "accepted_train.npz" else ROOT / name) == expected, name
    # New retry/report code is allowed; scientific inputs and prompt builders are not changed.
    for name in ("f2_prompts.py", "f2_instructions.md", "e2_response.schema.json", "e2_model_catalog.json", "common.py", "f_stats.py"):
        assert sha256_file(ROOT / name) == cfg["source_hashes"][name], name
    calls, retained = history(cfg)
    assert len(calls) == 15 and len(retained) == 5
    cfg["resume_v2"] = {"run_id": RUN_ID, "task_name": task_name, "configured_utc": now(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_hashes": {name: sha256_file(ROOT / name) for name in RESUME_SOURCES},
        "concurrency": 4, "arm_order": [["A3-M2", "A2-M2"], ["A3-M1"]],
        "max_additional_429_retries_per_material_arm_case": 3, "transport_retry_wait_seconds": list(DELAYS),
        "unused_user_delay_seconds": [480, 900], "delay_resolution": "max3retry cap takes precedence; use first3listed delays",
        "scientific_rounds": 2, "retry_scientific_input": "identical frozen prompt within each round; no new evidence",
        "retained_valid_count": 5, "historical_call_count": 15,
        "retained_response_hashes": {arm + "|" + key: response_hash(result["response"]) for (arm, key), result in retained.items()},
        "historical_valid_by_arm": {arm: sum(k[0] == arm for k in retained) for arm in ARMS},
        "pending_arm_cases": 895, "max_new_CLI_requests": 4475,
        "stop_rule": "explicit account usage exhaustion, experimental integrity violation, or3consecutive distinct fully exhausted material IDs; recovered CLI reconnects do not count",
        "started_utc": None, "completed_utc": None, "status": "configured"}
    cfg["current_stop_requires_human_input"] = False
    write_json(OUT / "config.json", cfg)
    print(json.dumps({"status": "configured", "retained": 5, "pending_numeric": 597, "pending_M1": 298,
                      "concurrency": 4, "delays_seconds": DELAYS}))


def numeric_priority(cfg, results):
    cases = cfg["cases"]
    groups = np.asarray([c["group"] for c in cases])
    y = np.asarray([c["evaluation"]["y"] for c in cases])
    sizes = {c["group"]: c["evaluation"]["original_group_size"] for c in cases}
    scores = baseline_scores(cases)
    common = np.ones(len(cases), dtype=bool)
    for arm in ("A3-M2", "A2-M2"):
        objs = [results.get((arm, c["id"]), {}).get("response") for c in cases]
        common &= np.asarray([obj is not None for obj in objs])
        scores[arm] = np.asarray([obj["p"] if obj is not None else np.nan for obj in objs])
    masks = {"all": np.ones(len(cases), dtype=bool), "exclude_93": groups != "93", "only_93": groups == "93"}
    return {key: summarize(y, scores, groups, sizes, mask & common) for key, mask in masks.items()}


def run(task_name):
    cfg = read_json(OUT / "config.json")
    resume = cfg["resume_v2"]
    assert resume["task_name"] == task_name and resume["status"] == "configured"
    os.environ["F1_CODEX_EXE"] = cfg["runtime"]["executable"]
    assert resolve_runtime() == cfg["runtime"]
    for name, expected in resume["source_hashes"].items():
        assert sha256_file(ROOT / name) == expected, name
    for name, expected in cfg["scientific_sources"].items():
        assert sha256_file(DATA if name == "accepted_train.npz" else ROOT / name) == expected, name
    calls, results = history(cfg)
    assert len(results) == 5
    initial_retained = {k: response_hash(v["response"]) for k, v in results.items()}
    for (arm, key), expected in initial_retained.items():
        assert expected == resume["retained_response_hashes"][arm + "|" + key]
    resume.update(status="running", started_utc=now())
    write_json(OUT / "config.json", cfg)
    assert ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    lock, stop = threading.RLock(), threading.Event()
    log_handle = safe_path(OUT / "run.log").open("a", encoding="utf-8", buffering=1)
    pending_rows, stop_info, streak = [], {}, []
    baseline = baseline_scores(cfg["cases"])
    positions = {c["id"]: i for i, c in enumerate(cfg["cases"])}
    started = time.monotonic()

    def log(text):
        with lock:
            log_handle.write(now() + " " + text + "\n")

    def append(record):
        with lock, safe_path(OUT / "llm_calls.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def rows_for(arm, case, result):
        obj = result["response"]
        p = float(obj["p"]) if obj is not None else ""
        used_hgb = obj is None or obj["abstain"]
        fallback = float(baseline["HGB127"][positions[case["id"]]]) if used_hgb else p
        return [dict(id=case["id"], fold=case["fold"], group=case["group"], arm=arm,
            status=("valid_abstain_forced" if obj["abstain"] else "valid_forced") if obj else result["status"],
            score=p, y_pred=int(p >= .5) if p != "" else ""),
            dict(id=case["id"], fold=case["fold"], group=case["group"], arm=arm + "_HGBfallback",
                 status="HGB_on_abstain_or_failure" if used_hgb else "LLM_direct", score=fallback, y_pred=int(fallback >= .5))]

    def flush(force=False):
        if len(pending_rows) >= 20 or (force and pending_rows):
            with safe_path(OUT / "predictions.csv").open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=FIELDS).writerows(pending_rows)
                handle.flush()
                os.fsync(handle.fileno())
            pending_rows.clear()

    def snapshot(status):
        for key, expected in initial_retained.items():
            assert response_hash(results[key]["response"]) == expected
        m = finish(cfg, results, calls, status, RUN_ID, time.monotonic() - started)
        m["numeric_priority_subsets"] = numeric_priority(cfg, results)
        m["transport_resume"] = {**resume, "status": status, "completed_utc": now() if status not in ("running_numeric_complete",) else None,
            "stop_reason": dict(stop_info), "historical_started_utc": cfg["runs"][0]["started_utc"],
            "historical_stopped_utc": cfg["runs"][0]["finished_utc"],
            "recovery_time_hint": stop_info.get("recovery_time_message"),
            "new_requests": sum(r.get("run_index") == RUN_ID for r in calls),
            "additional_429_retries": sum(r.get("run_index") == RUN_ID and r.get("transport_retry", 0) > 0 for r in calls),
            "consecutive_failure_ids": list(streak), "probe_calls": len(cfg.get("resume_probes", []))}
        m["recovery_time_hint"] = stop_info.get("recovery_time_message")
        m["limitations"].append("Five valid predictions were retained from the earlier transport/time snapshot; scientific prompts are unchanged, immutable backend weights remain unverified.")
        m["report_markdown"] = build_f2_report(m)
        write_json(OUT / "metrics.json", m)
        write_readme(status, m["case_counts"])
        return m

    def work(arm, case):
        last, obj, network_retries, requests = None, None, 0, 0
        for scientific_round in (1, 2):
            prompt = make_prompt(arm, case["packets"][arm], round_number=scientific_round, repair=scientific_round == 2)
            assert digest(prompt) == case["prompt_hashes"][arm][str(scientific_round)]
            while not stop.is_set():
                requests += 1
                metadata = {"experiment": "F2_mask", "run_index": RUN_ID, "arm": arm, "id": case["id"],
                    "attempt": scientific_round, "scientific_round": scientific_round, "transport_retry": network_retries,
                    "material_request_number": requests,
                    "call_id": f"{RUN_ID}|{arm}|{case['id']}|s{scientific_round}|n{requests}",
                    "prompt": prompt, "started_utc": now()}
                append({"event": "call_started", **metadata, "request_hash": digest(prompt)})
                record = call_once(prompt, schema_path=SCHEMA, instructions_path=INSTRUCTIONS,
                                   catalog_path=CATALOG, stop_event=stop, metadata=metadata)
                obj = validate(record)
                record.update(event="call_completed", finished_utc=now(), response_valid=obj is not None, parsed_response=obj)
                append(record)
                with lock:
                    calls.append(record)
                last = record
                if record["status"] == "terminal_usage_limit":
                    with lock:
                        stop_info.update(kind="explicit_usage_limit", errors=record.get("terminal_diagnostics"),
                            recovery_time_message=record.get("recovery_time_message"),
                            recovery_reset_fields=record.get("recovery_reset_fields"), call_id=record["call_id"])
                    stop.set()
                elif record["status"] in INTEGRITY:
                    with lock:
                        stop_info.update(kind="integrity_violation", status=record["status"], call_id=record["call_id"])
                    stop.set()
                if obj is not None or stop.is_set():
                    break
                if record["status"] == "rate_limited_final" and network_retries < 3:
                    delay = DELAYS[network_retries]
                    network_retries += 1
                    log(f"FINAL429 {arm} {case['id']}: wait{delay}s before transport retry{network_retries}/3; same frozen scientific prompt")
                    if stop.wait(delay):
                        break
                    continue
                break
            if obj is not None or stop.is_set() or last is None or last["status"] == "rate_limited_final":
                break
            # Only a non429 format/nonterminal transport repair may use scientific round2.
            if last["status"] not in {"schema_invalid", "cli_error", "backend_error", "incomplete_response", "transport_error"}:
                break
        status = last["status"] if last is not None else "not_dispatched_due_stop"
        return arm, case, {"response": obj, "status": status,
                           "attempts": last["scientific_round"] if last else 0,
                           "transport_retries": network_retries, "CLI_requests": requests,
                           "finished_monotonic": time.monotonic()}

    try:
        log("F2 HUMAN RESUME v2: retain5valid;4workers;numeric arms first;recover CLI reconnects;1/2/4min max3 transport retries")
        # Remove only obsolete failed rows. Reconstruct unique immutable baseline+retained rows.
        with safe_path(OUT / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for index, case in enumerate(cfg["cases"]):
                for arm, values in baseline.items():
                    value = float(values[index])
                    writer.writerow(dict(id=case["id"], fold=case["fold"], group=case["group"], arm=arm,
                        status="frozen_E2_reused", score=value, y_pred=int(value >= .5)))
                for arm in ARMS:
                    if (arm, case["id"]) in results:
                        writer.writerows(rows_for(arm, case, results[(arm, case["id"])]))
                if (index + 1) % 10 == 0:
                    handle.flush()
                    os.fsync(handle.fileno())
        new_finalized = 0
        for phase, arms in (("numeric_priority", ("A3-M2", "A2-M2")), ("M1", ("A3-M1",))):
            if stop.is_set():
                break
            jobs = [(arm, case) for case in cfg["cases"] for arm in arms if (arm, case["id"]) not in results]
            jobs.sort(key=lambda x: digest("F2-resume-v2|" + x[0] + "|" + x[1]["id"]))
            iterator = iter(jobs)
            log(f"PHASE {phase} pending={len(jobs)}")
            with ThreadPoolExecutor(max_workers=4) as pool:
                active = {}
                for _ in range(min(4, len(jobs))):
                    arm, case = next(iterator)
                    active[pool.submit(work, arm, case)] = (arm, case["id"])
                exhausted = False
                try:
                    while active:
                        ready, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                        for future in sorted(ready, key=lambda f: f.result()[2]["finished_monotonic"]):
                            active.pop(future)
                            arm, case, result = future.result()
                            assert (arm, case["id"]) not in results
                            if result["attempts"] > 0:
                                results[(arm, case["id"])] = result
                                pending_rows.extend(rows_for(arm, case, result))
                                new_finalized += 1
                                flush()
                            if result["response"] is not None:
                                streak.clear()
                            elif result["attempts"] > 0 and not stop.is_set():
                                if case["id"] not in streak:
                                    streak.append(case["id"])
                                if len(streak) >= 3:
                                    stop_info.update(kind="three_consecutive_exhausted_materials", material_ids=list(streak))
                                    stop.set()
                            log(f"RESUME {new_finalized}/895 phase={phase} {arm} {case['id']} {result['status']} requests={result['CLI_requests']} retry={result['transport_retries']}")
                            if new_finalized % 10 == 0:
                                write_json(OUT / "metrics.json", {"status": "running_resume_v2", "phase": phase,
                                    "retained_valid": 5, "new_finalized": new_finalized,
                                    "valid_by_arm": {a: sum(k[0] == a and v["response"] is not None for k, v in results.items()) for a in ARMS},
                                    "concurrency": 4, "stop_reason": dict(stop_info)})
                            if not stop.is_set() and not exhausted:
                                try:
                                    next_arm, next_case = next(iterator)
                                    active[pool.submit(work, next_arm, next_case)] = (next_arm, next_case["id"])
                                except StopIteration:
                                    exhausted = True
                except BaseException:
                    stop.set()
                    for future in active:
                        future.cancel()
                    raise
            flush(True)
            if phase == "numeric_priority" and not stop.is_set():
                snapshot("running_numeric_complete")
                log("Numeric priority phase finished; saved its independent common-valid tables before M1")
        valid_n = sum(r["response"] is not None for r in results.values())
        status = "complete" if valid_n == 900 else "stopped_partial" if stop.is_set() else "complete_with_failures"
        flush(True)
        m = snapshot(status)
        resume.update(status=status, completed_utc=now(), valid_decisions=valid_n, stop_reason=dict(stop_info))
        cfg["current_stop_requires_human_input"] = status != "complete"
        write_json(OUT / "config.json", cfg)
        log(f"F2 RESUME {status};valid={valid_n}/900;scientific_requests_total={len(calls)};wall={m['wall_seconds']:.2f}s;stop={stop_info};no Stage3")
        return 0 if status == "complete" else 75
    except BaseException:
        stop.set()
        flush(True)
        snapshot("interrupted_resume_error")
        resume.update(status="interrupted_resume_error", completed_utc=now(), stop_reason=dict(stop_info))
        cfg["current_stop_requires_human_input"] = True
        write_json(OUT / "config.json", cfg)
        log("FATAL resume error; progress preserved; " + traceback.format_exc())
        raise
    finally:
        log_handle.close()
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("configure", "run"))
    parser.add_argument("--task-name", required=True)
    args = parser.parse_args()
    try:
        if args.action == "configure":
            configure(args.task_name)
        else:
            raise SystemExit(run(args.task_name))
    except Exception:
        with safe_path(OUT / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(now() + " resume startup/runtime failure\n" + traceback.format_exc())
        raise
