"""Finite G4 input ablation; preparation never starts an evaluated LLM call."""
from common import ROOT, FIELDS, safe_path, write_json, load_data, np
from g4_inputs import (OUT, ARMS, SEED, SCHEMA_PATH, INSTRUCTIONS, CATALOG, SCHEMA,
                       sha, hashed, prompt, validate, frozen_inputs)
from g4_stats import statistics
from f2_transport_v2 import call_once, resolve_runtime
from e2_transport import input_budget, build_command
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import argparse
import csv
import ctypes
import json
import os
import subprocess
import threading
import time
import traceback

EXE = r'C:\Users\28908\AppData\Local\OpenAI\Codex\bin\896a0166b774037a\codex.exe'
MAX_CALLS = 10000
DELAYS = (60, 120, 240)
INTEGRITY = {'tool_activity_rejected', 'model_mismatch_rejected', 'backend_protocol_rejected',
             'preflight_rejected', 'actual_input_budget_violation'}
COLUMNS = FIELDS + ['cohort', 'G3_family', 'abstain']
SOURCES = ('g4_run.py', 'g4_inputs.py', 'g4_stats.py', 'g4_launch.ps1', 'common.py',
           'e2_prompts.py', 'e2_packets.py', 'e2_response.schema.json', 'e2_instructions.md',
           'f3_inputs.py', 'g3_inputs.py', 'f_stats.py', 'f_report.py', 'f3_stats.py',
           'f2_transport_v2.py', 'f1_transport.py', 'e2_transport.py', 'e2_model_catalog.json')


def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(path.read_text(encoding='utf-8'))


def readme(status):
    lines = ['# G4 input ablation (development results)', f'Status: {status}.',
        'Fixed E2 300 plus500 newly sampled materials; original IDs/groups/three folds unchanged.',
        'New500 selection sees only frozen G3 predicted family, original group size and IDs, never measured labels/Tc.',
        'Minimum1/nonempty family-size stratum; remaining quotas use Hamilton residual-capacity allocation and seed20261018 SHA ranks.',
        'E supplies only element set; F supplies only original proxy-cell formula.',
        'P retains full original composition/structure; only formula string and user-authorized formula_semantics change.',
        'P uses deterministic element-order permutation and exact whole-formula factor1/2 or2; composition occupancies never change.',
        'P original identity can still be reconstructed from the composition table; this is representation stability, not masking.',
        'Original A3 probabilities are read from F3, including reused E2 fixed300 outcomes; no A3 repeat calls.',
        'Original A3 instruction/schema retained; initial prompt scientific payload whitelist only, no IDs/family labels/truth.',
        'Fresh local codex exec gpt-6.1-sol/high; tools and web disabled; no alternate or professor API.',
        'One scientific E probe, then4 concurrent; tasks interleave E/F/P within deterministic material order.',
        'CLI Reconnecting awaited internally; final429 retries same prompt after1/2/4minutes, at most3 extra per case.',
        'Explicit usage-limit or3 consecutive distinct exhausted materials stops and saves; hard10000reservations across runs.',
        'Finite target2400new arm-cases, at most2scientific rounds; one diagnosed whole recomputation maximum.',
        'One-shot task PT0S/no restart/no triggers; SetThreadExecutionState; append/fsync every10completed arm-cases.',
        'Both weights/four metrics/three subsets, original families/sizes and all12G3 labels; 5000paired group bootstrap.',
        'Report800/fixed300/new500 separately; retain original full-group weights; sparse/empty strata explicit.',
        'Six audit files only; old experiments read-only; no later stage starts from this runner.']
    assert len(lines) <= 20
    safe_path(OUT / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def initial_csv(cfg):
    with safe_path(OUT / 'predictions.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS); writer.writeheader()
        for j, case in enumerate(cfg['cases'], 1):
            source = cfg['baseline_A3_rows'][case['id']]
            row = {key: source[key] for key in FIELDS}
            row.update(arm='A3_original', cohort=case['cohort'], G3_family=case['G3_family'],
                       abstain=str(source['status'].startswith('abstain')).lower())
            writer.writerow(row)
            if j % 10 == 0: handle.flush(); os.fsync(handle.fileno())
        handle.flush(); os.fsync(handle.fileno())


def prepare(task):
    assert not OUT.exists(), 'G4 already exists; inspect rather than overwrite'
    for ancestor in (ROOT, *ROOT.parents):
        assert not ancestor.is_symlink() and not ancestor.is_junction()
    assert 'onedrive' not in str(OUT.resolve()).lower()
    cfg = frozen_inputs()
    assert read(SCHEMA_PATH) == SCHEMA
    os.environ['F1_CODEX_EXE'] = EXE
    runtime = resolve_runtime()
    fixed_budget = input_budget('', schema_path=SCHEMA_PATH, instructions_path=INSTRUCTIONS)
    bounds, repair_bounds = [], []
    forbidden = {'id', 'i', 'fold', 'group', 'y', 'tc', 'Tc_K', 'label', 'target_K', 'G3_family', 'cohort', 'score', 'p'}
    def inspect(value):
        if isinstance(value, dict):
            assert not (set(value) & forbidden), set(value) & forbidden
            for item in value.values(): inspect(item)
        elif isinstance(value, list):
            for item in value: inspect(item)
    for case in cfg['cases']:
        for arm in ARMS:
            inspect(case['payloads'][arm])
            for repair, collection in ((False, bounds), (True, repair_bounds)):
                message = prompt(case, arm, repair)
                collection.append(fixed_budget['input_token_upper_bound'] + len(message.encode('utf-8')))
                if case is cfg['cases'][0]:
                    actual = input_budget(message, schema_path=SCHEMA_PATH, instructions_path=INSTRUCTIONS)
                    assert collection[-1] == actual['input_token_upper_bound']
            normalized = validate({'action': 'decide', 'p': .5, 'abstain': True,
                'selected_rule_ids': [], 'request_rule_ids': [], 'request_counterexamples': False,
                'reason': 'synthetic validator-only check'}, case, arm)
            assert normalized['p'] == .5 and normalized['abstain'] is True
    assert len(bounds) == len(repair_bounds) == 2400 and max(bounds + repair_bounds) <= 15000
    cfg.update(experiment='G4_input_ablation', status='prepared', created_utc=now(), task_name=task,
        runtime=runtime, source_hashes={name: sha(ROOT / name) for name in SOURCES},
        git_commit_at_prepare=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        runs=[], results={arm: {} for arm in ARMS},
        protocol={'target_new_arm_cases': 2400, 'max_calls': MAX_CALLS,
            'cap_interpretation': 'Execution ceiling against accidental call inflation, not a target; includes reservations from any diagnosed whole recomputation. CLI-internal Reconnecting is not a new reservation.',
            'concurrency': 4, 'first_scientific_case_is_probe': True, 'max_scientific_rounds': 2,
            'final429_delays_seconds': list(DELAYS), 'max_additional_429_retries_per_arm_material': 3,
            'max_process_restarts': 1, 'n_boot': 5000, 'seed': SEED,
            'ExecutionTimeLimit': 'PT0S', 'RestartCount': 0, 'triggers': 0,
            'job_order': 'Sort selected material IDs by SHA256(G4|seed|case-order|id); E,F,P adjacent per material.'},
        preflight={'real_LLM_calls': 0, 'all_800_original_A3_prompt_hashes_match_F3': True,
            'all_800_P_composition_structure_exact_after_original_8significant_digit_display': True,
            'all_P_formula_element_ratios_exact_Fraction': True,
            'P_changed_fields_only': ['formula', 'formula_semantics'],
            'E_only_field': 'element_set', 'F_only_fields': ['formula', 'formula_semantics'],
            'no_ID_fold_group_G3_family_truth_in_scientific_payload': True,
            'schema_matches_original_A3': True,
            'initial_input_upper_bound_min': min(bounds), 'initial_input_upper_bound_max': max(bounds),
            'repair_input_upper_bound_max': max(repair_bounds),
            'byte_budget_cached_fixed_overhead_verified_on_first_case_all_arms_both_rounds': True,
            'command': build_command(schema_path=SCHEMA_PATH, instructions_path=INSTRUCTIONS, catalog_path=CATALOG)})
    ordered = sorted(cfg['cases'], key=lambda c: hashed(f'G4|{SEED}|case-order|{c["id"]}'))
    cfg['job_order'] = [{'id': case['id'], 'arm': arm} for case in ordered for arm in ARMS]
    safe_path(OUT).mkdir()
    for filename in ('run.log', 'llm_calls.jsonl'): safe_path(OUT / filename).write_text('', encoding='utf-8')
    write_json(OUT / 'config.json', cfg); initial_csv(cfg); readme('prepared')
    write_json(OUT / 'metrics.json', {'status': 'prepared_LLM_not_started', 'requested_materials': 800,
        'fixed300': 300, 'new500': 500, 'requested_new_arm_cases': 2400, 'model_calls': 0,
        'new_valid_by_arm': {arm: 0 for arm in ARMS}})
    with safe_path(OUT / 'run.log').open('a', encoding='utf-8') as handle:
        handle.write(now() + ' PREPARED only; no LLM call dispatched; await root review/start.\n')
    print(json.dumps({'status': cfg['status'], 'n': len(cfg['cases']), 'runtime': runtime,
        'selection_nonempty_strata': sum(row['remaining_population_n'] > 0 for row in cfg['selection']['strata']),
        'preflight': {key: value for key, value in cfg['preflight'].items() if key != 'command'}}, ensure_ascii=False))


def run(task, restart_index):
    cfg, d = read(OUT / 'config.json'), load_data()
    assert cfg['task_name'] == task and restart_index == len(cfg['runs']) and restart_index <= 1
    assert cfg['status'] == 'prepared' or (restart_index == 1 and cfg['status'] == 'interrupted_error')
    os.environ['F1_CODEX_EXE'] = cfg['runtime']['executable']
    assert resolve_runtime() == cfg['runtime']
    for filename, expected in cfg['source_hashes'].items(): assert sha(ROOT / filename) == expected, filename
    for filename, expected in cfg['scientific_sources'].items(): assert sha(filename) == expected, filename
    with (OUT / 'llm_calls.jsonl').open(encoding='utf-8') as handle:
        reserved = sum(json.loads(line).get('event') == 'call_started' for line in handle)
    assert reserved <= MAX_CALLS
    meta = {'index': restart_index, 'started_utc': now(), 'reservations_before': reserved}
    cfg['runs'].append(meta); cfg.update(status='running', results={arm: {} for arm in ARMS})
    write_json(OUT / 'config.json', cfg)
    assert ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    lock, stop, budget = threading.RLock(), threading.Event(), threading.Event()
    calls, results, pending, streak, stop_info = [], {arm: {} for arm in ARMS}, [], [], {}
    started = time.monotonic()
    def log(message):
        with lock, safe_path(OUT / 'run.log').open('a', encoding='utf-8') as f: f.write(now() + ' ' + message + '\n')
    def append(record):
        with lock, safe_path(OUT / 'llm_calls.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
            f.flush(); os.fsync(f.fileno())
    def reserve(record):
        nonlocal reserved
        with lock:
            if reserved >= MAX_CALLS:
                budget.set(); stop_info.setdefault('kind', 'global_call_budget_exhausted'); return None
            if stop.is_set(): return None
            reserved += 1
            record.update(call_id=reserved, availability_probe=reserved == meta['reservations_before'] + 1)
            append({'event': 'call_started', **record}); return record
    def flush(force=False):
        if len(pending) >= 10 or (force and pending):
            with safe_path(OUT / 'predictions.csv').open('a', encoding='utf-8', newline='') as f:
                csv.DictWriter(f, fieldnames=COLUMNS).writerows(pending); f.flush(); os.fsync(f.fileno())
            pending.clear()
    def work(case, arm):
        response, last, retries, requests = None, None, 0, 0
        for science_round in (1, 2):
            message = prompt(case, arm, repair=science_round == 2)
            if science_round == 1: assert hashed(message) == case['prompt_hashes'][arm]
            while not stop.is_set():
                record = reserve({'experiment': 'G4_input_ablation', 'run_index': restart_index,
                    'arm': arm, 'id': case['id'], 'scientific_round': science_round,
                    'transport_retry': retries, 'prompt': message, 'request_hash': hashed(message), 'started_utc': now()})
                if record is None: break
                requests += 1
                record = call_once(message, schema_path=SCHEMA_PATH, instructions_path=INSTRUCTIONS,
                    catalog_path=CATALOG, stop_event=stop, metadata=record)
                if record['status'] == 'ok':
                    try: response = validate(json.loads(record['response_text']), case, arm)
                    except (ValueError, TypeError, KeyError) as exc:
                        record.update(status='protocol_error', protocol_error=str(exc))
                record.update(event='call_completed', finished_utc=now(), response_valid=response is not None, parsed_response=response)
                append(record)
                with lock: calls.append(record)
                last = record
                if record['status'] == 'terminal_usage_limit':
                    with lock: stop_info.update(kind='explicit_usage_limit', errors=record.get('terminal_diagnostics'),
                        recovery_time_message=record.get('recovery_time_message'), recovery_reset_fields=record.get('recovery_reset_fields'), call_id=record['call_id'])
                    stop.set()
                elif record['status'] in INTEGRITY:
                    with lock: stop_info.update(kind='integrity_violation', status=record['status'], call_id=record['call_id'])
                    stop.set()
                if response is not None or stop.is_set(): break
                if record['status'] == 'rate_limited_final' and retries < 3:
                    delay = DELAYS[retries]; retries += 1
                    log(f'FINAL429 {arm}/{case["id"]} wait={delay}s identical-prompt retry={retries}/3')
                    if stop.wait(delay): break
                    continue
                break
            if response is not None or stop.is_set() or budget.is_set() or last is None or last['status'] == 'rate_limited_final': break
            if last['status'] not in {'protocol_error', 'cli_error', 'backend_error', 'incomplete_response', 'transport_error'}: break
        return case, arm, {'response': response, 'status': last['status'] if last else 'not_dispatched',
            'requests': requests, 'retries': retries, 'finished_monotonic': time.monotonic()}
    def snapshot(status, final=False):
        cfg['results'] = {arm: {identity: value['response'] for identity, value in values.items() if value['response'] is not None}
                          for arm, values in results.items()}
        cfg['status'] = status; write_json(OUT / 'config.json', cfg)
        m = {'status': status, 'new_valid_by_arm': {arm: len(v) for arm, v in cfg['results'].items()},
             'new_finalized': sum(map(len, results.values())), 'requested_new_arm_cases': 2400,
             'reserved_across_runs': reserved, 'stop_reason': dict(stop_info),
             'new_calls': len(calls), 'wall_seconds': time.monotonic() - started}
        if final:
            report = statistics(cfg, d, results); report.update(m); m = report
            m['calls'] = {'reserved_across_runs': reserved, 'completed_this_run': len(calls),
                'additional_429_retries': sum(v.get('transport_retry', 0) > 0 for v in calls),
                'protocol_repairs': sum(v['scientific_round'] == 2 for v in calls),
                'tools_used': sum(len(v.get('tool_activity', [])) for v in calls), 'runtime': cfg['runtime']}
        write_json(OUT / 'metrics.json', m); readme(status)
    try:
        initial_csv(cfg)
        log(f'START G4 run={restart_index} materials=800 arm_cases=2400 cap={MAX_CALLS} concurrency=4')
        lookup = {case['id']: case for case in cfg['cases']}
        iterator = iter(cfg['job_order']); first = next(iterator)
        with ThreadPoolExecutor(max_workers=4) as pool:
            active = {pool.submit(work, lookup[first['id']], first['arm']): None}
            exhausted, probe_confirmed = False, False
            while active:
                ready, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                for future in sorted(ready, key=lambda f: f.result()[2]['finished_monotonic']):
                    active.pop(future); case, arm, result = future.result()
                    if result['requests']:
                        results[arm][case['id']] = result
                        response = result['response']
                        pending.append({'id': case['id'], 'fold': case['fold'], 'group': case['group'], 'arm': arm,
                            'status': ('abstain_forced_backup' if response['abstain'] else 'ok') if response else result['status'],
                            'score': response['p'] if response else '', 'y_pred': int(response['p'] >= .5) if response else '',
                            'cohort': case['cohort'], 'G3_family': case['G3_family'],
                            'abstain': str(response['abstain']).lower() if response else ''})
                        flush()
                    if result['response'] is not None: streak.clear(); probe_confirmed = True
                    elif result['requests'] and not stop.is_set() and not budget.is_set():
                        # Distinct material IDs: repeated arms of one material do not manufacture a three-material streak.
                        if case['id'] not in streak: streak.append(case['id'])
                        if len(streak) >= 3:
                            stop_info.update(kind='three_consecutive_distinct_exhausted_materials', material_ids=streak[-3:]); stop.set()
                    count = sum(map(len, results.values()))
                    log(f'PROGRESS {count}/2400 {arm}/{case["id"]} {result["status"]} requests={result["requests"]} retries={result["retries"]} total_reserved={reserved}')
                    if count % 10 == 0: snapshot('running')
                while not stop.is_set() and not budget.is_set() and not exhausted and len(active) < (4 if probe_confirmed else 1):
                    try:
                        job = next(iterator); active[pool.submit(work, lookup[job['id']], job['arm'])] = None
                    except StopIteration: exhausted = True
        flush(True)
        valid = sum(result['response'] is not None for values in results.values() for result in values.values())
        status = 'complete' if valid == 2400 else 'stopped_partial' if stop.is_set() or budget.is_set() else 'complete_with_failures'
        snapshot(status, final=True)
        meta.update(finished_utc=now(), reservations_end=reserved, status=status, stop_reason=dict(stop_info))
        write_json(OUT / 'config.json', cfg)
        log(f'END status={status} valid={valid}/2400 reserved={reserved} stop={stop_info}')
        return 0 if status == 'complete' else 75
    except BaseException:
        stop.set(); flush(True)
        meta.update(status='interrupted_error', finished_utc=now(), reservations_end=reserved)
        snapshot('interrupted_error', final=False); log('FATAL ' + traceback.format_exc()); raise
    finally: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=('prepare', 'run'))
    p.add_argument('--task-name', required=True); p.add_argument('--restart-index', type=int, default=0)
    a = p.parse_args()
    if a.action == 'prepare': prepare(a.task_name)
    else: raise SystemExit(run(a.task_name, a.restart_index))
