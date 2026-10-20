"""Finite full-cohort A2 then A3 evaluation, with F2's authorized transport policy."""
from __future__ import annotations

from common import ROOT, DATA, FIELDS, safe_path, write_json, load_data, np
from f1_inputs import sha256_file
from f2_transport_v2 import call_once, resolve_runtime
from f3_inputs import build_inputs, prompt_for, validate_output
from f3_stats import build_f3_statistics
from f3_report import build_f3_report
from f_stats import fraction_summary
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import argparse
import csv
import ctypes
import hashlib
import json
import os
import subprocess
import threading
import time
import traceback

OUT = ROOT / 'F3_full3764'
ARMS = ('A2', 'A3')
DELAYS = (60, 120, 240)
MAX_CALLS = 10000
SCHEMA, INSTRUCTIONS, CATALOG = [ROOT / n for n in ('e2_response.schema.json', 'e2_instructions.md', 'e2_model_catalog.json')]
INTEGRITY = {'tool_activity_rejected', 'model_mismatch_rejected', 'backend_protocol_rejected', 'preflight_rejected', 'actual_input_budget_violation'}
SOURCES = ('run_f3.py', 'launch_f3.ps1', 'f3_inputs.py', 'f3_stats.py', 'f3_report.py',
           'f2_transport_v2.py', 'f1_transport.py', 'f1_inputs.py', 'e2_transport.py', 'e2_packets.py',
           'e2_prompts.py', 'e2_instructions.md', 'e2_response.schema.json', 'e2_model_catalog.json',
           'e1_rules.py', 'common.py', 'f_stats.py', 'f_report.py')


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def repair_diagnostic(value):
    text = value.encode('ascii', errors='backslashreplace').decode('ascii')[:240]
    while len(json.dumps(text, ensure_ascii=False).encode('utf-8')) - 2 > 240:
        text = text[:-1]
    return text


def write_readme(status, counts=None):
    lines = ['# F3 full3764 development evaluation', f'Status: {status}; counts: {counts or {}}.',
      'Full3764/1117original groups; A2 first, then A3; each arm reuses fixed300 E2 predictions unchanged.',
      'Same E2 scientific packets, original prompts/instructions/schema, gpt-6.1-sol/high, tools and network disabled.',
      'First pending A2 material confirms availability, then four concurrent local logged-in calls; no alternative API.',
      '6928initial pending arm-cases; global cap10000 CLI reservations including repairs/retries/restarts.',
      'CLI Reconnecting runs to completion; recovered429 is successful when a valid final decision exists.',
      'Final429 retries identical material/prompt after1,2,4minutes; max3additional retries shared across scientific rounds.',
      'At most2scientific rounds, second only same-evidence protocol repair; first valid forced p retained even if abstain.',
      'Explicit usage-limit/try-again-at stops immediately; otherwise3consecutive distinct exhausted materials stop.',
      'One-shot scheduler has no triggers/runtime cap/restarts; SetThreadExecutionState prevents sleep.',
      'CSV appends/fsyncs every10completed arm-cases; no fabricated probabilities for failed/cancelled calls.',
      'Process interruption requires system/task-log diagnosis before at most1full recomputation; total call cap persists.',
      'HGB127, frozen E0 StrongML OOF and identical outer-TRAIN5NN; no new feature/model tuning.',
      'Both original-group inverse and material weights: BA/precision/recall/accuracy,5000paired group bootstrap.',
      'All/exclude93/only93 plus five families and original group-size bins1,2-5,6-20,>20; n/groups explicit.',
      'Only93 is one group; intervals degenerate. A15 requires declared SG223 and binary A3B ratio; overlap recorded.',
      'Main tables use all-method common valid materials; independent A2-priority tables retained if A3 is incomplete.',
      'Full tables and coverage: metrics.json/report_markdown; all previous experiment directories remain read-only.',
      'No full recall or automatic route selection. F4 collection is separate; its evaluation never overlaps F3.']
    assert len(lines) <= 20
    safe_path(OUT / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def reused_results(cfg):
    results = {}
    for arm in ARMS:
        for key, row in cfg['reused_predictions'][arm + '_forced'].items():
            results[(arm, key)] = {'response': {'p': float(row['score']), 'abstain': 'abstain' in row['status'].lower()},
                                  'status': 'frozen_E2_reused', 'attempts': 0, 'reused': True}
    assert len(results) == 600
    return results


def initial_csv(cfg):
    with safe_path(OUT / 'predictions.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for j, key in enumerate(cfg['case_order']):
            case = cfg['cases'][key]
            for arm, values in cfg['baselines'].items():
                p = float(values[j])
                writer.writerow(dict(id=key, fold=case['fold'], group=case['group'], arm=arm,
                    status='frozen_OOF_or_TRAIN5NN', score=p, y_pred=int(p >= .5)))
            for arm in ARMS:
                row = cfg['reused_predictions'][arm + '_forced'].get(key)
                if row is not None:
                    # Preserve every original scientific CSV value; provenance status is separate in config.
                    writer.writerow({field: row[field] for field in FIELDS})
            if (j + 1) % 10 == 0:
                handle.flush()
                os.fsync(handle.fileno())


def prepare(task_name):
    assert not OUT.exists(), 'F3 exists; inspect state instead of overwriting/restarting'
    for ancestor in (ROOT, *ROOT.parents):
        assert not ancestor.is_symlink() and not ancestor.is_junction()
    cfg = build_inputs(log=lambda s: print(s, flush=True))
    cfg.update(experiment='F3_full3764', scope='development only; no new-material external evaluation',
        created_utc=now(), status='prepared', task_name=task_name,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        runtime=resolve_runtime(), source_hashes={n: sha256_file(ROOT / n) for n in SOURCES},
        params={'n': 3764, 'n_groups': 1117, 'arms': list(ARMS), 'arm_order': list(ARMS), 'reused_E2_per_arm': 300,
            'pending_material_arm_cases': 6928, 'concurrency': 4, 'global_CLI_reservation_cap': MAX_CALLS,
            'max_scientific_rounds': 2, 'max_additional_429_retries': 3, 'transport_retry_wait_seconds': list(DELAYS),
            'first_A2_request_is_availability_probe_and_scientific_result': True,
            'unused_delay_seconds': [480, 900], 'max_process_restarts': 1, 'bootstrap_draws': 5000, 'seed': 20261016,
            'prediction_flush_every_arm_cases': 10, 'ExecutionTimeLimit': 'PT0S', 'RestartCount': 0}, runs=[])
    cfg['case_order'] = [str(cfg['cases'][key]['id']) for key in cfg['cases']]
    assert len(cfg['case_order']) == 3764 and len(set(cfg['case_order'])) == 3764
    assert [cfg['cases'][key]['i'] for key in cfg['case_order']] == list(range(3764))
    cfg['scientific_sources'] = {**cfg['metadata']['scientific_source_hashes'], str(DATA): sha256_file(DATA), **{str(ROOT / 'E2' / n): sha256_file(ROOT / 'E2' / n)
        for n in ('config.json', 'predictions.csv', 'metrics.json')},
        **{str(ROOT / 'E0' / n): sha256_file(ROOT / 'E0' / n) for n in ('config.json', 'predictions.csv', 'metrics.json')}}
    safe_path(OUT).mkdir()
    write_json(OUT / 'config.json', cfg)
    for name in ('llm_calls.jsonl', 'run.log'):
        safe_path(OUT / name).write_text('', encoding='utf-8')
    initial_csv(cfg)
    write_json(OUT / 'metrics.json', {'status': 'prepared', 'n': 3764, 'reused_valid': 600, 'pending': 6928})
    write_readme('prepared')
    print(json.dumps({'status': 'prepared', 'n': 3764, 'pending': 6928, 'runtime': cfg['runtime'],
                      'metadata_keys': list(cfg['metadata'])}, ensure_ascii=False), flush=True)


def finish(cfg, results, calls, status, run_index, seconds, stop_info, reserved):
    cases = [cfg['cases'][key] for key in cfg['case_order']]
    y = np.asarray([case['evaluation']['y'] for case in cases])
    groups = np.asarray([case['group'] for case in cases])
    sizes = {case['group']: case['evaluation']['original_group_size'] for case in cases}
    scores = {arm + '_forced': np.asarray([results.get((arm, case['id']), {}).get('response', None)['p']
        if results.get((arm, case['id']), {}).get('response') is not None else np.nan for case in cases]) for arm in ARMS}
    scores.update({name: np.asarray(values) for name, values in cfg['baselines'].items()})
    kwargs = {'family_masks': cfg['family_masks'], 'group_size_values': cfg['group_size_values'], 'a15_mask': cfg['raw_a15_mask']}
    metrics = build_f3_statistics(y, scores, groups, sizes, **kwargs)
    metrics.update(status=status, experiment='F3_full3764', scope='development only', method_order=list(scores),
        run_index=run_index, completed_utc=now(), wall_seconds=seconds, calls_reserved_across_runs=reserved,
        stop_reason=dict(stop_info), runtime=cfg['runtime'], params=cfg['params'],
        fixed300_reuse={'A2_forced': 300, 'A3_forced': 300, 'source': 'E2/predictions.csv; exact fields copied'},
        family_definition=cfg['metadata']['family_definitions'],
        provenance=cfg.get('metadata', {}), case_counts={}, calls={'new_completed_requests': len(calls),
            'by_arm': {arm: sum(r['arm'] == arm for r in calls) for arm in ARMS},
            'second_round_requests': sum(r['scientific_round'] == 2 for r in calls),
            'additional_429_retries': sum(r.get('transport_retry', 0) > 0 for r in calls),
            'input_tokens': sum(r.get('input_tokens') or 0 for r in calls),
            'output_tokens': sum(r.get('output_tokens') or 0 for r in calls),
            'tools_used': sum(len(r.get('tool_activity') or []) for r in calls)},
        F4_evaluation_started=False)
    masks = {'all': np.ones(len(y), dtype=bool), 'exclude_93': groups != '93', 'only_93': groups == '93'}
    metrics['rates'] = {name: {} for name in masks}
    for arm in ARMS:
        rs = [results.get((arm, case['id']), {}) for case in cases]
        valid = np.asarray([r.get('response') is not None for r in rs])
        done = np.asarray([bool(r.get('reused')) or r.get('attempts', 0) > 0 for r in rs])
        metrics['case_counts'][arm] = {'valid': int(valid.sum()), 'reused_E2': 300, 'new_valid': int(valid.sum()) - 300,
            'failed_or_cancelled': int((done & ~valid).sum()), 'not_attempted': int((~done).sum()),
            'abstain': sum(bool(r['response'].get('abstain')) for r in rs if r.get('response') is not None)}
        values = {'abstain': np.asarray([float(r['response']['abstain']) if r.get('response') else np.nan for r in rs]),
                  'failure': np.where(done, (~valid).astype(float), np.nan), 'not_attempted': (~done).astype(float)}
        for name, mask in masks.items():
            metrics['rates'][name][arm + '_forced'] = {}
            for event, vector in values.items():
                stat = fraction_summary(vector, groups, sizes, mask)
                stat['count'] = int(np.sum(np.isfinite(vector[mask]) & (vector[mask] == 1)))
                metrics['rates'][name][arm + '_forced'][event] = stat
    if metrics['case_counts']['A3']['valid'] < 3764:
        metrics['A2_priority'] = build_f3_statistics(y, {k: v for k, v in scores.items() if k != 'A3_forced'}, groups, sizes, **kwargs)
    metrics['report_markdown'] = build_f3_report(metrics)
    write_json(OUT / 'metrics.json', metrics)
    write_readme(status, metrics['case_counts'])
    return metrics


def run(task_name, restart_index):
    cfg = read_json(OUT / 'config.json')
    assert cfg['task_name'] == task_name and restart_index == len(cfg['runs']) and restart_index <= 1
    assert cfg['status'] == 'prepared' or (restart_index == 1 and cfg['status'] == 'interrupted_error')
    os.environ['F1_CODEX_EXE'] = cfg['runtime']['executable']
    assert resolve_runtime() == cfg['runtime']
    for name, expected in cfg['source_hashes'].items():
        assert sha256_file(ROOT / name) == expected, name
    for name, expected in cfg['scientific_sources'].items():
        from pathlib import Path
        assert sha256_file(Path(name)) == expected, name
    # Count old reservations only; do not recover or select prior scientific answers after process interruption.
    with (OUT / 'llm_calls.jsonl').open(encoding='utf-8') as handle:
        reserved = sum(json.loads(line).get('event') == 'call_started' for line in handle)
    assert reserved <= MAX_CALLS
    results, calls = reused_results(cfg), []
    run_meta = {'index': restart_index, 'started_utc': now(), 'calls_reserved_before_run': reserved,
        'policy': 'reuse E2 only; prior F3 interrupted predictions recomputed; global call cap persists'}
    cfg['runs'].append(run_meta)
    cfg['status'] = 'running'
    write_json(OUT / 'config.json', cfg)
    assert ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    lock, stop, budget_exhausted = threading.RLock(), threading.Event(), threading.Event()
    stop_info, streak, pending_rows = {}, [], []
    log_handle = safe_path(OUT / 'run.log').open('a', encoding='utf-8', buffering=1)
    started = time.monotonic()

    def log(message):
        with lock:
            log_handle.write(now() + ' ' + message + '\n')

    def append(record):
        with lock, safe_path(OUT / 'llm_calls.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
            handle.flush()
            os.fsync(handle.fileno())

    def reserve(metadata):
        nonlocal reserved
        with lock:
            if reserved >= MAX_CALLS:
                budget_exhausted.set()
                stop_info.setdefault('kind', 'global_call_budget_exhausted')
                return None
            if stop.is_set():
                return None
            reserved += 1
            metadata.update(call_id=reserved, request_hash=digest(metadata['prompt']), availability_probe=reserved == 1)
            append({'event': 'call_started', **metadata})
            return metadata

    def flush(force=False):
        if len(pending_rows) >= 10 or (force and pending_rows):
            with safe_path(OUT / 'predictions.csv').open('a', newline='', encoding='utf-8') as handle:
                csv.DictWriter(handle, fieldnames=FIELDS).writerows(pending_rows)
                handle.flush()
                os.fsync(handle.fileno())
            pending_rows.clear()

    def work(arm, case):
        last, obj, retries, requests, history = None, None, 0, 0, []
        for scientific_round in (1, 2):
            prompt = prompt_for(case, arm, history)
            if scientific_round == 1:
                assert digest(prompt) == case['prompt_hashes'][arm]
            while not stop.is_set():
                metadata = reserve({'experiment': 'F3_full3764', 'run_index': restart_index, 'arm': arm, 'id': case['id'],
                    'scientific_round': scientific_round, 'transport_retry': retries,
                    'material_request_number': requests + 1, 'prompt': prompt, 'started_utc': now()})
                if metadata is None:
                    break
                requests += 1
                record = call_once(prompt, schema_path=SCHEMA, instructions_path=INSTRUCTIONS,
                                   catalog_path=CATALOG, stop_event=stop, metadata=metadata)
                failure = record['status']
                if record['status'] == 'ok':
                    try:
                        obj = validate_output(json.loads(record['response_text']), case, arm)
                        assert obj is not None
                    except (ValueError, TypeError, KeyError, AssertionError) as exc:
                        failure = str(exc)
                        record.update(status='protocol_error', protocol_error=failure)
                record.update(event='call_completed', finished_utc=now(), response_valid=obj is not None, parsed_response=obj)
                append(record)
                with lock:
                    calls.append(record)
                last = record
                if record['status'] == 'terminal_usage_limit':
                    with lock:
                        stop_info.update(kind='explicit_usage_limit', errors=record.get('terminal_diagnostics'),
                            recovery_time_message=record.get('recovery_time_message'),
                            recovery_reset_fields=record.get('recovery_reset_fields'), call_id=record['call_id'])
                    stop.set()
                elif record['status'] in INTEGRITY:
                    with lock:
                        stop_info.update(kind='integrity_violation', status=record['status'], call_id=record['call_id'])
                    stop.set()
                if obj is not None or stop.is_set():
                    break
                if record['status'] == 'rate_limited_final' and retries < 3:
                    delay = DELAYS[retries]
                    retries += 1
                    log(f'FINAL429 {arm} {case["id"]}:wait{delay}s before identical-prompt retry{retries}/3')
                    if stop.wait(delay):
                        break
                    continue
                break
            if obj is not None or stop.is_set() or budget_exhausted.is_set() or last is None or last['status'] == 'rate_limited_final':
                break
            if last['status'] not in {'protocol_error', 'cli_error', 'backend_error', 'incomplete_response', 'transport_error'}:
                break
            # Operational diagnostics only: bounded ASCII avoids an impossible worst-case Unicode repair envelope.
            repair_error = repair_diagnostic(failure)
            history.append({'protocol_error': repair_error, 'instruction': 'Repair format/decision using exactly the same evidence.'})
        return arm, case, {'response': obj, 'status': last['status'] if last else 'not_dispatched_due_stop_or_budget',
            'attempts': last['scientific_round'] if last else 0, 'CLI_requests': requests,
            'transport_retries': retries, 'finished_monotonic': time.monotonic()}

    def snapshot(status):
        return finish(cfg, results, calls, status, restart_index, time.monotonic() - started, stop_info, reserved)

    try:
        initial_csv(cfg)
        log(f'F3 RUN{restart_index}:6928pending;A2 then A3;4workers;600frozen E2 decisions reused;cap10000 across runs')
        finalized = 0
        for arm in ARMS:
            if stop.is_set() or budget_exhausted.is_set():
                break
            jobs = [cfg['cases'][key] for key in cfg['case_order'] if (arm, key) not in results]
            jobs.sort(key=lambda case: digest('F3|' + arm + '|' + case['id']))
            iterator = iter(jobs)
            log(f'PHASE {arm} pending={len(jobs)}')
            with ThreadPoolExecutor(max_workers=4) as pool:
                initial_workers = 1 if arm == 'A2' else 4
                active = {pool.submit(work, arm, next(iterator)): None for _ in range(min(initial_workers, len(jobs)))}
                exhausted = False
                try:
                    while active:
                        ready, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                        for future in sorted(ready, key=lambda f: f.result()[2]['finished_monotonic']):
                            active.pop(future)
                            method, case, result = future.result()
                            assert (method, case['id']) not in results
                            if result['attempts'] > 0:
                                results[(method, case['id'])] = result
                                p = result['response']['p'] if result['response'] is not None else ''
                                pending_rows.append(dict(id=case['id'], fold=case['fold'], group=case['group'], arm=method + '_forced',
                                    status=('valid_abstain_forced' if result['response']['abstain'] else 'valid_forced') if p != '' else result['status'],
                                    score=p, y_pred=int(p >= .5) if p != '' else ''))
                                finalized += 1
                                flush()
                            if result['response'] is not None:
                                streak.clear()
                            elif result['attempts'] > 0 and not stop.is_set() and not budget_exhausted.is_set():
                                if case['id'] not in streak:
                                    streak.append(case['id'])
                                if len(streak) >= 3:
                                    stop_info.update(kind='three_consecutive_exhausted_materials', material_ids=list(streak))
                                    stop.set()
                            log(f'PROGRESS {finalized}/6928 {method} {case["id"]} {result["status"]} requests={result["CLI_requests"]} retries={result["transport_retries"]} total_reserved={reserved}')
                            if finalized % 10 == 0:
                                write_json(OUT / 'metrics.json', {'status': 'running', 'phase': arm, 'new_finalized': finalized,
                                    'calls_reserved_across_runs': reserved, 'stop_reason': dict(stop_info),
                                    'valid_by_arm': {a: sum(k[0] == a and v['response'] is not None for k, v in results.items()) for a in ARMS}})
                            while not stop.is_set() and not budget_exhausted.is_set() and not exhausted and len(active) < 4:
                                try:
                                    active[pool.submit(work, arm, next(iterator))] = None
                                except StopIteration:
                                    exhausted = True
                except BaseException:
                    stop.set()
                    for future in active:
                        future.cancel()
                    raise
            flush(True)
            if arm == 'A2' and not stop.is_set() and not budget_exhausted.is_set():
                snapshot('A2_complete_A3_pending')
                log('A2 phase finished; full A2-versus-baselines tables saved before A3')
        valid_n = sum(r['response'] is not None for r in results.values())
        status = 'complete' if valid_n == 7528 else 'stopped_partial' if stop.is_set() or budget_exhausted.is_set() else 'complete_with_failures'
        flush(True)
        snapshot(status)
        cfg['status'] = status
        run_meta.update(status=status, finished_utc=now(), calls_reserved_end=reserved, valid_decisions=valid_n, stop_reason=dict(stop_info))
        write_json(OUT / 'config.json', cfg)
        log(f'F3 {status};valid={valid_n}/7528;new_calls={len(calls)};total_reserved={reserved};stop={stop_info};no F4 evaluation')
        return 0 if status == 'complete' else 75
    except BaseException:
        stop.set()
        flush(True)
        cfg['status'] = 'interrupted_error'
        run_meta.update(status='interrupted_error', finished_utc=now(), calls_reserved_end=reserved, stop_reason=dict(stop_info))
        write_json(OUT / 'config.json', cfg)
        try:
            snapshot('interrupted_error')
        finally:
            log('FATAL ' + traceback.format_exc())
        raise
    finally:
        log_handle.close()
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--task-name', required=True)
    parser.add_argument('--restart-index', type=int, default=0)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.task_name)
    else:
        raise SystemExit(run(args.task_name, args.restart_index))
