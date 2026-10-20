"""Finite G3 complementarity: nested stacking then fresh formula-only families."""
from common import ROOT, DATA, FIELDS, safe_path, write_json, load_data, np
from g3_inputs import OUT, FAMILIES, SCHEMA, INSTRUCTIONS, CATALOG, sha, prompt, validate, frozen_inputs
from g3_ml import nested_stack, augment_family
from f2_transport_v2 import call_once, resolve_runtime
from f3_stats import build_f3_statistics
from f_report import _classification_section
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

DELAYS = (60, 120, 240)
MAX_CALLS = 8000
EXE = r'C:\Users\28908\AppData\Local\OpenAI\Codex\bin\896a0166b774037a\codex.exe'
INTEGRITY = {'tool_activity_rejected', 'model_mismatch_rejected', 'backend_protocol_rejected', 'preflight_rejected', 'actual_input_budget_violation'}
COLUMNS = FIELDS + ['family_label']
SOURCES = ('g3_run.py', 'g3_inputs.py', 'g3_ml.py', 'g3_family.schema.json', 'g3_instructions.md',
           'g3_launch.ps1', 'common.py', 'f_stats.py', 'f_report.py', 'f3_stats.py',
           'f2_transport_v2.py', 'f1_transport.py', 'e2_transport.py', 'e2_model_catalog.json')

def now(): return datetime.now(timezone.utc).isoformat()
def digest(s): return hashlib.sha256(s.encode('utf8')).hexdigest()
def read(path): return json.loads(path.read_text(encoding='utf8'))

def readme(status):
    lines = ['# G3 complementarity (development results)', f'Status: {status}.',
        'Original3764 materials/1117connected groups; original three outer folds unchanged.',
        'Stacking outerTRAIN uses two remaining original folds exchanged for strictly inner group-OOF HGB probabilities.',
        'Each HGB and scaler sees only its own TRAIN rows; held-fold HGB refits complete outerTRAIN and reproduces frozen original OOF.',
        'LR fixed C1/liblinear; TRAIN group-weight standardization and group/class-balanced loss; no tuning or held-label selection.',
        'Simple mean separately averages frozen original HGB and frozen F3 A3 probabilities.',
        'A3 historical input was formula plus structure. This experiment newly requests formula-only family labels, never Tc.',
        'Twelve exact predeclared labels, including not-superconductor; onehot joined to X127 then same frozen HGB outer-OOF recipe.',
        'Not-superconductor is an LLM prior judgment, not a pure chemical family and never a measured target label.',
        'Fresh logged-in local codex exec gpt-6.1-sol/high; tools and web disabled; no alternate or professor API.',
        'First scientific case probes availability alone; success expands to four concurrent processes.',
        'CLI Reconnecting is awaited; final429 retries identical prompt after1,2,4minutes, at most3additional retries per material.',
        'Explicit usage-limit or three consecutive distinct exhausted materials stops and saves progress.',
        'At most2scientific rounds for JSON format repair; hard8000CLI reservations including retries/restarts.',
        'One-shot task has no runtime cap or restart; SetThreadExecutionState prevents sleep; CSV fsync every10finalized materials.',
        'Interruption requires system/task-log diagnosis; at most1whole recomputation; no old family answers reused.',
        'Both weights/four classification metrics;5000paired group bootstrap; all/exclude93/only93 plus frozen family/size strata.',
        'Full original group weights retained within every subset; one-group intervals degenerate.',
        'Six audit files only; previous experiment directories read-only; no later stage or research-route choice.']
    assert len(lines) <= 20
    safe_path(OUT / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf8')

def append_scores(handle, d, scores):
    writer = csv.DictWriter(handle, fieldnames=COLUMNS)
    for method, values in scores.items():
        for i, p in enumerate(values):
            writer.writerow({'id': str(d['id'][i]), 'fold': int(d['fold'][i]), 'group': str(d['group'][i]),
                             'arm': method, 'status': 'valid_OOF', 'score': float(p), 'y_pred': int(p >= .5), 'family_label': ''})
            if (i + 1) % 10 == 0:
                handle.flush(); os.fsync(handle.fileno())
    handle.flush(); os.fsync(handle.fileno())

def initial_csv(cfg, d):
    with safe_path(OUT / 'predictions.csv').open('w', newline='', encoding='utf8') as handle:
        csv.DictWriter(handle, fieldnames=COLUMNS).writeheader()
        append_scores(handle, d, cfg['scores'])

def statistics(cfg, d, extra=None):
    scores = {name: np.asarray(v) for name, v in cfg['scores'].items()}
    scores.update({name: np.asarray(v) for name, v in (extra or {}).items()})
    sizes = {str(g): int(v) for g, v in zip(d['group'], d['group_size'])}
    m = build_f3_statistics(d['y'], scores, d['group'], sizes,
                           family_masks=cfg['family_masks'], group_size_values=d['group_size'],
                           a15_mask=cfg['raw_a15_mask'], seed=20261018)
    lines = ['# G3互补性开发结果', '',
             '原三折；堆叠中每个outerTRAIN的HGB概率严格来自内部按组OOF，held概率由完整outerTRAIN重新训练。',
             'LR固定C=1、不调参；simple mean单列。A3历史输入包含化学式和结构。',
             '新家族标注只给化学式、固定12类、不问Tc；“不是超导体”可携带先验判断，不能视作纯化学家族。',
             '所有CI为按component组重采样5000次，组加权=1/完整原组大小；不加权也按组抽样。', '']
    sets = [('主比较', m['subsets'])]
    sets += [('家族·' + key, values) for key, values in m['families'].items()]
    sets += [('原组大小·' + key, values) for key, values in m['group_size_bins'].items()]
    for title, values in sets:
        for subset, label in [('all', '全部'), ('exclude_93', '去掉93'), ('only_93', '只看93')]:
            _classification_section(lines, title + '：' + label, values[subset], list(scores), [tuple(p) for p in m['report_pairs']])
    m['report_markdown'] = '\n'.join(lines)
    return m

def prepare(task):
    assert not OUT.exists(), 'G3 exists; inspect before any restart'
    for ancestor in (ROOT, *ROOT.parents):
        assert not ancestor.is_symlink() and not ancestor.is_junction()
    d = load_data()
    cfg = frozen_inputs(d)
    os.environ['F1_CODEX_EXE'] = EXE
    from e2_transport import input_budget, build_command
    budgets = [input_budget(prompt(c['formula']), schema_path=SCHEMA, instructions_path=INSTRUCTIONS) for c in cfg['cases']]
    repair_budgets = [input_budget(prompt(c['formula'], repair=True), schema_path=SCHEMA, instructions_path=INSTRUCTIONS) for c in cfg['cases']]
    assert all(v['within_budget'] for v in budgets + repair_budgets)
    assert read(SCHEMA)['properties']['family']['enum'] == list(FAMILIES)
    cfg.update(experiment='G3_complementarity', status='preparing_ML', task_name=task,
               created_utc=now(), runtime=resolve_runtime(), source_hashes={n: sha(ROOT / n) for n in SOURCES},
               git_commit_at_prepare=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               protocol={'max_calls': MAX_CALLS, 'max_scientific_rounds': 2, 'max_429_retries': 3, 'delays_seconds': list(DELAYS),
                         'concurrency': 4, 'probe_is_first_scientific_case': True, 'max_process_restarts': 1,
                         'n_boot': 5000, 'seed': 20261018, 'ExecutionTimeLimit': 'PT0S', 'RestartCount': 0},
               family_vocabulary=list(FAMILIES), runs=[], family_results={},
               preflight={'all_3764_prompts_formula_only': True, 'no_structure_label_group_fold_in_prompt': True,
                          'initial_input_upper_bound_min': min(v['input_token_upper_bound'] for v in budgets),
                          'initial_input_upper_bound_max': max(v['input_token_upper_bound'] for v in budgets),
                          'repair_input_upper_bound_max': max(v['input_token_upper_bound'] for v in repair_budgets),
                          'command': build_command(schema_path=SCHEMA, instructions_path=INSTRUCTIONS, catalog_path=CATALOG),
                          'real_LLM_calls': 0})
    safe_path(OUT).mkdir()
    for name in ('run.log', 'llm_calls.jsonl'):
        safe_path(OUT / name).write_text('', encoding='utf8')
    write_json(OUT / 'config.json', cfg)
    def log(message):
        print(message, flush=True)
        with safe_path(OUT / 'run.log').open('a', encoding='utf8') as handle: handle.write(now() + ' ' + message + '\n')
    assert ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        scores, meta = nested_stack(d, np.asarray(cfg['scores']['A3']), log)
        cfg['scores'].update(scores)
        cfg['nested_stacking'] = meta
        cfg['status'] = 'prepared'
        initial_csv(cfg, d)
        metrics = statistics(cfg, d)
        metrics.update(status='prepared_family_LLM_not_started', family_valid=0, family_requested=3764, model_calls=0)
        write_json(OUT / 'metrics.json', metrics)
        write_json(OUT / 'config.json', cfg)
        readme(cfg['status'])
    finally: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    print(json.dumps({'status': cfg['status'], 'n': 3764, 'outer_HGB_max_abs_diff': [v['HGB_original_max_abs_diff'] for v in meta['outer_folds']],
                      'preflight': {k: v for k, v in cfg['preflight'].items() if k != 'command'}, 'runtime': cfg['runtime']}, ensure_ascii=False), flush=True)

def run(task, restart_index):
    cfg, d = read(OUT / 'config.json'), load_data()
    assert cfg['task_name'] == task and restart_index == len(cfg['runs']) and restart_index <= 1
    assert cfg['status'] == 'prepared' or (restart_index == 1 and cfg['status'] == 'interrupted_error')
    os.environ['F1_CODEX_EXE'] = cfg['runtime']['executable']
    assert resolve_runtime() == cfg['runtime']
    for name, value in cfg['source_hashes'].items(): assert sha(ROOT / name) == value, name
    for name, value in cfg['scientific_sources'].items(): assert sha(name) == value, name
    with (OUT / 'llm_calls.jsonl').open(encoding='utf8') as handle:
        reserved = sum(json.loads(line).get('event') == 'call_started' for line in handle)
    assert reserved <= MAX_CALLS
    meta = {'index': restart_index, 'started_utc': now(), 'reservations_before': reserved}
    cfg['runs'].append(meta)
    cfg.update(status='running_family', family_results={})
    write_json(OUT / 'config.json', cfg)
    assert ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    lock, stop, budget = threading.RLock(), threading.Event(), threading.Event()
    calls, results, pending, streak, stop_info = [], {}, [], [], {}
    started = time.monotonic()
    def log(s):
        with lock, safe_path(OUT / 'run.log').open('a', encoding='utf8') as handle: handle.write(now() + ' ' + s + '\n')
    def append(record):
        with lock, safe_path(OUT / 'llm_calls.jsonl').open('a', encoding='utf8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
            handle.flush(); os.fsync(handle.fileno())
    def reserve(record):
        nonlocal reserved
        with lock:
            if reserved >= MAX_CALLS:
                budget.set(); stop_info.setdefault('kind', 'global_call_budget_exhausted'); return None
            if stop.is_set(): return None
            reserved += 1
            record.update(call_id=reserved, availability_probe=len(calls) == 0 and reserved == meta['reservations_before'] + 1)
            append({'event': 'call_started', **record})
            return record
    def flush(force=False):
        if len(pending) >= 10 or (force and pending):
            with safe_path(OUT / 'predictions.csv').open('a', newline='', encoding='utf8') as handle:
                csv.DictWriter(handle, fieldnames=COLUMNS).writerows(pending)
                handle.flush(); os.fsync(handle.fileno())
            pending.clear()
    def work(case):
        response, last, retries, requests = None, None, 0, 0
        for scientific_round in (1, 2):
            message = prompt(case['formula'], repair=True if scientific_round == 2 else None)
            if scientific_round == 1: assert digest(message) == case['prompt_hash']
            while not stop.is_set():
                record = reserve({'experiment': 'G3_complementarity', 'run_index': restart_index, 'arm': 'G3_LLM_family',
                                  'id': case['id'], 'scientific_round': scientific_round, 'transport_retry': retries,
                                  'prompt': message, 'request_hash': digest(message), 'started_utc': now()})
                if record is None: break
                requests += 1
                record = call_once(message, schema_path=SCHEMA, instructions_path=INSTRUCTIONS, catalog_path=CATALOG, stop_event=stop, metadata=record)
                if record['status'] == 'ok':
                    try: response = validate(json.loads(record['response_text']))
                    except (ValueError, TypeError, KeyError) as exc: record.update(status='protocol_error', protocol_error=str(exc))
                record.update(event='call_completed', finished_utc=now(), response_valid=response is not None, parsed_response=response)
                append(record)
                with lock: calls.append(record)
                last = record
                if record['status'] == 'terminal_usage_limit':
                    with lock: stop_info.update(kind='explicit_usage_limit', errors=record.get('terminal_diagnostics'), recovery_time_message=record.get('recovery_time_message'), recovery_reset_fields=record.get('recovery_reset_fields'), call_id=record['call_id'])
                    stop.set()
                elif record['status'] in INTEGRITY:
                    with lock: stop_info.update(kind='integrity_violation', status=record['status'], call_id=record['call_id'])
                    stop.set()
                if response is not None or stop.is_set(): break
                if record['status'] == 'rate_limited_final' and retries < 3:
                    delay = DELAYS[retries]; retries += 1
                    log(f'FINAL429 {case["id"]} wait={delay}s identical-prompt retry={retries}/3')
                    if stop.wait(delay): break
                    continue
                break
            if response is not None or stop.is_set() or budget.is_set() or last is None or last['status'] == 'rate_limited_final': break
            if last['status'] not in {'protocol_error', 'cli_error', 'backend_error', 'incomplete_response', 'transport_error'}: break
        return case, {'response': response, 'status': last['status'] if last else 'not_dispatched', 'requests': requests,
                      'retries': retries, 'finished_monotonic': time.monotonic()}
    def snapshot(status, final=False):
        cfg['family_results'] = {key: value['response']['family'] for key, value in results.items() if value['response'] is not None}
        cfg['status'] = status
        write_json(OUT / 'config.json', cfg)
        m = {'status': status, 'family_valid': len(cfg['family_results']), 'family_finalized': len(results),
             'family_requested': 3764, 'reserved_across_runs': reserved, 'stop_reason': dict(stop_info),
             'new_calls': len(calls), 'wall_seconds': time.monotonic() - started}
        if final:
            extra = {}
            if len(cfg['family_results']) == 3764:
                labels = [cfg['family_results'][str(key)] for key in d['id']]
                values, details = augment_family(d, labels, FAMILIES, log)
                extra['G3_HGB_plus_family'] = values
                m['family_HGB'] = details
                with safe_path(OUT / 'predictions.csv').open('a', newline='', encoding='utf8') as handle: append_scores(handle, d, extra)
            full = statistics(cfg, d, extra)
            full.update(m)
            m = full
            m['family_label_counts'] = {label: sum(value == label for value in cfg['family_results'].values()) for label in FAMILIES}
            m['calls'] = {'reserved_across_runs': reserved, 'completed_this_run': len(calls),
                          'additional_429_retries': sum(v.get('transport_retry', 0) > 0 for v in calls),
                          'protocol_repairs': sum(v['scientific_round'] == 2 for v in calls),
                          'tools_used': sum(len(v.get('tool_activity', [])) for v in calls),
                          'runtime': cfg['runtime']}
        write_json(OUT / 'metrics.json', m)
        readme(status)
    try:
        initial_csv(cfg, d)
        log(f'START G3 run={restart_index} family_cases=3764 cap={MAX_CALLS} concurrency=4')
        jobs = sorted(cfg['cases'], key=lambda c: digest('G3|' + c['id']))
        iterator = iter(jobs)
        with ThreadPoolExecutor(max_workers=4) as pool:
            active = {pool.submit(work, next(iterator)): None}
            exhausted, probe_confirmed = False, False
            while active:
                ready, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                for future in sorted(ready, key=lambda f: f.result()[1]['finished_monotonic']):
                    active.pop(future)
                    case, result = future.result()
                    if result['requests']:
                        results[case['id']] = result
                        i = case['i']
                        label = result['response']['family'] if result['response'] else ''
                        pending.append({'id': case['id'], 'fold': int(d['fold'][i]), 'group': str(d['group'][i]), 'arm': 'G3_LLM_family',
                                        'status': 'valid_family' if label else result['status'], 'score': '', 'y_pred': '', 'family_label': label})
                        flush()
                    if result['response'] is not None:
                        streak.clear(); probe_confirmed = True
                    elif result['requests'] and not stop.is_set() and not budget.is_set():
                        streak.append(case['id'])
                        if len(streak) >= 3:
                            stop_info.update(kind='three_consecutive_exhausted_materials', material_ids=streak[-3:]); stop.set()
                    log(f'PROGRESS {len(results)}/3764 {case["id"]} {result["status"]} requests={result["requests"]} retries={result["retries"]} total_reserved={reserved}')
                    if len(results) % 10 == 0: snapshot('running_family')
                while not stop.is_set() and not budget.is_set() and not exhausted and len(active) < (4 if probe_confirmed else 1):
                    try: active[pool.submit(work, next(iterator))] = None
                    except StopIteration: exhausted = True
        flush(True)
        valid = sum(value['response'] is not None for value in results.values())
        status = 'complete' if valid == 3764 else 'stopped_partial' if stop.is_set() or budget.is_set() else 'complete_with_failures'
        snapshot(status, final=True)
        meta.update(finished_utc=now(), reservations_end=reserved, status=status, stop_reason=dict(stop_info))
        write_json(OUT / 'config.json', cfg)
        log(f'END status={status} valid={valid}/3764 reserved={reserved} stop={stop_info}')
        return 0 if status == 'complete' else 75
    except BaseException:
        stop.set(); flush(True)
        meta.update(status='interrupted_error', finished_utc=now(), reservations_end=reserved)
        snapshot('interrupted_error', final=False)
        log('FATAL ' + traceback.format_exc())
        raise
    finally: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('prepare', 'run'))
    p.add_argument('--task-name', required=True)
    p.add_argument('--restart-index', type=int, default=0)
    a = p.parse_args()
    if a.action == 'prepare': prepare(a.task_name)
    else: raise SystemExit(run(a.task_name, a.restart_index))
