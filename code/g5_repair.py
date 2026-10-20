"""One finite repair pass for G5 materials missing after a completed run."""
from __future__ import annotations

from common import ROOT, load_data, safe_path
from g5_inputs import OUT, SCHEMA, INSTRUCTIONS, CATALOG, prompt, validate, sha
from g5_stats import build_statistics
from g5_report import build_report
from f2_transport_v2 import call_once, resolve_runtime
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import csv, ctypes, hashlib, json, threading, time

MAX_CALLS = 8000
INTEGRITY = {'tool_activity_rejected','model_mismatch_rejected','backend_protocol_rejected','preflight_rejected','actual_input_budget_violation'}
COLUMNS = ['id','fold','group','arm','status','score','y_pred','threshold_K','Tc_est_K','p_gt_5K','p_gt_10K','p_gt_15K']

def now(): return datetime.now(timezone.utc).isoformat()
def digest(s): return hashlib.sha256(s.encode('utf8')).hexdigest()

def read_json(path): return json.loads(path.read_text(encoding='utf8'))
def append_jsonl(path, obj):
    with safe_path(path).open('a', encoding='utf8') as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(',',':'))+'\n')
        f.flush(); os_fsync(f)
def os_fsync(f):
    import os
    os.fsync(f.fileno())

def write_log(msg):
    with safe_path(OUT/'run.log').open('a', encoding='utf8') as f:
        f.write(now()+' '+msg+'\n'); f.flush(); os_fsync(f)

def main():
    cfg = read_json(OUT/'config.json'); d = load_data()
    assert cfg['status'] == 'complete_with_failures'
    assert len(cfg.get('runs', [])) == 1
    assert cfg['runtime'] == resolve_runtime()
    for name, expected in cfg['source_hashes'].items(): assert sha(ROOT/name) == expected, name
    cases = {c['id']: c for c in cfg['cases']}
    missing = [c for c in cfg['cases'] if c['id'] not in cfg['results']]
    assert 0 < len(missing) <= 20, len(missing)
    existing_reserved = sum(json.loads(line).get('event') == 'call_started' for line in (OUT/'llm_calls.jsonl').open(encoding='utf8'))
    assert existing_reserved <= MAX_CALLS
    run_index = 1
    cfg['runs'].append({'index': run_index, 'started_utc': now(), 'reservations_before': existing_reserved,
                        'repair_ids': [c['id'] for c in missing], 'G4_completed_handoff_authorized': True})
    cfg['status'] = 'running_A3_reg'; cfg['repair_status'] = 'running'; (OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf8')
    lock = threading.RLock(); stop = threading.Event(); reserved = existing_reserved; calls=[]; results={}; stop_info={}; started=time.monotonic()
    def reserve(rec):
        nonlocal reserved
        with lock:
            if reserved >= MAX_CALLS or stop.is_set(): return None
            reserved += 1; rec.update(call_id=reserved, availability_probe=(reserved == existing_reserved+1))
            append_jsonl(OUT/'llm_calls.jsonl', {'event':'call_started', **rec}); return rec
    def work(case):
        response=None; last=None; retries=0; requests=0
        for scientific_round in (1,2):
            message=prompt(case['material'], True if scientific_round==2 else None)
            if scientific_round==1: assert digest(message)==case['prompt_sha256']
            record=reserve({'experiment':'G5_tc_multithreshold','run_index':run_index,'arm':'A3_reg','id':case['id'],
                'scientific_round':scientific_round,'transport_retry':retries,'prompt':message,'request_hash':digest(message),'started_utc':now()})
            if record is None: break
            requests += 1
            record=call_once(message,schema_path=SCHEMA,instructions_path=INSTRUCTIONS,catalog_path=CATALOG,stop_event=stop,metadata=record)
            if record['status']=='ok':
                try: response=validate(json.loads(record['response_text']))
                except (ValueError,TypeError,KeyError) as exc: record.update(status='protocol_error', protocol_error=str(exc))
            record.update(event='call_completed',finished_utc=now(),response_valid=response is not None,parsed_response=response)
            append_jsonl(OUT/'llm_calls.jsonl',record); calls.append(record); last=record
            if record['status']=='terminal_usage_limit':
                stop_info.update(kind='explicit_usage_limit', recovery_time_message=record.get('recovery_time_message'), call_id=record['call_id']); stop.set(); break
            if record['status'] in INTEGRITY:
                stop_info.update(kind='integrity_violation', status=record['status'], call_id=record['call_id']); stop.set(); break
            if response is not None: break
            if record['status']=='rate_limited_final' and retries<3:
                delay=(60,120,240)[retries]; retries += 1; write_log(f'REPAIR_FINAL429 {case["id"]} wait={delay}s retry={retries}/3')
                if stop.wait(delay): break
                scientific_round -= 1
                continue
            if record['status'] not in {'protocol_error','cli_error','backend_error','incomplete_response','transport_error'}: break
        return case, {'response':response,'status':last['status'] if last else 'not_dispatched','requests':requests,'retries':retries}
    write_log(f'START G5 repair run=1 missing={len(missing)} concurrency=4')
    iterator=iter(missing); probe=next(iterator); active={}
    first_case, first=work(probe); results[probe['id']]=first; write_log(f'REPAIR {first_case["id"]} {first["status"]} requests={first["requests"]} retries={first["retries"]}')
    if first['response'] is None and first['status']=='terminal_usage_limit': stop.set()
    if first['response'] is not None:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for case in iterator: active[pool.submit(work,case)] = case
            while active:
                ready,_=wait(active, timeout=1, return_when=FIRST_COMPLETED)
                for fut in ready:
                    case=fut.result()[0]; results[case['id']]=fut.result()[1]
                    write_log(f'REPAIR {case["id"]} {results[case["id"]]["status"]} requests={results[case["id"]]["requests"]} retries={results[case["id"]]["retries"]}')
    for cid,res in results.items():
        if res['response'] is not None: cfg['results'][cid]=res['response']
    valid=sum(cid in cfg['results'] for cid in cases); final_status='complete' if valid==len(cases) else 'stopped_partial'
    # Replace the old failed A3_reg rows in the compact audit CSV, preserving one row/material/arm.
    csv_path=OUT/'predictions.csv'; rows=list(csv.DictReader(csv_path.open(encoding='utf8')))
    for row in rows:
        cid=row['id']
        if row['arm']=='A3_reg' and cid in results and results[cid]['response'] is not None:
            r=results[cid]['response']; row.update(status='valid_A3_reg',score=str(r['p_gt_10K']),y_pred=str(int(r['p_gt_10K']>=.5)),threshold_K='10',Tc_est_K=str(r['Tc_est_K']),p_gt_5K=str(r['p_gt_5K']),p_gt_10K=str(r['p_gt_10K']),p_gt_15K=str(r['p_gt_15K']))
    with safe_path(csv_path).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=COLUMNS); w.writeheader(); w.writerows(rows); f.flush(); os_fsync(f)
    cfg['status']=final_status; cfg['repair_status']=final_status; cfg['runs'][-1].update(finished_utc=now(),reservations_end=reserved,status=final_status,stop_reason=stop_info)
    full=build_statistics(cfg,d,cfg['results'],log=write_log); full.update(status=final_status,LLM_valid=valid,LLM_finalized=len(cases),LLM_requested=len(cases),reserved_across_runs=reserved,stop_reason=stop_info,new_calls=len(calls),wall_seconds=time.monotonic()-started)
    full['calls']={'reserved_across_runs':reserved,'completed_this_run':len(calls),'additional_429_retries':sum(v.get('transport_retry',0)>0 for v in calls),'protocol_repairs':sum(v['scientific_round']==2 for v in calls),'tools_used':sum(len(v.get('tool_activity',[])) for v in calls),'runtime':cfg['runtime']}
    full['report_markdown']=build_report(full)
    (OUT/'metrics.json').write_text(json.dumps(full,ensure_ascii=False,indent=2),encoding='utf8')
    (OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf8')
    write_log(f'END G5 repair status={final_status} valid={valid}/{len(cases)} reserved={reserved} stop={stop_info}')
    print(json.dumps({'status':final_status,'valid':valid,'missing_after':len(cases)-valid,'reserved':reserved,'stop':stop_info},ensure_ascii=False))

if __name__=='__main__': main()
