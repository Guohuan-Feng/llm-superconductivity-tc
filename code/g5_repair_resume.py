"""Resume the single authorized G5 repair pass after a local runner interruption."""
from __future__ import annotations
from common import ROOT, load_data, safe_path
from g5_inputs import OUT, SCHEMA, INSTRUCTIONS, CATALOG, prompt, validate, sha
from g5_stats import build_statistics
from g5_report import build_report
from f2_transport_v2 import call_once, resolve_runtime
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import csv, hashlib, json, os, threading, time

INTEGRITY={'tool_activity_rejected','model_mismatch_rejected','backend_protocol_rejected','preflight_rejected','actual_input_budget_violation'}
COLUMNS=['id','fold','group','arm','status','score','y_pred','threshold_K','Tc_est_K','p_gt_5K','p_gt_10K','p_gt_15K']
def now(): return datetime.now(timezone.utc).isoformat()
def digest(s): return hashlib.sha256(s.encode('utf8')).hexdigest()
def append_jsonl(path,obj):
    with safe_path(path).open('a',encoding='utf8') as f:
        f.write(json.dumps(obj,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n'); f.flush(); os.fsync(f.fileno())
def log(msg):
    with safe_path(OUT/'run.log').open('a',encoding='utf8') as f: f.write(now()+' '+msg+'\n'); f.flush(); os.fsync(f.fileno())

def main():
    cfg=json.loads((OUT/'config.json').read_text(encoding='utf8')); d=load_data()
    assert cfg['status']=='running_A3_reg' and len(cfg['runs'])==2 and cfg['runs'][-1]['index']==1
    assert cfg['runtime']==resolve_runtime()
    for name,expected in cfg['source_hashes'].items(): assert sha(ROOT/name)==expected,name
    cases={c['id']:c for c in cfg['cases']}
    # Recover the probe's valid response written before the runner interruption.
    recovered={}
    for line in (OUT/'llm_calls.jsonl').open(encoding='utf8'):
        x=json.loads(line)
        if x.get('event')=='call_completed' and x.get('run_index')==1 and x.get('response_valid') and x.get('parsed_response') is not None:
            recovered[x['id']]=x['parsed_response']
    cfg['results'].update(recovered)
    missing=[c for c in cfg['cases'] if c['id'] not in cfg['results']]
    assert len(missing)<=8, len(missing)
    reserved=sum(json.loads(line).get('event')=='call_started' for line in (OUT/'llm_calls.jsonl').open(encoding='utf8'))
    lock=threading.RLock(); stop=threading.Event(); stop_info={}; results={}; calls=[]
    def reserve(rec):
        nonlocal reserved
        with lock:
            if reserved>=8000 or stop.is_set(): return None
            reserved+=1; rec.update(call_id=reserved,availability_probe=False)
            append_jsonl(OUT/'llm_calls.jsonl',{'event':'call_started',**rec}); return rec
    def work(case):
        response=None; last=None; requests=0; retries=0
        for scientific_round in (1,2):
            msg=prompt(case['material'],True if scientific_round==2 else None)
            if scientific_round==1: assert digest(msg)==case['prompt_sha256']
            rec=reserve({'experiment':'G5_tc_multithreshold','run_index':1,'arm':'A3_reg','id':case['id'],'scientific_round':scientific_round,'transport_retry':retries,'prompt':msg,'request_hash':digest(msg),'started_utc':now()})
            if rec is None: break
            requests+=1; rec=call_once(msg,schema_path=SCHEMA,instructions_path=INSTRUCTIONS,catalog_path=CATALOG,stop_event=stop,metadata=rec)
            if rec['status']=='ok':
                try: response=validate(json.loads(rec['response_text']))
                except (ValueError,TypeError,KeyError) as e: rec.update(status='protocol_error',protocol_error=str(e))
            rec.update(event='call_completed',finished_utc=now(),response_valid=response is not None,parsed_response=response); append_jsonl(OUT/'llm_calls.jsonl',rec); calls.append(rec); last=rec
            if rec['status']=='terminal_usage_limit': stop_info.update(kind='explicit_usage_limit',recovery_time_message=rec.get('recovery_time_message'),call_id=rec['call_id']); stop.set(); break
            if rec['status'] in INTEGRITY: stop_info.update(kind='integrity_violation',status=rec['status'],call_id=rec['call_id']); stop.set(); break
            if response is not None: break
            # The original G5 runner permits one same-evidence format/backend retry.
            if rec['status'] in {'protocol_error','cli_error','backend_error','incomplete_response','transport_error'}: continue
            break
        return case,{'response':response,'status':last['status'] if last else 'not_dispatched','requests':requests,'retries':retries}
    log(f'RESUME G5 repair run=1 remaining={len(missing)} concurrency=4')
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(work,c) for c in missing]
        for fut in as_completed(futures):
            c,r=fut.result(); results[c['id']]=r; log(f'REPAIR {c["id"]} {r["status"]} requests={r["requests"]} retries={r["retries"]}')
    for cid,r in results.items():
        if r['response'] is not None: cfg['results'][cid]=r['response']
    valid=sum(cid in cfg['results'] for cid in cases); status='complete' if valid==len(cases) else 'stopped_partial'
    p=OUT/'predictions.csv'; rows=list(csv.DictReader(p.open(encoding='utf8')))
    for row in rows:
        if row['arm']=='A3_reg' and row['id'] in cfg['results'] and row['id'] in {c['id'] for c in missing}:
            r=cfg['results'][row['id']]; row.update(status='valid_A3_reg',score=str(r['p_gt_10K']),y_pred=str(int(r['p_gt_10K']>=.5)),threshold_K='10',Tc_est_K=str(r['Tc_est_K']),p_gt_5K=str(r['p_gt_5K']),p_gt_10K=str(r['p_gt_10K']),p_gt_15K=str(r['p_gt_15K']))
    with safe_path(p).open('w',newline='',encoding='utf8') as f: w=csv.DictWriter(f,fieldnames=COLUMNS); w.writeheader(); w.writerows(rows); f.flush(); os.fsync(f.fileno())
    cfg['status']=status; cfg['repair_status']=status; cfg['runs'][-1].update(finished_utc=now(),reservations_end=reserved,status=status,stop_reason=stop_info)
    full=build_statistics(cfg,d,cfg['results'],log=log); full.update(status=status,LLM_valid=valid,LLM_finalized=len(cases),LLM_requested=len(cases),reserved_across_runs=reserved,stop_reason=stop_info,new_calls=len(calls),wall_seconds=0)
    full['calls']={'reserved_across_runs':reserved,'completed_this_run':len(calls),'additional_429_retries':sum(v.get('transport_retry',0)>0 for v in calls),'protocol_repairs':sum(v['scientific_round']==2 for v in calls),'tools_used':sum(len(v.get('tool_activity',[])) for v in calls),'runtime':cfg['runtime']}
    full['report_markdown']=build_report(full)
    (OUT/'metrics.json').write_text(json.dumps(full,ensure_ascii=False,indent=2),encoding='utf8'); (OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf8')
    log(f'END G5 repair resume status={status} valid={valid}/{len(cases)} reserved={reserved} stop={stop_info}')
    print(json.dumps({'status':status,'valid':valid,'remaining':len(cases)-valid,'reserved':reserved,'stop':stop_info},ensure_ascii=False))
if __name__=='__main__': main()
