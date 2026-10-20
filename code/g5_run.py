"""Finite G5 A3-reg evaluation; preparation performs CPU fits and no LLM calls."""
from common import ROOT, FIELDS, safe_path, write_json, load_data, np
from g5_inputs import OUT, SCHEMA, INSTRUCTIONS, CATALOG, THRESHOLDS, sha, prompt, validate, frozen_inputs
from g5_ml import train_oof
from g5_stats import build_statistics
from g5_report import build_report
from f2_transport_v2 import call_once, resolve_runtime
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

MAX_CALLS=8000
DELAYS=(60,120,240)
EXE=r'C:\Users\28908\AppData\Local\OpenAI\Codex\bin\896a0166b774037a\codex.exe'
INTEGRITY={'tool_activity_rejected','model_mismatch_rejected','backend_protocol_rejected','preflight_rejected','actual_input_budget_violation'}
COLUMNS=FIELDS+['threshold_K','Tc_est_K','p_gt_5K','p_gt_10K','p_gt_15K']
SOURCES=('g5_run.py','g5_inputs.py','g5_ml.py','g5_stats.py','g5_report.py','g5_instructions.md','g5_response.schema.json',
         'g5_launch.ps1','g3_ml.py','common.py','f_stats.py','f3_stats.py','f_report.py',
         'f2_transport_v2.py','f1_transport.py','e2_transport.py','e2_model_catalog.json','e2_packets.py','e2_prompts.py')

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def digest(s):return hashlib.sha256(s.encode('utf8')).hexdigest()

def readme(status):
    lines=['# G5 Tc estimation and multiple thresholds',f'Status: {status}.',
        'Original3764 materials/1117connected groups; original three outer folds unchanged; development only.',
        'A3-reg material object exactly reuses original E2/F3 formula+composition+structure; no labels/rules/neighbors in prompt.',
        'One fresh call returns Tc_est_K and p(Tc>5/10/15K); non-superconductors estimated0K.',
        'Protocol requires finite numbers,Tc>=0,1>=p5>=p10>=p15>=0; outputs never clipped or repaired numerically.',
        'Point-estimate and direct-probability classifications evaluated separately; strict Tc>threshold; p>=.5 positive.',
        'HGB classifier frozen original parameters at each threshold; TRAIN-only group scaler and group/class-balanced loss.',
        'HGB regressor frozen same structure, squared_error on log1p(Tc), TRAIN group weights only; inverse max(0,expm1).',
        'No tuning or early stopping; HGB10 originalOOF reproduced before evaluated calls.',
        'Both weights/four classification metrics, MAEall and trueTc>5/10/15, Spearman;5000paired original-group bootstrap.',
        'Weighted Spearman=weighted Pearson of weighted mid-CDF ranks; ranks recomputed for every bootstrap draw.',
        'Three subsets all/exclude93/only93, originalF3fivefamilies, original group-size bins, frozen G3twelvefamilies.',
        'G3not-superconductor label is a prior judgment, not an independent measured target; sparse strata retain NA.',
        'Fresh local logged-in codex exec gpt-6.1-sol/high, tools/web disabled; first scientific case probes alone then4concurrent.',
        'CLI reconnect allowed to finish; final429 retries identical prompt after1,2,4min,max3additional retries/material.',
        'Explicit usage-limit or3consecutive exhausted distinct materials stops and saves; cap8000reservations incl repairs/restarts.',
        'At most2scientific rounds, second same-evidence protocol repair only; no professor/alternate API.',
        'One-shot task PT0S/restart0/no triggers; SetThreadExecutionState; resultCSV appends/fsyncs every10materials.',
        'Interruption: inspect system/task logs first, at most1whole recomputation; six audit files only; prior experiments read-only.']
    assert len(lines)<=20
    safe_path(OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

def rowbase(d,i,arm,status):
    return {'id':str(d['id'][i]),'fold':int(d['fold'][i]),'group':str(d['group'][i]),'arm':arm,'status':status,
            'score':'','y_pred':'','threshold_K':'','Tc_est_K':'','p_gt_5K':'','p_gt_10K':'','p_gt_15K':''}

def initial_csv(cfg,d):
    with safe_path(OUT/'predictions.csv').open('w',newline='',encoding='utf8') as handle:
        writer=csv.DictWriter(handle,fieldnames=COLUMNS);writer.writeheader()
        for i in range(3764):
            for t in THRESHOLDS:
                arm=f'HGB_classifier_gt_{t}K';p=cfg['ML_classifier_scores'][arm][i]
                r=rowbase(d,i,arm,'valid_original_fold_OOF');r.update(score=p,y_pred=int(p>=.5),threshold_K=t);writer.writerow(r)
            r=rowbase(d,i,'HGB_regressor_log1p','valid_original_fold_OOF');r['Tc_est_K']=cfg['HGB_regressor_Tc_K'][i];writer.writerow(r)
            p=cfg['original_A3_10K'][i]
            r=rowbase(d,i,'A3_original_10K','frozen_F3_reused');r.update(score=p,y_pred=int(p>=.5),threshold_K=10);writer.writerow(r)
            if (i+1)%10==0:handle.flush();os.fsync(handle.fileno())
        handle.flush();os.fsync(handle.fileno())

def prepare(task):
    assert not OUT.exists(),'G5 already exists; inspect before any recomputation'
    for path in (ROOT,*ROOT.parents):assert not path.is_symlink() and not path.is_junction()
    d=load_data();cfg=frozen_inputs(d)
    os.environ['F1_CODEX_EXE']=EXE
    from e2_transport import input_budget,build_command
    fixed=input_budget('',schema_path=SCHEMA,instructions_path=INSTRUCTIONS)
    overhead=fixed['input_token_upper_bound']
    initial=[len(prompt(case['material']).encode('utf8'))+overhead for case in cfg['cases']]
    repair=[len(prompt(case['material'],True).encode('utf8'))+overhead for case in cfg['cases']]
    assert max(initial+repair)<=15000
    cfg.update(experiment='G5_tc_multithreshold',status='preparing_ML',task_name=task,created_utc=now(),
        runtime=resolve_runtime(),source_hashes={n:sha(ROOT/n) for n in SOURCES},
        git_commit_at_prepare=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),runs=[],
        protocol={'concurrency':4,'max_calls':MAX_CALLS,'max_scientific_rounds':2,'max_final429_retries_per_material':3,
                  'retry_delays_seconds':list(DELAYS),'n_boot':5000,'seed':20261018,'ExecutionTimeLimit':'PT0S',
                  'RestartCount':0,'first_scientific_case_is_probe':True,'max_process_restarts':1,
                  'protocol_validation':'finite; Tc>=0;1>=p5>=p10>=p15>=0; never numerical correction',
                  'global_G4_G5_concurrency':'G5 LLM cannot start until G4 finishes and root explicitly authorizes handoff'},
        preflight={'all3764_original_A3_material_objects_reused':True,'input_initial_min':min(initial),'input_initial_max':max(initial),
                   'input_repair_max':max(repair),'input_limit':15000,'known_fixed_context_snapshot':fixed,
                   'prompt_schema_command':build_command(schema_path=SCHEMA,instructions_path=INSTRUCTIONS,catalog_path=CATALOG),
                   'real_LLM_calls':0},results={})
    safe_path(OUT).mkdir()
    for name in ('llm_calls.jsonl','run.log'):safe_path(OUT/name).write_text('',encoding='utf8')
    write_json(OUT/'config.json',cfg)
    def log(message):
        print(message,flush=True)
        with safe_path(OUT/'run.log').open('a',encoding='utf8') as f:f.write(now()+' '+message+'\n')
    assert ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        classifiers,reg,metadata=train_oof(d,log)
        cfg.update(ML_classifier_scores=classifiers,HGB_regressor_Tc_K=reg,ML_training=metadata,status='prepared_waiting_G4_handoff')
        initial_csv(cfg,d)
        write_json(OUT/'config.json',cfg)
        write_json(OUT/'metrics.json',{'status':cfg['status'],'LLM_calls':0,'n':3764,'groups':1117,
            'n_positive_at_threshold':{str(t):int(np.sum(d['tc']>t)) for t in THRESHOLDS},
            'classifier_fits':9,'regressor_fits':3,'HGB10_original_max_abs_diff':[r['original_HGB_max_abs_diff'] for r in metadata['fit_records'] if r.get('target')=='Tc>10K']})
        readme(cfg['status'])
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    print(json.dumps({'status':cfg['status'],'preflight':{k:v for k,v in cfg['preflight'].items() if k not in ('prompt_schema_command','known_fixed_context_snapshot')},
                      'HGB10_max_abs_diff':[r['original_HGB_max_abs_diff'] for r in metadata['fit_records'] if r.get('target')=='Tc>10K']},ensure_ascii=False),flush=True)

def run(task,restart_index,handoff):
    assert handoff,'LLM start requires explicit G4-completed root authorization'
    cfg,d=read(OUT/'config.json'),load_data()
    assert cfg['task_name']==task and restart_index==len(cfg['runs']) and restart_index<=1
    assert cfg['status']=='prepared_waiting_G4_handoff' or (restart_index==1 and cfg['status']=='interrupted_error')
    os.environ['F1_CODEX_EXE']=cfg['runtime']['executable'];assert resolve_runtime()==cfg['runtime']
    for name,value in cfg['source_hashes'].items():assert sha(ROOT/name)==value,name
    for name,value in cfg['scientific_sources'].items():assert sha(name)==value,name
    with (OUT/'llm_calls.jsonl').open(encoding='utf8') as f:reserved=sum(json.loads(line).get('event')=='call_started' for line in f)
    assert reserved<=MAX_CALLS
    meta={'index':restart_index,'started_utc':now(),'reservations_before':reserved,'G4_completed_handoff_authorized':True}
    cfg['runs'].append(meta);cfg.update(status='running_A3_reg',results={});write_json(OUT/'config.json',cfg)
    assert ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    lock,stop,budget=threading.RLock(),threading.Event(),threading.Event()
    calls,results,pending,streak,stop_info=[],{},[],[],{}
    started=time.monotonic()
    def log(message):
        with lock,safe_path(OUT/'run.log').open('a',encoding='utf8') as f:f.write(now()+' '+message+'\n')
    def append(record):
        with lock,safe_path(OUT/'llm_calls.jsonl').open('a',encoding='utf8') as f:
            f.write(json.dumps(record,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
    def reserve(record):
        nonlocal reserved
        with lock:
            if reserved>=MAX_CALLS:budget.set();stop_info.setdefault('kind','global_call_budget_exhausted');return None
            if stop.is_set():return None
            reserved+=1;record.update(call_id=reserved,availability_probe=reserved==meta['reservations_before']+1)
            append({'event':'call_started',**record});return record
    def flush(force=False):
        if len(pending)>=10 or (force and pending):
            with safe_path(OUT/'predictions.csv').open('a',newline='',encoding='utf8') as f:
                csv.DictWriter(f,fieldnames=COLUMNS).writerows(pending);f.flush();os.fsync(f.fileno())
            pending.clear()
    def work(case):
        response,last,retries,requests=None,None,0,0
        for scientific_round in (1,2):
            message=prompt(case['material'],True if scientific_round==2 else None)
            if scientific_round==1:assert digest(message)==case['prompt_sha256']
            while not stop.is_set():
                record=reserve({'experiment':'G5_tc_multithreshold','run_index':restart_index,'arm':'A3_reg','id':case['id'],
                    'scientific_round':scientific_round,'transport_retry':retries,'prompt':message,'request_hash':digest(message),'started_utc':now()})
                if record is None:break
                requests+=1
                record=call_once(message,schema_path=SCHEMA,instructions_path=INSTRUCTIONS,catalog_path=CATALOG,stop_event=stop,metadata=record)
                if record['status']=='ok':
                    try:response=validate(json.loads(record['response_text']))
                    except (ValueError,TypeError,KeyError) as exc:record.update(status='protocol_error',protocol_error=str(exc))
                record.update(event='call_completed',finished_utc=now(),response_valid=response is not None,parsed_response=response)
                append(record)
                with lock:calls.append(record)
                last=record
                if record['status']=='terminal_usage_limit':
                    with lock:stop_info.update(kind='explicit_usage_limit',errors=record.get('terminal_diagnostics'),recovery_time_message=record.get('recovery_time_message'),recovery_reset_fields=record.get('recovery_reset_fields'),call_id=record['call_id'])
                    stop.set()
                elif record['status'] in INTEGRITY:
                    with lock:stop_info.update(kind='integrity_violation',status=record['status'],call_id=record['call_id'])
                    stop.set()
                if response is not None or stop.is_set():break
                if record['status']=='rate_limited_final' and retries<3:
                    delay=DELAYS[retries];retries+=1;log(f'FINAL429 {case["id"]} wait={delay}s identical-prompt retry={retries}/3')
                    if stop.wait(delay):break
                    continue
                break
            if response is not None or stop.is_set() or budget.is_set() or last is None or last['status']=='rate_limited_final':break
            if last['status'] not in {'protocol_error','cli_error','backend_error','incomplete_response','transport_error'}:break
        return case,{'response':response,'status':last['status'] if last else 'not_dispatched','requests':requests,'retries':retries,'finished_monotonic':time.monotonic()}
    def snapshot(status,final=False):
        cfg['results']={key:value['response'] for key,value in results.items() if value['response'] is not None};cfg['status']=status
        write_json(OUT/'config.json',cfg)
        m={'status':status,'LLM_valid':len(cfg['results']),'LLM_finalized':len(results),'LLM_requested':3764,'reserved_across_runs':reserved,
           'stop_reason':dict(stop_info),'new_calls':len(calls),'wall_seconds':time.monotonic()-started}
        if final:
            log('BEGIN final5000groupbootstrap classification+regression; no more evaluated calls')
            full=build_statistics(cfg,d,cfg['results'],log=log);full.update(m);m=full
            m['calls']={'reserved_across_runs':reserved,'completed_this_run':len(calls),
                'additional_429_retries':sum(v.get('transport_retry',0)>0 for v in calls),'protocol_repairs':sum(v['scientific_round']==2 for v in calls),
                'tools_used':sum(len(v.get('tool_activity',[])) for v in calls),'runtime':cfg['runtime']}
            m['report_markdown']=build_report(m)
        write_json(OUT/'metrics.json',m);readme(status)
    try:
        initial_csv(cfg,d);log(f'START G5 run={restart_index} cases=3764 cap={MAX_CALLS} concurrency=4 G4handoff=true')
        jobs=sorted(cfg['cases'],key=lambda c:digest('G5|'+c['id']));iterator=iter(jobs)
        with ThreadPoolExecutor(max_workers=4) as pool:
            active={pool.submit(work,next(iterator)):None};exhausted,probe_confirmed=False,False
            while active:
                ready,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
                for future in sorted(ready,key=lambda f:f.result()[1]['finished_monotonic']):
                    active.pop(future);case,result=future.result()
                    if result['requests']:
                        results[case['id']]=result;r=rowbase(d,case['i'],'A3_reg','valid_A3_reg' if result['response'] else result['status'])
                        if result['response']:
                            r.update(result['response']);r.update(score=result['response']['p_gt_10K'],y_pred=int(result['response']['p_gt_10K']>=.5),threshold_K=10)
                        pending.append(r);flush()
                    if result['response'] is not None:streak.clear();probe_confirmed=True
                    elif result['requests'] and not stop.is_set() and not budget.is_set():
                        streak.append(case['id'])
                        if len(streak)>=3:stop_info.update(kind='three_consecutive_exhausted_materials',material_ids=streak[-3:]);stop.set()
                    log(f'PROGRESS {len(results)}/3764 {case["id"]} {result["status"]} requests={result["requests"]} retries={result["retries"]} total_reserved={reserved}')
                    if len(results)%10==0:snapshot('running_A3_reg')
                while not stop.is_set() and not budget.is_set() and not exhausted and len(active)<(4 if probe_confirmed else 1):
                    try:active[pool.submit(work,next(iterator))]=None
                    except StopIteration:exhausted=True
        flush(True);valid=sum(v['response'] is not None for v in results.values())
        status='complete' if valid==3764 else 'stopped_partial' if stop.is_set() or budget.is_set() else 'complete_with_failures'
        snapshot(status,True)
        meta.update(finished_utc=now(),reservations_end=reserved,status=status,stop_reason=dict(stop_info));write_json(OUT/'config.json',cfg)
        log(f'END status={status} valid={valid}/3764 reserved={reserved} stop={stop_info}')
        return 0 if status=='complete' else 75
    except BaseException:
        stop.set();flush(True);meta.update(status='interrupted_error',finished_utc=now(),reservations_end=reserved)
        snapshot('interrupted_error');log('FATAL '+traceback.format_exc());raise
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run'));p.add_argument('--task-name',required=True)
    p.add_argument('--restart-index',type=int,default=0);p.add_argument('--g4-completed-handoff',action='store_true');a=p.parse_args()
    if a.action=='prepare':prepare(a.task_name)
    else:raise SystemExit(run(a.task_name,a.restart_index,a.g4_completed_handoff))
