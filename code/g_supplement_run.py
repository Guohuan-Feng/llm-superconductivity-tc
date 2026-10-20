"""Finite G5c/G7b evaluator: one real probe, then four global CLI workers."""
from __future__ import annotations
import csv,ctypes,hashlib,json,os,threading,time,traceback
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from common import ROOT,safe_path
from supplement_io import write_json
from f2_transport_v2 import call_once,resolve_runtime
from f3_inputs import validate_output
from g5_inputs import validate as validate_reg

WORK=ROOT/'work/G_supplement_runtime'
FOLDERS={'G5c':'G5c_majority3764','G7b':'G7b_supplement'}
FIELDS=['id','fold','group','arm','status','score','y_pred']
DELAYS=(60,120,240)
EXPECTED_COUNTS={'G5c':5928,'G7b':534}
OPERATIONAL_ERRORS={'cli_error','backend_error','incomplete_response','transport_error'}
MAX_OPERATIONAL_RETRIES=3
INTEGRITY={'tool_activity_rejected','model_mismatch_rejected','backend_protocol_rejected','preflight_rejected','actual_input_budget_violation'}

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(s):return hashlib.sha256(s.encode('utf8')).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))

def run():
    import g5c_inputs,g7b_inputs
    for p in (ROOT,WORK,*ROOT.parents):
        assert not p.is_symlink() and not p.is_junction() and 'onedrive' not in str(p.resolve()).lower()
    WORK.mkdir(parents=True,exist_ok=True)
    assert not (WORK/'session.json').exists(),'Existing finite run: inspect status/logs before any recomputation.'
    cfg={k:read(ROOT/v/'config.json') for k,v in FOLDERS.items()}
    for k,c in cfg.items():assert c['status']=='prepared',(k,c['status'])
    runtimes=[c['runtime'] for c in cfg.values()];assert runtimes[0]==runtimes[1]
    os.environ['F1_CODEX_EXE']=runtimes[0]['executable'];assert resolve_runtime()==runtimes[0]
    cases={k:c['cases'] for k,c in cfg.items()}
    modules={'G5c':g5c_inputs,'G7b':g7b_inputs}
    jobs=[]
    for exp,c in cfg.items():
        for raw in c['pending_jobs']:
            j=dict(raw,experiment=exp);case=cases[exp][j['case_index']]
            assert j['id']==case['id'];j.setdefault('kind','a3');j.setdefault('score_key','p')
            msg=modules[exp].prompt_for(case,repair=False) if exp=='G5c' else modules[exp].prompt_for(case,j['arm'],repair=False)
            assert digest(msg)==j['prompt_sha256'],j['job_id']
            jobs.append(j)
    for exp,n in EXPECTED_COUNTS.items():assert sum(j['experiment']==exp for j in jobs)==n
    # Probe one actual missing G5c case; then complete the bounded external block first.
    first=next(j for j in jobs if j['experiment']=='G5c')
    jobs=[first]+sorted([j for j in jobs if j is not first],key=lambda j:(j['experiment']!='G7b',digest(j['experiment']+'|'+j['job_id'])))
    frozen_paths=[ROOT/'g_supplement_run.py',ROOT/'supplement_io.py',ROOT/'g5c_inputs.py',ROOT/'g7b_inputs.py',ROOT/'e2_response.schema.json',ROOT/'g5_response.schema.json',ROOT/'e2_instructions.md',ROOT/'g5_instructions.md',ROOT/'e2_model_catalog.json',ROOT/'f2_transport_v2.py',ROOT/'f1_transport.py',ROOT/'e2_transport.py']
    recovery=cfg['G5c'].get('recovery',{})
    recovery_count=cfg['G5c'].get('recovery_count',recovery.get('count',int(bool(recovery))))
    assert recovery_count in (0,1),'Only one recovery is authorized'
    session={'status':'prepared','created_utc':now(),'n_jobs':len(jobs),'global_concurrency':4,'first_actual_case_probe':True,'runtime':runtimes[0],
             'source_hashes':{str(p):sha(p) for p in frozen_paths},'prepared_config_hashes':{k:sha(ROOT/FOLDERS[k]/'config.json') for k in cfg},
             'task_name':cfg['G5c']['task_name'],'results_count':{},'stop_reason':{},'max_additional_429_retries':3,'delays_seconds':list(DELAYS),'max_scientific_rounds':2,
             'max_additional_operational_retries_per_scientific_prompt':MAX_OPERATIONAL_RETRIES,
             'recovery_count':recovery_count,'recovery':recovery,
             'recovery_record':cfg['G5c'].get('recovery_record'),
             'recovery_path':cfg['G5c'].get('recovery_path',cfg['G5c'].get('recovery_record',recovery.get('archive_path',recovery.get('path'))))}
    write_json(WORK/'session.json',session)
    stop=threading.Event();lock=threading.RLock();reserved=0;completed=0;failure_streak=[];stop_info={}
    results={k:dict(c.get('results',{})) for k,c in cfg.items()};pending={k:[] for k in cfg};seen=set();seen_counts={k:0 for k in cfg};finished_experiments=set()
    def log(msg,exp=None):
        paths=[ROOT/FOLDERS[exp]/'run.log'] if exp else [ROOT/v/'run.log' for v in FOLDERS.values()]
        with lock:
            for p in paths:
                with p.open('a',encoding='utf8') as h:h.write(now()+' '+msg+'\n');h.flush()
    def append(exp,record):
        with lock,(ROOT/FOLDERS[exp]/'llm_calls.jsonl').open('a',encoding='utf8') as h:
            h.write(json.dumps(record,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n');h.flush();os.fsync(h.fileno())
    def flush(exp,force=False):
        if len(pending[exp])>=10 or (force and pending[exp]):
            with (ROOT/FOLDERS[exp]/'predictions.csv').open('a',newline='',encoding='utf8') as h:
                csv.DictWriter(h,fieldnames=FIELDS).writerows(pending[exp]);h.flush();os.fsync(h.fileno())
            pending[exp].clear();cfg[exp]['results']=results[exp];cfg[exp]['new_completed']=len(results[exp]);write_json(ROOT/FOLDERS[exp]/'config.json',cfg[exp])
    def finish_exp(exp):
        """Commit each experiment independently, including its final <10 rows."""
        if exp in finished_experiments:return
        flush(exp,True);c=cfg[exp];c.update(results=results[exp],status='inference_completed' if len(results[exp])==len(c['pending_jobs']) else 'stopped_partial',stop_reason=dict(stop_info),new_completed=len(results[exp]),completed_tasks=seen_counts[exp],completed_utc=now())
        write_json(ROOT/FOLDERS[exp]/'config.json',c)
        write_json(ROOT/FOLDERS[exp]/'metrics.json',{'status':c['status'],'new_valid':len(results[exp]),'new_requested':len(c['pending_jobs']),'stop_reason':dict(stop_info)})
        finished_experiments.add(exp);session['results_count'][exp]=len(results[exp]);session.setdefault('experiment_status',{})[exp]=c['status'];write_json(WORK/'session.json',session)
        log('INFERENCE_END '+c['status']+' valid_new='+str(len(results[exp])),exp)
    def worker(j):
        nonlocal reserved
        exp=j['experiment'];case=cases[exp][j['case_index']];obj=None;last=None;retry=0;requests=0
        for rnd in (1,2):
            operational_retry=0
            msg=modules[exp].prompt_for(case,repair=rnd==2) if exp=='G5c' else modules[exp].prompt_for(case,j['arm'],repair=rnd==2)
            if rnd==1:assert digest(msg)==j['prompt_sha256']
            schema=ROOT/('g5_response.schema.json' if j['kind']=='reg' else 'e2_response.schema.json')
            instructions=ROOT/('g5_instructions.md' if j['kind']=='reg' else 'e2_instructions.md')
            while not stop.is_set():
                with lock:
                    reserved+=1;requests+=1
                    meta={'call_id':reserved,'experiment':exp,'id':j['id'],'arm':j['arm'],'job_id':j['job_id'],'scientific_round':rnd,'transport_retry':retry,'operational_retry':operational_retry,'started_utc':now(),'prompt':msg,'request_hash':digest(msg),'availability_probe':reserved==1}
                    append(exp,{'event':'call_started',**meta})
                try:r=call_once(msg,schema_path=schema,instructions_path=instructions,catalog_path=ROOT/'e2_model_catalog.json',stop_event=stop,metadata=meta)
                except Exception as e:r={**meta,'status':'transport_error','response_text':'','transport_exception':repr(e)}
                if r['status']=='protocol_error' and not str(r.get('response_text') or '').strip():
                    r.update(original_transport_status='protocol_error',status='transport_error')
                if r['status']=='ok':
                    try:
                        raw=json.loads(r['response_text']);obj=validate_reg(raw) if j['kind']=='reg' else validate_output(raw,{'id':j['id'],'rule_cards':{}},'A3')
                        val=obj[j['score_key']];assert isinstance(val,(float,int)) and not isinstance(val,bool) and 0<=val<=1
                    except Exception as e:obj=None;r.update(status='protocol_error',protocol_error=repr(e))
                r.update(event='call_completed',finished_utc=now(),response_valid=obj is not None,parsed_response=obj);append(exp,r);last=r
                if r['status']=='terminal_usage_limit':
                    stop_info.update(kind='explicit_usage_limit',recovery_time_message=r.get('recovery_time_message'),diagnostics=r.get('terminal_diagnostics',r.get('errors')),call_id=meta['call_id']);stop.set()
                elif r['status'] in INTEGRITY:stop_info.update(kind='integrity_violation',status=r['status']);stop.set()
                if obj is not None or stop.is_set():break
                if r['status']=='rate_limited_final' and retry<3:
                    delay=DELAYS[retry];retry+=1;log(f'FINAL429 {j["job_id"]} wait={delay}s retry={retry}/3',exp)
                    if stop.wait(delay):break
                    continue
                if r['status'] in OPERATIONAL_ERRORS and operational_retry<MAX_OPERATIONAL_RETRIES:
                    operational_retry+=1;log(f'OPERATIONAL_RETRY {j["job_id"]} status={r["status"]} retry={operational_retry}/{MAX_OPERATIONAL_RETRIES} same_scientific_round={rnd}',exp)
                    continue
                break
            if obj is not None or stop.is_set() or last is None or last['status']=='rate_limited_final':break
            # Only an invalid scientific response warrants the original repair prompt.
            # Exhausted operational failures never change the scientific prompt.
            if last['status']!='protocol_error':break
        return j,obj,(last['status'] if last else 'not_dispatched'),requests
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        for exp in cfg:
            cfg[exp]['status']='running';cfg[exp]['global_runner_source_sha256']=sha(ROOT/'g_supplement_run.py');cfg[exp]['global_session_path']=str(WORK/'session.json');write_json(ROOT/FOLDERS[exp]/'config.json',cfg[exp])
        session['status']='running';write_json(WORK/'session.json',session);log(f'START finite global queue{len(jobs)}; probe alone then4 workers; recovery_count={recovery_count}; no further automatic restart')
        probe=False;exhausted=False;it=iter(jobs)
        with ThreadPoolExecutor(max_workers=4) as pool:
            active={pool.submit(worker,next(it)):None}
            while active:
                done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
                for future in done:
                    active.pop(future);j,obj,status,nreq=future.result();exp=j['experiment'];key=(exp,j['job_id']);assert key not in seen;seen.add(key);completed+=1;seen_counts[exp]+=1
                    case=cases[exp][j['case_index']]
                    row={'id':j['id'],'fold':case.get('fold',''),'group':case.get('group',''),'arm':j['arm'],'status':'valid' if obj is not None else status,'score':'','y_pred':''}
                    if obj is not None:
                        score=float(obj[j['score_key']]);results[exp][j['job_id']]=obj;row.update(score=score,y_pred=int(score>=.5));failure_streak.clear();probe=True
                    elif j['id'] not in failure_streak:failure_streak.append(j['id'])
                    pending[exp].append(row);flush(exp)
                    log(f'PROGRESS global={completed}/{len(jobs)} exp={exp} valid_new={len(results[exp])} id={j["id"]} arm={j["arm"]} status={status} calls={reserved}',exp)
                    if len(failure_streak)>=3 and not stop.is_set():stop_info.update(kind='three_consecutive_distinct_exhausted_materials',ids=failure_streak[-3:]);stop.set()
                    if completed==1:log('PROBE_COMPLETED status='+status)
                    if seen_counts[exp]==len(cfg[exp]['pending_jobs']):finish_exp(exp)
                while not stop.is_set() and not exhausted and len(active)<(4 if probe else 1):
                    try:active[pool.submit(worker,next(it))]=None
                    except StopIteration:exhausted=True
        for exp in cfg:finish_exp(exp)
        session.update(status='inference_completed' if all(c['status']=='inference_completed' for c in cfg.values()) else 'stopped_partial',completed_utc=now(),calls=reserved,completed_tasks=completed,results_count={k:len(v) for k,v in results.items()},stop_reason=stop_info);write_json(WORK/'session.json',session)
        return 0 if session['status']=='inference_completed' else 75
    except BaseException:
        stop.set();log('FATAL '+traceback.format_exc())
        for exp in cfg:
            if exp in finished_experiments:continue
            flush(exp,True);cfg[exp].update(status='interrupted_error',results=results[exp],stop_reason={'kind':'process_error','traceback':traceback.format_exc()});write_json(ROOT/FOLDERS[exp]/'config.json',cfg[exp])
        session.update(status='interrupted_error',calls=reserved,completed_tasks=completed,traceback=traceback.format_exc());write_json(WORK/'session.json',session);raise
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':raise SystemExit(run())
