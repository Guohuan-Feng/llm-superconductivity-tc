"""Finite E2 only: frozen packets, local Codex, 10 concurrent calls, six files."""
import sys
from pathlib import Path
_ROOT=Path('C:/Users/28908/Codex/fung-research/v2-diagnosis').resolve()
_OUT=(_ROOT/'E2').resolve()
assert _OUT.is_relative_to(_ROOT) and 'onedrive' not in str(_OUT).lower()
_OUT.mkdir(exist_ok=True)
sys.stderr=(_OUT/'run.log').open('a',encoding='utf-8',buffering=1)
sys.stdout=sys.stderr
from common import *
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import ctypes
from datetime import datetime, timezone
import hashlib
import logging
import math
import subprocess
import threading
import time
import traceback
from e2_packets import build_packets, compact_evidence, requested_evidence
from e2_prompts import SCHEMA, STATIC_PREFIX, make_prompt
from e2_transport import codex_call, input_budget, MODEL, REASONING_EFFORT, CLI_VERSION
from e2_metrics import evaluate_e2

OUT=_OUT
SOURCE_SCHEMA=ROOT/'e2_response.schema.json'
SOURCE_INSTRUCTIONS=ROOT/'e2_instructions.md'
SOURCE_CATALOG=ROOT/'e2_model_catalog.json'
GIT_EXE='C:/Users/28908/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe'
MAX_CALLS=2500
MAX_ROUNDS={'A1':3,'A2':2,'A3':2}
SLOT=65536
PREFIX_HASH=hashlib.sha256(STATIC_PREFIX.encode()).hexdigest()
STOP_EVENT=threading.Event()

def utcnow(): return datetime.now(timezone.utc).isoformat()
def sha_file(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def head(): return subprocess.check_output([GIT_EXE,'rev-parse','HEAD'],cwd=ROOT,text=True).strip()

class CallStore:
    """One JSONL line per attempt, including an fsynced reservation before call.

    Whitespace-padded fixed-width lines permit crash-safe in-place completion
    without extra files or duplicate call-start/call-end records. An interrupted
    reservation remains counted against the budget and is never silently replayed.
    """
    def __init__(self):
        self.path=safe_path(OUT/'llm_calls.jsonl'); self.lock=threading.RLock(); self.consecutive_transport_failures=0
        self.path.touch(exist_ok=True); self.records=[]
        with self.path.open('rb') as f:
            for line in f:
                if len(line)!=SLOT+1: raise RuntimeError('Incomplete call-log reservation; stop for diagnosis')
                self.records.append(json.loads(line))
        for j,r in enumerate(self.records):
            if r['status']=='in_progress':
                self.update(j,dict(status='interrupted_unknown_completion',error='Previous process ended during this reserved attempt'))

    def _write(self,j,r):
        raw=json.dumps(r,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')
        if len(raw)>SLOT: raise RuntimeError('Call record exceeds fixed slot')
        with self.path.open('r+b') as f:
            f.seek(j*(SLOT+1)); f.write(raw+b' '*(SLOT-len(raw))+b'\n'); f.flush(); os.fsync(f.fileno())

    def reserve(self,arm,key,round_number,prompt):
        with self.lock:
            if STOP_EVENT.is_set(): raise RuntimeError('E2 stop already requested')
            if len(self.records)>=MAX_CALLS:
                STOP_EVENT.set(); raise RuntimeError('GLOBAL_CALL_BUDGET_EXHAUSTED')
            j=len(self.records)
            r=dict(call_id=j,arm=arm,id=key,round=round_number,status='in_progress',started_utc=utcnow(),
                request_hash=hashlib.sha256(prompt.encode()).hexdigest(),static_prefix_sha256=PREFIX_HASH,
                prompt=prompt,model=MODEL,reasoning_effort=REASONING_EFFORT,cli_version=CLI_VERSION,
                input_tokens=None,output_tokens=None,cached_input_tokens=None,elapsed_seconds=None)
            self.records.append(r); self._write(j,r)
            return j

    def update(self,j,fields):
        with self.lock:
            self.records[j].update(fields); self._write(j,self.records[j]); return dict(self.records[j])

    def invoke(self,arm,key,round_number,prompt):
        if not prompt.startswith(STATIC_PREFIX): raise RuntimeError('Static prefix changed')
        bound=input_budget(prompt,schema_path=SOURCE_SCHEMA,instructions_path=SOURCE_INSTRUCTIONS)
        if not bound['within_budget']: raise RuntimeError(f'Input too large before call: {arm}/{key}: {bound}')
        j=self.reserve(arm,key,round_number,prompt)
        result=codex_call(prompt,cwd=OUT,schema_path=SOURCE_SCHEMA,instructions_path=SOURCE_INSTRUCTIONS,
            catalog_path=SOURCE_CATALOG,timeout_seconds=None,
            metadata=dict(arm=arm,id=key,round=round_number))
        result['finished_utc']=utcnow()
        row=self.update(j,result)
        if row['status'] in ('tool_activity_rejected','actual_input_budget_violation','model_mismatch_rejected','backend_protocol_rejected'):
            STOP_EVENT.set()
            raise RuntimeError(f'EXPERIMENT_INTEGRITY_STOP: {row["status"]}')
        diagnostic=' '.join([str(row.get('error','')),str(row.get('errors','')),str(row.get('stderr_tail',''))]).lower()
        if row['status']!='ok' and any(x in diagnostic for x in ['usage limit','quota exceeded','insufficient_quota','not authenticated','unauthorized','authentication failed','rate_limit_exceeded']):
            STOP_EVENT.set(); raise RuntimeError('TERMINAL_PROVIDER_OR_QUOTA_STOP: '+row['status'])
        with self.lock:
            self.consecutive_transport_failures=0 if row['status']=='ok' else self.consecutive_transport_failures+1
            if self.consecutive_transport_failures>=10:
                STOP_EVENT.set(); raise RuntimeError('REPEATED_TRANSPORT_FAILURE_STOP')
        return j,row

def prepare(task_name):
    if (OUT/'config.json').exists(): raise RuntimeError('E2 already prepared; use existing frozen config')
    d=load_data(); prepared=build_packets(d,log=logging.info)
    cases=prepared['cases']
    config=dict(experiment='E2',stage=2,scope='development only',hypothesis='Separate rule retrieval/selection from chemical reasoning and nearest-neighbor evidence.',
        git_commit=head(),created_utc=utcnow(),task_name=task_name,
        model=MODEL,model_version_note='CLI exposes model slug; immutable backend weight snapshot not reported. Same explicit slug for all arms.',
        cli_version=CLI_VERSION,reasoning_effort=REASONING_EFFORT,configured_user_default_effort='ultra',
        params=dict(n=300,n_groups=81,max_total_calls=MAX_CALLS,planned_max_material_calls=2100,
            max_rounds=MAX_ROUNDS,concurrency=10,input_limit_tokens=15000,threshold=.5,
            bootstrap_draws=5000,bootstrap_seed=20261016,neighbors=5,min_rule_support_groups=20,
            top_candidate_rules=6,selected_rules=[1,3],max_counterexamples_per_rule=10,delivered_counterexamples_per_rule_cap=5,
            second_A2_A3_round='protocol/transport repair only; no new scientific evidence',
            A1_abstention_forced='Always select best1-3 rules even while abstaining; harness uses exact mean.',
            prompt_numeric_precision_significant_digits=8),
        source_hashes={str(p.name):sha_file(p) for p in [SOURCE_SCHEMA,SOURCE_INSTRUCTIONS,SOURCE_CATALOG,
            *[ROOT/name for name in ['e2_prompts.py','e2_packets.py','e2_transport.py','e2_metrics.py','run_stage2.py','common.py','stats.py','e1_rules.py','launch_stage2.ps1']]]},
        static_prefix_sha256=PREFIX_HASH,static_prefix=STATIC_PREFIX,
        metadata=prepared['metadata'],indices=prepared['indices'],cases=cases)
    sizes={a:[] for a in MAX_ROUNDS}
    for c in cases.values():
        for arm in MAX_ROUNDS:
            p=prompt_for(c,arm,[],{})
            bound=input_budget(p,schema_path=SOURCE_SCHEMA,instructions_path=SOURCE_INSTRUCTIONS)
            if not bound['within_budget']: raise RuntimeError(f'Initial prompt too large {arm}/{c["id"]}: {bound}')
            sizes[arm].append(bound['input_token_upper_bound'])
        # All evidence plus two maximum schema-size request histories is the
        # largest A1 followup planned; verify without looking at any outcome.
        history=[dict(action='request',p=None,abstain=False,selected_rule_ids=[],
            request_rule_ids=list(c['rule_cards']),request_counterexamples=True,reason='𠀀'*80)]*2
        all_evidence={k:v for k,v in c['rule_cards'].items()}
        bound=input_budget(prompt_for(c,'A1',history,all_evidence),schema_path=SOURCE_SCHEMA,instructions_path=SOURCE_INSTRUCTIONS)
        if not bound['within_budget']: raise RuntimeError(f'A1 followup too large {c["id"]}: {bound}')
    config['initial_input_bounds']={a:dict(min=min(v),max=max(v)) for a,v in sizes.items()}
    write_json(OUT/'config.json',config)
    safe_path(OUT/'llm_calls.jsonl').touch()
    PredictionWriter(OUT/'predictions.csv',d)
    write_json(OUT/'metrics.json',dict(status='prepared',n=300,n_groups=81,scientific_calls=0))
    safe_path(OUT/'README.md').write_text('# E2: prepared\nDevelopment data:300 materials /81 groups.\nNo model calls yet. Stage3 is not authorized.\n',encoding='utf8')
    logging.info('PREPARED n=300 groups=81 planned_material_call_bound=2100 initial_input_bounds=%s',config['initial_input_bounds'])

def prompt_for(c,arm,history,evidence):
    if arm=='A2':
        ev=compact_evidence(c['rule_cards'],rule_summaries=c['rule_summaries'])
    elif arm=='A1':
        ev=compact_evidence(evidence,rule_summaries=c['rule_summaries']) if evidence else {}
    else: ev={}
    positions={r['rule_id']:j for j,r in enumerate(c['rule_summaries'])}
    short_history=[]
    for response in history:
        if response.get('action')=='request':
            short_history.append(dict(action='request',request_initial_rule_indices=[positions[r] for r in response['request_rule_ids']],
                request_counterexamples=response['request_counterexamples'],reason=response['reason']))
        else: short_history.append(response)
    return make_prompt(arm,c['material'],c['rule_summaries'] if arm!='A3' else [],ev,
        c['neighbours'] if arm=='A2' else [],short_history,round_number=len(history)+1,
        force_decision=(arm=='A1' and len(history)>=2))

def validate_output(response,c,arm):
    if not isinstance(response,dict) or set(response)!=set(SCHEMA['required']): raise ValueError('JSON object fields differ from frozen schema')
    if response['action'] not in ('request','decide') or not isinstance(response['abstain'],bool): raise ValueError('Invalid action/abstain')
    if not isinstance(response['reason'],str) or len(response['reason'])>80: raise ValueError('Reason exceeds80 characters')
    for k in ('selected_rule_ids','request_rule_ids'):
        v=response[k]
        if not isinstance(v,list) or len(v)!=len(set(v)) or any(not isinstance(x,str) or x not in c['rule_cards'] for x in v):
            raise ValueError('Unknown or duplicate rule ID')
    if not isinstance(response['request_counterexamples'],bool): raise ValueError('Invalid evidence flag')
    if response['action']=='request':
        if arm!='A1' or not response['request_rule_ids'] or response['selected_rule_ids'] or response['p'] is not None: raise ValueError('Invalid evidence request')
        return None
    p=response['p']
    if isinstance(p,bool) or not isinstance(p,(int,float)) or not math.isfinite(p) or not 0<=p<=1: raise ValueError('Missing/invalid forced probability')
    if response['request_rule_ids'] or response['request_counterexamples']: raise ValueError('Decision also requested evidence')
    selected=response['selected_rule_ids']
    if arm=='A1':
        if not 1<=len(selected)<=3: raise ValueError('A1 requires1-3 backup rules, including during abstention')
        p=math.fsum(float(c['rule_cards'][r]['class_balanced_score']) for r in sorted(selected))/len(selected)
    elif len(selected)>3 or (arm=='A3' and selected): raise ValueError('Invalid selected rules for direct arm')
    return dict(id=c['id'],arm=arm,p=float(p),abstain=response['abstain'],selected_rules=selected,
        reported_p=response['p'],reason=response['reason'],status='ok')

def run_case(store,c,arm):
    if STOP_EVENT.is_set(): raise RuntimeError('E2 stop already requested')
    prior=[r for r in store.records if r['id']==c['id'] and r['arm']==arm]
    if any('case_result' in r for r in prior): return next(r['case_result'] for r in reversed(prior) if 'case_result' in r)
    history=[]; evidence={}; last=None; failure='no_final_decision'
    for row in prior:
        response=row.get('parsed_response')
        if response:
            history.append(response)
            if response['action']=='request':
                for item in requested_evidence(c,response['request_rule_ids'],include_counterexamples=response['request_counterexamples']):
                    evidence[item['rule_id']]={**evidence.get(item['rule_id'],{}),**item}
        else: history.append(dict(protocol_error=str(row.get('protocol_error',row['status']))[:240],instruction='Return one valid decision now.'))
    for round_number in range(len(prior)+1,MAX_ROUNDS[arm]+1):
        if STOP_EVENT.is_set(): raise RuntimeError('E2 stop already requested')
        prompt=prompt_for(c,arm,history,evidence)
        j,row=store.invoke(arm,c['id'],round_number,prompt); last=j
        response=None
        try:
            if row['status']!='ok': raise ValueError(row.get('error') or row['status'])
            response=json.loads(row['response_text'])
            final=validate_output(response,c,arm)
            if final is None:
                if round_number>=MAX_ROUNDS[arm]: raise ValueError('Final allowed round still requested evidence')
                for item in requested_evidence(c,response['request_rule_ids'],include_counterexamples=response['request_counterexamples']):
                    evidence[item['rule_id']]={**evidence.get(item['rule_id'],{}),**item}
                store.update(j,dict(parsed_response=response)); history.append(response); continue
            store.update(j,dict(parsed_response=response,case_result=final))
            return final
        except (ValueError,TypeError,KeyError) as exc:
            failure=str(exc)
            # Infrastructure failures retain their status; a successful CLI
            # response violating our protocol is an explicit protocol error.
            fields=dict(protocol_error=failure)
            if row['status']=='ok': fields['status']='protocol_error'
            store.update(j,fields)
            history.append(dict(protocol_error=failure[:240],instruction='Repair format/decision using exactly the same evidence.'))
    final=dict(id=c['id'],arm=arm,p=None,abstain=False,selected_rules=[],reason='',status='failure',error=failure)
    if last is None and prior: last=prior[-1]['call_id']
    if last is not None: store.update(last,dict(case_result=final))
    return final

def write_case_predictions(d,results):
    path=safe_path(OUT/'predictions.csv')
    with path.open(newline='',encoding='utf8') as f: done={(r['id'],r['arm']) for r in csv.DictReader(f)}
    by_id={str(v):i for i,v in enumerate(d['id'])}
    rows=[]
    for r in results:
        i=by_id[r['id']]; valid=r['status']=='ok'; p=r['p']
        for mode in ('forced','HGBfallback'):
            arm=r['arm']+'_'+mode
            if (r['id'],arm) in done: continue
            if mode=='forced': score=p if valid else None; status=('abstain_forced_backup' if r['abstain'] else 'ok') if valid else 'failure_no_forced_probability'
            elif not valid: score=float(d['hgb'][i]); status='failure_HGB_fallback'
            elif r['abstain']: score=float(d['hgb'][i]); status='explicit_abstention_HGB_fallback'
            else: score=p; status='ok'
            rows.append(dict(id=r['id'],fold=int(d['fold'][i]),group=str(d['group'][i]),arm=arm,status=status,
                score='' if score is None else score,y_pred='' if score is None else int(score>=.5)))
            done.add((r['id'],arm))
    if rows:
        with path.open('a',newline='',encoding='utf8') as f:
            csv.DictWriter(f,fieldnames=FIELDS).writerows(rows); f.flush(); os.fsync(f.fileno())

def read_config():
    config=json.loads((OUT/'config.json').read_text(encoding='utf8'))
    for name,value in config['source_hashes'].items():
        if sha_file(ROOT/name)!=value: raise RuntimeError('Frozen input source changed: '+name)
    if config['static_prefix_sha256']!=PREFIX_HASH: raise RuntimeError('Static prefix changed')
    return config

def render_summary(metrics):
    def fmt(x):
        if x['point'] is None:return 'NA'
        return f'{100*x["point"]:.1f} [{100*x["ci95"][0]:.1f},{100*x["ci95"][1]:.1f}]'
    lines=['# E2 Stage2: development results',
        f'All300 materials /81 association groups; model={MODEL}, effort={REASONING_EFFORT};5000 group bootstrap draws;95%CI.',
        '|Mode|n/groups|BA% [CI]|Precision% [CI]|Recall% [CI]|','|---|---|---|---|---|']
    for arm in MAX_ROUNDS:
        a=metrics['arms'][arm]
        for key,name in [('forced_valid_cases',arm+'_forced'),('abstention_or_failure_HGB_fallback_all_cases',arm+'_HGBfallback')]:
            s=a[key]
            if s:
                m=s['metrics'][name];lines.append(f'|{name}|{s["n"]}/{s["n_groups"]}|{fmt(m["balanced_accuracy"])}|{fmt(m["precision"])}|{fmt(m["recall"])}|')
    for name,m in metrics['baselines']['metrics'].items():
        lines.append(f'|{name}|300/81|{fmt(m["balanced_accuracy"])}|{fmt(m["precision"])}|{fmt(m["recall"])}|')
    lines+=['Paired differences, abstention/failure rates, tokens and call time (each with group bootstrap CI) are in metrics.json.',
        'Full prompts, <=80-character reasons, requested evidence and selected rules remain in llm_calls.jsonl for later authorized error analysis.',
        'A1 forced scores use the exact mean of selected frozen scores; even abstainers supply1-3 backup rules.',
        'Stage2 complete. Stop here; Stage3 requires a new human confirmation.']
    assert len(lines)<=20
    return '\n'.join(lines)+'\n'

def evaluate(d,config,store):
    results={a:{} for a in MAX_ROUNDS}
    for row in store.records:
        if 'case_result' in row: results[row['arm']][row['id']]=row['case_result']
    cases=config['cases']; indices=config['indices']
    controls={'R-info':np.asarray([cases[str(d['id'][i])]['rinfo_score'] for i in indices]),
              '5NN':np.asarray([cases[str(d['id'][i])]['knn_score'] for i in indices])}
    calls=[r for r in store.records if r['arm'] in MAX_ROUNDS]
    metrics=evaluate_e2(d,indices,results,calls,controls)
    complete=all(len(results[a])==300 for a in MAX_ROUNDS)
    metrics['completion']=dict(status='complete' if complete else 'partial',finished_utc=utcnow(),git_commit=config['git_commit'],
        actual_total_codex_calls=len(store.records),material_calls=len(calls),model=MODEL,reasoning_effort=REASONING_EFFORT,
        cli_version=CLI_VERSION,static_prefix_sha256=PREFIX_HASH,case_results={a:len(v) for a,v in results.items()})
    write_json(OUT/'metrics.json',metrics)
    if complete: safe_path(OUT/'README.md').write_text(render_summary(metrics),encoding='utf8')
    return metrics

def execute(task_name):
    started=time.monotonic()
    config=read_config(); d=load_data(); store=CallStore()
    cases=config['cases']
    prediction_path=safe_path(OUT/'predictions.csv')
    raw=prediction_path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        keep=raw.rfind(b'\n')+1
        if keep==0: raise RuntimeError('Prediction header incomplete')
        with prediction_path.open('r+b') as f: f.truncate(keep)
        logging.warning('Removed torn final CSV line; completed call results will restore it')
    writer=PredictionWriter(prediction_path,d)
    ix=np.asarray(config['indices'])
    for name,values in [('R-info',[cases[str(d['id'][i])]['rinfo_score'] for i in ix]),
                        ('5NN',[cases[str(d['id'][i])]['knn_score'] for i in ix]),('HGB127',d['hgb'][ix])]:
        writer.append(name,ix,values,['deterministic_TRAIN_only_control']*len(ix))
    finished={(r['arm'],r['id']):r['case_result'] for r in store.records if 'case_result' in r}
    write_case_predictions(d,list(finished.values()))
    jobs=[(arm,key) for key in cases for arm in MAX_ROUNDS if (arm,key) not in finished]
    jobs.sort(key=lambda x:hashlib.sha256(('E2-interleaved-20261016|'+x[0]+'|'+x[1]).encode()).digest())
    logging.info('RUN task=%s pending=%d existing_final=%d calls_consumed=%d concurrency=10',task_name,len(jobs),len(finished),len(store.records))
    pending=[]; count=len(finished)
    pool=ThreadPoolExecutor(max_workers=10,thread_name_prefix='E2')
    try:
        futures={pool.submit(run_case,store,cases[key],arm):(arm,key) for arm,key in jobs}
        for future in as_completed(futures):
            result=future.result(); pending.append(result); count+=1
            if len(pending)>=10:
                write_case_predictions(d,pending); pending=[]
                logging.info('CHECKPOINT completed_arm_cases=%d/900 calls=%d',count,len(store.records))
    except BaseException:
        STOP_EVENT.set()
        for future in futures: future.cancel()
        raise
    finally:
        pool.shutdown(wait=True,cancel_futures=True)
    write_case_predictions(d,pending)
    metrics=evaluate(d,config,store)
    metrics['completion']['run_wall_clock_seconds']=time.monotonic()-started
    write_json(OUT/'metrics.json',metrics)
    assert metrics['completion']['status']=='complete'
    assert {f.name for f in OUT.iterdir()}=={'config.json','predictions.csv','llm_calls.jsonl','metrics.json','run.log','README.md'}
    logging.info('COMPLETE E2 all900 arm-cases totalcalls=%d; no Stage3 execution',len(store.records))

def probe():
    read_config(); store=CallStore()
    if any(r['arm']=='PREFLIGHT' and r['status']=='ok' for r in store.records): raise RuntimeError('Successful probe already recorded')
    prompt=STATIC_PREFIX+'\nTransport conformance only, no real material: return action=decide,p=0.5,abstain=true,selected_rule_ids=[],request_rule_ids=[],request_counterexamples=false,reason="transport probe". Do not call any tool.'
    j,row=store.invoke('PREFLIGHT','synthetic_transport',1,prompt)
    if row['status']!='ok': raise RuntimeError('Local Codex conformance probe failed: '+row['status'])
    response=json.loads(row['response_text'])
    assert response['action']=='decide' and response['p']==.5 and response['abstain'] is True
    store.update(j,dict(parsed_response=response))
    logging.info('PREFLIGHT PASS model=%s version=%s actual_input_tokens=%s output_tokens=%s elapsed=%s',
        MODEL,row.get('model_version_reported'),row['input_tokens'],row['output_tokens'],row['elapsed_seconds'])

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','probe','run','evaluate']);p.add_argument('--task-name',default='manual')
    args=p.parse_args()
    logging.basicConfig(filename=str(OUT/'run.log'),level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',encoding='utf8',force=True)
    awake=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    if not awake:raise RuntimeError('SetThreadExecutionState failed')
    try:
        if args.mode=='prepare':prepare(args.task_name)
        elif args.mode=='probe':probe()
        elif args.mode=='run':execute(args.task_name)
        else:evaluate(load_data(),read_config(),CallStore())
    except BaseException:
        logging.error('STOPPED\n%s',traceback.format_exc());raise
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__': main()
