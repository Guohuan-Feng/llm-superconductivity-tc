"""One-workday Stage4 P1 only; frozen data, CUDA verifier and three rule versions."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
import ctypes
from datetime import datetime,timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('C:/Users/28908/Codex/fung-research/v2-diagnosis').resolve()
OUT=ROOT/'P1'
FIELDS=['id','fold','group','arm','status','score','y_pred']
SLOT=65536
SCHEMA=ROOT/'p1_response.schema.json'
INSTRUCTIONS=ROOT/'p1_instructions.md'
SOURCES=['run_p1.py','launch_p1.ps1','p1_data.py','p1_verifier.py','p1_baseline.py',
         'p1_rules.py','p1_response.schema.json','p1_instructions.md','e2_transport.py','e2_model_catalog.json']

def safe(path):
    path=Path(path).resolve()
    if 'onedrive' in str(path).casefold() or not path.is_relative_to(ROOT):raise ValueError('Forbidden write path: '+str(path))
    return path

def utc():return datetime.now(timezone.utc).isoformat()
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    raw=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    with safe(path).open('w',encoding='utf-8') as h:h.write(raw);h.flush();os.fsync(h.fileno())

def logger():
    log=logging.getLogger('P1');log.setLevel(logging.INFO);log.handlers.clear()
    handler=logging.FileHandler(safe(OUT/'run.log'),encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'));log.addHandler(handler)
    return log.info

def check_deadline(cfg):
    if datetime.now(timezone.utc)>=datetime.fromisoformat(cfg['deadline_utc']):
        raise RuntimeError('One-workday deadline reached; stop P1 and report partial results, never continue Stage5')

class Writer:
    def __init__(self,d,threshold):
        self.d=d;self.threshold=threshold;self.path=safe(OUT/'predictions.csv');self.done=set()
        if self.path.exists():
            with self.path.open(encoding='utf-8',newline='') as h:
                for r in csv.DictReader(h):self.done.add((r['arm'],r['id']))
        else:
            with self.path.open('x',encoding='utf-8',newline='') as h:csv.DictWriter(h,fieldnames=FIELDS).writeheader()
    def append(self,arm,indices,values,statuses=None):
        statuses=['ok']*len(indices) if statuses is None else statuses
        rows=[]
        for i,value,status in zip(indices,values,statuses):
            identity=self.d['ids'][i];key=(arm,identity)
            if key in self.done:continue
            finite=value is not None and math.isfinite(float(value))
            rows.append({'id':identity,'fold':-1 if self.d['splits'][i]=='TRAIN' else 0,
                         'group':self.d['groups'][i],'arm':arm,'status':status,
                         'score':float(value) if finite else '',
                         'y_pred':int(float(value)<self.threshold) if finite else ''})
            self.done.add(key)
            if len(rows)==10:self.flush(rows);rows=[]
        if rows:self.flush(rows)
    def flush(self,rows):
        with self.path.open('a',encoding='utf-8',newline='') as h:
            csv.DictWriter(h,fieldnames=FIELDS).writerows(rows);h.flush();os.fsync(h.fileno())

def prepare(task_name):
    from p1_data import load_prepare,source_metadata
    from e2_transport import MODEL,CLI_VERSION,REASONING_EFFORT
    if OUT.exists():raise RuntimeError('P1 exists; resume frozen configuration')
    d=load_prepare()
    safe(OUT).mkdir()
    cfg={'experiment':'P1','stage':4,'hypothesis':'A local approximate energy verifier can support a bounded, transparent rule-generation/feedback loop within one workday.',
         'created_utc':utc(),'authorized_start_utc':'2026-10-17T02:02:00+00:00',
         'deadline_utc':'2026-10-17T10:02:00+00:00','scope':'Feasibility pilot only; no Stage5, no independent-pretraining generalization claim',
         'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         'task_name':task_name,'model':MODEL,'cli_version':CLI_VERSION,'reasoning_effort':REASONING_EFFORT,
         'source_hashes':{name:digest(ROOT/name) for name in SOURCES},
         'dataset':{**source_metadata(),**d['metadata']},
         'targets_by_id':dict(zip(d['ids'],d['targets'])),
         'params':{'total_structures':10000,'train':8000,'holdout':2000,'verifier_train_bank':1024,'verifier_holdout_test':200,
                   'rule_versions':3,'interpretation':'Initial generation plus two TRAIN-feedback revisions; five full predictors/version, equal-weight mean.',
                   'rules_per_version':5,'max_calls':6,'max_attempts_per_version':2,'input_limit_tokens':15000,
                   'classification_threshold_eV_per_atom':0.0,'positive_definition':'formation_energy_per_atom < threshold, NOT hull stability',
                   'bootstrap_draws':2000,'bootstrap_seed':20261017,'automatic_process_restarts_allowed':1,
                   'LLM_concurrency':'One dependent generation/version; three sequential versions, no independent parallel calls.',
                   'checkpoint_prediction_rows':10},
         'prediction_columns':{'score':'formation energy eV/atom; not a probability','y_pred':'score < configurable threshold','fold':'-1 TRAIN,0 HOLDOUT'},
         'LLM_information_boundary':'ONLY1024 TRAIN-bank CHGNet formation references and structure descriptors; no DFT HOLDOUT labels, metrics, identifiers or feedback.',
         'verifier_input_boundary':'Elements/lattice/coordinates only; ALL site properties including DFT magmom removed.',
         'feasibility_gates':['10k dataset with license/hash and composition-disjoint split',
                              'CUDA energy inference and reference conversion on200heldout attempts with explicit coverage/error',
                              'Three versions of five executable rules; TRAIN feedback and heldout scores',
                              'Magpie+RF heldout control and public GNN context',
                              'Within8hours, finite calls, no Stage5 dispatch'],
         'model_version_note':'Explicit local model alias; immutable backend weight version not reported.'}
    write(OUT/'config.json',cfg)
    write(OUT/'metrics.json',{'status':'prepared','created_utc':utc(),'verifier_records':{},'rule_versions':[]})
    Writer(d,0.0)
    safe(OUT/'llm_calls.jsonl').touch(exist_ok=False)
    safe(OUT/'run.log').write_text(utc()+' frozen P1 data/protocol prepared\n',encoding='utf-8')
    safe(OUT/'README.md').write_text('# P1 formation-energy feasibility\nPrepared. Stage4 only; no Stage5.\n',encoding='utf-8')
    print('P1_PREPARED')

def verify(cfg):
    from p1_data import PILOT_PATH
    for name,expected in cfg['source_hashes'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Frozen P1 source changed: '+name)
    if digest(PILOT_PATH)!=cfg['dataset']['pilot_sha256']:raise ValueError('Frozen pilot changed')

def read_calls():
    records=[]
    with (OUT/'llm_calls.jsonl').open('rb') as h:
        for n,line in enumerate(h):
            if len(line)!=SLOT+1:raise ValueError('Incomplete P1 call-log slot; stop for diagnosis')
            record=json.loads(line)
            if record['call_id']!=n:raise ValueError('P1 call identity mismatch')
            records.append(record)
    return records

def save_call(index,record):
    raw=json.dumps(record,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')
    if len(raw)>SLOT:raise ValueError('Full P1 call exceeds fixed log slot; no truncation')
    with safe(OUT/'llm_calls.jsonl').open('r+b') as h:
        h.seek(index*(SLOT+1));h.write(raw+b' '*(SLOT-len(raw))+b'\n');h.flush();os.fsync(h.fileno())

def bounded_prompt(iteration,context,previous_code,train_feedback):
    from e2_transport import input_budget
    from p1_rules import make_prompt
    for examples,bins,counterexamples in [(8,100,8),(4,24,4),(0,12,2)]:
        reduced=copy.deepcopy(context)
        reduced['labelblind_TRAIN_examples']=reduced['labelblind_TRAIN_examples'][:examples]
        reduced['conditional_TRAIN_statistics']=reduced['conditional_TRAIN_statistics'][:bins]
        fb=copy.deepcopy(train_feedback)
        if fb is not None:fb['counterexamples']=fb['counterexamples'][:counterexamples]
        prompt=make_prompt(iteration,reduced,previous_code,fb)
        budget=input_budget(prompt,schema_path=SCHEMA,instructions_path=INSTRUCTIONS)
        if budget['within_budget']:return prompt,budget
    raise ValueError('P1 prompt cannot fit frozen input budget without losing required evidence')

def generation(iteration,prompt,feature_rows,cfg,log):
    from e2_transport import codex_call,input_budget
    from p1_rules import execute_rules
    prior=[r for r in read_calls() if r['iteration']==iteration]
    for record in prior:
        if record['status']=='in_progress':
            record.update(status='interrupted_unknown_completion',finished_utc=utc())
            save_call(record['call_id'],record)
        if record.get('valid'):
            return record['parsed_response'],execute_rules(record['parsed_response']['rules_code'],feature_rows)
    if len(prior)>=2:raise RuntimeError('P1 version exhausted two attempts')
    for attempt in range(len(prior)+1,3):
        check_deadline(cfg)
        actual=prompt
        if prior:
            actual+='\nFORMAT/RUNTIME REPAIR ONLY; same TRAIN evidence. Previous error: '+str(prior[-1].get('validation_error',prior[-1].get('status')))[:180]
        budget=input_budget(actual,schema_path=SCHEMA,instructions_path=INSTRUCTIONS)
        if not budget['within_budget']:raise ValueError('P1 repair exceeds input limit')
        calls=read_calls();index=len(calls)
        if index>=6:raise RuntimeError('P1 six-call cap reached')
        record={'call_id':index,'iteration':iteration,'attempt':attempt,'prompt':actual,'request_hash':hashlib.sha256(actual.encode()).hexdigest(),
                'started_utc':utc(),'status':'in_progress','valid':False,'input_budget':budget}
        save_call(index,record)
        remaining=(datetime.fromisoformat(cfg['deadline_utc'])-datetime.now(timezone.utc)).total_seconds()
        result=codex_call(actual,cwd=ROOT,schema_path=SCHEMA,instructions_path=INSTRUCTIONS,
                          catalog_path=ROOT/'e2_model_catalog.json',timeout_seconds=remaining,
                          metadata={'iteration':iteration,'attempt':attempt})
        record.update(result)
        record['finished_utc']=utc()
        execution=None
        if record['status']=='ok':
            try:
                parsed=json.loads(record['response_text'])
                if set(parsed)!={'iteration','rules_code','summary'} or parsed['iteration']!=iteration:raise ValueError('Rule response schema/iteration mismatch')
                record['parsed_response']=parsed
                execution=execute_rules(parsed['rules_code'],feature_rows)
                record['valid']=True
            except Exception as exc:
                record['validation_error']=type(exc).__name__+': '+str(exc)
                record['status']='rule_protocol_error'
        save_call(index,record)
        log(f'P1 rules version{iteration} attempt{attempt} status={record["status"]} valid={record["valid"]}')
        if record['valid']:return record['parsed_response'],execution
        if record.get('tool_activity') or record['status'] in {'actual_input_budget_violation','model_mismatch_rejected','backend_protocol_rejected'}:
            raise RuntimeError('P1 integrity failure; no further calls')
        prior.append(record)
    raise RuntimeError('P1 rule generation invalid after final allowed repair')

def timing_summary(records,groups):
    import numpy as np
    times=np.asarray([r['elapsed_seconds'] for r in records],dtype=float)
    group=np.asarray(groups)
    unique=np.unique(group)
    bygroup=[np.flatnonzero(group==g) for g in unique]
    rng=np.random.default_rng(20261017);draws=[]
    for _ in range(2000):
        ix=np.concatenate([bygroup[j] for j in rng.integers(0,len(unique),len(unique))])
        draws.append([float(times[ix].mean()),float(np.median(times[ix])),float(np.percentile(times[ix],95))])
    draws=np.asarray(draws)
    return {'n':len(records),'n_groups':len(unique),'bootstrap_draws':2000,
            'seconds':{name:{'point':point,'ci95':np.percentile(draws[:,j],[2.5,97.5]).tolist()}
                       for j,(name,point) in enumerate([('mean',float(times.mean())),('median',float(np.median(times))),('p95',float(np.percentile(times,95)))])}}

def run(task_name):
    import numpy as np
    from p1_data import load_prepare
    from p1_verifier import load_model,model_metadata,predict_energies
    from p1_baseline import run_magpie_rf,evaluate_p1
    from p1_rules import descriptors,training_context,feedback,execute_rules
    cfg=read(OUT/'config.json')
    if cfg['task_name']!=task_name:raise ValueError('Task identity mismatch')
    verify(cfg);check_deadline(cfg)
    metrics=read(OUT/'metrics.json')
    if metrics['status']=='complete':print('P1 already complete; no new work');return
    log=logger();start=time.perf_counter();log('Starting/resuming finite Stage4 P1 only')
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        d=load_prepare(as_pymatgen=True)
        for structure in d['structures']:
            for key in list(structure.site_properties):structure.remove_site_property(key)
        groups=np.asarray(d['groups']); y=np.asarray(d['targets'],dtype=float)
        sizes=dict(Counter(d['groups']));threshold=cfg['params']['classification_threshold_eV_per_atom']
        writer=Writer(d,threshold)
        cache=safe(ROOT/'work/p1-cache');os.chdir(cache)
        # Independent RF work shares no model/data labels with the rule generator.
        with ThreadPoolExecutor(max_workers=1) as pool:
            rf_future=pool.submit(run_magpie_rf,d['structures'],y,d['train_indices'],d['holdout_indices'],logger=log)
            cold=time.perf_counter();model=load_model();metrics['model_load_seconds']=time.perf_counter()-cold
            metrics['verifier_metadata']=model_metadata(model)
            # Synthetic warm-up is independent of all dataset structures.
            from pymatgen.core import Lattice,Structure
            silicon=Structure(Lattice.cubic(5.43),['Si']*8,[[0,0,0],[.25,.25,.25],[0,.5,.5],[.25,.75,.75],[.5,0,.5],[.75,.25,.75],[.5,.5,0],[.75,.75,.25]])
            metrics['synthetic_warmup']=predict_energies([silicon],model=model)[0]
            if metrics['synthetic_warmup']['status']!='ok':raise RuntimeError('CUDA warm-up failed')
            write(OUT/'metrics.json',metrics)
            for arm,key in [('CHGNet_verifier_test','verifier_test_indices'),('CHGNet_verifier_TRAIN','verifier_train_indices')]:
                indices=d[key]
                records=metrics['verifier_records']
                # Recover CSV from durably checkpointed records before any new inference.
                present=[i for i in indices if d['ids'][i] in records]
                writer.append(arm,present,[records[d['ids'][i]]['formation_energy_per_atom_eV'] for i in present],
                              [records[d['ids'][i]]['status'] for i in present])
                todo=[i for i in indices if d['ids'][i] not in records]
                pending=[]
                def checkpoint():
                    if not pending:return
                    write(OUT/'metrics.json',metrics)
                    writer.append(arm,[i for i,_ in pending],[r['formation_energy_per_atom_eV'] for _,r in pending],[r['status'] for _,r in pending]);pending.clear()
                def completed(position,result):
                    check_deadline(cfg)
                    i=todo[position];result.update(i=i,id=d['ids'][i],group=d['groups'][i],split=d['splits'][i])
                    records[d['ids'][i]]=result;pending.append((i,result))
                    if len(pending)==10:
                        checkpoint();log(f'{arm} checkpoint {sum(d["ids"][j] in records for j in indices)}/{len(indices)}')
                predict_energies([d['structures'][i] for i in todo],model=model,progress_callback=completed)
                checkpoint()
            del model
            import torch
            torch.cuda.empty_cache()
            feature_start=time.perf_counter()
            feature_rows=[descriptors(s) for s in d['structures']]
            metrics['structure_descriptor_seconds']=time.perf_counter()-feature_start
            bank=[i for i in d['verifier_train_indices'] if metrics['verifier_records'][d['ids'][i]]['status']=='ok']
            if len(bank)<1000:raise RuntimeError('Less than1000 valid TRAIN verifier-bank results; report limited coverage')
            bank_features=[feature_rows[i] for i in bank]
            bank_reference=np.asarray([metrics['verifier_records'][d['ids'][i]]['formation_energy_per_atom_eV'] for i in bank])
            context=training_context(bank_features,bank_reference,[d['ids'][i] for i in bank])
            metrics['TRAIN_reference_context']=context
            prev_code=None;prev_feedback=None
            for iteration in range(1,4):
                check_deadline(cfg)
                prompt,budget=bounded_prompt(iteration,context,prev_code,prev_feedback)
                parsed,bank_execution=generation(iteration,prompt,bank_features,cfg,log)
                # Freeze the TRAIN-valid code before applying it outside the
                # bank. HOLDOUT runtime/quality failures never cause LLM repair.
                execution=execute_rules(parsed['rules_code'],feature_rows)
                prediction=execution['predictions'];per_rule=execution['per_rule']
                score_started=time.perf_counter()
                train_feedback=feedback(bank_features,bank_reference,prediction[bank],per_rule[bank],
                                        [d['ids'][i] for i in bank],[d['groups'][i] for i in bank])
                evaluation=evaluate_p1(bank_reference,{'rules':prediction[bank]},groups[bank],sizes,threshold=threshold)
                version={'iteration':iteration,'summary':parsed['summary'],'rules_code':parsed['rules_code'],
                         'TRAIN_verifier_score':evaluation,'TRAIN_feedback':train_feedback,
                         'seconds_execute_bank':bank_execution['seconds'],
                         'seconds_score_bank':bank_execution['seconds']+time.perf_counter()-score_started,
                         'seconds_predict_all10000':execution['seconds'],
                         'input_budget':budget,'holdout_status':'Predictions saved; labels/metrics NEVER returned to generator'}
                metrics['rule_versions']=[v for v in metrics['rule_versions'] if v['iteration']!=iteration]+[version]
                metrics['rule_versions'].sort(key=lambda v:v['iteration'])
                write(OUT/'metrics.json',metrics)
                writer.append('Rules_v'+str(iteration),d['holdout_indices'],prediction[d['holdout_indices']].tolist())
                writer.append('Rules_v'+str(iteration)+'_TRAIN',bank,prediction[bank].tolist())
                log(f'Rules version{iteration} complete: bankMAE={train_feedback["TRAIN_ensemble_MAE"]:.6f}; no HOLDOUT feedback')
                prev_code=parsed['rules_code'];prev_feedback=train_feedback
            if rf_future is not None:
                rf=rf_future.result();metrics['rf_metadata']=rf['metadata']
                writer.append('Magpie_RF',rf['holdout_indices'].tolist(),rf['predictions'].tolist())
                write(OUT/'metrics.json',metrics)
        # HOLDOUT evaluation happens only after all rule generations are frozen.
        saved={}
        with (OUT/'predictions.csv').open(encoding='utf-8',newline='') as h:
            for r in csv.DictReader(h):
                if r['score']!='':saved[(r['arm'],r['id'])]=float(r['score'])
        ho=d['holdout_indices'];arms=['Magpie_RF','Rules_v1','Rules_v2','Rules_v3']
        predictions={arm:np.asarray([saved[(arm,d['ids'][i])] for i in ho]) for arm in arms}
        metrics['holdout_DFT']=evaluate_p1(y[ho],predictions,groups[ho],sizes,threshold=threshold)
        vt=d['verifier_test_indices'];valid=[i for i in vt if metrics['verifier_records'][d['ids'][i]]['status']=='ok']
        vpred={arm:np.asarray([saved[(arm,d['ids'][i])] for i in valid]) for arm in arms}
        vpred['CHGNet']=np.asarray([metrics['verifier_records'][d['ids'][i]]['formation_energy_per_atom_eV'] for i in valid])
        metrics['verifier200_DFT']=evaluate_p1(y[valid],vpred,groups[valid],sizes,threshold=threshold)
        metrics['verifier200_coverage']={'n_requested':200,'n_valid':len(valid),'n_failed':200-len(valid),
                                       'n_groups_requested':len(set(groups[vt])),'n_groups_valid':len(set(groups[valid]))}
        metrics['verifier_timing']=timing_summary([metrics['verifier_records'][d['ids'][i]] for i in vt],groups[vt])
        calls=read_calls()
        metrics['calls']={'n':len(calls),'successful_versions':sum(bool(c.get('valid')) for c in calls),
                          'input_tokens':sum(c.get('input_tokens') or 0 for c in calls),
                          'output_tokens':sum(c.get('output_tokens') or 0 for c in calls),
                          'cached_input_tokens':sum(c.get('cached_input_tokens') or 0 for c in calls),
                          'seconds':sum(c.get('elapsed_seconds') or 0 for c in calls),
                          'tools_used':sum(len(c.get('tool_activity') or []) for c in calls)}
        measured=metrics['calls'];bank_score=np.mean([v['seconds_score_bank'] for v in metrics['rule_versions']])
        verifier_per_bank=metrics['verifier_timing']['seconds']['mean']['point']*1024
        metrics['hundred_iteration_projection']={'basis':'3observed versions; linear operational projection, not a confidence interval',
             'LLM_input_tokens':measured['input_tokens']/3*100,'LLM_output_tokens':measured['output_tokens']/3*100,
             'cached_reference_seconds':100*(measured['seconds']/3+bank_score),
             'uncached1024_new_structures_each_iteration_seconds':100*(measured['seconds']/3+bank_score+verifier_per_bank),
             'cash_cost':'Local logged-in Codex only; no professor/API billing. CLI does not establish subscription/credit monetary cost; no invented USD conversion.',
             'does_not_include':'Dataset download, environment setup, one-time descriptors/model load, RF fit; future context-length variation.'}
        metrics['dataset_counts']={k:cfg['dataset'][k] for k in ('n','n_groups','n_train','n_train_groups','n_holdout','n_holdout_groups','n_verifier','n_verifier_groups','n_verifier_train','n_verifier_train_groups')}
        check_deadline(cfg)
        gates={'data_10k_license_hash_group_split':len(d['ids'])==10000 and cfg['dataset']['license']=='MIT' and cfg['dataset']['composition_overlap_train_holdout']==0,
               'CUDA_200_finite_formation_energies':len(valid)==200 and metrics['verifier_metadata']['model_device'].startswith('cuda'),
               'three_versions_five_executable_rules':len(metrics['rule_versions'])==3 and sum(bool(c.get('valid')) for c in calls)==3,
               'Magpie_RF_complete_heldout':all(('Magpie_RF',d['ids'][i]) in saved for i in ho),
               'public_GNN_context_recorded':'public_gnn_reference' in cfg['dataset'],
               'within_one_workday':datetime.now(timezone.utc)<datetime.fromisoformat(cfg['deadline_utc'])}
        metrics['decision']={'operational_feasibility':all(gates.values()),'gate_results':gates,
                             'scope':'Can run this bounded approximate-verifier loop; this is not proof that LLM rules beat RF, nor a DFT-validating oracle.',
                             'Stage5_started':False}
        metrics['status']='complete';metrics['completed_utc']=utc();metrics['run_wall_seconds']=time.perf_counter()-start
        write(OUT/'metrics.json',metrics)
        lines=['# P1 formation-energy feasibility','Complete; Stage4 only. Stage5 has not started.',
               '10,000 structures: TRAIN8,000/7,372 compositions; HOLDOUT2,000/1,821 compositions.',
               'DFT-relaxed Matbench structures; site properties including DFT magmom removed from model/LLM inputs.',
               'CUDA CHGNet0.3.0:200holdout structures timed and compared with DFT;1024TRAIN references cached.',
               'Formation energy subtracts public MP elemental references; AtomRef is NOT a thermodynamic reference.',
               'CHGNet pretraining overlaps MP data; historical label/correction conventions may differ.',
               'Three rule versions = initial plus two TRAIN-only revisions; five Python predictors/version, equal-weight mean.',
               'Rule predictions and Magpie+RF evaluated on same untouched HOLDOUT; no HOLDOUT feedback.',
               'Binary positive = Eform<0eV/atom (configurable); not convex-hull stability or synthesizability.',
               'Composition-group bootstrap2000, fixed original composition sizes; raw and group-weighted metrics saved.',
               'See metrics.json for feasibility table,200-test error/timing,three-version scores,RF and100-iteration projections.',
               'Public GNN score is full Matbench benchmark context, not a directly matched subset control.',
               'config.json records license,source hashes,split IDs,parameters and Git commit.',
               'llm_calls.jsonl records full source functions,request hash,token usage,time,status and validation.',
               'Only six required experiment files retained. Await human confirmation before Stage5.']
        safe(OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        log('P1 COMPLETE; no Stage5 dispatch');print('P1_COMPLETE')
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
    sys.dont_write_bytecode=True;sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);parser.add_argument('--task-name',required=True)
    args=parser.parse_args()
    try:(prepare if args.action=='prepare' else run)(args.task_name)
    except Exception:
        import traceback
        traceback.print_exc()
        if (OUT/'run.log').exists():
            with safe(OUT/'run.log').open('a',encoding='utf-8') as h:traceback.print_exc(file=h)
        raise
