"""Freeze G5c majority-vote inputs/reused outputs. No model calls on import/prepare."""
from __future__ import annotations
import argparse,csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
from common import ROOT,DATA,HGB,load_data,np,write_json,safe_path
from e2_prompts import make_prompt
from f2_transport_v2 import resolve_runtime

OUT=ROOT/'G5c_majority3764'
ARMS=('A3_historical','A3_repeat_1','A3_repeat_2')
FIELDS=('id','fold','group','arm','status','score','y_pred')
SCHEMA=ROOT/'e2_response.schema.json';INSTRUCTIONS=ROOT/'e2_instructions.md';CATALOG=ROOT/'e2_model_catalog.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(s):return hashlib.sha256(s.encode('utf8')).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def prompt_for(case,repair=False):
    hist=[] if not repair else [{'operational_error':'Previous attempt failed the decision JSON contract. Return a forced decision with numeric p.'}]
    return make_prompt('A3',case['material'],[],{},[],hist,round_number=2 if repair else 1,force_decision=False)
def csv_rows(p):
    with Path(p).open(newline='',encoding='utf8') as h:return list(csv.DictReader(h))
def prepare(task_name='Codex.V2Diagnosis.G5cG7b.20261020'):
    safe_path(OUT);assert not OUT.exists(),'Never overwrite existing G5c; inspect current state'
    d=load_data();f3=read(ROOT/'F3_full3764/config.json');g4=read(ROOT/'G4_input_ablation/config.json');g5b=read(ROOT/'G5b_repeatability/config.json');g3=read(ROOT/'G3_complementarity/config.json')
    ids=list(map(str,d['id']));assert len(ids)==len(set(ids))==3764
    selected=set(map(int,g4['selected_indices']));assert len(selected)==800
    family=g3['family_results'];historical={r['id']:r for r in csv_rows(ROOT/'F3_full3764/predictions.csv') if r['arm']=='A3_forced'}
    assert set(historical)==set(ids)
    cases=[];pending=[];reused=[]
    for i,k in enumerate(ids):
        src=f3['cases'][k];fam=family[k];fam=fam['family'] if isinstance(fam,dict) else fam
        c={'i':i,'id':k,'fold':int(d['fold'][i]),'group':str(d['group'][i]),'material':src['material'],'prompt_sha256':src['prompt_hashes']['A3'],'G3_family':fam,'reuse_G5b':i in selected}
        assert digest(prompt_for(c))==c['prompt_sha256']
        cases.append(c);r=historical[k];p=float(r['score'])
        reused.append(dict(zip(FIELDS,(k,c['fold'],c['group'],'A3_historical','reused_valid',p,int(p>=.5),'F3_full3764','A3_forced'))))
        for arm in ARMS[1:]:
            if i in selected:
                result=g5b['results'][k+'|'+arm];p=float(result['p'])
                reused.append(dict(zip(FIELDS,(k,c['fold'],c['group'],arm,'reused_valid',p,int(p>=.5),'G5b_repeatability',arm))))
            else:pending.append({'job_id':k+'|'+arm,'id':k,'arm':arm,'case_index':i,'prompt_sha256':c['prompt_sha256']})
    assert len(pending)==5928 and len(reused)==5364
    pending.sort(key=lambda j:digest('G5c|20261020|'+j['job_id']))
    # All compared ML scores are frozen OOF outputs; no fitting or tuning occurs.
    byid={k:i for i,k in enumerate(ids)};baselines={'HGB127':np.asarray(d['hgb'],float),'StrongML':np.full(3764,np.nan),'CGCNN_scratch':np.full(3764,np.nan)}
    for path,arm in [(ROOT/'E0/predictions.csv','StrongML'),(ROOT/'G1_structure_baselines/predictions.csv','CGCNN_scratch')]:
        for r in csv_rows(path):
            if r['arm']==arm:baselines[arm][byid[r['id']]]=float(r['score'])
    assert all(np.isfinite(v).all() for v in baselines.values())
    source_paths=[DATA,HGB,ROOT/'F3_full3764/config.json',ROOT/'F3_full3764/predictions.csv',ROOT/'G4_input_ablation/config.json',ROOT/'G5b_repeatability/config.json',ROOT/'G5b_repeatability/predictions.csv',ROOT/'G3_complementarity/config.json',ROOT/'E0/predictions.csv',ROOT/'G1_structure_baselines/predictions.csv',SCHEMA,INSTRUCTIONS,CATALOG,ROOT/'e2_prompts.py',ROOT/'f2_transport_v2.py',ROOT/'e2_transport.py',ROOT/'g5c_inputs.py']
    cfg={'experiment':'G5c_majority3764','status':'prepared','created_utc':now(),'task_name':task_name,'n':3764,'n_groups':len(set(c['group'] for c in cases)),'cases':cases,'pending_jobs':pending,'n_new_jobs':5928,'n_reused_historical':3764,'n_reused_repeat_results':1600,'arms':list(ARMS),'primary_method':'A3_majority','majority_definition':'Each of three forced p>=0.5 predictions supplies one binary vote; at least2 positive votes gives positive. This is not probability averaging.','runtime':resolve_runtime(),'schema_path':str(SCHEMA),'instructions_path':str(INSTRUCTIONS),'catalog_path':str(CATALOG),'source_hashes':{str(p):sha(p) for p in source_paths},'family_masks':f3['family_masks'],'raw_a15_mask':f3['raw_a15_mask'],'group_size_values':d['group_size'].tolist(),'G3_labels':[c['G3_family'] for c in cases],'baseline_scores':{k:v.tolist() for k,v in baselines.items()},'runs':[],'protocol':{'model':'gpt-6.1-sol','reasoning_effort':'high','tools':0,'global_concurrency':4,'first_call_probe':True,'retry_delays_seconds':[60,120,240],'max_transport_retries':3,'max_scientific_rounds':2,'threshold':.5,'fallback':False,'bootstrap_draws':5000,'seed':20261020,'scheduled_task_runtime_limit':'PT0S','automatic_restarts_max':1,'development_only':True,'runtime_backend_time_boundary':'Historical/F3, reused/G5b800 and fresh2964 runs occur at different times; backend weights are unobservable. Reuse creates a temporally mixed repetition cohort.'}}
    cfg['results']={}
    cfg['reuse_sources']={'A3_historical':'F3_full3764/predictions.csv:A3_forced','A3_repeat_1_G4_800':'G5b_repeatability/config.json:results|A3_repeat_1','A3_repeat_2_G4_800':'G5b_repeatability/config.json:results|A3_repeat_2','fixed800_indices':sorted(selected)}
    OUT.mkdir();write_json(OUT/'config.json',cfg)
    with (OUT/'predictions.csv').open('w',newline='',encoding='utf8') as h:w=csv.DictWriter(h,fieldnames=FIELDS);w.writeheader();w.writerows(reused)
    (OUT/'llm_calls.jsonl').write_text('',encoding='utf8');(OUT/'run.log').write_text(now()+' PREPARED only; 5364 reused rows and5928 pending calls; no inference dispatched\n',encoding='utf8')
    write_json(OUT/'metrics.json',{'status':'prepared','n':3764,'n_reused_rows':5364,'n_pending_calls':5928})
    lines=['# G5c majority3764','Status: prepared; inference dispatch belongs to the unified global-four-worker runner.','Three original A3 runs: historical3764, repeat1, repeat2.','G4 fixed800 repeats reuse valid G5b outputs; remaining2964 require two new calls each (5928).','First prompt bytes exactly match frozen F3 A3; same model/high/runtime/system/schema; no tools or network.','Forced p>=0.5; primary majority uses binary votes, not averaged probabilities; no fallback.','All/exclude93/only93, five F3 families, G3 predicted labels, original group-size bins.','Both weightings; BA/precision/recall/accuracy; shared5000 group bootstrap and pairedCI.','All three pairwise run agreements and each single run are reported.','Baselines are readonly HGB127/StrongML/CGCNN_scratch OOF scores.','Calls and result rows are logged by the unified runner; results appended each10.','Historical/reused/fresh calls occur at different times; backend weights cannot be locked.','Development results only. Earlier experiment directories remain readonly.']
    assert len(lines)<=20;(OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    assert len(list(OUT.iterdir()))==6
    return cfg
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--task-name',default='Codex.V2Diagnosis.G5cG7b.20261020');v=a.parse_args();c=prepare(v.task_name);print(json.dumps({'status':c['status'],'cases':c['n'],'pending':c['n_new_jobs'],'reused_rows':5364}))
