"""Frozen G7b supplement input contract; no LLM calls on import or preparation."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json
from common import ROOT,FIELDS,write_json,safe_path
from e2_prompts import make_prompt
from g5_inputs import prompt as regression_prompt

OUT=ROOT/'G7b_supplement'; WORK=ROOT/'work/G7b_runtime'
LABELS=ROOT/'work/G7_sources/G7_evaluation_labels_frozen.json'
ARMS=tuple(f'GPT_A3_condition10_r{i}' for i in (1,2,3))+tuple(f'GPT_A3reg5_r{i}' for i in (1,2,3))
CONDITION='标签是常压、零外场下测得的 Tc'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(s):return hashlib.sha256(s.encode('utf8')).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()

def prompt_for(case,arm,repair=False):
    assert arm in ARMS
    material=case['material'];assert set(material)=={'formula'}
    if 'A3reg5' in arm:return regression_prompt(material,repair=True if repair else None)
    history=[] if not repair else [{'operational_error':'Previous attempt violated decision JSON contract. Return a forced decision with numeric p and exactly the requested fields.'}]
    return make_prompt('A3',material,[],{},[],history,round_number=2 if repair else 1,force_decision=False)+'\n'+CONDITION

def threshold_label(record,threshold):
    tc=record.get('tc_k')
    if tc is not None:return int(float(tc)>threshold),'reported_tc'
    upper=record.get('tc_upper_bound_k');lower=record.get('tc_lower_bound_k')
    if upper is not None and float(upper)<=threshold:return 0,'upper_bound'
    if lower is not None and float(lower)>threshold:return 1,'lower_bound'
    return None,'undetermined_bound'

def prepare():
    assert not OUT.exists(),'Do not overwrite a prepared or running supplement'
    for p in (ROOT,*ROOT.parents):assert not p.is_symlink() and not p.is_junction()
    prior=read(ROOT/'work/G7_eval/config.json');main=read(ROOT/'G7_postcutoff_materials/config.json');labels=read(LABELS)
    eligible=main['primary_external']['eligible_ids'];assert len(eligible)==69
    cases=[]
    for c in prior['cases']:
        k=c['id'];m=c['material'];assert set(m)=={'formula'} and m['formula']==c['formula']
        cases.append({'id':k,'fold':'','group':labels[k]['parent_family_key'],'material':m,'formula':c['formula']})
    assert len(cases)==89 and len({c['id'] for c in cases})==89
    assert {c['id'] for c in cases}==set(main['frozen_pre_QA_screening_ids'])
    jobs=[]
    for arm in ARMS:
        for i,c in enumerate(cases):jobs.append({'job_id':c['id']+'|'+arm,'id':c['id'],'arm':arm,'case_index':i,'prompt_sha256':digest(prompt_for(c,arm)),
            'kind':'reg' if 'A3reg5' in arm else 'a3','score_key':'p_gt_5K' if 'A3reg5' in arm else 'p'})
    assert len(jobs)==534
    truth={}
    for c in cases:
        r=labels[c['id']];y5,b5=threshold_label(r,5);y10,b10=threshold_label(r,10);assert y10==r['y']
        truth[c['id']]={'y5':y5,'y10':y10,'label5_basis':b5,'label10_basis':b10,'tc_k':r.get('tc_k'),'tc_upper_bound_k':r.get('tc_upper_bound_k')}
    counts={}
    for name,ids in [('primary69',eligible),('frozen89',[c['id'] for c in cases])]:
        counts[name]={str(t):{'n':len(ids),'n_positive':sum(truth[k]['y'+str(t)]==1 for k in ids),'n_unknown':sum(truth[k]['y'+str(t)] is None for k in ids)} for t in (5,10)}
    sources=[Path(__file__),ROOT/'e2_prompts.py',ROOT/'g5_inputs.py',ROOT/'e2_response.schema.json',ROOT/'g5_response.schema.json',ROOT/'e2_instructions.md',ROOT/'g5_instructions.md',LABELS,ROOT/'G7_postcutoff_materials/config.json',ROOT/'work/G7_eval/config.json']
    cfg={'experiment':'G7b_supplement','status':'prepared','created_utc':now(),'cases':cases,'pending_jobs':jobs,'results':{},'n':89,'n_jobs':534,'primary_eligible_ids':eligible,'truth_evaluation_only':truth,'label_counts':counts,
         'llm_model':'gpt-6.1-sol','reasoning_effort':'high','concurrency':4,'no_tools':True,'no_web':True,'majority_rule':'Three fresh repeats, each score>=0.5 positive; majority of three hard decisions. No probability averaging replaces majority.',
         'prompt_contract':{'10K':'Original G7 A3 prompt plus newline and exact condition sentence; formula-only unchanged','condition_exact':CONDITION,'5K':'Original G5 regression prompt and four-number schema, formula-only; take p_gt_5K, no reason field'},
         'source_hashes':{str(p):sha(p) for p in sources},'label_isolation':'Cases contain only original formula in material; groups/fold/labels are never serialized by prompt_for. Truth retained evaluation-only.',
         'statistical_protocol':{'bootstrap':5000,'unit':'specific-parent groups','weights':'1/cohort-specific parent size; primary69 and frozen89 computed independently','paired_CI':True,'thresholds_strictly_greater_than_K':[5,10]},'old_directories_modified':False}
    safe_path(OUT).mkdir();safe_path(WORK).mkdir(parents=True,exist_ok=True);write_json(OUT/'config.json',cfg)
    write_json(OUT/'metrics.json',{'status':'pending_inference','label_counts':counts})
    with (OUT/'predictions.csv').open('w',newline='',encoding='utf8') as h:csv.DictWriter(h,fieldnames=FIELDS).writeheader()
    (OUT/'llm_calls.jsonl').write_text('',encoding='utf8');(OUT/'run.log').write_text(now()+' PREPARED 534 tasks; CPU baseline writes scratch only\n',encoding='utf8')
    (OUT/'README.md').write_text('# G7b supplement\nPrepared: 89 frozen formula-only inputs; primary69 fixed.\n10K: original A3 + exact ambient/zero-field condition, three fresh repeats.\n5K: original four-number A3-reg, three fresh repeats, majority p5 decisions.\nNo labels are in scientific prompts. Labels and cohorts remain frozen.\nML fixed composition118 HGB, original threefold OOF plus full-data fit.\nGroup bootstrap5000; primary69/frozen89 weights computed separately.\nFinal results pending unified runner and statistics.\n',encoding='utf8')
    print(json.dumps({'out':str(OUT),'jobs':len(jobs),'label_counts':counts,'config_sha256':sha(OUT/'config.json')},ensure_ascii=False))

if __name__=='__main__':prepare()
