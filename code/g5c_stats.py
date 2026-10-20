"""G5c three-run binary majority and clustered descriptive comparisons; no LLM."""
from __future__ import annotations
from itertools import combinations
import csv,json,hashlib,os
from common import ROOT,load_data,np,write_json
from f3_stats import build_f3_statistics
from f_stats import summarize,fraction_summary
from f_report import _classification_section
from g5c_inputs import OUT,ARMS,FIELDS,read,now

N_BOOT=5000;SEED=20261020;MAJORITY='A3_majority'
BASELINES=('HGB127','StrongML','CGCNN_scratch')

def build_statistics(cfg,d,scores):
    """Scores align to all3764 original rows; force every run independently first."""
    assert len(cfg['cases'])==len(d['y'])
    y=np.asarray(d['y']);groups=np.asarray(d['group']).astype(str)
    sizes={str(g):int(s) for g,s in zip(d['group'],d['group_size'])}
    arr={k:np.asarray(scores[k],float) for k in ARMS}
    assert all(v.shape==y.shape for v in arr.values())
    valid=np.logical_and.reduce([np.isfinite(v) for v in arr.values()])
    majority=np.full(len(y),np.nan);votes=np.stack([arr[k]>=.5 for k in ARMS]).sum(axis=0)
    majority[valid]=votes[valid]/3
    arr[MAJORITY]=majority
    arr.update({k:np.asarray(cfg['baseline_scores'][k],float) for k in BASELINES})
    m=build_f3_statistics(y,arr,groups,sizes,family_masks={k:np.asarray(v,bool) for k,v in cfg['family_masks'].items()},group_size_values=np.asarray(d['group_size']),a15_mask=np.asarray(cfg['raw_a15_mask'],bool),n_boot=N_BOOT,seed=SEED)
    m.update(status='complete' if valid.all() else 'partial',primary_method=MAJORITY,majority_definition=cfg['majority_definition'],development_only=True,n_boot=N_BOOT,seed=SEED,n_scientific_runs=3,n_all_three_valid=int(valid.sum()),temporal_boundary=cfg['protocol']['runtime_backend_time_boundary'])
    masks={'all':np.ones(len(y),bool),'exclude_93':groups!='93','only_93':groups=='93'}
    g3=np.asarray(cfg['G3_labels']);m['G3_predicted_families']={}
    for label in dict.fromkeys(g3.tolist()):
        m['G3_predicted_families'][label]={s:summarize(y,arr,groups,sizes,mask=mask&(g3==label)&valid,n_boot=N_BOOT,seed=SEED) for s,mask in masks.items()}
    strata={'subsets':masks,'families':{k:np.asarray(v,bool) for k,v in cfg['family_masks'].items()},'G3_predicted_families':{k:g3==k for k in dict.fromkeys(g3.tolist())},'group_size_bins':{'1':np.asarray(d['group_size'])==1,'2-5':(np.asarray(d['group_size'])>=2)&(np.asarray(d['group_size'])<=5),'6-20':(np.asarray(d['group_size'])>=6)&(np.asarray(d['group_size'])<=20),'>20':np.asarray(d['group_size'])>20}}
    m['pairwise_run_agreement']={}
    for category,collection in strata.items():
        m['pairwise_run_agreement'][category]={}
        for label,mask in collection.items():
            submasks={'all':mask} if category=='subsets' else {s:mask&v for s,v in masks.items()}
            tables={}
            for subset,selected in submasks.items():
                tables[subset]={}
                for a,b in combinations(ARMS,2):
                    good=selected&np.isfinite(arr[a])&np.isfinite(arr[b]);value=np.full(len(y),np.nan);value[good]=((arr[a][good]>=.5)==(arr[b][good]>=.5)).astype(float)
                    tables[subset][a+' = '+b]=fraction_summary(value,groups,sizes,mask=selected,n_boot=N_BOOT,seed=SEED)
            m['pairwise_run_agreement'][category][label]=tables['all'] if category=='subsets' else tables
    m['vote_count_distribution']={str(k):int(((votes==k)&valid).sum()) for k in range(4)}
    m['score_distributions']={k:{'min':float(v[valid].min()) if valid.any() else None,'max':float(v[valid].max()) if valid.any() else None,'n_predicted_positive':int(((v>=.5)&valid).sum()),'n_unique':int(len(np.unique(v[valid])))} for k,v in arr.items()}
    m['boundaries']=['All estimates are development results on the original3764; no external test is implied.','Majority voting uses three thresholded binary predictions, not probability averaging.','The historical F3 run, reused G5b800 repeat calls, and fresh2964 calls have different run times; backend model weights are unobservable and cannot be locked.','Pairwise comparisons use the same5000 group draws for all methods and both weightings within each reported stratum.','Group weighted means1/full original component size even after any subset/family mask. Unweighted uncertainty still resamples groups.','One-group CIs are degenerate; BA/recall can be undefined for single-class bootstrap draws; valid/undefined counts remain in metrics.','Multiple descriptive comparisons have no multiplicity adjustment.']
    return m,majority

def report_markdown(m):
    methods=list(ARMS)+[MAJORITY]+list(BASELINES)
    pairs=[(MAJORITY,b) for b in BASELINES]+[(a,MAJORITY) for a in ARMS]+list(combinations(ARMS,2))+[(a,b) for a in ARMS for b in BASELINES]
    lines=['# G5c 全3764三次A3与多数投票','',m['majority_definition'],'',m['temporal_boundary'],'']
    for sub,t in m['subsets'].items():_classification_section(lines,'全量·'+sub,t,methods,pairs)
    for cat in ('families','G3_predicted_families','group_size_bins'):
        for label,subsets in m[cat].items():
            for sub,t in subsets.items():_classification_section(lines,cat+'·'+label+'·'+sub,t,methods,pairs)
    lines.extend(['','## 三次运行两两判断一致率','', '|分层|子集|方法对|n|组数|组加权一致率95%CI|不加权一致率95%CI|','|---|---|---|---:|---:|---|---|'])
    def fmt(r):return 'NA' if r['point'] is None else f'{100*r["point"]:.2f} [{100*r["ci95"][0]:.2f},{100*r["ci95"][1]:.2f}]'
    for cat,coll in m['pairwise_run_agreement'].items():
        for label,items in coll.items():
            subsets={'all':items} if cat=='subsets' else items
            for sub,table in subsets.items():
                for pair,r in table.items():lines.append(f'|{cat}={label}|{sub}|{pair}|{r["n"]}|{r["n_groups"]}|{fmt(r["group_weighted"])}|{fmt(r["unweighted"])}|')
    lines+=['','## 证据边界','']+['- '+s for s in m['boundaries']]
    return '\n'.join(lines)+'\n'

def finalize():
    cfg=read(OUT/'config.json');d=load_data();byid={str(k):i for i,k in enumerate(d['id'])}
    with (OUT/'predictions.csv').open(newline='',encoding='utf8') as h:rows=list(csv.DictReader(h))
    scores={a:np.full(len(byid),np.nan) for a in ARMS};seen=set()
    for row in rows:
        if row['arm'] not in ARMS:continue
        key=(row['id'],row['arm']);assert key not in seen;seen.add(key)
        if row['score']!='':scores[row['arm']][byid[row['id']]]=float(row['score'])
    assert all(np.isfinite(v).all() for v in scores.values()),'Do not finalize before all three complete valid runs'
    metrics,majority=build_statistics(cfg,d,scores)
    metrics['input_source_hashes']=cfg['source_hashes'];metrics['reused_results']=cfg['reuse_sources'];metrics['report_markdown']=report_markdown(metrics)
    rows=[r for r in rows if r['arm']!=MAJORITY]
    for c,p in zip(cfg['cases'],majority):rows.append({'id':c['id'],'fold':c['fold'],'group':c['group'],'arm':MAJORITY,'status':'derived_binary_majority','score':float(p),'y_pred':int(p>=.5)})
    scratch=ROOT/'work/G5c_final_predictions.csv'
    with scratch.open('w',newline='',encoding='utf8') as h:w=csv.DictWriter(h,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    metrics['final_predictions_sha256']=hashlib.sha256(scratch.read_bytes()).hexdigest();write_json(OUT/'metrics.json',metrics);os.replace(scratch,OUT/'predictions.csv')
    cfg.update(status='complete',statistics_completed_utc=now(),majority_rows=len(majority));write_json(OUT/'config.json',cfg)
    text=(OUT/'README.md').read_text(encoding='utf8').replace('Status: prepared; inference dispatch belongs to the unified global-four-worker runner.','Status: complete; majority and single-run comparisons are in metrics.json/report_markdown.')
    (OUT/'README.md').write_text(text,encoding='utf8')
    with (OUT/'run.log').open('a',encoding='utf8') as h:h.write(now()+' G5c statistics complete; no new LLM calls\n')
    return metrics
if __name__=='__main__':
    m=finalize();print(json.dumps({'status':m['status'],'n_all_three_valid':m['n_all_three_valid'],'vote_counts':m['vote_count_distribution']}))
