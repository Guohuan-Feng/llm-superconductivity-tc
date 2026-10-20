"""G7b fixed-cohort majority statistics; no LLM calls or model fitting."""
from __future__ import annotations
import argparse,csv,json,os,ctypes
from pathlib import Path
from itertools import combinations
from common import ROOT,FIELDS,load_data,np,write_json
from g7b_inputs import OUT,WORK,ARMS,LABELS,read,sha,now,prompt_for,digest,threshold_label
from f_stats import summarize
from f3_stats import build_f3_statistics
from f_report import _classification_section,_table,_cell
from g5_inputs import validate as validate_regression

SEED=20261020;N_BOOT=5000

def collect(cfg,ml):
    cases=cfg['cases'];ids=[c['id'] for c in cases];groups=np.asarray([c['group'] for c in cases]);results=cfg['results']
    assert len(ids)==89 and len(results)==534 and set(results)=={j['job_id'] for j in cfg['pending_jobs']}
    for j in cfg['pending_jobs']:
        assert digest(prompt_for(cases[j['case_index']],j['arm']))==j['prompt_sha256']
    labels=read(LABELS);truth={};scores={};votes={};raw={};derived=[]
    prior=read(ROOT/'work/G7_eval/config.json')['results']
    for threshold,stem,key in ((10,'GPT_A3_condition10','p'),(5,'GPT_A3reg5','p_gt_5K')):
        yy=[threshold_label(labels[k],threshold)[0] for k in ids];valid=np.asarray([v is not None for v in yy]);truth[threshold]=(np.asarray([0 if v is None else v for v in yy],int),valid)
        ss={};vv=[]
        for rep in (1,2,3):
            arm=f'{stem}_r{rep}';rr=[results[k+'|'+arm] for k in ids]
            if threshold==5:
                for r in rr:validate_regression(r)
            else:
                for r in rr:
                    assert isinstance(r,dict) and not isinstance(r.get('p'),bool) and isinstance(r.get('p'),(int,float))
                    assert isinstance(r.get('reason'),str) and len(r['reason'])<=80
            pp=np.asarray([float(r[key]) for r in rr]);assert np.isfinite(pp).all() and np.all((pp>=0)&(pp<=1))
            ss[arm]=pp;vv.append(pp>=.5);raw[arm]=rr
        vv=np.stack(vv,axis=1);votes[threshold]=vv
        majority=(vv.sum(1)/3);ss={stem+'_majority':majority,**ss,'HGB_composition118':np.asarray([ml['thresholds'][str(threshold)]['new89_fullfit'][k] for k in ids])}
        if threshold==10:ss['GPT_A3_original_G7_single']=np.asarray([float(prior[k]['p']) for k in ids])
        scores[threshold]=ss
        for arm in (stem+'_majority','HGB_composition118'):
            for c,p in zip(cases,ss[arm]):derived.append({'id':c['id'],'fold':'','group':c['group'],'arm':arm+('5K' if threshold==5 and arm=='HGB_composition118' else '10K' if threshold==10 and arm=='HGB_composition118' else ''),'status':'valid','score':float(p),'y_pred':int(p>=.5)})
    return cases,ids,groups,truth,scores,votes,raw,derived

def build():
    cfg=read(OUT/'config.json');ml=read(WORK/'ML_baselines.json');assert ml['status']=='complete'
    for path,h in cfg['source_hashes'].items():assert sha(path)==h,path
    cases,ids,groups,truth,scores,votes,raw,derived=collect(cfg,ml)
    primary=set(cfg['primary_eligible_ids']);cohorts={};family_tables={};repetition={}
    for name,mask in [('primary69',np.asarray([k in primary for k in ids])),('frozen89',np.ones(89,bool))]:
        gg=groups[mask];fullsizes={str(k):int(v) for k,v in zip(*np.unique(gg,return_counts=True))};cohort={};families={};reps={}
        for threshold in (5,10):
            y,v=truth[threshold];selected=mask&v;stem='GPT_A3_condition10' if threshold==10 else 'GPT_A3reg5';major=stem+'_majority'
            ss={k:a[mask] for k,a in scores[threshold].items()};yy=y[mask];valid=v[mask]
            pairs=[(major,'HGB_composition118')]+[(major,f'{stem}_r{i}') for i in (1,2,3)]
            pairs+=list(combinations([f'{stem}_r{i}' for i in (1,2,3)],2))
            if threshold==10:pairs.append((major,'GPT_A3_original_G7_single'))
            table=summarize(yy,ss,gg,fullsizes,valid,n_boot=N_BOOT,seed=SEED,pairs=pairs)
            table.update(n_requested=int(mask.sum()),n_undetermined=int((mask&~v).sum()),undetermined_ids=[k for k,m in zip(ids,mask&~v) if m],n_positive_groups=int(len(np.unique(groups[selected&(y==1)]))),n_negative_groups=int(len(np.unique(groups[selected&(y==0)]))),method_order=list(ss),report_pairs=[list(p) for p in pairs])
            cohort[str(threshold)]=table
            # Identity-stable repetition agreement and its clustered paired uncertainty.
            vi=votes[threshold][mask];agreements={f'r{i+1}_vs_r{j+1}':float((vi[:,i][valid]==vi[:,j][valid]).mean()) for i,j in combinations(range(3),2)}
            reps[str(threshold)]={'pairwise_decision_agreement':agreements,'all_three_agree':float(np.all(vi[valid]==vi[valid,:1],axis=1).mean()),'majority_positive_count':int((vi[valid].sum(1)>=2).sum()),'individual_positive_counts':[int(vi[valid,i].sum()) for i in range(3)],'note':'BA and all other paired differences with clustered CI are in report_pairs, not inferred from agreement alone.'}
        for group in np.unique(gg):
            fm=mask&(groups==group);rows=[];details={}
            for threshold in (5,10):
                y,v=truth[threshold];select=fm&v;stem='GPT_A3_condition10' if threshold==10 else 'GPT_A3reg5';maj=stem+'_majority'
                ss={k:scores[threshold][k][mask] for k in (maj,'HGB_composition118')}
                table=summarize(y[mask],ss,gg,fullsizes,(groups[mask]==group)&v[mask],n_boot=N_BOOT,seed=SEED,pairs=[(maj,'HGB_composition118')])
                details[str(threshold)]=table
                rows.append({'threshold_K':threshold,'n':int(select.sum()),'n_positive':int(y[select].sum()),'GPT_correct':int(((scores[threshold][maj]>=.5)==y)[select].sum()),'HGB_correct':int(((scores[threshold]['HGB_composition118']>=.5)==y)[select].sum()),'GPT_error_ids':[ids[i] for i in np.flatnonzero(select&((scores[threshold][maj]>=.5)!=y))]})
            # Actual reasons are exclusively from conditional10K A3 calls; reg5 has no reason.
            wrong10=fm&truth[10][1]&((scores[10]['GPT_A3_condition10_majority']>=.5)!=truth[10][0]);wrong5=fm&truth[5][1]&((scores[5]['GPT_A3reg5_majority']>=.5)!=truth[5][0])
            examples=[]
            for i in np.flatnonzero(wrong10)[:3]:
                majority=int(votes[10][i].sum()>=2);rep=next(j for j in range(3) if int(votes[10][i,j])==majority);r=raw[f'GPT_A3_condition10_r{rep+1}'][i]
                examples.append({'id':ids[i],'formula':cases[i]['formula'],'repeat':rep+1,'p':float(r['p']),'y10':int(truth[10][0][i]),'majority_prediction10':majority,'reason_verbatim':r['reason'],'also_wrong_at5K':bool(wrong5[i])})
            newly=fm&truth[5][1]&truth[10][1]&(truth[5][0]==1)&(truth[10][0]==0)
            families[str(group)]={'n':int(fm.sum()),'n_new_positive_lowering_10K_to5K':int(newly.sum()),'new_positive_ids':[ids[i] for i in np.flatnonzero(newly)],'threshold_counts_and_correct':rows,'statistics':details,'error_reason_examples10K':examples,'reason_example_availability':f'{int(wrong10.sum())} erroneous materials; show at most3; fewer than2 means fewer than2 errors available' if wrong10.sum()<2 else f'{int(wrong10.sum())} erroneous materials; first3 fixed source order, one majority-voting actual reason per material','5K_reason_boundary':'A3-reg schema contains only four numeric fields and no reason. No 5K error explanation can be inferred or invented from those outputs; 10K reasons above concern a different task.'}
        cohorts[name]=cohort;family_tables[name]=families;repetition[name]=reps
    # Original3764 baseline OOF tables (development reference), preserving full component weights.
    d=load_data();g=d['group'];sizes={str(k):int(v) for k,v in zip(*np.unique(g,return_counts=True))};f3=read(ROOT/'F3_full3764/config.json');historical={}
    for threshold in (5,10):
        ss={'HGB_composition118':np.asarray([ml['thresholds'][str(threshold)]['original3764_OOF'][str(k)] for k in d['id']])}
        if threshold==10:ss['HGB127']=d['hgb']
        historical[str(threshold)]=build_f3_statistics((d['tc']>threshold).astype(int),ss,g,sizes,family_masks={k:np.asarray(v,bool) for k,v in f3['family_masks'].items()},group_size_values=d['group_size'],a15_mask=np.asarray(f3['raw_a15_mask'],bool),n_boot=N_BOOT,seed=SEED)
    out={'status':'complete','completed_utc':now(),'experiment':'G7b_supplement','n_boot':N_BOOT,'seed':SEED,'cohorts':cohorts,'specific_parent_families':family_tables,'repeat_diagnostics':repetition,'historical3764_HGB_OOF':historical,'ML_training':ml,'source_hashes':{str(p):sha(p) for p in (Path(__file__),OUT/'config.json',WORK/'ML_baselines.json',LABELS,ROOT/'f_stats.py',ROOT/'f3_stats.py')},
         'boundaries':['Thresholds are strictly >5K and >10K; exactly5K is negative at5K.','Primary69 and sensitivity89 use independent cohort-specific parent sizes, while each family stratum retains its full cohort group weight.','Tc upper bounds are not exact zero. All7 null-Tc records have upper bound2K, hence both thresholds are determinate negative.','G7_V4Ti2W frozenTc5.0K uses zero-field heat-capacity equal-entropy criterion; reported resistive5.57/onset5.75/zero5.24K would cross5K. No labels adjusted.','G7_RO1_Re0.4Os0.4Ta0.1Zr0.1 frozenTc5.0K is negative by strict5K rule.','Percentile CI is conditional on finite bootstrap draws; sparse families may be single-class or one-parent with undefined/degenerate intervals.','10K new prompt adds stated measurement condition only, and uses three-repeat majority. Comparison with original G7 single run changes condition and aggregation jointly; it cannot isolate a causal effect of the condition alone.','All repeats use original formula-only frozen89 material, not measured Tc, labels, DOI, dates, structural facts, or family hints.','5Kreg provides numeric estimates/probabilities only; verbatim reasons shown are from the10K conditional classifier and cannot explain5K decisions.']}
    out['report_markdown']=report(out);return out,derived

def report(stats):
    lines=['# G7b 新材料补充检验','', '主分析名单固定69例；冻结89例为敏感性分析。每张表同时给组加权与不加权；95% CI均为specific-parent按组bootstrap5000次，配对差共享draw。三次LLM独立判断以多数票为主结果，score为正票数/3。','']
    for cohort,bythreshold in stats['cohorts'].items():
        lines+=['## '+cohort,'']
        for threshold,t in bythreshold.items():
            lines+=[f'### Tc>{threshold}K：n={t["n"]}，正类={t["n_positive"]}，组={t["n_groups"]}，正类支持组={t["n_positive_groups"]}，负类支持组={t["n_negative_groups"]}，标签不定={t["n_undetermined"]}','']
            _classification_section(lines,cohort+' '+threshold+'K',t,t['method_order'],t['report_pairs'])
            lines+=['',json.dumps(stats['repeat_diagnostics'][cohort][threshold],ensure_ascii=False),'']
        families=stats['specific_parent_families'][cohort];rows=[]
        for name,f in families.items():
            for r in f['threshold_counts_and_correct']:rows.append([name,r['threshold_K'],r['n'],r['n_positive'],r['GPT_correct'],r['HGB_correct'],f['n_new_positive_lowering_10K_to5K']])
        lines+=['### specific-parent家族判对数','',_table(['家族','阈值K','n','正类','GPT多数判对','HGB判对','降至5K新增正类'],rows),'']
        lines+=['新增正类仅按固定标签降阈值计数；不是新增收集材料。','']
        for name,f in families.items():
            if f['n_new_positive_lowering_10K_to5K']:lines+=[name+' 新增正类：'+', '.join(f['new_positive_ids']),'']
            lines+=['#### '+name+'·10K错例真实理由',f['reason_example_availability'],'']
            ex=[[e['id'],e['formula'],e['repeat'],e['p'],e['y10'],e['majority_prediction10'],e['reason_verbatim']] for e in f['error_reason_examples10K']]
            lines+=[_table(['ID','公式','repeat','p','标签10K','多数判断10K','原文理由'],ex) if ex else '无10K错误，故无可展示的错误理由。',f['5K_reason_boundary'],'']
            for threshold,t in f['statistics'].items():
                _classification_section(lines,f'{cohort}/{name}/>{threshold}K',t,list(t['group_weighted']['metrics']),[(list(t['group_weighted']['metrics'])[0],'HGB_composition118')])
    lines+=['## 历史3764成分HGB OOF参照（开发结果）','']
    for threshold,h in stats['historical3764_HGB_OOF'].items():
        for subset,t in h['subsets'].items():_classification_section(lines,f'历史3764>{threshold}K/{subset}',t,h['method_order'],h['report_pairs'])
        for category in ('families','group_size_bins'):
            for name,by in h[category].items():
                for subset,t in by.items():_classification_section(lines,f'历史>{threshold}K/{category}/{name}/{subset}',t,h['method_order'],h['report_pairs'])
    lines+=['## 边界','']+['- '+s for s in stats['boundaries']]
    return '\n'.join(lines)+'\n'

def main(finalize=False):
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:stats,derived=build();write_json(WORK/'statistics.json',stats)
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    print('STATS_READY',WORK/'statistics.json',flush=True)
    if finalize:
        cfg=read(OUT/'config.json');assert len(cfg['results'])==534
        # Keep all raw-repeat rows from the unified runner and append/replace derived rows deterministically.
        with (OUT/'predictions.csv').open(newline='',encoding='utf8') as h:rows=[r for r in csv.DictReader(h) if r['arm'] in ARMS]
        assert len(rows)==len({(r['id'],r['arm']) for r in rows})==534
        assert all(r['status']=='valid' for r in rows)
        jobs={j['job_id']:j for j in cfg['pending_jobs']}
        for r in rows:
            job=jobs[r['id']+'|'+r['arm']];p=float(cfg['results'][job['job_id']][job['score_key']])
            assert float(r['score'])==p and int(r['y_pred'])==int(p>=.5)
        # Only mark MAIN complete after all534 raw CSV outcomes were revalidated.
        cfg.update(status='complete',completed_utc=now(),ML_training=stats['ML_training'],statistics_sha256=sha(WORK/'statistics.json'),statistics_source_sha256=sha(Path(__file__)));write_json(OUT/'config.json',cfg)
        with (OUT/'predictions.csv').open('w',newline='',encoding='utf8') as h:w=csv.DictWriter(h,fieldnames=FIELDS);w.writeheader();w.writerows(rows+derived);h.flush();os.fsync(h.fileno())
        write_json(OUT/'metrics.json',stats)
        (OUT/'README.md').write_text('# G7b supplement\nComplete: primary69 fixed; frozen89 sensitivity.\n10K: original A3 plus exact ambient/zero-field sentence; three-repeat majority.\n5K: original four-number A3-reg; majority p5 decisions across three repeats.\nFormula-only inputs identical to original G7; no evaluation labels in prompts.\nFixed HGB composition118, original threefold OOF and full3764 fit, no tuning.\nmetrics.json contains full results, paired CIs, family tables, and report_markdown.\nBoth weighting schemes; parent-group bootstrap5000; cohorts independently weighted.\n7 null Tc records remain upperbound2K, never imputed Tc0.\nTc exactly5K negative; V4Ti2W transition-criterion sensitivity explicitly disclosed.\n10K reasons shown verbatim;5K numeric-only schema cannot explain decisions.\nAll old experiment directories unchanged.\n',encoding='utf8')
        with (OUT/'run.log').open('a',encoding='utf8') as h:h.write(now()+' STATISTICS_COMPLETE primary69/frozen89 two thresholds\n')
        assert len(list(OUT.iterdir()))==6

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root-finalize-authorized',action='store_true');a=p.parse_args();main(a.root_finalize_authorized)
