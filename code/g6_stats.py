"""Read-only G6 evaluation, frozen original group bootstrap statistics."""
from common import ROOT, load_data, np, write_json,DATA,HGB
from g6_local import OUT, read, log, sha
from g5_inputs import validate as validate_reg
from f3_stats import build_f3_statistics
from f_stats import summarize, fraction_summary
from g5_stats import regression_summary
from f_report import _classification_section,_table,_cell
from g5_report import numeric
import csv,itertools,ctypes

def load_arrays(path,d,arms):
    names={str(v):i for i,v in enumerate(d['id'])};result={arm:np.full(len(names),np.nan) for arm in arms}
    with path.open(newline='',encoding='utf8') as f:
        for r in csv.DictReader(f):
            if r['arm'] in result and r['score']!='':
                i=names[r['id']];assert not np.isfinite(result[r['arm']][i]);result[r['arm']][i]=float(r['score'])
    return result
def main():
    cfg=read(OUT/'config.json');assert cfg['status']=='inference_completed'
    d=load_data();g=d['group'].astype(str);sizes={str(x):int(s) for x,s in zip(g,d['group_size'])}
    local=load_arrays(OUT/'predictions.csv',d,['local_A3','local_A3_reg','local_A3_M1'])
    baseline=load_arrays(ROOT/'F3_full3764/predictions.csv',d,['A3_forced'])
    strong=load_arrays(ROOT/'E0/predictions.csv',d,['StrongML'])['StrongML']
    g5=read(ROOT/'G5_tc_multithreshold/config.json')
    for source in (DATA,HGB):assert sha(source)==g5['scientific_sources'][str(source)]
    llmreg={name:np.full(3764,np.nan) for name in ('Tc_est_K','p_gt_5K','p_gt_10K','p_gt_15K')}
    gptreg={name:np.full(3764,np.nan) for name in llmreg}
    lookup={str(v):i for i,v in enumerate(d['id'])}
    for path,arm,arrays in [(OUT/'predictions.csv','local_A3_reg',llmreg)]:
        with path.open(newline='',encoding='utf8') as f:
            for r in csv.DictReader(f):
                if r['arm']==arm:
                    for name in arrays:
                        if r.get(name,'')!='':arrays[name][lookup[r['id']]]=float(r[name])
    assert set(g5['results'])==set(lookup),'canonicalG5results must cover all3764'
    for key,raw in g5['results'].items():
        obj=validate_reg(raw)
        for name in gptreg:gptreg[name][lookup[key]]=obj[name]
    inherited_discrepancies=[]
    with (ROOT/'G5_tc_multithreshold/predictions.csv').open(newline='',encoding='utf8') as f:
        for row in csv.DictReader(f):
            if row['arm']=='A3_reg':
                i=lookup[row['id']]
                if any(row.get(name,'')=='' or not np.isclose(float(row[name]),gptreg[name][i],rtol=0,atol=1e-12) for name in gptreg):
                    inherited_discrepancies.append({'id':row['id'],'CSV_status':row['status'],'canonical_source':'G5 config.results validated output; inherited CSV not modified'})
    assert all(np.isfinite(v).all() for v in gptreg.values())
    families={k:np.asarray(v,dtype=bool) for k,v in cfg['family_masks'].items()}
    g3={k:np.asarray(v,dtype=bool) for k,v in cfg['G3_family_masks'].items()}
    sub={'all':np.ones(3764,dtype=bool),'exclude_93':g!='93','only_93':g=='93'}
    sizebin={'1':d['group_size']==1,'2-5':(d['group_size']>=2)&(d['group_size']<=5),'6-20':(d['group_size']>=6)&(d['group_size']<=20),'>20':d['group_size']>20}
    partitions={'main':{'cohort':np.ones(3764,dtype=bool)},'F3_families':families,'G3_families':g3,'group_size_bins':sizebin}
    result={'status':'statistical_finalization','scope':'development only; original3764, no external validation','classification':{},'regression':{},'agreement':{},
            'method_boundary':cfg['cutoff_boundary'],'inherited_G5_canonical_outputs':{'source':'G5_tc_multithreshold/config.json results','validated_n':3764,'CSV_config_discrepancies':inherited_discrepancies,'old_files_modified':False},
            'bootstrap':{'n_boot':5000,'seed':20261019,'unit':'original component groups','weights':'1/full original group size and one/material; shared draws within stratum'}}
    for t in (5,10,15):
        scores={'local_A3_reg_direct':llmreg[f'p_gt_{t}K'],'local_A3_reg_from_Tc':np.where(np.isfinite(llmreg['Tc_est_K']),(llmreg['Tc_est_K']>t).astype(float),np.nan),
                'gpt_A3_reg_direct':gptreg[f'p_gt_{t}K'],'gpt_A3_reg_from_Tc':(gptreg['Tc_est_K']>t).astype(float),
                'HGB_classifier':np.asarray(g5['ML_classifier_scores'][f'HGB_classifier_gt_{t}K'])}
        if t==10:scores.update(local_A3=local['local_A3'],gpt_A3=baseline['A3_forced'],HGB127=d['hgb'],StrongML=strong)
        stats=build_f3_statistics((d['tc']>t).astype(int),scores,g,sizes,family_masks=families,a15_mask=np.asarray(cfg['raw_a15_mask'],dtype=bool),n_boot=5000,seed=20261019)
        valid=np.logical_and.reduce([np.isfinite(v) for v in scores.values()])
        stats['G3_families']={k:{label:summarize((d['tc']>t).astype(int),scores,g,sizes,mask&m&valid,n_boot=5000,seed=20261019) for label,m in sub.items()} for k,mask in g3.items()}
        result['classification'][f'Tc_gt_{t}K']=stats;log(f'STATS classification >{t} complete')
    f2=load_arrays(ROOT/'F2_mask/predictions.csv',d,['A3-M1'])['A3-M1']
    fixed=np.isin(d['id'].astype(str),[c['id'] for c in cfg['M1_cases']]);assert int(fixed.sum())==300
    scores={'local_A3':local['local_A3'],'local_A3_M1':local['local_A3_M1'],'gpt_A3':baseline['A3_forced'],'gpt_A3_M1':f2,'HGB127':d['hgb']}
    fixed_valid=fixed&np.logical_and.reduce([np.isfinite(v) for v in scores.values()])
    stats=build_f3_statistics(d['y'],scores,g,sizes,family_masks=families,a15_mask=np.asarray(cfg['raw_a15_mask'],dtype=bool),valid_mask=fixed,n_boot=5000,seed=20261019)
    def counts_for_fixed(stat,mask):
        requested=mask&fixed
        stat.update(n_requested=int(requested.sum()),n_requested_groups=len(np.unique(g[requested])),n_missing=int(np.sum(requested&~fixed_valid)),n_missing_groups=len(np.unique(g[requested&~fixed_valid])))
    for label,mask in sub.items():counts_for_fixed(stats['subsets'][label],mask)
    for kind,part in [('families',families),('group_size_bins',sizebin)]:
        for name,mask in part.items():
            for label,m in sub.items():counts_for_fixed(stats[kind][name][label],mask&m)
    stats['G3_families']={k:{label:summarize(d['y'],scores,g,sizes,mask&m&fixed_valid,n_boot=5000,seed=20261019) for label,m in sub.items()} for k,mask in g3.items()}
    for name,mask in g3.items():
        for label,m in sub.items():counts_for_fixed(stats['G3_families'][name][label],mask&m)
    stats['coverage']['requested']={'n':int(fixed.sum()),'n_groups':len(np.unique(g[fixed])),'n_positive':int(d['y'][fixed].sum()),'n_negative':int((1-d['y'][fixed]).sum())}
    stats['coverage']['common_valid']={'n':int(fixed_valid.sum()),'n_groups':len(np.unique(g[fixed_valid])),'n_positive':int(d['y'][fixed_valid].sum()),'n_negative':int((1-d['y'][fixed_valid]).sum())}
    result['classification']['fixed300_M1_10K']=stats;log('STATS fixed300 M1 complete')
    pred={'local_A3_reg':llmreg['Tc_est_K'],'gpt_A3_reg':gptreg['Tc_est_K'],'HGB_regressor':np.asarray(g5['HGB_regressor_Tc_K'])}
    reg_cache={}
    def regression_once(mask):
        key=mask.tobytes()
        if key not in reg_cache:reg_cache[key]=regression_summary(d['tc'],pred,g,sizes,mask,n_boot=5000,seed=20261019)
        return reg_cache[key]
    result['HGB10_reproduction_max_abs_difference']=float(np.max(np.abs(np.asarray(g5['ML_classifier_scores']['HGB_classifier_gt_10K'])-d['hgb'])))
    for kind,part in partitions.items():
        result['regression'][kind]={}
        for key,mask in part.items():
            result['regression'][kind][key]={label:{pop:regression_once(mask&m&(np.ones(3764,dtype=bool) if pop=='all_Tc' else d['tc']>10)) for pop in ('all_Tc','Tc_gt_10K')} for label,m in sub.items()}
        log('STATS regression '+kind+' complete')
    av={'local_A3_vs_gpt_A3':np.where(np.isfinite(local['local_A3']),((local['local_A3']>=.5)==(baseline['A3_forced']>=.5)).astype(float),np.nan),
        'local_reg_vs_gpt_reg':np.where(np.isfinite(local['local_A3_reg']),((local['local_A3_reg']>=.5)==(gptreg['p_gt_10K']>=.5)).astype(float),np.nan)}
    result['agreement']={kind:{name:{label:{key:fraction_summary(value,g,sizes,mask&m,n_boot=5000,seed=20261019) for key,value in av.items()} for label,m in sub.items()} for name,mask in part.items()} for kind,part in partitions.items()}
    result['coverage']={'requested_A3':3764,'valid_A3':int(np.isfinite(local['local_A3']).sum()),'requested_reg':3764,'valid_reg':int(np.isfinite(llmreg['Tc_est_K']).sum()),'requested_M1':300,'valid_M1':int(np.isfinite(local['local_A3_M1']).sum())}
    result['source_sha256']={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'predictions.csv',OUT/'llm_calls.jsonl',ROOT/'F3_full3764/predictions.csv',ROOT/'G5_tc_multithreshold/config.json',ROOT/'G5_tc_multithreshold/predictions.csv',ROOT/'F2_mask/predictions.csv',ROOT/'g6_stats.py']}
    result['status']='complete';result['report_markdown']=report(result);write_json(OUT/'metrics.json',result)
    cfg['status']='complete';write_json(OUT/'config.json',cfg);log('COMPLETE all G6 statistics')
    path=OUT/'README.md';path.write_text(path.read_text(encoding='utf8').replace('Status: prepared;','Status: complete;'),encoding='utf8')
def report(m):
    lines=['# G6 本地早截止模型开发结果','','模型：Meta Llama3.1 8B Instruct Q4_K_M，官方预训练截止2023-12。训练容量、数据、对齐和量化与GPT均不同；不能把性能差直接归为知识日期。','两种权重、四项指标及配对差均按原component组bootstrap5000次。组加权使用1/完整原组大小；子集继续原权重。单组CI退化。','模型提示词无标签、工具或联网；A3/回归/M1分别复用历史科学payload。所有方法共同有效分母，失败不插补。','A3→M1同时改变元素身份、物性精度和提示词；保留的比例、空间群和几何仍可能提供身份线索，性能变化不能换算为记忆比例。','G3家族来自既有模型的冻结预测，含“不是超导体”先验判断，不作为独立真实家族标签。','']
    if 'post_run_degenerate_A3_diagnosis' in m:
        diag=m['post_run_degenerate_A3_diagnosis']
        lines+=['## A3退化诊断与解释限制','',
            f"冻结A3提示词下，{diag['A3_exact_zero_scores']}/{diag['A3_n']}个概率全部精确为0；abstain分布{diag['A3_abstain_histogram']}。M1概率分布{diag['M1_score_histogram']}。",
            f"同一模型的A3-reg仍给出{diag['A3_reg_direct_positive_10K']}个直接概率正类、{diag['A3_reg_Tc_est_positive_10K']}个点估计>10K。因此不能将A3全负与GPT之间的差距解释成知识截止或记忆量的效应。",
            '这首先是该模型/运行配置在当前A3输出契约下的常数负类退化；原始响应中也有把化合物判断误写为元素判断等错误化学理由。没有改提示词、调参数或追加调用。',
            '若本地模型在历史和新材料均全负，两者BA都为50%；正的“本地变化−HGB变化”仅是HGB下降的反号算术结果，不能作为本地新材料能力或稳健性证据。','']
        for example in diag['illustrative_same_material_cross_task_outputs']:
            lines += [f"- {example['id']}，公式{example['formula']}：A3 p={example['A3_output']['p']}；同一科学材料A3-reg估Tc={example['A3_reg_output']['Tc_est_K']}K、p10={example['A3_reg_output']['p_gt_10K']}。A3理由：{example['A3_output']['reason']}"]
        lines+=['','这些只是跨任务差异的说明例，不用于估计知识或记忆比例。','']
    subs=[('all','全部'),('exclude_93','去93'),('only_93','仅93')]
    for task,stats in m['classification'].items():
        parts=[('主比较',stats['subsets'])]+[(kind+' '+name,data) for kind in ('families','group_size_bins','G3_families') for name,data in stats[kind].items()]
        for title,part in parts:
            for sub,label in subs:_classification_section(lines,task+' '+title+' '+label,part[sub],stats['method_order'],[tuple(p) for p in stats['report_pairs']])
    for kind,part in m['regression'].items():
        for name,data in part.items():
            for sub,label in subs:
                for pop,stat in data[sub].items():
                    lines+=['## 回归 '+kind+' '+name+' '+label+' '+pop,'',f"n={stat['n']}，组数={stat['n_groups']}；MAE单位K；Spearman为相关系数。",'']
                    rows=[]
                    for arm in stat['group_weighted']['metrics']:
                        rows.append([arm]+[numeric(stat[s]['metrics'][arm][k]) for s in ('group_weighted','unweighted') for k in ('MAE_K','Spearman')])
                    lines+=[_table(['方法','组加权 MAE K','组加权 Spearman','不加权 MAE K','不加权 Spearman'],rows),'']
                    rows=[]
                    for pair in stat['group_weighted']['paired_differences']:
                        rows.append([pair]+[numeric(stat[s]['paired_differences'][pair][k],True) for s in ('group_weighted','unweighted') for k in ('MAE_K','Spearman')])
                    lines+=[_table(['配对差','组加权 ΔMAE K','组加权 ΔSpearman','不加权 ΔMAE K','不加权 ΔSpearman'],rows),'']
    for kind,part in m['agreement'].items():
        for name,data in part.items():
            rows=[]
            for sub,label in subs:
                for pair,stat in data[sub].items():rows.append([label,pair,stat['n'],stat['n_groups'],_cell(stat['group_weighted']),_cell(stat['unweighted'])])
            lines+=['## 一致率 '+kind+' '+name,'',_table(['子集','比较','n','组数','组加权','不加权'],rows),'']
    return '\n'.join(lines)
if __name__=='__main__':
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:main()
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
