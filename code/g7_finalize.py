"""Stage G7 merge/statistics. Build scratch only unless root authorizes main write."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,shutil
from datetime import datetime,timezone
from pathlib import Path
from common import ROOT,load_data,np,write_json
from g7_stats import build_g7_statistics,historical_stratified_appendix,METHODS
from g7_eval import material_prompt,composition_features
from f_report import _classification_section

MAIN=ROOT/'G7_postcutoff_materials';SCR=ROOT/'work/G7_eval';COR=ROOT/'work/G7_sources/eligibility_correction.json'
ARCHIVED_LABELS=ROOT/'work/G7_sources/G7_evaluation_labels_frozen.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def rows(p):
    with Path(p).open(newline='',encoding='utf8') as h:return list(csv.DictReader(h))
def aligned(p,ids,arm=None):
    rr=[r for r in rows(p) if arm is None or r.get('arm')==arm];lookup={r['id']:r for r in rr};assert len(lookup)==len(rr)
    assert all(str(k) in lookup for k in ids),(p,arm)
    out=np.asarray([float(lookup[str(k)]['score']) for k in ids]);assert np.isfinite(out).all();return out

def evaluation_audit(ev):
    started={};completed={};events=[];byid={c['id']:c for c in ev['cases']}
    with (SCR/'llm_calls.jsonl').open(encoding='utf8') as h:
        for line in h:
            r=json.loads(line)
            (started if r['event']=='call_started' else completed)[r['call_id']]=r
    assert len(started)==len(completed)==89
    for cid,r in completed.items():
        c=byid[r['id']];expected=material_prompt(c['material']);assert r['prompt']==expected and hashlib.sha256(expected.encode('utf8')).hexdigest()==c['prompt_sha256']==r['request_hash']
        assert r['runtime']==ev['runtime'] and r['model']=='gpt-6.1-sol' and r['reasoning_effort']=='high'
        assert not r['tool_activity'] and r['status']=='ok' and r['response_valid'] and r['returncode']==0
        assert r['scientific_round']==1
        events.extend([(datetime.fromisoformat(started[cid]['started_utc']).timestamp(),1),(datetime.fromisoformat(r['finished_utc']).timestamp(),-1)])
    n=peak=0
    for _,v in sorted(events):n+=v;peak=max(peak,n)
    assert peak<=4 and min(datetime.fromisoformat(r['started_utc']).timestamp() for cid,r in started.items() if cid!=1)>=datetime.fromisoformat(completed[1]['finished_utc']).timestamp()
    ml=read(SCR/'HGB_composition_training.json');assert ml['evaluation_labels_used'] is False and ml['no_tuning'] is True
    local=read(SCR/'local_run.json')
    assert local['status']=='complete' and local['n_completed']==local['n_valid']==89
    assert local['all_first_round_prompt_hashes_matched']==89
    la=local['final_audit'];assert la['status']=='passed' and la['prompt_system_schema_CSV_all_matched'] is True
    assert la['tool_calls']==0 and la['max_completion_tokens']<=512 and la['first_probe_before_four_workers'] is True
    assert la['server_slots']==4 and la['context_per_slot']==8192
    system_path=ROOT/'e2_instructions.md'
    assert sha(system_path)==ev['source_hashes'][str(system_path)]
    system_text=system_path.read_text(encoding='utf8')
    assert system_path.read_bytes().decode('utf8').replace('\r\n','\n')==system_text
    assert local['system_sha256']==hashlib.sha256(system_text.encode('utf8')).hexdigest()
    with (SCR/'local_llm_calls.jsonl').open(encoding='utf8') as h:
        lr=[json.loads(line) for line in h]
    assert len(lr)==89
    for r in lr:
        assert r['instructions']==system_text and r['prompt']==material_prompt(byid[r['id']]['material'])
    local['instruction_line_ending_audit']={'frozen_file_bytes_sha256':sha(system_path),'local_read_text_UTF8_sha256':local['system_sha256'],'text_equal_after_universal_newline_read':True,'boundary':'Local Python read_text normalizes CRLF to LF; GPT CLI receives the unchanged frozen model_instructions_file. Instruction text is equivalent but raw line-ending bytes are not asserted identical.'}
    xx=composition_features([c['formula'] for c in ev['cases']]);assert xx.shape==(89,118) and np.isfinite(xx).all()
    return {'status':'pass','initial_prompt_count':89,'exact_frozen_prompt_bytes_and_hashes_match':True,'runtime':ev['runtime'],'all_runtime_records_match':True,'tools_used':0,'model':'gpt-6.1-sol','reasoning_effort':'high','max_concurrency':peak,'probe_completed_before_second_call':True,'all_finished_not_cancelled':True,'all_returncodes_zero':True,'system_instructions_path':str(ROOT/'e2_instructions.md'),'system_instructions_sha256':ev['source_hashes'][str(ROOT/'e2_instructions.md')],'ML_labels_not_read':True,'ML_composition118_F3_reproduction_max_abs_diff':ml['composition118_F3_formula_reproduction_max_abs_diff'],'new_formula118_shape':[89,118],'task_completion_audit':ev.get('task_completion_audit'),'local_execution_audit_complete_copy':local}

def report_markdown(stats):
    from itertools import combinations
    lines=['# G7 新材料检验','',f'状态：{stats["status"]}。严格主分析n={stats["primary_n"]}，正类={stats["primary_n_positive"]}，specific-parent组={stats["primary_n_groups"]}。','',
           '组权重=1/严格eligible子集的specific-parent组大小；冻结89单独按其组大小。所有CI按组bootstrap5000次。仅两个正类支持组时CI很宽或单类bootstrap未定义，有限draw已逐表保留。','']
    if stats.get('local_A3_constant_negative_both_cohorts'):
        lines.extend(['本地Local_A3在历史3764和新89/严格69上均逐例p=0.0、全判负，属于本任务适配失败。BA恒为50%（仅双类draw有定义）；若相对HGB的DiD为正，只反映常数基线不变与HGB下降的算术组合，不能解释为本地模型泛化好或保留知识优势。',''])
    def section(title,m):
        t=m['new_all'];lines.extend([f'## {title}：n={t["n"]}，正类={t["n_positive"]}，组={t["n_groups"]}；正类支持组={t["n_positive_groups"]}，负类支持组={t["n_negative_groups"]}',''])
        _classification_section(lines,title,t,list(METHODS),list(combinations(METHODS,2)))
        lines.extend(['','|权重|方法|指标|有效draw|未定义draw|CI宽度pp|','|---|---|---|---:|---:|---:|'])
        for w in ('group_weighted','unweighted'):
            for a,mm in t[w]['metrics'].items():
                for metric,v in mm.items():
                    ci=v['ci95'];width='NA' if ci[0] is None else f'{100*(ci[1]-ci[0]):.2f}'
                    lines.append(f'|{w}|{a}|{metric}|{v["n_valid_boot"]}|{v["n_undefined_boot"]}|{width}|')
        lines.extend(['','仅有限draw形成的95% percentile区间是条件区间；新69仅2个正类支持组，按16组重采样有约12% draw缺正类。不能把排除这些draw后的区间当作完整稳定外部泛化不确定性。',''])
        for sub,t in m['historical3764'].items():
            lines.extend([f'历史3764 {sub}：n={t["n"]}，正类={t["n_positive"]}，组={t["n_groups"]}。',''])
            _classification_section(lines,title+'历史·'+sub,t,list(METHODS),list(combinations(METHODS,2)))
        for key in ('cohort_change_controls','cohort_change_controls_old_exclude93_sensitivity'):
            c=m[key];lines.extend([f'### {title}·{key}（old n={c["old_n"]}/正{c["old_positive"]}/{c["old_groups"]}组；new n={c["new_n"]}/正{c["new_positive"]}/{c["new_groups"]}组）','', '|类型|权重|方法|指标|差值pp|95%CIpp|CI宽度pp|有效/未定义draw|','|---|---|---|---|---:|---:|---:|---:|'])
            for typ in ('changes','difference_in_differences'):
                for w,methods in c[typ].items():
                    for a,metrics in methods.items():
                        for metric,v in metrics.items():
                            point=v['point'];ci=v['ci95'];p='NA' if point is None else f'{100*point:.2f}';cis='NA' if ci[0] is None else f'[{100*ci[0]:.2f},{100*ci[1]:.2f}]';width='NA' if ci[0] is None else f'{100*(ci[1]-ci[0]):.2f}'
                            lines.append(f'|{typ}|{w}|{a}|{metric}|{p}|{cis}|{width}|{v["n_valid_boot"]}/{v["n_undefined_boot"]}|')
    if stats['status']=='complete':section('严格外部主分析',stats['primary_external'])
    section('冻结89敏感性（不等于严格外部验证）',stats['frozen89_sensitivity'])
    lines+=['','## 历史3764分层附表（计算一次，适用于主分析与冻结89背景）','',
            '家族复用只读F3冻结五家族；组大小用完整原component大小。各层再分all/exclude_93/only_93；空层/单类BA为NA，仅一组CI退化。','']
    appendix=stats['historical3764_stratified_appendix']
    for category in ('families','group_size_bins'):
        for label,subsets in appendix[category].items():
            for sub,t in subsets.items():
                title=f'历史3764·{category}={label}·{sub}'
                lines.extend([f'### {title}：n={t["n"]}，正类={t["n_positive"]}，组={t["n_groups"]}',''])
                _classification_section(lines,title,t,list(METHODS),list(combinations(METHODS,2)))
    lines+=['','## 解释边界','']+['- '+x for x in stats['boundaries']]
    return '\n'.join(lines)+'\n'

def finalize(root_write_main_authorized=False):
    ev=read(SCR/'config.json');assert ev['status']=='complete';ids=[c['id'] for c in ev['cases']]
    assert len(ids)==89 and len(set(ids))==89
    snapshot=(MAIN/'predictions.csv').read_bytes();source_rows=rows(MAIN/'predictions.csv');main=read(MAIN/'config.json')
    assert ev['source_hashes'][str(MAIN/'predictions.csv')]==sha(MAIN/'predictions.csv')
    assert ev['source_hashes'][str(MAIN/'config.json')]==sha(MAIN/'config.json')
    local_state=read(SCR/'local_run.json');assert local_state['status']=='complete' and local_state['n_completed']==local_state['n_valid']==89
    labels=read(MAIN/'labels.json');correction=read(COR)
    assert sha(COR)=='83737f24f4693aaeb5498b8dcc8711d09fae15a8dccd3e0bf92de036951fc1b5','Root-frozen eligibility correction SHA differs'
    assert correction['reviewed_without_predictions'] is True
    assert correction['status']=='complete','Correction review not final'
    correction_by_id={r['material_id']:r for r in correction['predeclared_corrections'] if r['action']=='exclude_from_primary_external_analysis'}
    excluded=set(correction_by_id)
    y=np.asarray([int(labels[k]['y']) for k in ids]);groups=np.asarray([str(labels[k].get('parent_family_key') or next(r['parent_family_key'] for r in source_rows if r['material_id']==k)) for k in ids])
    new={'y':y,'groups':groups,'scores':{
        'GPT_A3':aligned(SCR/'gpt_predictions.csv',ids,'GPT_A3'),
        'Local_A3':aligned(SCR/'local_predictions.csv',ids,'Local_A3'),
        'HGB_composition118':aligned(SCR/'ml_predictions.csv',ids,'HGB_composition118')}}
    d=load_data();oldids=list(map(str,d['id']));old={'y':d['y'],'groups':d['group'],'scores':{
        'GPT_A3':aligned(ROOT/'F3_full3764/predictions.csv',oldids,'A3_forced'),
        'Local_A3':aligned(ROOT/'G6_local_cutoff/predictions.csv',oldids,'local_A3'),
        'HGB_composition118':aligned(SCR/'original3764_HGB_composition_OOF.csv',oldids)}}
    eligible=np.asarray([k not in excluded for k in ids]);n=int(eligible.sum());npos=int(y[eligible].sum());ng=len(np.unique(groups[eligible]))
    status='complete' if n>=60 and npos>=15 else 'stopped_insufficient_pool'
    assert (n,npos,ng)==(69,32,16),'Strict external identities differ from root-frozen69/32/16'
    allstats=build_g7_statistics(old,new,primary_eligible_mask=eligible if status=='complete' else None)
    f3config=read(ROOT/'F3_full3764/config.json')
    oldappendix=historical_stratified_appendix(old,f3config['family_masks'],d['group_size'],f3config.get('raw_a15_mask'))
    oldappendix['frozen_family_source_config_sha256']=sha(ROOT/'F3_full3764/config.json')
    primary=allstats.pop('strict_primary_eligible',None);stats={'status':status,'primary_n':n,'primary_n_positive':npos,'primary_n_groups':ng,'primary_n_positive_groups':int(len(np.unique(groups[eligible & (y==1)]))),'primary_n_negative_groups':int(len(np.unique(groups[eligible & (y==0)]))),
        'primary_external':primary if status=='complete' else {'status':status,'n':n,'n_positive':npos,'n_groups':ng,'reason':'Strict pool below60 total or15 positive; no external primary performance conclusion.'},
        'frozen89_sensitivity':allstats,'historical3764_stratified_appendix':oldappendix,'eligibility_correction':correction,'eligibility_correction_sha256':sha(COR),
        'eligibility_review_blind_to_predictions':True,'frozen_candidate_table_sha256':hashlib.sha256(snapshot).hexdigest(),
        'boundaries':['Independent source/date QA corrected20 entries after89-case scientific input freeze, blind to all model predictions. Strict69 is the primary external set; frozen89 contains known old/ambiguous entries and is audit/sensitivity only.',
          'Only2 strict-primary specific-parent groups support positive labels; conditional finite bootstrap intervals do not cover the uncertainty of resamples lacking a positive class.',
          'CompositionHGB historical performance uses original three-fold OOF (each fit2/3 data); external model fit all3764.',
          'Historical GPT/local A3 receives formula+composition+proxystructure; current no actual retrieved CIF means formula-only external inputs.',
          'G7 labels require ambient pressure and zero applied field, but the frozen original A3 STATIC_PREFIX asks only measuredTc>10 without an explicit pressure/field constraint. Formula-only input cannot uniquely identify phase or measurement conditions. Historical ML labels were not uniformly re-audited for pressure, so cohort changes/DiD cannot isolate knowledge-cutoff effects or fully adjust condition differences.',
          'Official pretraining cutoff does not certify an alignment/post-training data cutoff; historical3764 first-report dates are not all verified pre-cutoff.',
          'Some new specific families can have known prior relatives; strict formula/date filtering does not establish model-unseen identity.',
          'Independent-cohort DiD is descriptive and cannot fully deduct new-material difficulty, family mix, information availability or model capacity.',
          'Five archived deposition identifiers were found but no actual CIF bytes were retrieved; structure model subset absent.'],
        'prediction_sources_sha256':{str(p):sha(p) for p in [SCR/'gpt_predictions.csv',SCR/'local_predictions.csv',SCR/'ml_predictions.csv',SCR/'original3764_HGB_composition_OOF.csv',ROOT/'G6_local_cutoff/predictions.csv',ROOT/'F3_full3764/predictions.csv']}}
    def score_distribution(scores):
        result={}
        for arm,values in scores.items():
            values=np.asarray(values,float);unique,counts=np.unique(values,return_counts=True)
            result[arm]={'n':len(values),'min':float(values.min()),'max':float(values.max()),'n_unique':len(unique),'n_predicted_positive':int((values>=.5).sum()),'value_counts':{str(float(k)):int(v) for k,v in zip(unique,counts)}}
        return result
    stats['score_distributions']={'historical3764':score_distribution(old['scores']),'frozen89':score_distribution(new['scores']),'primary69':score_distribution({k:np.asarray(v)[eligible] for k,v in new['scores'].items()})}
    constant=bool(np.all(old['scores']['Local_A3']==0) and np.all(new['scores']['Local_A3']==0))
    stats['local_A3_constant_negative_both_cohorts']=constant
    if constant:stats['boundaries'].append('Local_A3 is exactlyp=0.0 for all historical3764 and external89/strict69, a constant-negative task-adaptation failure. Its BA50→50 and any positive DiD versus decliningHGB are arithmetic consequences, not evidence of good generalization or remaining knowledge advantage.')
    augmented=[];byid={k:i for i,k in enumerate(ids)}
    for r in source_rows:
        q=dict(r);k=r['material_id']
        q['evaluated_frozen89']='false';q['primary_external_eligible']='false';q['primary_external_exclusion_reason']=r.get('reason','')
        if k in byid:
            i=byid[k];q['evaluated_frozen89']='true';q['primary_external_eligible']=str(bool(eligible[i])).lower()
            q['primary_external_exclusion_reason']=correction_by_id[k]['predeclared_reason'] if k in correction_by_id else ''
            for a in METHODS:q[a+'_score']=float(new['scores'][a][i]);q[a+'_y_pred']=int(new['scores'][a][i]>=.5)
        augmented.append(q)
    fields=list(source_rows[0])+['evaluated_frozen89','primary_external_eligible','primary_external_exclusion_reason']+[z for a in METHODS for z in (a+'_score',a+'_y_pred')]
    # Always materialize and audit scratch first. Main mutation occurs only after every check succeeds.
    outcsv=SCR/'merged_predictions.csv'
    with outcsv.open('w',newline='',encoding='utf8') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(augmented)
    stats['evaluation_execution_audit']=evaluation_audit(ev);stats['ML_training_audit']=read(SCR/'HGB_composition_training.json')
    stats['final_prediction_table_sha256']=sha(outcsv)
    local_calls=SCR/'local_llm_calls.jsonl';assert local_calls.is_file()
    combined=SCR/'merged_llm_calls.jsonl';call_counts={}
    with combined.open('w',encoding='utf8') as h:
        for provider,path in [('GPT_A3',SCR/'llm_calls.jsonl'),('Local_A3',local_calls)]:
            count=0
            with path.open(encoding='utf8') as r:
                for line in r:
                    obj=json.loads(line);obj['evaluation_provider_arm']=provider;h.write(json.dumps(obj,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n');count+=1
            call_counts[provider]=count
    stats['merged_call_record_counts']=call_counts
    stats['evaluation_labels_storage']={'path':str(ARCHIVED_LABELS),'sha256':sha(MAIN/'labels.json'),'original_path':str(MAIN/'labels.json'),'migration_policy':'Only after all inference and final main audit succeed: copy, verify identical SHA, unlink original non-recursively. Labels remain separate from all scientific prompts.'}
    stats['boundaries'].append('Evaluation labels remain a separate frozen file at '+str(ARCHIVED_LABELS)+'; labels were absent from all model prompts and ML fitting inputs.')
    stats['report_markdown']=report_markdown(stats)
    write_json(SCR/'merged_metrics.json',stats)
    if root_write_main_authorized:
        main['frozen_pre_QA_screening_counts']={k:main.get(k) for k in ('qualified_rows_after_dedup_and_primary_evidence','qualified_positive_rows')}
        main['frozen_pre_QA_screening_ids']=main.get('accepted_material_ids',list(ids))
        assert set(main['frozen_pre_QA_screening_ids'])==set(ids)
        main.update(qualified_rows_after_dedup_and_primary_evidence=n,qualified_positive_rows=npos,accepted_material_ids=[k for k,b in zip(ids,eligible) if b])
        main.update(status=status,completed_utc=datetime.now(timezone.utc).isoformat(),primary_external={'n':n,'n_positive':npos,'n_groups':ng,'n_positive_groups':stats['primary_n_positive_groups'],'n_negative_groups':stats['primary_n_negative_groups'],'eligible_ids':[k for k,b in zip(ids,eligible) if b]},frozen89_sensitivity={'n':89,'n_positive':int(y.sum()),'n_groups':len(np.unique(groups))},eligibility_correction=correction,eligibility_correction_sha256=sha(COR),frozen_candidate_table_sha256=hashlib.sha256(snapshot).hexdigest(),frozen_candidate_table_utf8=snapshot.decode('utf8'),final_prediction_table_sha256=sha(outcsv))
        main['evaluation_execution_audit']=stats['evaluation_execution_audit'];main['merged_call_record_counts']=call_counts;main['evaluation_labels_storage']=stats['evaluation_labels_storage']
        write_json(SCR/'merged_config.json',main)
        readme=['# G7 post-cutoff external evaluation',f'Status: {status}.',f'Strict primary: n={n}, positive={npos}, specific-parent groups={ng}.',
          'Frozen89 predictions all retained as audit/sensitivity, separate from strict primary eligibility.',
          'CSV status retains frozen screening meaning; select primary_external_eligible=true for primary69. Exclusion reasons are explicit.',
          'Eligibility corrected by independent source/date review blind to model predictions; full correction and hashes in config/metrics.',
          'GPT A3: local codex exec gpt-6.1-sol/high, no tools/network; first probe then4concurrent.',
          'Local A3: frozen G6 local model; no API; local call audit retained.',
          'HGB composition118: original3fold OOF for historical3764 and finalall3764 fit for external query.',
          'No actual CIF bytes retrieved; five known archival IDs were access blocked. External prompts formula-only.',
          'New groups=specific parent; original3764 groups=component_id; bothweights and5000groupbootstrap.',
          'Tables include BA/precision/recall/accuracy, pairedCI, old all/ex93/only93 and independent-cohort DiD.',
          'Few positive support groups yield wide/undefined CIs; finite bootstrap counts retained.',
          'Historical3764 not all confirmed before cutoff; alignment-data cutoff not certified; DiD cannot fully deduct difficulty.',
          'Full report text: metrics.json/report_markdown. Frozen labels remain separately in work/G7_sources/G7_evaluation_labels_frozen.json, never in prompts.']
        assert len(readme)<=20
        (SCR/'merged_README.md').write_text('\n'.join(readme)+'\n',encoding='utf8')
        existing=(MAIN/'run.log').read_text(encoding='utf8');(SCR/'merged_run.log').write_text(existing+'\n'+datetime.now(timezone.utc).isoformat()+f' FINAL {status}; strict{n}/{npos}positive/{ng}groups; frozen89 retained; blind eligibility correction sha={sha(COR)}; GPT89complete; local89complete; no actualCIF\n',encoding='utf8')
        for src,dst in [('merged_predictions.csv','predictions.csv'),('merged_llm_calls.jsonl','llm_calls.jsonl'),('merged_metrics.json','metrics.json'),('merged_README.md','README.md'),('merged_run.log','run.log'),('merged_config.json','config.json')]:os.replace(SCR/src,MAIN/dst)
        # Only successful finalization may migrate the label file, and only after lossless copy/audit.
        label_sha=stats['evaluation_labels_storage']['sha256'];assert sha(MAIN/'labels.json')==label_sha
        assert 'onedrive' not in str(ARCHIVED_LABELS.parent.resolve()).lower()
        assert ARCHIVED_LABELS.parent.resolve().is_relative_to(ROOT.resolve())
        if ARCHIVED_LABELS.exists():assert sha(ARCHIVED_LABELS)==label_sha
        else:shutil.copyfile(MAIN/'labels.json',ARCHIVED_LABELS)
        assert sha(ARCHIVED_LABELS)==label_sha
        assert sha(MAIN/'predictions.csv')==stats['final_prediction_table_sha256']
        final_config=read(MAIN/'config.json');final_metrics=read(MAIN/'metrics.json')
        assert final_config['status']==final_metrics['status']==status
        assert final_config['evaluation_labels_storage']==final_metrics['evaluation_labels_storage']
        assert len((MAIN/'README.md').read_text(encoding='utf8').splitlines())<=20
        expected={'config.json','predictions.csv','llm_calls.jsonl','metrics.json','run.log','README.md'}
        assert {p.name for p in MAIN.iterdir()}==expected|{'labels.json'}
        (MAIN/'labels.json').unlink()
        assert {p.name for p in MAIN.iterdir()}==expected and sha(ARCHIVED_LABELS)==label_sha
    print(json.dumps({'status':status,'primary_n':n,'primary_positive':npos,'groups':ng,'main_written':root_write_main_authorized},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root-write-main-authorized',action='store_true');a=p.parse_args();finalize(a.root_write_main_authorized)
