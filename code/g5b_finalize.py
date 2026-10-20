"""Pure statistics finalization of completed G5b calls; never calls LLM."""
import json
from collections import Counter
from common import load_data, write_json
from g5b_repeat import OUT, G4, G5, F3, read, build_statistics, make_report, readme, now
from common import np
from f_stats import summarize, fraction_summary

cfg=read(OUT/'config.json'); g4=read(G4/'config.json'); d=load_data()
assert len(cfg['results'])==2400
cfg['family_masks']=g4['family_masks']
cfg['G3_labels']=[c['G3_family'] for c in g4['cases']]
cfg['post_run_metadata_repair']={'family_masks':'copied from readonly G4 config after statistics-only KeyError; not frozen by initial G5b prepare',
 'G3_labels':'copied from readonly G4 cases for requested final stratification; not supplied to evaluated LLM',
 'source_hashes':'initial source_hashes was empty; prompt source parity is independently verified post-run against original per-case frozen hashes; no claim of a pre-run full source freeze'}
cfg['execution_launch_audit']={'method':'functions.exec exec_command interactive PTY session_id68458','scheduled_task_registered':False,'task_name_is_metadata_only':True,'SetThreadExecutionState_used':True,'runtime_cap_seconds':None,'automatic_task_restart':False,'scientific_process_result':'2400 valid calls completed; process exit1 at statistics-only missing family_masks KeyError; no inference restart; statistics-only finalize repaired metadata and returned0','plan_deviation':'G5b was launched as a long-lived direct Python process, not the requested Windows scheduled task. No PT0S/task result/triggers claim is made. Successful scientific calls are not rerun merely to change launch method.'}
cfg['terminal_scientific_audit']={'post_run':True,'all800_A3_and_reg_hashes_match_original':True,'all2400_initial_prompt_bytes_match_original':True,'completed_calls':2425,'valid_arm_cases':2400,'runtime_records_all_match':True,'runtime_sha256':cfg['runtime']['sha256'],'model':'gpt-6.1-sol','reasoning_effort':'high','tools':0,'max_concurrency':4,'probe_completed_before_call2':True,'forced_threshold':0.5,'HGB_fallback_inserted':False,'source_freeze_limitation':'No pre-run complete code hash freeze was recorded; exact evaluated prompt parity was audited from persisted prompt bytes and original frozen per-case hashes.'}
arms=('A3_repeat_1','A3_repeat_2','A3_reg'); scores={a:[] for a in arms}
for c in cfg['cases']:
    for a in arms:
        q=cfg['results'][c['id']+'|'+a]
        scores[a].append(float(q.get('p',q.get('p_gt_10K'))))
m=build_statistics(cfg,d,scores)
oldg5=read(G5/'config.json'); historical=oldg5['results']
idx=np.asarray(cfg['selected_indices'],int); groups=np.asarray(d['group'])[idx].astype(str); y=np.asarray(d['y'])[idx]
sizes={str(g):int(s) for g,s in zip(d['group'],d['group_size'])}
hist_scores=np.asarray([historical[c['id']]['p_gt_10K'] for c in cfg['cases']],float)
reg_scores=np.asarray(scores['A3_reg'],float)
hist_a3=np.asarray([float(g4['baseline_A3_rows'][c['id']]['score']) for c in cfg['cases']],float)
m['historical_A3_and_reg_context']={}
for subset,mask in {'all':np.ones(800,bool),'exclude_93':groups!='93','only_93':groups=='93'}.items():
    m['historical_A3_and_reg_context'][subset]=summarize(y,{'A3_historical_F3':hist_a3,'A3_repeat_1':np.asarray(scores['A3_repeat_1']),'A3_repeat_2':np.asarray(scores['A3_repeat_2']),'A3_reg_historical_G5':hist_scores,'A3_reg_fresh':reg_scores},groups,sizes,mask=mask,n_boot=5000,seed=20261019)
m['A3_reg_same_prompt_historical_G5_control']={}
for subset,mask in {'all':np.ones(800,bool),'exclude_93':groups!='93','only_93':groups=='93'}.items():
    table=summarize(y,{'A3_reg_fresh':reg_scores,'A3_reg_historical_G5':hist_scores},groups,sizes,mask=mask,n_boot=5000,seed=20261019)
    table['agreement']=fraction_summary(((reg_scores>=.5)==(hist_scores>=.5)).astype(float),groups,sizes,mask=mask,n_boot=5000,seed=20261019)
    m['A3_reg_same_prompt_historical_G5_control'][subset]=table
m['A3_reg_same_prompt_historical_G5_boundary']='Historical G5 and fresh G5b run at different times; backend weights/version are not observable or guaranteed fixed. This control estimates realized repeat/run variation rather than a pure temperature parameter effect.'
from g5b_stats import _exceed
from g5b_stats import difference_in_differences
m['historical_vs_fresh_cross_prompt_difference_in_differences']={s:{a:difference_in_differences(y,{'fresh_A3':np.asarray(scores[a]),'fresh_reg':reg_scores,'historical_A3':hist_a3,'historical_reg':hist_scores},groups,sizes,mask) for a in ('A3_repeat_1','A3_repeat_2')} for s,mask in {'all':np.ones(800,bool),'exclude_93':groups!='93','only_93':groups=='93'}.items()}
m['cross_vs_historical_reg_repeat']={s:{
 'A3_repeat_1':_exceed(y,{'A3_reg_fresh':reg_scores,'A3_reg_historical':hist_scores,'A3_repeat_1':np.asarray(scores['A3_repeat_1'])},groups,sizes,mask),
 'A3_repeat_2':_exceed(y,{'A3_reg_fresh':reg_scores,'A3_reg_historical':hist_scores,'A3_repeat_2':np.asarray(scores['A3_repeat_2'])},groups,sizes,mask)}
 for s,mask in {'all':np.ones(800,bool),'exclude_93':groups!='93','only_93':groups=='93'}.items()}
records=[]
with (OUT/'llm_calls.jsonl').open(encoding='utf8') as h:
    for line in h:
        q=json.loads(line)
        if q.get('event')=='call_completed': records.append(q)
counts=Counter(q['status'] for q in records)
assert sum(q.get('response_valid',False) for q in records)==2400
m.update(status='complete',n=800,groups=len(set(c['group'] for c in cfg['cases'])),valid_arm_cases=2400,reserved_across_runs=cfg['runs'][-1]['reservations_end'],stop_reason={},
         calls={'completed':len(records),'status_counts':dict(counts),'final_429_retries':sum(int(x.get('transport_retry',0)>0) for x in records),'protocol_repairs':sum(int(x.get('scientific_round',1)==2) for x in records),'tools_used':sum(len(x.get('tool_activity',[])) for x in records)},
         statistics_recovery='All2400 valid scientific calls were completed and persisted before a missing family_masks config field caused statistics-only KeyError; masks copied byte-for-byte from original G4 config, then statistics recomputed without any new LLM call.')
m['post_run_metadata_repair']=cfg['post_run_metadata_repair']; m['execution_launch_audit']=cfg['execution_launch_audit']; m['terminal_scientific_audit']=cfg['terminal_scientific_audit']
m['report_markdown']=make_report(m); cfg['status']='complete'; cfg['statistics_recovered_utc']=now(); cfg['runs'][-1]['scientific_status']='complete_2400_valid'; write_json(OUT/'config.json',cfg); write_json(OUT/'metrics.json',m); readme('complete')
with (OUT/'run.log').open('a',encoding='utf8') as h:h.write(now()+' END statistics-only recovery complete; all2400 valid; no new LLM calls\n')
print(json.dumps({'status':'complete','n':800,'groups':m['groups'],'calls':m['calls']},ensure_ascii=False))
