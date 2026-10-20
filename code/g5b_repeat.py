"""G5b repeatability: two fresh original A3 calls and one A3-reg call on G4's 800 cases."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path
import argparse, csv, ctypes, hashlib, json, os, subprocess, threading, time, traceback

from common import ROOT, load_data, safe_path, write_json
from f2_transport_v2 import call_once, resolve_runtime
from e2_prompts import make_prompt
from f3_inputs import validate_output as validate_a3
from g5_inputs import prompt as prompt_reg, validate as validate_reg
from g5b_stats import build_statistics

OUT=ROOT/'G5b_repeatability'; G4=ROOT/'G4_input_ablation'; F3=ROOT/'F3_full3764'; G5=ROOT/'G5_tc_multithreshold'
SCHEMA_E2=ROOT/'e2_response.schema.json'; INST_E2=ROOT/'e2_instructions.md'; CAT=ROOT/'e2_model_catalog.json'
SCHEMA_REG=ROOT/'g5_response.schema.json'; INST_REG=ROOT/'g5_instructions.md'
EXE=r'C:\Users\28908\AppData\Local\OpenAI\Codex\bin\896a0166b774037a\codex.exe'
DELAYS=(60,120,240); MAX_CALLS=8000; INTEGRITY={'tool_activity_rejected','model_mismatch_rejected','backend_protocol_rejected','preflight_rejected','actual_input_budget_violation'}
ARMS=('A3_repeat_1','A3_repeat_2','A3_reg')

def now(): return datetime.now(timezone.utc).isoformat()
def sha(s): return hashlib.sha256(s.encode('utf8')).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf8'))

def a3_prompt(material,repair=False):
    history=[] if not repair else [{'operational_error':'Previous attempt failed the decision JSON contract. Return a forced decision with numeric p.'}]
    return make_prompt('A3',material,[],{},[],history,round_number=2 if repair else 1,force_decision=False)

def prepare(task):
    assert not OUT.exists(),'G5b directory already exists; inspect rather than overwrite'
    d=load_data(); g4=read(G4/'config.json'); f3=read(F3/'config.json'); g5=read(G5/'config.json')
    ids=[str(v) for v in d['id']]; cases=[]; selected=list(map(int,g4['selected_indices']))
    by_id=f3['cases']; raw_g5=g5.get('cases',{})
    g5_by=({str(k):v for k,v in raw_g5.items()} if isinstance(raw_g5,dict) else {str(v['id']):v for v in raw_g5})
    for i in selected:
        identity=ids[i]; source=by_id[identity]; material=source['material']
        p1=a3_prompt(material); p2=prompt_reg(material)
        assert sha(p1)==source['prompt_hashes']['A3']
        if g5_by: assert sha(p2)==g5_by[identity]['prompt_sha256']
        cases.append({'i':i,'id':identity,'fold':int(d['fold'][i]),'group':str(d['group'][i]),'material':material,
                      'A3_prompt_sha256':sha(p1),'A3_reg_prompt_sha256':sha(p2)})
    cfg={'experiment':'G5b_repeatability','status':'prepared','task_name':task,'created_utc':now(),
         'selected_indices':selected,'cases':cases,'n':800,'groups':int(len(set(c['group'] for c in cases))),
         'arms':list(ARMS),'arm_order':list(ARMS),'concurrency':4,'max_calls':MAX_CALLS,'n_boot':5000,'seed':20261019,'family_masks':g4['family_masks'],
         'runtime':resolve_runtime(),'source_hashes':{},'runs':[],'results':{},'protocol':{
             'exact_original_A3_prompt_reused':True,'A3_reg_prompt':'G5 original A3-reg prompt','tools_disabled':True,
             '429_retry_delays_seconds':list(DELAYS),'max_additional_retries':3,'max_scientific_rounds':2,
             'first_scientific_case_is_probe':True,'development_only':True},
         'g4_config_sha256':sha((G4/'config.json').read_text(encoding='utf8')),'g5_config_sha256':sha((G5/'config.json').read_text(encoding='utf8'))}
    OUT.mkdir(); write_json(OUT/'config.json',cfg)
    (OUT/'llm_calls.jsonl').write_text('',encoding='utf8'); (OUT/'run.log').write_text('',encoding='utf8')
    with (OUT/'predictions.csv').open('w',newline='',encoding='utf8') as h:
        csv.DictWriter(h,fieldnames=['id','fold','group','arm','status','score','y_pred']).writeheader()
    readme('prepared')

def readme(status):
    lines=['# G5b repeatability',f'Status: {status}; 800 G4 materials; development only.',
      'Fresh local logged-in codex exec gpt-6.1-sol/high with tools and network disabled.',
      'A3 repeat 1/2 use the original E2 A3 prompt bytes; A3-reg uses the frozen G5 numeric-output prompt.',
      'Each arm is a new CLI process; first case probes alone, then four concurrent calls.',
      'CLI Reconnecting is allowed to finish; final 429 retries identical prompt after 1,2,4 minutes, max3.',
      'Explicit usage-limit diagnostics stop immediately; three distinct exhausted materials also stop.',
      'At most two scientific rounds; second round only same-evidence JSON contract repair.',
      'Forced prediction is p>=0.5; no fallback or post-hoc numerical correction.',
      'Group-weighted uses 1/full original component size; unweighted uses unit material weight.',
      'All/exclude93/only93, F3 family and original group-size strata; 5000 shared group bootstrap.',
      'Repeatability agreement and BA differences are computed on common valid rows.',
      'No changes are made to G4, G5, F3, or any earlier experiment directory.']
    assert len(lines)<=20; (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

def run(task,restart=0):
    cfg=read(OUT/'config.json'); d=load_data(); assert cfg['task_name']==task and restart==len(cfg['runs']) and restart<=1
    assert cfg['status']=='prepared' or (restart==1 and cfg['status']=='interrupted_error')
    cfg['runs'].append({'index':restart,'started_utc':now(),'reservations_before':0}); cfg['status']='running'; write_json(OUT/'config.json',cfg)
    lock=threading.RLock(); stop=threading.Event(); budget=threading.Event(); results={}; pending=[]; calls=[]; reserved=0; stop_info={}; failed_ids=[]; probe=False
    def log(msg):
        with lock: 
            with (OUT/'run.log').open('a',encoding='utf8') as h:h.write(now()+' '+msg+'\n')
    def append(obj):
        with lock:
            with (OUT/'llm_calls.jsonl').open('a',encoding='utf8') as h:h.write(json.dumps(obj,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n'); h.flush(); os.fsync(h.fileno())
    def reserve(meta):
        nonlocal reserved
        with lock:
            if reserved>=MAX_CALLS: budget.set(); stop_info.setdefault('kind','global_call_budget_exhausted'); return None
            if stop.is_set(): return None
            reserved+=1; meta.update(call_id=reserved,availability_probe=(reserved==1)); append({'event':'call_started',**meta}); return meta
    def work(job):
        case=job['case']; arm=job['arm']; response=None; last=None; retries=0; req=0
        for rnd in (1,2):
            if arm=='A3_reg': msg=prompt_reg(case['material'],True if rnd==2 else None); schema,inst=SCHEMA_REG,INST_REG; validator=validate_reg
            else: msg=a3_prompt(case['material'],rnd==2); schema,inst=SCHEMA_E2,INST_E2; validator=lambda obj: validate_a3(obj,{'id':case['id'],'rule_cards':{}},'A3')
            if rnd==1:
                expected=case['A3_reg_prompt_sha256'] if arm=='A3_reg' else case['A3_prompt_sha256']; assert sha(msg)==expected
            while not stop.is_set():
                meta=reserve({'experiment':'G5b_repeatability','run_index':restart,'arm':arm,'id':case['id'],'scientific_round':rnd,'transport_retry':retries,'prompt':msg,'request_hash':sha(msg),'started_utc':now()})
                if meta is None: break
                req+=1; rec=call_once(msg,schema_path=schema,instructions_path=inst,catalog_path=CAT,stop_event=stop,metadata=meta)
                if rec['status']=='ok':
                    try: response=validator(json.loads(rec['response_text']))
                    except Exception as exc: rec.update(status='protocol_error',protocol_error=str(exc))
                rec.update(event='call_completed',finished_utc=now(),response_valid=response is not None,parsed_response=response); append(rec); calls.append(rec); last=rec
                if rec['status']=='terminal_usage_limit': stop_info.update(kind='explicit_usage_limit',diagnostics=rec.get('errors'),call_id=rec['call_id']); stop.set()
                elif rec['status'] in INTEGRITY: stop_info.update(kind='integrity_violation',status=rec['status'],call_id=rec['call_id']); stop.set()
                if response is not None or stop.is_set(): break
                if rec['status']=='rate_limited_final' and retries<3:
                    delay=DELAYS[retries]; retries+=1; log(f'FINAL429 {case["id"]}/{arm} wait={delay}s retry={retries}/3');
                    if stop.wait(delay): break
                    continue
                break
            if response is not None or stop.is_set() or budget.is_set() or last is None or last['status']=='rate_limited_final': break
            if last['status'] not in {'protocol_error','cli_error','backend_error','incomplete_response','transport_error'}: break
        return job,{'response':response,'status':last['status'] if last else 'not_dispatched','requests':req,'retries':retries}
    def flush(force=False):
        if len(pending)>=10 or (force and pending):
            with (OUT/'predictions.csv').open('a',newline='',encoding='utf8') as h: csv.DictWriter(h,fieldnames=['id','fold','group','arm','status','score','y_pred']).writerows(pending); h.flush(); os.fsync(h.fileno())
            pending.clear()
    jobs=[{'case':c,'arm':a} for c in cfg['cases'] for a in ARMS]; jobs.sort(key=lambda j:sha('G5b|'+j['arm']+'|'+j['case']['id']))
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            it=iter(jobs); active={pool.submit(work,next(it)):None}; exhausted=False
            while active:
                done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
                for fut in done:
                    active.pop(fut); job,res=fut.result(); key=(job['case']['id'],job['arm']); results[key]=res
                    if res['response'] is not None:
                        p=float(res['response'].get('p',res['response'].get('p_gt_10K'))); pending.append({'id':job['case']['id'],'fold':job['case']['fold'],'group':job['case']['group'],'arm':job['arm'],'status':'valid','score':p,'y_pred':int(p>=.5)})
                    else:
                        pending.append({'id':job['case']['id'],'fold':job['case']['fold'],'group':job['case']['group'],'arm':job['arm'],'status':res['status'],'score':'','y_pred':''})
                        if job['case']['id'] not in failed_ids: failed_ids.append(job['case']['id'])
                    flush(); log(f'PROGRESS {len(results)}/{len(jobs)} {job["case"]["id"]}/{job["arm"]} {res["status"]} requests={res["requests"]} retries={res["retries"]} total_reserved={reserved}')
                    if res['response'] is not None: probe=True
                    if len(failed_ids)>=3 and not stop.is_set(): stop_info.update(kind='three_consecutive_exhausted_materials',material_ids=failed_ids[-3:]); stop.set()
                while not stop.is_set() and not budget.is_set() and not exhausted and len(active)<(4 if probe else 1):
                    try: active[pool.submit(work,next(it))]=None
                    except StopIteration: exhausted=True
        flush(True); valid=sum(1 for x in results.values() if x['response'] is not None); complete=valid==len(jobs)
        cfg['results']={f'{k[0]}|{k[1]}':v['response'] for k,v in results.items() if v['response'] is not None}; cfg['status']='complete' if complete else 'stopped_partial'; cfg['runs'][-1].update(finished_utc=now(),reservations_end=reserved,status=cfg['status']); write_json(OUT/'config.json',cfg)
        if complete:
            scores={a:[] for a in ARMS}; lookup={(c['id'],a):c for c in cfg['cases'] for a in ARMS}
            for c in cfg['cases']:
                for a in ARMS: scores[a].append(float(results[(c['id'],a)]['response'].get('p',results[(c['id'],a)]['response'].get('p_gt_10K'))))
            metrics=build_statistics(cfg,d,scores)
            metrics.update({'status':'complete','n':800,'groups':len(set(c['group'] for c in cfg['cases'])), 'valid_arm_cases':valid,'reserved_across_runs':reserved,'stop_reason':stop_info,
                            'calls':{'reserved':reserved,'completed':len(calls),'final_429_retries':sum(int(x.get('transport_retry',0)>0) for x in calls),'protocol_repairs':sum(int(x.get('scientific_round',1)==2) for x in calls),'tools_used':sum(len(x.get('tool_activity',[])) for x in calls)},
                            'report_markdown':make_report(metrics)})
            write_json(OUT/'metrics.json',metrics); readme('complete')
        else:
            write_json(OUT/'metrics.json',{'status':cfg['status'],'valid_arm_cases':valid,'reserved_across_runs':reserved,'stop_reason':stop_info}); readme(cfg['status'])
        log(f'END status={cfg["status"]} valid={valid}/{len(jobs)} reserved={reserved} stop={stop_info}'); return 0 if complete else 75
    except BaseException:
        stop.set(); flush(True); cfg['status']='interrupted_error'; cfg['runs'][-1].update(status='interrupted_error',finished_utc=now(),reservations_end=reserved); write_json(OUT/'config.json',cfg); log('FATAL '+traceback.format_exc()); raise
    finally: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

def _fmt(row,metric='balanced_accuracy'):
    if row is None or row.get('point') is None:return 'NA'
    return f'{100*row["point"]:.2f} [{100*row["ci95"][0]:.2f},{100*row["ci95"][1]:.2f}]'
def make_report(m):
    lines=['# G5b 重复性检查','',f'n=800；组数={m["groups"]}；5000次按原component_id组bootstrap；全部结果均为开发集。','',
           '指标格式：BA/精确率/召回率/准确率（百分比；方括号为95% CI）。']
    for subset,label in [('all','全部'),('exclude_93','去掉93'),('only_93','仅93')]:
        t=m['subsets'][subset]; lines += [f'## {label}（n={t["n"]}，组={t["n_groups"]}）','', '|权重|方法|BA|精确率|召回率|准确率|','|---|---|---:|---:|---:|---:|']
        for w,wl in [('group_weighted','组加权'),('unweighted','不加权')]:
            for name in ('A3_repeat_1','A3_repeat_2','A3_reg'):
                q=t[w]['metrics'][name]; lines.append('|'+wl+'|'+name+'|'+'|'.join(_fmt(q,x) for x in ('balanced_accuracy','precision','recall','accuracy'))+'|')
        lines += ['', '一致率（组加权 / 不加权）：']
        for pair,r in m['agreement'][subset].items(): lines.append(f'- {pair}: {_fmt(r["group_weighted"])} / {_fmt(r["unweighted"])}')
        ex=m['exceedance'][subset]; lines += ['',f'A3 与 A3-reg 的 BA 差（点估计）及其相对 A3 重复波动的 bootstrap 超越比例：{ex}','','']
    from f_report import _classification_section
    for title,collections in [('F3家族',m.get('families',{})),('原组大小',m.get('group_size_bins',{})),('G3预测家族',m.get('G3_predicted_families',{}))]:
        for name,tables in collections.items():
            for subset,label in [('all','全部'),('exclude_93','去掉93'),('only_93','仅93')]:
                t=tables[subset]; _classification_section(lines,f'{title}·{name}·{label}',t,list(ARMS),[('A3_repeat_1','A3_repeat_2'),('A3_repeat_1','A3_reg'),('A3_repeat_2','A3_reg')])
    lines += ['', '## A3-reg 同提示词历史G5对照', '',m.get('A3_reg_same_prompt_historical_G5_boundary','')]
    for subset,t in m.get('A3_reg_same_prompt_historical_G5_control',{}).items():
        _classification_section(lines,'A3-reg历史对照·'+subset,t,['A3_reg_fresh','A3_reg_historical_G5'],[('A3_reg_fresh','A3_reg_historical_G5')])
    for subset,t in m.get('historical_A3_and_reg_context',{}).items():
        ns=['A3_historical_F3','A3_repeat_1','A3_repeat_2','A3_reg_historical_G5','A3_reg_fresh']
        from itertools import combinations
        _classification_section(lines,'历史与新调用共同800材料·'+subset,t,ns,list(combinations(ns,2)))
    lines += ['', '## 配对差超过重复波动的直接bootstrap检验', '', '统计量=|BA(跨提示)-BA(参照)|−|BA(同提示重复1)-BA(同提示重复2)|；共享5000组draw直接取CI，非比较独立CI。Bootstrap超越比例不是p值；仅93为单组CI退化。', '', '|子集|参照|权重|差值pp|95% CI pp|bootstrap超越比例|有效draw|','|---|---|---|---:|---:|---:|---:|']
    for subset,refs in m.get('exceedance',{}).items():
        for ref,ws in refs.items():
            for w,r in ws.items():
                if r:lines.append(f'|{subset}|{ref}|{w}|{100*r["point_abs_cross_minus_abs_within_BA"]:.2f}|{[round(100*x,3) for x in r["ci95_abs_cross_minus_abs_within_BA"]]}|{r["bootstrap_fraction_abs_A3_vs_reg_exceeds_repeat"]:.4f}|{r["n_valid_boot"]}|')
    lines += ['', '## 历史与本轮跨提示差的difference-in-differences', '', '定义=(fresh A3−fresh reg)−(historical A3−historical reg)，共享5000组draw；不同run/backend不可锁定，不能作纯提示因果效应。', '', '|子集|freshA3参照|权重|DiD pp|95% CI pp|有效draw|','|---|---|---|---:|---:|---:|']
    for subset,refs in m.get('historical_vs_fresh_cross_prompt_difference_in_differences',{}).items():
        for ref,ws in refs.items():
            for w,r in ws.items():
                if r:lines.append(f'|{subset}|{ref}|{w}|{100*r["point"]:.2f}|{[round(100*x,3) for x in r["ci95"]]}|{r["n_valid_boot"]}|')
    return '\n'.join(lines)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=('prepare','run')); p.add_argument('--task-name',required=True); p.add_argument('--restart-index',type=int,default=0); a=p.parse_args()
    if a.action=='prepare': prepare(a.task_name)
    else: raise SystemExit(run(a.task_name,a.restart_index))
