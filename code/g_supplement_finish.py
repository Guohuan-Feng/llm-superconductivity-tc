"""One finite, read-only evaluator watcher followed by audits/statistics/export.

No model calls, automatic retries, or subsequent research stages.
"""
from __future__ import annotations
import ctypes,hashlib,json,time,traceback,csv,subprocess
from pathlib import Path
from datetime import datetime,timezone
from html.parser import HTMLParser
from common import ROOT,safe_path
from supplement_io import write_json
from g_stage1_export import safe,inline,table,interval,markdown_tables

WORK=ROOT/'work/G_supplement_runtime'
FOLDERS=('G5c_majority3764','G7b_supplement')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()

def main_task_state(task_name):
    """Read the finite Windows task, without starting or changing it."""
    # task_name is frozen metadata, carried through the environment rather than shell text.
    import os
    env=dict(os.environ);env['CODEX_SUPPLEMENT_TASK_TO_QUERY']=task_name
    script="$ErrorActionPreference='Stop'; (Get-ScheduledTask -TaskName $env:CODEX_SUPPLEMENT_TASK_TO_QUERY).State.ToString()"
    try:
        r=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',script],capture_output=True,text=True,timeout=20,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    except Exception as e:return 'QueryError:'+type(e).__name__
    return r.stdout.strip() if r.returncode==0 else 'QueryError'

def await_terminal(state,task_state_fn=main_task_state,sleep_fn=time.sleep):
    """Three consecutive five-second non-Running observations detect dead jobs."""
    inactive=0;query_errors=0
    while True:
        s=read(WORK/'session.json')
        if s['status'] not in ('prepared','running'):return s
        actual=task_state_fn(s['task_name'])
        if actual not in {'Running','Ready','Disabled','Queued','NotFound'}:
            query_errors+=1;inactive=0
            if query_errors>=3:
                reason={'kind':'task_state_query_failed','task_name':s['task_name'],'last_observation':actual,'consecutive_query_errors':query_errors}
                state.update(status='monitor_query_error',completed_utc=now(),stop_reason=reason);write_json(WORK/'completion.json',state)
                # A monitoring failure is not evidence that the scientific task died.
                # Preserve the real session rather than overwriting its running state.
                return {**s,'status':'monitor_query_error','stop_reason':reason}
            sleep_fn(5);continue
        query_errors=0
        inactive=0 if actual=='Running' else inactive+1
        if inactive>=3:
            reason={'kind':'main_task_not_running_with_nonterminal_session','task_name':s['task_name'],'last_observed_task_state':actual,'consecutive_nonrunning_observations':inactive,'poll_interval_seconds':5}
            s.update(status='interrupted_detected',detected_utc=now(),stop_reason=reason);write_json(WORK/'session.json',s)
            state.update(status='interrupted_detected',completed_utc=now(),inference_status=s['status'],stop_reason=reason);write_json(WORK/'completion.json',state)
            return s
        sleep_fn(5)

def conclusions(g5,g7):
    def comparison(s,w,a,b,key):
        for p in s[w]['paired_differences'].values():
            if p['arm_a']==a and p['arm_b']==b:return interval(p[key])
        raise KeyError((w,a,b,key))
    lines=[]
    for sub,label in [('all','全部'),('exclude_93','去掉93组')]:
        s=g5['subsets'][sub]
        values=[]
        for w,wl in [('group_weighted','组加权'),('unweighted','不加权')]:
            values.append(wl+'多数票BA '+interval(s[w]['metrics']['A3_majority']['balanced_accuracy'])+'；相对HGB127 ΔBA '+comparison(s,w,'A3_majority','HGB127','balanced_accuracy')+' pp')
        lines.append('G5c '+label+'：'+'；'.join(values)+'。')
    s=g5['subsets']['all'];parts=[]
    for w,label in [('group_weighted','组加权'),('unweighted','不加权')]:
        parts.append(label+'多数票精确率 '+interval(s[w]['metrics']['A3_majority']['precision'])+'；相对HGB127差 '+comparison(s,w,'A3_majority','HGB127','precision')+' pp')
    lines.append('G5c 精确率：'+'；'.join(parts)+'。三次一致率及各基线的配对差见完整表。')
    parts=[]
    for threshold,stem in [('5','GPT_A3reg5'),('10','GPT_A3_condition10')]:
        s=g7['cohorts']['primary69'][threshold]
        parts.append('>'+threshold+'K：'+', '.join(label+' ΔBA '+comparison(s,w,stem+'_majority','HGB_composition118','balanced_accuracy')+' pp' for w,label in [('group_weighted','组加权'),('unweighted','不加权')]))
    lines.append('G7b 严格69例中GPT多数票相对成分HGB：'+'；'.join(parts)+'。完整89例仅作敏感性分析。')
    lines.append('边界：G5c是开发结果；G7b新材料的日期门槛来自本地模型预训练截止，不能保证晚于GPT全部训练。10K条件句与多数聚合同时改变，不能单独识别条件句效应；5K回归无理由字段，错因原文只解释10K调用。')
    return lines

def audit(session):
    for p,h in session['source_hashes'].items():assert sha(p)==h,('Changed frozen evaluator source',p)
    records=[];out={};events=[]
    for folder in FOLDERS:
        cfg=read(ROOT/folder/'config.json');assert cfg['status']=='inference_completed'
        for p,h in cfg['source_hashes'].items():assert sha(p)==h,('Changed input source',p)
        jobs={j['job_id']:j for j in cfg['pending_jobs']}
        assert len(cfg['results'])==len(jobs)
        seen=set();valid={};starts={};completed={}
        with (ROOT/folder/'llm_calls.jsonl').open(encoding='utf8') as f:
            for line in f:
                r=json.loads(line);key=r['call_id'];j=jobs[r['job_id']]
                assert r['id']==j['id'] and r['arm']==j['arm']
                assert hashlib.sha256(r['prompt'].encode('utf8')).hexdigest()==r['request_hash']
                if r['scientific_round']==1:assert r['request_hash']==j['prompt_sha256']
                if r['event']=='call_started':
                    assert key not in starts;starts[key]=r
                    events.append((r['started_utc'],1,key))
                else:
                    assert r['event']=='call_completed' and key not in completed;completed[key]=r
                    events.append((r['finished_utc'],-1,key))
                    assert not r.get('tool_activity')
                    if r['response_valid']:
                        assert r['status']=='ok' and r['model']=='gpt-6.1-sol'
                        assert r['reasoning_effort']=='high' and r['runtime']==session['runtime']
                        assert r['job_id'] not in valid;valid[r['job_id']]=r['parsed_response']
                records.append(r)
        assert set(starts)==set(completed) and valid==cfg['results']
        with (ROOT/folder/'predictions.csv').open(newline='',encoding='utf8') as f:
            rows=list(csv.DictReader(f))
        for r in rows:
            job_id=r['id']+'|'+r['arm']
            if job_id not in jobs:continue
            assert job_id not in seen;seen.add(job_id);j=jobs[job_id]
            score=float(valid[job_id][j.get('score_key','p')])
            assert r['status']=='valid' and float(r['score'])==score and int(r['y_pred'])==int(score>=.5)
        assert seen==set(jobs)
        out[folder]={'valid_new_tasks':len(valid),'actual_requests':len(starts),'no_tools':True,'prompts_and_csv_verified':True}
    active=peak=0
    for _,delta,key in sorted(events,key=lambda v:(v[0],v[1])):
        active+=delta;peak=max(peak,active);assert 0<=active<=4
    assert active==0
    assert sum(v['actual_requests'] for v in out.values())==session['calls']
    return {'status':'passed','audited_utc':now(),'global_max_concurrent':peak,'experiments':out,'frozen_sources_unchanged':True}

def export(audit_result):
    g5=read(ROOT/FOLDERS[0]/'metrics.json');g7=read(ROOT/FOLDERS[1]/'metrics.json')
    assert g5['status']==g7['status']=='complete'
    lines=conclusions(g5,g7)
    blocks=['<h1>G5c／G7b 补充结果</h1>',
        '<p>G5c 为原 3764 条开发结果；G7b 固定严格 69 例为主、完整 89 例为敏感性分析。LLM 主结果均使用三次二元判断的多数票。所有区间为按组 bootstrap 5000 次的 95% CI；表内均注明样本数与组数。两项完成后停止，未进入 G8／G9。</p>',
        '<h2>结论与边界</h2>',''.join('<p>'+inline(s)+'</p>' for s in lines),'<h2>G5c 主结果</h2>']
    keys=('balanced_accuracy','precision','recall','accuracy')
    for subset,s in g5['subsets'].items():
        blocks.append(f'<h3>{inline(subset)} · n={s["n"]}，组数={s["n_groups"]}，正类={s["n_positive"]}</h3>')
        for w,label in [('group_weighted','组加权'),('unweighted','不加权')]:
            rows=[[name]+[interval(v[k]) for k in keys] for name,v in s[w]['metrics'].items()]
            blocks.append('<h4>'+label+'</h4>'+table(['方法','BA','精确率','召回率','准确率'],rows))
    blocks.append('<h2>G7b 主结果</h2>')
    for cohort,by_threshold in g7['cohorts'].items():
        for threshold,s in by_threshold.items():
            blocks.append(f'<h3>{inline(cohort)} / Tc&gt;{threshold}K · n={s["n"]}，正类={s["n_positive"]}，组数={s["n_groups"]}</h3>')
            for w,label in [('group_weighted','组加权'),('unweighted','不加权')]:
                rows=[[name]+[interval(v[k]) for k in keys] for name,v in s[w]['metrics'].items()]
                blocks.append('<h4>'+label+'</h4>'+table(['方法','BA','精确率','召回率','准确率'],rows))
    blocks.append('<h2>执行审计</h2><pre>'+inline(json.dumps(audit_result,ensure_ascii=False,indent=2))+'</pre>')
    for title,m in [('G5c 全表：单次、投票、配对差、家族、组大小及一致率',g5),('G7b 全表：两个阈值、固定队列、家族及错误理由原文',g7)]:
        blocks.append('<details><summary>'+title+'</summary>'+markdown_tables(m['report_markdown'])+'</details>')
    text='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>G5c／G7b 补充结果</title><style>body{font:15px/1.6 system-ui,sans-serif;margin:28px;color:#17212b}h1,h2,h3{line-height:1.3}.scroll{overflow:auto;margin:12px 0}table{border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{border:1px solid #c9d2da;padding:7px 10px;white-space:nowrap;text-align:left}th{background:#edf2f7}summary{cursor:pointer;font-weight:650;margin:18px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><main>'+'\n'.join(blocks)+'</main></html>'
    parser=HTMLParser(convert_charrefs=True);parser.feed(text);parser.close()
    out=safe(ROOT/'outputs/G5c_G7b_results.html');out.parent.mkdir(parents=True,exist_ok=True);out.write_text(text,encoding='utf8')
    return {'path':str(out),'sha256':sha(out),'bytes':out.stat().st_size,'conclusion_lines':lines}

def main():
    for p in (ROOT,WORK,*ROOT.parents):assert not p.is_symlink() and not p.is_junction() and 'onedrive' not in str(p.resolve()).lower()
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    sources=[ROOT/n for n in ('g_supplement_finish.py','supplement_io.py','g5c_stats.py','g7b_stats.py','f_stats.py','f3_stats.py','f_report.py','g_stage1_export.py')]
    state={'status':'waiting_for_inference','created_utc':now(),'no_model_calls':True,'finite_one_shot':True,'source_hashes':{str(p):sha(p) for p in sources}}
    write_json(WORK/'completion.json',state)
    try:
        s=await_terminal(state)
        if s['status']!='inference_completed':
            status=s['status'] if s['status'] in {'interrupted_detected','monitor_query_error'} else 'stopped_without_statistics'
            state.update(status=status,inference_status=s['status'],stop_reason=s.get('stop_reason'),completed_utc=now());write_json(WORK/'completion.json',state);return 75
        for p,h in state['source_hashes'].items():assert sha(p)==h,('Postprocessor source changed',p)
        state['status']='auditing';write_json(WORK/'completion.json',state);checked=audit(s)
        import g5c_stats,g7b_stats
        # These modules historically imported the shared non-atomic writer. Override
        # only this new supplement process; old code and experiment files stay unchanged.
        g5c_stats.write_json=write_json;g7b_stats.write_json=write_json
        g5c_stats.finalize();g7b_stats.main(finalize=True)
        for folder in FOLDERS:
            p=ROOT/folder;m=read(p/'metrics.json');m['terminal_scientific_audit']=checked;write_json(p/'metrics.json',m)
            assert {q.name for q in p.iterdir()}=={'config.json','predictions.csv','llm_calls.jsonl','metrics.json','run.log','README.md'}
            assert len((p/'README.md').read_text(encoding='utf8').splitlines())<=20
        output=export(checked);state.update(status='complete',completed_utc=now(),audit=checked,report=output);write_json(WORK/'completion.json',state);return 0
    except BaseException:
        state.update(status='error',completed_utc=now(),traceback=traceback.format_exc());write_json(WORK/'completion.json',state);raise
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':raise SystemExit(main())
