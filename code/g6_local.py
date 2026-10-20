"""Finite local Llama G6; localhost inference has no tools or remote requests."""
from common import ROOT, load_data, write_json, np
import argparse, csv, ctypes, hashlib, json, math, os, subprocess, time, traceback, http.client, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from e2_prompts import make_prompt, MODEL_INSTRUCTIONS, SCHEMA as A3_SCHEMA
from f2_prompts import make_prompt as m1_prompt
from g5_inputs import prompt as reg_prompt, validate as reg_validate

OUT=ROOT/'G6_local_cutoff'; WORK=ROOT/'work/G6_runtime'
EXE=WORK/'bin/llama-server.exe'; MODEL=WORK/'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf'
PORT=18631; SEED=20261019
HTTP=threading.local()
COLUMNS=['id','fold','group','arm','status','score','y_pred','Tc_est_K','p_gt_5K','p_gt_10K','p_gt_15K','abstain','requests']
def now(): return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(2**20),b''): h.update(x)
    return h.hexdigest()
def digest(s): return hashlib.sha256(s.encode('utf8')).hexdigest()
def api(route,data=None,timeout=600):
    # Explicit loopback connection bypasses system proxy; reuse sockets to avoid
    # Windows ephemeral-port exhaustion during the7828-case tokenization audit.
    if not hasattr(HTTP,'connection'):HTTP.connection=http.client.HTTPConnection('127.0.0.1',PORT,timeout=timeout)
    c=HTTP.connection;c.timeout=timeout
    try:
        c.request('GET' if data is None else 'POST','/'+route,body=None if data is None else json.dumps(data,ensure_ascii=False).encode('utf8'),headers={'Content-Type':'application/json','Connection':'keep-alive'})
        r=c.getresponse();raw=r.read()
        if r.status!=200:raise RuntimeError(f'localhost HTTP {r.status}: '+raw.decode('utf8',errors='replace')[:500])
        return json.loads(raw)
    except Exception:
        c.close();del HTTP.connection;raise
def log(text):
    with (OUT/'run.log').open('a',encoding='utf8') as f:f.write(now()+' '+text+'\n')
def a3_validate(obj):
    if not isinstance(obj,dict) or set(obj)!=set(A3_SCHEMA['required']):raise ValueError('exact schema keys required')
    if obj['action']!='decide' or obj['selected_rule_ids']!=[] or obj['request_rule_ids']!=[] or obj['request_counterexamples'] is not False:raise ValueError('direct decision only')
    if not isinstance(obj['abstain'],bool) or not isinstance(obj['reason'],str) or len(obj['reason'])>80:raise ValueError('abstain/reason contract')
    p=obj['p']
    if isinstance(p,bool) or not isinstance(p,(int,float)) or not math.isfinite(p) or not 0<=p<=1:raise ValueError('finite p required')
    return obj
def cases_from_sources():
    f3=read(ROOT/'F3_full3764/config.json'); g5=read(ROOT/'G5_tc_multithreshold/config.json');f2=read(ROOT/'F2_mask/config.json')
    g5_by={str(c['id']):c for c in g5['cases']}
    cases=[]
    for key in f3['case_order']:
        source=f3['cases'][key]; material=source['material']
        a3=make_prompt('A3',material,[],{},[],[],round_number=1)
        reg=reg_prompt(material)
        assert digest(a3)==source['prompt_hashes']['A3']
        assert digest(reg)==g5_by[key]['prompt_sha256']
        assert a3.split('Scientific inputs:\n')[1]==reg.split('Scientific inputs:\n')[1]
        cases.append(dict(id=key,i=source['i'],A3_prompt=a3,A3_reg_prompt=reg))
    m1=[]
    for c in f2['cases']:
        p=m1_prompt('A3-M1',c['packets']['A3-M1'],round_number=1)
        expected=c['prompt_hashes']['A3-M1']
        if isinstance(expected,dict):expected=expected.get('1',expected.get('round1'))
        assert digest(p)==expected,(c['id'],expected)
        m1.append(dict(id=c['id'],i=c['i'],prompt=p))
    return f3,g5,f2,cases,m1
def prepare():
    for p in (ROOT,WORK,*ROOT.parents):assert 'onedrive' not in str(p.resolve()).lower() and not p.is_symlink() and not p.is_junction()
    assert MODEL.is_file() and EXE.is_file()
    previous=read(OUT/'config.json') if (OUT/'config.json').exists() else None
    f3,g5,f2,cases,m1=cases_from_sources()
    cfg={'experiment':'G6_local_cutoff','status':'prepared','created_utc':now(),'model':'Meta-Llama-3.1-8B-Instruct','quantization':'Q4_K_M',
         'official_cutoff_month':'2023-12','conservative_eligibility_date':'2024-01-31','official_cutoff_source':'https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md',
         'cutoff_quote':'The pretraining data has a cutoff of December 2023.',
         'cutoff_boundary':'Official pretraining cutoff is not a guarantee that alignment/distillation contains no later knowledge; comparison also changes capacity, training and quantization.',
         'checkpoint_repository':'bartowski/Meta-Llama-3.1-8B-Instruct-GGUF','checkpoint_commit':'bf5b95e96dac0462e2a09145ec66cae9a3f12067',
         'checkpoint_sha256':sha(MODEL),'checkpoint_bytes':MODEL.stat().st_size,'checkpoint_original':'meta-llama/Meta-Llama-3.1-8B-Instruct; quantization only, no new finetuning',
         'runtime_release':'llama.cpp b11744 / commit 7adf160c4','runtime_archive_sha256':sha(WORK/'llama-runtime.zip'),'cudart_archive_sha256':sha(WORK/'cudart.zip'),
         'runtime_files_sha256':{p.name:sha(p) for p in (WORK/'bin').iterdir() if p.suffix in ('.exe','.dll')},
         'parameters':{'temperature':0,'seed':SEED,'max_tokens':512,'slots':4,'total_context':32768,'context_per_slot':8192,'flash_attention':'on','kv_dtype':'f16','gpu_layers':99,
                       'max_scientific_rounds':2,'flush_every':10,'tools':False,'network':'local loopback only; no model web/tools','json_schema_constraint':True},
         'chat_template':'GGUF embedded official Llama3.1 template; captured from /props before inference',
         'cases':cases,'M1_cases':m1,'A3_instructions':MODEL_INSTRUCTIONS,'reg_instructions':(ROOT/'g5_instructions.md').read_text(encoding='utf8'),'M1_instructions':(ROOT/'f2_instructions.md').read_text(encoding='utf8'),
         'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'F3_full3764/config.json',ROOT/'G5_tc_multithreshold/config.json',ROOT/'F2_mask/config.json',ROOT/'e2_prompts.py',ROOT/'f2_prompts.py',ROOT/'g5_inputs.py',ROOT/'g6_local.py']},
         'family_masks':f3['family_masks'],'raw_a15_mask':f3['raw_a15_mask'],'G3_family_masks':g5['G3_family_masks'],'task_name':'Codex.V2Diagnosis.G6.20261019'}
    if previous and previous.get('status')=='stopped_error':
        cfg['preflight_restart_audit']={'previous_status':previous['status'],'previous_traceback':previous.get('traceback'),'scientific_outputs_before_restart':0,'task_result_code':1,
            'system_task_logs_checked':True,'related_system_errors':[],'repair':'direct loopback persistent HTTP connections; cached system token counts; no scientific prompt/model/generation parameter changes','whole_recomputations':1}
    assert cfg['checkpoint_sha256']=='7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c'
    OUT.mkdir(exist_ok=True);write_json(OUT/'config.json',cfg)
    write_json(OUT/'metrics.json',{'status':'prepared','requested_n':3764,'requested_M1_n':300})
    (OUT/'run.log').write_text(now()+' PREPARED; previous premature operational block superseded after direct curl succeeded. No scientific calls yet.\n',encoding='utf8')
    (OUT/'llm_calls.jsonl').write_text('',encoding='utf8')
    with (OUT/'predictions.csv').open('w',newline='',encoding='utf8') as f:csv.DictWriter(f,fieldnames=COLUMNS).writeheader()
    (OUT/'README.md').write_text('\n'.join(['# G6 local early-cutoff model','Status: prepared; development-only original cohort.', 'Meta Llama3.1 8B Instruct, original frozen A3 and G5 A3-reg prompts; F2 A3-M1 fixed300.',
        'Official pretraining cutoff December2023; this does not certify no later alignment/distillation information.',
        'Q4_K_M checkpoint and llama.cpp release/archives/binary SHAs frozen in config.',
        'Temperature0, seed20261019, max512outputtokens, four slots,8192context/slot, no truncation.',
        'Fresh slot conversation each call; no tools/web; localhost inference only.',
        'Original prompts and model instructions byte/hash revalidated; max2 scientific rounds, second format-only repair.',
        'Every valid/invalid response saved; failed records never imputed; p retained even when abstain=true.',
        'Group-weighted1/original group size and unweighted; three subsets; family/G3/size;5000paired group bootstrap.',
        'ML probabilities reuse original3fold OOF; no new tuning.',
        'One-shot scheduled task PT0S/restart0; SetThreadExecutionState; append/fsync every10completed tasks.',
        'Interruption diagnosed from task/system logs first; no silent per-record continuation or unlimited restarts.'])+'\n',encoding='utf8')
    print('prepared',len(cases),len(m1))
def call_task(task,cfg):
    arm,c=task; key='A3_prompt' if arm=='local_A3' else 'A3_reg_prompt' if arm=='local_A3_reg' else 'prompt'
    prompt=c[key]; system=cfg['reg_instructions' if arm=='local_A3_reg' else 'M1_instructions' if arm=='local_A3_M1' else 'A3_instructions']
    schema=read(ROOT/'g5_response.schema.json') if arm=='local_A3_reg' else A3_SCHEMA
    records=[];obj=None
    for round_no in (1,2):
        actual=prompt if round_no==1 else prompt+'\nFormat-only repair: return exactly the required JSON schema using the same scientific evidence. Do not add prose or request evidence.'
        payload={'model':'local','messages':[{'role':'system','content':system},{'role':'user','content':actual}],
                 'temperature':0,'seed':SEED,'max_tokens':512,'stream':False,'cache_prompt':False,
                 'response_format':{'type':'json_schema','json_schema':{'name':'scientific_output','strict':True,'schema':schema}}}
        start=time.monotonic(); rec={'id':c['id'],'arm':arm,'round':round_no,'started_utc':now(),'prompt':actual,'instructions':system,'prompt_sha256':digest(actual),'tools':[]}
        try:
            raw=api('v1/chat/completions',payload)
            rec['raw_response']=raw; content=raw['choices'][0]['message']['content']; parsed=json.loads(content)
            obj=reg_validate(parsed) if arm=='local_A3_reg' else a3_validate(parsed)
            if raw['choices'][0].get('finish_reason')=='length':raise ValueError('output length exhaustion')
            rec['status']='ok';rec['validated_output']=obj
        except Exception as exc:rec['status']='invalid';rec['error']=repr(exc);obj=None
        rec['seconds']=time.monotonic()-start; records.append(rec)
        if obj is not None:break
    return task,obj,records
def run():
    cfg=read(OUT/'config.json'); assert cfg['status']=='prepared'
    assert sha(MODEL)==cfg['checkpoint_sha256']
    for path,h in cfg['source_sha256'].items():assert sha(ROOT/path)==h,(path,'source_changed')
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    cmd=[str(EXE),'-m',str(MODEL),'--host','127.0.0.1','--port',str(PORT),'-ngl','99','-c','32768','-np','4','-fa','on','--seed',str(SEED),'--jinja','--no-webui']
    err=(WORK/'server.stderr.log').open('w',encoding='utf8');out=(WORK/'server.stdout.log').open('w',encoding='utf8')
    proc=subprocess.Popen(cmd,cwd=str(WORK/'bin'),stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
    log('SERVER_STARTED pid='+str(proc.pid));cfg['server_command']=cmd
    try:
        for attempt in range(120):
            if proc.poll() is not None:raise RuntimeError('llama-server exited '+str(proc.returncode))
            try:
                props=api('props',timeout=2)
                if props.get('default_generation_settings') is not None:break
            except Exception:pass
            time.sleep(2)
        else:raise TimeoutError('server readiness')
        cfg['server_props']=props;cfg['chat_template']=props.get('chat_template');cfg['status']='running';cfg['started_utc']=now();write_json(OUT/'config.json',cfg)
        tasks=[('local_A3',c) for c in cfg['cases']]+[('local_A3_reg',c) for c in cfg['cases']]+[('local_A3_M1',c) for c in cfg['M1_cases']]
        # Tokenization preflight of every scientific prompt, no labels/outcome-dependent setting changes.
        max_input=0
        system_counts={key:len(api('tokenize',{'content':cfg[key]})['tokens']) for key in ('A3_instructions','M1_instructions','reg_instructions')}
        for number,(arm,c) in enumerate(tasks,1):
            p=c['A3_prompt' if arm=='local_A3' else 'A3_reg_prompt' if arm=='local_A3_reg' else 'prompt']
            # conservative instruction/template allowance; tokenize exact user text and systems independently
            n=len(api('tokenize',{'content':p})['tokens'])+system_counts['reg_instructions' if arm=='local_A3_reg' else 'M1_instructions' if arm=='local_A3_M1' else 'A3_instructions']+160
            assert n+512+100<8192,(arm,c['id'],n)
            max_input=max(max_input,n)
            if number%500==0:log(f'TOKEN_AUDIT {number}/{len(tasks)}')
        cfg['max_prompt_token_bound']=max_input;write_json(OUT/'config.json',cfg);log('PREFLIGHT max_input_bound='+str(max_input))
        d=load_data(); pending=[];valid=0;done=0
        with (OUT/'llm_calls.jsonl').open('a',encoding='utf8') as calls,(OUT/'predictions.csv').open('a',newline='',encoding='utf8') as pf:
            writer=csv.DictWriter(pf,fieldnames=COLUMNS)
            # First real scientific task runs alone as operational probe; then fixed4concurrent.
            results=[call_task(tasks[0],cfg)]
            def save(result):
                nonlocal valid,done,pending
                (arm,c),obj,recs=result;done+=1;valid+=obj is not None
                for r in recs:calls.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
                calls.flush();os.fsync(calls.fileno())
                i=c['i'];row=dict(id=c['id'],fold=int(d['fold'][i]),group=str(d['group'][i]),arm=arm,status='ok' if obj is not None else 'invalid_after_two_rounds',score='',y_pred='',Tc_est_K='',p_gt_5K='',p_gt_10K='',p_gt_15K='',abstain='',requests=len(recs))
                if obj is not None:
                    if arm=='local_A3_reg':row.update(obj);row.update(score=obj['p_gt_10K'],y_pred=int(obj['p_gt_10K']>=.5))
                    else:row.update(score=obj['p'],y_pred=int(obj['p']>=.5),abstain=obj['abstain'])
                pending.append(row)
                if len(pending)>=10:writer.writerows(pending);pending=[];pf.flush();os.fsync(pf.fileno());log(f'PROGRESS {done}/{len(tasks)} valid={valid} arm={arm}')
            save(results[0])
            if results[0][1] is None:raise RuntimeError('scientific probe failed both rounds; inspect operational diagnostics')
            with ThreadPoolExecutor(max_workers=4) as pool:
                for res in pool.map(lambda task:call_task(task,cfg),tasks[1:]):save(res)
            if pending:writer.writerows(pending);pf.flush();os.fsync(pf.fileno())
        cfg['status']='inference_completed';cfg['completed_utc']=now();cfg['n_completed']=done;cfg['n_valid']=valid;write_json(OUT/'config.json',cfg);log(f'INFERENCE_COMPLETED n={done} valid={valid}')
    except Exception:
        cfg['status']='stopped_error';cfg['traceback']=traceback.format_exc();write_json(OUT/'config.json',cfg);log(cfg['traceback']);raise
    finally:
        proc.terminate()
        try:proc.wait(timeout=20)
        except subprocess.TimeoutExpired:proc.kill()
        out.close();err.close();ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);args=p.parse_args()
    (prepare if args.mode=='prepare' else run)()
