"""G7 same-checkpoint local A3; label-free scratch inputs, explicit root handoff."""
from common import ROOT
from g6_local import WORK,EXE,MODEL,PORT,SEED,api,read,sha,digest,a3_validate
from e2_prompts import SCHEMA
from g7_eval import material_prompt,validate_scientific_material
from concurrent.futures import ThreadPoolExecutor
import argparse,csv,ctypes,json,os,subprocess,time,traceback
OUT=ROOT/'work/G7_eval'
def run(authorized):
    assert authorized,'Requires root final-freeze authorization and G6 GPU handoff'
    g6=read(ROOT/'G6_local_cutoff/config.json');assert g6['status'] in ('inference_completed','complete')
    assert sha(MODEL)==g6['checkpoint_sha256'];assert all(sha(WORK/'bin'/p)==h for p,h in g6['runtime_files_sha256'].items())
    cfg=read(OUT/'config.json');assert cfg['root_final_freeze_authorized'] and cfg['n']>=60
    for path,h in cfg['source_hashes'].items():assert sha(path)==h,path
    assert not (OUT/'local_predictions.csv').exists(),'No silent continuation; diagnose any interruption first'
    system=(ROOT/'e2_instructions.md').read_text(encoding='utf8')
    cases=cfg['cases'];assert len({c['id'] for c in cases})==len(cases)
    for c in cases:
        validate_scientific_material(c['material'],bool('structure' in c['material']))
        assert digest(material_prompt(c['material']))==c['prompt_sha256']
    state={'status':'running','model':g6['model'],'checkpoint_sha256':g6['checkpoint_sha256'],'parameters':g6['parameters'],'root_final_freeze_authorized':True,
           'official_pretraining_cutoff_month':g6['official_cutoff_month'],'official_cutoff_source':g6['official_cutoff_source'],'cutoff_boundary':g6['cutoff_boundary'],
           'runtime_files_sha256':g6['runtime_files_sha256'],'G7_frozen_scientific_source_hashes':cfg['source_hashes'],'all_first_round_prompt_hashes_matched':len(cases),
           'label_files_opened':False,'scientific_cases_sha256':digest(json.dumps(cases,ensure_ascii=False,sort_keys=True,separators=(',',':'))),
           'helper_source_sha256':sha(ROOT/'g7_local.py'),'g6_config_sha256':sha(ROOT/'G6_local_cutoff/config.json'),'system_sha256':digest(system)}
    def save_state(): (OUT/'local_run.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8')
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);save_state()
    stdout=(WORK/'g7_server.stdout.log').open('w',encoding='utf8');stderr=(WORK/'g7_server.stderr.log').open('w',encoding='utf8')
    proc=subprocess.Popen(g6['server_command'],cwd=str(WORK/'bin'),stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
    def call(c):
        records=[];obj=None
        for rnd in (1,2):
            prompt=material_prompt(c['material'],repair=rnd==2)
            start=time.monotonic()
            rec={'id':c['id'],'arm':'Local_A3','round':rnd,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'prompt':prompt,'instructions':system,'prompt_sha256':digest(prompt),'tools':[],'checkpoint_sha256':g6['checkpoint_sha256']}
            try:
                raw=api('v1/chat/completions',{'model':'local','messages':[{'role':'system','content':system},{'role':'user','content':prompt}],
                    'temperature':0,'seed':SEED,'max_tokens':512,'stream':False,'cache_prompt':False,
                    'response_format':{'type':'json_schema','json_schema':{'name':'scientific_output','strict':True,'schema':SCHEMA}}})
                rec['raw_response']=raw;obj=a3_validate(json.loads(raw['choices'][0]['message']['content']))
                assert raw['choices'][0]['finish_reason']!='length','output exhaustion'
                rec.update(status='ok',validated_output=obj)
            except Exception as exc:rec.update(status='invalid',error=repr(exc));obj=None
            rec['elapsed_seconds']=time.monotonic()-start;records.append(rec)
            if obj is not None:break
        return c,obj,records
    try:
        for _ in range(120):
            if proc.poll() is not None:raise RuntimeError('server exited '+str(proc.returncode))
            try:
                props=api('props',timeout=2)
                if props.get('default_generation_settings') is not None:break
            except Exception:pass
            time.sleep(2)
        else:raise TimeoutError('server readiness')
        assert props['total_slots']==4 and props['default_generation_settings']['n_ctx']==8192
        assert props['chat_template']==g6['chat_template'];state['server_props']=props;save_state()
        max_bound=0
        syst=len(api('tokenize',{'content':system})['tokens'])
        for c in cases:
            bound=syst+len(api('tokenize',{'content':material_prompt(c['material'])})['tokens'])+160
            assert bound+512+100<8192,'inputbudget';max_bound=max(max_bound,bound)
        state['max_prompt_token_bound']=max_bound;pending=[];valid=0;done=0
        with (OUT/'local_predictions.csv').open('w',newline='',encoding='utf8') as pf,(OUT/'local_llm_calls.jsonl').open('w',encoding='utf8') as log:
            w=csv.DictWriter(pf,fieldnames=['id','arm','status','score','y_pred']);w.writeheader()
            def save(item):
                nonlocal done,valid,pending
                c,obj,recs=item;done+=1;valid+=obj is not None
                for rec in recs:log.write(json.dumps(rec,ensure_ascii=False,allow_nan=False)+'\n')
                log.flush();os.fsync(log.fileno())
                pending.append({'id':c['id'],'arm':'Local_A3','status':'valid' if obj else 'invalid_after_two_rounds','score':obj['p'] if obj else '', 'y_pred':int(obj['p']>=.5) if obj else ''})
                if len(pending)>=10:w.writerows(pending);pending=[];pf.flush();os.fsync(pf.fileno())
            first=call(cases[0]);save(first)
            assert first[1] is not None,'first real scientific probe failed'
            with ThreadPoolExecutor(max_workers=4) as pool:
                for result in pool.map(call,cases[1:]):save(result)
            if pending:w.writerows(pending);pf.flush();os.fsync(pf.fileno())
        # Revalidate all transcript/CSV pairs before handing scratch results to root.
        byid={c['id']:c for c in cases};completed={};max_completion=0;repairs=0
        with (OUT/'local_llm_calls.jsonl').open(encoding='utf8') as audit:
            for line in audit:
                rec=json.loads(line);assert rec['id'] in byid and rec['arm']=='Local_A3' and rec['tools']==[]
                expected=material_prompt(byid[rec['id']]['material'],repair=rec['round']==2)
                assert rec['prompt']==expected and rec['prompt_sha256']==digest(expected) and rec['instructions']==system
                repairs+=int(rec['round']==2)
                if rec['status']=='ok':
                    raw=rec['raw_response'];choice=raw['choices'][0];message=choice['message']
                    assert not message.get('tool_calls') and not message.get('function_call') and choice['finish_reason']!='length'
                    assert a3_validate(json.loads(message['content']))==rec['validated_output']
                    max_completion=max(max_completion,raw.get('usage',{}).get('completion_tokens',0));completed[rec['id']]=rec['validated_output']
        assert max_completion<=512
        with (OUT/'local_predictions.csv').open(newline='',encoding='utf8') as audit:
            rows=list(csv.DictReader(audit));assert len(rows)==len(cases) and len({r['id'] for r in rows})==len(cases)
            for row in rows:
                if row['status']=='valid':assert float(row['score'])==completed[row['id']]['p'] and int(row['y_pred'])==int(completed[row['id']]['p']>=.5)
        state.update(status='complete' if valid==len(cases) else 'partial',n_completed=done,n_valid=valid,
            final_audit={'status':'passed','prompt_system_schema_CSV_all_matched':True,'tool_calls':0,'max_completion_tokens':max_completion,'format_repair_requests':repairs,
                         'first_probe_before_four_workers':True,'server_slots':props['total_slots'],'context_per_slot':props['default_generation_settings']['n_ctx']})
        save_state()
    except Exception:state.update(status='stopped_error',traceback=traceback.format_exc());save_state();raise
    finally:
        proc.terminate()
        try:proc.wait(timeout=20)
        except subprocess.TimeoutExpired:proc.kill()
        stdout.close();stderr.close();ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root-final-freeze-authorized',action='store_true');a=p.parse_args();run(a.root_final_freeze_authorized)
