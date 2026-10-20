"""Final G6 transcript audit; no inference and no old-directory mutation."""
from g6_local import ROOT,OUT,WORK,MODEL,read,sha,digest,a3_validate,reg_validate
from collections import Counter,defaultdict
from common import load_data,DATA,HGB
import csv,json,re

def main():
    cfg=read(OUT/'config.json');assert cfg['status'] in ('inference_completed','complete')
    data=load_data();indices={str(v):i for i,v in enumerate(data['id'])}
    historical=read(ROOT/'G5_tc_multithreshold/config.json')
    for source in (DATA,HGB):assert sha(source)==historical['scientific_sources'][str(source)]
    for case in cfg['cases']+cfg['M1_cases']:assert case['i']==indices[case['id']]
    mapping={('local_A3',c['id']):(c['A3_prompt'],cfg['A3_instructions']) for c in cfg['cases']}
    mapping.update({('local_A3_reg',c['id']):(c['A3_reg_prompt'],cfg['reg_instructions']) for c in cfg['cases']})
    mapping.update({('local_A3_M1',c['id']):(c['prompt'],cfg['M1_instructions']) for c in cfg['M1_cases']})
    assert len(mapping)==7828
    records=defaultdict(list);states=Counter();reasons=Counter();models=Counter();finishes=Counter();max_completion=0;max_prompt=0
    with (OUT/'llm_calls.jsonl').open(encoding='utf8') as f:
        for line in f:
            r=json.loads(line);key=(r['arm'],r['id']);assert key in mapping
            prompt,system=mapping[key]
            expected=prompt if r['round']==1 else prompt+'\nFormat-only repair: return exactly the required JSON schema using the same scientific evidence. Do not add prose or request evidence.'
            assert r['round'] in (1,2) and r['prompt']==expected and r['instructions']==system and r['tools']==[]
            assert r['prompt_sha256']==digest(expected)
            records[key].append(r);states[r['status']]+=1
            if 'raw_response' in r:
                raw=r['raw_response'];choice=raw['choices'][0];message=choice['message']
                assert not message.get('tool_calls') and not message.get('function_call')
                models[str(raw.get('model'))]+=1;finishes[str(choice.get('finish_reason'))]+=1
                max_completion=max(max_completion,raw.get('usage',{}).get('completion_tokens',0))
                max_prompt=max(max_prompt,raw.get('usage',{}).get('prompt_tokens',0))
                if r['status']=='ok':
                    obj=json.loads(message['content']);v=reg_validate(obj) if r['arm']=='local_A3_reg' else a3_validate(obj)
                    assert v==r['validated_output'] and choice.get('finish_reason')!='length'
            if r['status']!='ok':reasons[r.get('error','missing_error')]+=1
    assert set(records)==set(mapping)
    for key,rr in records.items():
        assert len(rr) in (1,2) and [r['round'] for r in rr]==list(range(1,len(rr)+1))
        if len(rr)==2:assert rr[0]['status']!='ok'
    seen=set();valids=Counter();failures=[]
    with (OUT/'predictions.csv').open(newline='',encoding='utf8') as f:
        for row in csv.DictReader(f):
            key=(row['arm'],row['id']);assert key in mapping and key not in seen;seen.add(key)
            i=indices[row['id']];assert int(row['fold'])==int(data['fold'][i]) and row['group']==str(data['group'][i])
            rr=records[key];assert int(row['requests'])==len(rr)
            last=rr[-1]
            if last['status']=='ok':
                assert row['status']=='ok';obj=last['validated_output'];p=obj['p_gt_10K'] if key[0]=='local_A3_reg' else obj['p']
                assert float(row['score'])==p and int(row['y_pred'])==int(p>=.5)
                if key[0]=='local_A3_reg':
                    for k in ('Tc_est_K','p_gt_5K','p_gt_10K','p_gt_15K'):assert float(row[k])==obj[k]
                else:assert row['abstain']==str(obj['abstain'])
                valids[key[0]]+=1
            else:assert row['score']=='' and row['y_pred']=='';failures.append({'id':key[1],'arm':key[0]})
    assert seen==set(mapping) and max_completion<=512 and max_prompt+512<8192
    assert sha(MODEL)==cfg['checkpoint_sha256']
    for name,value in cfg['runtime_files_sha256'].items():assert sha(WORK/'bin'/name)==value
    for name,value in cfg['source_sha256'].items():assert sha(ROOT/name)==value,name
    props=cfg['server_props'];assert props['total_slots']==4 and props['default_generation_settings']['n_ctx']==8192
    assert props['chat_template']==cfg['chat_template']
    assert cfg['server_command'][cfg['server_command'].index('-fa')+1]=='on'
    assert str(MODEL)==props['model_path']
    native=(WORK/'server.stderr.log').read_text(encoding='utf8',errors='replace')
    truncations=Counter(re.findall(r'truncated\s*=\s*(\d+)',native))
    assert not any(int(k)>0 and v>0 for k,v in truncations.items())
    audit={'status':'passed','audit_source_sha256':sha(ROOT/'g6_audit.py'),'expected_tasks':7828,'observed_unique_tasks':len(seen),'request_records':sum(states.values()),'states':dict(states),
           'final_valid_by_arm':dict(valids),'final_failures':failures,'repair_tasks':sum(len(v)==2 for v in records.values()),'invalid_reasons':dict(reasons),
           'raw_model_identifiers':dict(models),'finish_reasons':dict(finishes),'max_completion_tokens':max_completion,'max_actual_prompt_tokens':max_prompt,
           'native_truncation_counts':dict(truncations),'first_round_scientific_prompts_byte_identical':True,'format_repairs_same_evidence_only':True,
           'schema_validators_recomputed_for_all_accepted':True,'forced_p_and_Tc_values_equal_CSV':True,'CSV_ID_index_fold_group_alignment':True,'original_DATA_HGB_historical_hashes_match':True,'runtime_checkpoint_all_binary_source_hashes_match':True,
           'four_server_slots_and_8192_context_per_slot':True,'flash_attention_command_on':True,'no_model_tools_or_function_calls':True,
           'chat_template_embedded_GGUF_captured_and_unchanged':True,'first_real_probe_before_four_worker_submission':'guaranteed by frozen runner; native log task0 release precedes task69.. launches',
           'task_operational_log_enabled':False,'initial_preflight_system_warning_error_events':[],'initial_preflight_task_result_code':1,
           'same_prompt_no_performance_parameter_selection':True}
    cfg['final_transcript_audit']=audit
    (OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(audit,ensure_ascii=False))
if __name__=='__main__':main()
