"""Read-only strict audit of completed G5b scientific calls."""
import hashlib,json
from collections import Counter
from datetime import datetime
from g5b_repeat import ROOT,OUT,F3,G5,G4,read,a3_prompt,sha
from g5_inputs import prompt as reg_prompt

cfg=read(OUT/'config.json'); f3=read(F3/'config.json'); g5=read(G5/'config.json'); g4=read(G4/'config.json')
assert len(cfg['cases'])==800 and cfg['selected_indices']==g4['selected_indices']
case={c['id']:c for c in cfg['cases']}; reg_cases={c['id']:c for c in g5['cases']}
for c in cfg['cases']:
 assert sha(a3_prompt(c['material']))==f3['cases'][c['id']]['prompt_hashes']['A3']==c['A3_prompt_sha256']
 assert sha(reg_prompt(c['material']))==reg_cases[c['id']]['prompt_sha256']==c['A3_reg_prompt_sha256']
started={}; completed={}; events=[]
with (OUT/'llm_calls.jsonl').open(encoding='utf8') as h:
 for line in h:
  r=json.loads(line); cid=r['call_id']
  if r['event']=='call_started': started[cid]=r
  if r['event']=='call_completed': completed[cid]=r
assert len(started)==len(completed)==2425
valid=[]
for cid,r in completed.items():
 c=case[r['id']]; p=r['prompt']; assert sha(p)==r['request_hash']
 if r['scientific_round']==1:
  expected=reg_prompt(c['material']) if r['arm']=='A3_reg' else a3_prompt(c['material'])
  assert p==expected
 assert r['model']=='gpt-6.1-sol' and r['reasoning_effort']=='high' and not r['tool_activity']
 assert r['runtime']==cfg['runtime']
 if r['status']=='ok':
  valid.append(r)
  q=r['parsed_response']; stored=cfg['results'][r['id']+'|'+r['arm']]; assert q==stored
assert len(valid)==2400
for c in cfg['cases']:
 for arm in ('A3_repeat_1','A3_repeat_2','A3_reg'): assert sum(r['id']==c['id'] and r['arm']==arm for r in valid)==1
def tm(s):return datetime.fromisoformat(s).timestamp()
for cid,r in started.items():
 events.append((tm(r['started_utc']),1,cid)); events.append((tm(completed[cid]['finished_utc']),-1,cid))
n=peak=0
for t,delta,cid in sorted(events): n+=delta;peak=max(n,peak)
assert peak<=4
assert min(tm(r['started_utc']) for cid,r in started.items() if cid!=1)>=tm(completed[1]['finished_utc'])
out={'status':'pass','initial_exact_prompt_bytes':sum(r['scientific_round']==1 for r in completed.values()),'all800_A3_and_reg_hashes_match_original':True,'valid_arm_cases':2400,'completed_calls':2425,'tools':0,'max_concurrency':peak,'probe_completed_before_call2':True,'runtime':cfg['runtime'],'all_runtime_records_equal_config':True,'status_counts':dict(Counter(r['status'] for r in completed.values()))}
print(json.dumps(out,ensure_ascii=False))
