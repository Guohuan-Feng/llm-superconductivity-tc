"""G7 label-isolated GPT scientific evaluator and frozen composition HGB baseline.

Prepare only after root confirms candidate freeze. All outputs go to scratch
work/G7_eval; main audit directory is merged by the coordinating agent.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import argparse,csv,ctypes,hashlib,json,os,threading,time
from common import ROOT,DATA,load_data,np,write_json,safe_path
from g3_ml import PARAMS,weights
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from e2_prompts import make_prompt
from f3_inputs import validate_output
from f2_transport_v2 import call_once,resolve_runtime

OUT=ROOT/'work/G7_eval'; MAIN=ROOT/'G7_postcutoff_materials'
SCHEMA=ROOT/'e2_response.schema.json'; INST=ROOT/'e2_instructions.md'; CATALOG=ROOT/'e2_model_catalog.json'
EXE=r'C:\Users\28908\AppData\Local\OpenAI\Codex\bin\896a0166b774037a\codex.exe'
DELAYS=(60,120,240); MAX_CALLS=1000
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(s):return hashlib.sha256(s.encode('utf8')).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))

def material_prompt(material,repair=False):
    history=[] if not repair else [{'operational_error':'Previous attempt violated decision JSON contract. Return a forced decision with numeric p and exactly the requested fields.'}]
    return make_prompt('A3',material,[],{},[],history,round_number=2 if repair else 1,force_decision=False)

def validate_scientific_material(material,has_cif):
    """Only formula without CIF; exact scientific summary whitelist otherwise."""
    assert isinstance(material,dict) and isinstance(material.get('formula'),str) and material['formula']
    if not has_cif:assert set(material)=={'formula'};return
    assert set(material)<= {'formula','formula_semantics','composition','structure'}
    if 'formula_semantics' in material:assert material['formula_semantics'] in {'occupancy formula of the supplied proxy structure cell; not reduced','reported formula of the measured material'}
    if 'composition' in material:
        assert isinstance(material['composition'],list)
        for r in material['composition']:
            assert set(r)=={'element','cell_occupancy','atomic_fraction'}
            assert isinstance(r['element'],str) and r['element'].isalpha() and len(r['element'])<=2
            assert all(isinstance(r[k],(float,int)) and np.isfinite(r[k]) for k in ('cell_occupancy','atomic_fraction'))
    if 'structure' in material:
        s=material['structure'];assert isinstance(s,dict)
        assert set(s)<= {'coordinate_source_proxy','measured_specimen_match','space_group_declared_in_source_CIF','lattice_lengths_A','lattice_angles_alpha_beta_gamma_deg','cell_volume_A3','n_sites','existing_global_structure_features'}
        if 'coordinate_source_proxy' in s:assert isinstance(s['coordinate_source_proxy'],bool)
        if 'measured_specimen_match' in s:assert s['measured_specimen_match'] in {'unknown','matched','verified'}
        if 'space_group_declared_in_source_CIF' in s:
            sg=s['space_group_declared_in_source_CIF'];assert set(sg)=={'symbol','number'}
            assert sg['number'] is None or isinstance(sg['number'],int) and 1<=sg['number']<=230
            assert sg['symbol'] is None or isinstance(sg['symbol'],str) and len(sg['symbol'])<=20
        for key in ('lattice_lengths_A','lattice_angles_alpha_beta_gamma_deg'):
            if key in s:assert len(s[key])==3 and all(isinstance(v,(int,float)) and np.isfinite(v) for v in s[key])
        for key in ('cell_volume_A3','n_sites'):
            if key in s:assert isinstance(s[key],(int,float)) and np.isfinite(s[key]) and s[key]>0
        if 'existing_global_structure_features' in s:
            permitted={'volume_per_occupied_atom_A3','nearest_positive_PBC_distance_site_occ_mean_A','nearest_positive_PBC_distance_site_occ_std_A','nearest_positive_PBC_distance_min_A','nearest_positive_PBC_distance_max_A','coordination6A_site_occ_mean','coordination6A_site_occ_std','PBC_edge_distance_pair_occ_mean_A','PBC_edge_distance_pair_occ_std_A','cell_site_count','cell_positive_site_count','cell_total_absolute_occupancy','cell_volume_A3','cell_a_length_A','cell_b_length_A','cell_c_length_A','cell_alpha_cosine','cell_beta_cosine','cell_gamma_cosine','site_absolute_occupancy_unweighted_mean','site_absolute_occupancy_unweighted_std','occupied_mass_on_partial_sites_fraction'}
            fs=s['existing_global_structure_features'];assert isinstance(fs,dict) and set(fs)<=permitted
            assert all(v is None or isinstance(v,(int,float)) and np.isfinite(v) for v in fs.values())

def prepare(freeze_authorized=False):
    assert freeze_authorized,'Requires root-confirmed final candidate freeze, not just provisional accepted status.'
    assert not OUT.exists(),'Scratch exists; inspect before overwrite/recompute'
    for p in (ROOT,*ROOT.parents):assert not p.is_symlink() and not p.is_junction()
    with (MAIN/'predictions.csv').open(newline='',encoding='utf8') as h: rows=[r for r in csv.DictReader(h) if r['status']=='accepted']
    assert len(rows)>=60
    assert len({r['material_id'] for r in rows})==len(rows),'Duplicate G7 material IDs'
    composition_features([r['formula'] for r in rows])
    # The scientific source table intentionally omits labels and Tc. No label file is opened here.
    cases=[];main_config=read(MAIN/'config.json');struct_materials=main_config.get('scientific_materials',{})
    retrieved_manifest=main_config.get('retrieved_CIF_manifest',{})
    for r in rows:
        assert r['first_report_date']>='2024-02-01'
        identity=r['material_id']; material={'formula':r['formula']}
        # Label-free scientific packets live inside the finite main config, not an extra audit file.
        # A public landing/deposition URL is not proof that CIF bytes were retrieved.
        has_cif=identity in retrieved_manifest
        if has_cif:
            entry=retrieved_manifest[identity];assert set(entry)>={'path','sha256'}
            cp=Path(entry['path']).resolve();assert cp.is_file() and sha(cp)==entry['sha256'],f'Unverified CIF bytes {identity}'
            supplied=struct_materials.get(identity)
            assert supplied is not None,f'CIF declared for {identity} but frozen scientific summary absent; do not silently fall back to formula-only'
            material=supplied
        validate_scientific_material(material,has_cif)
        message=material_prompt(material)
        cases.append({'id':identity,'formula':r['formula'],'material':material,'prompt_sha256':digest(message)})
    os.environ['F1_CODEX_EXE']=EXE;runtime=resolve_runtime()
    from e2_transport import input_budget
    bounds=[input_budget(material_prompt(c['material']),schema_path=SCHEMA,instructions_path=INST)['input_token_upper_bound'] for c in cases]
    assert max(bounds)<=15000
    sourcepaths=[MAIN/'predictions.csv',MAIN/'config.json',ROOT/'g7_eval.py',ROOT/'g7_eval_launch.ps1',ROOT/'e2_prompts.py',ROOT/'f2_transport_v2.py',ROOT/'f1_transport.py',ROOT/'e2_transport.py',ROOT/'f3_inputs.py',ROOT/'g3_ml.py',ROOT/'common.py',SCHEMA,INST,CATALOG,DATA]
    sourcepaths += [Path(v['path']).resolve() for k,v in retrieved_manifest.items() if k in {c['id'] for c in cases}]
    sources={str(p):sha(p) for p in sourcepaths}
    cfg={'experiment':'G7_eval_scratch','status':'prepared','created_utc':now(),'cases':cases,'n':len(cases),'runtime':runtime,'source_hashes':sources,'concurrency':4,'max_calls':MAX_CALLS,'root_final_freeze_authorized':True,'labels_opened_by_prompt_preparation':False,'input_bound_max':max(bounds),'input_limit':15000}
    safe_path(OUT).mkdir(parents=True);write_json(OUT/'config.json',cfg)
    (OUT/'llm_calls.jsonl').write_text('',encoding='utf8');(OUT/'run.log').write_text('',encoding='utf8')
    with (OUT/'gpt_predictions.csv').open('w',newline='',encoding='utf8') as h:csv.DictWriter(h,fieldnames=['id','arm','status','score','y_pred']).writeheader()

def composition_features(formulas):
    from pymatgen.core import Composition,Element
    X=np.zeros((len(formulas),118),float)
    for i,f in enumerate(formulas):
        c=Composition(f);total=sum(c.values());assert total>0
        for e,a in c.items():X[i,Element(str(e)).Z-1]=float(a/total)
    assert np.allclose(X.sum(1),1.) and np.isfinite(X).all()
    return X

def ml_baseline():
    cfg=read(OUT/'config.json');d=load_data();X=d['X127'][:,:118]
    # Validate atomic-number columns against known frozen F3 material compositions before novel prediction.
    f3=read(ROOT/'F3_full3764/config.json');formulas=[f3['cases'][str(k)]['material']['formula'] for k in d['id']]
    expected=composition_features(formulas);assert np.allclose(expected,X,atol=2e-8,rtol=2e-8),'Composition118 column mismatch'
    Q=composition_features([c['formula'] for c in cfg['cases']]);ws,wc=weights(d['group'],d['y'])
    oof=np.full(len(d['y']),np.nan);fold_meta=[]
    for outer in range(3):
        train=np.flatnonzero(d['fold']!=outer);test=np.flatnonzero(d['fold']==outer)
        assert not set(d['group'][train])&set(d['group'][test])
        wt,wc_fold=weights(d['group'][train],d['y'][train])
        with threadpool_limits(limits=4):
            sf=StandardScaler();Xt=sf.fit_transform(X[train],sample_weight=wt);Xh=sf.transform(X[test]);mf=HistGradientBoostingClassifier(**PARAMS);mf.fit(Xt,d['y'][train],sample_weight=wc_fold);oof[test]=mf.predict_proba(Xh)[:,1]
        fold_meta.append({'outer_fold':outer,'n_train':len(train),'n_held':len(test),'group_overlap':0,'no_tuning':True})
    assert np.isfinite(oof).all()
    with threadpool_limits(limits=4):
        sc=StandardScaler();Xt=sc.fit_transform(X,sample_weight=ws);Xq=sc.transform(Q)
        model=HistGradientBoostingClassifier(**PARAMS);model.fit(Xt,d['y'],sample_weight=wc);p=model.predict_proba(Xq)[:,1]
    meta={'train_n':3764,'train_groups':1117,'n_features':118,'fixed_HGB_params':PARAMS,'weights':'1/group weighted scaler; group/class balanced HGB loss','no_tuning':True,'evaluation_labels_used':False,'composition118_F3_formula_reproduction_max_abs_diff':float(np.max(np.abs(expected-X))),'train_data_sha256':sha(DATA),'n_iter':int(model.n_iter_),'original_three_fold_OOF':fold_meta}
    write_json(OUT/'HGB_composition_training.json',meta)
    with (OUT/'ml_predictions.csv').open('w',newline='',encoding='utf8') as h:
        w=csv.DictWriter(h,fieldnames=['id','arm','status','score','y_pred']);w.writeheader()
        for c,v in zip(cfg['cases'],p):w.writerow({'id':c['id'],'arm':'HGB_composition118','status':'valid','score':float(v),'y_pred':int(v>=.5)})
    with (OUT/'original3764_HGB_composition_OOF.csv').open('w',newline='',encoding='utf8') as h:
        w=csv.DictWriter(h,fieldnames=['id','fold','group','score','y_pred']);w.writeheader()
        for i,v in enumerate(oof):w.writerow({'id':str(d['id'][i]),'fold':int(d['fold'][i]),'group':str(d['group'][i]),'score':float(v),'y_pred':int(v>=.5)})

def run():
    cfg=read(OUT/'config.json');assert cfg['status']=='prepared'
    for path,h in cfg['source_hashes'].items():assert sha(path)==h,path
    os.environ['F1_CODEX_EXE']=cfg['runtime']['executable'];assert resolve_runtime()==cfg['runtime']
    stop=threading.Event();lock=threading.RLock();reserved=0;results={};pending=[];stop_info={};failed=[]
    def log(msg):
        with lock,(OUT/'run.log').open('a',encoding='utf8') as h:h.write(now()+' '+msg+'\n')
    def append(r):
        with lock,(OUT/'llm_calls.jsonl').open('a',encoding='utf8') as h:h.write(json.dumps(r,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n');h.flush();os.fsync(h.fileno())
    def worker(c):
        nonlocal reserved
        response=None;last=None;retries=0
        for rnd in (1,2):
            msg=material_prompt(c['material'],rnd==2)
            if rnd==1:assert digest(msg)==c['prompt_sha256']
            while not stop.is_set():
                with lock:
                    if reserved>=MAX_CALLS:stop_info['kind']='call_budget';stop.set();break
                    reserved+=1;meta={'id':c['id'],'arm':'GPT_A3','call_id':reserved,'scientific_round':rnd,'transport_retry':retries,'started_utc':now(),'prompt':msg,'request_hash':digest(msg),'availability_probe':reserved==1};append({'event':'call_started',**meta})
                r=call_once(msg,schema_path=SCHEMA,instructions_path=INST,catalog_path=CATALOG,stop_event=stop,metadata=meta)
                if r['status']=='ok':
                    try:response=validate_output(json.loads(r['response_text']),{'id':c['id'],'rule_cards':{}},'A3')
                    except Exception as exc:r.update(status='protocol_error',protocol_error=str(exc))
                r.update(event='call_completed',finished_utc=now(),response_valid=response is not None,parsed_response=response);append(r);last=r
                if r['status']=='terminal_usage_limit':stop_info.update(kind='explicit_usage_limit',recovery_time_message=r.get('recovery_time_message'),diagnostics=r.get('terminal_diagnostics'));stop.set()
                elif r['status'] in {'tool_activity_rejected','model_mismatch_rejected','backend_protocol_rejected','preflight_rejected','actual_input_budget_violation'}:stop_info.update(kind='integrity',status=r['status']);stop.set()
                if response is not None or stop.is_set():break
                if r['status']=='rate_limited_final' and retries<3:
                    delay=DELAYS[retries];retries+=1;log(f'FINAL429 {c["id"]} wait{delay}s retry{retries}');
                    if stop.wait(delay):break
                    continue
                break
            if response is not None or stop.is_set() or last is None or last['status']=='rate_limited_final':break
            if last['status'] not in {'protocol_error','cli_error','backend_error','incomplete_response','transport_error'}:break
        return c,response,last['status'] if last else 'not_dispatched'
    def flush():
        if pending:
            with (OUT/'gpt_predictions.csv').open('a',newline='',encoding='utf8') as h:csv.DictWriter(h,fieldnames=['id','arm','status','score','y_pred']).writerows(pending);h.flush();os.fsync(h.fileno())
            pending.clear()
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        cfg['status']='running';write_json(OUT/'config.json',cfg);it=iter(sorted(cfg['cases'],key=lambda c:digest('G7|'+c['id'])));probe=False;doneall=False
        with ThreadPoolExecutor(max_workers=4) as pool:
            active={pool.submit(worker,next(it)):None}
            while active:
                ready,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
                for f in ready:
                    active.pop(f);c,q,status=f.result();results[c['id']]=q
                    if q is not None:probe=True;failed.clear();p=q['p'];pending.append({'id':c['id'],'arm':'GPT_A3','status':'valid','score':p,'y_pred':int(p>=.5)})
                    else:
                        pending.append({'id':c['id'],'arm':'GPT_A3','status':status,'score':'','y_pred':''})
                        if c['id'] not in failed:failed.append(c['id'])
                        if len(failed)>=3:stop_info.update(kind='three_consecutive_exhausted',ids=failed[-3:]);stop.set()
                    if len(pending)>=10:flush()
                    log(f'PROGRESS {len(results)}/{len(cfg["cases"])} {c["id"]} {status}')
                while not stop.is_set() and not doneall and len(active)<(4 if probe else 1):
                    try:active[pool.submit(worker,next(it))]=None
                    except StopIteration:doneall=True
        flush();cfg['results']=results;cfg['status']='complete' if all(v is not None for v in results.values()) and len(results)==len(cfg['cases']) else 'stopped_partial';cfg.update(stop_reason=stop_info,completed_utc=now(),calls=reserved);write_json(OUT/'config.json',cfg);log('END '+cfg['status'])
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','ml','run'));p.add_argument('--root-final-freeze-authorized',action='store_true');a=p.parse_args()
    if a.action=='prepare':prepare(a.root_final_freeze_authorized)
    elif a.action=='ml':ml_baseline()
    else:run()
