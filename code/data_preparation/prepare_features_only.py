"""Root-authorized one pass TRAIN-only geometry X127; no labels or models."""
from pathlib import Path
from datetime import datetime,timezone
import gzip,hashlib,json,os
import numpy as np
import structural_features as sf

ROOT=Path(__file__).resolve().parent
EXPECTED=Path('C:/Users/28908/Codex/fung-research/2026-10-09-superconductivity-structure-review')
PLAN=Path('C:/Users/28908/Codex/fung-research/2026-10-09-structure-classification/SCIENTIFIC_PLAN.json')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def guard(path):
    path=Path(path)
    assert ROOT==EXPECTED and path.resolve()==path and path.is_relative_to(EXPECTED)
    for a in [path,*path.parents]:
        assert 'onedrive' not in str(a.resolve()).lower()
        if a.exists():
            assert not a.is_symlink() and not a.is_junction()
            assert not (getattr(a.stat(follow_symlinks=False),'st_file_attributes',0)&0x400)
    return path
def save(name,value):
    p=guard(ROOT/name)
    with p.open('x',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
    return {'path':str(p),'sha256':sha(p)}

source=ROOT/'TRAIN_ONLY_STRUCTURE_DESCRIPTORS.jsonl.gz'
identity=ROOT/'TRAIN_ONLY_STRUCTURE_IDENTITY.json'
assert sha(source)=='5a01dc66d8851334f5a1b1a330592c954c21b969bf65596a31cf00565878ea05'
assert sha(identity)=='07e78dcd02ddfaeb9b757b09bef9b2490a466aaf3939bf80a5f5743246942b1d'
assert sha(ROOT/'structural_features.py')=='f04015effecdc1509ef9ae87b252d3369a0c469e5e8938c0be850c008710fb42'
id_data=json.loads(identity.read_text(encoding='utf-8'))
assert id_data['target_values_included'] is False and id_data['record_count']==3764
ids=[r['material_id'] for r in id_data['records']]
assert len(ids)==len(set(ids))==3764
plan=json.loads(PLAN.read_text(encoding='utf-8'))
start=save('FEATURE_PREPARATION_ONCE.json',{'started_utc':datetime.now(timezone.utc).isoformat(),
    'scientific_plan':{'path':str(PLAN),'sha256':sha(PLAN)},'fixed_recipe':sf.RECIPE_VERSION,
    'target_labels_model_fits_model_queries_authorized':False,'geometry_record_budget':3764})
seen=[]; features=[]; witnesses=[]; current_id=None
try:
    with gzip.open(source,'rb') as f:
        for line in f:
            record=json.loads(line); current_id=record['material_id']; seen.append(current_id)
            assert current_id==ids[len(seen)-1]
            vector,witness=sf.feature_vector(record)
            assert vector.dtype==np.float64 and vector.shape==(127,) and np.isfinite(vector).all()
            features.append(vector); witnesses.append({'material_id':current_id,**witness})
    assert seen==ids
    X=np.vstack(features); assert X.shape==(3764,127)
    path=guard(ROOT/'FEATURES_ONLY.npz')
    with path.open('xb') as f:
        np.savez_compressed(f,X127=X,material_id=np.asarray(ids,dtype='U24'),feature_names=np.asarray(sf.ALL_COLUMNS))
        f.flush(); os.fsync(f.fileno())
    witness_binding=save('FEATURE_GEOMETRY_WITNESSES.json',{'records':witnesses,'target_labels_included':False})
    receipt=save('FEATURE_PREPARATION_RECEIPT.json',{
        'checks_passed':True,'completed_utc':datetime.now(timezone.utc).isoformat(),'once':start,
        'scientific_plan':{'path':str(PLAN),'sha256':sha(PLAN)},
        'inputs':{str(p):sha(p) for p in [source,identity,ROOT/'structural_features.py',ROOT/'verify_structural_features.py',ROOT/'STRUCTURAL_FEATURE_PURE_CONTRACTS_v2.json',Path(__file__)]},
        'feature_output':{'path':str(path),'sha256':sha(path)},'witnesses':witness_binding,
        'array_schema':{'X127':['float64',[3764,127]],'material_id':['str',[3764]],'feature_names':['str',[127]]},
        'feature_columns':list(sf.ALL_COLUMNS),'geometry_records_decoded_this_pass':3764,
        'sites_checked':sum(w['n_sites'] for w in witnesses),'PBC_edges_recomputed_this_pass':sum(w['cached_PBC_edges'] for w in witnesses),
        'every_positive_occupied_site_has_at_least_one_neighbor':True,
        'cutoff_A':sf.NEIGHBOR_CUTOFF_A,'cutoff_slack_A':sf.NEIGHBOR_BOUND_TOLERANCE_A,
        'native_cached_distances_are_never_used_as_feature_values':True,
        'max_neighbor_coordinate_cache_difference_A':max(w['neighbor_coordinate_cache_max_abs_difference_A'] for w in witnesses),
        'max_nearest_matrix_edge_crosscheck_difference_A':max(w['nearest_matrix_edge_crosscheck_max_abs_difference_A'] for w in witnesses),
        'max_volume_cache_difference_A3':max(w['volume_cache_abs_difference_A3'] for w in witnesses),
        'target_values_read':0,'old_validation_label_reads':0,'old_retrospective_label_reads':0,
        'old_validation_geometry_reads':0,'old_retrospective_geometry_reads':0,
        'model_fits':0,'model_queries':0,'feature_searches':0,'OneDrive_writes':0,
        'native_geometry_parse_rerun':False,'source_status':'Original matched computed/artificially doped MP structures, not proven experiment sample structures'})
    print(json.dumps({'checks_passed':True,'feature_receipt':receipt,'features':{'path':str(path),'sha256':sha(path)},
                      'feature_code_sha256':sha(ROOT/'structural_features.py'),'sites_checked':sum(w['n_sites'] for w in witnesses),
                      'edges_recomputed':sum(w['cached_PBC_edges'] for w in witnesses)},ensure_ascii=False))
except Exception as error:
    failure=save('FEATURE_PREPARATION_FAILURE.json',{'checks_passed':False,'failure_class':type(error).__name__,
                'failure_code':str(error) if isinstance(error,sf.StructureInputError) else None,
                'last_identity':current_id,'geometry_records_started':len(seen),'target_values_read':0,'model_fits':0,'model_queries':0,
                'automatic_retry':False,'once':start})
    print(json.dumps({'checks_passed':False,'failure':failure})); raise
