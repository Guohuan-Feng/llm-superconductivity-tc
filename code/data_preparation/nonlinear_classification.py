"""One fixed nonlinear, balanced TRAIN classifier. Root freezes describe() once.

Only run reads real NPZ payloads, fits, or calls predict_proba. Root-controlled
SCIENTIFIC_PLAN binds code, sources, runtime, parameters, and the compact schema.
Manual numeric tree walks use saved leaf values already including shrinkage.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import uuid
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble._hist_gradient_boosting.common import PREDICTOR_RECORD_DTYPE
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent
EXPECTED_ROOT=Path('C:/Users/28908/Codex/fung-research/2026-10-10-structure-nonlinear-classification')
ORIGINAL=Path('C:/Users/28908/Codex/fung-research/2026-10-09-structure-classification')
BALANCED=Path('C:/Users/28908/Codex/fung-research/2026-10-09-structure-balanced-classification')
for directory in (ORIGINAL,BALANCED):
    if str(directory) not in sys.path:sys.path.insert(0,str(directory))
import balanced_math as bm
import structure_classification as common
RUN=ROOT/'run';INPUT=ORIGINAL/'inputs/accepted_train.npz'
CODE_FILES=('nonlinear_classification.py','verify_nonlinear.py')
PARAMS={'learning_rate':.05,'max_iter':150,'max_leaf_nodes':7,'min_samples_leaf':20,
    'l2_regularization':1.,'max_bins':64,'early_stopping':False,'random_state':20261010,
    'class_weight':None,'categorical_features':None,'loss':'log_loss','max_features':1.,
    'max_depth':None,'validation_fraction':None,'warm_start':False}
PINS={str(INPUT):'d34ac344c62067f15ccdd48a97f611d7ed6e14763a11ec6e31a7076df0b39fa2',
    str(ORIGINAL/'PREFLIGHT.json'):'704cdea0c3124846023bd7e91546c4b462f5dfb28e07bca3ab59bb05ff6af6ea',
    str(BALANCED/'run/summary.json'):'50b1bf2fcd964425869f01f08047b2203420aa77d658817c4930399fb6825446',
    str(BALANCED/'run/oof_predictions.npz'):'2bd01a9245a7d8ded01f4d391e3abb0db9bf469bda231ebdcc986562a7122acd',
    str(BALANCED/'audit_balanced_classification.json'):'7af8d3d7525ed87665d2c9e04abda7370eb6887ad52c8ed56fb2e04c835c71c1',
    str(BALANCED/'CURRENT_STATE.json'):'4b541b1511a48fc6542c1e07fcae180bfccd9dc50ffa387f6687835337065043',
    str(BALANCED/'balanced_math.py'):'18d58c7849c64c1a44c4a132b73b4bf3a620a21731fa6397866f7bfbede87cea',
    str(ORIGINAL/'structure_classification_math.py'):'f1f21553b1762d93ffe0afd196c7a2e25b102f45fabcbd3d592de464d7dad855',
    str(ORIGINAL/'structure_classification.py'):'b6de6c83d21a0c255bd3ab10ba413f61b3f53aac981adbd27ed83108a8c7b060'}
COUNTS_BEFORE={'total_successful_fits':989,'development_fits':975,'original_development_fits':957,
    'source_record_development_fits':18,'deployment_fits':14,'development_scored_records':382,'source_record_experiment_records':5}
require=bm.require

def checked(path,output=False):
    p=Path(path);require(p.is_absolute() and p.resolve(strict=p.exists())==p,'Unresolved path '+str(p))
    for q in (p,*p.parents):
        require(not q.is_symlink() and not q.is_junction(),'Linked ancestor '+str(q))
        if output:require('onedrive' not in str(q).casefold(),'OneDrive output forbidden')
    if output:require(ROOT==EXPECTED_ROOT and p.is_relative_to(ROOT),'Wrong output root')
    return p

def sha(path):
    h=hashlib.sha256()
    with checked(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def utc():return datetime.now(timezone.utc).isoformat()
def load(path):return json.loads(checked(path).read_text(encoding='utf-8-sig'))

def save(path,value):
    with checked(path,True).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def save_npz(path,**arrays):
    with checked(path,True).open('xb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())

def mkdir(path):checked(path,True).mkdir()
def code_binding():return {str(ROOT/n):sha(ROOT/n) for n in CODE_FILES}

def runtime():
    info=common.runtime();hashes=dict(info['library_code_sha256'])
    base='sklearn.ensemble._hist_gradient_boosting.'
    for name in ('gradient_boosting','grower','predictor','common','binning','_binning','_predictor',
                 '_gradient_boosting','splitting','histogram'):
        module=importlib.import_module(base+name);path=Path(module.__file__).resolve();hashes[str(path)]=sha(path)
    native_dir=Path(importlib.import_module(base+'predictor').__file__).resolve().parent
    for name in ('_predictor.pyx','common.pyx'):
        path=native_dir/name;hashes[str(path)]=sha(path)
    info['library_code_sha256']=hashes
    return info

def new_pipeline():return Pipeline([('scaler',StandardScaler(**common.SCALER_PARAMS)),
                                    ('classifier',HistGradientBoostingClassifier(**PARAMS))])

def source_binding():
    for path,h in PINS.items():require(sha(path)==h,'Pinned source changed '+path)
    state=load(BALANCED/'CURRENT_STATE.json');audit=load(BALANCED/'audit_balanced_classification.json')
    require(state['accepted_counts']==audit['proposed_counts_after']==COUNTS_BEFORE
        and audit['checks_passed'] is True and audit['final_completion_verified'] is True
        and audit['classifier_fits_verified']==3 and audit.get('errors',[])==[],'Prior accepted evidence differs')
    return dict(PINS)

def describe():
    return {'code_sha256':code_binding(),'input_sha256':source_binding(),'runtime':runtime(),
        'parameters':new_pipeline().named_steps['classifier'].get_params(),
        'scaler_parameters':StandardScaler(**common.SCALER_PARAMS).get_params(),
        'accepted_counts_before':COUNTS_BEFORE,'temperature_threshold_K':10.,'probability_cutoff':.5,
        'planned_classifier_fits':3,'planned_predict_proba':3,'auditor_native_queries':0,
        'minimum_leaf_samples_means_raw_rows_not_groups':True,'early_stopping_validation_subset':False,
        'classbalanced_weights_affect_binning_and_objective_initial_prediction_gradients':True,
        'standardization_keeps_same_preprocessing_not_added_tree_expressiveness':True,
        'adaptive_same_TRAIN_development_population':True,'fresh_blind_validation':False,
        'score_semantics':'class-balanced classifier score, not natural-prevalence calibrated probability',
        'selection_rule':'advance only if weightedBA and weightedprecision both >=balancedLR, at least one strictly greater',
        'tolerance':{'atol':2e-10,'rtol':2e-12},
        'schema':{'prefit':'run/prefit_arrays.npz fold{f}_train_index/held_index/scaler_train_weight/classifier_train_weight',
            'models':'run/models/fold{f}_{pipeline.joblib,witness.npz,support.json}',
            'witness_keys':['train_index','held_index','scaler_train_weight','classifier_train_weight','original_Tc_train_K',
                'original_Tc_held_K','labels_train','labels_held','X_train','X_held','mean','var','scale','n_samples_seen',
                'hgb_baseline','classes','n_iter','manual_raw_score','probability','prediction','baseline_probability',
                'null_probability','temperature_threshold_K'],
            'tree_keys':'tree_nodes_000 through tree_nodes_149; one binary tree per iteration',
            'node_dtype_descr':PREDICTOR_RECORD_DTYPE.descr,'hgb_baseline_shape':[1,1],
            'tree_value_contains_learning_rate':True,'categorical_nodes_forbidden':True,
            'OOF':'run/oof_predictions.npz','OOF_keys':['X127','original_Tc_K','label','component_id','fold','material_id',
                'source_index','role','probability','prediction','baseline_probability','null_probability'],
            'OOF_vector_shape':'N; one candidate only','trace':'run/execution_trace.jsonl','summary':'run/summary.json'}}

def validate_nodes(nodes,n_features):
    require(nodes.dtype==PREDICTOR_RECORD_DTYPE and nodes.ndim==1 and 1<=len(nodes)<=13
        and not nodes['is_categorical'].any(),'Numeric HGB nodes / size / dtype')
    require(int(nodes['is_leaf'].sum())<=7,'Maximum seven leaves')
    seen=set();pending=[0]
    while pending:
        i=pending.pop();require(0<=i<len(nodes) and i not in seen,'Cycle / duplicate or invalid node pointer');seen.add(i)
        node=nodes[i]
        if node['is_leaf']:require(np.isfinite(node['value']),'Finite leaf value')
        else:
            require(0<=int(node['feature_idx'])<n_features and np.isfinite(node['num_threshold']),'Finite numeric split')
            pending.extend([int(node['left']),int(node['right'])])
    require(len(seen)==len(nodes),'Unreachable node')

def tree_values(nodes,X):
    require(X.ndim==2 and np.isfinite(X).all(),'Finite standardized numeric X')
    validate_nodes(nodes,X.shape[1]);values=np.empty(len(X))
    for row,x in enumerate(X):
        i=0
        while not nodes[i]['is_leaf']:
            node=nodes[i];i=int(node['left'] if x[int(node['feature_idx'])]<=node['num_threshold'] else node['right'])
        values[row]=nodes[i]['value']
    return values

def sigmoid(raw):
    raw=np.asarray(raw,dtype=np.float64);require(np.isfinite(raw).all(),'Finite raw scores')
    out=np.empty(raw.shape);positive=raw>=0;out[positive]=1/(1+np.exp(-raw[positive]))
    z=np.exp(raw[~positive]);out[~positive]=z/(1+z);return out

def manual_scores(baseline,trees,X):
    require(np.asarray(baseline).shape==(1,1) and np.isfinite(baseline).all(),'Binary HGB baseline')
    raw=np.full(len(X),float(np.asarray(baseline)[0,0]))
    for nodes in trees:raw+=tree_values(nodes,X)
    return raw,sigmoid(raw)

def integrity():
    plan=load(ROOT/'SCIENTIFIC_PLAN.json')
    # JSON serializes dtype descriptors as lists; normalize the pure description.
    binding=json.loads(json.dumps(describe()))
    require(plan.get('root_frozen') is True and plan.get('authorize_actual_run') is True
        and plan.get('producer_binding')==binding and plan.get('planned_classifier_fits')==plan.get('planned_predict_proba')==3,
        'Root scientific plan is not frozen or code/source/runtime binding changed')
    return plan

def trace(event,**fields):
    with checked(RUN/'execution_trace.jsonl',True).open('a',encoding='utf-8') as f:
        f.write(json.dumps({'utc':utc(),'event':event,**fields},allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

@threadpool_limits.wrap(limits=1)
def run():
    plan=integrity();require(not RUN.exists(),'No retry of a prior real run');mkdir(RUN)
    started=returned=durable=query_started=query_returned=containers=target_rows=old_values=0
    try:
        owner={'pid':os.getpid(),'utc':utc(),'attempt_id':str(uuid.uuid4())}
        require(datetime.fromisoformat(plan['created_utc'])<=datetime.fromisoformat(owner['utc']),'Plan postdates execution')
        save(RUN/'protocol.json',plan);save(RUN/'execution_owner.json',{**owner,'protocol_sha256':sha(RUN/'protocol.json')})
        containers+=1
        with np.load(checked(INPUT),allow_pickle=False) as z:d={k:z[k] for k in common.INPUT_KEYS}
        target_rows+=len(d['target_K']);tau=plan['producer_binding']['temperature_threshold_K']
        cohort=bm.cohort_contract(d,load(ORIGINAL/'PREFLIGHT.json'),tau)
        X,raw,y,g,f=d['X127'],d['target_K'],d['label'],np.asarray(d['component_id'],dtype=str),d['fold']
        containers+=1
        with np.load(checked(BALANCED/'run/oof_predictions.npz'),allow_pickle=False) as z:
            oldX=z['X127'];oldraw=z['original_Tc_K'];target_rows+=len(oldraw)
            require(oldX.dtype==X.dtype and oldX.shape==X.shape and oldX.tobytes()==X.tobytes()
                and oldraw.dtype==raw.dtype and oldraw.tobytes()==raw.tobytes(),'Cached input feature / target mismatch')
            for key in ('label','component_id','fold','material_id','source_index','role'):
                a=z[key];require(a.dtype==d[key].dtype and a.shape==d[key].shape and a.tobytes()==d[key].tobytes(),'Cached identity '+key)
            baseline=z['probability'].copy();null=z['null_probability'].copy();old_values=baseline.size+null.size
        require(baseline.shape==null.shape==y.shape and np.isfinite(baseline).all() and np.isfinite(null).all()
            and ((baseline>=0)&(baseline<=1)&(null>=0)&(null<=1)).all(),'Cached balanced scores')
        scopes=bm.all_support(y,g,f);plans=[];arrays={};proofs=[]
        for fold in range(3):
            train,held=np.flatnonzero(f!=fold),np.flatnonzero(f==fold)
            ws,wc,proof=bm.training_weight_vectors(g[train],y[train]);plans.append((train,held,ws,wc));proofs.append(proof)
            for key,val in (('train_index',train),('held_index',held),('scaler_train_weight',ws),('classifier_train_weight',wc)):
                arrays[f'fold{fold}_{key}']=val
        save_npz(RUN/'prefit_arrays.npz',**arrays)
        save(RUN/'prefit_support.json',{'checks_passed':True,'cohort':cohort,'support':scopes,'weight_proofs':proofs,
            'all_three_scopes_and_weights_before_first_fit':True,'prefit_arrays_sha256':sha(RUN/'prefit_arrays.npz'),
            'cached_baseline_aligned_bitwise':True})
        probabilities=np.full(len(y),np.nan);predictions=np.full(len(y),-1,dtype=np.int64);artifacts={};mkdir(RUN/'models')
        for fold in range(3):
            train,held,ws,wc=plans[fold];model=new_pipeline();started+=1
            trace('fit_started',fold=fold,classifier_fits_started=started,prefit_arrays_sha256=sha(RUN/'prefit_arrays.npz'))
            model.fit(X[train],y[train],scaler__sample_weight=ws,classifier__sample_weight=wc)
            returned+=1;trace('fit_returned',fold=fold,classifier_fits_returned=returned)
            scaler,clf=model.named_steps['scaler'],model.named_steps['classifier']
            require(np.array_equal(clf.classes_,np.asarray([0,1])) and clf.n_iter_==150
                and len(clf._predictors)==150 and all(len(stage)==1 for stage in clf._predictors)
                and not clf.do_early_stopping_ and (clf.is_categorical_ is None or not clf.is_categorical_.any()),'HGB fitted schema differs')
            query_started+=1;trace('predict_proba_started',fold=fold,native_calls_started=query_started)
            pp=model.predict_proba(X[held]);query_returned+=1
            trace('predict_proba_returned',fold=fold,native_calls_returned=query_returned)
            require(pp.shape==(len(held),2) and np.isfinite(pp).all() and ((pp>=0)&(pp<=1)).all()
                and np.allclose(pp.sum(axis=1),1,atol=2e-12,rtol=2e-12),'Native probability schema')
            trees=[stage[0].nodes.copy() for stage in clf._predictors]
            scaled=(X[held]-scaler.mean_)/scaler.scale_
            manual_raw,manual_probability=manual_scores(clf._baseline_prediction,trees,scaled)
            prob=pp[:,1].copy();pred=(prob>=.5).astype(np.int64)
            require(np.allclose(prob,manual_probability,atol=2e-10,rtol=2e-12),'Numeric HGB walk differs from sole native call')
            stem=RUN/'models'/f'fold{fold}';mp,wp,sp=[Path(str(stem)+suffix) for suffix in ('_pipeline.joblib','_witness.npz','_support.json')]
            with checked(mp,True).open('xb') as fh:joblib.dump(model,fh,compress=3);fh.flush();os.fsync(fh.fileno())
            save_npz(wp,train_index=train,held_index=held,scaler_train_weight=ws,classifier_train_weight=wc,
                original_Tc_train_K=raw[train],original_Tc_held_K=raw[held],labels_train=y[train],labels_held=y[held],
                X_train=X[train],X_held=X[held],mean=scaler.mean_,var=scaler.var_,scale=scaler.scale_,
                n_samples_seen=np.asarray(scaler.n_samples_seen_),hgb_baseline=clf._baseline_prediction,classes=clf.classes_,
                n_iter=np.asarray(clf.n_iter_),manual_raw_score=manual_raw,probability=prob,prediction=pred,
                baseline_probability=baseline[held],null_probability=null[held],temperature_threshold_K=np.asarray(tau),
                **{f'tree_nodes_{j:03d}':nodes for j,nodes in enumerate(trees)})
            save(sp,{'fold':fold,'support':scopes[fold],'weight_proof':proofs[fold],'classifier_get_params':clf.get_params(),
                'scaler_get_params':scaler.get_params(),'model_sha256':sha(mp),'witness_sha256':sha(wp),
                'trees':150,'max_leaves':7,'manual_tree_walks':150,'native_predict_proba_calls':1})
            for path in (mp,wp,sp):artifacts[str(path)]=sha(path)
            durable+=1;trace('fit_complete',fold=fold,classifier_fits_durable=durable,model_sha256=sha(mp),witness_sha256=sha(wp))
            probabilities[held],predictions[held]=prob,pred
        require(started==returned==durable==query_started==query_returned==3 and np.isfinite(probabilities).all(),'OOF fit/query coverage')
        save_npz(RUN/'oof_predictions.npz',X127=X,original_Tc_K=raw,label=y,
            **{key:d[key] for key in ('component_id','fold','material_id','source_index','role')},
            probability=probabilities,prediction=predictions,baseline_probability=baseline,null_probability=null)
        trace('oof_committed_before_metrics',oof_sha256=sha(RUN/'oof_predictions.npz'))
        newscore,oldscore,nullscore=[bm.metrics(y,score,g) for score in (probabilities,baseline,null)];foldscores=[]
        for fold in range(3):
            mask=f==fold;foldscores.append({'fold':fold,'classifier':bm.metrics(y[mask],probabilities[mask],g[mask]),
                'baseline':bm.metrics(y[mask],baseline[mask],g[mask]),'null':bm.metrics(y[mask],null[mask],g[mask])})
        accepted=load(BALANCED/'run/summary.json')['classifier']
        for scope in ('group_weighted','row'):
            for key in ('balanced_accuracy','accuracy','precision','recall','f1','roc_auc','average_precision'):
                require(np.isclose(oldscore[scope][key],accepted[scope][key],atol=2e-10,rtol=2e-12),'Cached LR metric differs')
        n,o=newscore['group_weighted'],oldscore['group_weighted'];delta=n['balanced_accuracy']-o['balanced_accuracy']
        advance=n['balanced_accuracy']>=o['balanced_accuracy'] and n['precision']>=o['precision'] and (
            n['balanced_accuracy']>o['balanced_accuracy'] or n['precision']>o['precision'])
        integrity();artifacts[str(RUN/'oof_predictions.npz')]=sha(RUN/'oof_predictions.npz')
        save(RUN/'summary.json',{'checks_passed':True,'completed':True,'independent_audit_pending':True,
            'producer_binding':plan['producer_binding'],'cohort':cohort,'classifier':newscore,'baseline':oldscore,'null':nullscore,
            'folds':foldscores,'primary_groupweighted_BA_HGB_minus_balancedLR':delta,
            'development_candidate_advances':advance,'precision_difference':n['precision']-o['precision'],
            'successful_classifier_fits':durable,'classifier_fits_started':started,'classifier_fits_returned':returned,
            'producer_predict_proba_started':query_started,'producer_predict_proba_returned':query_returned,
            'manual_tree_walks':450,'regressor_fits':0,'old_models_loaded':0,'old_model_queries':0,'auditor_native_query_budget':0,
            'null_model_fits':0,'new_configuration_scores':1,'accepted_counts_before':COUNTS_BEFORE,
            'payload_NPZ_containers_read':containers,'original_Tc_target_rows_read':target_rows,'old_prediction_values_decoded':old_values,
            'artifact_sha256':artifacts,'oof_sha256':sha(RUN/'oof_predictions.npz'),'trace_sha256':sha(RUN/'execution_trace.jsonl'),
            'prefit_arrays_sha256':sha(RUN/'prefit_arrays.npz'),'prefit_support_sha256':sha(RUN/'prefit_support.json'),
            'protocol_sha256':sha(RUN/'protocol.json'),'scientific_plan_sha256':sha(ROOT/'SCIENTIFIC_PLAN.json'),
            'adaptive_same_TRAIN_development_cohort':True,'fresh_blind_validation':False,'automatic_promotion':False})
        return {'checks_passed':True,'successful_classifier_fits':durable,'predict_proba_calls':query_returned,
            'summary_sha256':sha(RUN/'summary.json')}
    except Exception as exc:
        if not (RUN/'failure.json').exists():save(RUN/'failure.json',{'checks_passed':False,'error':type(exc).__name__+': '+str(exc),
            'classifier_fits_started':started,'classifier_fits_returned':returned,'classifier_fits_durable':durable,
            'predict_proba_calls_started':query_started,'predict_proba_calls_returned':query_returned,
            'payload_NPZ_containers_read':containers,'original_Tc_target_rows_read':target_rows,
            'old_prediction_values_decoded':old_values,'retry_authorized':False})
        raise

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('describe','run'));args=parser.parse_args()
    print(json.dumps(describe() if args.stage=='describe' else run(),ensure_ascii=False,allow_nan=False))

if __name__=='__main__':main()
