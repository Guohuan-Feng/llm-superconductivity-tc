"""Exploratory, grouped OOF residual-rule learning on the audited 3DSC-MP pilot.

No upstream scripts, LLM API, DFT, or in-sample residual discovery are used.
See PROTOCOL.md, fixed before the outer evaluations.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
from pymatgen.core import Element
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.tree import DecisionTreeRegressor
import sklearn

ROOT = Path(__file__).resolve().parent
SEEDS = [20260929, 20261001, 20261002, 20261003]
COUNTS = [0, 1, 2, 4, 8]
STRENGTHS = [0.25, 0.5, 1.0]
RIDGE = 10.0
ORDINAL_SYMMETRY = ['struct_sym_'+c for c in
                    ['cubic','hexagonal','monoclinic','orthorhombic','tetragonal','triclinic','trigonal']]


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def groups_from_parents(df):
    parent = list(range(len(df)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for col in ['chemical_system', 'material_id_2']:
        seen = {}
        for i, value in enumerate(df[col]):
            if pd.isna(value):
                continue
            if value in seen:
                parent[find(i)] = find(seen[value])
            else:
                seen[value] = i
    return np.array([find(i) for i in range(len(df))])


def baseline(seed, workers):
    return ExtraTreesRegressor(n_estimators=160, min_samples_leaf=2,
                               max_features=1.0, random_state=seed, n_jobs=workers)


def metrics(y_log, pred_log, weights):
    p = np.maximum(0, pred_log)
    y, yp = np.expm1(y_log), np.expm1(p)
    return {'MSLE_log1p': float(np.average((y_log-p)**2, weights=weights)),
            'MAE_K': float(np.average(abs(y-yp), weights=weights)),
            'RMSE_K': float(np.sqrt(np.average((y-yp)**2, weights=weights)))}


def oof_predictions(x, y, w, groups, row_ids, seed, workers, path):
    pred = np.full(len(y), np.nan)
    folds = np.full(len(y), -1)
    audit = []
    cv = GroupKFold(n_splits=4, shuffle=True, random_state=seed+211)
    for fold, (tr, va) in enumerate(cv.split(x, groups=groups)):
        assert not set(groups[tr]) & set(groups[va])
        model = baseline(seed+fold+1, workers)
        model.fit(x[tr], y[tr], sample_weight=w[tr])
        pred[va] = model.predict(x[va])
        folds[va] = fold
        audit.append({'fold': fold, 'train_rows': len(tr), 'validation_rows': len(va),
                      'group_overlap': 0})
    assert np.isfinite(pred).all()
    pd.DataFrame({'row_id': row_ids, 'component': groups, 'oof_fold': folds,
                  'tc_log1p': y, 'baseline_oof_log1p': pred,
                  'residual_log1p': y-pred, 'weight': w}).to_csv(path, index=False)
    return pred, audit


def predicate_text(p):
    if p['op'] == 'any_positive':
        return '('+' OR '.join(c.replace('comp_fraction_Z', 'element_Z')+' present' for c in p['features'])+')'
    if p['op'] == 'none_positive':
        return 'all present elements classified as metals by pymatgen'
    return f"{p['feature']} {p['op']} {p['threshold']:.8g}"


def rule_text(rule):
    return ' AND '.join(predicate_text(p) for p in rule['predicates'])


def gate(df, rule):
    result = np.ones(len(df), dtype=bool)
    for p in rule['predicates']:
        op = p['op']
        if op in ['any_positive', 'none_positive']:
            hit = np.any(df[p['features']].to_numpy(float)>1e-12, axis=1)
            result &= hit if op == 'any_positive' else ~hit
        elif op == '>':
            result &= df[p['feature']].to_numpy(float)>p['threshold']
        elif op == '<=':
            result &= df[p['feature']].to_numpy(float)<=p['threshold']
        else:
            raise ValueError(op)
    return result


def chemistry_rules(columns):
    def positive(z):
        return {'feature': f'comp_fraction_Z{z:03d}', 'op': '>', 'threshold': 1e-12}
    def exists(z):
        return f'comp_fraction_Z{z:03d}' in columns
    out = []
    if exists(29) and exists(8):
        out.append({'name':'Cu_and_O_present', 'predicates':[positive(29), positive(8)]})
    fe_partners = [f'comp_fraction_Z{z:03d}' for z in [15,33,16,34,52] if exists(z)]
    if exists(26) and fe_partners:
        out.append({'name':'Fe_pnictogen_or_chalcogen_present',
                    'predicates':[positive(26), {'features':fe_partners, 'op':'any_positive'}]})
    if exists(5):
        out.append({'name':'B_present', 'predicates':[positive(5)]})
    nonmetal = [c for c in columns if c.startswith('comp_fraction_Z') and
                not Element.from_Z(int(c.rsplit('Z',1)[1])).is_metal]
    out.append({'name':'all_metal_elements', 'predicates':[{'features':nonmetal,'op':'none_positive'}]})
    out.extend([{'name':'ordered_structure','predicates':[{'feature':'struct_is_ordered','op':'>','threshold':0.5}]},
                {'name':'disordered_structure','predicates':[{'feature':'struct_is_ordered','op':'<=','threshold':0.5}]}])
    for r in out:
        r['source'] = 'predefined_chemistry_or_data_quality_gate'
    return out


def candidate_bank(df, y, w, columns, seed):
    raw = chemistry_rules(columns)
    x = df[columns].to_numpy(float)
    for depth in [2,3]:
        for leaf in [40,100]:
            tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf, random_state=seed)
            tree.fit(x, y, sample_weight=w)
            t = tree.tree_
            def walk(node, path):
                if path:
                    raw.append({'name': f'tree_d{depth}_leaf{leaf}_node{node}',
                                'source': 'training_target_tree_path', 'predicates': path})
                if t.children_left[node] != t.children_right[node]:
                    f, threshold = columns[t.feature[node]], float(t.threshold[node])
                    walk(t.children_left[node], path+[{'feature':f,'op':'<=','threshold':threshold}])
                    walk(t.children_right[node], path+[{'feature':f,'op':'>','threshold':threshold}])
            walk(0, [])
    kept, logs, seen = [], [], set()
    for i, rule in enumerate(raw):
        rule['candidate_id'] = f'r{i:03d}'
        rule['text'] = rule_text(rule)
        active = gate(df, rule)
        support = int(active.sum())
        inside_components = int(df.loc[active,'component'].nunique())
        outside_components = int(df.loc[~active,'component'].nunique())
        reason = None
        key = np.packbits(active).tobytes()
        if len(rule['predicates']) > 3:
            reason = 'more_than_three_predicates'
        elif min(support, len(df)-support) < 40:
            reason = 'fewer_than_40_records_inside_or_outside'
        elif min(inside_components,outside_components) < 5:
            reason = 'fewer_than_5_components_inside_or_outside'
        elif key in seen:
            reason = 'duplicate_training_gate'
        if reason is None:
            seen.add(key)
            kept.append(rule)
        logs.append({**rule, 'training_support_rows':support,
                     'training_support_components':inside_components,
                     'training_outside_components':outside_components,
                     'accepted':reason is None, 'rejection_reason':reason})
    matrix = np.column_stack([gate(df,r) for r in kept]).astype(float) if kept else np.zeros((len(df),0))
    return kept, matrix, logs


def fit_rule_path(r, y, w, with_intercept=False):
    """Forward weighted-correlation selection, jointly refitted ridge coefficients."""
    weights = w/w.mean()
    mean_y = float(np.average(y, weights=weights)) if with_intercept else 0.0
    means = np.average(r, axis=0, weights=weights) if with_intercept else np.zeros(r.shape[1])
    a, target = r-means, y-mean_y
    states = [{'selected':[], 'coefficients':[], 'intercept':mean_y}]
    selected = []
    residual = target.copy()
    for step in range(min(8,r.shape[1])):
        numerator = (a.T@(weights*residual))**2
        denominator = (a*a).T@weights + RIDGE
        scores = numerator/denominator
        scores[selected] = -np.inf
        best = int(np.argmax(scores))
        if not np.isfinite(scores[best]) or scores[best] <= 1e-20:
            break
        selected.append(best)
        x = a[:,selected]
        coefficients = np.linalg.solve(x.T@(weights[:,None]*x)+RIDGE*np.eye(len(selected)),
                                       x.T@(weights*target))
        residual = target-x@coefficients
        intercept = mean_y-float(means[selected]@coefficients)
        states.append({'selected':selected.copy(), 'coefficients':coefficients.tolist(),
                       'intercept':intercept})
    return states


def predict_rules(r, state):
    return state['intercept'] + (r[:,state['selected']]@np.array(state['coefficients'])
                                if state['selected'] else np.zeros(r.shape[0]))


def evaluate_candidates(df_val, y_val, w_val, bank, rtrain, target, wtrain,
                        base_val, path, mode):
    rval = np.column_stack([gate(df_val,r) for r in bank]).astype(float) if bank else np.zeros((len(df_val),0))
    records = []
    base_score = metrics(y_val,base_val,w_val)['MSLE_log1p']
    for j, rule in enumerate(bank):
        states = fit_rule_path(rtrain[:,[j]], target, wtrain, with_intercept=(mode=='rule_only'))
        state = states[-1]
        value = predict_rules(rval[:,[j]],state)
        predicted = value if mode=='rule_only' else base_val+value
        rec = {'candidate_id':rule['candidate_id'],'text':rule['text'],
               'train_support_rows':int(rtrain[:,j].sum()),'validation_support_rows':int(rval[:,j].sum()),
               'validation_support_components':int(df_val.loc[rval[:,j].astype(bool),'component'].nunique()),
               'single_rule_coefficient':float(state['coefficients'][0]) if state['coefficients'] else 0.0,
               'single_rule_intercept':state['intercept'],
               **metrics(y_val,predicted,w_val)}
        rec['delta_MSLE_vs_global_composition_baseline'] = rec['MSLE_log1p']-base_score
        records.append(rec)
    pd.DataFrame(records).to_csv(path,index=False)
    return rval


def serialized_model(bank, state, strength, train, val=None, test=None):
    rules=[]
    for index, coef in zip(state['selected'],state['coefficients']):
        rule={**bank[index], 'coefficient_log1p':float(coef),
              'applied_coefficient_log1p':float(strength*coef),
              'train_support_rows':int(gate(train,bank[index]).sum()),
              'train_support_components':int(train.loc[gate(train,bank[index]),'component'].nunique())}
        if val is not None:
            rule['validation_support_rows']=int(gate(val,bank[index]).sum())
            rule['validation_support_components']=int(val.loc[gate(val,bank[index]),'component'].nunique())
        if test is not None:
            rule['test_support_rows']=int(gate(test,bank[index]).sum())
            rule['test_support_components']=int(test.loc[gate(test,bank[index]),'component'].nunique())
        rules.append(rule)
    return {'strength':strength,'intercept_log1p':state['intercept'],
            'applied_intercept_log1p':float(strength*state['intercept']), 'rules':rules,
            'prediction_convention':'raw_log_prediction = baseline_log1p + applied_intercept_log1p + sum(applied_coefficient_log1p * gate); then max(raw_log_prediction,0); then expm1. For rule-only baseline_log1p=0.'}


def split_run(df, comp_columns, rule_columns, components, weights, seed, workers, root):
    out=root/f'seed_{seed}'
    out.mkdir(parents=True,exist_ok=True)
    y=np.log1p(df.tc.to_numpy(float));xc=df[comp_columns].to_numpy(float)
    tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(df,groups=components))
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed+101).split(df.iloc[tr],groups=components[tr]))
    itr,iva=tr[a],tr[b]
    for left,right in [(tr,te),(itr,iva)]:
        assert not set(components[left]) & set(components[right])
        for col in ['chemical_system','material_id_2','canonical_composition']:
            assert not set(df.iloc[left][col]) & set(df.iloc[right][col])
    roles=np.full(len(df),'test',dtype=object);roles[itr]='inner_train';roles[iva]='inner_validation'
    assignments=df[['row_id','chemical_system','material_id_2','canonical_composition']].copy()
    assignments['component']=components;assignments['role']=roles
    assignments.to_csv(out/'split_assignments.csv',index=False)
    print(f'{seed}: inner OOF residuals ({len(itr)} rows)',flush=True)
    oof,fold_audit=oof_predictions(xc[itr],y[itr],weights[itr],components[itr],df.iloc[itr].row_id.to_numpy(),
                                  seed,workers,out/'inner_train_oof.csv')
    inner_base=baseline(seed,workers).fit(xc[itr],y[itr],sample_weight=weights[itr])
    base_val=inner_base.predict(xc[iva]);resid=y[itr]-oof
    pd.DataFrame({'row_id':df.iloc[iva].row_id,'tc_log1p':y[iva],
                  'baseline_log1p':base_val,'weight':weights[iva]}).to_csv(out/'inner_validation_baseline.csv',index=False)
    selections={};tuning=[];inner_models={}
    for mode,target in [('residual',resid),('rule_only',y[itr])]:
        bank,r,log=candidate_bank(df.iloc[itr],target,weights[itr],rule_columns,seed)
        dump(out/f'inner_{mode}_candidates.json',log)
        states=fit_rule_path(r,target,weights[itr],with_intercept=(mode=='rule_only'))
        rv=evaluate_candidates(df.iloc[iva],y[iva],weights[iva],bank,r,target,weights[itr],base_val,
                               out/f'inner_{mode}_candidate_validation.csv',mode)
        candidates=[]
        for requested in COUNTS:
            k=min(requested,len(states)-1);state=states[k]
            for strength in ([1.0] if mode=='rule_only' else ([0.0] if k==0 else STRENGTHS)):
                correction=predict_rules(rv,state)
                pred=correction if mode=='rule_only' else base_val+strength*correction
                rec={'mode':mode,'requested_rule_count':requested,'actual_rule_count':k,'strength':strength,
                     **metrics(y[iva],pred,weights[iva]),'outer_test_used':False}
                candidates.append(rec);tuning.append(rec)
        winner=min(candidates,key=lambda d:(d['MSLE_log1p'],d['actual_rule_count'],d['strength']))
        selections[mode]=winner
        state=states[winner['actual_rule_count']]
        inner_models[mode]=serialized_model(bank,state,winner['strength'],df.iloc[itr],val=df.iloc[iva])
    pd.DataFrame(tuning).to_csv(out/'inner_validation_grid.csv',index=False)
    dump(out/'inner_selected_rules.json',inner_models)
    dump(out/'selection.json',selections)
    print(f'{seed}: selected residual k={selections["residual"]["actual_rule_count"]}, strength={selections["residual"]["strength"]}; refitting full-train OOF',flush=True)
    outer_oof,outer_audit=oof_predictions(xc[tr],y[tr],weights[tr],components[tr],df.iloc[tr].row_id.to_numpy(),
                                        seed+1000,workers,out/'outer_train_oof.csv')
    final_base=baseline(seed,workers).fit(xc[tr],y[tr],sample_weight=weights[tr])
    base_test=final_base.predict(xc[te])
    models={};test_logs={'composition_baseline':base_test};final_counts={}
    for mode,target in [('residual',y[tr]-outer_oof),('rule_only',y[tr])]:
        bank,r,log=candidate_bank(df.iloc[tr],target,weights[tr],rule_columns,seed)
        dump(out/f'final_{mode}_candidates.json',log)
        states=fit_rule_path(r,target,weights[tr],with_intercept=(mode=='rule_only'))
        count=min(selections[mode]['actual_rule_count'],len(states)-1)
        state=states[count];strength=selections[mode]['strength']
        rt=np.column_stack([gate(df.iloc[te],rule) for rule in bank]).astype(float) if bank else np.zeros((len(te),0))
        correction=predict_rules(rt,state)
        models[mode]=serialized_model(bank,state,strength,df.iloc[tr],test=df.iloc[te])
        final_counts[mode]=count
        test_logs['composition_plus_rules' if mode=='residual' else 'rule_only']=base_test+strength*correction if mode=='residual' else correction
    dump(out/'final_rules.json',models)
    records=[]
    prediction=df.iloc[te][['row_id','formula_sc','chemical_system','canonical_composition','material_id_2','sc_class','tc','struct_is_ordered','synth_doped']].copy()
    prediction['component']=components[te];prediction['weight']=weights[te]
    for arm,values in test_logs.items():
        prediction[arm+'_raw_log1p']=values
        prediction[arm+'_tc_K']=np.expm1(np.maximum(0,values))
        records.append({'seed':seed,'arm':arm,'test_rows':len(te),'train_rows':len(tr),
                        **metrics(y[te],values,weights[te]),
                        'final_rule_count':final_counts['residual'] if arm=='composition_plus_rules' else final_counts['rule_only'] if arm=='rule_only' else 0})
    prediction.to_csv(out/'test_predictions.csv',index=False)
    pd.DataFrame(records).to_csv(out/'metrics.csv',index=False)
    families=[]
    for cls,part in prediction.groupby('sc_class'):
        for arm in test_logs:
            families.append({'seed':seed,'family':cls,'arm':arm,'n_rows':len(part),
                             **metrics(np.log1p(part.tc),part[arm+'_raw_log1p'],part.weight)})
    pd.DataFrame(families).to_csv(out/'family_errors.csv',index=False)
    dump(out/'split_audit.json',{'seed':seed,'train_rows':len(tr),'test_rows':len(te),
         'inner_train_rows':len(itr),'inner_validation_rows':len(iva),
         'outer_component_overlap':0,'inner_component_overlap':0,
         'outer_chemical_system_overlap':0,'outer_MP_parent_overlap':0,'outer_composition_overlap':0,
         'inner_oof_folds':fold_audit,'outer_train_oof_folds':outer_audit})
    print(f'{seed}: outer evaluation complete',flush=True)
    return records,prediction


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,default=ROOT.parent/'superconductivity'/'results')
    ap.add_argument('--workers',type=int,default=3)
    args=ap.parse_args();start=time.time()
    result=ROOT/'results';result.mkdir(exist_ok=True)
    df=pd.read_csv(args.source/'features.csv')
    schema=json.loads((args.source/'feature_schema.json').read_text(encoding='utf-8'))
    comp_columns=schema['composition']
    rule_columns=comp_columns+[c for c in schema['structure'] if c not in ORDINAL_SYMMETRY]
    assert len(df)==5773 and df.row_id.is_unique
    assert df.tc.notna().all() and (df.tc>=0).all()
    assert np.isfinite(df[rule_columns].to_numpy(float)).all()
    components=groups_from_parents(df)
    df['component']=components
    weights=1/df.groupby('canonical_composition').canonical_composition.transform('size').to_numpy()
    dump(result/'input_manifest.json',{'source_features_path':str(args.source/'features.csv'),
         'features_sha256':sha(args.source/'features.csv'),'schema_sha256':sha(args.source/'feature_schema.json'),
         'protocol_sha256':sha(ROOT/'PROTOCOL.md'),'learner_sha256':sha(Path(__file__)),
         'n_rows':len(df),'n_composition_features':len(comp_columns),'n_rule_features':len(rule_columns),
         'composition_feature_whitelist':comp_columns,'rule_feature_whitelist':rule_columns,
         'excluded_ordinal_point_group_encodings':ORDINAL_SYMMETRY,
         'seeds':SEEDS,'n_components':int(len(np.unique(components))),
         'formula':'Tc from existing source data, unchanged; y=log1p(Tc); weight=1/matches per canonical composition',
         'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__})
    all_metrics=[];all_predictions=[]
    for seed in SEEDS:
        m,p=split_run(df,comp_columns,rule_columns,components,weights,seed,args.workers,result)
        all_metrics.extend(m);p['seed']=seed;all_predictions.append(p)
    table=pd.DataFrame(all_metrics);table.to_csv(result/'metrics_all.csv',index=False)
    pd.concat(all_predictions,ignore_index=True).to_csv(result/'test_predictions_all.csv',index=False)
    overlaps=[]
    for i,first in enumerate(SEEDS):
        a=set(all_predictions[i].row_id)
        for j in range(i+1,len(SEEDS)):
            b=set(all_predictions[j].row_id)
            overlaps.append({'seed_a':first,'seed_b':SEEDS[j],'shared_test_rows':len(a&b)})
    pd.DataFrame(overlaps).to_csv(result/'test_set_overlap.csv',index=False)
    dump(result/'run_summary.json',{'elapsed_seconds':time.time()-start,'seeds':SEEDS,
         'protocol_frozen_before_outer_evaluation':True,'independent_confirmatory_claim':False,
         'primary_endpoint':'weighted MSLE of log1p Tc','metrics':all_metrics,
         'limitations':['Existing dataset and earlier holdout already inspected; all results exploratory.',
                       'Repeated holdouts overlap and are not independent replications.',
                       'No label changes, LLM calls, DFT, GNN, or autonomous PRIS execution.',
                       'Rule coefficients are conditional predictive corrections, not physical laws.']})
    print(table.to_string(index=False),flush=True)
    print('COMPLETE',flush=True)


if __name__=='__main__':
    main()
