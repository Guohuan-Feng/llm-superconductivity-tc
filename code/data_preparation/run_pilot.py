"""Pinned, reproducible 3DSC-MP audit and small prediction pilot (no PRIS training).

Run with Python containing numpy, pandas, scipy, scikit-learn, and pymatgen.
All downloads are public. Upstream code is not executed. No CIF is force-ordered.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, shutil, time, urllib.request, warnings, zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element, Structure
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
import sklearn

PIN = '2471dd51a298a854cb4f365ebd39e72c7cbf3634'
URL = f'https://codeload.github.com/aimat-lab/3DSC/zip/{PIN}'
SEED = 20260929
FEATURE_VERSION = '2-periodic-sphere-self-images'
ROOT = Path(__file__).resolve().parent
SYM = ['cubic','hexagonal','monoclinic','orthorhombic','tetragonal','triclinic','trigonal',
       'primitive','base-centered','body-centered','face-centered']

def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()

def prepare(work):
    data = ROOT / 'data'
    cifs = data / 'cifs'
    cifs.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    csv_path=data/'3DSC_MP.csv'
    if (data/'source_manifest.json').exists() and csv_path.exists():
        manifest=json.loads((data/'source_manifest.json').read_text(encoding='utf-8'))
        if manifest['commit']==PIN and sha(csv_path)==manifest['csv_sha256']:
            df=pd.read_csv(csv_path,comment='#')
            if set(Path(x).name for x in df.cif).issubset(set(os.listdir(cifs))):
                return df
    archive = work / '3DSC-pinned.zip'
    if not archive.exists(): urllib.request.urlretrieve(URL, archive)
    prefix = f'3DSC-{PIN}/superconductors_3D/'
    with zipfile.ZipFile(archive) as z:
        csv_path = data / '3DSC_MP.csv'
        csv_path.write_bytes(z.read(prefix + 'data/final/MP/3DSC_MP.csv'))
        df = pd.read_csv(csv_path, comment='#')
        manifest = []
        for rel in df.cif:
            name = Path(rel).name
            dest = cifs / name
            if dest.resolve().parent != cifs.resolve(): raise ValueError('Unsafe path')
            if not dest.exists(): dest.write_bytes(z.read(prefix + rel))
            manifest.append({'cif': name, 'sha256': sha(dest), 'bytes': dest.stat().st_size})
        for name in ['LICENSE.md','README.md']:
            (data / ('UPSTREAM_' + name)).write_bytes(z.read(f'3DSC-{PIN}/' + name))
    pd.DataFrame(manifest).to_csv(data / 'cif_manifest.csv', index=False)
    dump(data / 'source_manifest.json', {'repository':'https://github.com/aimat-lab/3DSC',
         'commit':PIN,'archive_url':URL,'archive_sha256':sha(archive),
         'csv_sha256':sha(csv_path),'structure_count':len(manifest),
         'data_license':'CC BY 4.0','code_license':'upstream MIT',
         'download_note':'Only referenced final CIFs extracted; upstream code not executed.'})
    return df

def composition_features(formula):
    c = Composition(formula)
    elems = list(c.elements)
    w = np.array([c.get_atomic_fraction(e) for e in elems])
    out = {'comp_n_elements':len(elems), 'comp_entropy':float(-sum(w*np.log(w))),
           'comp_max_fraction':float(max(w))}
    for e in elems: out[f'comp_fraction_Z{e.Z:03d}'] = c.get_atomic_fraction(e)
    props = {'Z':[float(e.Z) for e in elems], 'mass':[float(e.atomic_mass) for e in elems],
             'electronegativity':[float(e.X) if e.X is not None else np.nan for e in elems],
             'row':[float(e.row) for e in elems], 'group':[float(e.group) for e in elems],
             'atomic_radius':[float(e.atomic_radius) if e.atomic_radius is not None else np.nan for e in elems]}
    for key, vals in props.items():
        a=np.array(vals); ok=np.isfinite(a)
        if not ok.any():
            for stat in ['mean','std','min','max']: out[f'comp_{key}_{stat}']=np.nan
        else:
            ww=w[ok]/sum(w[ok]); aa=a[ok]; mean=float(sum(ww*aa))
            out.update({f'comp_{key}_mean':mean, f'comp_{key}_std':float(np.sqrt(sum(ww*(aa-mean)**2))),
                        f'comp_{key}_min':float(min(aa)), f'comp_{key}_max':float(max(aa))})
    return out, '-'.join(sorted(e.symbol for e in elems)), tuple(sorted((e.symbol, round(c.get_atomic_fraction(e),10)) for e in elems))

def structure_features(arg):
    row_id, name = arg
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            s=Structure.from_file(ROOT/'data'/'cifs'/name, primitive=False)
        occ=np.array([sum(float(v) for v in site.species.values()) for site in s])
        dm=s.distance_matrix.copy(); np.fill_diagonal(dm, np.inf)
        # Include the shortest nonzero self-image for small primitive cells.
        # A lattice basis vector supplies an upper bound. The sphere search
        # includes translations beyond [-1, 1] for non-reduced lattice bases.
        _,self_distances,_,_=s.lattice.get_points_in_sphere(
            [[0,0,0]],[0,0,0],r=min(s.lattice.abc)+1e-7,zip_results=False)
        self_distance=float(np.min(self_distances[self_distances>1e-8]))
        nn=np.minimum(dm.min(axis=1),self_distance)
        lengths=np.sort(s.lattice.abc); angles=np.sort(s.lattice.angles)
        out={'struct_n_sites':len(s), 'struct_n_atoms_occupancy':float(occ.sum()),
             'struct_volume_per_atom':float(s.volume/occ.sum()), 'struct_density':float(s.density),
             'struct_length_ratio_mid':float(lengths[1]/lengths[0]),
             'struct_length_ratio_max':float(lengths[2]/lengths[0]),
             'struct_angle_min':float(angles[0]),'struct_angle_mid':float(angles[1]),'struct_angle_max':float(angles[2]),
             'struct_occupancy_min':float(occ.min()),'struct_occupancy_mean':float(occ.mean()),
             'struct_disordered_site_fraction':float(np.mean([not site.is_ordered for site in s])),
             'struct_is_ordered':int(s.is_ordered),
             'struct_nearest_distance_min':float(nn.min()),'struct_nearest_distance_mean':float(nn.mean()),
             'struct_nearest_distance_std':float(nn.std()),'struct_nearest_distance_max':float(nn.max())}
        return row_id,out,None
    except Exception as exc: return row_id,{},f'{type(exc).__name__}: {exc}'

def weighted_metrics(y, pred, weights):
    return {'n_rows':len(y),'MAE_K':float(mean_absolute_error(y,pred,sample_weight=weights)),
            'RMSE_K':float(mean_squared_error(y,pred,sample_weight=weights)**.5),
            'MSLE_log1p':float(mean_squared_error(np.log1p(y),np.log1p(pred),sample_weight=weights)),
            'R2':float(r2_score(y,pred,sample_weight=weights)) if len(y)>1 else None}

def component_groups(df):
    """Join records sharing either an element set OR original MP parent."""
    parent=list(range(len(df)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for col in ['chemical_system','material_id_2']:
        seen={}
        for i,value in enumerate(df[col]):
            if pd.isna(value):continue
            if value in seen:parent[find(i)]=find(seen[value])
            else:seen[value]=i
    return np.array([find(i) for i in range(len(df))])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work-dir',type=Path,default=ROOT/'work')
    ap.add_argument('--workers',type=int,default=3);ap.add_argument('--prepare-only',action='store_true')
    args=ap.parse_args(); start=time.time()
    for p in ['data','results']: (ROOT/p).mkdir(exist_ok=True,parents=True)
    df=prepare(args.work_dir);df.insert(0,'row_id',np.arange(len(df)))
    print(f'Prepared {len(df)} referenced structures.',flush=True)
    if args.prepare_only:return
    result=ROOT/'results';cache=result/'features.csv'
    cache_meta=result/'feature_cache_metadata.json'
    expected_meta={'feature_version':FEATURE_VERSION,'commit':PIN,
                   'csv_sha256':sha(ROOT/'data'/'3DSC_MP.csv'),
                   'cif_manifest_sha256':sha(ROOT/'data'/'cif_manifest.csv')}
    if cache.exists() and cache_meta.exists() and json.loads(cache_meta.read_text(encoding='utf-8'))==expected_meta:
        features=pd.read_csv(cache);print('Using saved feature cache.',flush=True)
    else:
        rows=[]
        for row in df.itertuples():
            out,system,canonical=composition_features(row.formula_sc)
            out.update({'row_id':row.row_id,'formula_sc':row.formula_sc,'chemical_system':system,
                        'canonical_composition':repr(canonical),'tc':float(row.tc),'sc_class':row.sc_class,
                        'material_id_2':row.material_id_2,'cif':Path(row.cif).name,
                        'synth_doped':bool(row.synth_doped),'upstream_weight':float(row.weight)})
            rows.append(out)
        features=pd.DataFrame(rows)
        for col in [c for c in features if c.startswith('comp_fraction_')]:features[col]=features[col].fillna(0)
        struc={};fails=[]
        tasks=list(zip(df.row_id,[Path(x).name for x in df.cif]))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i,(row_id,out,error) in enumerate(pool.map(structure_features,tasks,chunksize=25)):
                struc[row_id]=out
                if error:fails.append({'row_id':row_id,'error':error})
                if (i+1)%500==0:print(f'Structure features {i+1}/{len(tasks)}',flush=True)
        sdf=pd.DataFrame.from_dict(struc,orient='index');sdf.index.name='row_id'
        features=features.merge(sdf,on='row_id',how='left',validate='one_to_one')
        for col in SYM:features['struct_sym_'+col]=pd.to_numeric(df[col],errors='coerce')
        features['structure_parse_ok']=features.row_id.map(lambda x:bool(struc[x]))
        features.to_csv(cache,index=False)
        dump(cache_meta,expected_meta)
        pd.DataFrame(fails,columns=['row_id','error']).to_csv(result/'parse_failures.csv',index=False)
    # Do not silently discard labels or merge conflicting temperatures.
    conflicts=features.groupby('canonical_composition').tc.nunique()
    df.groupby('formula_sc').agg(n_matches=('tc','size'),n_tc=('tc','nunique'),tc_min=('tc','min'),tc_max=('tc','max')).query('n_matches > 1').to_csv(result/'multiple_matches.csv')
    audit={'rows':len(df),'unique_formula_strings':int(df.formula_sc.nunique()),'unique_cif_paths':int(df.cif.nunique()),
           'unique_canonical_compositions':int(features.canonical_composition.nunique()),
           'chemical_systems':int(features.chemical_system.nunique()),'missing_tc':int(df.tc.isna().sum()),
           'negative_tc':int((df.tc<0).sum()),'tc_zero_rows':int((df.tc==0).sum()),'tc_positive_rows':int((df.tc>0).sum()),
           'zero_unique_formulas':int(df.loc[df.tc==0,'formula_sc'].nunique()),'positive_unique_formulas':int(df.loc[df.tc>0,'formula_sc'].nunique()),
           'formula_strings_with_multiple_matches':int((df.groupby('formula_sc').size()>1).sum()),
           'canonical_compositions_with_conflicting_tc':int((conflicts>1).sum()),
           'max_tc_K':float(df.tc.max()),'synthetic_doped_rows':int(df.synth_doped.sum()),
           'structure_parse_success':int(features.structure_parse_ok.sum()),
           'ordered_rows':int((features.struct_is_ordered==1).sum()),
           'disordered_rows':int((features.struct_is_ordered==0).sum()),
           'labels':'Reported SuperCon experimental critical temperature; recorded nonsuperconductors have tc=0. Missing labels are never filled with zero.',
           'geometry':'Matched MP geometry, often artificially doped without relaxation; not paired experimentally measured geometry.',
           'source_group_string_order_differences':int((features.chemical_system!=df.chemical_composition_sc).sum()),
           'source_group_element_set_mismatches':int(sum(set(a.split('-'))!=set(b.split('-')) for a,b in zip(features.chemical_system,df.chemical_composition_sc)))}
    dump(result/'data_audit.json',audit)
    # Same successfully parsed, labelled cohort for both feature sets.
    cohort=features.loc[features.structure_parse_ok & features.tc.notna() & (features.tc>=0)].reset_index(drop=True)
    comp_cols=[c for c in cohort if c.startswith('comp_')]
    struct_cols=[c for c in cohort if c.startswith('struct_')]
    components=component_groups(cohort)
    component_sizes=pd.Series(components).value_counts()
    dump(result/'strict_components.json',{'grouping':'Connected components: same chemical system OR same material_id_2',
         'n_components':len(component_sizes),'largest_component_rows':int(component_sizes.max()),
         'largest_component_fraction':float(component_sizes.max()/len(cohort)),
         'top_10_component_sizes':component_sizes.head(10).tolist()})
    dump(result/'feature_schema.json',{'composition':comp_cols,'structure':struct_cols,
         'excluded_predictors':['tc','sc_class','formula_sc','chemical_system','material_id_2','synth_doped','upstream_weight'],
         'structure_symmetry_note':'Upstream symmetry labels describe matched geometry; all other structure features recomputed from final CIFs.',
         'nearest_distance_note':'Minimum-image pair distances plus shortest nonzero self-image from an exhaustive lattice sphere search bounded by the shortest basis-vector length; simple descriptor, not full coordination analysis.'})
    all_metrics=[];all_predictions=[];all_splits=[];tuning=[];importance=[]
    for split_name,groups in [('chemical_system',cohort.chemical_system),('chemical_system_parent_components',components),
                             ('formula_random_diagnostic',cohort.canonical_composition)]:
        tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED).split(cohort,groups=groups))
        inner_groups=components[tr] if split_name=='chemical_system_parent_components' else cohort.iloc[tr].chemical_system
        inner_tr,inner_va=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED+1).split(cohort.iloc[tr],groups=inner_groups))
        itr,iva=tr[inner_tr],tr[inner_va]
        weights=1/cohort.groupby('canonical_composition').canonical_composition.transform('size').to_numpy()
        test_set,validation_set=set(te),set(iva)
        for i in range(len(cohort)):
            all_splits.append({'split':split_name,'row_id':int(cohort.iloc[i].row_id),
                'role':'test' if i in test_set else ('inner_validation' if i in validation_set else 'inner_train'),
                'parent_component':int(components[i]),
                'chemical_system':cohort.iloc[i].chemical_system,'canonical_composition':cohort.iloc[i].canonical_composition})
        med=float(np.median(np.log1p(cohort.iloc[tr].tc)))
        dummy=np.full(len(te),np.expm1(med))
        all_metrics.append({'split':split_name,'features':'dummy_train_median','model':'constant',
                            **weighted_metrics(cohort.iloc[te].tc,dummy,weights[te])})
        for feat_name,cols in [('composition',comp_cols),('composition_structure',comp_cols+struct_cols)]:
            x=cohort[cols].to_numpy(float);y=np.log1p(cohort.tc.to_numpy(float))
            imp=SimpleImputer(strategy='median',keep_empty_features=True)
            a=imp.fit_transform(x[itr]);b=imp.transform(x[iva]);best=None
            for leaf in [2,8]:
                model=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=leaf,max_features=1.0,
                                         random_state=SEED,n_jobs=args.workers)
                model.fit(a,y[itr],sample_weight=weights[itr]);pr=np.maximum(0,model.predict(b))
                score=float(mean_squared_error(y[iva],pr,sample_weight=weights[iva]))
                tuning.append({'split':split_name,'features':feat_name,'min_samples_leaf':leaf,
                               'inner_validation_MSLE_log1p':score,'outer_test_used':False})
                if best is None or score<best[0]:best=(score,leaf)
            imp=SimpleImputer(strategy='median',keep_empty_features=True)
            a=imp.fit_transform(x[tr]);b=imp.transform(x[te])
            model=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=best[1],max_features=1.0,
                                     random_state=SEED,n_jobs=args.workers)
            model.fit(a,y[tr],sample_weight=weights[tr]);pred=np.expm1(np.maximum(0,model.predict(b)))
            rec={'split':split_name,'features':feat_name,'model':'ExtraTrees','min_samples_leaf':best[1],
                 'n_train_rows':len(tr),'train_test_chemical_system_overlap':len(set(cohort.iloc[tr].chemical_system)&set(cohort.iloc[te].chemical_system)),
                 'train_test_canonical_composition_overlap':len(set(cohort.iloc[tr].canonical_composition)&set(cohort.iloc[te].canonical_composition)),
                 'train_test_parent_MP_overlap':len(set(cohort.iloc[tr].material_id_2)&set(cohort.iloc[te].material_id_2)),
                 **weighted_metrics(cohort.iloc[te].tc,pred,weights[te])}
            all_metrics.append(rec);print(rec,flush=True)
            for j,i in enumerate(te):
                row=cohort.iloc[i]
                all_predictions.append({'split':split_name,'features':feat_name,'row_id':int(row.row_id),
                    'formula_sc':row.formula_sc,'chemical_system':row.chemical_system,'sc_class':row.sc_class,
                    'tc_true_K':float(row.tc),'tc_pred_K':float(pred[j]),'weight':float(weights[i]),
                    'absolute_error_K':float(abs(pred[j]-row.tc)),
                    'squared_log1p_error':float((np.log1p(pred[j])-np.log1p(row.tc))**2),
                    'struct_is_ordered':int(row.struct_is_ordered),'synth_doped':bool(row.synth_doped)})
            importance.extend({'split':split_name,'features':feat_name,'feature':c,'impurity_importance':float(v)} for c,v in zip(cols,model.feature_importances_))
    pd.DataFrame(all_splits).to_csv(result/'split_assignments.csv',index=False)
    pd.DataFrame(tuning).to_csv(result/'train_only_tuning.csv',index=False)
    pd.DataFrame(all_metrics).to_csv(result/'metrics.csv',index=False)
    pred=pd.DataFrame(all_predictions);pred.to_csv(result/'test_predictions.csv',index=False)
    pred.sort_values('absolute_error_K',ascending=False).head(100).to_csv(result/'largest_errors.csv',index=False)
    family=[]
    for key,g in pred.groupby(['split','features','sc_class']):
        family.append(dict(zip(['split','features','sc_class'],key),**weighted_metrics(g.tc_true_K,g.tc_pred_K,g.weight)))
    pd.DataFrame(family).to_csv(result/'family_errors.csv',index=False)
    pd.DataFrame(importance).to_csv(result/'feature_importances.csv',index=False)
    dump(result/'run_summary.json',{'seed':SEED,'elapsed_seconds':time.time()-start,'cohort_rows':len(cohort),
         'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,
         'metrics':all_metrics,'caveats':['One deterministic outer holdout; no confidence intervals or significance claim.',
         'Formula-random diagnostic prevents exact composition overlap, but allows chemical-system/prototype overlap.',
         'Chemical-system split includes dopants and can still share parent MP structures across different dopants.',
         'Simple feature pilot; not a reproduction of original MAGPIE+DSOAP XGBoost scores.',
         'No PRIS agent loop or GNN was trained. No missing Tc was converted to zero.']})
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
