"""Fixed CPU composition118 baselines; never writes the runner-owned MAIN config."""
from __future__ import annotations
import ctypes,json,sys
from common import ROOT,DATA,load_data,np,write_json
from g7b_inputs import OUT,WORK,read,sha,now
from g7_eval import composition_features
from g3_ml import PARAMS,weights
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

def run():
    # Element parsing is supplied by the existing P1 environment, without replacing the frozen ML stack.
    sys.path.append(str(ROOT/'work/p1-env/Lib/site-packages'))
    cfg=read(OUT/'config.json');d=load_data();X=d['X127'][:,:118]
    f3=read(ROOT/'F3_full3764/config.json');formulas=[f3['cases'][str(k)]['material']['formula'] for k in d['id']]
    expected=composition_features(formulas);assert np.allclose(X,expected,atol=2e-8,rtol=2e-8)
    Q=composition_features([c['formula'] for c in cfg['cases']]);results={}
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        for threshold in (5,10):
            y=(d['tc']>threshold).astype(int);oof=np.full(3764,np.nan);folds=[]
            for outer in (0,1,2):
                train=np.flatnonzero(d['fold']!=outer);held=np.flatnonzero(d['fold']==outer)
                assert not set(d['group'][train])&set(d['group'][held]);ws,wc=weights(d['group'][train],y[train])
                with threadpool_limits(limits=4):
                    sc=StandardScaler();Xt=sc.fit_transform(X[train],sample_weight=ws)
                    m=HistGradientBoostingClassifier(**PARAMS);m.fit(Xt,y[train],sample_weight=wc);oof[held]=m.predict_proba(sc.transform(X[held]))[:,1]
                folds.append({'fold':outer,'train_n':len(train),'held_n':len(held),'group_overlap':0,'n_iter':int(m.n_iter_)})
            assert np.isfinite(oof).all();ws,wc=weights(d['group'],y)
            with threadpool_limits(limits=4):
                sc=StandardScaler();Xt=sc.fit_transform(X,sample_weight=ws);m=HistGradientBoostingClassifier(**PARAMS)
                m.fit(Xt,y,sample_weight=wc);p=m.predict_proba(sc.transform(Q))[:,1]
            results[str(threshold)]={'n_positive3764':int(y.sum()),'original3764_OOF':{str(k):float(v) for k,v in zip(d['id'],oof)},'new89_fullfit':{c['id']:float(v) for c,v in zip(cfg['cases'],p)},'folds':folds,'fullfit_n_iter':int(m.n_iter_)}
            print(now(),f'HGB>{threshold}K complete',flush=True)
        # Fixed 10K reproduction against prior G7 baseline; no model selection.
        import csv
        with (ROOT/'work/G7_eval/ml_predictions.csv').open(newline='',encoding='utf8') as h:prior={r['id']:float(r['score']) for r in csv.DictReader(h)}
        diff=max(abs(results['10']['new89_fullfit'][k]-v) for k,v in prior.items());assert diff<1e-12
        obj={'status':'complete','completed_utc':now(),'thresholds':results,'training_meta':{'train_n':3764,'train_groups':1117,'features':'atomic-fraction Z1..118 only','fixed_params':PARAMS,'no_tuning':True,'scaler_weights':'1/TRAIN component size','HGB_weights':'1/TRAIN component size, then class-balanced','outer_folds':'original unchanged three folds; scaler and all fits TRAIN-only','fullfit_labels':'original3764 only','new89_labels_used_for_fit':False,'original_composition_reconstruction_max_abs_diff':float(np.max(np.abs(X-expected))),'G7_prior10K_max_abs_probability_diff':diff},'source_hashes':{str(p):sha(p) for p in (DATA,ROOT/'g7b_ml.py',ROOT/'g3_ml.py',ROOT/'g7_eval.py',ROOT/'common.py')}}
        write_json(WORK/'ML_baselines.json',obj);print('SAVED',WORK/'ML_baselines.json',flush=True)
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':run()
