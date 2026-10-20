"""Fixed original-three-fold HGB classification and log1p(Tc) regression."""
from common import np
from g3_ml import PARAMS as CLASSIFIER_PARAMS, weights, indices_digest
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import time

REGRESSOR_PARAMS={k:v for k,v in CLASSIFIER_PARAMS.items() if k!='class_weight'}
REGRESSOR_PARAMS['loss']='squared_error'
THRESHOLDS=(5,10,15)

def train_oof(d,log):
    assert np.all(d['tc']>=0) and np.isfinite(d['tc']).all()
    scores={f'HGB_classifier_gt_{t}K':np.full(3764,np.nan) for t in THRESHOLDS}
    reg=np.full(3764,np.nan)
    records=[]
    for fold in range(3):
        train,test=np.flatnonzero(d['fold']!=fold),np.flatnonzero(d['fold']==fold)
        assert not set(d['group'][train]) & set(d['group'][test])
        unique,inv,count=np.unique(d['group'][train],return_inverse=True,return_counts=True)
        ws=1./count[inv]
        with threadpool_limits(limits=4):
            scaler=StandardScaler(copy=True,with_mean=True,with_std=True)
            Xt=scaler.fit_transform(d['X127'][train],sample_weight=ws)
            Xq=scaler.transform(d['X127'][test])
            for t in THRESHOLDS:
                y=(d['tc']>t).astype(int)
                _,wc=weights(d['group'][train],y[train])
                started=time.monotonic()
                model=HistGradientBoostingClassifier(**CLASSIFIER_PARAMS)
                model.fit(Xt,y[train],sample_weight=wc)
                p=model.predict_proba(Xq)[:,1]
                scores[f'HGB_classifier_gt_{t}K'][test]=p
                record={'fold':fold,'target':f'Tc>{t}K','n_train':len(train),'n_test':len(test),'n_train_groups':len(unique),
                        'train_indices_sha256':indices_digest(train),'test_indices_sha256':indices_digest(test),
                        'group_overlap':0,'n_iter':int(model.n_iter_),'seconds':time.monotonic()-started,
                        'weights':'TRAIN group/class balanced; scaler TRAIN group only','class_group_masses':[float(ws[y[train]==c].sum()) for c in (0,1)]}
                if t==10:
                    record['original_HGB_max_abs_diff']=float(np.max(np.abs(p-d['hgb'][test])))
                    assert np.allclose(p,d['hgb'][test],atol=2e-10,rtol=2e-12)
                records.append(record)
                log(f'FIT fold={fold} HGB_classifier_gt_{t}K seconds={record["seconds"]:.3f}')
            started=time.monotonic()
            model=HistGradientBoostingRegressor(**REGRESSOR_PARAMS)
            model.fit(Xt,np.log1p(d['tc'][train]),sample_weight=ws)
            raw=model.predict(Xq)
            pred=np.maximum(0.,np.expm1(raw))
            assert np.isfinite(pred).all()
            reg[test]=pred
            record={'fold':fold,'target':'log1p(Tc)','n_train':len(train),'n_test':len(test),'n_train_groups':len(unique),
                    'train_indices_sha256':indices_digest(train),'test_indices_sha256':indices_digest(test),'group_overlap':0,
                    'n_iter':int(model.n_iter_),'seconds':time.monotonic()-started,'weights':'TRAIN group weights only; no class balancing',
                    'inverse':'max(0,expm1(predicted_log1p_Tc))','negative_inverse_clipped_count':int(np.sum(np.expm1(raw)<0))}
            records.append(record)
            log(f'FIT fold={fold} HGB_regressor_log1p_Tc seconds={record["seconds"]:.3f}')
    assert all(np.isfinite(p).all() for p in scores.values()) and np.isfinite(reg).all()
    return {k:v.tolist() for k,v in scores.items()},reg.tolist(),{
        'original_outer_folds':[0,1,2],'classifier_parameters':CLASSIFIER_PARAMS,'regressor_parameters':REGRESSOR_PARAMS,
        'target_definition':'strict Tc>threshold, thresholds5/10/15K; Tc=threshold is negative',
        'fit_records':records,'n_classifier_fits':9,'n_regressor_fits':3,'hyperparameter_tuning':'none',
        'early_stopping':False,'preprocessing':'TRAIN-only group weighted StandardScaler; no held data in fitting'}
