"""Strict nested group-isolated stacking and frozen HGB family augmentation."""
from common import ROOT, load_data, np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import time

PARAMS = {'learning_rate': .05, 'max_iter': 150, 'max_leaf_nodes': 7, 'min_samples_leaf': 20,
          'l2_regularization': 1., 'max_bins': 64, 'early_stopping': False, 'random_state': 20261010,
          'class_weight': None, 'categorical_features': None, 'loss': 'log_loss', 'max_features': 1.,
          'max_depth': None, 'validation_fraction': None, 'warm_start': False}
LR_PARAMS = {'C': 1., 'solver': 'liblinear', 'max_iter': 3000, 'tol': 1e-7, 'random_state': 20261018,
             'class_weight': None, 'penalty': 'l2', 'fit_intercept': True}

def weights(groups, y):
    unique, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    ws = 1. / counts[inv]
    masses = np.asarray([ws[y == c].sum() for c in (0, 1)])
    assert min(masses) > 0
    wc = ws * (float(ws.sum()) / (2 * masses))[y]
    return ws, wc

def indices_digest(ids):
    import hashlib
    return hashlib.sha256(np.asarray(ids, dtype='<i8').tobytes()).hexdigest()

def hgb_fit_predict(d, train, query, X, log, name):
    assert not set(d['group'][train]) & set(d['group'][query])
    ws, wc = weights(d['group'][train], d['y'][train])
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    start = time.monotonic()
    with threadpool_limits(limits=4):
        Xt = scaler.fit_transform(X[train], sample_weight=ws)
        Xq = scaler.transform(X[query])
        model = HistGradientBoostingClassifier(**PARAMS)
        model.fit(Xt, d['y'][train], sample_weight=wc)
        p = model.predict_proba(Xq)[:, 1]
    meta = {'name': name, 'train_n': len(train), 'query_n': len(query),
            'train_groups': len(set(d['group'][train])), 'query_groups': len(set(d['group'][query])),
            'train_indices_sha256': indices_digest(train), 'query_indices_sha256': indices_digest(query),
            'group_overlap': 0, 'fit_seconds': time.monotonic() - start, 'hgb_n_iter': int(model.n_iter_),
            'scaler_fit_scope': 'this fit TRAIN only', 'early_stopping': False}
    log(f"FIT {name} train={len(train)} held={len(query)} seconds={meta['fit_seconds']:.3f}")
    return p, meta

def nested_stack(d, a3, log):
    n = len(d['y'])
    outer_oof, stacked = np.full(n, np.nan), np.full(n, np.nan)
    meta = {'hgb_parameters': PARAMS, 'LR_parameters': LR_PARAMS, 'HGB_fit_records': [], 'outer_folds': [],
            'tuning': 'none; fixed C=1 and all HGB parameters; no held labels used in fitting/selection',
            'meta_features': ['inner_OOF_HGB_probability', 'frozen_A3_probability'],
            'weights': 'TRAIN 1/group_size scaler; TRAIN group/class-balanced classifier weights',
            'A3_is_label_independent': True, 'old_pooled_HGB_OOF_used_for_meta_training': False}
    for outer in range(3):
        train, test = np.flatnonzero(d['fold'] != outer), np.flatnonzero(d['fold'] == outer)
        inner_p = np.full(n, np.nan)
        inner = [v for v in range(3) if v != outer]
        for inner_held in inner:
            inner_train = np.flatnonzero((d['fold'] != outer) & (d['fold'] != inner_held))
            inner_query = np.flatnonzero(d['fold'] == inner_held)
            assert not set(inner_train) & set(test) and not set(inner_query) & set(test)
            p, record = hgb_fit_predict(d, inner_train, inner_query, d['X127'], log, f'outer{outer}_innerheld{inner_held}')
            inner_p[inner_query] = p
            meta['HGB_fit_records'].append(record)
        assert np.isfinite(inner_p[train]).all() and np.isnan(inner_p[test]).all()
        outer_p, record = hgb_fit_predict(d, train, test, d['X127'], log, f'outer{outer}_allTRAIN')
        outer_oof[test] = outer_p
        meta['HGB_fit_records'].append(record)
        maxdiff = float(np.max(np.abs(outer_p - d['hgb'][test])))
        assert np.allclose(outer_p, d['hgb'][test], atol=2e-10, rtol=2e-12), ('Original HGB reproduction failed', outer, maxdiff)
        Xm = np.column_stack((inner_p[train], a3[train]))
        Xq = np.column_stack((outer_p, a3[test]))
        ws, wc = weights(d['group'][train], d['y'][train])
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        with threadpool_limits(limits=4):
            Xt = scaler.fit_transform(Xm, sample_weight=ws)
            Xtest = scaler.transform(Xq)
            lr = LogisticRegression(**LR_PARAMS)
            lr.fit(Xt, d['y'][train], sample_weight=wc)
            assert np.all(lr.n_iter_ < LR_PARAMS['max_iter'])
            stacked[test] = lr.predict_proba(Xtest)[:, 1]
        meta['outer_folds'].append({'outer': outer, 'train_n': len(train), 'test_n': len(test),
            'HGB_original_max_abs_diff': maxdiff, 'HGB_original_matches': True,
            'meta_train_indices_sha256': indices_digest(train), 'meta_test_indices_sha256': indices_digest(test),
            'inner_HGB_train_probabilities': inner_p[train].tolist(),
            'inner_HGB_train_indices': train.tolist(), 'LR_coef_scaled': lr.coef_.tolist(),
            'LR_intercept_scaled': lr.intercept_.tolist(), 'LR_n_iter': lr.n_iter_.tolist()})
    assert np.isfinite(outer_oof).all() and np.isfinite(stacked).all()
    return {'G3_HGB_reproduced': outer_oof.tolist(), 'G3_stacked_LR': stacked.tolist(),
            'G3_simple_mean': ((d['hgb'] + a3) / 2).tolist()}, meta

def augment_family(d, labels, family_values, log):
    assert len(labels) == 3764 and all(v in family_values for v in labels)
    # The category vocabulary is a fixed, label-free list, not fitted using held outcomes.
    onehot = np.asarray([[float(label == value) for value in family_values] for label in labels])
    X = np.column_stack((d['X127'], onehot))
    scores = np.full(3764, np.nan)
    records = []
    for outer in range(3):
        train, test = np.flatnonzero(d['fold'] != outer), np.flatnonzero(d['fold'] == outer)
        scores[test], meta = hgb_fit_predict(d, train, test, X, log, f'family_outer{outer}')
        records.append(meta)
    assert np.isfinite(scores).all()
    return scores.tolist(), {'HGB_parameters': PARAMS, 'n_features': 139, 'family_vocabulary': list(family_values),
                            'onehot_categories_predeclared': True, 'fit_records': records,
                            'not_superconductor_label_boundary': 'LLM prior judgment, not an independent pure-chemistry family; never equated with measured label.'}
