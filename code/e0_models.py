"""Clean finite E0 reimplementation; old scientific inputs are read only."""
from common import BASE, BANKS, DIAGNOSTIC, PROGRAMME, np
from pathlib import Path
import hashlib
import json
import math
import warnings

FAMILIES = ('D_COMMON969', 'D_KB_NORMALIZED1015', 'RF_SHARED1015', 'LR_SHARED1015')
WEIGHTS = np.asarray(((.25, .25, .25, .25), (.625, .125, .125, .125),
    (.125, .625, .125, .125), (.125, .125, .625, .125), (.125, .125, .125, .625)))
SALT = 'original-component-label-independent-crossfit/v1/20261012'

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode('utf8')).hexdigest()

def subfold(group):
    return int.from_bytes(hashlib.sha256(canonical([SALT, str(group)]).encode()).digest()[:8], 'big') % 2

def scope_for(name, ids, d, outer):
    ids = np.asarray(sorted(ids), dtype=int)
    assert np.all(d['fold'][ids] != outer)
    body = dict(scope_version='nested-conventional-controls-SOURCE/v1', name=name,
        original_source_indices=[int(d['source_index'][i]) for i in ids],
        original_component_ids=sorted(set(str(g) for g in d['group'][ids])),
        outer_fold=int(outer), original_denominators_preserved=True)
    return dict(body, scope_sha256=digest(body), indices=ids)

def scopes(d):
    result = []
    sf = np.asarray([subfold(g) for g in d['group']])
    for outer in range(3):
        result.append(scope_for('outerTRAIN'+str(outer), np.flatnonzero(d['fold'] != outer), d, outer))
        for tf in range(3):
            if tf == outer:
                continue
            name = 'outer'+str(outer)+'_innerTRAIN'+str(tf)
            ids = np.flatnonzero(d['fold'] == tf)
            result.append(scope_for(name, ids, d, outer))
            for hs in (0, 1):
                result.append(scope_for(name+'_exclude_subfold'+str(hs), ids[sf[ids] != hs], d, outer))
    return result

def find_inner_banks(d):
    found = {}
    roots = {0: BASE/'workflow-review-2026-10-10/full-comparison-campaign-022/science',
             1: BASE/'workflow-review-2026-10-10/full-comparison-campaign-035/science',
             2: BASE/'workflow-review-2026-10-15/strong-ml-priority-campaign-001/run/science'}
    for s in scopes(d):
        if s['name'].startswith('outerTRAIN'):
            continue
        path = roots[s['outer_fold']]/(digest(['inner-bank:'+s['scope_sha256'], 'finalized'])+'.json')
        if not path.is_file():
            raise FileNotFoundError(path)
        found[s['scope_sha256']] = path
    return found

def load_agent(d):
    doc = json.loads(DIAGNOSTIC.read_text(encoding='utf-8'))
    by_source = {int(v): i for i,v in enumerate(d['source_index'])}
    indices, scores, status = [], [], []
    for case in doc['per_case_provenance']:
        i = by_source[case['original_source_index']]
        mechanism = case['derived_mechanism']
        if mechanism == 'rule_prediction':
            original = json.loads(Path(case['actual_original_output_ref']['path']).read_text(encoding='utf-8'))
            payload = original['payload']
            assert payload['identity']['original_source_index'] == int(d['source_index'][i])
            result = payload['actual_numeric_result']
            assert not result['fallback_used'] and result['numeric_rule_prediction_performed']
            score = float(result['score'])
            assert int(score >= .5) == result['prediction']
        else:
            score = float(d['hgb'][i])
        indices.append(i); scores.append(score); status.append(mechanism)
    assert len(indices) == len(set(indices)) == 300
    assert status.count('rule_prediction') == 100
    result = np.full(len(d['y']), np.nan)
    statuses = np.full(len(d['y']), 'outside_fixed300', dtype=object)
    result[indices] = scores
    statuses[indices] = status
    return dict(indices=np.asarray(indices), rule_mask=np.asarray([v == 'rule_prediction' for v in status]),
        scores=result, statuses=statuses, metadata=dict(mechanism_counts=doc['derived_mechanism_counts'],
        original_rule_outputs_reused=100, hgb_fixed_fallbacks=200))

def training_weights(d, ids):
    ws = 1./d['group_size'][ids]
    W = math.fsum(ws)
    C = [math.fsum(float(w) for w,y in zip(ws,d['y'][ids]) if y == v) for v in (0,1)]
    assert min(C) > 0
    wc = np.asarray([float(w)*(W/(2.*C[int(y)])) for w,y in zip(ws,d['y'][ids])])
    return ws, wc, W, C

def kb20(columns):
    if not columns:
        return [0.]*20
    def col(j): return [r[j] for r in columns]
    def mean(j): return math.fsum(col(j))/len(columns)
    p = mean(2)
    return [1.,float(len(columns)),mean(0),min(col(0)),max(col(0)),
        mean(1),min(col(1)),max(col(1)),p,
        math.sqrt(math.fsum((v-p)**2 for v in col(2))/len(columns)),min(col(2)),max(col(2)),
        mean(3),mean(4),mean(5),mean(6),mean(7),max(col(7)),mean(8),max(col(8))]

def load_kb_features(d, scope, path, outer_bank=False):
    doc = json.loads(path.read_text(encoding='utf-8'))
    if outer_bank:
        bank = doc['finalized_bank']
    else:
        assert doc['stage'] == 'finalized' and doc['operation_id'] == 'inner-bank:'+scope['scope_sha256']
        bank = doc['payload']['finalized_bank']
    ids = scope['indices']
    partition = bank['partition']
    assert sorted(r['source_index'] for r in partition['TRAIN_rows']) == scope['original_source_indices']
    ws,wc,W,C = training_weights(d,ids)
    assert np.allclose(C,partition['TRAIN_class_ws_mass'],rtol=1e-12,atol=1e-12)
    C = partition['TRAIN_class_ws_mass']
    N,G = len(ids),len(set(d['group'][ids]))
    context = [math.log1p(N),math.log1p(G),math.log1p(W),C[1]/W,min(C)/W,1/math.sqrt(G)]
    envelopes = sorted(bank['rule_envelopes'],key=lambda r:(-r['card']['training_group_support']['n_rows'],r['card']['rule_id']))
    rows = [[] for _ in range(len(d['y']))]
    train_groups = set(d['group'][ids])
    query_ids = np.asarray([i for i,g in enumerate(d['group']) if g not in train_groups],dtype=int)
    for rule in envelopes:
        card,stats = rule['card'],rule['statistics']
        support = card['training_group_support']
        U0,U1 = stats['matched_class_ws_mass']
        A0,A1 = U0*(W/(2*C[0])), U1*(W/(2*C[1]))
        p = (A1+1.)/((A0+A1)+2.)
        atoms = card['applicability']['all']
        raw = [float(support['n_rows']),float(support['n_groups']),p,U0,U1,A0,A1,
               float(len(card['counterexamples'])),float(len(atoms))]
        n,ng,p,U0,U1,A0,A1,nc,na = raw
        norm = [n/N,ng/G,p,U0/C[0],U1/C[1],math.fsum((U0,U1))/W,
                math.fsum((A0,A1))/W,nc/N,na/8.]
        applicable = np.ones(len(query_ids),dtype=bool)
        for atom in atoms:
            v = d['X127'][query_ids,atom['feature_index']]
            applicable &= v <= atom['threshold'] if atom['operator'] == '<=' else v > atom['threshold']
        for i in query_ids[applicable]:
            if len(rows[i]) < 6:
                rows[i].append((raw,norm))
    features = np.full((len(d['y']),46),np.nan)
    for i in query_ids:
        raw = [r[0] for r in rows[i]]
        norm = [r[1] for r in rows[i]]
        nkb = kb20(norm)
        if norm:
            nkb[1] /= 6.
        features[i] = kb20(raw)+nkb+context
    return features,dict(path=str(path),scope=scope['name'],n_rows=N,n_groups=G,n_rules=len(envelopes))

def weighted_metric(d, ids, scores):
    pred = scores >= .5
    y = d['y'][ids]
    ws = 1./d['group_size'][ids]
    mass = [[math.fsum(float(w) for w,t,p in zip(ws,y,pred) if t==a and p==b)
             for b in (0,1)] for a in (0,1)]
    ba = .5*(mass[0][0]/math.fsum(mass[0])+mass[1][1]/math.fsum(mass[1]))
    denominator = math.fsum((mass[0][1],mass[1][1]))
    precision = mass[1][1]/denominator if denominator else 0.
    return dict(balanced_accuracy=ba,precision=precision)

def choose(d, ids, candidates):
    metrics = [weighted_metric(d,ids,p) for p in candidates]
    index = min(range(len(metrics)),key=lambda j:(-metrics[j]['balanced_accuracy'],-metrics[j]['precision'],j))
    return index,metrics

def lr_transform(train, query, ws):
    rows = train.tolist()
    weights = ws.tolist()
    W = math.fsum(weights)
    mean = np.asarray([math.fsum(w*row[j] for w,row in zip(weights,rows))/W for j in range(train.shape[1])])
    var = np.asarray([math.fsum(w*(row[j]-mean[j])**2 for w,row in zip(weights,rows))/W for j in range(train.shape[1])])
    scale = np.where(var>0,np.sqrt(var),1.)
    return (train-mean)/scale,(query-mean)/scale

def run_strong(d, callback, log):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    programme = json.loads(PROGRAMME.read_text(encoding='utf-8'))
    specs = {(s['family'],s['configuration_index']):s for s in programme['native_spec_catalogue']}
    counts = {f:sum(s['family']==f for s in programme['native_spec_catalogue']) for f in FAMILIES}
    constructors = dict(HistGradientBoostingClassifier=HistGradientBoostingClassifier,
        RandomForestClassifier=RandomForestClassifier,LogisticRegression=LogisticRegression)
    all_scopes = {s['name']:s for s in scopes(d)}
    inner_paths = find_inner_banks(d)
    kb = {}
    bank_metadata = []
    for name,scope in all_scopes.items():
        is_outer = name.startswith('outerTRAIN')
        path = BANKS/('fold'+str(scope['outer_fold']))/'ROOT_REVIEWABLE_BANK.json' if is_outer else inner_paths[scope['scope_sha256']]
        log('Loading scientific rule bank %s',name)
        kb[name],meta = load_kb_features(d,scope,path,is_outer)
        bank_metadata.append(meta)
    sf = np.asarray([subfold(g) for g in d['group']])
    def matrix(family, ids, banks):
        X = d['X969'][ids]
        if family != 'D_COMMON969':
            K = np.asarray([kb[banks[int(i)]][i] for i in ids])
            X = np.concatenate([X,K],axis=1)
        assert np.isfinite(X).all()
        return X
    fallback_models = {}
    fallback_scores = {}
    def fallback(train, query):
        # The original programme rejects unconverged LR and resolves it with a
        # TRAIN-only X127 HGB for inner folds, original HGB127 for outer HELD.
        tf = np.unique(d['fold'][train])
        if len(tf) != 1:
            return d['hgb'][query]
        tf = int(tf[0])
        ef = int(np.unique(d['fold'][query])[0])
        if tf not in fallback_models:
            spec = specs[('INNER_RAW_X127_FALLBACK',0)]
            model = constructors[spec['native_classifier_kind']](**spec['parameters'])
            _,wc,_,_ = training_weights(d,train)
            log('Fit inner raw X127 convergence fallback trainfold=%d',tf)
            with threadpool_limits(limits=8):
                model.fit(d['X127'][train],d['y'][train],sample_weight=wc)
            fallback_models[tf] = model
            metadata['fallback_native_fit_count'] += 1
        if (tf,ef) not in fallback_scores:
            with threadpool_limits(limits=8):
                fallback_scores[(tf,ef)] = fallback_models[tf].predict_proba(d['X127'][query])[:,1]
        return fallback_scores[(tf,ef)]
    def fit_predict(family,cfg,train,query,bank_train,bank_query):
        Xt,Xq = matrix(family,train,bank_train),matrix(family,query,bank_query)
        ws,wc,_,_ = training_weights(d,train)
        if family == 'LR_SHARED1015':
            Xt,Xq = lr_transform(Xt,Xq,ws)
        spec = specs[(family,cfg)]
        model = constructors[spec['native_classifier_kind']](**spec['parameters'])
        log('Fit %s config=%d n_train=%d n_query=%d',family,cfg,len(train),len(query))
        with threadpool_limits(limits=8):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model.fit(Xt,d['y'][train],sample_weight=wc)
            for warning in caught:
                log('Native warning %s: %s',warning.category.__name__,str(warning.message))
            if family == 'LR_SHARED1015' and np.any(model.n_iter_ >= spec['parameters']['max_iter']):
                rejection = dict(family=family,configuration_index=cfg,n_train=len(train),
                    query_fold=int(np.unique(d['fold'][query])[0]),n_iter=model.n_iter_.tolist())
                metadata['convergence_rejections'].append(rejection)
                log('Reject unconverged LR %s',rejection)
                return fallback(train,query),'fixed_fallback_convergence_rejected'
            score = model.predict_proba(Xq)[:,1]
        return score,'recomputed_nested_oof'
    result = np.full(len(d['y']),np.nan)
    metadata = dict(families=list(FAMILIES),rule_bank_inputs=bank_metadata,choices=[],
        ensemble_weights_candidates=WEIGHTS.tolist(),ensemble_choices=[],native_fit_count=0,
        fallback_native_fit_count=0,convergence_rejections=[],lr_convergence_policy='reject_and_retain',
        frozen_programme=str(PROGRAMME),protocol='Recomputed frozen four-family nested group OOF; no scientific LLM')
    for outer in range(3):
        train = np.flatnonzero(d['fold']!=outer)
        held = np.flatnonzero(d['fold']==outer)
        train_pos = {int(i):j for j,i in enumerate(train)}
        components_inner,components_held = [],[]
        bank_train_final = {int(i):'outer'+str(outer)+'_innerTRAIN'+str(next(f for f in range(3) if f!=outer and f!=d['fold'][i])) for i in train}
        bank_query_final = {int(i):'outerTRAIN'+str(outer) for i in held}
        for family in FAMILIES:
            candidates = [np.full(len(train),np.nan) for _ in range(counts[family])]
            for tf in range(3):
                if tf==outer: continue
                ef=next(f for f in range(3) if f!=outer and f!=tf)
                inner_train = np.flatnonzero(d['fold']==tf)
                inner_query = np.flatnonzero(d['fold']==ef)
                name = 'outer'+str(outer)+'_innerTRAIN'+str(tf)
                bt = {int(i):name+'_exclude_subfold'+str(sf[i]) for i in inner_train}
                bq = {int(i):name for i in inner_query}
                positions = [train_pos[int(i)] for i in inner_query]
                for cfg in range(counts[family]):
                    candidates[cfg][positions],status = fit_predict(family,cfg,inner_train,inner_query,bt,bq)
                    metadata['native_fit_count'] += 1
            assert all(np.isfinite(v).all() for v in candidates)
            selected,metrics = choose(d,train,candidates)
            metadata['choices'].append(dict(outer_fold=outer,family=family,selected_config=selected,inner_metrics=metrics))
            components_inner.append(candidates[selected])
            score,status = fit_predict(family,selected,train,held,bank_train_final,bank_query_final)
            metadata['native_fit_count'] += 1
            components_held.append(score)
            callback(family,held,score,[status]*len(held))
        inner = np.asarray(components_inner)
        candidates = [np.asarray([math.fsum(float(w)*float(v) for w,v in zip(weight,row)) for row in inner.T]) for weight in WEIGHTS]
        selected,metrics = choose(d,train,candidates)
        weight = WEIGHTS[selected]
        metadata['ensemble_choices'].append(dict(outer_fold=outer,weights_index=selected,weights=weight.tolist(),inner_metrics=metrics))
        score = np.asarray([math.fsum(float(w)*float(v) for w,v in zip(weight,row)) for row in np.asarray(components_held).T])
        result[held] = score
        callback('StrongML',held,score,['recomputed_nested_oof']*len(held))
        log('Completed strong outer fold %d selected_weight=%d',outer,selected)
    assert np.isfinite(result).all() and metadata['native_fit_count']==78
    return dict(scores=result,metadata=metadata)
