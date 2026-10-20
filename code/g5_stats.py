"""G5 exact group-bootstrap multi-threshold classification and regression."""
from common import np
from f_stats import summarize, fraction_summary
from f3_stats import build_f3_statistics
from itertools import combinations

SUBSETS={'all':lambda g:np.ones(len(g),dtype=bool),'exclude_93':lambda g:g!='93','only_93':lambda g:g=='93'}
SCHEMES=('group_weighted','unweighted')
THRESHOLDS=(5,10,15)

def _interval(point,draws):
    draws=np.asarray(draws,dtype=float)
    finite=draws[np.isfinite(draws)]
    return {'point':float(point) if np.isfinite(point) else None,
            'ci95':np.percentile(finite,[2.5,97.5]).tolist() if len(finite) else [None,None],
            'n_valid_boot':len(finite),'n_undefined_boot':len(draws)-len(finite)}

def rank_plan(values):
    x=np.asarray(values,dtype=float)
    order=np.argsort(x,kind='stable')
    starts=np.r_[0,np.flatnonzero(np.diff(x[order])!=0)+1]
    lengths=np.diff(np.r_[starts,len(x)])
    return order,starts,lengths

def weighted_midranks(weights,plan):
    """Weighted mid-CDF positions (unscaled); weight-zero rows never contribute."""
    w=np.atleast_2d(np.asarray(weights,dtype=float))
    order,starts,lengths=plan
    masses=np.add.reduceat(w[:,order],starts,axis=1)
    blocks=np.cumsum(masses,axis=1)-.5*masses
    result=np.empty_like(w)
    result[:,order]=np.repeat(blocks,lengths,axis=1)
    return result

def spearman_batch(y,pred,w,yplan=None,pplan=None):
    w=np.atleast_2d(np.asarray(w,dtype=float))
    ry=weighted_midranks(w,rank_plan(y) if yplan is None else yplan)
    rp=weighted_midranks(w,rank_plan(pred) if pplan is None else pplan)
    total=w.sum(axis=1)
    safe=np.where(total>0,total,1.)
    cy=ry-(w*ry).sum(axis=1,keepdims=True)/safe[:,None]
    cp=rp-(w*rp).sum(axis=1,keepdims=True)/safe[:,None]
    numerator=(w*cy*cp).sum(axis=1)
    denominator=np.sqrt((w*cy*cy).sum(axis=1)*(w*cp*cp).sum(axis=1))
    return np.divide(numerator,denominator,out=np.full(len(w),np.nan),where=denominator>0)

def regression_summary(tc,preds,groups,sizes,mask,*,n_boot=5000,seed=20261018):
    if n_boot!=5000: raise ValueError('Exactly5000 group bootstrap draws required')
    tc=np.asarray(tc,dtype=float); groups=np.asarray(groups).astype(str); mask=np.asarray(mask,dtype=bool)
    arrays={name:np.asarray(v,dtype=float) for name,v in preds.items()}
    valid=np.logical_and.reduce([np.isfinite(v)&(v>=0) for v in arrays.values()])
    selected=mask&valid
    names=list(arrays); y=tc[selected]
    observed,inv=np.unique(groups[selected],return_inverse=True)
    n,g=len(y),len(observed)
    output={'n_requested':int(mask.sum()),'n_requested_groups':len(np.unique(groups[mask])),
            'n':n,'n_groups':g,'n_missing':int(np.sum(mask&~valid)),
            'positive_group_description_threshold_K':10,'n_positive_groups':len(np.unique(groups[selected&(tc>10)])),
            'n_negative_groups':len(np.unique(groups[selected&(tc<=10)])),
            'n_boot':n_boot,'seed':seed,'bootstrap_unit':'observed original component groups, sampled uniformly with replacement',
            'one_group_CI_degenerate':g==1,
            'Spearman_definition':{'unweighted':'Pearson correlation of ordinary average ranks in the expanded group-resampled sample; ties receive average ranks.',
                                   'group_weighted':'Weighted Pearson correlation of weighted mid-CDF ranks, using1/full original group size per observation and group multiplicity; ranks recomputed inside every bootstrap draw.'}}
    if not n:
        empty=_interval(np.nan,np.full(n_boot,np.nan))
        for scheme in SCHEMES:
            output[scheme]={'metrics':{name:{'MAE_K':dict(empty),'Spearman':dict(empty)} for name in names},
                            'paired_differences':{a+' - '+b:{'arm_a':a,'arm_b':b,'MAE_K':dict(empty),'Spearman':dict(empty)} for a,b in combinations(names,2)}}
        return output
    original=np.asarray([1./sizes[str(v)] for v in groups[selected]])
    P=np.asarray([arrays[name][selected] for name in names])
    E=np.abs(P-y)
    unit=np.column_stack((original,np.ones(n)))
    denom=np.zeros((g,2)); numer=np.zeros((g,len(names),2))
    for scheme in range(2):
        np.add.at(denom[:,scheme],inv,unit[:,scheme])
        for j in range(len(names)): np.add.at(numer[:,j,scheme],inv,E[j]*unit[:,scheme])
    point_mae=numer.sum(axis=0)/denom.sum(axis=0)[None,:]
    point_spear=np.asarray([[spearman_batch(y,P[j],unit[:,s])[0] for s in range(2)] for j in range(len(names))])
    boot_mae=np.full((n_boot,len(names),2),np.nan); boot_spear=np.full_like(boot_mae,np.nan)
    if g==1:
        boot_mae[:]=point_mae; boot_spear[:]=point_spear
    else:
        rng=np.random.default_rng(seed)
        yp=rank_plan(y); pp=[rank_plan(p) for p in P]
        for start in range(0,n_boot,128):
            stop=min(start+128,n_boot)
            draws=rng.multinomial(g,np.full(g,1./g),size=stop-start)
            D=draws@denom
            N=(draws@numer.reshape(g,-1)).reshape(-1,len(names),2)
            boot_mae[start:stop]=N/D[:,None,:]
            for s in range(2):
                w=draws[:,inv]*unit[:,s][None,:]
                for j in range(len(names)): boot_spear[start:stop,j,s]=spearman_batch(y,P[j],w,yp,pp[j])
    for s,scheme in enumerate(SCHEMES):
        output[scheme]={'weight_sum':float(unit[:,s].sum()),
            'metrics':{name:{'MAE_K':_interval(point_mae[j,s],boot_mae[:,j,s]),
                             'Spearman':_interval(point_spear[j,s],boot_spear[:,j,s])} for j,name in enumerate(names)},
            'paired_differences':{a+' - '+b:{'arm_a':a,'arm_b':b,
                'MAE_K':_interval(point_mae[names.index(a),s]-point_mae[names.index(b),s],boot_mae[:,names.index(a),s]-boot_mae[:,names.index(b),s]),
                'Spearman':_interval(point_spear[names.index(a),s]-point_spear[names.index(b),s],boot_spear[:,names.index(a),s]-boot_spear[:,names.index(b),s])} for a,b in combinations(names,2)}}
    return output

def build_statistics(cfg,d,responses,log=lambda s:None):
    groups=d['group'].astype(str); sizes={str(g):int(n) for g,n in zip(groups,d['group_size'])}
    n=len(groups)
    llm_tc=np.full(n,np.nan); ps={t:np.full(n,np.nan) for t in THRESHOLDS}
    for case in cfg['cases']:
        obj=responses.get(case['id'])
        if obj is not None:
            i=case['i']; llm_tc[i]=obj['Tc_est_K']
            for t in THRESHOLDS: ps[t][i]=obj[f'p_gt_{t}K']
    reg=np.asarray(cfg['HGB_regressor_Tc_K'])
    result={'status':'statistical_finalization','classification':{},'regression':{},'agreement_original_A3_10K':{},
            'scope':'development only; original3764cohort, no external validation',
            'protocol':{'n_boot':5000,'seed':20261018,'original_group_weights_retained':True,
                        'strict_temperature_thresholds_K':list(THRESHOLDS),'probability_cutoff':.5,
                        'LLM_missing_outputs':'not imputed; common-valid paired denominators',
                        'point_and_probability_outputs':'evaluated separately; no enforced consistency beyond monotone probabilities'}}
    masks={name:func(groups) for name,func in SUBSETS.items()}
    families={name:np.asarray(v,dtype=bool) for name,v in cfg['family_masks'].items()}
    g3families={name:np.asarray(v,dtype=bool) for name,v in cfg['G3_family_masks'].items()}
    sizebins={'1':d['group_size']==1,'2-5':(d['group_size']>=2)&(d['group_size']<=5),
              '6-20':(d['group_size']>=6)&(d['group_size']<=20),'>20':d['group_size']>20}
    collections={'subsets':{'cohort':np.ones(n,dtype=bool)},'F3_families':families,'G3_families':g3families,'group_size_bins':sizebins}
    for t in THRESHOLDS:
        y=(d['tc']>t).astype(int)
        scores={'A3_reg_direct':ps[t], 'A3_reg_from_Tc':np.where(np.isfinite(llm_tc),(llm_tc>t).astype(float),np.nan),
                'HGB_classifier':np.asarray(cfg['ML_classifier_scores'][f'HGB_classifier_gt_{t}K']),
                'HGB_regressor_from_Tc':(reg>t).astype(float)}
        if t==10: scores['A3_original_10K']=np.asarray(cfg['original_A3_10K'])
        main=build_f3_statistics(y,scores,groups,sizes,family_masks=families,a15_mask=np.asarray(cfg['raw_a15_mask'],dtype=bool),
                                 group_size_values=d['group_size'],n_boot=5000,seed=20261018)
        main['G3_families']={}
        valid=np.logical_and.reduce([np.isfinite(v) for v in scores.values()])
        def annotate(stat,mask):
            stat.update(n_positive_groups=len(np.unique(groups[mask&valid&(y==1)])),n_negative_groups=len(np.unique(groups[mask&valid&(y==0)])))
        for name,sub in masks.items():annotate(main['subsets'][name],sub)
        for family,selected in families.items():
            for name,sub in masks.items():annotate(main['families'][family][name],selected&sub)
        for label,selected in sizebins.items():
            for name,sub in masks.items():annotate(main['group_size_bins'][label][name],selected&sub)
        for family,selected in g3families.items():
            main['G3_families'][family]={}
            for name,sub in masks.items():
                requested=selected&sub
                a=summarize(y,scores,groups,sizes,requested&valid,n_boot=5000,seed=20261018)
                a.update(n_requested=int(requested.sum()),n_requested_groups=len(np.unique(groups[requested])),n_missing=int(np.sum(requested&~valid)))
                annotate(a,requested)
                main['G3_families'][family][name]=a
        result['classification'][f'Tc_gt_{t}K']=main
        log(f'STATS classification Tc>{t} complete')
    for kind,partition in collections.items():
        result['regression'][kind]={}
        for label,selected in partition.items():
            container={}
            for name,sub in masks.items():
                scope={}
                for positive_t in (None,*THRESHOLDS):
                    mask=selected&sub
                    if positive_t is not None: mask=mask&(d['tc']>positive_t)
                    title='all_Tc' if positive_t is None else f'Tc_gt_{positive_t}K'
                    scope[title]=regression_summary(d['tc'],{'A3_reg':llm_tc,'HGB_regressor':reg},groups,sizes,mask)
                container[name]=scope
            result['regression'][kind][label]=container
        log(f'STATS regression {kind} complete')
    old=np.asarray(cfg['original_A3_10K'])>=.5
    agreement_vectors={
        'direct_probability':np.where(np.isfinite(ps[10]),((ps[10]>=.5)==old).astype(float),np.nan),
        'point_estimate':np.where(np.isfinite(llm_tc),((llm_tc>10)==old).astype(float),np.nan)}
    for name,sub in masks.items():
        result['agreement_original_A3_10K'][name]={key:fraction_summary(value,groups,sizes,sub,n_boot=5000,seed=20261018) for key,value in agreement_vectors.items()}
    result['agreement_original_A3_10K_G3_families']={family:{
        name:{key:fraction_summary(value,groups,sizes,selected&sub,n_boot=5000,seed=20261018) for key,value in agreement_vectors.items()}
        for name,sub in masks.items()} for family,selected in g3families.items()}
    result['coverage']={'requested_n':n,'requested_groups':len(np.unique(groups)),'valid_LLM':int(np.isfinite(llm_tc).sum()),
                        'ML_all3764':True}
    return result
