"""G7 paired clustered statistics and independent-cohort change controls.

No I/O or model calls on import. New cohorts use frozen specific-parent groups;
the historical3764 cohort retains its original component grouping. All claims
are descriptive; the historical cohort is not verified entirely pre-cutoff.
"""
from __future__ import annotations
from common import np
from f_stats import summarize,_metric,_interval
from f3_stats import build_f3_statistics

SEED=20261019;N_BOOT=5000
METHODS=('Local_A3','GPT_A3','HGB_composition118')
METRICS=('balanced_accuracy','precision','recall','accuracy')

def historical_stratified_appendix(old,family_masks,group_size_values,raw_a15_mask=None):
    """One shared old-cohort appendix; new69/89 cohort eligibility is irrelevant."""
    groups=np.asarray(old['groups']).astype(str)
    sizes=old.get('full_group_sizes') or {str(k):int(v) for k,v in zip(*np.unique(groups,return_counts=True))}
    full=build_f3_statistics(old['y'],old['scores'],groups,sizes,
        family_masks={k:np.asarray(v,bool) for k,v in family_masks.items()},
        group_size_values=np.asarray(group_size_values),
        a15_mask=None if raw_a15_mask is None else np.asarray(raw_a15_mask,bool),
        n_boot=N_BOOT,seed=SEED)
    return {k:full[k] for k in ('method_order','report_pairs','families','group_size_bins','coverage','statistical_protocol','a15_component93_overlap')}

def cohort_draws(y,scores,groups,*,seed,full_sizes=None):
    y=np.asarray(y,int);groups=np.asarray(groups).astype(str)
    assert y.ndim==1 and groups.shape==y.shape and np.all(np.isin(y,[0,1]))
    names=list(METHODS);arr={k:np.asarray(scores[k],float) for k in names}
    assert all(v.shape==y.shape and np.all((v>=0)&(v<=1)) for v in arr.values())
    obs,inv,count=np.unique(groups,return_inverse=True,return_counts=True);ng=len(obs)
    assert ng>0
    sizes={str(g):int(c) for g,c in zip(obs,count)} if full_sizes is None else {str(k):int(v) for k,v in full_sizes.items()}
    assert all(sizes[g]>=c for g,c in zip(obs,count))
    w=np.stack(([1./sizes[g] for g in groups],np.ones(len(y))),axis=1)
    conf=np.zeros((ng,len(names),2,4),float)
    for j,name in enumerate(names):
        cells=2*y+(arr[name]>=.5).astype(int)
        for k in range(2):np.add.at(conf[:,j,k,:],(inv,cells),w[:,k])
    point=_metric(conf.sum(0));boot_conf=np.empty((N_BOOT,len(names),2,4),float);rng=np.random.default_rng(seed)
    for st in range(0,N_BOOT,256):
        en=min(N_BOOT,st+256);draw=rng.multinomial(ng,np.full(ng,1/ng),size=en-st)
        boot_conf[st:en]=(draw@conf.reshape(ng,-1)).reshape(-1,len(names),2,4)
    return point,_metric(boot_conf),sizes

def independent_cohort_controls(old,new):
    """Separate independent group draws for old and new, shared methods within each."""
    p0,b0,s0=cohort_draws(old['y'],old['scores'],old['groups'],seed=SEED,full_sizes=old.get('full_group_sizes'))
    p1,b1,s1=cohort_draws(new['y'],new['scores'],new['groups'],seed=SEED+1,full_sizes=new.get('full_group_sizes'))
    out={'definition':'new minus historical3764; different cohorts independently bootstrap their own groups, while all methods/weights within each cohort share draws',
         'old_n':len(old['y']),'new_n':len(new['y']),'old_groups':len(np.unique(old['groups'])),'new_groups':len(np.unique(new['groups'])),'old_positive':int(np.asarray(old['y']).sum()),'new_positive':int(np.asarray(new['y']).sum()),'changes':{},'difference_in_differences':{}}
    for wi,weight in enumerate(('group_weighted','unweighted')):
        out['changes'][weight]={};out['difference_in_differences'][weight]={}
        for j,name in enumerate(METHODS):
            out['changes'][weight][name]={metric:_interval(p1[metric][j,wi]-p0[metric][j,wi],b1[metric][:,j,wi]-b0[metric][:,j,wi]) for metric in METRICS}
        for j,name in enumerate(METHODS[:2]):
            out['difference_in_differences'][weight][name+' vs HGB_composition118']={metric:_interval(
                (p1[metric][j,wi]-p0[metric][j,wi])-(p1[metric][2,wi]-p0[metric][2,wi]),
                (b1[metric][:,j,wi]-b0[metric][:,j,wi])-(b1[metric][:,2,wi]-b0[metric][:,2,wi])) for metric in METRICS}
    return out

def build_g7_statistics(old,new,primary_eligible_mask=None):
    assert all(set(c['scores'])==set(METHODS) for c in (old,new))
    oldg=np.asarray(old['groups']).astype(str);newg=np.asarray(new['groups']).astype(str)
    new_sizes={str(k):int(v) for k,v in zip(*np.unique(newg,return_counts=True))}
    old_sizes=old.get('full_group_sizes') or {str(k):int(v) for k,v in zip(*np.unique(oldg,return_counts=True))}
    current=summarize(new['y'],new['scores'],newg,new_sizes,n_boot=N_BOOT,seed=SEED)
    yy=np.asarray(new['y']);current.update(n_positive_groups=int(len(np.unique(newg[yy==1]))),n_negative_groups=int(len(np.unique(newg[yy==0]))),bootstrap_interval_boundary='Percentile interval is conditional on finite metric draws; if a resample lacks a class, BA is undefined and excluded, with counts explicitly retained.')
    historical={s:summarize(old['y'],old['scores'],oldg,old_sizes,mask=mask,n_boot=N_BOOT,seed=SEED) for s,mask in {'all':np.ones(len(oldg),bool),'exclude_93':oldg!='93','only_93':oldg=='93'}.items()}
    ex=oldg!='93';oldex={**old,'y':np.asarray(old['y'])[ex],'groups':oldg[ex],'scores':{k:np.asarray(v)[ex] for k,v in old['scores'].items()},'full_group_sizes':old_sizes}
    result={'status':'complete','n_boot':N_BOOT,'seed':SEED,'new_all':current,'historical3764':historical,
            'cohort_change_controls':independent_cohort_controls({**old,'full_group_sizes':old_sizes},{**new,'full_group_sizes':new_sizes}),
            'cohort_change_controls_old_exclude93_sensitivity':independent_cohort_controls(oldex,{**new,'full_group_sizes':new_sizes}),
            'new_group_definition':'Frozen specific-parent family ID; weight=1/new group size, unit-material stats also bootstrap same groups',
            'historical_group_definition':'Original3764 component_id; weight=1/full original component size',
            'boundaries':['Historical3764 records are not all independently verified to have first appeared before the local model cutoff.',
                          'Independent-cohort difference-in-differences is descriptive and cannot fully remove changes in material difficulty, model capacity, structural information availability, or family composition.',
                          'All multiple comparisons are descriptive; no multiplicity-adjusted confirmatory claim is made.',
                          'New-data small sample and few parent families can yield wide or undefined bootstrap intervals; finite/undefined draw counts retained.']}
    if primary_eligible_mask is not None:
        mask=np.asarray(primary_eligible_mask,bool);assert mask.shape==newg.shape and mask.any()
        sub={**new,'y':np.asarray(new['y'])[mask],'groups':newg[mask],'scores':{k:np.asarray(v)[mask] for k,v in new['scores'].items()}}
        sub.pop('full_group_sizes',None) # strict primary uses 1/eligible specific-parent group size
        strict=build_g7_statistics(old,sub)
        result['frozen_full_dataset_sensitivity']={'n':len(newg),'n_positive':int(np.asarray(new['y']).sum()),'n_groups':len(np.unique(newg)),'definition':'All frozen evaluated cases retained regardless of post-freeze identity/date corrections; predictions not used to screen eligibility.'}
        result['strict_primary_eligible']=strict
        result['eligibility_correction_boundary']='Strict primary subset is defined by independent date/formula source review after the89-case input freeze, blind to all model predictions; full frozen set retained as sensitivity. Group weighting renormalized by actual eligible specific-parent group size, separately from frozen-full sensitivity.'
    return result
