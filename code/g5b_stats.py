"""Repeatability statistics for G5b (three fresh LLM arms on the G4 sample)."""
from __future__ import annotations
from itertools import combinations
from common import np
from f3_stats import build_f3_statistics
from f_stats import fraction_summary
from f_stats import summarize

SEED = 20261019
N_BOOT = 5000

def _metric(conf):
    tn, fp, fn, tp = np.moveaxis(conf, -1, 0)
    def div(a,b,zero=np.nan): return np.divide(a,b,out=np.full(np.shape(a),zero,dtype=float),where=b>0)
    return {'balanced_accuracy':(div(tp,tp+fn)+div(tn,tn+fp))/2,
            'precision':div(tp,tp+fp,0.0),'recall':div(tp,tp+fn),
            'accuracy':div(tp+tn,tp+tn+fp+fn)}

def _exceed(y, scores, groups, full_sizes, mask):
    """Shared group bootstrap probability |A3-A3reg|>|A3_1-A3_2| for BA."""
    selected=np.asarray(mask,bool)&np.logical_and.reduce([np.isfinite(v) for v in scores.values()])
    if not selected.any(): return {k:None for k in ('group_weighted','unweighted')}
    g=np.asarray(groups).astype(str); yy=np.asarray(y)[selected]; gg=g[selected]
    names=list(scores); vals=[np.asarray(scores[n])[selected] for n in names]
    obs, inv=np.unique(gg,return_inverse=True); ng=len(obs)
    sizes={str(k):int(v) for k,v in full_sizes.items()}
    w=np.stack(([1/sizes[z] for z in gg],np.ones(len(gg))),axis=1)
    # order: A3_rep1, A3_rep2, A3_reg
    conf=np.zeros((ng,3,2,4),float)
    for j,v in enumerate(vals):
        cell=2*yy.astype(int)+(v>=.5).astype(int)
        for s in range(2): np.add.at(conf[:,j,s,:],(inv,cell),w[:,s])
    point=_metric(conf.sum(0))['balanced_accuracy']
    rng=np.random.default_rng(SEED); counts=np.zeros((N_BOOT,3,2,4),float)
    for st in range(0,N_BOOT,256):
        en=min(st+256,N_BOOT); draws=rng.multinomial(ng,np.full(ng,1/ng),size=en-st)
        counts[st:en]=(draws@conf.reshape(ng,-1)).reshape(-1,3,2,4)
    ba=_metric(counts)['balanced_accuracy']
    out={}
    for s,key in enumerate(('group_weighted','unweighted')):
        within=np.abs(ba[:,0,s]-ba[:,1,s]); cross=np.abs(ba[:,2,s]-ba[:,0,s]); margin=cross-within
        point_margin=abs(point[2,s]-point[0,s])-abs(point[0,s]-point[1,s])
        finite=np.isfinite(margin)
        point_ex=point_margin>0; draw=margin>0
        out[key]={'point_A3_minus_A3reg':float(point[0,s]-point[2,s]),
                  'point_A3_repeat_difference':float(point[0,s]-point[1,s]),
                  'point_abs_A3_vs_reg_exceeds_repeat':bool(point_ex),
                  'bootstrap_fraction_abs_A3_vs_reg_exceeds_repeat':float(np.mean(draw[finite])) if finite.any() else None,
                  'point_abs_cross_minus_abs_within_BA':float(point_margin),
                  'ci95_abs_cross_minus_abs_within_BA':np.percentile(margin[np.isfinite(margin)],[2.5,97.5]).tolist(),
                  'n_valid_boot':int(finite.sum()),'n_undefined_boot':int((~finite).sum()),
                  'n_boot':N_BOOT,'n_groups':ng,'n':int(selected.sum())}
    return out

def difference_in_differences(y,scores,groups,full_sizes,mask):
    """(fresh A3 - fresh reg) - (historical A3 - historical reg), shared draws."""
    g=np.asarray(groups).astype(str); selected=np.asarray(mask,bool)
    if not selected.any():return {w:None for w in ('group_weighted','unweighted')}
    yy=np.asarray(y)[selected];gg=g[selected]; obs,inv=np.unique(gg,return_inverse=True);ng=len(obs)
    names=['fresh_A3','fresh_reg','historical_A3','historical_reg']; w=np.stack(([1/full_sizes[z] for z in gg],np.ones(len(gg))),axis=1)
    conf=np.zeros((ng,4,2,4),float)
    for j,name in enumerate(names):
        cell=2*yy.astype(int)+(np.asarray(scores[name])[selected]>=.5).astype(int)
        for s in range(2):np.add.at(conf[:,j,s,:],(inv,cell),w[:,s])
    ba=_metric(conf.sum(0))['balanced_accuracy']; point=(ba[0]-ba[1])-(ba[2]-ba[3])
    rng=np.random.default_rng(SEED);boot=np.empty((N_BOOT,2),float)
    for st in range(0,N_BOOT,256):
        en=min(st+256,N_BOOT);draw=rng.multinomial(ng,np.full(ng,1/ng),size=en-st);q=_metric((draw@conf.reshape(ng,-1)).reshape(-1,4,2,4))['balanced_accuracy'];boot[st:en]=(q[:,0]-q[:,1])-(q[:,2]-q[:,3])
    return {key:{'point':float(point[s]),'ci95':np.percentile(boot[np.isfinite(boot[:,s]),s],[2.5,97.5]).tolist(),'n_valid_boot':int(np.isfinite(boot[:,s]).sum()),'n_undefined_boot':int((~np.isfinite(boot[:,s])).sum()),'n':int(selected.sum()),'n_groups':ng} for s,key in enumerate(('group_weighted','unweighted'))}

def build_statistics(cfg,d,scores,log=None):
    idx=np.asarray(cfg['selected_indices'],int); y=np.asarray(d['y'])[idx]; groups=np.asarray(d['group'])[idx].astype(str)
    sizes={str(g):int(s) for g,s in zip(d['group'],d['group_size'])}
    # scores are aligned to selected G4 order
    names=['A3_repeat_1','A3_repeat_2','A3_reg']
    arr={k:np.asarray(scores[k],float) for k in names}
    masks={'all':np.ones(len(idx),bool),'exclude_93':groups!='93','only_93':groups=='93'}
    out={'protocol':{'n_boot':N_BOOT,'seed':SEED,'weighting':'1/full original component size; unit material','bootstrap':'group bootstrap shared across all methods and paired differences','development_only':True},'subsets':{},'families':{},'exceedance':{},'agreement':{}}
    basefam={k:np.asarray(v,bool) for k,v in cfg['family_masks'].items()}
    canonical=build_f3_statistics(y,arr,groups,sizes,family_masks=basefam,group_size_values=np.asarray(d['group_size'])[idx],n_boot=N_BOOT,seed=SEED)
    for subset,m in masks.items():
        out['subsets'][subset]=canonical['subsets'][subset]
        out['exceedance'][subset]={
            'A3_repeat_1_reference':_exceed(y,arr,groups,sizes,m),
            'A3_repeat_2_reference':_exceed(y,{'A3_repeat_2':arr['A3_repeat_2'],'A3_repeat_1':arr['A3_repeat_1'],'A3_reg':arr['A3_reg']},groups,sizes,m)}
        # Binary consistency and disagreement rates, with group-bootstrap CIs.
        out['agreement'][subset]={}
        for a,b in [('A3_repeat_1','A3_repeat_2'),('A3_repeat_1','A3_reg'),('A3_repeat_2','A3_reg')]:
            vals=np.full(len(idx),np.nan); valid=m&np.isfinite(arr[a])&np.isfinite(arr[b]); vals[valid]=((arr[a][valid]>=.5)==(arr[b][valid]>=.5)).astype(float)
            out['agreement'][subset][a+' = '+b]=fraction_summary(vals,groups,sizes,mask=m,n_boot=N_BOOT,seed=SEED)
    out['families']=canonical['families']; out['group_size_bins']=canonical['group_size_bins']
    out['G3_predicted_families']={}
    g3labels=np.asarray(cfg.get('G3_labels',[]))
    for label in dict.fromkeys(g3labels.tolist()):
        out['G3_predicted_families'][label]={}
        for subset,m in masks.items():
            mm=m&(g3labels==label); common=mm&np.logical_and.reduce([np.isfinite(v) for v in arr.values()])
            out['G3_predicted_families'][label][subset]=summarize(y,arr,groups,sizes,mask=common,n_boot=N_BOOT,seed=SEED)
    return out
