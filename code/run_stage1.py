"""Finite Stage 1 only. Each experiment retains exactly six output files."""
import sys
from pathlib import Path
# Capture startup errors too; scheduled tasks do not inherit the tool's PATH.
_startup_root = Path('C:/Users/28908/Codex/fung-research/v2-diagnosis').resolve()
if len(sys.argv)>1 and sys.argv[1] in ('E0','E1'):
    _startup_out = (_startup_root/sys.argv[1]).resolve()
    assert 'onedrive' not in str(_startup_out).lower() and _startup_out.is_relative_to(_startup_root)
    _startup_out.mkdir(exist_ok=True)
    sys.stderr = (_startup_out/'run.log').open('a',encoding='utf-8',buffering=1)
    sys.stdout = sys.stderr
from common import *
import argparse
import ctypes
import hashlib
import logging
import subprocess
import time
import traceback
from datetime import datetime, timezone
import sklearn
import joblib
from stats import summarize

def group_sizes(d):
    return {str(g): int(s) for g, s in zip(d['group'], d['group_size'])}

def json_safe(obj):
    if isinstance(obj, np.ndarray):
        return [json_safe(x) for x in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj

def proportion_summary(values, groups, sizes, seed=20261016):
    """Both material and group-weighted proportions, cluster percentile CIs."""
    _, gi = np.unique(groups, return_inverse=True)
    ng = int(gi.max()) + 1
    v = np.asarray(values, dtype=float)
    totals = np.bincount(gi)
    hits = np.bincount(gi, weights=v)
    weighted_totals = np.bincount(gi, weights=1/np.asarray(sizes))
    weighted_hits = np.bincount(gi, weights=v/np.asarray(sizes))
    rng = np.random.default_rng(seed)
    raw, weighted = [], []
    for start in range(0, 5000, 250):
        m = rng.multinomial(ng, np.full(ng, 1/ng), size=min(250, 5000-start))
        raw.extend((m@hits)/(m@totals))
        weighted.extend((m@weighted_hits)/(m@weighted_totals))
    return dict(n=len(v), n_groups=ng, n_boot=5000,
        material_fraction=dict(point=float(v.mean()), ci95=np.percentile(raw,[2.5,97.5]).tolist()),
        group_weighted_fraction=dict(point=float(weighted_hits.sum()/weighted_totals.sum()),
            ci95=np.percentile(weighted,[2.5,97.5]).tolist()))

def e1_run(d, writer, log):
    from e1_rules import run_e1
    result = run_e1(d, bank_root=BANKS, progress=writer.append, log=log)
    scores = result['scores']
    scores['HGB127'] = d['hgb']
    writer.append('HGB127', np.arange(len(d['y'])), d['hgb'], ['cached_original_oof']*len(d['y']))
    metrics = summarize(d['y'], scores, d['group'], group_sizes(d))
    diagnostics = json_safe(result['diagnostics'])
    # Locate the explicitly named diagnostic array without interpreting label outcomes.
    max_key = next(k for k in diagnostics if 'max_abs' in k)
    mx = np.asarray(diagnostics[max_key], dtype=float)
    assert mx.shape == d['y'].shape and np.isfinite(mx).all()
    bins = [0,.05,.10,.15,.20,.25,.30,.35,.40,.45,.5000000001]
    counts, _ = np.histogram(mx, bins=bins)
    hist = []
    for a,b,c in zip(bins[:-1],bins[1:],counts):
        ci = proportion_summary((mx>=a)&(mx<b), d['group'], d['group_size'])
        hist.append(dict(lower=a, upper=min(b,.5), count=int(c), **ci))
    metrics['diagnostics'] = diagnostics
    metrics['rule_information_coverage'] = proportion_summary(mx>=.15,d['group'],d['group_size'])
    metrics['rule_information_coverage']['n_materials_ge_0_15'] = int(np.sum(mx>=.15))
    metrics['max_rule_deviation_histogram'] = hist
    metrics['max_rule_deviation_quantiles'] = {str(q):float(np.quantile(mx,q)) for q in [0,.25,.5,.75,.9,.95,1]}
    return metrics

def e0_run(d, writer, log):
    from e0_models import run_strong, load_agent
    strong_result = run_strong(d, callback=writer.append, log=log)
    agent = load_agent(d)
    # e0_models interface is frozen once sources are committed.
    scores = dict(Agent=agent['scores'], HGB127=d['hgb'], StrongML=strong_result['scores'])
    ix = np.asarray(agent['indices'], dtype=int)
    rule = np.asarray(agent['rule_mask'], dtype=bool)
    assert len(ix)==300 and int(rule.sum())==100
    subset = np.zeros(len(d['y']),dtype=bool); subset[ix]=True
    rule_mask = np.zeros(len(d['y']),dtype=bool); rule_mask[ix[rule]]=True
    fallback = subset & ~rule_mask
    writer.append('Agent',ix,scores['Agent'][ix],np.asarray(agent['statuses'])[ix])
    writer.append('HGB127',ix,d['hgb'][ix],['cached_original_oof']*len(ix))
    metrics = {name:summarize(d['y'],scores,d['group'],group_sizes(d),mask=mask)
        for name,mask in [('all300',subset),('rule100',rule_mask),('fallback200',fallback)]}
    v=scores['Agent'][rule_mask]
    metrics['rule100_score_distribution'] = dict(n=100,n_groups=int(len(np.unique(d['group'][rule_mask]))),
        min=float(v.min()),max=float(v.max()),mean=float(v.mean()),std=float(v.std()),
        quantiles={str(q):float(np.quantile(v,q)) for q in [0,.25,.5,.75,1]},
        all_in_0_488_to_0_510=bool(np.all((v>=.488)&(v<=.510))),
        n_lt_half=int(np.sum(v<.5)),n_eq_half=int(np.sum(v==.5)),n_ge_half=int(np.sum(v>=.5)))
    metrics['strong_ml_selection'] = json_safe(strong_result.get('metadata',{}))
    metrics['agent_metadata'] = json_safe(agent.get('metadata',{}))
    return metrics

def fmt(v):
    p=v['point']; lo,hi=v['ci95']
    return 'undefined' if p is None else f'{100*p:.1f} [{100*lo:.1f},{100*hi:.1f}]'

def readme(experiment, metrics):
    lines=[f'# {experiment}: Stage 1 development results',
        'Group weights = 1/original group size; 5000 association-group bootstrap draws; 95% percentile CI; positive Tc>10K.',
        '|Subset; n/groups|Arm|Balanced accuracy % [CI]|Precision % [CI]|Recall % [CI]|',
        '|---|---|---|---|---|']
    subsets = {'all3764':metrics} if experiment=='E1' else {k:metrics[k] for k in ['all300','rule100','fallback200']}
    for name,sub in subsets.items():
        for arm,m in sub['metrics'].items():
            lines.append(f'|{name}; {sub["n"]}/{sub["n_groups"]}|{arm}|{fmt(m["balanced_accuracy"])}|{fmt(m["precision"])}|{fmt(m["recall"])}|')
    if experiment=='E1':
        cov=metrics['rule_information_coverage']
        lines += [f'Max |rule score-0.5| >=0.15: {cov["n_materials_ge_0_15"]}/3764; material fraction {100*cov["material_fraction"]["point"]:.2f}%; CI in metrics.json.',
            'Histogram of max rule deviation (range:count; bars scale to largest bin):']
        hist=metrics['max_rule_deviation_histogram']; peak=max(x['count'] for x in hist)
        # A single-line SVG plot keeps the experiment to the six mandated files.
        bars=[]
        for j,h in enumerate(hist):
            height=130*h['count']/max(peak,1); x=65+j*53
            bars.append(f'<rect x="{x}" y="{155-height:.1f}" width="40" height="{height:.1f}" fill="#277da1"/><text x="{x}" y="175" font-size="10">{h["lower"]:.2f}</text><text x="{x}" y="{150-height:.1f}" font-size="10">{h["count"]}</text>')
        lines.append('<svg xmlns="http://www.w3.org/2000/svg" width="640" height="205" viewBox="0 0 640 205"><rect width="640" height="205" fill="white"/><text x="30" y="17" font-size="13">All 3764 materials / 1117 groups: max |applicable rule score - 0.5|</text>'+''.join(bars)+'<text x="240" y="198" font-size="12">Deviation (bin width 0.05)</text></svg>')
        lines += ['Full paired differences and per-material rule information are in metrics.json; fold-specific selection details are included.',
            'Pure-rule no-match score is 0.5 (positive), explicitly status-tagged; no HGB fallback. C selection is grouped TRAIN-only CV.']
    else:
        lines += [f'Rule100 score range: {metrics["rule100_score_distribution"]["min"]:.9f} to {metrics["rule100_score_distribution"]["max"]:.9f}.',
            'StrongML predictions include all3764 OOF; the three comparison subsets use exactly the fixed original300.',
            'Paired differences and frozen TRAIN-only model/ensemble choices are in metrics.json.']
    lines += ['No scientific LLM calls. Stage 2 requires a new human instruction: continue.']
    assert len(lines)<=20, len(lines)
    return '\n'.join(lines)+'\n'

def main():
    p=argparse.ArgumentParser(); p.add_argument('experiment',choices=['E0','E1']); p.add_argument('--task-name',default='manual')
    args=p.parse_args(); out=safe_path(ROOT/args.experiment); out.mkdir(exist_ok=True)
    logging.basicConfig(filename=str(safe_path(out/'run.log')),level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',encoding='utf-8',force=True)
    log=logging.info
    awake=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    if not awake: raise RuntimeError('SetThreadExecutionState failed')
    start=time.monotonic()
    git_exe='C:/Users/28908/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe'
    commit=subprocess.check_output([git_exe,'rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    config=dict(experiment=args.experiment,stage=1,scope='adaptive development only',git_commit=commit,
        scientific_llm_calls=0,task_name=args.task_name,execution_time_limit_seconds=0,
        hypotheses={'E0':'Original v1 rule decisions versus HGB127 and frozen four-family strong ML.',
        'E1':'Existing TRAIN rule banks may contain useful information that coverage ranking hides.'}[args.experiment],
        threshold=.5,bootstrap=dict(draws=5000,seed=20261016,unit='association group'),
        runtime=dict(python=sys.version,numpy=np.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__,site=str(SITE)),
        inputs=dict(data=str(DATA),hgb=str(HGB),common=str(COMMON),banks=str(BANKS),programme=str(PROGRAMME),diagnostic=str(DIAGNOSTIC)),
        params=dict(R_cov_top_k=3,R_info_top_k=3,R_info_min_support_groups=20,
        information_cutoff=.15,logistic_C_grid=[.001,.01,.1,1,10,100],tree_depths=[3,4],
        logistic_solver='liblinear',logistic_max_iter=3000,logistic_tol=1e-7,
        inner_cv_splits=3,seed=20261016,
        tree_class_balance='equal TRAIN inverse-group-size mass for both true classes') if args.experiment=='E1' else {})
    if args.experiment=='E0':
        from e0_models import FAMILIES, WEIGHTS, SALT
        frozen=json.loads(PROGRAMME.read_text(encoding='utf-8'))
        config['params']=dict(native_spec_catalogue=[s for s in frozen['native_spec_catalogue'] if s['family'] in FAMILIES or s['family']=='INNER_RAW_X127_FALLBACK'],
            ensemble_weights_candidates=WEIGHTS.tolist(),subfold_salt=SALT,
            family_selection='pooled grouped inner OOF BA, precision, configuration index',
            ensemble_selection='pooled grouped inner OOF BA, precision, candidate index',
            training_weights='1/original group size * TRAIN total group mass /(2 * TRAIN class group mass)',
            feature_views={'D_COMMON969':969,'D_KB_NORMALIZED1015':1015,'RF_SHARED1015':1015,'LR_SHARED1015':1015},
            native_fit_count=78,maximum_additional_scoped_fallback_fits=3,
            lr_convergence_policy='reject_and_retain; inner scoped X127 HGB fallback; outer original HGB127 fallback',
            old_strong_predictions_reused=False)
    write_json(out/'config.json',config)
    safe_path(out/'llm_calls.jsonl').touch(exist_ok=True)
    try:
        log('START %s git=%s awake=%s task=%s',args.experiment,commit,awake,args.task_name)
        d=load_data(); writer=PredictionWriter(out/'predictions.csv',d)
        result=(e1_run if args.experiment=='E1' else e0_run)(d,writer,log)
        result['completion']=dict(status='complete',elapsed_seconds=time.monotonic()-start,scientific_llm_calls=0,
            git_commit=commit,finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(out/'metrics.json',json_safe(result))
        safe_path(out/'README.md').write_text(readme(args.experiment,result),encoding='utf-8')
        expected={'config.json','predictions.csv','llm_calls.jsonl','metrics.json','run.log','README.md'}
        assert {f.name for f in out.iterdir()}==expected
        log('COMPLETE elapsed_seconds=%.3f files=6',time.monotonic()-start)
    except BaseException:
        log('FAILED\n%s',traceback.format_exc())
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__': main()
