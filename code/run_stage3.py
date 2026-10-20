"""Bounded Stage3: post-hoc E3 + E4 only. Prepare freezes every prompt before calls."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import subprocess
import sys

from common import ROOT, FIELDS, load_data, safe_path, write_json, np
from e2_transport import MODEL, CLI_VERSION, REASONING_EFFORT, input_budget
from e2_metrics import mean_summary
from e3_inputs import build_e3_inputs
from e4_inputs import build_e4_inputs
from stage3_stats import evaluate_stage3
from stage3_prompts import make_e3, make_e4, validate_annotation
from stage3_runtime import annotate, read_calls

SCHEMA = ROOT / 'stage3_response.schema.json'
INSTRUCTIONS = ROOT / 'stage3_instructions.md'
SOURCE_NAMES = ['run_stage3.py','stage3_runtime.py','stage3_prompts.py','stage3_stats.py',
                'e3_inputs.py','e4_inputs.py','e2_transport.py','e2_metrics.py','stats.py',
                'common.py','stage3_response.schema.json','stage3_instructions.md',
                'e2_model_catalog.json','launch_stage3.ps1']
CATEGORY_NAMES = {'1':'规则不相关','2':'特征误读','3':'正确化学判断被规则覆盖',
                  '4':'疑似标签噪声或代理结构错配','5':'其他／证据不足'}

def utcnow():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

def log_for(stage):
    logger = logging.getLogger(stage)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(safe_path(ROOT / stage / 'run.log'), encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(handler)
    return logger.info

def write_predictions(stage, arms):
    rows = {}
    wanted = set(arms)
    ids = set(read_json(ROOT / 'E2/config.json')['cases'])
    for source in ('E1','E2'):
        with (ROOT / source / 'predictions.csv').open(encoding='utf-8', newline='') as handle:
            for row in csv.DictReader(handle):
                key = (row['arm'],row['id'])
                if row['arm'] in wanted and row['id'] in ids:
                    if key in rows and float(rows[key]['score']) != float(row['score']):
                        raise ValueError('Frozen prediction sources disagree')
                    row['status'] = 'reused_frozen_OOF'
                    rows[key] = row
    assert len(rows) == 300 * len(arms)
    path = safe_path(ROOT / stage / 'predictions.csv')
    with path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        ordered = [rows[key] for key in sorted(rows)]
        for start in range(0,len(ordered),10):
            writer.writerows(ordered[start:start+10])
            handle.flush(); os.fsync(handle.fileno())

def prepare(task_name):
    if any((ROOT / stage).exists() for stage in ('E3','E4')):
        raise RuntimeError('Stage3 directories already exist; use run to resume frozen configuration')
    d = load_data()
    e3 = build_e3_inputs(d=d)
    e4 = build_e4_inputs(data=d, config=e3['config'])
    stats = evaluate_stage3(d=d, config=e3['config'])
    packets = {'E3':e3['selected'], 'E4':e4['top30_rules']['rule_annotation_packets']}
    items = {}
    for stage, records in packets.items():
        make = make_e3 if stage == 'E3' else make_e4
        items[stage] = [{**make(record), 'packet':record} for record in records]
        assert len(items[stage]) == 30 and len({r['key'] for r in items[stage]}) == 30
        for item in items[stage]:
            bound = input_budget(item['prompt'], schema_path=SCHEMA, instructions_path=INSTRUCTIONS)
            if not bound['within_budget']:
                raise ValueError(f'Input overrun: {stage} {item["key"]} {bound}')
            item['input_budget'] = bound
    scientific_sources = {f'{stage}/{name}':sha(ROOT/stage/name)
                          for stage,names in {'E1':['predictions.csv','metrics.json'],
                                              'E2':['predictions.csv','config.json','llm_calls.jsonl']}.items()
                          for name in names}
    source_hashes = {name:sha(ROOT/name) for name in SOURCE_NAMES}
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip()
    for stage in ('E3','E4'):
        out = safe_path(ROOT/stage)
        out.mkdir()
        cfg = {'stage':3,'experiment':stage,'scope':'adaptive development only; no Stage4 work',
               'hypothesis':('Separate irrelevant-rule/feature/override/mismatch/other explanations of A1 and A2 errors.' if stage=='E3' else 'Compare concise rule citations, provisional physical meaning, and shallow-tree accuracy on the same cohort.'),
               'created_utc':utcnow(),'task_name':task_name,'git_commit':commit,
               'model':MODEL,'cli_version':CLI_VERSION,'reasoning_effort':REASONING_EFFORT,
               'model_version_note':'Frozen explicit slug; immutable backend weights not reported.',
               'params':{'concurrency':10,'max_attempts_per_item':2,'planned_calls':30,
                         'max_calls':60,'max_total_stage3_calls':120,'input_limit_tokens':15000,
                         'bootstrap_draws':5000,'bootstrap_seed':20261016},
               'source_hashes':source_hashes,'frozen_scientific_inputs':scientific_sources,
               'items':items[stage],
               'no_new_predictions':'Every prediction is reused unchanged; annotations are post-hoc.',
               'review_status':'local Codex provisional; human review pending'}
        if stage == 'E3':
            cfg.update(sampling_design=e3['sampling_design'],error_population=e3['counts'],
                       category_names=CATEGORY_NAMES,
                       packet_scope='Compact historical evidence; 22 redundant global scalars omitted; rule target values retained.')
            metrics = {'status':'prepared','stage':3,'experiment':stage,'family_and_control_statistics':stats,
                       'sampling_design':e3['sampling_design'],'error_population':e3['counts']}
            arms = ['A1_forced','A2_forced','A2_HGBfallback','HGB127']
        else:
            cfg.update(top30_selection={k:v for k,v in e4['top30_rules'].items() if k!='rule_annotation_packets'},
                       examples_selection=e4['examples']['selection'])
            metrics = {'status':'prepared','stage':3,'experiment':stage,**e4}
            arms = ['R-info','A2_forced','Tree3','Tree4','HGB127']
        write_json(out/'config.json',cfg)
        write_json(out/'metrics.json',metrics)
        write_predictions(stage,arms)
        safe_path(out/'llm_calls.jsonl').touch(exist_ok=False)
        safe_path(out/'run.log').write_text(utcnow()+' prepared frozen post-hoc annotation inputs\n',encoding='utf-8')
        safe_path(out/'README.md').write_text(f'# {stage} Stage3\nPrepared; bounded post-hoc analysis only.\nSee metrics.json for tables and config.json for frozen inputs.\nNo Stage4 work.\n',encoding='utf-8')
    print(json.dumps({stage:{'n':len(rows),'max_input_upper_bound':max(r['input_budget']['input_token_upper_bound'] for r in rows)} for stage,rows in items.items()}))

def verify_config(cfg):
    for name,expected in cfg['source_hashes'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Frozen Stage3 source changed: '+name)
    for name,expected in cfg['frozen_scientific_inputs'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Frozen scientific inputs changed: '+name)

def call_summary(stage):
    calls = read_calls(stage)
    return {'n_calls':len(calls),'n_valid_annotations':sum(bool(c.get('annotation_valid')) for c in calls),
            'input_tokens':sum(c.get('input_tokens') or 0 for c in calls),
            'output_tokens':sum(c.get('output_tokens') or 0 for c in calls),
            'cached_input_tokens':sum(c.get('cached_input_tokens') or 0 for c in calls),
            'sum_call_seconds':sum(c.get('elapsed_seconds') or 0 for c in calls),
            'models':sorted({c.get('model','') for c in calls}),
            'tools_used':sum(len(c.get('tool_activity') or []) for c in calls),
            'max_input_tokens':max([c.get('input_tokens') or 0 for c in calls] or [0])}

def finish_e3(cfg, annotations, d):
    rows = [{**item['packet'], 'annotation':annotations[item['key']],
             'human_review':'pending'} for item in cfg['items']]
    indices = np.asarray([row['i'] for row in rows])
    category_stats = {}
    for category,name in CATEGORY_NAMES.items():
        subset = [row for row in rows if row['annotation']['category']==category]
        summary = mean_summary([int(row['annotation']['category']==category) for row in rows],
                               d['group'][indices],d['group_size'][indices])
        pool = sorted(subset,key=lambda r:hashlib.sha256(('E3-typical|'+r['id']).encode()).hexdigest())
        typical, seen = [], set()
        for row in pool:
            if row['group'] not in seen and len(typical)<3:
                typical.append(row); seen.add(row['group'])
        typical += [r for r in pool if r not in typical][:3-len(typical)]
        category_stats[category] = {'name':name,'n':len(subset),'n_groups':len({r['group'] for r in subset}),
                                    'by_primary_arm':dict(Counter(r['primary_arm'] for r in subset)),
                                    'fraction_in_selected30':summary,
                                    'typical_examples':typical,
                                    'example_availability':'No fabricated examples; fewer than two if category absent/small.'}
    path=ROOT/'E3/metrics.json'; metrics=read_json(path)
    metrics.update(status='complete',completed_utc=utcnow(),annotations=rows,category_results=category_stats,
                   annotation_frequency_scope='Purposive stratified30/13groups only; NOT prevalence among all errors or the deployment population.',
                   annotation_review_status='Provisional local-Codex annotation; human review pending; no causal validation.',
                   calls=call_summary('E3'))
    write_json(path,metrics)
    lines=['# E3: error analysis','Complete; post-hoc only. A2 is the designated best LLM arm.',
           '30 unique materials /13 groups: primary A1=15, A2=15; Cu=6, Fe=3, other=21, hydride=0.',
           'Category frequencies describe the selected stratified sample, not all errors.',
           'All annotation labels remain provisional, pending human review.',
           'A2 has 35 raw errors /3 groups; 33 occur in one original association group.',
           'Point metrics: pooled OOF; weight 1/full original group size; 5000 group bootstrap draws.',
           'Iron family: one group; CI is degenerate. Hydride family: no observations.',
           'Full category tables, per-case evidence, family comparisons: metrics.json.',
           'Original reused predictions: predictions.csv; all annotation calls: llm_calls.jsonl.',
           'Stage4 has not started.']
    safe_path(ROOT/'E3/README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def finish_e4(cfg, annotations):
    path=ROOT/'E4/metrics.json'; metrics=read_json(path)
    rows=metrics['top30_rules']['rule_annotation_packets']
    for row in rows:
        a=annotations[row['marker']]
        row['annotation']={**a,'label':a['category'],'human_review':'pending'}
    metrics.update(status='complete',completed_utc=utcnow(),review_table=rows,
                   provisional_rule_label_counts=dict(Counter(a['category'] for a in annotations.values())),
                   review_status='ALL30 pending human review; physical_meaning is not a causal validation.',
                   calls=call_summary('E4'))
    write_json(path,metrics)
    lines=['# E4: interpretability analysis','Complete; all 30 physical/proxy labels are provisional, human-review-pending.',
           'Fixed cohort: 300 materials /81 original association groups.',
           'R-info executes 3 rules; A2 self-reports selected IDs, including 84 zero-rule cases.',
           'A2 citation count is not proof of causal faithfulness or actual internal rule use.',
           'Top30: combined R-info+A2 material citation counts; fold-scoped cards; 908/1194 citations.',
           'Full 30-row review table with exact predicates, support and evidence: metrics.json/review_table.',
           'Five complete paired chains: metrics.json/examples/chains; one zero-rule supplement is separate.',
           'Saved E1 Tree3/Tree4 predictions reused on identical300, without refit.',
           '5000 paired association-group bootstrap draws; original full-group weights retained.',
           'All call prompts/responses/tokens/time: llm_calls.jsonl.',
           'Stage4 has not started.']
    safe_path(ROOT/'E4/README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def run(task_name):
    sys.stdout.reconfigure(encoding='utf-8')
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        d=load_data()
        for stage in ('E3','E4'):
            cfg=read_json(ROOT/stage/'config.json')
            if cfg['task_name'] != task_name:
                raise ValueError('Task identity mismatch')
            verify_config(cfg)
            log=log_for(stage)
            if read_json(ROOT/stage/'metrics.json')['status']=='complete':
                log('Already complete; skip without new calls'); continue
            log('Starting/resuming frozen finite annotation batch')
            lookup={item['key']:item for item in cfg['items']}
            validator=lambda key,parsed:validate_annotation(stage,lookup[key],parsed)
            annotations=annotate(stage,cfg['items'],validator,log=log)
            assert len(annotations)==30
            if stage=='E3': finish_e3(cfg,annotations,d)
            else: finish_e4(cfg,annotations)
            log('COMPLETE; no Stage4 dispatch')
        print('STAGE3_COMPLETE')
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['prepare','run']);p.add_argument('--task-name',required=True)
    args=p.parse_args()
    try:
        (prepare if args.action=='prepare' else run)(args.task_name)
    except Exception:
        import traceback
        traceback.print_exc()
        for stage in ('E3','E4'):
            if (ROOT/stage/'run.log').exists():
                with safe_path(ROOT/stage/'run.log').open('a',encoding='utf-8') as handle:
                    traceback.print_exc(file=handle)
        raise
