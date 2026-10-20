"""Frozen A3 scientific material inputs for G5 regression/multiple thresholds."""
from common import ROOT, DATA, HGB, load_data, np
from pathlib import Path
import hashlib
import json
import math
import csv
from e2_packets import round_prompt_numbers
from e2_prompts import make_prompt

OUT = ROOT / 'G5_tc_multithreshold'
SCHEMA = ROOT / 'g5_response.schema.json'
INSTRUCTIONS = ROOT / 'g5_instructions.md'
CATALOG = ROOT / 'e2_model_catalog.json'
THRESHOLDS = (5, 10, 15)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def prompt(material, repair=None):
    text=('Estimate superconducting Tc (kelvin) from the supplied proxy material formula, composition, and structural summary. '
          'Use your existing chemical knowledge and these supplied facts. For a nonsuperconducting material estimate 0 K. '
          'Give Tc_est_K and p_gt_5K=P(Tc>5 K), p_gt_10K=P(Tc>10 K), p_gt_15K=P(Tc>15 K). '
          'All values must be finite numbers, Tc_est_K>=0, and 1>=p_gt_5K>=p_gt_10K>=p_gt_15K>=0. '
          'The point estimate and threshold probabilities are separate outputs; they need not yield identical hard decisions. '
          'Return only the required JSON object.')
    if repair is not None:
        text+='\nThe preceding attempt violated the numeric JSON contract or monotone probability ordering. Return the required four numeric fields using exactly the same material evidence.'
    packet=round_prompt_numbers({'material':material,'previous_rounds':[]})
    return text+'\nScientific inputs:\n'+json.dumps(packet,ensure_ascii=False,separators=(',',':'),allow_nan=False)

def validate(obj):
    names=('Tc_est_K','p_gt_5K','p_gt_10K','p_gt_15K')
    if not isinstance(obj,dict) or set(obj)!=set(names): raise ValueError('Expected exact four-field JSON object')
    if any(isinstance(obj[k],bool) or not isinstance(obj[k],(int,float)) or not math.isfinite(obj[k]) for k in names):
        raise ValueError('All outputs must be finite numbers')
    if obj['Tc_est_K']<0: raise ValueError('Tc estimate must be nonnegative')
    if not 1>=obj['p_gt_5K']>=obj['p_gt_10K']>=obj['p_gt_15K']>=0:
        raise ValueError('Probabilities must lie in[0,1] and p5>=p10>=p15')
    return {key:float(obj[key]) for key in names}

def frozen_inputs(d=None):
    d=load_data() if d is None else d
    f3=ROOT/'F3_full3764/config.json'
    cfg=json.loads(f3.read_text(encoding='utf8'))
    g3p=ROOT/'G3_complementarity/config.json'
    g3=json.loads(g3p.read_text(encoding='utf8'))
    labels=g3['family_results']; vocab=g3['family_vocabulary']
    assert len(labels)==3764 and len(vocab)==12
    cases=[]
    for i,key in enumerate(cfg['case_order']):
        original=cfg['cases'][key]
        assert key==str(d['id'][i]) and original['fold']==int(d['fold'][i]) and original['group']==str(d['group'][i])
        material=original['material']
        original_prompt=make_prompt('A3',material,[],{},[],[],round_number=1)
        assert hashlib.sha256(original_prompt.encode('utf8')).hexdigest()==original['prompt_hashes']['A3']
        original_packet=original_prompt.split('\nScientific inputs:\n',1)[1]
        current_packet=prompt(material).split('\nScientific inputs:\n',1)[1]
        assert json.loads(current_packet)==json.loads(original_packet) and current_packet==original_packet
        cases.append({'i':i,'id':key,'material':material,'material_sha256':hashlib.sha256(json.dumps(material,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode('utf8')).hexdigest(),
                      'scientific_packet_sha256':hashlib.sha256(current_packet.encode('utf8')).hexdigest(),
                      'rounded_material_sha256':hashlib.sha256(json.dumps(round_prompt_numbers(material),sort_keys=True,ensure_ascii=False,separators=(',',':')).encode('utf8')).hexdigest(),
                      'original_A3_prompt_hash_revalidated':True,'scientific_packet_byte_equal_original_A3':True,
                      'prompt_sha256':hashlib.sha256(prompt(material).encode('utf8')).hexdigest()})
    a3=np.full(3764,np.nan)
    lookup={str(v):i for i,v in enumerate(d['id'])}
    with (ROOT/'F3_full3764/predictions.csv').open(newline='',encoding='utf8') as f:
        for row in csv.DictReader(f):
            if row['arm']=='A3_forced': a3[lookup[row['id']]]=float(row['score'])
    assert np.isfinite(a3).all()
    sources=[DATA,HGB,f3,ROOT/'F3_full3764/predictions.csv',ROOT/'F3_full3764/metrics.json',
             g3p,ROOT/'G3_complementarity/predictions.csv',ROOT/'G3_complementarity/metrics.json',
             ROOT.parent/'2026-10-10-structure-nonlinear-classification/nonlinear_classification.py',
             ROOT.parent/'2026-10-10-structure-nonlinear-classification/run/protocol.json']
    return {'cases':cases,'original_A3_10K':a3.tolist(),'family_masks':cfg['family_masks'],'raw_a15_mask':cfg['raw_a15_mask'],
            'group_size_values':d['group_size'].tolist(),'G3_family_masks':{v:[labels[str(key)]==v for key in d['id']] for v in vocab},
            'G3_family_vocabulary':vocab,'family_definition':cfg['metadata']['family_definitions'],
            'scientific_sources':{str(path):sha(path) for path in sources},
            'input_identity':'The material object is exactly original E2/F3 A3 formula+composition+structure; no labels, folds, groups, rules, or neighbors in evaluated prompt.'}
