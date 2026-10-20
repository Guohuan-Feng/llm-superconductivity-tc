"""Post-run descriptive degeneration diagnostic; no new inference or tuning."""
from common import ROOT,write_json
from g6_local import OUT,read,sha
from g6_stats import report
from collections import Counter
import csv,json

def main():
    cfg=read(OUT/'config.json');metrics=read(OUT/'metrics.json');assert cfg['status']==metrics['status']=='complete'
    with (OUT/'predictions.csv').open(newline='',encoding='utf8') as f:rows=list(csv.DictReader(f))
    arms={arm:[r for r in rows if r['arm']==arm] for arm in ('local_A3','local_A3_reg','local_A3_M1')}
    h={arm:dict(Counter(r['score'] for r in rr)) for arm,rr in arms.items()}
    reg={r['id']:r for r in arms['local_A3_reg']};first={c['id']:c for c in cfg['cases']}
    examples=[];picked={'3dsc_mp_row_000115','3dsc_mp_row_001556','3dsc_mp_row_001557'}
    with (OUT/'llm_calls.jsonl').open(encoding='utf8') as f:
        for line in f:
            r=json.loads(line)
            if r['arm']=='local_A3' and r['id'] in picked and r['status']=='ok':
                material=json.loads(first[r['id']]['A3_prompt'].split('Scientific inputs:\n',1)[1])['material'];q=reg[r['id']]
                examples.append({'id':r['id'],'formula':material['formula'],'A3_output':r['validated_output'],
                    'A3_reg_output':{key:float(q[key]) for key in ('Tc_est_K','p_gt_5K','p_gt_10K','p_gt_15K')},
                    'scientific_material_byte_identical_across_tasks':True})
    assert len(examples)==3
    diag={'status':'observed_degeneration','scope':'post-run descriptive audit only; no prompts, parameters, predictions or primary statistics changed',
        'A3_n':len(arms['local_A3']),'A3_score_histogram':h['local_A3'],'A3_exact_zero_scores':sum(float(r['score'])==0 for r in arms['local_A3']),
        'A3_abstain_histogram':dict(Counter(r['abstain'] for r in arms['local_A3'])),
        'M1_score_histogram':h['local_A3_M1'],'A3_reg_direct_positive_10K':sum(float(r['score'])>=.5 for r in arms['local_A3_reg']),
        'A3_reg_Tc_est_positive_10K':sum(float(r['Tc_est_K'])>10 for r in arms['local_A3_reg']),
        'illustrative_same_material_cross_task_outputs':examples,
        'illustrative_examples_not_for_estimating_a_knowledge_or_memory_fraction':True,
        'interpretation_boundary':'The frozen A3 contract produced a constant-negative classifier in this model/runtime. A3-reg gives nonzero Tc/high probabilities for the same material inputs. This does not establish absent superconductivity knowledge, nor identify a cutoff/knowledge/memory effect. Model capacity, instruction following, output contract, post-training and quantization change together.',
        'diagnostic_source_sha256':sha(ROOT/'g6_diagnosis.py')}
    with (ROOT/'work/G7_eval/local_predictions.csv').open(newline='',encoding='utf8') as f:g7=list(csv.DictReader(f))
    diag['G7_frozen89_Local_A3_score_histogram']=dict(Counter(r['score'] for r in g7))
    diag['constant_classifier_DiD_boundary']='If both cohorts contain both classes and Local_A3 is constant-negative on each, its BA is exactly50% on both. A positive (Local change minus HGB change) is then algebraically the negative HGB BA change, not evidence of local learning, transfer, robustness, or a corrected novelty advantage.'
    cfg['post_run_degenerate_A3_diagnosis']=diag;metrics['post_run_degenerate_A3_diagnosis']=diag
    metrics['post_run_diagnostic_report_source_sha256']=sha(ROOT/'g6_stats.py')
    metrics['report_markdown']=report(metrics)
    write_json(OUT/'config.json',cfg);write_json(OUT/'metrics.json',metrics)
    print(json.dumps(diag,ensure_ascii=False))
if __name__=='__main__':main()
