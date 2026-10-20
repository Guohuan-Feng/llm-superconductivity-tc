"""Prepare auditable pre-freeze packet; no evaluations, no presumed approval."""
import csv,datetime,hashlib,json
from collections import Counter
from pathlib import Path
ROOT=Path(r'C:\Users\28908\Codex\fung-research\v2-diagnosis');OUT=ROOT/'G7_postcutoff_materials'
assert 'onedrive' not in str(OUT.resolve()).lower()
j=json.loads((OUT/'config.json').read_text(encoding='utf8'));labs=json.loads((OUT/'labels.json').read_text(encoding='utf8'))
rows=list(csv.DictReader((OUT/'predictions.csv').open(encoding='utf8')))
for r in rows:
 mid=r['material_id'];lab=labs[mid]
 if mid.startswith('G7_SCE_'):
  # Per-composition exact values come from published author data, not high-field reentry.
  r['first_report_date']='2025-08-20'
  lab.update(first_material_report_date='2025-08-20',first_material_report_date_precision='earliest bound for series, not a per-point exact first day',first_material_date_interval=['2025-08-20','2026-03-26'],exact_date_known=False,first_ambient_label_date='2026-03-26',material_series_first_report_date='2025-08-20',label_source='https://doi.org/10.6084/m9.figshare.31861684',source_version='author Fig1 XLSX published2026-03-26; arxiv2508.14666v3 dated2026-04-25',magnetic_field_t=0,tc_criterion='ambient-pressure zero-external-field onset Tc; Fig1 XLSX sheet c dot column B only; high-field reentrant column C never used',no_transition_detection_limit_source='arxiv2508.14666v3 Methods p13: resistivity down to2K; Fig1 zero-field series and individual curves ExtendedData2')
  suffix='; exact x/Tc pair available no later than author table2026-03-26; v1 plotted series first2025-08-20, per-point first-day not asserted'
  if suffix not in r['reason']:r['reason']+=suffix
 if mid=='G7_SCE_0.2000':
  r['first_report_date']='2025-03-20';lab['first_material_report_date']='2025-03-20';lab['first_material_date_interval']=['2025-03-20','2025-03-20'];lab['prior_material_source']='https://doi.org/10.1038/s41586-025-08893-4';lab['first_report_evidence']='Exact Sm0.75Eu0.2Ca0.05NiO2 in Nature2025; current table zero-field onset32.34K is a later sample measurement. Both classifications above10K.'
 if mid in ['F4_candidate_aupbbi','F4_candidate_pdpbbi','F4_candidate_rhpbbi']:r['space_group']='136'
 if mid=='G7_TaWMoReOs_sigma':lab['first_material_report_date_precision']='year-month; first day is interval start, not asserted publication day';lab['first_material_report_date_interval']=['2025-07-01','2025-07-31']
 if mid=='G7_Mo20Re6Si4':lab['first_material_report_date_precision']='arxiv first-day; same experiment at earlier ML4PS2024 workshop';lab['first_material_report_date_interval']=['2024-11-01','2024-12-17']
 if r['status']=='accepted':
  lab.setdefault('magnetic_field_t',0)
  r['specific_parent_review']=r.get('specific_parent_review') or 'No exact3764 match; necessary specific parent core absent or new crystal topology documented. See config dedup_evidence.'
  # Source titles, dates, SG-only data, labels and audit text are forbidden from packets.
  r['cif_url']=''

fields=['material_id','formula','first_report_date','doi_or_arxiv','cif_url','space_group','status','source','reason','parent_family_key','exact_3764_matches','specific_parent_review']
with (OUT/'predictions.csv').open('w',encoding='utf8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r.get(k,'') for k in fields} for r in rows)
(OUT/'labels.json').write_text(json.dumps(labs,ensure_ascii=False,indent=2),encoding='utf8')
a=[r for r in rows if r['status']=='accepted'];ps=Counter(r['parent_family_key'] for r in a if labs[r['material_id']]['y']==1)
j.update(status='awaiting_root_review_not_frozen',retrieved_CIF_manifest={},scientific_materials={r['material_id']:{'formula':r['formula']} for r in a},accepted_material_ids=[r['material_id'] for r in a],screening_rows=rows,screening_input_csv_sha256=hashlib.sha256((OUT/'predictions.csv').read_bytes()).hexdigest(),screening_input_labels_sha256=hashlib.sha256((OUT/'labels.json').read_bytes()).hexdigest(),evaluation_prompt_policy='Whitelist material.formula only when no retrieved CIF. Never include audit reasons, titles, links, date, pressure or labels. SG alone is audit-only.',freeze_state='NOT_FROZEN_AWAITING_ROOT_REVIEW')
j['known_CIF_depositions']={'G7_BaClNbS2':'2395029','G7_BaClNbSe2':'2395036','F4_candidate_aupbbi':'2562542','F4_candidate_pdpbbi':'2562544','F4_candidate_rhpbbi':'2562543'}
j['CIF_availability']={'known_deposition_n':5,'actually_retrieved_CIF_n':0,'structure_model_evaluable_n':0,'accepted_with_known_space_group_n':sum(bool(r['space_group']) for r in a),'reason':'CCDC public Search/LoadSearch returned General Enquiry requiring license/sign-in; Wiley SI403. No bypass, no SG-only invented structures. Known deposition is distinguished from actual CIF.'}
j['positive_support_by_specific_parent']=dict(ps)
j['collection_scope_and_end']='arXiv supr-con date-window inventory5818 entries, title/abstract lead screen plus primary full-text follow-up and publisher/author archives; not a claim that all5818 full texts were read.143 material candidates manually curated.89 accepted exceeds60/15floor but misses100-material target. Search ended after several hours and first-material/specific-parent checks; no arbitrary ratio padding.'
j['remaining_limits']=['Only3 specific-parent groups contain positive materials; composition-series rows are correlated and bootstrap must resample groups.','33 positive labels are dominated by the infinite-layer NiO2 series; broad nickelate family knowledge predates cutoff, so this tests new compositions rather than proof of no family prior.','First-public-report dates are finite-search findings; exact-series point dates have explicit intervals.','All89 have no actually retrieved CIF; A3 formula-only and structure baseline unassessed (n0).','Nominal stoichiometries, varied transition criteria and thin-film strain/sample effects limit composition-only causal interpretations.']
(OUT/'config.json').write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf8')
m=j['final_review_summary'];m.update(accepted_with_known_space_group_n=j['CIF_availability']['accepted_with_known_space_group_n'],actually_retrieved_CIF_n=0,positive_parent_groups_n=len(ps),positive_support_by_specific_parent=dict(ps),first_report_earliest=min(r['first_report_date'] for r in a),target_n_reached=len(a)>=100,target_positive_reached=sum(labs[r['material_id']]['y']==1 for r in a)>=25,hard_floor_met=len(a)>=60 and sum(labs[r['material_id']]['y']==1 for r in a)>=15)
(OUT/'metrics.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf8')
(OUT/'config.json').write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf8')
readme=['# G7 post-cutoff material collection (awaiting root review)',f'- Accepted89 materials:33 positive,56 negative;20 specific-parent groups.','- Target100/25: material target unmet, positive target met; minimum60/15 met.','- Official Llama3.1 cutoffDecember2023; conservative eligibility starts2024-02-01.','- Literature inventory5818 arXiv entries;143 material candidates curated; full-text evidence retained in config.','- predictions.csv is screening metadata, not model predictions; labels.json holds evaluation labels.','- config.scientific_materials is label-free; all evaluation packets formula-only.','- Exact reduced-formula and specific-parent checks use original3764; all accepted exact matches absent.','- New Tc measurements of old materials excluded; unresolved carbon/mixed phases retained pending.','- Zero-field Tc used; no-field-transition bounds kept null with upper bound, never fabricated0K.','- Three positive parent groups; bootstrap must resample parent groups, not treat a series as independent.','- Known CIF depositions5; actual retrievedCIF0; structure baseline not evaluable(n0).','- CCDC returned license/sign-in enquiry; WileySI403; no structures inferred from SG.','- SCE per-ratio first days not fixed: source date interval wholly post-cutoff; CSV earliest bound is not an exact first day.','- No evaluation LLM calls or baseline fits by collector. Input not frozen until root reviews.']
(OUT/'README.md').write_text('\n'.join(readme)+'\n',encoding='utf8')
with (OUT/'run.log').open('a',encoding='utf8') as f:f.write('\n'+datetime.datetime.now(datetime.timezone.utc).isoformat()+' PREPARED_FOR_REVIEW '+json.dumps(m)+'\n')
print(json.dumps(m,indent=2))
