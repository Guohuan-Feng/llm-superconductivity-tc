"""Final literature curation before evaluation; no model calls. Idempotent."""
import csv, datetime, hashlib, io, json, re
from collections import Counter
from pathlib import Path
ROOT=Path(r'C:\Users\28908\Codex\fung-research\v2-diagnosis')
OUT=ROOT/'G7_postcutoff_materials'
assert 'onedrive' not in str(OUT.resolve()).lower()
j=json.loads((OUT/'config.json').read_text(encoding='utf8'))
labs=json.loads((OUT/'labels.json').read_text(encoding='utf8'))
rows=list(csv.DictReader((OUT/'predictions.csv').open(encoding='utf8')))
cases=json.loads((ROOT/'F3_full3764/config.json').read_text(encoding='utf8'))['cases']
ref={mid:{x['element']:x['atomic_fraction'] for x in c['material']['composition']} for mid,c in cases.items()}
def comp(f):
 p=re.findall(r'([A-Z][a-z]?)([0-9.]*)',f); assert ''.join(a+b for a,b in p)==f,f
 d={}
 for a,b in p:d[a]=d.get(a,0)+float(b or 1)
 s=sum(d.values());return {a:v/s for a,v in d.items()}
def exact(f):
 d=comp(f);return [mid for mid,r in ref.items() if set(d)==set(r) and all(abs(d[e]-r[e])<1e-7 for e in d)]
def add(mid,f,tc,date,url,group,sg,note,upper=None,status='accepted'):
 hit=exact(f)
 if hit:status='excluded_exact_3764_reduced_formula'
 r=dict(material_id=mid,formula=f,first_report_date=date,doi_or_arxiv=url,cif_url='',space_group=str(sg or ''),status=status,source='primary source; full text/table retained in config',reason=note,parent_family_key=group,exact_3764_matches=';'.join(hit),specific_parent_review='Reviewed specific chemical parent and phase against 3764; no matching parent unless explicitly excluded')
 old=next((i for i,x in enumerate(rows) if x['material_id']==mid),None)
 if old is None:rows.append(r)
 else:rows[old]=r
 labs[mid]=dict(tc_k=tc,tc_upper_bound_k=upper,y=int(tc>10) if tc is not None else (0 if upper is not None and upper<=10 else None),pressure_gpa=0,pressure_semantics='ambient measurement; synthesis pressure is not measurement pressure',status=status,label_source=url,tc_criterion=note,parent_family_key=group,first_material_report_date=date,first_ambient_label_date=date,first_report_evidence=note)
def change(mid,status,why,source=None,first=None):
 r=next(x for x in rows if x['material_id']==mid);r['status']=status;r['reason']=why
 labs[mid]['status']=status;labs[mid]['first_report_evidence']=why
 if source:labs[mid]['prior_report_source']=source
 if first:labs[mid]['first_material_report_date']=first

# A new Tc measurement does not turn an old material into a post-cutoff material.
change('F4_candidate_bilayer_pr','excluded_preexisting_material','Same La2PrNi2O7 material reported in 2023 under pressure; first ambient film Tc is not first material report.','https://arxiv.org/abs/2311.08212','2023-11-14')
change('F4_candidate_nd_eu03','excluded_preexisting_material','Nd1-xEuxNiO2 x=0.15-0.4 experimentally reported in 2023; later growth/reentrance measurements are not new material.','https://doi.org/10.1126/sciadv.adh3327','2023-07-05')
change('G7_Al13Os4','excluded_preexisting_material','Monoclinic Al13Os4 superconductivity already reported in 1968; 2025 paper is new characterization.','https://www.e-periodica.ch/cntmng?pid=hpa-001%3A1968%3A41%3A%3A1563','1968')
change('G7_HEA_antimonide','excluded_preexisting_material','Original high-entropy antimonide article is Inorg. Chem.62,14207-14215 (2023); 2025 arXiv upload is not first report.','https://www.nature.com/articles/s41427-024-00579-z','2023')
change('G7_TaPtSi','excluded_preexisting_material','Same compound and superconductivity reported in 2000; also exact reduced formula in 3764.','https://doi.org/10.1016/S0022-3697(00)00044-5','2000')
change('G7_Zr6FeBi2','excluded_preexisting_material','2024 primary introduction explicitly says Zr6FeBi2 was previously synthesized and its resistivity measured (refs11,20,25); first low-Tc measurement is not new material.','https://arxiv.org/pdf/2410.16684')
change('G7_Ta4Rh2C','pending_exact_carbon_occupancy','Primary reports Ta4Rh2C1-delta; carbon vacancy delta is unresolved, so a precise evaluation composition is not yet established.','https://arxiv.org/abs/2506.02209')
change('G7_bst_pressure_quenched','pending_metastable_phase_identity','Bi0.5Sb1.5Te3 formula is old. 2025 pressure quench retains a mixed phase with 10.2 K attributed to BST-II/III or another unresolved phase; cannot assign unique new material identity from formula alone.','https://arxiv.org/pdf/2502.01881')
change('F4_candidate_la6ni5o12','excluded_preexisting_theoretical_identity','La6Ni5O12 same formula/structure and predicted 11-19 K Tc were public in 2023 arXiv2311.13674; first bulk experimental superconductivity in 2026 is not a clean unseen-identity test.','https://arxiv.org/abs/2311.13674','2023-11-22')
change('F4_candidate_luru3b2','excluded_specific_parent_family','Earlier GPT window exclusion is withdrawn. However original 3764 contains YRu3B2 of the same specific RRu3B2 parent, so excluded under parent-family dedup.')

# Author figure text and supporting full text supply explicit compositions/Tc.
add('G7_SECS_079','Sm0.79Eu0.12Ca0.04Sr0.05NiO2',32.3,'2024-09-30','https://arxiv.org/abs/2410.00144','infinite_layer_NiO2',123,'Primary Fig3 exact formula and 90%-resistivity Tc32.3 K; new Sm-based film composition, rather than an old Nd/Sr sample.')
add('G7_SECS_085','Sm0.85Eu0.06Ca0.08Sr0.01NiO2',7.0,'2024-09-30','https://arxiv.org/abs/2410.00144','infinite_layer_NiO2',123,'Primary p5 explicitly states underdoped exact formula and onset approximately7 K.')
add('G7_SCE_Ca007_020','Sm0.73Eu0.2Ca0.07NiO2',26.5,'2025-03-20','https://doi.org/10.1038/s41586-025-08893-4','infinite_layer_NiO2',123,'Nature Extended Data Fig4 exact formula; zero-resistance Tc26.5 K, hence positive independently of onset definition.')
add('G7_SCE_Ca007_040','Sm0.53Eu0.4Ca0.07NiO2',None,'2025-03-20','https://doi.org/10.1038/s41586-025-08893-4','infinite_layer_NiO2',123,'Nature Extended Data Fig4: outside superconducting phase, metallic throughout measured2-300 K; null Tc with upper bound2 K.',2)
add('G7_LaPrSm_bilayer','La2.46Pr0.24Sm0.3Ni2O7',52.1,'2025-08-22','https://arxiv.org/abs/2508.16412','RP_bilayer_hybrid_nickelate',None,'Primary p3 exact new co-substituted film; ExtendedData5 gives onset52.1 K, zero14.6 K. No external pressure during measurement.')
# Methods of hybrid paper specify nominal La:Pr=0.67:0.33 with ~1% composition precision.
add('G7_hybrid_1212','La3.35Pr1.65Ni3O11',50.0,'2025-09-03','https://arxiv.org/abs/2509.03502','RP_bilayer_hybrid_nickelate',None,'Primary Methodsp16 La0.67Pr0.33Ox target; nominal1212 formula from5x0.67/0.33. Onset50K. Formula represents nominal composition, not measured stoichiometry.')
add('G7_hybrid_2323','La4.69Pr2.31Ni5O17',46.0,'2026-04-13','https://arxiv.org/abs/2509.03502v3','RP_bilayer_hybrid_nickelate',None,'Nominal La:Pr0.67:0.33 target;2323 material introduced in v3 (not v1). Onset46K. ~1% composition precision.')
change('G7_hybrid_1313','pending_phase_first_report_and_detection_bound','1313 control nominal La:Pr ratio now resolved from Methods, but exact phase first-report and lowest no-transition temperature require separate verification; not counted.')

# Chemical-avenue paper: new rare-earth co-substitution compositions; the old Nd-Eu parent itself is excluded.
for key,f,tc,criterion in [
 ('NdPr010','Nd0.585Pr0.065Eu0.35NiO2',20,'zero-resistance approximately20K, primaryp6'),
 ('NdSm010','Nd0.585Sm0.065Eu0.35NiO2',18,'zero-resistance approximately18K, primaryp6'),
 ('NdDy010','Nd0.585Dy0.065Eu0.35NiO2',14,'zero-resistance approximately14K, primaryp6'),
 ('NdPr020','Nd0.52Pr0.13Eu0.35NiO2',40.1,'onset40.1K, primaryp7/S24-S25'),
 ('NdSm040','Nd0.39Sm0.26Eu0.35NiO2',38.1,'onset38.1K, primaryp7/S24-S25'),
 ('NdPrSm030','Nd0.26Pr0.195Sm0.195Eu0.35NiO2',38.4,'onset38.4K, primaryp7/S24-S25')]:
 add('G7_REmix_'+key,f,tc,'2025-11-27','https://arxiv.org/abs/2511.22026','infinite_layer_NiO2',123,'New co-substituted material grown by chemical avenue; '+criterion+'; algebraically expanded reported nominal formula. First known public material report in present search; no claim of exhaustive absence proof.')

# Open author supplemental table labels these sixteen FCC compositions "This text".
text=j['primary_SI_reads']['10.1007/s10853-024-09780-5']['text']
lines=[x.strip() for x in text.splitlines() if x.strip()]
fcc=[]
for i,s in enumerate(lines):
 if s=='This text':
  f,structure,tc=lines[i-6:i-3]
  assert structure=='fcc',(f,structure,tc)
  fcc.append((f,float(tc)))
for i,(f,tc) in enumerate(fcc):
 add('G7_FCC_'+str(i+1).zfill(2),f,tc,'2024-06-09','https://doi.org/10.1007/s10853-024-09780-5','W_Rh_Ir_Pt_fcc_HEA',225,'Author SI final table, source column This text; exact reported nominal atomic percentages, ambient resistivity Tc. Public2024-06-09; 2023 precursor work concerned distinct equimolar MoReRuRhPt hcp systems.')
assert len(fcc)==16
add('G7_Mo20Re6Si4','Mo20Re6Si4',5.4,'2024-12-17','https://arxiv.org/abs/2412.13012','Mo_Re_Si_sigma',None,'Primaryabstract/section4.1: synthesized new ternary compound, bulk Tc5.4K at ambient pressure. Earlier same-title ML4PS2024 workshop also post-cutoff; exact earliest workshop posting date unresolved within2024.',status='accepted')
add('G7_UNbTiVZr','UNbTiVZr',2.1,'2024-07-02','https://doi.org/10.1016/j.actamat.2025.121519','U_Nb_Ti_V_Zr_bcc_HEA',229,'First known public primary conference abstract July2,2024 reports novel equimolar UNbTiVZr, Tc2.1K; later ActaMat2025 article says about2K. Conference URL retained in config.')
add('G7_TaWMoReOs_sigma','Ta7W10Mo30Re33Os20',5.9,'2025-07-01','https://doi.org/10.1016/j.jpcs.2025.112630','Ta_W_Mo_Re_Os_sigma_HEA',None,'Primarypublisherabstract explicitly new composition x=0 and bulkmaximumTc5.9K; first publication dated2025 year/month, exact day unresolved. Only x=0 included; unknown per-x compositions/Tc are not invented.')
# This is an old synthesized material despite the genuinely new 2025 Tc result.
add('G7_Zr6RuTe2','Zr6RuTe2',1.1,'1996','https://arxiv.org/abs/2507.02228','Zr6MX2_Zr6CoAl2',189,'2025 primaryp1 explicitly cites1996 Wang/Hughbanks for existing structure; electronic properties newly measured, material not new.',status='excluded_preexisting_material')

# Harmonize grouping without looking at any model predictions. Same parent spans papers.
for r in rows:
 mid=r['material_id']
 if r.get('parent_family_key') in ['SmCaEu_infinite_layer_NiO2','infinite_layer_NiO2'] or mid.startswith('G7_SCE_'):
  r['parent_family_key']='infinite_layer_NiO2'
 if mid.startswith('F4_candidate_bilayer_') or mid in ['G7_lapr_bilayer_285','G7_hybrid_1212','G7_hybrid_2323','G7_LaPrSm_bilayer']:
  r['parent_family_key']='RP_bilayer_hybrid_nickelate'
 if mid in ['G7_Zr6RuBi2','G7_Zr6FeBi2','G7_Zr6RuTe2']:r['parent_family_key']='Zr6MX2_Zr6CoAl2'
 if not r.get('parent_family_key'):
  r['parent_family_key']={'F4_candidate_aupbbi':'MPbBi_layered','F4_candidate_pdpbbi':'MPbBi_layered','F4_candidate_rhpbbi':'MPbBi_layered','F4_candidate_re_mea':'Re_Nb_Ta_V_sigma','F4_candidate_ynbtizrhf':'Y_Nb_Ti_Zr_Hf_bcc_HEA','G7_tinbtan3':'Ti_Nb_Ta_nitride_rocksalt'}.get(mid,'audit_only_'+mid)
 labs[mid]['parent_family_key']=r['parent_family_key']
 labs[mid].setdefault('first_material_report_date',r['first_report_date'])
 labs[mid].setdefault('first_ambient_label_date',r['first_report_date'])
 labs[mid].setdefault('first_report_evidence',r.get('reason',''))
 # screening input excludes labels but retains audit reasons; eval runner must whitelist formula/structure.
 if r['status']=='accepted':
  hit=exact(r['formula']);r['exact_3764_matches']=';'.join(hit)
  assert not hit,(mid,hit)
  assert labs[mid]['y'] in [0,1],mid

fields=['material_id','formula','first_report_date','doi_or_arxiv','cif_url','space_group','status','source','reason','parent_family_key','exact_3764_matches','specific_parent_review']
with (OUT/'predictions.csv').open('w',encoding='utf8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r.get(k,'') for k in fields} for r in rows)
(OUT/'labels.json').write_text(json.dumps(labs,ensure_ascii=False,indent=2),encoding='utf8')
a=[r for r in rows if r['status']=='accepted']
groups=Counter(r['parent_family_key'] for r in a)
m={'status':'awaiting_root_review_not_frozen','screened_material_rows':len(rows),'qualified_n':len(a),'positive_n':sum(labs[r['material_id']]['y']==1 for r in a),'negative_n':sum(labs[r['material_id']]['y']==0 for r in a),'specific_parent_groups_n':len(groups),'group_sizes':dict(groups),'status_counts':dict(Counter(r['status'] for r in rows)),'evaluation':None}
j.update(status=m['status'],reason_for_stop=None,qualified_rows_after_dedup_and_primary_evidence=len(a),qualified_positive_rows=m['positive_n'],candidate_rows_screened=len(rows),group_definition='specific chemical parent and phase; cross-paper composition series merged before predictions; all infinite-layer NiO2 substitution samples grouped together; RP bilayer/hybrid samples conservatively merged',final_review_summary=m)
j['cutoff_boundary']='Official month December2023 conservatively closed2023-12-31; require2024-02-01 or later. Public theory/old material cannot be rescued by new Tc measurement.'
j['conference_primary_sources']={'UNbTiVZr':'https://airdrive.eventsair.com/eventsairwesteuprod/production-iop-public/5963ec00b1a648f68b8a1db6ef914086'}
j['first_report_limitations']='First known report established by finite primary-literature and exact-composition searches, not proof of complete historical absence; nominal compositions and mixed Tc criteria are recorded. Online metadata posting is distinguished from material identity. CIF files not yet collected; SG alone is not a complete structure.'
j['dedup_evidence']={'reference_sha256':hashlib.sha256((ROOT/'F3_full3764/config.json').read_bytes()).hexdigest(),'accepted_exact_formula_matches':[],'specific_parent_necessary_core_witnesses':{core:[{'id':mid,'formula':cases[mid]['material']['formula'],'sg':cases[mid]['material']['structure']['space_group_declared_in_source_CIF']} for mid,d in ref.items() if set(core.split())<=set(d)] for core in ['W Rh Ir Pt','Mo Re Si','U Nb Ti V Zr','Zr Ru Bi','Mg Pd As','Nb Ta Os','Re Os','Ta W Mo Re Os','Ti Nb Ta N']}}
(OUT/'config.json').write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf8')
(OUT/'metrics.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf8')
with (OUT/'run.log').open('a',encoding='utf8') as f:f.write('\n'+datetime.datetime.now(datetime.timezone.utc).isoformat()+' FIRST_MATERIAL_FINAL_REVIEW '+json.dumps(m)+'\n')
print(json.dumps(m,indent=2))
