"""Evidence-led G7 collection expansion; never calls an evaluated model."""
import csv,datetime,hashlib,io,json,re,urllib.request,zipfile,xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
ROOT=Path(r'C:\Users\28908\Codex\fung-research\v2-diagnosis'); OUT=ROOT/'G7_postcutoff_materials'
j=json.loads((OUT/'config.json').read_text(encoding='utf8')); labels=json.loads((OUT/'labels.json').read_text(encoding='utf8'))
rows=list(csv.DictReader((OUT/'predictions.csv').open(encoding='utf8')))
cases=json.loads((ROOT/'F3_full3764/config.json').read_text(encoding='utf8'))['cases']
def parse(f):
 parts=re.findall(r'([A-Z][a-z]?)([0-9.]*)',f); d={}
 if ''.join(a+b for a,b in parts)!=f:raise ValueError(f)
 for a,b in parts:d[a]=d.get(a,0)+float(b or 1)
 s=sum(d.values());return {a:v/s for a,v in d.items()}
ref={mid:{x['element']:x['atomic_fraction'] for x in c['material']['composition']} for mid,c in cases.items()}
def matches(f):
 d=parse(f)
 return [mid for mid,r in ref.items() if set(d)==set(r) and all(abs(d[e]-r[e])<1e-7 for e in d)]
def parent_witness(required,forbidden=()):
 req=set(required.split()); bad=set(forbidden.split()) if isinstance(forbidden,str) else set(forbidden)
 return [mid for mid,r in ref.items() if req<=set(r) and not (bad&set(r))]
inv={e['id'].rsplit('/',1)[-1].split('v')[0]:e for e in j['arxiv_primary_inventory']['entries']}
def add(mid,formula,tc,aid,parent,required,sg=None,note='',status='accepted',date=None,upper=None):
 hit=matches(formula)
 if hit:status='excluded_exact_3764_reduced_formula'
 rec={'material_id':mid,'formula':formula,'first_report_date':date or inv.get(aid,{}).get('published','')[:10],
      'doi_or_arxiv':'https://arxiv.org/abs/'+aid,'cif_url':'','space_group':str(sg or ''),'status':status,'source':'primary literature + stored PDF/metadata',
      'reason':note,'parent_family_key':parent,'exact_3764_matches':';'.join(hit),'specific_parent_review':required}
 # repeated script execution replaces by stable id, never duplicates a row.
 old=next((i for i,x in enumerate(rows) if x['material_id']==mid),None)
 if old is None:rows.append(rec)
 else:rows[old]=rec
 labels[mid]={'tc_k':tc,'y':int(tc>10) if tc is not None else (0 if upper is not None and upper<=10 else None),'tc_upper_bound_k':upper,
  'pressure_gpa':0,'pressure_semantics':'measurement at ambient/no applied pressure; synthesis pressure separately described in source',
  'tc_criterion':note,'status':status,'label_source':rec['doi_or_arxiv'],'parent_family_key':parent}

# Source-data table, not digitized or inferred Tc. The paper reports nominal
# Sm(0.95-x)Ca0.05Eu(x)NiO2; XLSX Fig 1 sheet c dot gives x and onset Tc.
u='https://ndownloader.figshare.com/files/63139627'; b=urllib.request.urlopen(u,timeout=40).read(); z=zipfile.ZipFile(io.BytesIO(b)); ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
ss=[''.join(s.itertext()) for s in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si',ns)]
raw=[]
for row in ET.fromstring(z.read('xl/worksheets/sheet5.xml')).findall('.//m:sheetData/m:row',ns)[1:]:
 d={}
 for c in row.findall('m:c',ns):
  v=c.findtext('m:v',namespaces=ns)
  if c.attrib.get('t')=='s':v=ss[int(v)]
  d[re.sub('[0-9]','',c.attrib['r'])]=v
 raw.append(d)
 for key in ['A','B']:
  if key not in d:break
 else:
  x=round(float(d['A']),4); tc=None if d['B']=='--' else float(d['B']); f=f'Sm{0.95-x:.4f}Ca0.05'+(f'Eu{x:g}' if x else '')+'NiO2'
  add('G7_SCE_'+f'{x:.4f}',f,tc,'2508.14666','SmCaEu_infinite_layer_NiO2','No infinite-layer NiO2 parent in 3764; original nickelates are perovskite LaNiO3 or RP monolayer La2NiO4, not NiO2',123,
   note='onset Tc from author Fig 1 XLSX c dot; -- means no transition down to 2 K, stored as upper bound rather than exact zero',upper=2 if tc is None else None)
j['author_source_data']={'doi':'10.6084/m9.figshare.31861684','file_url':u,'file_sha256':hashlib.sha256(b).hexdigest(),'sheet':'c dot','records':raw,'criterion':'Tc onset per paper; no-transition -- retained as bound <=2 K; no regression zero invented'}

# Explicit Tc and composition measurements from the primary paper text/tables.
manual=[
 ('G7_V4Ti2W','V4Ti2W',5.0,'2403.05115','V_Ti_W_bcc_MEA','V Ti W',229,'first report bulk type-II; abstract and resistivity/magnetization/heat capacity'),
 ('G7_Mg4Pd7As6','Mg4Pd7As6',5.45,'2408.06813','Mg4Pd7As6_U4Re7Si6','Mg Pd As',229,'new compound; bulk Tc in abstract; U4Re7Si6 structure'),
 ('G7_Zr6RuBi2','Zr6RuBi2',4.9,'2410.16684','Zr6M_Bi2_Zr6CoAl2','Zr Ru Bi',189,'new Zr6CoAl2-type Ru-Bi variant; abstract bulk Tc'),
 ('G7_Zr6FeBi2','Zr6FeBi2',1.4,'2410.16684','Zr6M_Bi2_Zr6CoAl2','Zr Fe Bi',189,'new Zr6CoAl2-type Fe-Bi variant; abstract bulk Tc'),
 ('G7_BaClNbS2','Ba0.75ClNbS2',1.0,'2411.12231','BaCl_H_NbX2_superlattice','Ba Cl Nb S',None,'new BaCl spacer superlattice; abstract resistivity and magnetization Tc'),
 ('G7_BaClNbSe2','Ba0.75ClNbSe2',1.25,'2411.12231','BaCl_H_NbX2_superlattice','Ba Cl Nb Se',None,'new BaCl spacer superlattice; abstract resistivity and magnetization Tc'),
 ('G7_WBe2','WBe2',1.05,'2411.17504','WBe2_hexagonal','W Be',194,'bulk ambient Tc onset1.05 K, zero0.86 K; known Be-rich WBe13/WBe22 are distinct phases'),
 ('G7_Nb2TiW','Nb2TiW',4.86,'2412.08361','Nb_Ti_W_MEA','Nb Ti W',229,'arc-melted BCC MEA, bulk Tc via transport/magnetization/specific heat'),
 ('G7_Nb2TiMo','Nb2TiMo',3.22,'2412.08361','Nb_Ti_Mo_MEA','Nb Ti Mo',229,'arc-melted BCC MEA, bulk Tc via transport/magnetization/specific heat'),
 ('G7_Al13Os4','Al13Os4',5.47,'2501.00554','Al13Os4_decagonal_approximant','Al Os',None,'monoclinic approximant; transport/magnetization/specific heat/muSR bulk Tc'),
 ('G7_ScTiVNbCu','Sc0.2Ti0.2V0.2Nb0.2Cu0.2',7.21,'2501.01887','ScTiVNbCu_bcc_HEA','Sc Ti V Nb Cu',229,'equiatomic light HEA, bulk Tc confirmed by transport/magnetization/heat capacity'),
 ('G7_HEA_antimonide','Ru0.2Rh0.2Pd0.2Ir0.2Pt0.2Sb',2.15,'2503.23692','HEA_MPtSb_NiAs','Ru Rh Pd Ir Pt Sb',194,'new high-entropy antimonide, bulk transition x0.2, abstract'),
 ('G7_Ta4Rh2C','Ta4Rh2C',6.4,'2506.02209','Ta4Rh2C_eta_carbide','Ta Rh C',227,'reported Ta4Rh2C1-delta, delta unresolved: nominal formula C1 retained; bulk Tc abstract'),
 ('G7_NbTaOs2','NbTaOs2',2.4,'2506.12449','NbTaOs2_alphaMn','Nb Ta Os',217,'Table I Tc2.4(2) K; original reference has no ternary Nb-Ta-Os'),
 ('G7_TaPtSi','TaPtSi',3.64,'2602.22793','TaPtSi_TiNiSi','Ta Pt Si',62,'Table S1 magnetic Tc3.64(1) K; heat capacity3.57 K'),
]
for mid,f,tc,aid,pkey,req,sg,note in manual:
 hits=parent_witness(req)
 status='accepted' if not hits else 'pending_specific_parent_family'
 add(mid,f,tc,aid,pkey,'required chemical core '+req+'; TRAIN matching-core rows '+str(hits),sg,note,status)
for f,tc in [
 ('Re0.3Os0.3Ta0.1Nb0.1W0.1Zr0.1',4.0),('Re0.35Os0.35Ta0.1W0.1Zr0.1',4.4),('Re0.35Os0.35Nb0.1W0.1Zr0.1',4.5),('Re0.35Os0.35Mo0.1Ta0.1Zr0.1',4.6),('Re0.4Os0.4Ta0.1Zr0.1',5.0)]:
 add('G7_RO1_'+f,f,tc,'2504.02556','ReOs_alphaMn_MEA','Only Re/Os TRAIN row Be44Os0.1Re1.9 is Fd-3m beryllide, not alpha-Mn; no Re-Os alpha-Mn parent in 3764',217,'Conclusion p19 gives composition-specific Tc and newly synthesized identity')
for f,tc in [('Re3.5Os3.5Ta0.5Hf0.5Nb3',4.52),('Re3Os3Ta0.5Hf0.5Nb3',4.20),('Re3.5Os3.5Mo0.5Hf0.5Nb3',4.94),('Re3Os3Mo0.5Hf0.5Nb3',4.77),('Re3.5Os3.5Mo0.5W0.5Nb3',5.11)]:
 add('G7_RO2_'+f,f,tc,'2508.05010','ReOs_alphaMn_MEA','Only Re/Os TRAIN row is Be44Os0.1Re1.9 Fd-3m, not alpha-Mn; no Re-Os alpha-Mn parent in 3764',217,'Table2 and resistivity p7 provide composition-specific Tc and newly synthesized identity')

# Existing nickelates were pending because parent checks had not been performed.
# The actual original packets show no bilayer/hybrid or infinite-layer NiO2 parent.
nickel_proof=[{'id':mid,'formula':c['material']['formula'],'sg':c['material']['structure']['space_group_declared_in_source_CIF']} for mid,c in cases.items()
 if {'Ni','O'}<=set(ref[mid]) and 'Cu' not in ref[mid] and len(ref[mid])<=5]
j['specific_nickelate_parent_audit']={'reference_rows':nickel_proof,'finding':'No bilayer RP n=2, hybrid 1212/2323, quintuple square-planar or infinite-layer NiO2 parent. NiO225, LaNiO3167 and La2NiO4138 are distinct specific parents. Chemical family Ni-O alone is not used as exclusion.'}
for r in rows:
 mid=r['material_id']
 if mid.startswith('F4_candidate_bilayer_') or mid.startswith('F4_candidate_trilayer_'):
  r['status']='accepted';labels[mid]['status']='accepted';r['specific_parent_review']='No matching layered parent in original 3764; see config specific_nickelate_parent_audit'
  r['parent_family_key']='bilayer_RP_Ni2O7' if 'bilayer' in mid else 'square_planar_trilayer_Ni3O8'
 if mid in {'G7_lapr_bilayer_285','G7_hybrid_1212','G7_hybrid_2323','G7_hybrid_1313'}:
  # Exact La:Pr fractions for hybrid notation are not fully specified in the
  # abstract; retain those three pending rather than invent a numeric formula.
  if mid=='G7_lapr_bilayer_285':
   r['status']='accepted';labels[mid]['status']='accepted';labels[mid]['tc_k']=45.0
   r['parent_family_key']='bilayer_RP_Ni2O7';r['specific_parent_review']='No bilayer parent in 3764; original paper p3-4 gives onset45K'
  else:r['status']='pending_exact_La_Pr_fraction';labels[mid]['status']=r['status']
 if mid=='F4_candidate_la6ni5o12':
  # Early publication before GPT cutoff is irrelevant for Llama cutoff. The
  # material first-report date is verified separately before accepting.
  r['status']='pending_earliest_material_report_after_2024_02';labels[mid]['status']=r['status']
 if mid=='F4_candidate_nd_eu03':
  r['status']='pending_earliest_material_report_after_2024_02';labels[mid]['status']=r['status']

fields=['material_id','formula','first_report_date','doi_or_arxiv','cif_url','space_group','status','source','reason','parent_family_key','exact_3764_matches','specific_parent_review']
with (OUT/'predictions.csv').open('w',newline='',encoding='utf8') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r.get(k,'') for k in fields} for r in rows)
(OUT/'labels.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2),encoding='utf8')
qual=[r for r in rows if r['status']=='accepted' and r['first_report_date']>='2024-02-01']
j.update({'status':'collection_in_progress','model_cutoff_date':'2023-12-31','eligibility_window_start':'2024-02-01','candidate_rows_screened':len(rows),'qualified_rows_after_dedup_and_primary_evidence':len(qual),'qualified_positive_rows':sum(labels[r['material_id']]['y']==1 for r in qual),'evaluation_started':False})
(OUT/'config.json').write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf8')
m={'status':'collection_in_progress','candidate_rows_screened':len(rows),'qualified_n':len(qual),'qualified_positive_n':sum(labels[r['material_id']]['y']==1 for r in qual),'qualified_negative_n':sum(labels[r['material_id']]['y']==0 for r in qual),'status_counts':dict(Counter(r['status'] for r in rows)),'evaluation':None,'label_file_sha256':hashlib.sha256((OUT/'labels.json').read_bytes()).hexdigest()}
(OUT/'metrics.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf8')
with (OUT/'run.log').open('a',encoding='utf8') as f:f.write('\n'+datetime.datetime.now(datetime.timezone.utc).isoformat()+' EXPAND '+json.dumps(m)+'\n')
print(json.dumps(m,indent=2))
