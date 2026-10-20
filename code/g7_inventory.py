import datetime,json,urllib.request,urllib.parse,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(r'C:\Users\28908\Codex\fung-research\v2-diagnosis')
ns={'a':'http://www.w3.org/2005/Atom'}
q='cat:cond-mat.supr-con AND submittedDate:[202402010000 TO 202610192359]'
url='https://export.arxiv.org/api/query?'+urllib.parse.urlencode({'search_query':q,'start':0,'max_results':2000,'sortBy':'submittedDate','sortOrder':'ascending'})
entries=[]
for start in [0,2000,4000,6000]:
 u='https://export.arxiv.org/api/query?'+urllib.parse.urlencode({'search_query':q,'start':start,'max_results':2000,'sortBy':'submittedDate','sortOrder':'ascending'})
 r=urllib.request.urlopen(u,timeout=90).read(); root=ET.fromstring(r)
 es=root.findall('a:entry',ns)
 for e in es:
  entries.append({k:(e.findtext('a:'+k,namespaces=ns) or '').strip().replace('\n',' ') for k in ['id','title','summary','published','updated']})
 if len(es)<2000: break
p=ROOT/'G7_postcutoff_materials/config.json'; j=json.loads(p.read_text(encoding='utf8'))
j['arxiv_primary_inventory']={'query':q,'source_url':url,'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'entry_n':len(entries),'entries':entries}
p.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf8')
print('entries',len(entries))
for e in entries:
 t=e['title'].lower(); a=e['summary'].lower()
 if any(s in t for s in ['discovery','new ','superconductivity in','superconductivity of','synthesis']) and not any(s in t for s in ['theory','theoretical','calculation','predict','model','qubit','circuit','josephson','vortex','microwave','numerical','hubbard','pressur','graphene','hydride','moir','sr2ruo4','ut e','ute','cuprate']):
  if any(s in a for s in ['synthes','synthesis','resistivity','resistance','specific heat','magnetization','experiment']):
   print(e['id'].rsplit('/',1)[-1],e['published'][:10],e['title'])
