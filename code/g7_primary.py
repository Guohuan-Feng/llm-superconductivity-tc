import datetime,hashlib,io,json,sys,urllib.request
from pathlib import Path
from pypdf import PdfReader
ROOT=Path(r'C:\Users\28908\Codex\fung-research\v2-diagnosis')
p=ROOT/'G7_postcutoff_materials/config.json'
j=json.loads(p.read_text(encoding='utf8')); docs=j.setdefault('primary_pdf_reads',{})
for a in sys.argv[1:]:
 if a in docs: continue
 u='https://arxiv.org/pdf/'+a
 try:
  b=urllib.request.urlopen(u,timeout=40).read(); rd=PdfReader(io.BytesIO(b))
  docs[a]={'url':u,'sha256':hashlib.sha256(b).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pages':[x.extract_text() or '' for x in rd.pages]}
  p.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf8'); print('read',a,len(rd.pages))
 except Exception as e:print('ERROR',a,str(e))
