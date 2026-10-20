"""Atomic JSON persistence for the new supplements only; no import-time writes."""
from __future__ import annotations
import json,os,threading,uuid
from pathlib import Path
from common import ROOT

def checked(path):
    path=Path(path)
    assert path.is_absolute()
    resolved=path.resolve()
    assert resolved.is_relative_to(ROOT.resolve()) and 'onedrive' not in str(resolved).lower()
    for parent in (path,*path.parents):
        assert not parent.is_symlink() and not parent.is_junction(),str(parent)
    return resolved

def write_json(path,obj):
    path=checked(path)
    temp=checked(path.with_name('.'+path.name+'.'+str(os.getpid())+'.'+str(threading.get_ident())+'.'+uuid.uuid4().hex+'.tmp'))
    try:
        with temp.open('x',encoding='utf8',newline='\n') as h:
            json.dump(obj,h,ensure_ascii=False,indent=2,allow_nan=False)
            h.write('\n');h.flush();os.fsync(h.fileno())
        os.replace(temp,path)
    finally:
        if temp.exists():temp.unlink()
