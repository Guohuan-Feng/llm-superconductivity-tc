from pathlib import Path
import json,hashlib,os
BASE=Path("C:/Users/28908/Codex/fung-research")
def checked(p):
 p=Path(p)
 assert p.is_absolute() and p.resolve(strict=p.exists())==p and p.is_relative_to(BASE),str(p)
 for q in (p,*p.parents):
  assert "onedrive" not in str(q.resolve(strict=q.exists())).casefold()
  assert not q.is_symlink() and not q.is_junction()
  if q.exists():assert not(getattr(q.stat(follow_symlinks=False),"st_file_attributes",0)&0x400)
 return p

import sys,types,time,struct,traceback,subprocess,shutil,io
from datetime import datetime,timezone
R=BASE/"workflow-review-2026-10-10";D=R/"common-facts-cohort-actual-001"
C=BASE/"classification-loop/LOOP_STATE.json";B=BASE/"classification-loop/CONTINUOUS_RUNBOOK.json";LOCK=BASE/"classification-loop/RUNNER_LOCK.json"
def canon(v):return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)
def sha(v):return hashlib.sha256(v).hexdigest()
def ref(p,h=None):return {"path":str(p),"sha256":h or sha(checked(p).read_bytes())}
def pin(r,decode=False):
 raw=checked(r["path"]).read_bytes();assert sha(raw)==r["sha256"],r["path"]
 return json.loads(raw) if decode else raw
def save(name,v):
 p=checked(D/name);p.parent.mkdir(parents=True,exist_ok=True);checked(p)
 with p.open("xb") as f:f.write((canon(v)+"\n").encode());f.flush();os.fsync(f.fileno())
 return ref(p)
PINS={
"cohort":ref(R/"common-facts-cohort-source-002/common_facts_cohort.py","2c0805ec6e7aca809c58ab253923dba807675b6051802943e8878df719a85144"),
"fixture":ref(R/"common-facts-cohort-source-002/common_facts_cohort_invented_cache_contracts.py","4c109654220e73b9bf9728e248ec88cae27836d6dc1fd2a449c8f664c8e89c00"),
"cohort_SOURCE_review":ref(R/"common-facts-cohort-source-002/INDEPENDENT_CACHE_SOURCE_REVIEW_002.json","6d594c84494edf1a5c491fc1a9f669336eb129ba241c7d9664aac4afedd32ead"),
"common":ref(R/"common-structural-facts-source-001/common_structural_facts.py","69472dfff328a8d7eb90992dc4f5a26a11a79aeef3309ca4536dea5ff9214540"),
"reference":ref(R/"common-facts-reference-source-001/common_facts_reference.py","e303bbee2c0ba3735e8b5791de34409ed3d2c163f33fe565d45681ebc4b623e3")}
ATOM=ref(R/"shared-structure-packet-cohort-001/ROOT_RESULT.json","373cc1dd9a2f9c315ba8d910382699958ac5c422c5571555909ce362da34a750")
CLOSE=ref(R/"shared-structure-packet-cohort-001/ROOT_METADATA_CLOSURE.json","8e92dce88f170f2f24dc04a1187699e8de550926a2b55d9421d5daef6f8c3471")
REG=ref(R/"shared-structure-packet-cohort-001/COHORT_REGISTRATION_AND_HISTORICAL_IDENTITIES.json","0a67a461631732910d491da0dbd0526c62afeae933e27064872b488e9f927cea")
PACKET=ref(R/"shared-structure-packet-cohort-001/shared_structure_packets_fullTRAIN.jsonl","abb3f28f20050b9e6e52d9b0a989a366c03755424e5ded1e86176f1ac7978cef")
OLD_COMMON=ref(R/"common-facts-actual-001/ROOT_RESULT.json","1e042f46bdc4346fdad299116fe5d45800a648e4d1e3f456bc50943aca792f9f")
OLD_ACCEPT=ref(R/"human-pause-2026-10-11-001/CASE_AGENT_PREPARATION_ACCEPTANCE_007.json","410e0b6cc697b15a0fe55b703fbc58a3af83c1065501ffcfcd504932079e2617")
NP_BINS=[
{"path":"C:/Users/28908/Documents/Codex/2026-10-05/tc-performance-push/env/Lib/site-packages/numpy/_core/_multiarray_umath.cp312-win_amd64.pyd","sha256":"a9d5c4202cf109bf24305f3f1180e0509aaf661ba850612bd75df7cfd9f928b8"},
{"path":"C:/Users/28908/Documents/Codex/2026-10-05/tc-performance-push/env/Lib/site-packages/numpy/linalg/_umath_linalg.cp312-win_amd64.pyd","sha256":"15884ab8ded82bc5f834e429a63cc9cfdcc24f61737a5c563efb2c2f274213a8"}]
plan={"version":"ROOT-full-originalTRAIN-common969/v1","source":PINS,"inputs":{"full_atomic_result":ATOM,"atomic_metadata_closure":CLOSE,"registration":REG,"full_packets":PACKET,"completed_common3":OLD_COMMON,"completed3_acceptance":OLD_ACCEPT},"maximum":{"new_invented_groups":2,"invented_mock_common_calls":10,"invented_mock_assembler_calls":1,"packet_decodes":3764,"new_common_started":3761,"new_independent_reference_started":3761,"saved_common_reuses":3,"saved_audit_reuses":3,"output_rows":3764,"output_bytes":2147483648,"npz_output_bytes":67108864,"wall_seconds":1800},"forbidden":{"new_packet_assembly_calls":0,"original_geometry_or_NPZ_reads":0,"old3_common_or_numeric_audit_replays":0,"old_passed_contract_reruns":0,"native_fits_or_queries":0,"scaler_fits":0,"bank_builds_or_rehydrations":0,"LLM_calls":0,"Tc_or_label_member_reads":0,"new_performance_score_records":0,"new_neighbor_searches":0,"new_CIF_parses":0,"physical_experiments":0},"runtime":{"python_version":"3.12.14 (main, Aug 25 2026, 14:01:42) [MSC v.1944 64 bit (AMD64)]","numpy_version":"2.5.3","binary_refs":NP_BINS},"scope":"All3764 original adaptive-developmentTRAIN structural inputs; old3reuse +new3761 same algorithm; not predictions or blind evidence.","output_budget_scope":"output_bytes is JSONL-only; separate npz_output_bytes<=64MiB checked in memory before any NPZ file write","failure_policy":"Retain unavailable geometry and failed independent audit with original row/denominator; NaN vector with valid=False, never synthetic zero; no retry or scope selection."}
usage={k:0 for k in ["new_invented_groups","invented_mock_common_calls","invented_mock_assembler_calls","packet_decodes","new_common_started","new_common_returned","new_independent_reference_started","new_independent_reference_returned","saved_common_reuses","saved_audit_reuses","output_rows","output_bytes","npz_output_bytes","numeric_valid_rows","unavailable_rows",*plan["forbidden"]]}
checked(D);D.mkdir(parents=True,exist_ok=True);checked(D);assert not (D/"ROOT_ONCE.json").exists() and not checked(LOCK).exists()
cr=checked(C).read_bytes();br=checked(B).read_bytes();c=json.loads(cr)
assert c["active_goal_continue"] and not c["stop_requested"] and c["cycle_in_flight"] is None
assert c["current_native_fit_returns"]==1062 and c["current_classification_fit_returns"]==91 and c["current_development_score_records"]==414
assert sys.version==plan["runtime"]["python_version"] and shutil.disk_usage(D).free>=3*1073741824
cmd="Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|codex' -and $_.CommandLine -match 'fung-research.*(classification|case|rulebank|common-facts|packet-cohort)' } | Select-Object ProcessId,ParentProcessId,ExecutablePath | ConvertTo-Json -Compress"
ps=subprocess.run(["powershell.exe","-NoProfile","-Command",cmd],capture_output=True,text=True,check=True).stdout.strip();pids=json.loads(ps) if ps else [];pids=pids if isinstance(pids,list) else [pids]
expected=Path("C:/Users/28908/Documents/Codex/2026-10-05/tc-performance-push/env/Scripts/python.exe")
assert Path(sys.executable)==expected and any(p["ProcessId"]==os.getpid() and p["ParentProcessId"]==os.getppid() for p in pids)
assert all(p["ProcessId"]==os.getpid() or(p["ProcessId"]==os.getppid() and Path(p["ExecutablePath"])==expected) for p in pids)
plan.update(live_control=ref(C,sha(cr)),live_runbook=ref(B,sha(br)),driver_SOURCE=ref(Path(__file__)),launch_pids=pids)
pr=save("ROOT_PLAN.json",plan);save("ROOT_ONCE.json",{"plan":pr,"pid":os.getpid(),"utc":datetime.now(timezone.utc).isoformat(),"replay_allowed":False})
started=time.monotonic();locked=False;error=None;coordinator=None;output=None;npzref=None;summaryref=None;failures=[];max_numeric_error=0.0
try:
 with checked(LOCK).open("xb") as f:f.write((canon({"pid":os.getpid(),"purpose":"fullTRAIN_common969","plan":pr})+"\n").encode());f.flush();os.fsync(f.fileno())
 locked=True
 def live():
  state=json.loads(checked(C).read_bytes());assert state["active_goal_continue"] and not state["stop_requested"],"Human stop"
  assert time.monotonic()-started<=1800,"Wall budget reached"
 def module(r,name):
  raw=pin(r);m=types.ModuleType(name);m.__file__=r["path"];sys.modules[name]=m;exec(compile(raw,m.__file__,"exec"),m.__dict__);return m
 pin(PINS["cohort_SOURCE_review"])
 for r in NP_BINS:
  p=Path(r["path"]);assert p.resolve(strict=True)==p and "onedrive" not in str(p).casefold();assert sha(p.read_bytes())==r["sha256"]
 cohort=module(PINS["cohort"],"root_full_common_cohort_v2")
 common=module(PINS["common"],"root_full_common_calculator")
 reference=module(PINS["reference"],"root_full_common_independent_reference")
 fixtures=module(PINS["fixture"],"root_full_common_new_cache_fixtures")
 assert reference.np.__version__=="2.5.3"
 invented=fixtures.run_new_contract_groups(cohort)
 assert len(invented["groups"])==2 and all(g["passed"] for g in invented["groups"])
 assert invented["mock_common_callbacks"]<=10 and invented["mock_assembly_callbacks"]<=1 and invented["actual_production_callback_calls"]==0
 usage["new_invented_groups"]=2;usage["invented_mock_common_calls"]=invented["mock_common_callbacks"];usage["invented_mock_assembler_calls"]=invented["mock_assembly_callbacks"]
 save("NEW_CACHE_INVENTED_RESULT.json",invented)
 atom=pin(ATOM,True);close=pin(CLOSE,True);assert atom["status"]=="PASS" and atom["all_structures_available"] and close["status"]=="PASS"
 assert atom["counts"]["output_rows"]==3764 and atom["counts"]["new_assembler_returns"]==3752 and atom["counts"]["accepted_packet_reuses"]==12
 data=pin(REG,True);reg=data["registration"];reg["registration_source_ref"]=REG
 hist=data["historical_identities"];assert len(hist)==len(reg["records"])==3764
 old=pin(OLD_COMMON,True);pin(OLD_ACCEPT);assert old["status"]=="PASS" and len(old["cases"])==3
 cached={}
 for case in old["cases"]:
  facts=pin(case["common_fact_output"],True);audit=pin(case["independent_audit"],True)
  assert audit["passed"] and audit["checked_newcolumns"]==842 and audit["checked_masks"]==969
  cached[case["fixed_identity"]["source_index"]]=(case,facts,audit)
 class NoAssembler:
  def assemble_shared_packet(self,*args,**kw):
   usage["new_packet_assembly_calls"]+=1;raise AssertionError("Original packet assembly replay forbidden")
 class CountedCommon:
  def make_common_facts(self,packet):
   usage["new_common_started"]+=1;assert usage["new_common_started"]<=3761
   result=common.make_common_facts(packet);usage["new_common_returned"]+=1;return result
 coordinator=cohort.CommonFactsCohort(reg,assembler_api=NoAssembler(),common_api=CountedCommon(),authenticated_api_source_sha256=cohort.API_PINS)
 np=reference.np;X=np.full((3764,969),np.nan,dtype=np.float64);missing=np.ones((3764,969),dtype=bool);valid=np.zeros(3764,dtype=bool);names=None;units=None
 packets=pin(PACKET).splitlines(keepends=True);assert len(packets)==3764
 op=checked(D/"common_facts_fullTRAIN.jsonl")
 with op.open("xb") as out:
  for i,line in enumerate(packets):
   live();packet=json.loads(line);usage["packet_decodes"]+=1
   row=reg["records"][i];identity=hist[i];idx=row["source_index"]
   assert packet["source_index"]==idx and packet["material_id"]==row["material_id"] and packet["source_lineage"]["identity_manifest_ordinal0"]==i
   assert sha(line)==atom["row_bindings"][i]["packet_line_sha256"]
   raw=packet["raw_structure"];oldX=packet["existing_numeric_facts"]["X127"]["values"];fn=packet["existing_numeric_facts"]["X127"]["feature_names"]
   saved={"packet_artifact_ref":PACKET,"packet_acceptance_ref":ATOM,"packet_line_binary_sha256":sha(line),"packet_canonical_document_sha256":sha(canon(packet).encode()),"ROOT_authenticated":True}
   if idx in cached:
    case,facts,audit=cached[idx]
    assert saved["packet_canonical_document_sha256"]==case["canonical_packet_sha256"]
    saved.update({"common_fact_ref":case["common_fact_output"],"common_fact_acceptance_ref":OLD_ACCEPT,"common_fact_canonical_document_sha256":sha(canon(facts).encode()),"independent_numeric_audit_ref":case["independent_audit"],"independent_numeric_audit_passed":True,"independent_numeric_audit_bound_vector_sha256":case["common_vector_sha256"],"independent_numeric_audit_bound_packet_sha256":case["canonical_packet_sha256"]})
    result=coordinator.reuse_completed_row(identity,oldX,raw,feature_names=fn,saved_packet=packet,saved_facts=facts,saved_refs=saved)
    usage["saved_common_reuses"]+=1;usage["saved_audit_reuses"]+=1
    audit_result={"status":"accepted_prior_audit_reused","prior_audit":case["independent_audit"],"new_numeric_audit_calls":0}
   else:
    result=coordinator.reuse_packet_row(identity,oldX,raw,feature_names=fn,saved_packet=packet,saved_refs=saved)
    audit_result=None
   numeric_ok=result["status"]=="complete"
   if numeric_ok and idx not in cached:
    usage["new_independent_reference_started"]+=1;assert usage["new_independent_reference_started"]<=3761
    try:
     audit_result=reference.audit_common_facts(packet,result["common_facts"]);usage["new_independent_reference_returned"]+=1
     assert audit_result["passed"];max_numeric_error=max(max_numeric_error,audit_result["maximum_newcolumn_absolute_error"])
    except Exception as e:
     numeric_ok=False;audit_result={"status":"failed_new_numeric_audit","type":type(e).__name__,"message":str(e)}
   if numeric_ok:
    facts=result["common_facts"]
    if names is None:names=facts["feature_names"];units=facts["feature_units"]
    assert facts["feature_names"]==names and facts["feature_units"]==units
    X[i]=facts["values"];missing[i]=facts["missing_mask"];valid[i]=True;usage["numeric_valid_rows"]+=1
   else:
    usage["unavailable_rows"]+=1;failures.append({"registered_identity":row,"coordinator_failure":result["failure_diagnostic"],"independent_numeric_audit":audit_result})
   record={"coordinator_record":result,"numeric_valid":numeric_ok,"independent_numeric_audit":audit_result,"original_group_denominator_retained":True}
   rendered=(canon(record)+"\n").encode()
   assert usage["output_bytes"]+len(rendered)<=2147483648,"Write would exceed frozen output budget"
   out.write(rendered);usage["output_rows"]+=1;usage["output_bytes"]+=len(rendered)
   if(i+1)%256==0:print(canon({"stage":"common969_plus_independent_reference","rows":i+1,"total":3764,"new_common":usage["new_common_returned"],"valid":usage["numeric_valid_rows"],"reused3":usage["saved_common_reuses"]}),flush=True)
  out.flush();os.fsync(out.fileno())
 summary=coordinator.finish();summaryref=save("COORDINATOR_ACCOUNTING_SUMMARY.json",json.loads(summary["canonical_json"]))
 output=ref(op);live()
 assert usage["output_rows"]==usage["packet_decodes"]==3764 and usage["saved_common_reuses"]==usage["saved_audit_reuses"]==3
 assert usage["new_common_started"]<=3761 and usage["new_independent_reference_started"]<=3761
 ledger=coordinator.operation_counts();assert ledger["packet_assembly_started"]==ledger["packet_assembly_returned"]==0
 assert ledger["common_fact_started"]==usage["new_common_started"] and ledger["common_fact_returned"]==usage["new_common_returned"]
 for k in plan["forbidden"]:assert usage[k]==0
 # Target-free, explicitly aligned feature export; unavailable vectors retain NaN/False.
 npzp=checked(D/"COMMON_FACTS_FEATURES_ONLY.npz")
 buffer=io.BytesIO()
 np.savez_compressed(buffer,X969=X,missing_mask=missing,valid=valid,material_id=np.asarray([r["material_id"] for r in reg["records"]]),source_index=np.asarray([r["source_index"] for r in reg["records"]],dtype=np.int64),component_id=np.asarray([r["component_id"] for r in reg["records"]]),fold=np.asarray([r["outer_fold"] for r in reg["records"]],dtype=np.int64),feature_names=np.asarray(names),feature_units=np.asarray(units))
 npz_bytes=buffer.getvalue();assert len(npz_bytes)<=67108864,"NPZ would exceed separate frozen cap"
 with npzp.open("xb") as f:f.write(npz_bytes);f.flush();os.fsync(f.fileno())
 usage["npz_output_bytes"]=len(npz_bytes);npzref=ref(npzp);live()
except BaseException as e:error={"type":type(e).__name__,"message":str(e),"traceback":traceback.format_exc()}
finally:
 if locked:
  l=json.loads(checked(LOCK).read_bytes());assert l["pid"]==os.getpid() and l["plan"]==pr;checked(LOCK).unlink()
receipt={"version":plan["version"],"status":"PASS" if error is None else "FAIL","plan":pr,"usage":usage,"coordinator_ledger":coordinator.operation_counts() if coordinator else None,"coordinator_accounting_summary":summaryref,"common_facts_output":output,"features_only_npz":npzref,"failures":failures,"error":error,"maximum_new_numeric_absolute_error":max_numeric_error,"wall_seconds":time.monotonic()-started,"all_originalTRAIN_rows_retained":error is None,"all969_vectors_independently_valid":error is None and usage["numeric_valid_rows"]==3764,"agent_superiority_established":False,"new_accuracy_result":False,"not_a_new_blind_cohort":True,"strong_graph_comparator_fitted":False,"goal_active":True}
save("ROOT_RESULT.json",receipt);print(canon({"status":receipt["status"],"usage":usage,"error":error,"features_only_npz":npzref}))
if error:raise SystemExit(1)
