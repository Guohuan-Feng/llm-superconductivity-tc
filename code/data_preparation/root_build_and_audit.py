from pathlib import Path
import json, hashlib, os
BASE=Path("C:/Users/28908/Codex/fung-research")
def checked(p):
 p=Path(p)
 assert p.is_absolute() and p.resolve(strict=p.exists())==p and p.is_relative_to(BASE), str(p)
 for q in (p,*p.parents):
  assert "onedrive" not in str(q.resolve(strict=q.exists())).casefold()
  assert not q.is_symlink() and not q.is_junction()
  if q.exists(): assert not (getattr(q.stat(follow_symlinks=False),"st_file_attributes",0)&0x400)
 return p

from datetime import datetime, timezone
from collections import Counter
import importlib.util, time
import numpy as np
D=checked(BASE/"workflow-review-2026-10-10/actual-train-rulebank-run-001")
def file_sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(v): return json.dumps(v,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(",",":"))
def digest(v): return hashlib.sha256(canonical(v).encode("utf-8")).hexdigest()
def save_new(p,v):
 p=checked(p);payload=canonical(v).encode("utf-8")
 assert len(payload)<=134217728
 with p.open("xb") as out:out.write(payload);out.flush();os.fsync(out.fileno())
 assert file_sha(p)==digest(v)
 return {"path":str(p),"sha256":digest(v)}
plan=json.loads(checked(D/"ROOT_PLAN.json").read_text(encoding="utf-8"))
assert file_sha(checked(D/"root_build_and_audit.py"))==plan["wrapper_sha256"]
for pin in plan["pins"]:
 assert file_sha(checked(pin["path"]))==pin["sha256"],pin["path"]
state=json.loads(checked(BASE/"classification-loop/LOOP_STATE.json").read_text(encoding="utf-8"))
assert state["active_goal_continue"] and not state["stop_requested"] and state["cycle_in_flight"] is None
lock=checked(BASE/"classification-loop/RUNNER_LOCK.json");assert not lock.exists()
with lock.open("x",encoding="utf-8") as out:
 json.dump({"pid":os.getpid(),"purpose":"three-original-TRAIN-rulebank-builds-and-independent-audits","plan_sha256":file_sha(D/"ROOT_PLAN.json")},out)
 out.flush();os.fsync(out.fileno())
usage={"producer_arrays_decoded":0,"auditor_arrays_decoded":0,"producer_original_TRAIN_labels_decoded":0,
 "auditor_original_TRAIN_labels_decoded":0,"supervised_nonparametric_bank_builds_started":0,
 "supervised_nonparametric_bank_builds_returned":0,"supervised_nonparametric_bank_builds_durable":0,
 "independent_numeric_audits_started":0,"independent_numeric_audits_passed":0,"finalized_reviewable_banks":0,
 "native_classifier_fits":0,"scaler_fits":0,"native_model_queries":0,"new_score_records":0,
 "geometry_records":0,"new_physical_measurements":0,"old_passed_contract_reruns":0}
def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,checked(path));module=importlib.util.module_from_spec(spec)
 spec.loader.exec_module(module);return module
def load_rows(who):
 arrays={}
 with np.load(checked(plan["input"]["path"]),allow_pickle=False) as z:
  for key in plan["allowed_arrays"]:
   arrays[key]=z[key];usage[who+"_arrays_decoded"]+=1
 assert arrays["X127"].shape==(3764,127) and arrays["X127"].dtype==np.dtype("float64")
 assert np.isfinite(arrays["X127"]).all()
 for key in ("source_index","fold","label"):
  assert arrays[key].shape==(3764,) and arrays[key].dtype.kind in "iu"
 assert set(arrays["fold"].tolist())=={0,1,2} and set(arrays["label"].tolist())=={0,1}
 assert arrays["role"].shape==(3764,) and all(str(v)=="TRAIN" for v in arrays["role"])
 assert arrays["component_id"].shape==(3764,) and arrays["component_id"].dtype.kind in "SU"
 rows=[{"source_index":int(arrays["source_index"][i]),"original_row_index":i,
 "component_id":str(arrays["component_id"][i]),"fold":int(arrays["fold"][i]),"role":"TRAIN",
 "X127":arrays["X127"][i].tolist(),"label":int(arrays["label"][i])} for i in range(3764)]
 counts=dict(sorted(Counter(r["component_id"] for r in rows).items()))
 assert len(counts)==1117 and len({r["source_index"] for r in rows})==3764
 groupfold={}
 for r in rows:
  g=r["component_id"]
  assert g not in groupfold or groupfold[g]==r["fold"];groupfold[g]=r["fold"]
 usage[who+"_original_TRAIN_labels_decoded"]+=3764
 return rows,counts
started=time.perf_counter();outputs=[]
try:
 with checked(D/"ROOT_ONCE.json").open("x",encoding="utf-8") as out:
  json.dump({"pid":os.getpid(),"plan_sha256":file_sha(D/"ROOT_PLAN.json"),"maximum_bank_builds":3},out)
 factory=load_module("root_actual_fixed_factory",plan["factory"]["path"])
 adapter=load_module("root_actual_adapter",plan["adapter"]["path"])
 auditor=load_module("root_actual_independent_auditor",plan["auditor"]["path"])
 assert factory.digest(factory.recipe())==plan["recipe_sha256"]
 assert tuple(plan["feature_names"])==factory.FEATURE_NAMES
 producer_rows,producer_counts=load_rows("producer")
 for fold in range(3):
  outdir=checked(D/("fold%d"%fold));outdir.mkdir()
  rows=[r for r in producer_rows if r["fold"]!=fold]
  scope={"partition_id":"shared-packet-engineering-outer%d-TRAIN"%fold,"held_fold":fold,
   "source_container_sha256":plan["input"]["sha256"],"feature_schema_sha256":plan["feature_schema_sha256"],
   "TRAIN_rows":[{k:r[k] for k in ("source_index","component_id","role","fold")} for r in rows],
   "held_component_ids":sorted({r["component_id"] for r in producer_rows if r["fold"]==fold}),
   "TRAIN_original_row_indices":[r["original_row_index"] for r in rows],"rule_factory_recipe_sha256":plan["recipe_sha256"]}
  save_new(outdir/"SCOPE.json",scope)
  usage["supervised_nonparametric_bank_builds_started"]+=1
  save_new(outdir/"BUILD_STARTED.json",{"usage_before_call":dict(usage),"scope_sha256":digest(scope)})
  built=adapter.build_training_work_product(rows,producer_counts,scope,factory_api=factory,trusted_scope_sha256=digest(scope))
  usage["supervised_nonparametric_bank_builds_returned"]+=1
  workref=save_new(outdir/"WORK_PRODUCT.json",built["work_product"])
  receiptreff=save_new(outdir/"ADAPTER_BUILD_RECEIPT.json",built["adapter_receipt"])
  usage["supervised_nonparametric_bank_builds_durable"]+=1
  outputs.append({"fold":fold,"work_product":workref,"adapter_receipt":receiptreff})
 # Distinct reload of authenticated original input, not reuse producer primitives.
 audit_rows,audit_counts=load_rows("auditor")
 for item in outputs:
  fold=item["fold"];outdir=checked(D/("fold%d"%fold));rows=[r for r in audit_rows if r["fold"]!=fold]
  scope=json.loads((outdir/"SCOPE.json").read_text(encoding="utf-8"))
  assert file_sha(item["work_product"]["path"])==item["work_product"]["sha256"]
  work=json.loads(Path(item["work_product"]["path"]).read_text(encoding="utf-8"))
  receipt=json.loads(Path(item["adapter_receipt"]["path"]).read_text(encoding="utf-8"))
  binding={"execution_kind":"ROOT_actual","ROOT_authorization_sha256":file_sha(D/"ROOT_PLAN.json"),
   "auditor_source_sha256":plan["auditor"]["sha256"],"factory_source_sha256":plan["factory"]["sha256"],
   "executor_source_sha256":plan["executor"]["sha256"],"source_container_sha256":plan["input"]["sha256"],
   "feature_schema_sha256":plan["feature_schema_sha256"],"scope_sha256":digest(scope),
   "primitive_rows_sha256":digest(rows),"groupcounts_sha256":digest(audit_counts),"work_product_sha256":digest(work)}
  save_new(outdir/"ROOT_AUDIT_BINDING.json",binding)
  usage["independent_numeric_audits_started"]+=1
  audit=auditor.audit_training_work_product(rows,audit_counts,scope,work,root_binding=binding)
  assert audit["checks_passed"] is True and audit["execution_kind"]=="ROOT_actual" and audit["certificate"] is not None
  auditref=save_new(outdir/"INDEPENDENT_STATISTICS_AUDIT.json",audit)
  certref=save_new(outdir/"INDEPENDENT_STATISTICS_CERTIFICATE.json",audit["certificate"])
  assert file_sha(auditref["path"])==auditref["sha256"] and file_sha(certref["path"])==certref["sha256"]
  usage["independent_numeric_audits_passed"]+=1
  finalized=adapter.finalize_authenticated_work_product(work,receipt,factory_api=factory,root_binding=binding,
   independent_statistics_certificate=json.loads(Path(certref["path"]).read_text(encoding="utf-8")),statistics_audit_sha256=auditref["sha256"])
  bankref=save_new(outdir/"ROOT_REVIEWABLE_BANK.json",finalized)
  usage["finalized_reviewable_banks"]+=1
  item.update(independent_audit=auditref,certificate=certref,bank=bankref,retained_rules=len(finalized["finalized_bank"]["rule_envelopes"]),
   canonical_predicates=work["canonical_candidates"],eligible_median_columns=len(work["eligible_paired_median_columns"]),
   numeric_rule_use_allowed=work["diagnostics"]["numeric_rule_use_allowed"])
 result={"status":"PASS","created_utc":datetime.now(timezone.utc).isoformat(),"wall_seconds":time.perf_counter()-started,
 "purpose":plan["purpose"],"usage":usage,"outputs":outputs,"agent_superiority_established":False,
 "new_blind_cohort":False,"owner_approval_map_issued":False,"whole_cohort_scoring_performed":False}
except Exception as exc:
 result={"status":"FAIL","created_utc":datetime.now(timezone.utc).isoformat(),"wall_seconds":time.perf_counter()-started,
 "error_type":type(exc).__name__,"error":str(exc)[:1500],"usage":usage,"partial_outputs":outputs,
 "automatic_rerun_allowed":False}
finally:
 save_new(D/"ROOT_RESULT.json",result)
 if lock.exists() and json.loads(lock.read_text(encoding="utf-8")).get("pid")==os.getpid():lock.unlink()
print(json.dumps(result,ensure_ascii=False,allow_nan=False))
