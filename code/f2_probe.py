"""One synthetic, tool-disabled F2 availability probe; no scientific material."""
from common import ROOT, safe_path, write_json
from f1_transport import resolve_runtime
from e2_transport import codex_call
from datetime import datetime, timezone
import json
import os
import re


def main():
    out = ROOT / "F2_mask"
    runtime = resolve_runtime()
    prompt = ('Synthetic availability probe only, not a material prediction. Do not use any tools. '
              'Return exactly {"action":"decide","p":0.5,"abstain":true,"selected_rule_ids":[], '
              '"request_rule_ids":[],"request_counterexamples":false,"reason":"状态探测"}.')
    metadata = {"experiment": "F2_mask", "arm": "PROBE", "id": "synthetic_resume_probe_20261017_1",
                "call_id": "synthetic_resume_probe_20261017_1", "is_probe": True,
                "run_index": "resume_probe_1", "attempt": 1, "prompt": prompt,
                "started_utc": datetime.now(timezone.utc).isoformat()}

    def append(record):
        with safe_path(out / "llm_calls.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    append({"event": "call_started", **metadata})
    record = codex_call(prompt, schema_path=ROOT / "e2_response.schema.json",
                        instructions_path=ROOT / "f2_instructions.md",
                        catalog_path=ROOT / "e2_model_catalog.json", metadata=metadata)
    record.update(event="call_completed", finished_utc=datetime.now(timezone.utc).isoformat())
    diagnostic = " ".join(str(record.get(k, "")) for k in ("errors", "warnings", "error", "stderr_tail"))
    usage = re.search(r"usage[ _-]*limit|you(?:'ve| have) hit.{0,30}limit|try again at|usage_limit_reached|insufficient_quota", diagnostic, re.I)
    record["explicit_usage_limit"] = bool(usage)
    append(record)
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    cfg.setdefault("resume_probes", []).append({"call_id": record["call_id"], "status": record["status"],
        "finished_utc": record["finished_utc"], "explicit_usage_limit": bool(usage), "runtime": runtime,
        "errors": record.get("errors", []), "warnings": record.get("warnings", [])})
    write_json(out / "config.json", cfg)
    with (out / "run.log").open("a", encoding="utf-8") as log:
        log.write(record["finished_utc"] + " human-authorized synthetic probe status=" + record["status"] +
                  " explicit_usage_limit=" + str(bool(usage)) + "\n")
    print(json.dumps({k: record.get(k) for k in ("status", "explicit_usage_limit", "response_text", "errors", "warnings", "elapsed_seconds")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
