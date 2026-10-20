"""Stateless, audited local logged-in Codex transport for the frozen E2 protocol.

No scientific call is made on import. Subprocess output stays in memory; the
runner owns the six permitted experiment files and the global call budget.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

ROOT = Path(r"C:\Users\28908\Codex\fung-research\v2-diagnosis")
CODEX_EXE = Path(r"C:\Users\28908\AppData\Local\OpenAI\Codex\bin\e78bad9887fa50cf\codex.exe")
CLI_VERSION = "codex-cli 0.163.0-alpha.10.2"
TRANSPORT_VERSION = "2.1-recovered-cli-warnings"
MODEL = "gpt-6.1-sol"
REASONING_EFFORT = "high"
INPUT_LIMIT = 15_000

# Native shell registration also gates apply_patch in this CLI family. Features
# below remove alternate interpreters, connectors, browsing and file readers.
FALSE_FEATURES = (
    "apps", "plugins", "hooks", "shell_tool", "unified_exec", "shell_snapshot",
    "code_mode", "code_mode_host", "code_mode_only", "multi_agent", "multi_agent_v2",
    "browser_use", "browser_use_external", "computer_use", "image_generation",
    "view_image", "workspace_dependencies", "goals", "memories", "skill_search",
    "sleep_tool", "tool_suggest", "in_app_browser", "in_app_local_automation",
    "recommended_plugins", "remote_plugin", "deferred_executor", "agent_message_board",
)
BASE_OVERRIDES = (
    'model_provider="openai"', 'model_reasoning_effort="high"', 'web_search="disabled"',
    "project_doc_max_bytes=0", "mcp_servers={}",
    "include_apps_instructions=false", "include_environment_context=false",
    "include_environment_context_time=false", "include_permissions_instructions=false",
    "include_collaboration_mode_instructions=false", "skills.include_instructions=false",
    "tools.update_plan.enabled=false", "tools.experimental_request_user_input.enabled=false",
    'history.persistence="none"', "check_for_update_on_startup=false",
    "suppress_unstable_features_warning=true",
)


def checked_path(path: str | Path, *, must_exist: bool = True) -> Path:
    """Resolve junctions/symlinks before allowing any task-controlled destination."""
    resolved = Path(path).resolve(strict=must_exist)
    if any("onedrive" in p.casefold() for p in resolved.parts):
        raise ValueError(f"OneDrive destination is forbidden: {resolved}")
    return resolved


def _toml_string(value: str | Path) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def build_command(*, cwd: str | Path = ROOT, schema_path: str | Path | None = None,
                  instructions_path: str | Path | None = None,
                  catalog_path: str | Path | None = None) -> list[str]:
    cwd = checked_path(cwd)
    if catalog_path is None:
        raise ValueError("E2 requires its explicit frozen tool-disabled model catalog")
    catalog = json.loads(checked_path(catalog_path).read_text(encoding="utf-8-sig"))
    entries = catalog.get("models", [])
    expected = {"slug": MODEL, "shell_type": "disabled", "tool_mode": "direct",
                "apply_patch_tool_type": None, "node_repl_disabled": True,
                "multi_agent_version": "disabled", "experimental_supported_tools": []}
    if (len(entries) != 1 or not isinstance(entries[0], dict)
            or any(entries[0].get(k) != value for k, value in expected.items())):
        raise ValueError("E2 model catalog violates frozen model/tool isolation")
    command = [str(checked_path(CODEX_EXE)), "--no-daemon", "exec", "--ignore-user-config",
               "--ignore-rules", "--ephemeral", "--skip-git-repo-check", "--json",
               "--color", "never", "--model", MODEL, "--cd", str(cwd)]
    overrides = list(BASE_OVERRIDES)
    overrides += [f"features.{name}=false" for name in FALSE_FEATURES]
    if instructions_path is not None:
        overrides.append("model_instructions_file=" + _toml_string(checked_path(instructions_path)))
    if catalog_path is not None:
        overrides.append("model_catalog_json=" + _toml_string(checked_path(catalog_path)))
    for override in overrides:
        command += ["-c", override]
    if schema_path is not None:
        command += ["--output-schema", str(checked_path(schema_path))]
    return command + ["-"]


def _global_instruction_bytes() -> int:
    # project_doc_max_bytes=0 intentionally leaves the user's global storage-only
    # AGENTS guidance. Do not edit it or duplicate credentials into another home.
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    for name in ("AGENTS.override.md", "AGENTS.md"):
        candidate = codex_home / name
        if candidate.is_file():
            raw = candidate.read_bytes()
            if raw.strip():
                return len(raw)
    return 0


def input_budget(prompt: str, *, schema_path: str | Path | None = None,
                 instructions_path: str | Path | None = None,
                 fixed_context_allowance: int = 1500) -> dict[str, Any]:
    """A conservative byte upper bound avoids an unfrozen tokenizer dependency.

    UTF-8 byte-level BPE cannot require more tokens than input bytes. The fixed
    allowance covers role delimiters and other small CLI framing. Root must pass
    a short model_instructions_file so the coding default is never hidden here.
    """
    if instructions_path is None:
        raise ValueError("E2 requires an explicit short model_instructions_file")
    prompt_bytes = len(prompt.encode("utf-8"))
    source_bytes = len(checked_path(instructions_path).read_bytes())
    schema_bytes = len(checked_path(schema_path).read_bytes()) if schema_path else 0
    fixed_bytes = _global_instruction_bytes() + source_bytes + schema_bytes
    bound = prompt_bytes + fixed_bytes + fixed_context_allowance
    return {"method": "utf8_byte_upper_bound_plus_1500_framing", "prompt_bytes": prompt_bytes,
            "known_fixed_context_bytes": fixed_bytes, "input_token_upper_bound": bound,
            "input_limit": INPUT_LIMIT, "within_budget": bound <= INPUT_LIMIT}


def parse_events(stdout: str) -> dict[str, Any]:
    messages: list[str] = []
    activity: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    usage: dict[str, Any] = {}
    event_types: list[str] = []
    bad_lines = 0
    protocol_errors: list[str] = []
    reported_models: list[str] = []
    completed = False
    completion_count = 0
    failed_turn = False
    thread_id = None
    model_version = None
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            bad_lines += 1
            continue
        if not isinstance(event, dict):
            protocol_errors.append("Backend JSON event is not an object")
            continue
        kind = event.get("type", "")
        event_types.append(kind)
        if kind not in {"thread.started", "turn.started", "turn.completed", "turn.failed",
                        "item.started", "item.updated", "item.completed", "error"}:
            protocol_errors.append(f"Unknown backend event type: {kind!r}")
        if kind == "thread.started":
            thread_id = event.get("thread_id")
        if event.get("model"):
            model_version = event["model"]
            reported_models.append(str(model_version))
        if kind == "turn.completed":
            completed = True
            completion_count += 1
            candidate_usage = event.get("usage")
            if not isinstance(candidate_usage, dict):
                protocol_errors.append("Completed turn has no usage object")
            else:
                usage = candidate_usage
        if kind in ("error", "turn.failed"):
            errors.append(str(event.get("message") or event.get("error") or event))
        if kind == "turn.failed":
            failed_turn = True
        item = event.get("item") or {}
        if not isinstance(item, dict):
            protocol_errors.append("Backend item is not an object")
            continue
        item_kind = item.get("type", "")
        if kind.startswith("item.") and not item_kind:
            protocol_errors.append("Backend item has no type")
        if item.get("model"):
            model_version = item["model"]
            reported_models.append(str(model_version))
        if kind == "item.completed" and item_kind == "agent_message":
            if isinstance(item.get("text"), str):
                messages.append(item["text"])
            else:
                protocol_errors.append("Agent message text is not a string")
        if item_kind == "error":
            # The CLI emits these progress/error items for failed websocket
            # attempts before successfully completing the same turn over HTTP.
            # They are diagnostics, not model tool calls. Preserve their text.
            warning = str(item.get("message") or item.get("text") or item.get("error") or item)
            warnings.append(warning)
        # Record any tool activity even on failed or still-started items. This
        # fails closed if the CLI gains a new non-message execution item type.
        if item_kind and item_kind not in ("agent_message", "reasoning", "error"):
            activity.append({"event": kind, "item_type": item_kind,
                             "id": item.get("id"), "status": item.get("status")})
    if completion_count > 1:
        protocol_errors.append("Multiple turns completed in one reserved generation")
    if bad_lines:
        protocol_errors.append("Unparsed backend stdout lines")
    if completed:
        for key in ("input_tokens", "output_tokens", "cached_input_tokens"):
            value = usage.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                protocol_errors.append(f"Missing/invalid usage field: {key}")
    return {"response_text": messages[-1] if messages else "", "messages": messages,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "cached_input_tokens": usage.get("cached_input_tokens", usage.get("cache_read_input_tokens")),
            "usage_raw": usage, "tool_activity": activity, "errors": errors,
            "warnings": warnings, "failed_turn": failed_turn,
            "completion_count": completion_count,
            "completed": completed, "event_types": event_types,
            "backend_protocol_errors": protocol_errors, "reported_models": reported_models,
            "unparsed_stdout_lines": bad_lines, "thread_id": thread_id,
            "model_version_reported": model_version}


def codex_call(prompt: str, *, cwd: str | Path = ROOT,
               schema_path: str | Path | None = None,
               instructions_path: str | Path | None = None,
               catalog_path: str | Path | None = None,
               timeout_seconds: float | None = None,
               metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """One generation, no retry and no local output files; safe for 10 threads.

    Each invocation is a fresh ephemeral context. For round 2/3 the runner
    explicitly includes the earlier transcript, preserving a stable prefix.
    None timeout respects the user's no-runtime-cap requirement. Any retry is
    a separate, budgeted generation owned by the runner, never hidden here.
    """
    budget = input_budget(prompt, schema_path=schema_path, instructions_path=instructions_path)
    result: dict[str, Any] = dict(metadata or {})
    result.update({"request_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                   "requested_model": MODEL, "model": MODEL, "reasoning_effort": REASONING_EFFORT,
                   "cli_version": CLI_VERSION, "transport_version": TRANSPORT_VERSION,
                   "input_budget": budget,
                   "input_tokens": None, "output_tokens": None, "cached_input_tokens": None,
                   "elapsed_seconds": 0.0, "status": "preflight_rejected"})
    if not budget["within_budget"]:
        result["error"] = "Conservative complete-input token upper bound exceeds 15000"
        return result
    command = build_command(cwd=cwd, schema_path=schema_path,
                            instructions_path=instructions_path, catalog_path=catalog_path)
    environment = dict(os.environ)
    # Existing ChatGPT login is the only authentication path. Never print/read
    # auth.json and never accept a possibly inherited professor/API credential.
    for key in tuple(environment):
        if key.upper() in ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL"):
            environment.pop(key, None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.monotonic()
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = subprocess.run(command, input=prompt, text=True, encoding="utf-8",
                                   errors="replace", capture_output=True,
                                   cwd=checked_path(cwd), env=environment,
                                   timeout=timeout_seconds, creationflags=creationflags,
                                   check=False)
        result.update(parse_events(completed.stdout))
        result["returncode"] = completed.returncode
        # Operational diagnostics only, capped; output JSON is already retained
        # as response_text, and hidden reasoning is intentionally not persisted.
        result["stderr_tail"] = completed.stderr[-8000:]
        if result["tool_activity"]:
            result["status"] = "tool_activity_rejected"
        elif any(model != MODEL for model in result["reported_models"]):
            result["status"] = "model_mismatch_rejected"
        elif result["backend_protocol_errors"]:
            result["status"] = "backend_protocol_rejected"
        elif completed.returncode:
            result["status"] = "cli_error"
        elif result["failed_turn"]:
            result["status"] = "backend_error"
        elif not result["completed"] or not result["response_text"]:
            result["status"] = "incomplete_response"
        elif result["input_tokens"] is not None and result["input_tokens"] > INPUT_LIMIT:
            result["status"] = "actual_input_budget_violation"
        else:
            result["status"] = "ok"
            if result["errors"] or result["warnings"]:
                result["recovered_cli_diagnostics"] = True
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Explicit caller per-call timeout expired"
    except Exception as exc:
        result["status"] = "transport_error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_seconds"] = time.monotonic() - started
    result["latency_seconds"] = result["elapsed_seconds"]
    return result
