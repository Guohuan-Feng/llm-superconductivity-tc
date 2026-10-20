"""F2 single-attempt streaming transport with recoverable CLI reconnects.

Only explicit account-usage exhaustion sets the provider stop immediately.
Final HTTP 429 failures are returned to the finite runner, which owns all
material retries and backoff. No scientific call or file write runs on import.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any

import e2_transport as _e2
from f1_transport import resolve_runtime


TRANSPORT_VERSION = "F2-streaming-reconnect-aware-2.0"
# Inspect these expressions only in CLI/provider diagnostics, never agent text.
# A bare 429, rate limit, reconnect, or generic word 'quota' is not exhaustion.
_USAGE_LIMIT = re.compile(
    r"usage[\s_-]*limit|try\s+again\s+at\b|insufficient[_ -]quota|"
    r"exceeded\s+(?:your\s+)?current\s+quota|"
    r"(?:your|account(?:'s)?)\s+(?:usage\s+)?quota\s+(?:is\s+|has\s+been\s+)?"
    r"(?:exhausted|exceeded|depleted)|"
    r"(?:hit|reached|exceeded)\s+(?:your\s+|the\s+)?(?:account\s+)?usage\s+limit|"
    r"(?:使用|调用|用量)\s*上限|(?:账户|账号|您的|你的).{0,12}(?:额度|配额).{0,8}"
    r"(?:耗尽|用完|不足|已满)|(?:额度|配额)(?:已)?(?:耗尽|用完)",
    re.IGNORECASE,
)
_FINAL_RATE_LIMIT = re.compile(
    r"(?<![\d.])429(?![\d.])|too\s+many\s+requests|rate[\s_-]*limit|"
    r"速率限制|请求过多|频率限制", re.IGNORECASE,
)
_TRY_AGAIN_AT = re.compile(r"try\s+again\s+at\b[^\r\n]*", re.IGNORECASE)
_RESET_KEYS = {"resets_at", "reset_at", "reset_time", "reset_timestamp", "retry_at"}
_FAILURE_WORDS = re.compile(
    r"\berror\b|\bfailed\b|\bfailure\b|\bfatal\b|\bunauthorized\b|"
    r"\bforbidden\b|\breconnecting\b|\bretrying\b|429|rate[\s_-]*limit|"
    r"too\s+many\s+requests", re.IGNORECASE,
)
_COSMETIC_LOG_LEVEL = re.compile(r"(?:^|\s)(?:TRACE|DEBUG|INFO)(?:\s|:)")


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    for key in tuple(environment):
        if key.upper() in {"OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL"}:
            environment.pop(key, None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _diagnostic_event(line: str) -> tuple[str | None, dict[str, Any] | None]:
    """Return diagnostics only; an ordinary answer containing 'usage' is inert."""
    try:
        event = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return None, None
    if not isinstance(event, dict):
        return None, None
    item = event.get("item")
    item = item if isinstance(item, dict) else {}
    if event.get("type") in {"error", "turn.failed", "warning", "turn.warning"}:
        return json.dumps(event, ensure_ascii=False), event
    if item.get("type") in {"error", "warning"}:
        return json.dumps(item, ensure_ascii=False), item
    return None, None


def _reset_values(value: Any) -> list[dict[str, Any]]:
    """Retain numeric server reset fields, with no guessed timezone/conversion."""
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key.casefold() in _RESET_KEYS and isinstance(child, (int, float)) and not isinstance(child, bool):
                found.append({"field": key, "value": child})
            found.extend(_reset_values(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_reset_values(child))
    return found


def _recovery_texts(value: Any) -> list[str]:
    """Keep original message values instead of JSON-escaped diagnostic tails."""
    if isinstance(value, str):
        return [value] if _TRY_AGAIN_AT.search(value) else []
    if isinstance(value, dict):
        return [text for child in value.values() for text in _recovery_texts(child)]
    if isinstance(value, list):
        return [text for child in value for text in _recovery_texts(child)]
    return []


def _final_failure_diagnostic(diagnostics: list[str]) -> str:
    """Classify the final failure, rather than any historical reconnect error.

    The list preserves observed streaming arrival order under the diagnostic
    lock. A turn.failed event is authoritative; otherwise the latest meaningful
    provider/CLI failure is used. Cosmetic stderr INFO/DEBUG/TRACE lines cannot
    overwrite the final diagnostic merely by appearing during process shutdown.
    """
    terminal_turns = [text for text in diagnostics
                      if text.startswith("stdout_provider_error:")
                      and re.search(r'"type"\s*:\s*"turn\.failed"', text)]
    if terminal_turns:
        return terminal_turns[-1]
    meaningful = [text for text in diagnostics
                  if _FAILURE_WORDS.search(text)
                  and not (text.startswith("stderr:") and _COSMETIC_LOG_LEVEL.search(text))]
    return meaningful[-1] if meaningful else ""


def _parse_stream(lines: list[str]) -> dict[str, Any]:
    """Use E2's frozen parser, accepting only additional diagnostic warnings."""
    parsed_lines: list[str] = []
    extra_warnings: list[str] = []
    last_turn_status = None
    for line in lines:
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            parsed_lines.append(line)
            continue
        if not isinstance(event, dict):
            parsed_lines.append(line)
            continue
        kind = event.get("type")
        if kind in {"turn.completed", "turn.failed"}:
            last_turn_status = kind
        item = event.get("item")
        # Warning events are operational diagnostics, never permitted tools.
        if kind in {"warning", "turn.warning"}:
            extra_warnings.append(json.dumps(event, ensure_ascii=False))
            continue
        if isinstance(item, dict) and item.get("type") == "warning":
            event = dict(event)
            event["item"] = {**item, "type": "error"}
            line = json.dumps(event, ensure_ascii=False) + "\n"
        parsed_lines.append(line)
    result = _e2.parse_events("".join(parsed_lines))
    result["warnings"].extend(extra_warnings)
    result["last_turn_status"] = last_turn_status
    return result


def call_once(
    prompt: str, *, schema_path: str | Path, instructions_path: str | Path,
    catalog_path: str | Path, stop_event: threading.Event,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Own exactly one fresh CLI process; no Python retry or call time limit.

    CLI reconnects run to completion. Explicit usage exhaustion and integrity
    failures set the shared event; other final failures are decisions for the
    finite runner. Caller must persist a reservation before invoking this API.
    """
    result: dict[str, Any] = dict(metadata or {})
    result.update({
        "prompt": prompt,
        "request_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "requested_model": _e2.MODEL, "model": _e2.MODEL,
        "reasoning_effort": _e2.REASONING_EFFORT,
        "cli_version": _e2.CLI_VERSION, "transport_version": TRANSPORT_VERSION,
        "input_tokens": None, "output_tokens": None,
        "cached_input_tokens": None, "elapsed_seconds": 0.0,
        "status": "preflight_rejected", "response_text": "",
        "tool_activity": [], "errors": [], "warnings": [],
    })
    started = time.monotonic()
    process: subprocess.Popen[str] | None = None
    readers: list[threading.Thread] = []
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    reader_errors: list[str] = []
    usage_here = threading.Event()
    diagnostic_lock = threading.Lock()
    provider_diagnostics: list[str] = []
    usage_diagnostics: list[str] = []
    recovery_messages: list[str] = []
    reset_values: list[dict[str, Any]] = []
    cancelled = False

    def inspect_diagnostic(diagnostic: str, source: str, structured: Any = None) -> None:
        with diagnostic_lock:
            provider_diagnostics.append(f"{source}: {diagnostic}")
            if not _USAGE_LIMIT.search(diagnostic):
                return
            usage_diagnostics.append(f"{source}: {diagnostic}")
            recovery_messages.extend(_recovery_texts(structured if structured is not None else diagnostic))
            if structured is not None:
                reset_values.extend(_reset_values(structured))
        usage_here.set()
        stop_event.set()
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass

    def consume_stdout() -> None:
        assert process is not None and process.stdout is not None
        try:
            for line in process.stdout:
                stdout_lines.append(line)
                diagnostic, structured = _diagnostic_event(line)
                if diagnostic is not None:
                    inspect_diagnostic(diagnostic, "stdout_provider_error", structured)
        except Exception as exc:
            reader_errors.append(f"stdout {type(exc).__name__}: {exc}")

    def consume_stderr() -> None:
        assert process is not None and process.stderr is not None
        try:
            for line in process.stderr:
                stderr_lines.append(line)
                inspect_diagnostic(line.rstrip(), "stderr")
        except Exception as exc:
            reader_errors.append(f"stderr {type(exc).__name__}: {exc}")

    try:
        if stop_event.is_set():
            result["status"] = "interrupted_due_terminal_stop"
            result["error"] = "Shared provider/integrity stop was set before dispatch"
            return result
        result["runtime"] = resolve_runtime()
        budget = _e2.input_budget(prompt, schema_path=schema_path, instructions_path=instructions_path)
        result["input_budget"] = budget
        if not budget["within_budget"]:
            result["error"] = "Conservative complete-input upper bound exceeds 15000"
            stop_event.set()
            return result
        command = _e2.build_command(
            cwd=_e2.ROOT, schema_path=schema_path,
            instructions_path=instructions_path, catalog_path=catalog_path,
        )
        if stop_event.is_set():
            result["status"] = "interrupted_due_terminal_stop"
            result["error"] = "Shared stop was set during preflight"
            return result
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
            cwd=_e2.checked_path(_e2.ROOT), env=_environment(), bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        result["pid"] = process.pid
        for target, name in ((consume_stdout, "stdout"), (consume_stderr, "stderr")):
            reader = threading.Thread(target=target, name=f"F2-{process.pid}-{name}")
            readers.append(reader)
            reader.start()
        assert process.stdin is not None
        try:
            process.stdin.write(prompt)
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        while process.poll() is None:
            if stop_event.is_set():
                cancelled = not usage_here.is_set()
                try:
                    process.kill()
                except OSError:
                    pass
                break
            time.sleep(0.05)
        result["returncode"] = process.wait()
        for reader in readers:
            reader.join()
        result.update(_parse_stream(stdout_lines))
        result["stderr_tail"] = "".join(stderr_lines)[-8000:]
        result["stream_reader_errors"] = list(reader_errors)
        result["provider_diagnostics"] = list(provider_diagnostics)
        result["final_failure_diagnostic"] = _final_failure_diagnostic(provider_diagnostics)
        result["terminal_diagnostics"] = list(usage_diagnostics)
        result["recovery_time_message"] = recovery_messages[0] if recovery_messages else None
        result["recovery_time_messages"] = list(recovery_messages)
        result["recovery_reset_fields"] = list(reset_values)
        successful = (result["completed"] and bool(result["response_text"])
                      and result["last_turn_status"] == "turn.completed")
        # Integrity controls remain fail-closed even after a completed answer.
        if result["tool_activity"]:
            result["status"] = "tool_activity_rejected"
            stop_event.set()
        elif any(model != _e2.MODEL for model in result["reported_models"]):
            result["status"] = "model_mismatch_rejected"
            stop_event.set()
        elif usage_here.is_set():
            result["status"] = "terminal_usage_limit"
            result["error"] = "Explicit account usage-limit diagnostic detected"
        elif cancelled:
            result["status"] = "interrupted_due_terminal_stop"
            result["error"] = "Cancelled because another call set the shared stop"
        elif result["backend_protocol_errors"] or reader_errors:
            result["status"] = "backend_protocol_rejected"
            stop_event.set()
        elif result["input_tokens"] is not None and result["input_tokens"] > _e2.INPUT_LIMIT:
            result["status"] = "actual_input_budget_violation"
            stop_event.set()
        elif successful:
            # Historical failed websocket attempts, including 429, do not
            # invalidate the CLI's subsequently completed final response.
            result["status"] = "ok"
            if result["errors"] or result["warnings"] or stderr_lines or result["returncode"]:
                result["recovered_cli_diagnostics"] = True
        elif _FINAL_RATE_LIMIT.search(result["final_failure_diagnostic"]):
            result["status"] = "rate_limited_final"
            result["error"] = "CLI exhausted its own retries and no completed response was returned"
        elif result["returncode"]:
            result["status"] = "cli_error"
        elif result["failed_turn"]:
            result["status"] = "backend_error"
        else:
            result["status"] = "incomplete_response"
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            stop_event.set()
            result["status"] = "interrupted_due_terminal_stop"
        elif process is None:
            result["status"] = "preflight_rejected"
            stop_event.set()
        else:
            result["status"] = ("interrupted_due_terminal_stop" if stop_event.is_set() else "transport_error")
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if process is not None:
            if process.poll() is None:
                try:
                    process.kill()
                except OSError:
                    pass
            process.wait()
            for reader in readers:
                reader.join()
            for pipe in (process.stdin, process.stdout, process.stderr):
                if pipe is not None:
                    try:
                        pipe.close()
                    except OSError:
                        pass
            if "stderr_tail" not in result:
                result["stderr_tail"] = "".join(stderr_lines)[-8000:]
        elapsed = time.monotonic() - started
        result["elapsed_seconds"] = elapsed
        result["latency_seconds"] = elapsed
    return result
