"""Finite E3/E4 local-Codex annotation transport; no calls on import.

The caller creates the six-file experiment directory and scientific prompts.
Each item has at most two logged generations, ten generations may run at once,
and every reserved generation consumes its attempt even after interruption.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
import threading

from common import ROOT, safe_path
from e2_transport import codex_call, input_budget, MODEL, REASONING_EFFORT, CLI_VERSION


SLOT = 65536
MAX_ATTEMPTS = 2
MAX_CALLS_PER_STAGE = 120
CONCURRENCY = 10
_INTEGRITY_STOPS = {
    'tool_activity_rejected', 'actual_input_budget_violation',
    'model_mismatch_rejected', 'backend_protocol_rejected', 'preflight_rejected',
}
_TERMINAL_TEXT = (
    'usage limit', 'quota exceeded', 'insufficient_quota', 'not authenticated',
    'unauthorized', 'authentication failed', 'rate_limit_exceeded',
)


def _utcnow():
    return datetime.now(timezone.utc).isoformat()


def _hash(prompt):
    return hashlib.sha256(prompt.encode('utf-8')).hexdigest()


def _stage_dir(stage):
    if stage not in ('E3', 'E4'):
        raise ValueError('Only the explicitly authorized E3/E4 stages are supported')
    out = safe_path(ROOT / stage)
    if not out.is_dir():
        raise ValueError('Caller must create the six-file experiment directory first')
    return out


class AnnotationFailure(RuntimeError):
    """Explicit partial result; no exhausted item is silently marked complete."""
    def __init__(self, stage, failures, annotations):
        self.stage = stage
        self.failures = failures
        self.annotations = annotations
        super().__init__(f'{stage} annotation incomplete: {len(failures)} items; {failures}')


def read_calls(stage):
    """Read existing fixed-slot calls without modifying records or starting work."""
    path = _stage_dir(stage) / 'llm_calls.jsonl'
    if not path.exists():
        return []
    records = []
    with path.open('rb') as handle:
        for index, line in enumerate(handle):
            if len(line) != SLOT + 1:
                raise RuntimeError(f'{stage}: incomplete call-log slot {index}; stop for diagnosis')
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise RuntimeError(f'{stage}: corrupt call-log slot {index}; stop for diagnosis') from exc
            if record.get('stage') != stage or record.get('call_id') != index:
                raise ValueError('Persisted call-log stage/index identity mismatch')
            records.append(record)
    return records


class _CallStore:
    def __init__(self, stage, items, validator, log):
        self.stage = stage
        self.out = _stage_dir(stage)
        self.path = safe_path(self.out / 'llm_calls.jsonl')
        self.items = items
        self.validator = validator
        self.log = log
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.consecutive_failures = 0
        self.records = read_calls(stage)
        if not self.path.exists():
            self.path.touch(exist_ok=False)
        if len(self.records) > MAX_CALLS_PER_STAGE:
            raise ValueError('Existing stage exceeds its finite annotation-call cap')
        for index, record in enumerate(self.records):
            key = record.get('key')
            if key not in items:
                raise ValueError('Existing annotation call does not belong to this fixed item set')
            if record.get('original_prompt_hash') != _hash(items[key]['prompt']):
                raise ValueError('Scientific annotation prompt changed during resume')
            if record.get('request_hash') != _hash(record.get('prompt', '')):
                raise ValueError('Existing request hash does not match its saved prompt')
            attempt = record.get('attempt')
            if not isinstance(attempt, int) or isinstance(attempt, bool) or not 1 <= attempt <= MAX_ATTEMPTS:
                raise ValueError('Invalid persisted annotation attempt')
            if record['status'] == 'in_progress':
                record.update(status='interrupted_unknown_completion',
                              error='Reserved generation interrupted; its attempt remains consumed.',
                              finished_utc=_utcnow(), annotation_valid=False)
                self._write(index, record)
        for key in items:
            attempts = [r['attempt'] for r in self.records if r['key'] == key]
            if attempts != list(range(1, len(attempts) + 1)) or len(attempts) > MAX_ATTEMPTS:
                raise ValueError('Persisted attempts must be consecutive and bounded')

    def _write(self, index, record):
        raw = json.dumps(record, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
        if len(raw) > SLOT:
            raise RuntimeError('Full annotation call exceeds fixed-slot capacity; stop without truncation')
        with self.path.open('r+b') as handle:
            handle.seek(index * (SLOT + 1))
            handle.write(raw + b' ' * (SLOT - len(raw)) + b'\n')
            handle.flush()
            os.fsync(handle.fileno())

    def reserve(self, key, prompt):
        with self.lock:
            if self.stop.is_set():
                raise RuntimeError('Finite annotation stage stopped after terminal/repeated failure')
            previous = [record for record in self.records if record['key'] == key]
            attempt = len(previous) + 1
            if attempt > MAX_ATTEMPTS:
                raise RuntimeError('Annotation item has exhausted both attempts')
            if len(self.records) >= MAX_CALLS_PER_STAGE:
                self.stop.set()
                raise RuntimeError('Finite stage annotation-call budget exhausted')
            index = len(self.records)
            record = {
                'call_id': index, 'call_key': f'{self.stage}|{key}|attempt{attempt}',
                'stage': self.stage, 'key': key, 'attempt': attempt,
                'status': 'in_progress', 'annotation_valid': False,
                'started_utc': _utcnow(), 'finished_utc': None,
                'original_prompt_hash': _hash(self.items[key]['prompt']),
                'request_hash': _hash(prompt), 'prompt': prompt,
                'model': MODEL, 'reasoning_effort': REASONING_EFFORT, 'cli_version': CLI_VERSION,
                'input_tokens': None, 'output_tokens': None, 'cached_input_tokens': None,
                'elapsed_seconds': None, 'response_text': None,
            }
            # Serialize before changing in-memory state, so oversize requests
            # never leave a phantom successful reservation.
            self._write(index, record)
            self.records.append(record)
            return index

    def complete(self, index, result):
        with self.lock:
            self.records[index].update(result)
            record = self.records[index]
            record['finished_utc'] = _utcnow()
            self._write(index, record)
            if record.get('annotation_valid'):
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
            diagnostics = ' '.join(str(record.get(key, '')) for key in ('error', 'errors', 'stderr_tail')).lower()
            if record['status'] in _INTEGRITY_STOPS or any(text in diagnostics for text in _TERMINAL_TEXT):
                self.stop.set()
            if self.consecutive_failures >= 10:
                self.stop.set()
            self.log(f"{self.stage}: {record['call_key']} {record['status']}; "
                     f"annotation_valid={record.get('annotation_valid', False)}")

    def accepted(self, key):
        with self.lock:
            records = [dict(record) for record in self.records if record['key'] == key]
        valid = [r for r in records if r.get('annotation_valid')]
        for record in valid:
            if record.get('status') != 'ok' or record.get('transport_status', 'ok') != 'ok':
                raise ValueError('Accepted annotation has a failed transport status')
            parsed = record['parsed_response']
            if parsed.get('key') != key:
                raise ValueError('Persisted annotation response key mismatch')
            self.validator(key, parsed)
        if len(valid) > 1:
            raise ValueError('An annotation item has multiple accepted generations')
        return valid[0]['parsed_response'] if valid else None


def _repair_prompt(original, previous):
    """No new scientific evidence; repair JSON/schema/transport only."""
    failure = previous.get('validation_error') or previous.get('error') or previous.get('status')
    response = previous.get('response_text') or ''
    # Capping the prior malformed answer prevents an accidental token-budget
    # expansion. All supplied scientific evidence remains in original verbatim.
    base = original + '\nFORMAT/TRANSPORT REPAIR ONLY, final allowed attempt. Return the required JSON using the same scientific evidence.\n'
    for cap in (400, 100, 0):
        candidate = base + 'Previous error: ' + str(failure)[:100] + '\nPrevious output excerpt: ' + str(response)[:cap]
        bound = input_budget(candidate, schema_path=ROOT/'stage3_response.schema.json',
                             instructions_path=ROOT/'stage3_instructions.md')
        if bound['within_budget']:
            return candidate
    return base


def annotate(stage, items, validator, log=None):
    """Annotate a finite fixed item list with logged local Codex calls.

    ``items`` is ``[{"key": str, "prompt": str}, ...]``. ``validator(key,
    parsed)`` must raise on an invalid or scientifically unsupported label.
    Return ``{key: parsed_response}``; raise AnnotationFailure carrying explicit
    ``failures`` and valid partial ``annotations`` if any item remains invalid.
    This function is the only entry point that starts generations.
    """
    log = log or (lambda message: None)
    by_key = {}
    for item in items:
        key, prompt = item.get('key'), item.get('prompt')
        if not isinstance(key, str) or not key or not isinstance(prompt, str) or not prompt:
            raise ValueError('Annotation items need nonempty string keys and prompts')
        if key in by_key:
            raise ValueError('Duplicate finite annotation item key')
        by_key[key] = dict(item)
    if not by_key:
        return {}
    if len(by_key) * MAX_ATTEMPTS > MAX_CALLS_PER_STAGE:
        raise ValueError('Item set exceeds the fixed finite stage call cap')
    store = _CallStore(stage, by_key, validator, log)
    schema = safe_path(ROOT / 'stage3_response.schema.json')
    instructions = safe_path(ROOT / 'stage3_instructions.md')
    catalog = safe_path(ROOT / 'e2_model_catalog.json')
    for source in (schema, instructions, catalog):
        if not source.is_file():
            raise ValueError(f'Missing frozen annotation source: {source}')
    annotations = {}
    pending = []
    for key in by_key:
        accepted = store.accepted(key)
        if accepted is not None:
            annotations[key] = accepted
        else:
            pending.append(key)

    def one(key):
        while True:  # Bounded by reserve's two-attempt assertion and this check.
            accepted = store.accepted(key)
            if accepted is not None:
                return accepted
            with store.lock:
                previous = [dict(r) for r in store.records if r['key'] == key]
            if len(previous) >= MAX_ATTEMPTS:
                raise RuntimeError('Both annotation attempts invalid/interrupted')
            prompt = by_key[key]['prompt'] if not previous else _repair_prompt(by_key[key]['prompt'], previous[-1])
            index = store.reserve(key, prompt)
            try:
                result = codex_call(prompt, cwd=ROOT, schema_path=schema,
                    instructions_path=instructions, catalog_path=catalog,
                    timeout_seconds=None,
                    metadata={'stage': stage, 'key': key, 'attempt': len(previous) + 1})
            except Exception as exc:
                result = {'status': 'transport_exception', 'error': f'{type(exc).__name__}: {exc}'}
            result['transport_status'] = result.get('status')
            result['annotation_valid'] = False
            if result.get('status') == 'ok':
                try:
                    parsed = json.loads(result['response_text'])
                    result['parsed_response'] = parsed
                    if not isinstance(parsed, dict) or parsed.get('key') != key:
                        raise ValueError('Annotation JSON response key mismatch')
                    validator(key, parsed)
                    result['annotation_valid'] = True
                except Exception as exc:
                    result['validation_error'] = f'{type(exc).__name__}: {exc}'
                    result['status'] = 'annotation_protocol_error'
            store.complete(index, result)
            if result['annotation_valid']:
                return result['parsed_response']
            if store.stop.is_set():
                raise RuntimeError('Terminal integrity/provider or ten consecutive annotation failures')

    failures = {}
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {executor.submit(one, key): key for key in pending}
        for future in as_completed(futures):
            key = futures[future]
            try:
                annotations[key] = future.result()
            except Exception as exc:
                failures[key] = f'{type(exc).__name__}: {exc}'
    if failures:
        raise AnnotationFailure(stage, failures, annotations)
    return annotations
