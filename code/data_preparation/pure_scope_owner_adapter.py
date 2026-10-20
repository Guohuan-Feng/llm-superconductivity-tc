"""Pure ROOT-injected scope/bank/query owner adapter; SOURCE only.

No data/file/model/process/LLM loading. Actual execution is exclusively ROOT-owned
and requires a new campaign freeze. The accepted factory/primitive adapter and
5ffe score/selection mathematics are reused. INNER QUERY BASELINE AUTHENTICATION
IS A NEW PROTOCOL: the old kernel hardcodes the original outer HGB reference.
This module extends that binding for honest legitimate-inner/null references;
it must be independently reviewed and cannot be called unchanged 5ffe protocol.
"""
from copy import deepcopy
import hashlib
import json
import math

VERSION = 'pure-scope-owner-adapter/v1'
FACTORY_SHA = '54c3e8587c9e273fa95ad39b8ce5409624bdadb7c925cade5437dfe5959fcbe6'
PRIMITIVE_ADAPTER_SHA = 'f7b5fdb7362682f29971ee4e7652cae1499cbfd3fd3e50f8dfc9d26ff2e1465d'
MATH_SHA = '5ffe51ef90244ce87ef1b0a03f78466a9ff4573226b8d95d68d563d25fef8c01'
FAST_SHA = '364790426c3e308e3e62ff28bb6d084195106fd9fb24fcf3c229d93767fbfe69'
SEMANTIC_SHA = '6a1c6065d03fdb0efd6693d240d57db75aa217b39adb1d53c4577b47b00cabba'
CONTROLS_SHA = '7c4b6cefd7b22a1cc0f546f19cecf5b2241506361988bd84ca7dda0f1965f674'
MAX_INNER_BUILDS = 18
MAX_LIVE_CASES = 32
INNER_REFERENCE_FIELDS = ('reference_protocol', 'policy', 'score_artifact_ref', 'audit_ref',
                          'TRAIN_scope_sha256', 'TRAIN_source_indices_sha256', 'failure_ref_or_null')
BASELINE_BINDING_FIELDS = ('score', 'kind', 'reference', 'TRAIN_source_indices',
                          'TRAIN_scope_sha256', 'failure_ref_or_null')
ROOT_CALLBACK_FIELDS = ('persist_stage', 'independent_audit', 'approve_bank',
                        'authenticate_registration', 'authenticate_baseline')

class ScopeOwnerError(ValueError): pass

def need(ok, message):
    if not ok:
        raise ScopeOwnerError(message)

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode('utf8')).hexdigest()

def clone(value):
    return deepcopy(value)

def sha(value):
    need(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
         'Frozen lowercase SHA256 required')
    return value

def ref(value, *, nullable=False):
    if nullable and value is None:
        return
    need(type(value) is dict and set(value) == {'path', 'sha256'}
         and type(value['path']) is str and bool(value['path']), 'ROOT durable reference required')
    sha(value['sha256'])

def vector(value):
    need(type(value) in (list, tuple) and len(value) == 127
         and all(type(v) is float and math.isfinite(v) for v in value), 'Original binary64 X127 required')
    return list(value)

def decision_score(value):
    need(type(value) is float and math.isfinite(value) and 0.0 <= value <= 1.0,
         'Finite binary64 score in [0,1]')
    return value

def neutral_feature_baseline():
    """No supervised target/reference library is read for TRAIN-pair queries."""
    reference = dict(reference_protocol='scoped-baseline-reference/v1',
        policy='feature_only_class_balanced_null0.5', score_artifact_ref=None, audit_ref=None,
        TRAIN_scope_sha256=None, TRAIN_source_indices_sha256=digest([]), failure_ref_or_null=None)
    return dict(score=0.5, kind='feature_only_null', reference=reference,
                TRAIN_source_indices=[], TRAIN_scope_sha256=None, failure_ref_or_null=None)

def make_inner_reference(*, policy, TRAIN_source_indices, TRAIN_scope_sha256,
                         score_artifact_ref, audit_ref, failure_ref_or_null):
    """ROOT supplies real artifacts; this only makes explicit metadata bindings."""
    return dict(reference_protocol='scoped-baseline-reference/v1', policy=policy,
        score_artifact_ref=clone(score_artifact_ref), audit_ref=clone(audit_ref),
        TRAIN_scope_sha256=TRAIN_scope_sha256,
        TRAIN_source_indices_sha256=digest(TRAIN_source_indices),
        failure_ref_or_null=clone(failure_ref_or_null))

def make_scoped_fast_owner(kernel_api, fast_api):
    """Only query/fallback-reference authentication changes; numeric math inherits 5ffe.

    Trusted adapter authorizes each exact pool/opaque query/X127/group/score/reference
    before issuance. Agent transports never receive this authority or owner object.
    Native old outer owner is created separately and retains its old protocol.
    """
    class ScopedExecutor(kernel_api.TrustedNumericExecutor):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._scope_baseline_authorizations = {}
        def authorize_query_baseline(self, pool, *, opaque_query_id, X127,
                                     held_component_id, baseline_score, baseline_reference):
            self._pool(pool)
            payload = dict(X127_hex=[v.hex() for v in vector(X127)],
                held_component_id=held_component_id, baseline_score_hex=decision_score(baseline_score).hex(),
                baseline_reference=clone(baseline_reference), pool_sha256=pool.payload_sha256)
            key = (pool.nonce, opaque_query_id)
            old = self._scope_baseline_authorizations.get(key)
            need(old is None or canonical(old) == canonical(payload), 'Baseline authorization cannot change')
            self._scope_baseline_authorizations[key] = payload
        def _authorized(self, pool, opaque_id, xs, group, score, reference):
            key = (pool.nonce, opaque_id)
            expected = self._scope_baseline_authorizations.get(key)
            actual = dict(X127_hex=[v.hex() for v in vector(xs)], held_component_id=group,
                baseline_score_hex=decision_score(score).hex(), baseline_reference=clone(reference),
                pool_sha256=pool.payload_sha256)
            need(expected is not None and canonical(expected) == canonical(actual),
                 'Query baseline is not ROOT-authorized for this exact scope and query')
        def issue_query(self, pool, *, opaque_query_id, X127, held_component_id,
                        baseline_score, baseline_reference):
            p = self._pool(pool)
            kernel_api.text(opaque_query_id, 'opaque query')
            xs = vector(X127)
            need(held_component_id in p['partition']['held_component_ids'], 'Query group outside held complement')
            self._authorized(pool, opaque_query_id, xs, held_component_id, baseline_score, baseline_reference)
            return self._issue('numeric_query', pool.partition_id,
                dict(pool_sha256=pool.payload_sha256, opaque_query_id=opaque_query_id, X127=xs,
                     held_component_id=held_component_id, baseline_score=baseline_score,
                     baseline_reference=clone(baseline_reference)))
        def _bound(self, pool, query):
            p = self._pool(pool)
            q = self._open(query, 'numeric_query', pool.partition_id)
            kernel_api.exact(q, ('pool_sha256','opaque_query_id','X127','held_component_id',
                                'baseline_score','baseline_reference'), 'scoped sealed numeric query')
            need(q['pool_sha256'] == pool.payload_sha256
                 and q['held_component_id'] in p['partition']['held_component_ids'], 'Different scoped query/pool')
            self._authorized(pool, q['opaque_query_id'], q['X127'], q['held_component_id'],
                             q['baseline_score'], q['baseline_reference'])
            return p, q
        def forget_query_baseline(self, pool, opaque_query_id):
            self._scope_baseline_authorizations.pop((pool.nonce, opaque_query_id), None)
    class Facade:
        TrustedNumericExecutor = ScopedExecutor
        def __getattr__(self, name):
            return getattr(kernel_api, name)
    return fast_api.build_fast_numeric_owner(Facade(), authenticated_kernel_source_sha256=MATH_SHA)

class PureScopeOwnerAdapter:
    """Inject physically authenticated modules and ROOT-only actual capabilities.

    APIs: factory, primitive_adapter, kernel, fast_owner, semantic, controls.
    Each build uses actual accepted primitive_adapter.build_training_work_product
    and finalize_authenticated_work_product; ROOT independent auditing/approval
    is mandatory. Outer register never mines/audits the old bank again.
    """
    def __init__(self, *, APIs, authenticated_source_pins, ROOT_callbacks,
                 feature_names, unit_dictionary, semantic_source_pins):
        expected = dict(factory=FACTORY_SHA, primitive_adapter=PRIMITIVE_ADAPTER_SHA,
                        kernel=MATH_SHA, fast_owner=FAST_SHA, semantic=SEMANTIC_SHA, controls=CONTROLS_SHA)
        need(type(APIs) is dict and set(APIs) == set(expected), 'Injected API catalogue')
        need(type(authenticated_source_pins) is dict and authenticated_source_pins == expected,
             'ROOT must physically authenticate exact accepted sources first')
        need(type(ROOT_callbacks) is dict and set(ROOT_callbacks) == set(ROOT_CALLBACK_FIELDS)
             and all(callable(c) for c in ROOT_callbacks.values()), 'ROOT audit/authority callbacks required')
        self.api = dict(APIs)
        self.root = dict(ROOT_callbacks)
        self.feature_names = clone(feature_names)
        self.unit_dictionary = clone(unit_dictionary)
        self.semantic_source_pins = clone(semantic_source_pins)
        self.outer_owner = APIs['fast_owner'].build_fast_numeric_owner(
            APIs['kernel'], authenticated_kernel_source_sha256=MATH_SHA)
        self.inner_owner = make_scoped_fast_owner(APIs['kernel'], APIs['fast_owner'])
        self._banks = {}
        self._bank_by_scope = {}
        self._attempted_scope_builds = set()
        self._cases = {}
        self.ledger = dict(supervised_builds_started=0, supervised_builds_returned=0,
            new_bank_pool_registrations=0, outer_bank_capability_rehydrations=0,
            case_capability_issuances=0, case_capability_releases=0,
            independent_audit_callbacks=0, selected_numeric_calls=0,
            native_classifier_fits=0, scalers=0, native_model_queries=0, LLM_calls=0)

    def _persist(self, operation_id, stage, payload):
        receipt = self.root['persist_stage'](operation_id, stage, clone(payload))
        ref(receipt)
        return receipt

    def _runner_scope(self, scope):
        need(type(scope) is dict and 'scope_sha256' in scope, 'Frozen runner TRAIN scope')
        body = {k: v for k, v in scope.items() if k != 'scope_sha256'}
        need(scope['scope_sha256'] == digest(body), 'Runner scope digest differs')
        ids = scope['original_source_indices']
        need(type(ids) is list and ids == sorted(set(ids)) and bool(ids), 'Nonempty unique sorted TRAIN identities')
        need(type(scope['outer_fold']) is int and scope['outer_fold'] in (0,1,2), 'Outer fold')
        return scope

    def build_inner_bank(self, *, runner_scope, bank_role, TRAIN_rows,
                         full_original_group_counts, factory_scope, inner_train_fold):
        """ROOT supplies seven-field original-order primitives and exact eight-field manifest.

        This is ONE future supervised nonparametric build, not a native fit.
        No retries follow any returned work/audit/approval failure. ROOT must
        restore saved unapproved work or finalized bank after a partial failure.
        """
        self._runner_scope(runner_scope)
        need(bank_role in ('inner_full', 'inner_crossfit')
             and type(inner_train_fold) is int and inner_train_fold in (0,1,2)
             and inner_train_fold != runner_scope['outer_fold'], 'Inner bank role/fold')
        need(set(r['source_index'] for r in TRAIN_rows) == set(runner_scope['original_source_indices'])
             and all(r['fold'] == inner_train_fold for r in TRAIN_rows), 'Inner bank exact TRAIN membership')
        need(factory_scope['held_fold'] == runner_scope['outer_fold'], 'Factory outer HELD exclusion')
        scope_hash = runner_scope['scope_sha256']
        need(scope_hash not in self._attempted_scope_builds, 'No repeat of a started scope bank build')
        need(self.ledger['supervised_builds_started'] < MAX_INNER_BUILDS, '18-bank maximum')
        op = 'inner-bank:'+scope_hash
        self._persist(op, 'started', dict(runner_scope=runner_scope, factory_scope=factory_scope,
            bank_role=bank_role, supervised_reference_builds_reserved=1, native_fits=0))
        self._attempted_scope_builds.add(scope_hash)
        self.ledger['supervised_builds_started'] += 1
        try:
            built = self.api['primitive_adapter'].build_training_work_product(
                TRAIN_rows, full_original_group_counts, factory_scope,
                factory_api=self.api['factory'], trusted_scope_sha256=digest(factory_scope))
            self.ledger['supervised_builds_returned'] += 1
            work_ref = self._persist(op, 'work_returned', built)
            # ROOT's independent audit reconstructs statistics from original
            # primitives; it is not another producer bank mining operation.
            audit = self.root['independent_audit'](clone(built), clone(TRAIN_rows),
                clone(full_original_group_counts), clone(factory_scope), clone(work_ref))
            self.ledger['independent_audit_callbacks'] += 1
            need(type(audit) is dict and set(audit) == {'root_binding','certificate',
                 'statistics_audit_sha256','durable_audit_ref'}, 'Actual independent audit receipt')
            ref(audit['durable_audit_ref']); sha(audit['statistics_audit_sha256'])
            finalized = self.api['primitive_adapter'].finalize_authenticated_work_product(
                built['work_product'], built['adapter_receipt'], factory_api=self.api['factory'],
                root_binding=audit['root_binding'], independent_statistics_certificate=audit['certificate'],
                statistics_audit_sha256=audit['statistics_audit_sha256'])
            bank = finalized['finalized_bank']
            bank_ref = self._persist(op, 'finalized', finalized)
            approved = self.root['approve_bank'](clone(bank), clone(audit), clone(bank_ref))
            need(type(approved) is dict and set(approved) == {'approved_rule_digests','durable_approval_ref'},
                 'ROOT approval is separate from review_digest_index')
            ref(approved['durable_approval_ref'])
            return self.register_authenticated_bank(runner_scope=runner_scope, bank_role=bank_role,
                bank=bank, approved_rule_digests=approved['approved_rule_digests'],
                durable_refs=dict(bank_ref=bank_ref, audit_ref=audit['durable_audit_ref'],
                                  approval_ref=approved['durable_approval_ref']),
                inner_train_fold=inner_train_fold)
        except BaseException as exc:
            self._persist(op, 'failed', dict(error_type=type(exc).__name__, error=str(exc)[:1000],
                runner_scope=runner_scope, ledger=clone(self.ledger),
                retry_allowed=False, completed_work_must_be_reused=True))
            raise

    def register_authenticated_bank(self, *, runner_scope, bank_role, bank,
                                    approved_rule_digests, durable_refs, inner_train_fold=None):
        """Reuse accepted outer3 or register newly audited inner18; no mining/audit."""
        self._runner_scope(runner_scope)
        need(bank_role in ('outer_full','inner_full','inner_crossfit'), 'Bank role')
        need(type(durable_refs) is dict and set(durable_refs) == {'bank_ref','audit_ref','approval_ref'}, 'Durable bank refs')
        for r in durable_refs.values():
            ref(r)
        scope_hash = runner_scope['scope_sha256']
        need(scope_hash not in self._bank_by_scope, 'Pool already registered in this adapter')
        ids = [r['source_index'] for r in bank['partition']['TRAIN_rows']]
        need(set(ids) == set(runner_scope['original_source_indices'])
             and len(ids) == len(set(ids)), 'Bank TRAIN membership differs')
        need({r['component_id'] for r in bank['partition']['TRAIN_rows']} == set(runner_scope['original_component_ids']),
             'Bank original groups differ from declared runner scope')
        need(bank['partition']['held_fold'] == runner_scope['outer_fold'], 'Current outerfold differs')
        if bank_role == 'outer_full':
            need(inner_train_fold is None, 'Outer bank has no inner train fold')
        else:
            need(type(inner_train_fold) is int and inner_train_fold in (0,1,2)
                 and inner_train_fold != runner_scope['outer_fold']
                 and all(r['fold'] == inner_train_fold for r in bank['partition']['TRAIN_rows']), 'Singlefold inner bank')
        need(self.root['authenticate_registration'](clone(runner_scope), bank_role, clone(bank),
             clone(approved_rule_digests), clone(durable_refs)) is True, 'ROOT actual bank/approval authentication required')
        owner = self.outer_owner if bank_role == 'outer_full' else self.inner_owner
        pool = owner.issue_pool(bank['partition'], bank['rule_envelopes'],
                                approved_rule_digests=approved_rule_digests)
        token = object()
        envelopes = {e['card']['rule_id']: clone(e) for e in bank['rule_envelopes']}
        need(len(envelopes) == len(bank['rule_envelopes']), 'Distinct rules')
        catalog_by_id = {}
        for rid, envelope in envelopes.items():
            approved = approved_rule_digests[rid]
            card = envelope['card']
            catalog_by_id[rid] = dict(issued_rule_id=rid, card_sha256=approved['card_sha256'],
                statistics_sha256=approved['statistics_sha256'], envelope_sha256=approved['envelope_sha256'],
                support_reference=dict(ref_id='support:'+rid, sha256=digest(card['provenance']['source_indices']),
                                       count=len(card['provenance']['source_indices'])),
                counterexample_reference=dict(ref_id='counter:'+rid, sha256=digest(card['counterexamples']),
                                              count=len(card['counterexamples'])))
        self._banks[token] = dict(scope=clone(runner_scope), role=bank_role, inner_train_fold=inner_train_fold,
            owner=owner, pool=pool, envelopes=envelopes, catalogue=catalog_by_id,
            approved=clone(approved_rule_digests), refs=clone(durable_refs))
        self._bank_by_scope[scope_hash] = token
        key = 'outer_bank_capability_rehydrations' if bank_role == 'outer_full' else 'new_bank_pool_registrations'
        self.ledger[key] += 1
        self._persist('register-bank:'+scope_hash, 'returned', dict(bank_role=bank_role,
            TRAIN_scope_sha256=scope_hash, pool_sha256=pool.payload_sha256, durable_refs=durable_refs,
            bank_builds_in_registration=0))
        return dict(bank_handle=token, partition_id=pool.partition_id,
            TRAIN_source_indices=sorted(ids), scope_sha256=scope_hash,
            approved_bank_ref=clone(durable_refs['bank_ref']))

    def _bank(self, token):
        need(token in self._banks, 'Unregistered ROOT bank handle')
        return self._banks[token]

    def _query_metadata(self, bank, query):
        need(type(query) is dict and set(query) == {'source_index','component_id','fold'}, 'Target-free query identity')
        need(type(query['source_index']) is int and query['source_index'] >= 0
             and type(query['component_id']) is str and query['component_id']
             and type(query['fold']) is int and query['fold'] in (0,1,2), 'Query metadata types')
        scope = bank['scope']
        need(query['source_index'] not in scope['original_source_indices']
             and query['component_id'] not in scope['original_component_ids'], 'Whole own group must be excluded')
        if bank['role'] == 'outer_full':
            need(query['fold'] == scope['outer_fold'], 'Outer query must be original outer HELD')
        elif bank['role'] == 'inner_full':
            need(query['fold'] not in (scope['outer_fold'], bank['inner_train_fold']), 'Inner eval is remaining originalfold')
        else:
            need(query['fold'] == bank['inner_train_fold'], 'Crossfit queries only excluded TRAIN subfold')

    def _baseline(self, bank, query, supplied):
        if bank['role'] == 'inner_crossfit':
            expected = neutral_feature_baseline()
            need(supplied is None or canonical(supplied) == canonical(expected),
                 'TRAIN feature-only queries cannot use a learned self-trained fallback')
            return expected
        need(type(supplied) is dict and set(supplied) == set(BASELINE_BINDING_FIELDS), 'Explicit scoped baseline')
        decision_score(supplied['score'])
        ids = supplied['TRAIN_source_indices']
        need(type(ids) is list and ids == sorted(set(ids)), 'Baseline TRAIN identities')
        if bank['role'] == 'outer_full':
            need(supplied['kind'] == 'original_outer_HGB127'
                 and canonical(supplied['reference']) == canonical(self.api['kernel'].BASELINE_REFERENCE)
                 and ids == bank['scope']['original_source_indices']
                 and supplied['failure_ref_or_null'] is None, 'Exact original outer HGB fallback required')
            sha(supplied['TRAIN_scope_sha256'])
        else:
            reference = supplied['reference']
            need(type(reference) is dict and set(reference) == set(INNER_REFERENCE_FIELDS)
                 and reference['reference_protocol'] == 'scoped-baseline-reference/v1', 'New scoped inner protocol required')
            if supplied['kind'] == 'legitimate_inner_rawX127_HGB':
                need(reference['policy'] == 'legitimate_inner_rawX127_HGB'
                     and ids == bank['scope']['original_source_indices']
                     and reference['TRAIN_scope_sha256'] == supplied['TRAIN_scope_sha256']
                     and reference['TRAIN_source_indices_sha256'] == digest(ids)
                     and supplied['failure_ref_or_null'] is None
                     and reference['failure_ref_or_null'] is None, 'Singlefold-only rawX127 baseline binding')
                # The one deduplicated model scope has a different descriptive
                # name/outer_marker than these two bank scopes. Compare actual
                # full TRAIN identities; ROOT authenticates model hash, weights,
                # parameters and seed. Literal bank/model scope hashes need not
                # be identical and never grant extra label access.
                sha(supplied['TRAIN_scope_sha256'])
                ref(reference['score_artifact_ref']); ref(reference['audit_ref'])
            else:
                need(supplied['kind'] == 'inner_model_failure_null0.5'
                     and reference['policy'] == 'inner_model_failure_null0.5'
                     and supplied['score'].hex() == (0.5).hex() and ids == []
                     and supplied['TRAIN_scope_sha256'] is None and reference['TRAIN_scope_sha256'] is None
                     and reference['TRAIN_source_indices_sha256'] == digest([])
                     and reference['score_artifact_ref'] is None and reference['audit_ref'] is None,
                     'Explicit nonlearned failure null required')
                ref(supplied['failure_ref_or_null'])
                need(reference['failure_ref_or_null'] == supplied['failure_ref_or_null'], 'Missing null failure provenance')
        need(self.root['authenticate_baseline'](clone(bank['scope']), bank['role'],
             clone(query), clone(supplied)) is True, 'ROOT must authenticate actual score/source/legitimate TRAIN scope')
        return clone(supplied)

    def prepare_case(self, bank_token, *, query_metadata, X127, baseline_binding=None):
        """One live capability session, complete applicable catalogue, deterministic first6.

        Supplied X127 is target-free original evidence. A session is reused by
        agent/control callbacks until release_case; a controls batch may release
        after projection and explicitly rehydrate capabilities for later selection.
        Such rehydration is charged separately, never a bank build/model query.
        """
        need(len(self._cases) < MAX_LIVE_CASES, 'Live case capability budget; release completed cases')
        bank = self._bank(bank_token)
        self._query_metadata(bank, query_metadata)
        xs = vector(X127)
        baseline = self._baseline(bank, query_metadata, baseline_binding)
        owner, pool = bank['owner'], bank['pool']
        opaque = digest(dict(partition_id=pool.partition_id, source_index=query_metadata['source_index'],
                             component_id=query_metadata['component_id'], X127_hex=[v.hex() for v in xs]))
        need(not any(c['bank_token'] is bank_token and c['opaque'] == opaque
                     for c in self._cases.values()),
             'Reuse the existing case token; duplicate live scoped query is forbidden')
        if bank['role'] != 'outer_full':
            owner.authorize_query_baseline(pool, opaque_query_id=opaque, X127=xs,
                held_component_id=query_metadata['component_id'], baseline_score=baseline['score'],
                baseline_reference=baseline['reference'])
        query = owner.issue_query(pool, opaque_query_id=opaque, X127=xs,
            held_component_id=query_metadata['component_id'], baseline_score=baseline['score'],
            baseline_reference=baseline['reference'])
        token = object()
        handles = owner.applicable_rules(pool, query)
        by_id = {}
        for handle in handles:
            payload = owner._open(handle, 'applicable_rule', pool.partition_id)
            rid = payload['rule_id']
            need(rid not in by_id and payload['pool_sha256'] == pool.payload_sha256
                 and payload['query_sha256'] == query.payload_sha256, 'Issued applicability binding')
            by_id[rid] = handle
        catalog = [clone(bank['catalogue'][rid]) for rid in sorted(by_id)]
        first6 = [r['issued_rule_id'] for r in sorted(catalog,
                   key=lambda r: (-r['support_reference']['count'], r['issued_rule_id']))[:6]]
        self._cases[token] = dict(bank_token=bank_token, query=query, metadata=clone(query_metadata),
            X127=xs, opaque=opaque, handles=by_id, catalogue=catalog, first6=first6,
            inspections={}, baseline=baseline, numeric_cache={})
        self.ledger['case_capability_issuances'] += 1
        self._persist('case:'+opaque+':'+query.nonce, 'issued', dict(
            query_metadata=query_metadata, partition_id=pool.partition_id, pool_sha256=pool.payload_sha256,
            query_sha256=query.payload_sha256, baseline=baseline, original_math_source_sha256=MATH_SHA,
            baseline_protocol='original_outer' if bank['role']=='outer_full' else 'scoped-baseline-reference/v1',
            approved_applicable_count=len(catalog), first6=first6))
        return token

    def case_binding(self, case_token):
        case = self._case(case_token)
        bank = self._bank(case['bank_token'])
        return dict(partition_id=bank['pool'].partition_id, opaque_query_id=case['opaque'],
            pool_sha256=bank['pool'].payload_sha256, query_sha256=case['query'].payload_sha256,
            approved_rule_catalog=clone(case['catalogue']), first6_rule_ids=list(case['first6']),
            baseline_binding=clone(case['baseline']),
            counterexample_refs=[clone(r['counterexample_reference']) for r in case['catalogue']])

    def _case(self, token):
        need(token in self._cases, 'Case capability released/unissued')
        return self._cases[token]

    def inspect_rule(self, case_token, rid, *, fixed_six_only=True):
        case = self._case(case_token); bank = self._bank(case['bank_token'])
        need(rid in case['handles'] and (not fixed_six_only or rid in case['first6']),
             'Same prefetched six-card policy; broader requests require another frozen estimand')
        if rid not in case['inspections']:
            case['inspections'][rid] = bank['owner'].inspect_applicable(bank['pool'], case['query'], case['handles'][rid])
        return clone(case['inspections'][rid])

    def inspect_six(self, case_token):
        case = self._case(case_token)
        return dict(applicable_catalog=clone(case['catalogue']),
                    inspections=[self.inspect_rule(case_token, rid) for rid in case['first6']])

    def semantic_catalog(self, case_token):
        """Complete applicable semantic locator; no truncation or invented thresholds."""
        case = self._case(case_token); bank = self._bank(case['bank_token'])
        envelopes = [clone(bank['envelopes'][r['issued_rule_id']]) for r in case['catalogue']]
        def trusted_owner_check(envelope, row, partition_id):
            rid = row['issued_rule_id']
            return (partition_id == bank['pool'].partition_id and rid in case['handles']
                    and digest(envelope) == digest(bank['envelopes'][rid])
                    and canonical(row) == canonical(bank['catalogue'][rid]))
        return self.api['semantic'].build_semantic_index(envelopes, clone(case['catalogue']),
            clone(self.feature_names), clone(self.unit_dictionary), partition_id=bank['pool'].partition_id,
            source_pins=clone(self.semantic_source_pins), trusted_owner_check=trusted_owner_check)

    def read_counterexamples(self, case_token, ref_id, *, max_items=3):
        need(type(max_items) is int and 0 <= max_items <= 3, 'Frozen <=3 counterexample samples')
        case = self._case(case_token)
        rows = [r for r in case['catalogue'] if r['counterexample_reference']['ref_id'] == ref_id]
        need(len(rows) == 1 and rows[0]['issued_rule_id'] in case['first6'], 'Same-six counterexample reference')
        rid = rows[0]['issued_rule_id']
        card = self.inspect_rule(case_token, rid)['card']
        return [dict(index=k, value=clone(card['counterexamples'][k]))
                for k in range(min(max_items, len(card['counterexamples'])))]

    def matrix_projection(self, case_token, common969):
        """Both 20/1010 representations derive from identical actually inspected cards."""
        need(type(common969) is list and len(common969) == 969
             and all(type(v) is float and math.isfinite(v) for v in common969), 'Common facts')
        case = self._case(case_token); bank = self._bank(case['bank_token'])
        need([v.hex() for v in common969[:127]] == [v.hex() for v in case['X127']], 'Copied prefix differs')
        inspected = self.inspect_six(case_token)['inspections']
        partition = bank['owner']._pool(bank['pool'])['partition']
        # Exactly the kernel's legitimate TRAIN totals; no second ws balancing.
        weights = dict(original_row_count=len(partition['TRAIN_rows']),
            original_group_count=len({r['component_id'] for r in partition['TRAIN_rows']}),
            W=math.fsum(partition['TRAIN_class_ws_mass']), C=clone(partition['TRAIN_class_ws_mass']))
        bound = dict(bank['scope'], partition_id=bank['pool'].partition_id)
        controls = self.api['controls']
        return dict(KB20=controls.kb20(inspected, bound, weights),
            pair1010=[dict(rule_id=r['rule_id'],
                         values=controls.pair_vector(common969, r, bound, weights)) for r in inspected],
            actual_inspected_rule_ids=[r['rule_id'] for r in inspected],
            complete_inspections_sha256=digest(inspected), no_new_rule_or_ws_conversion=True)

    def execute_selected(self, case_token, selected_ids):
        """Unchanged 5ffe selection math on ROOT-issued handles; IDs only."""
        case = self._case(case_token); bank = self._bank(case['bank_token'])
        need(bank['role'] != 'inner_crossfit', 'Crossfit TRAIN pair capability is feature-only')
        need(type(selected_ids) is list and 1 <= len(selected_ids) <= 3
             and all(type(r) is str and r in case['first6'] for r in selected_ids)
             and len(set(selected_ids)) == len(selected_ids), 'Select at most3 same inspected six IDs')
        ids = sorted(selected_ids)
        for rid in ids:
            self.inspect_rule(case_token, rid)
        key = tuple(ids)
        if key not in case['numeric_cache']:
            self.ledger['selected_numeric_calls'] += 1
            value = bank['owner'].execute_selected(bank['pool'], case['query'],
                                                    [case['handles'][rid] for rid in ids])
            need(value['partition_id'] == bank['pool'].partition_id and value['opaque_query_id'] == case['opaque']
                 and value['pool_sha256'] == bank['pool'].payload_sha256,
                 'Selected numerical result belongs to another scope/query')
            if not value['fallback_used']:
                need(value['query_sha256'] == case['query'].payload_sha256
                     and value['selected_rule_ids'] == ids, 'Selected-set/query binding differs')
            case['numeric_cache'][key] = clone(value)
            self._persist('selected:'+case['opaque']+':'+digest(ids), 'returned', value)
        return clone(case['numeric_cache'][key])

    def release_case(self, case_token):
        case = self._case(case_token); bank = self._bank(case['bank_token'])
        bank['owner'].release_query(bank['pool'], case['query'])
        if bank['role'] != 'outer_full':
            bank['owner'].forget_query_baseline(bank['pool'], case['opaque'])
        del self._cases[case_token]
        self.ledger['case_capability_releases'] += 1

