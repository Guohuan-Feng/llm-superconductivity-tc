"""ROOT-only primitive adapter to the frozen TRAIN factory; source-stage only.

This adapter does not load datasets/models, mine new formulas, issue certificates,
owner approvals, or predictions. ROOT supplies audited assigned TRAIN labels and
source-authenticated original-order primitives, then injects the pinned factory.
The independent auditor must not import this adapter or the factory statistics.
JSON declarations are consistency checks, not authentication of actual input.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import math
import re
import struct

FACTORY_SOURCE_SHA256 = '54c3e8587c9e273fa95ad39b8ce5409624bdadb7c925cade5437dfe5959fcbe6'
EXECUTOR_SOURCE_SHA256 = '5ffe51ef90244ce87ef1b0a03f78466a9ff4573226b8d95d68d563d25fef8c01'
INPUT_SOURCE_SHA256 = 'd34ac344c62067f15ccdd48a97f611d7ed6e14763a11ec6e31a7076df0b39fa2'
ROW_KEYS = ('source_index','original_row_index','component_id','fold','role','X127','label')
SCOPE_KEYS = ('partition_id','held_fold','source_container_sha256','feature_schema_sha256',
              'TRAIN_rows','held_component_ids','TRAIN_original_row_indices','rule_factory_recipe_sha256')
BINDING_KEYS = ('execution_kind','ROOT_authorization_sha256','auditor_source_sha256',
                'factory_source_sha256','executor_source_sha256','source_container_sha256',
                'feature_schema_sha256','scope_sha256','primitive_rows_sha256',
                'groupcounts_sha256','work_product_sha256')
CERTIFICATE_KEYS = ('schema_version','checks_passed','work_product_sha256','partition_id',
                    'source_container_sha256','feature_schema_sha256','recipe_sha256','auditor_source_sha256')

class AdapterContractError(ValueError):
    pass

def need(ok, message):
    if not ok:
        raise AdapterContractError(message)

def exact(value, keys, name):
    need(type(value) is dict and set(value)==set(keys), 'Invalid fields: '+name)

def sha(value, name):
    need(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None, 'Invalid SHA: '+name)
    return value

def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode('utf8')

def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()

def validate_training_primitives(TRAIN_rows, full_original_group_counts, scope, *, trusted_scope_sha256):
    """Check identities/order/full-group denominators; never compute rule scores."""
    exact(scope,SCOPE_KEYS,'ROOT-bound scope')
    sha(trusted_scope_sha256,'scope binding')
    need(digest(scope)==trusted_scope_sha256,'Different frozen scope')
    need(scope['source_container_sha256']==INPUT_SOURCE_SHA256,'Different original input')
    for key in ('feature_schema_sha256','rule_factory_recipe_sha256'):
        sha(scope[key],key)
    need(type(scope['partition_id']) is str and bool(scope['partition_id'].strip()),'Partition ID')
    held=scope['held_fold']
    need(type(held) is int and held in (0,1,2),'Held fold must be 0, 1 or 2')
    need(type(TRAIN_rows) is list and 1<=len(TRAIN_rows)<3764,'Explicit outer TRAIN primitive rows')
    need(type(full_original_group_counts) is dict and len(full_original_group_counts)==1117,
         'Complete original 1117-group identity counts required')
    need(all(type(g) is str and bool(g) and type(n) is int and n>0
             for g,n in full_original_group_counts.items()),'Original group/count types')
    need(sum(full_original_group_counts.values())==3764,'Complete original 3764-row count')
    n=len(TRAIN_rows)
    need(type(scope['TRAIN_rows']) is list and len(scope['TRAIN_rows'])==n
         and type(scope['TRAIN_original_row_indices']) is list and len(scope['TRAIN_original_row_indices'])==n,
         'Scope/primitive lengths')
    source_ids=[]; original_ids=[]; groups=[]; group_folds={}; binary64=hashlib.sha256()
    for k,row in enumerate(TRAIN_rows):
        exact(row,ROW_KEYS,'TRAIN primitive row')
        s,o,g,f=row['source_index'],row['original_row_index'],row['component_id'],row['fold']
        need(type(s) is int and s>=0 and type(o) is int and 0<=o<3764,'Original identity indices')
        need(type(g) is str and bool(g) and g in full_original_group_counts,'Original group ID')
        need(row['role']=='TRAIN' and type(f) is int and f in (0,1,2) and f!=held,'Held/non-TRAIN row')
        need(type(row['label']) is int and row['label'] in (0,1),'Assigned curated binary label only')
        x=row['X127']
        need(type(x) is list and len(x)==127 and all(type(v) is float and math.isfinite(v) for v in x),
             'Finite original binary64 X127; no implicit cast/feature extraction')
        identity={'source_index':s,'component_id':g,'role':'TRAIN','fold':f}
        need(scope['TRAIN_rows'][k]==identity and scope['TRAIN_original_row_indices'][k]==o,
             'Primitive identities/order differ from ROOT scope')
        if g in group_folds:
            need(group_folds[g]==f,'Original group split across folds')
        group_folds[g]=f; source_ids.append(s); original_ids.append(o); groups.append(g)
        binary64.update(struct.pack('<127d',*x))
    need(len(set(source_ids))==n and len(set(original_ids))==n,'Duplicate original identity')
    need(original_ids==sorted(original_ids),'Original array order must be preserved, not source/cache sorting')
    observed=Counter(groups)
    need(all(observed[g]==full_original_group_counts[g] for g in observed),'Incomplete TRAIN group denominator')
    excluded=scope['held_component_ids']
    need(type(excluded) is list and len(excluded)>0 and all(type(g) is str and bool(g) for g in excluded)
         and len(set(excluded))==len(excluded),'Explicit unique held group exclusion set')
    need(set(excluded)==set(full_original_group_counts)-set(observed),
         'Held groups must be exact complete-cohort complement of TRAIN groups')
    return {'scope_sha256':trusted_scope_sha256,'primitive_rows_sha256':digest(TRAIN_rows),
            'groupcounts_sha256':digest(full_original_group_counts),
            'TRAIN_X127_binary64_le_sha256':binary64.hexdigest(),
            'TRAIN_rows':n,'TRAIN_groups':len(observed),'held_groups':len(excluded)}

def check_factory_api(factory_api):
    """ROOT must first authenticate physical factory source bytes through its loader."""
    need(factory_api.FACTORY_VERSION=='TRAIN-X127-weighted-quantile-overlap/v1'
         and factory_api.MAX_CANDIDATES==434 and factory_api.MIN_ROWS==40
         and factory_api.MIN_GROUPS==10,'Different fixed factory contract')

def build_training_work_product(TRAIN_rows, full_original_group_counts, scope, *,
                                factory_api, trusted_scope_sha256):
    """One call equals one supervised nonparametric bank build; native fits = 0."""
    proof=validate_training_primitives(TRAIN_rows,full_original_group_counts,scope,
                                      trusted_scope_sha256=trusted_scope_sha256)
    check_factory_api(factory_api)
    need(scope['rule_factory_recipe_sha256']==factory_api.digest(factory_api.recipe()),'Different factory recipe')
    work=factory_api.mine_training_pool([r['X127'] for r in TRAIN_rows],[r['label'] for r in TRAIN_rows],
        [r['component_id'] for r in TRAIN_rows],[r['source_index'] for r in TRAIN_rows],scope['partition_id'],
        trusted_partition_manifest=scope,trusted_scope_sha256=trusted_scope_sha256)
    need(work['scope']==scope and work['trusted_scope_sha256']==trusted_scope_sha256,'Factory scope changed')
    need(work['generated_candidates']==434 and len(work['quantile_witnesses'])==145,'Fixed candidate arithmetic changed')
    need(work['resource_ledger']['supervised_reference_builds']==1
         and work['claims']['audited_or_owner_approved'] is False,'Unexpected factory authority/ledger')
    receipt={'schema_version':'TRAIN-rulebank-adapter-build/v1',**proof,
        'partition_id':scope['partition_id'],'source_container_sha256':scope['source_container_sha256'],
        'feature_schema_sha256':scope['feature_schema_sha256'],'recipe_sha256':work['recipe_sha256'],
        'factory_source_sha256':FACTORY_SOURCE_SHA256,'executor_source_sha256':EXECUTOR_SOURCE_SHA256,
        'work_product_sha256':digest(work),'supervised_reference_bank_builds_returned':1,
        'native_estimator_fits':0,'native_preprocessing_fits':0,'native_model_queries':0,
        'model_deserializations':0,'held_labels_or_outcomes_consumed':0,
        'owner_approval_map_issued':False,'independent_statistics_certificate_issued':False,
        'durability_is_external_ROOT_IO_fact':True}
    # Work is returned unchanged. It contains every quantile, support, class mass,
    # balanced mass, fixed score, rejected origin and TRAIN counterexample needed
    # by the independent auditor. No second predicate/statistical pass occurs here.
    return {'work_product':work,'adapter_receipt':receipt}

def finalize_authenticated_work_product(work_product, adapter_receipt, *, factory_api,
        root_binding, independent_statistics_certificate, statistics_audit_sha256):
    """Consistency-only finalize after ROOT authenticates physical audit/cert refs.

    This never manufactures a PASS certificate or approved digest map. Hashes in
    review_digest_index are a ROOT review aid, not authority to issue a pool.
    """
    check_factory_api(factory_api); exact(root_binding,BINDING_KEYS,'independent ROOT binding')
    need(root_binding['execution_kind']=='ROOT_actual','Invented/unverified audit cannot finalize actual bank')
    for key in BINDING_KEYS[1:]:
        sha(root_binding[key],key)
    sha(statistics_audit_sha256,'actual statistics audit file')
    need(root_binding['factory_source_sha256']==FACTORY_SOURCE_SHA256
         and root_binding['executor_source_sha256']==EXECUTOR_SOURCE_SHA256,'Different fixed sources')
    for key in ('source_container_sha256','feature_schema_sha256','scope_sha256','primitive_rows_sha256',
                'groupcounts_sha256','work_product_sha256'):
        need(root_binding[key]==adapter_receipt[key],'Independent binding/producer receipt differ: '+key)
    need(adapter_receipt['work_product_sha256']==digest(work_product)
         and adapter_receipt['scope_sha256']==digest(work_product['scope']),'Modified work product/scope')
    cert=independent_statistics_certificate
    exact(cert,CERTIFICATE_KEYS,'actual independent certificate')
    need(cert['auditor_source_sha256']==root_binding['auditor_source_sha256'],'Different independent auditor')
    bank=factory_api.finalize_rule_bank(work_product,independent_statistics_certificate=cert,
        statistics_audit_sha256=statistics_audit_sha256,executor_source_sha256=EXECUTOR_SOURCE_SHA256)
    review=[]
    for envelope in bank['rule_envelopes']:
        review.append({'rule_id':envelope['card']['rule_id'],'card_sha256':digest(envelope['card']),
                       'statistics_sha256':digest(envelope['statistics']),'envelope_sha256':digest(envelope)})
    need(bank['owner_approval_map_issued'] is False and bank['factory_bank_rebuilds']==0,
         'Finalization must not build or approve')
    return {'finalized_bank':bank,'review_digest_index':review,
            'review_digest_index_is_owner_approval':False,'additional_supervised_reference_bank_builds':0,
            'additional_native_estimator_fits':0,'additional_predictions':0}
