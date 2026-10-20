"""Frozen post-hoc annotation prompts. No calls or writes on import."""
import json

E3_PREFIX = '''Post-hoc E3: assign ONE primary error category to primary_error only.
1=irrelevant rule association drives the wrong decision; not merely an imperfect rule.
2=feature misreading: quote an explicit factual/numeric claim contradicted by a supplied feature.
3=correct chemical caution explicitly overridden by rules. Require both stated caution and rule-driven direction; A1's enforced score mean can establish the latter. A2 is a direct probability: cited rules alone do not establish override.
4=suspected label/structure mismatch supported by case-specific positive discrepancy evidence. ALL targets are proxies: proxy=true, unknown specimen match, class0/Tc0, or a plausible high-Tc family ALONE are insufficient; never assert proven label noise.
5=other/insufficient evidence, including chemical-family or nearest-neighbor overgeneralization, missed doping/phase dependence, or ambiguity.
Use the strongest observable explanation; do not infer hidden thoughts. Secondary-arm text is context only. Evidence is historical prediction-time content; true Tc is exposed only for this post-hoc audit. Rule scores are class-balanced scores, not calibrated natural probabilities. evidence_quote must be an EXACT nonempty substring of primary_error.reason. rationale_cn <=180 Chinese characters, limitation_cn <=100. If category1/2/3/4 lacks its required evidence, choose5 and explain uncertainty. Return key exactly as provided.
'''

E4_PREFIX = '''Post-hoc E4: annotate this EXACT frozen rule predicate, not its predictive performance.
Choose physical_meaning if the literal predicate is a recognizable materials-physics descriptor or a specific QUALIFIED mechanism proxy with a concrete interpretation. Explain the link and its limits; elemental composition does not establish layers, pairing, pressure, phase, doping or Tc. Choose proxy_only if its interpretation is only a dataset association (e.g. rare-element absence, generic coordinates/occupancy, arbitrary geometry/cell-size cutoffs) or no specific mechanism is supported. Generic theoretical_basis text and a high score do not establish physical meaning. Do not add unseen facts or references. This binary label is provisional and EVERY row is human-review-pending; it is not a causal claim. evidence_quote must be one EXACT human_predicate string from conditions. rationale_cn <=120 Chinese characters; limitation_cn <=80. Return key exactly as provided.
'''

def make_e3(row):
    key = row['primary_arm'] + ':' + row['id']
    return {'key': key, 'prompt': E3_PREFIX + json.dumps({'key': key, **row['annotation_packet']}, ensure_ascii=False, separators=(',', ':'))}

def make_e4(row):
    key = row['marker']
    packet = {'key': key, 'fold': row['fold'], 'rule_id': row['rule_id'],
              'card': row['card'], 'TRAIN_n': row['TRAIN_n'],
              'TRAIN_n_groups': row['TRAIN_n_groups'],
              'usage': {arm: {k: v for k, v in usage.items() if k != 'citation_fraction'}
                        for arm, usage in row['usage'].items()}}
    return {'key': key, 'prompt': E4_PREFIX + json.dumps(packet, ensure_ascii=False, separators=(',', ':'))}

def validate_annotation(stage, item, parsed):
    if not isinstance(parsed, dict) or set(parsed) != {'key','category','confidence','rationale_cn','evidence_quote','limitation_cn'}:
        raise ValueError('Response fields violate frozen schema')
    if parsed['key'] != item['key']:
        raise ValueError('Annotation identity mismatch')
    categories = {'1','2','3','4','5'} if stage == 'E3' else {'physical_meaning','proxy_only'}
    if parsed['category'] not in categories or parsed['confidence'] not in {'high','medium','low'}:
        raise ValueError('Invalid annotation category/confidence')
    for key in ('rationale_cn','evidence_quote','limitation_cn'):
        if not isinstance(parsed[key], str) or not parsed[key].strip():
            raise ValueError('Empty annotation field: ' + key)
    lengths = (180,100) if stage == 'E3' else (120,80)
    if len(parsed['rationale_cn']) > lengths[0] or len(parsed['limitation_cn']) > lengths[1]:
        raise ValueError('Annotation rationale/limitation exceeds frozen character limit')
    packet = item['packet']
    quote = parsed['evidence_quote']
    if stage == 'E3':
        if quote not in packet['annotation_packet']['primary_error']['reason']:
            raise ValueError('Evidence quote is not verbatim primary reason')
    elif quote not in [c['human_predicate'] for c in packet['card']['conditions']]:
        raise ValueError('Evidence quote is not an exact rule predicate')
    return parsed
