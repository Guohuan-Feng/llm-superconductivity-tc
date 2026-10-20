"""Frozen E2 prompts. One identical task/schema prefix starts EVERY call."""
import json
from e2_packets import round_prompt_numbers

SCHEMA = {
    'type':'object', 'additionalProperties':False,
    'properties':{
        'action':{'type':'string','enum':['decide','request']},
        'p':{'type':['number','null'],'minimum':0,'maximum':1},
        'abstain':{'type':'boolean'},
        'selected_rule_ids':{'type':'array','items':{'type':'string'},'maxItems':3},
        'request_rule_ids':{'type':'array','items':{'type':'string'},'maxItems':6},
        'request_counterexamples':{'type':'boolean'},
        'reason':{'type':'string','maxLength':80},
    },
    'required':['action','p','abstain','selected_rule_ids','request_rule_ids','request_counterexamples','reason'],
}

STATIC_PREFIX = '''Scientific task: classify whether the material's measured superconducting critical temperature Tc is STRICTLY greater than 10 kelvin. Positive means Tc>10K; Tc<=10K is negative. The target Tc/label is unknown to you. This is a development experiment. Use only the supplied scientific evidence and your prior chemical knowledge. Do not call tools, run commands, browse, or read files. Structure-derived composition and geometry may describe a proxy structure; do not claim it is a verified measured sample.
Return exactly one JSON object and no surrounding prose. The output schema is identical in every call and arm:
{"action":"decide or request","p":"number in [0,1], or null only while requesting evidence","abstain":"boolean","selected_rule_ids":"array of at most3 rule IDs","request_rule_ids":"array of at most6 rule IDs","request_counterexamples":"boolean","reason":"at most80 characters, preferably concise Chinese"}.
For a decision set action=decide, request_rule_ids=[], request_counterexamples=false and provide p even when abstaining, so forced and fallback evaluations are both possible. The abstain flag independently requests HGB fallback. Do not interpret a class-balanced rule score as a calibrated prevalence probability. Reasons must describe scientific evidence, not this experiment's implementation.
'''

MODEL_INSTRUCTIONS = '''You are a materials-science classifier in a controlled experiment. Follow the user-provided task and JSON schema. You have no tools. Never read local files or look up the target's measured Tc. Only use scientific information explicitly included in the user prompt plus prior chemical knowledge. Return a single concise JSON object.'''

def dumps(x):
    return json.dumps(x, ensure_ascii=False, separators=(',',':'),allow_nan=False)

def make_prompt(arm, target, rule_briefs, evidence, neighbors, history, *, round_number, force_decision=False):
    """Arguments are whitelist-built scientific packets; never a full data dict."""
    if arm=='A1':
        instruction = '''Arm A1: choose rules. Six applicable TRAIN-only rules are ranked by |score-0.5|, restricted to at least20 supporting TRAIN association groups. You may request their cards and/or their TRAIN counterexamples before deciding. The harness will return only requested evidence; at most3 total rounds are allowed. Select the most scientifically relevant1–3 of these six rules. The score used for evaluation is the exact equal-weight mean of your selected rules' supplied scores; do not replace that score with a chemical guess. Even when abstain=true, supply your best1–3 rules as the forced-choice backup, and p equal to their mean. Abstention still sends the operational prediction to HGB. Requests use action=request, p=null, selected_rule_ids=[], and specify request_rule_ids plus whether counterexamples are wanted.'''
        payload = dict(material=target, applicable_rule_summaries=rule_briefs,
                       returned_rule_evidence=evidence, previous_rounds=history)
        if force_decision:
            instruction += ' This is the FINAL allowed round: action must be decide; choose1–3 supplied rules and set abstain honestly.'
    elif arm=='A2':
        instruction = '''Arm A2: direct prediction with rules and neighbors. You receive the same material and the full available evidence for the same six A1 rules, plus five nearest TRAIN materials under Euclidean distance after X127 standardization fitted exclusively on this outer fold TRAIN. Each neighbor includes structure-derived formula, measured Tc, and the Tc>10K label. Weigh chemical relevance and uncertainty, then directly output p(Tc>10K), abstain, and a reason of at most80 characters. You may use0–3 rule IDs in selected_rule_ids to identify rules actually used; direct p need not equal their mean. Do not request more evidence; action=decide.'''
        payload = dict(material=target, applicable_rule_summaries=rule_briefs,
                       full_rule_evidence=evidence, nearest_TRAIN_materials=neighbors,
                       previous_rounds=history)
    elif arm=='A3':
        instruction = '''Arm A3: chemical knowledge only. You receive only the material's structure-derived formula/composition and available structural summary; no rules, neighbors, measured Tc, or labels. Missing space group is genuinely unavailable. Directly output p(Tc>10K), abstain, and a scientific reason of at most80 characters. selected_rule_ids=[], action=decide; do not request evidence.'''
        payload = dict(material=target, previous_rounds=history)
    else:
        raise ValueError(arm)
    return STATIC_PREFIX+'\n'+instruction+'\nRound '+str(round_number)+'\nScientific inputs:\n'+dumps(round_prompt_numbers(payload))
