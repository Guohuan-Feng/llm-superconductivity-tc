"""Transparent structure-derived Python rules and TRAIN-only feedback. No I/O."""
from __future__ import annotations
import ast
from collections import Counter
import hashlib
import json
import math
import time
import numpy as np

CONTRACT = '''Each s is a label-free dictionary derived from a pymatgen Structure.
Keys: n_sites,n_elements,volume_per_atom_A3,density_g_cm3,min_distance_A;
mean_Z,mean_mass,mean_X,std_X,range_X; metal_fraction,transition_fraction,
lanthanide_fraction,actinide_fraction,anion_fraction;
H_fraction,O_fraction,N_fraction,F_fraction,S_fraction,Cl_fraction,
Br_fraction,I_fraction,C_fraction,Si_fraction,P_fraction;
fractions (dictionary of element-symbol:atomic-fraction).
X is Pauling electronegativity; missing elemental X uses0 and X_missing_fraction is its atomic fraction.
anion_fraction is only the listed O/N/F/S/Cl/Br/I/Se/Te composition proxy, not an oxidation-state inference. n_sites depends on the supplied cell.
All scalars are finite floats. Element fractions sum to1.
Functions receive these structure descriptors, NOT target labels, IDs or split membership.
'''

PREFIX = '''P1 formation-energy feasibility pilot, development only.
Return JSON {iteration:int,rules_code:string,summary:string}.
rules_code must define EXACTLY five functions rule_1(s),rule_2(s),rule_3(s),rule_4(s),rule_5(s).
Each predicts formation energy in eV/atom for every structure. The executed predictor is their UNWEIGHTED MEAN; do not fit a meta-model.
Favor simple chemically motivated, qualified empirical formulas or short piecewise rules; complementary variants are fine. Constants may be revised from supplied TRAIN feedback.
Allowed Python: assignments, if/elif/else, return; numeric arithmetic, comparisons, boolean operators, dictionary lookup and s.get/fractions.get, min,max,abs,float,sum,len, and math.sqrt/log/log1p/exp/tanh.
No imports, loops/comprehensions, lambdas, recursion, classes, exception handling, file/network/tool access, random values, IDs, target lookups, code outside the five functions, or list/tuple/dict construction. No external learned model calls. At most40 statements/function and4500 source characters. Numeric constants have magnitude<=10000 and execute as floats. Powers only constant exponents with magnitude<=4. String literals only specify dictionary keys. Always return finite values.
Use atomic fractions and volume units exactly as specified. Unary elements should be near zero formation energy. Formation energy<tau(default0) is the binary positive class: favorable relative to elemental references; this is NOT convex-hull stability or synthesizability.
TRAIN references are CHGNet0.3.0 static energies with public MP elemental-reference subtraction. They are approximate, not exact DFT. All HOLDOUT labels/metrics are withheld from revisions. Three versions total: initial plus two feedback revisions. Do not extrapolate claims from in-domain evaluation to unseen chemistry.
'''


def descriptors(structure):
    composition = structure.composition.fractional_composition
    entries = [(element,float(amount)) for element,amount in composition.items()]
    fractions = {str(element):amount for element,amount in entries}
    x = [(float(element.X) if element.X is not None and np.isfinite(element.X) else 0.0,amount) for element,amount in entries]
    mean_x = sum(value*amount for value,amount in x)
    radius=max(6.0,float(min(structure.lattice.abc))+1e-5)
    _,_,_,distances=structure.get_neighbor_list(radius, numerical_tol=1e-8, exclude_self=True)
    nearest=float(np.min(distances))
    result = {'n_sites':float(len(structure)), 'n_elements':float(len(entries)),
              'volume_per_atom_A3':float(structure.volume/structure.composition.num_atoms),
              'density_g_cm3':float(structure.density),'min_distance_A':nearest,
              'mean_Z':sum(element.Z*amount for element,amount in entries),
              'mean_mass':sum(float(element.atomic_mass)*amount for element,amount in entries),
              'mean_X':mean_x,'std_X':math.sqrt(sum(amount*(value-mean_x)**2 for value,amount in x)),
              'range_X':max(value for value,_ in x)-min(value for value,_ in x),
              'X_missing_fraction':sum(amount for element,amount in entries if element.X is None or not np.isfinite(element.X)),
              'metal_fraction':sum(amount for element,amount in entries if element.is_metal),
              'transition_fraction':sum(amount for element,amount in entries if element.is_transition_metal),
              'lanthanide_fraction':sum(amount for element,amount in entries if element.is_lanthanoid),
              'actinide_fraction':sum(amount for element,amount in entries if element.is_actinoid),
              'anion_fraction':sum(fractions.get(e,0.0) for e in ('O','N','F','S','Cl','Br','I','Se','Te')),
              'fractions':fractions}
    for symbol in ('H','O','N','F','S','Cl','Br','I','C','Si','P'):
        result[symbol+'_fraction']=fractions.get(symbol,0.0)
    if not all(math.isfinite(v) for k,v in result.items() if k!='fractions'):
        raise ValueError('Nonfinite structure descriptor')
    return result


_ALLOWED_NODES = (ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Assign,ast.Return,
                 ast.If,ast.Expr,ast.Name,ast.Load,ast.Store,ast.Constant,ast.Subscript,
                 ast.BinOp,ast.UnaryOp,ast.BoolOp,ast.Compare,ast.IfExp,ast.Call,
                 ast.Attribute,ast.Add,ast.Sub,ast.Mult,
                 ast.Div,ast.Pow,ast.Mod,ast.USub,ast.UAdd,ast.Not,ast.And,ast.Or,
                 ast.Eq,ast.NotEq,ast.Lt,ast.LtE,ast.Gt,ast.GtE,ast.In,ast.NotIn,ast.keyword)
_BUILTINS = {'min':min,'max':max,'abs':abs,'float':float,'sum':sum,'len':lambda obj:float(len(obj))}
_MATH = {'sqrt','log','log1p','exp','tanh'}

def _scalar_binary(kind,left,right):
    if not isinstance(left,(int,float,bool)) or not isinstance(right,(int,float,bool)):
        raise ValueError('Rule arithmetic is scalar numeric only')
    a,b=float(left),float(right)
    if not math.isfinite(a) or not math.isfinite(b) or abs(a)>1e12 or abs(b)>1e12:
        raise ValueError('Rule arithmetic magnitude limit')
    if kind=='Add':value=a+b
    elif kind=='Sub':value=a-b
    elif kind=='Mult':value=a*b
    elif kind=='Div':value=a/b
    elif kind=='Pow':value=a**b
    elif kind=='Mod':value=a%b
    else:raise ValueError('Unknown arithmetic operation')
    if isinstance(value,complex) or not math.isfinite(value) or abs(value)>1e12:
        raise ValueError('Rule arithmetic result outside finite bounds')
    return float(value)

class _BoundArithmetic(ast.NodeTransformer):
    def visit_BinOp(self,node):
        self.generic_visit(node)
        return ast.copy_location(ast.Call(func=ast.Name(id='_p1_binary',ctx=ast.Load()),
            args=[ast.Constant(type(node.op).__name__),node.left,node.right],keywords=[]),node)


def compile_rules(code):
    if not isinstance(code,str) or len(code)>4500:
        raise ValueError('Rule source missing or oversized')
    tree=ast.parse(code)
    if len(tree.body)!=5 or any(not isinstance(node,ast.FunctionDef) for node in tree.body):
        raise ValueError('Exactly five function definitions required')
    names=[node.name for node in tree.body]
    if names!=['rule_'+str(i) for i in range(1,6)]:
        raise ValueError('Function names/order must be rule_1 through rule_5')
    for fn in tree.body:
        if (fn.decorator_list or fn.returns is not None or fn.args.defaults or fn.args.kw_defaults or fn.args.vararg
            or fn.args.kwarg or fn.args.kwonlyargs or fn.args.posonlyargs or len(fn.args.args)!=1
            or fn.args.args[0].arg!='s' or fn.args.args[0].annotation is not None):
            raise ValueError('Only undecorated one-argument functions supported')
        if sum(isinstance(node,ast.stmt) for node in ast.walk(fn))>40:
            raise ValueError('Too many rule statements')
    parents={child:parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    for node in ast.walk(tree):
        if not isinstance(node,_ALLOWED_NODES):
            raise ValueError('Forbidden rule syntax: '+type(node).__name__)
        if isinstance(node,ast.Name) and ('__' in node.id or node.id in {'eval','exec','open','globals','locals','getattr','setattr'}):
            raise ValueError('Forbidden rule name')
        if isinstance(node,ast.Constant) and isinstance(node.value,str) and '__' in node.value:
            raise ValueError('Forbidden string field')
        if isinstance(node,ast.Constant):
            if isinstance(node.value,str):
                parent=parents.get(node)
                if not ((isinstance(parent,ast.Subscript) and parent.slice is node) or
                        (isinstance(parent,ast.Call) and isinstance(parent.func,ast.Attribute) and
                         parent.func.attr=='get' and parent.args and parent.args[0] is node)):
                    raise ValueError('String literals may only name dictionary keys')
            elif isinstance(node.value,(int,float)) and not isinstance(node.value,bool):
                if not math.isfinite(float(node.value)) or abs(node.value)>10000:
                    raise ValueError('Numeric constant outside finite allowed range')
                node.value=float(node.value)
            elif node.value is not None and not isinstance(node.value,bool):
                raise ValueError('Unsupported literal')
        if isinstance(node,ast.Assign):
            if len(node.targets)!=1 or not isinstance(node.targets[0],ast.Name) or node.targets[0].id in {'s','math',*_BUILTINS}:
                raise ValueError('Assignments must target a plain local name')
        if isinstance(node,ast.Attribute):
            if node.attr.startswith('_') or node.attr not in (_MATH|{'get'}):
                raise ValueError('Forbidden attribute')
        if isinstance(node,ast.Call):
            target=node.func
            if isinstance(target,ast.Name):
                if target.id not in _BUILTINS: raise ValueError('Only whitelisted pure builtins may be called')
            elif isinstance(target,ast.Attribute):
                if target.attr in _MATH:
                    if not isinstance(target.value,ast.Name) or target.value.id!='math':raise ValueError('Only math functions supported')
                elif target.attr!='get': raise ValueError('Only dict.get and math calls supported')
            else:raise ValueError('Indirect rule calls forbidden')
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Pow):
            exponent=node.right
            value=exponent.value if isinstance(exponent,ast.Constant) else None
            if isinstance(exponent,ast.UnaryOp) and isinstance(exponent.op,ast.USub) and isinstance(exponent.operand,ast.Constant):value=-exponent.operand.value
            if not isinstance(value,(int,float)) or abs(value)>4:raise ValueError('Power exponent must be a constant <=4 in magnitude')
    tree=ast.fix_missing_locations(_BoundArithmetic().visit(tree))
    namespace={'__builtins__':{},'math':math,'_p1_binary':_scalar_binary,**_BUILTINS}
    exec(compile(tree,'<P1-reviewed-rule-functions>','exec'),namespace)
    return [namespace[name] for name in names]


def execute_rules(code, feature_rows):
    functions=compile_rules(code)
    started=time.perf_counter()
    per_rule=np.asarray([[float(fn(dict(s))) for s in feature_rows] for fn in functions],dtype=float).T
    if per_rule.shape!=(len(feature_rows),5) or not np.isfinite(per_rule).all():
        raise ValueError('All five rules must return finite values on every structure')
    if np.max(np.abs(per_rule))>50:
        raise ValueError('Rule prediction magnitude exceeds50eV/atom; check units')
    return {'per_rule':per_rule,'predictions':per_rule.mean(axis=1),'seconds':time.perf_counter()-started}


def _rounded(value):
    if isinstance(value,(float,np.floating)):return float(f'{float(value):.5g}')
    if isinstance(value,dict):return {k:_rounded(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_rounded(v) for v in value]
    return value


def training_context(features, targets, ids):
    y=np.asarray(targets,dtype=float)
    result={'n_TRAIN':len(y),'target_quantiles':np.quantile(y,[0,.1,.25,.5,.75,.9,1]).tolist(),'conditional_TRAIN_statistics':[]}
    keys=['O_fraction','F_fraction','N_fraction','S_fraction','anion_fraction','range_X','std_X','metal_fraction','volume_per_atom_A3']
    for key in keys:
        values=np.asarray([s[key] for s in features])
        thresholds=np.unique(np.quantile(values,[0,.25,.5,.75,1]))
        if len(thresholds)<2:continue
        for lower,upper in zip(thresholds[:-1],thresholds[1:]):
            mask=(values>=lower)&(values<=upper if upper==thresholds[-1] else values<upper)
            if mask.sum():result['conditional_TRAIN_statistics'].append([key,float(lower),float(upper),int(mask.sum()),float(y[mask].mean()),float(np.median(y[mask]))])
    chosen=sorted(range(len(ids)),key=lambda i:hashlib.sha256(('P1-TRAIN-examples|'+str(ids[i])).encode()).hexdigest())[:8]
    compact_keys=['O_fraction','F_fraction','N_fraction','S_fraction','range_X','std_X','metal_fraction','volume_per_atom_A3','n_elements']
    result['labelblind_TRAIN_examples']=[{'s':{k:features[i][k] for k in compact_keys},'fractions':features[i]['fractions'],'reference_e_form':float(y[i])} for i in chosen]
    return _rounded(result)


def feedback(features, reference, prediction, per_rule, ids, groups):
    y=np.asarray(reference,dtype=float); p=np.asarray(prediction,dtype=float)
    order=np.argsort(-np.abs(y-p),kind='stable')
    chosen=[];seen=set()
    for i in order:
        if groups[i] not in seen:
            chosen.append(int(i));seen.add(groups[i])
        if len(chosen)==8:break
    keys=['O_fraction','F_fraction','N_fraction','S_fraction','range_X','std_X','metal_fraction','volume_per_atom_A3','n_elements']
    return _rounded({'TRAIN_ensemble_MAE':float(np.abs(y-p).mean()),
                     'TRAIN_per_rule_MAE':np.abs(per_rule-y[:,None]).mean(axis=0).tolist(),
                     'counterexamples':[{'s':{k:features[i][k] for k in keys},'fractions':features[i]['fractions'],
                                         'reference_e_form':float(y[i]),'prediction':float(p[i]),'residual_pred_minus_reference':float(p[i]-y[i])} for i in chosen]})


def make_prompt(iteration, context, previous_code=None, train_feedback=None):
    packet={'iteration':iteration,'TRAIN_context':context}
    if previous_code is not None:packet.update(previous_five_functions=previous_code,TRAIN_feedback=train_feedback)
    return PREFIX+CONTRACT+json.dumps(packet,ensure_ascii=False,separators=(',',':'))
