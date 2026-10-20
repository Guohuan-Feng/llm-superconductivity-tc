"""Finite G2 ranking/calibration/selective statistics, with no I/O or fitting.

Score ties are selected fractionally. Cluster draws change the material counts,
so top-k and requested coverage boundaries are recomputed within every draw.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
from itertools import combinations
from common import np
from f_stats import _interval, _validate_groups

WEIGHTINGS = ('group_weighted', 'unweighted')
KS = (50, 100, 200, 500)
COVERAGES = tuple(round(i / 10, 1) for i in range(1, 11))


def confidence_values(p):
    """Exact displayed-decimal complement, removing binary subtraction artifacts."""
    with localcontext() as context:
        context.prec = 50
        one = Decimal(1)
        values = [Decimal(str(float(value))) for value in p]
        return np.asarray([float(max(value, one-value)) for value in values])


def ratio(a, b, zero=np.nan):
    a, b = np.broadcast_arrays(a, b)
    return np.divide(a, b, out=np.full(a.shape, zero, dtype=float), where=b > 0)


def ordered_buckets(values):
    order = np.argsort(-values, kind='stable')
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(values[order]))]
    return order, starts, values[order[starts]]


def _select(counts, weights, numerators, target, cumulative=None):
    """Fractional selection from descending buckets, independently per draw."""
    if cumulative is None:
        cumulative = tuple(np.cumsum(x, axis=1) for x in (counts, weights, numerators))
    count_cum, weight_cum, numerator_cum = cumulative
    target = np.broadcast_to(target, (len(counts),)).astype(float)
    valid = (target > 0) & (target <= count_cum[:, -1] + 1e-10)
    boundary = np.argmax(count_cum >= target[:, None] - 1e-10, axis=1)
    row = np.arange(len(counts))
    before = np.maximum(boundary - 1, 0)
    n_before = np.where(boundary > 0, count_cum[row, before], 0)
    w_before = np.where(boundary > 0, weight_cum[row, before], 0)
    z_before = np.where(boundary > 0, numerator_cum[row, before], 0)
    fraction = ratio(target - n_before, counts[row, boundary])
    selected_w = w_before + fraction * weights[row, boundary]
    selected_z = z_before + fraction * numerators[row, boundary]
    precision = ratio(selected_z, selected_w)
    for value in (precision, selected_w):
        value[~valid] = np.nan
    return precision, selected_w


def _tie_boundary(values, target):
    if len(values) == 0 or target > len(values) or target <= 0:
        return {'requested_material_count': float(target), 'defined': False}
    order, starts, bucket_values = ordered_buckets(values)
    counts = np.diff(np.r_[starts, len(order)])
    cumulative = np.cumsum(counts)
    j = int(np.searchsorted(cumulative, target - 1e-10))
    before = int(cumulative[j - 1]) if j else 0
    return {'requested_material_count': float(target), 'defined': True,
            'boundary_score': float(bucket_values[j]),
            'strictly_above_boundary_n': before, 'boundary_tie_n': int(counts[j]),
            'selected_from_boundary_n': float(target - before),
            'selected_boundary_fraction': float((target - before) / counts[j])}


def _method_draws(y, p, inverse, base_weights, multiplicities, ks, coverages,
                  batch_size=128):
    """Point estimate in row zero; identical precomputed cluster draws thereafter."""
    n_rows = len(multiplicities)
    fields = ['balanced_accuracy', 'precision', 'recall', 'accuracy', 'ece', 'brier']
    fields += [f'pk.{k}.{m}' for k in ks for m in ('precision', 'weighted_coverage')]
    fields += [f'select.{q:.1f}.{m}' for q in coverages for m in ('accuracy', 'weighted_coverage')]
    fields += [f'bin.{j}.{m}' for j in range(10)
               for m in ('weight', 'weight_fraction', 'mean_p', 'positive_fraction')]
    out = {name: np.full((n_rows, 2), np.nan) for name in fields}
    predicted = p >= .5
    correct = (predicted == y).astype(float)
    rank_order, rank_starts, _ = ordered_buckets(p)
    confidence = confidence_values(p)
    conf_order, conf_starts, _ = ordered_buckets(confidence)
    bins = np.minimum((p * 10).astype(int), 9)
    bin_order = np.argsort(bins, kind='stable')
    bin_starts = np.r_[0, 1 + np.flatnonzero(np.diff(bins[bin_order]))]
    present_bins = bins[bin_order[bin_starts]]
    for start in range(0, n_rows, batch_size):
        stop = min(start + batch_size, n_rows)
        row_counts = multiplicities[start:stop, inverse].astype(float)
        n_materials = row_counts.sum(axis=1)
        rank_counts = np.add.reduceat(row_counts[:, rank_order], rank_starts, axis=1)
        conf_counts = np.add.reduceat(row_counts[:, conf_order], conf_starts, axis=1)
        for scheme in range(2):
            weights = row_counts * base_weights[:, scheme]
            total = weights.sum(axis=1)
            tp = np.sum(weights * (predicted & (y == 1)), axis=1)
            tn = np.sum(weights * (~predicted & (y == 0)), axis=1)
            positive = np.sum(weights * y, axis=1)
            pred_positive = np.sum(weights * predicted, axis=1)
            out['recall'][start:stop, scheme] = ratio(tp, positive)
            out['precision'][start:stop, scheme] = ratio(tp, pred_positive, zero=0)
            out['balanced_accuracy'][start:stop, scheme] = (ratio(tp, positive) + ratio(tn, total-positive)) / 2
            out['accuracy'][start:stop, scheme] = ratio(tp + tn, total)
            out['brier'][start:stop, scheme] = ratio(np.sum(weights * (p-y)**2, axis=1), total)
            rank_w = np.add.reduceat(weights[:, rank_order], rank_starts, axis=1)
            rank_pos = np.add.reduceat((weights*y)[:, rank_order], rank_starts, axis=1)
            rank_cumulative = tuple(np.cumsum(x, axis=1) for x in (rank_counts, rank_w, rank_pos))
            for k in ks:
                if k > len(p):
                    continue
                precision, selected_w = _select(rank_counts, rank_w, rank_pos, k, rank_cumulative)
                out[f'pk.{k}.precision'][start:stop, scheme] = precision
                out[f'pk.{k}.weighted_coverage'][start:stop, scheme] = ratio(selected_w, total)
            conf_w = np.add.reduceat(weights[:, conf_order], conf_starts, axis=1)
            conf_correct = np.add.reduceat((weights*correct)[:, conf_order], conf_starts, axis=1)
            conf_cumulative = tuple(np.cumsum(x, axis=1) for x in (conf_counts, conf_w, conf_correct))
            for q in coverages:
                accuracy, selected_w = _select(conf_counts, conf_w, conf_correct, q * n_materials, conf_cumulative)
                out[f'select.{q:.1f}.accuracy'][start:stop, scheme] = accuracy
                out[f'select.{q:.1f}.weighted_coverage'][start:stop, scheme] = ratio(selected_w, total)
            bin_w = np.zeros((len(weights), 10))
            bin_p, bin_y = bin_w.copy(), bin_w.copy()
            bin_w[:, present_bins] = np.add.reduceat(weights[:, bin_order], bin_starts, axis=1)
            bin_p[:, present_bins] = np.add.reduceat((weights*p)[:, bin_order], bin_starts, axis=1)
            bin_y[:, present_bins] = np.add.reduceat((weights*y)[:, bin_order], bin_starts, axis=1)
            out['ece'][start:stop, scheme] = ratio(np.sum(np.abs(bin_y-bin_p), axis=1), total)
            for j in range(10):
                out[f'bin.{j}.weight'][start:stop, scheme] = bin_w[:, j]
                out[f'bin.{j}.weight_fraction'][start:stop, scheme] = ratio(bin_w[:, j], total)
                out[f'bin.{j}.mean_p'][start:stop, scheme] = ratio(bin_p[:, j], bin_w[:, j])
                out[f'bin.{j}.positive_fraction'][start:stop, scheme] = ratio(bin_y[:, j], bin_w[:, j])
    return out


def _iv(values):
    return _interval(values[0], values[1:])


def _method_summary(p, values, ks, coverages):
    unique, counts = np.unique(p, return_counts=True)
    result = {'n_unique_probabilities': int(len(unique)),
              'n_in_repeated_probability_buckets': int(counts[counts > 1].sum()),
              'largest_probability_tie_n': int(counts.max()) if len(counts) else 0,
              'probability_value_counts': [{'p': float(v), 'n': int(c)} for v, c in zip(unique, counts)],
              'top_k_boundaries': {str(k): _tie_boundary(p, k) for k in ks},
              'selective_boundaries': {f'{q:.1f}': _tie_boundary(confidence_values(p), q*len(p)) for q in coverages}}
    for scheme, weighting in enumerate(WEIGHTINGS):
        result[weighting] = {
            'classification': {m: _iv(values[m][:, scheme]) for m in ('balanced_accuracy', 'precision', 'recall', 'accuracy')},
            'calibration': {'ece': _iv(values['ece'][:, scheme]), 'brier': _iv(values['brier'][:, scheme]),
                'bins': [{'lower': j/10, 'upper': (j+1)/10, 'upper_inclusive': j == 9,
                    **{m: _iv(values[f'bin.{j}.{m}'][:, scheme]) for m in ('weight', 'weight_fraction', 'mean_p', 'positive_fraction')}} for j in range(10)]},
            'precision_at_k': {str(k): {m: _iv(values[f'pk.{k}.{m}'][:, scheme])
                for m in ('precision', 'weighted_coverage')} for k in ks},
            'selective': {f'{q:.1f}': {'target_material_coverage': q,
                **{m: _iv(values[f'select.{q:.1f}.{m}'][:, scheme])
                   for m in ('accuracy', 'weighted_coverage')}} for q in coverages}}
    return result


def summarize(y, scores, groups, full_group_sizes, mask=None, *, n_boot=5000,
              seed=20261018, ks=KS, coverages=COVERAGES, pairs=None):
    y = np.asarray(y)
    if y.ndim != 1 or not np.all(np.isin(y, [0, 1])):
        raise ValueError('y must be aligned binary labels')
    groups, sizes = _validate_groups(groups, full_group_sizes, len(y))
    mask = np.ones(len(y), dtype=bool) if mask is None else np.asarray(mask)
    if mask.shape != y.shape or mask.dtype != np.bool_:
        raise ValueError('mask must be aligned Boolean')
    if n_boot < 5000:
        raise ValueError('G2 requires 5000 cluster draws')
    names = list(scores)
    if not names:
        raise ValueError('methods are required')
    for name in names:
        p = np.asarray(scores[name], dtype=float)
        if p.shape != y.shape or not np.isfinite(p[mask]).all() or np.any((p[mask] < 0) | (p[mask] > 1)):
            raise ValueError(f'invalid probabilities {name}')
    pairs = list(combinations(names, 2)) if pairs is None else list(pairs)
    observed, inverse = np.unique(groups[mask], return_inverse=True)
    ng, n = len(observed), int(mask.sum())
    result = {'n': n, 'n_groups': ng, 'n_positive': int(y[mask].sum()),
              'n_negative': int(n-y[mask].sum()), 'n_boot': n_boot, 'seed': seed,
              'single_group_warning': 'One group; all cluster-bootstrap intervals are degenerate and cannot estimate between-group uncertainty.' if ng == 1 else None,
              'empty_warning': 'Empty subset; all metrics undefined.' if not n else None,
              'methods': {}, 'paired_differences': {}}
    if not n:
        return result
    base_weights = np.stack(([1 / sizes[g] for g in groups[mask]], np.ones(n)), axis=1)
    draws = np.empty((n_boot + 1, ng), dtype=np.int16)
    draws[0] = 1
    rng = np.random.default_rng(seed)
    for start in range(0, n_boot, 256):
        stop = min(start + 256, n_boot)
        draws[1+start:1+stop] = rng.multinomial(ng, np.full(ng, 1/ng), size=stop-start)
    raw = {}
    for name in names:
        p = np.asarray(scores[name], dtype=float)[mask]
        raw[name] = _method_draws(y[mask], p, inverse, base_weights, draws, ks, coverages)
        result['methods'][name] = _method_summary(p, raw[name], ks, coverages)
    paired_fields = ['balanced_accuracy', 'precision', 'recall', 'accuracy', 'ece', 'brier']
    paired_fields += [f'pk.{k}.precision' for k in ks]
    paired_fields += [f'select.{q:.1f}.accuracy' for q in coverages]
    for a, b in pairs:
        result['paired_differences'][f'{a} - {b}'] = {'arm_a': a, 'arm_b': b,
            **{weighting: {field: _iv(raw[a][field][:, i] - raw[b][field][:, i])
                for field in paired_fields} for i, weighting in enumerate(WEIGHTINGS)}}
    return result


def self_check():
    """Expanded-cluster oracle: recomputation, ties, weights and unavailable k."""
    y = np.array([1, 0, 1, 0, 1, 0])
    p = np.array([.9, .9, .9, .4, .2, .1])
    group = np.array([0, 0, 1, 2, 2, 2])
    base = np.stack((1 / np.array([2, 2, 1, 3, 3, 3]), np.ones(6)), axis=1)
    mult = np.array([[1, 1, 1], [0, 2, 1], [2, 1, 0], [0, 0, 3]])
    raw = _method_draws(y, p, group, base, mult, (2, 5, 50), (.5, 1.))
    checks = 0
    for j, multiplicity in enumerate(mult):
        expanded = np.repeat(np.arange(6), multiplicity[group])
        ey, ep, ew = y[expanded], p[expanded], base[expanded]
        for scheme in range(2):
            w = ew[:, scheme]
            for k in (2, 5, 50):
                if k > len(ep):
                    assert np.isnan(raw[f'pk.{k}.precision'][j, scheme])
                else:
                    threshold = np.sort(ep)[-k]
                    above, tied = ep > threshold, ep == threshold
                    selected = above.astype(float) + tied * ((k-above.sum())/tied.sum())
                    oracle = np.sum(w*ey*selected) / np.sum(w*selected)
                    assert np.isclose(raw[f'pk.{k}.precision'][j, scheme], oracle)
                checks += 1
            for q in (.5, 1.):
                confidence = np.maximum(ep, 1-ep)
                target = q*len(ep)
                threshold = np.sort(confidence)[-int(np.ceil(target))]
                above, tied = confidence > threshold, confidence == threshold
                selected = above.astype(float) + tied * ((target-above.sum())/tied.sum())
                correct = ((ep >= .5) == ey)
                oracle = np.sum(w*correct*selected) / np.sum(w*selected)
                assert np.isclose(raw[f'select.{q:.1f}.accuracy'][j, scheme], oracle)
                assert np.isclose(raw[f'select.{q:.1f}.weighted_coverage'][j, scheme], np.sum(w*selected)/w.sum())
                checks += 2
            oracle_ece = 0
            for b in range(10):
                selected = np.minimum((ep*10).astype(int), 9) == b
                oracle_ece += abs(np.sum(w[selected]*(ey[selected]-ep[selected])))
            assert np.isclose(raw['ece'][j, scheme], oracle_ece/w.sum())
            assert np.isclose(raw['brier'][j, scheme], np.sum(w*(ep-ey)**2)/w.sum())
            checks += 2
    confidence = confidence_values(np.array([.82, .18, .93, .07, np.nextafter(.82, 1), np.nextafter(.82, 0)]))
    assert confidence[0] == confidence[1]
    assert confidence[2] == confidence[3]
    assert confidence[4] > confidence[0]
    assert confidence[0] > confidence[5]
    return {'expanded_cluster_oracle_checks': checks, 'confidence_decimal_tie_checks': 4,
            'passed': True}
