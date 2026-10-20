"""Read-only G3 family membership helpers for stage-2 analyses.

Importing does not read experiment sources, fit models, call LLMs, or write files.
Returned labels are model predictions, never experimentally verified families.
"""
from __future__ import annotations

import csv
from pathlib import Path
from common import ROOT, np

G3_FAMILY_VALUES = ('铜基', '铁基', 'A15', 'MgB₂ 类', '重费米子', 'Chevrel',
                    '硼碳化物', '富勒烯', '镍基', '单质或合金', '其他', '不是超导体')
FIXED_POSITIVE_FAMILIES = ('铜基', '铁基', 'A15', '镍基', '富勒烯')
INTERNAL_FAMILIES = ('铜基', '铁基', 'A15', '硼碳化物')
DEFAULT_G3_SOURCE = ROOT/'G3_complementarity/predictions.csv'


def load_g3_family_labels(ids, source=None):
    """Read frozen G3 labels aligned to arbitrary unique requested material IDs.

    This supports the full cohort or G4/G5 subsets; no labels or outcomes are
    inferred for missing IDs. The source CSV is opened read-only, once per call.
    """
    ids = np.asarray(ids).astype(str)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError('ids must be a one-dimensional unique source-ID array')
    by_id = {}
    with Path(source or DEFAULT_G3_SOURCE).open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['arm'] != 'G3_LLM_family':
                continue
            mid, family = row['id'], row['family_label']
            if mid in by_id or family not in G3_FAMILY_VALUES or row['status'] != 'valid_family':
                raise ValueError(f'invalid or duplicate frozen G3 family: {mid}')
            by_id[mid] = family
    missing = set(ids)-set(by_id)
    if missing:
        raise ValueError(f'missing requested G3 family IDs: {sorted(missing)[:5]}')
    return np.asarray([by_id[mid] for mid in ids])


def g3_family_masks(ids, labels=None, source=None):
    """Return all 12 exclusive masks in requested-ID order, with no writes.

    If aligned labels are supplied this is a pure in-memory operation. Otherwise
    it calls load_g3_family_labels to read the frozen G3 CSV. Empty families are
    retained. Passing a model-predicted family never turns it into ground truth.
    """
    ids = np.asarray(ids).astype(str)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError('ids must be a one-dimensional unique source-ID array')
    labels = load_g3_family_labels(ids, source) if labels is None else np.asarray(labels).astype(str)
    if labels.shape != ids.shape or not np.isin(labels, G3_FAMILY_VALUES).all():
        raise ValueError('labels must be aligned, valid frozen G3 family strings')
    return {label: labels == label for label in G3_FAMILY_VALUES}


def fixed_family_rule(labels):
    """Frozen five-label whitelist, never selected/tuned using outcomes."""
    labels = np.asarray(labels).astype(str)
    if labels.ndim != 1 or not np.isin(labels, G3_FAMILY_VALUES).all():
        raise ValueError('valid one-dimensional frozen G3 labels required')
    return np.isin(labels, FIXED_POSITIVE_FAMILIES).astype(float)
