"""Visible G2 scientific curves from frozen bootstrap summaries; no refitting."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from common import ROOT, np, safe_path, write_json


def make_figures(metrics, directory=None):
    directory = safe_path(directory or ROOT/'work/g2_report')
    directory.mkdir(parents=True, exist_ok=True)
    cache = safe_path(directory/'matplotlib-cache')
    cache.mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(cache)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = metrics['method_order']
    labels = {'A3_forced': 'A3 (highest p=.98: 104 tied)', 'A2_forced': 'A2',
              'HGB127': 'HGB127', 'StrongML': 'Strong ML', 'G1best': 'G1 best'}
    colors = dict(zip(names, ('#1766B2', '#1C9B99', '#D78017', '#B34747', '#4D8C3A')))
    figure_records = {}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.labelsize': 11, 'axes.titlesize': 11,
                         'legend.fontsize': 9, 'axes.spines.top': False,
                         'axes.spines.right': False})
    for subset, title in (('all', 'All materials'), ('exclude_93', 'Excluding component 93')):
        result = metrics['subsets'][subset]
        fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.6), dpi=180)
        legend_handles = []
        for column, weighting in enumerate(('group_weighted', 'unweighted')):
            calibration, selective = axes[0, column], axes[1, column]
            calibration.plot([0, 1], [0, 1], '--', color='#7D7D7D', lw=1, zorder=0)
            for name in names:
                value = result['methods'][name][weighting]
                bins = value['calibration']['bins']
                bins = [b for b in bins if b['mean_p']['point'] is not None and b['positive_fraction']['point'] is not None]
                x = np.asarray([b['mean_p']['point'] for b in bins])
                y = np.asarray([b['positive_fraction']['point'] for b in bins])
                lo = np.asarray([b['positive_fraction']['ci95'][0] for b in bins])
                hi = np.asarray([b['positive_fraction']['ci95'][1] for b in bins])
                color = colors[name]
                line, = calibration.plot(x, y, 'o-', ms=3.8, lw=1.5,
                                         color=color, label=labels.get(name, name))
                calibration.fill_between(x, lo, hi, color=color, alpha=.10, linewidth=0)
                xlo = np.asarray([b['mean_p']['ci95'][0] for b in bins])
                xhi = np.asarray([b['mean_p']['ci95'][1] for b in bins])
                calibration.errorbar((xlo+xhi)/2, y, xerr=(xhi-xlo)/2,
                                     fmt='none', ecolor=color, alpha=.30, capsize=2, lw=.8)
                if column == 0:
                    legend_handles.append(line)
                curve = list(value['selective'].values())
                cx = np.asarray([r['weighted_coverage']['point'] for r in curve])
                cy = np.asarray([r['accuracy']['point'] for r in curve])
                clo = np.asarray([r['accuracy']['ci95'][0] for r in curve])
                chi = np.asarray([r['accuracy']['ci95'][1] for r in curve])
                selective.plot(cx, cy, 'o-', ms=3.8, lw=1.5, color=color)
                selective.fill_between(cx, clo, chi, color=color, alpha=.10, linewidth=0)
                if weighting == 'group_weighted':
                    covlo = np.asarray([r['weighted_coverage']['ci95'][0] for r in curve])
                    covhi = np.asarray([r['weighted_coverage']['ci95'][1] for r in curve])
                    selective.errorbar((covlo+covhi)/2, cy, xerr=(covhi-covlo)/2,
                                       fmt='none', ecolor=color, alpha=.30, capsize=2, lw=.8)
            scheme = 'Inverse-original-group weighted' if weighting == 'group_weighted' else 'Unweighted materials'
            calibration.set(title=f'Calibration | {scheme}', xlabel='Mean predicted p in bin',
                            ylabel='Observed positive fraction', xlim=(-.02, 1.02), ylim=(-.02, 1.02))
            selective.set(title=f'Selective accuracy | {scheme}',
                          xlabel='Actual retained group weight / total group weight' if weighting == 'group_weighted' else 'Retained material fraction',
                          ylabel='Accuracy of retained materials', xlim=(0, 1.02), ylim=(0, 1.02))
            for axis in (calibration, selective):
                axis.grid(alpha=.18, linewidth=.7)
                axis.set_xticks(np.linspace(0, 1, 6))
                axis.set_yticks(np.linspace(0, 1, 6))
        fig.suptitle(f'G2 development analysis: {title}\nn={result["n"]:,} | groups={result["n_groups"]:,} | positives={result["n_positive"]:,}',
                     y=.985, fontsize=14, fontweight='bold')
        fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(.5, .929), ncol=5, frameon=False)
        fig.text(.05, .060, 'Bands: pointwise 95% group-bootstrap CI (5,000 shared draws). Horizontal bars: bootstrap CI of x coordinate.', fontsize=9)
        fig.text(.05, .040, 'Calibration: fixed 10 equal-width bins, unchanged p. Selection: fractional confidence ties; material targets 10%-100%.', fontsize=9)
        fig.text(.05, .020, 'Weighted panels use 1/original component size. A3 ties do not define unique top-50/top-100 lists. G1 best: adaptive OOF selection.', fontsize=9)
        fig.subplots_adjust(left=.075, right=.98, top=.86, bottom=.13, wspace=.27, hspace=.32)
        path = safe_path(directory/f'G2_{subset}_curves.png')
        fig.savefig(path, dpi=180, facecolor='white')
        plt.close(fig)
        figure_records[subset] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'panels': ['group-weighted calibration', 'unweighted calibration', 'actual-group-weight coverage vs weighted accuracy', 'material coverage vs unweighted accuracy'],
            'source': 'metrics.json/subsets/'+subset, 'n': result['n'], 'n_groups': result['n_groups'],
            'confidence_bands': 'pointwise percentile 95% group-bootstrap intervals already computed from5000shared draws; no additional resampling, model calls or refitting',
            'x_uncertainty': 'horizontal intervals use stored bootstrap CIs of bin mean p and actual retained group weight fraction',
            'development_model_selection': 'G1best adaptively selected on same development OOF weighted BA; curves are descriptive development results'}
    return figure_records


if __name__ == '__main__':
    output = ROOT/'G2_ranking_calibration'
    metrics = json.loads((output/'metrics.json').read_text(encoding='utf-8'))
    assert metrics['status'] == 'complete' and metrics['extra_G1best'] is not None, 'Plot once after complete G1best is included'
    figures = make_figures(metrics)
    metrics['figures'] = figures
    write_json(output/'metrics.json', metrics)
    config = json.loads((output/'config.json').read_text(encoding='utf-8'))
    config['report_figures'] = figures
    script = ROOT/'g2_figures.py'
    config['source_sha256'][str(script)] = hashlib.sha256(script.read_bytes()).hexdigest()
    write_json(output/'config.json', config)
    with safe_path(output/'run.log').open('a', encoding='utf-8') as log:
        log.write(datetime.now(timezone.utc).isoformat()+' final G1best curves: '+json.dumps(figures,ensure_ascii=False)+'\n')
        log.flush()
        os.fsync(log.fileno())
    print(json.dumps(figures,ensure_ascii=False))
