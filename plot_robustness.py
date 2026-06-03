"""Plot robustness curves from robustness.json (matplotlib non-interactive)."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')   # MUST be before pyplot
import matplotlib.pyplot as plt

ROOT = '/home/server/Documents/yping/LR-SAF-LSD'
LOG = os.path.join(ROOT, 'logs', 'robustness.json')

with open(LOG) as f:
    R = json.load(f)

# Make 2x3 grid: rows = metrics (F, sAP-10), cols = degradations
fig, axes = plt.subplots(2, 3, figsize=(14, 7))

KIND_TITLE = {'noise': 'Gaussian Noise',
              'blur':  'Motion Blur',
              'lowlight': 'Low Light'}
KIND_XLABEL = {'noise': 'noise sigma (0-255)',
               'blur':  'blur kernel size (px)',
               'lowlight': 'brightness multiplier'}

for ci, kind in enumerate(['noise', 'blur', 'lowlight']):
    rows = R[kind]
    xs = [r['level'] for r in rows]
    for ri, (metric, ylabel) in enumerate([('F', 'F-measure'),
                                            ('sAP10', 'sAP-10')]):
        ax = axes[ri, ci]
        ax.plot(xs, [r[f'{metric}_afm']     for r in rows],
                 'o--', label='AFM baseline', color='#888888', linewidth=2)
        ax.plot(xs, [r[f'{metric}_lrsaf']   for r in rows],
                 's-',  label='LR-SAF (geom only)', color='#1f77b4', linewidth=2)
        ax.plot(xs, [r[f'{metric}_lrsaf_s'] for r in rows],
                 '^-',  label='LR-SAF + semantic', color='#2ca02c', linewidth=2.5)
        ax.set_title(f'{KIND_TITLE[kind]} - {ylabel}', fontsize=11)
        ax.set_xlabel(KIND_XLABEL[kind])
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if ri == 0 and ci == 2:
            ax.legend(loc='upper right', fontsize=9)

plt.tight_layout()
out_png = os.path.join(ROOT, 'logs', 'robustness_curves.png')
out_pdf = os.path.join(ROOT, 'logs', 'robustness_curves.pdf')
plt.savefig(out_png, dpi=140)
plt.savefig(out_pdf)
print(f"saved {out_png}")
print(f"saved {out_pdf}")

# Also print a markdown table summary
print('\n=== MARKDOWN SUMMARY ===\n')
for kind in ['noise', 'blur', 'lowlight']:
    print(f'### {KIND_TITLE[kind]}\n')
    print('| level | F (AFM/LR/LR+sem) | sAP10 (AFM/LR/LR+sem) | Δ sAP10 (sem vs AFM) |')
    print('|---|---|---|---|')
    for r in R[kind]:
        d = r['sAP10_lrsaf_s'] - r['sAP10_afm']
        print(f"| {r['level']} | "
              f"{r['F_afm']:.3f}/{r['F_lrsaf']:.3f}/{r['F_lrsaf_s']:.3f} | "
              f"{r['sAP10_afm']:.3f}/{r['sAP10_lrsaf']:.3f}/{r['sAP10_lrsaf_s']:.3f} | "
              f"{d:+.3f} |")
    print()
