"""Render the saved reciprocal-core diagnostic without new trajectory solves."""
from pathlib import Path
from time import perf_counter
import hashlib
import json
import os
import platform
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / os.environ.get('SM_RECIPROCAL_FIGURE_INPUT', 'results/reciprocal_encounter_core')
OUT = ROOT / os.environ.get('SM_RECIPROCAL_FIGURE_OUTPUT', str(DATA))
names = ('reciprocal_equations', 'core_balance', 'reciprocal_validation')
expected = [OUT / (name + '.' + ext) for name in names for ext in ('svg', 'png')]
expected.append(OUT / 'publication_render_receipt.json')
if any(path.exists() for path in expected):
    raise FileExistsError('Preserve rendered evidence; select a fresh SM_RECIPROCAL_FIGURE_OUTPUT')
OUT.mkdir(parents=True, exist_ok=True)
started = perf_counter()
summary = json.loads((DATA / 'summary.json').read_text())
balance = json.loads((DATA / 'saved_balance.json').read_text())
inputs, results = summary['inputs'], summary['results']
fine = [row for row in balance['cases'] if row['case'].endswith('N2048') or row['case'] == 'alternate_partition']
for row in balance['cases']:
    if not np.allclose(np.asarray(row['total']), np.asarray(row['core']) + row['escaping'], rtol=0, atol=2e-16):
        raise ValueError('Saved core and escaping components do not reconstruct the total')
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'legend.fontsize': 9,
                     'figure.facecolor': 'white', 'svg.fonttype': 'path'})
outputs = []
def save(fig, name):
    for ext in ('svg', 'png'):
        path = OUT / (name + '.' + ext)
        fig.savefig(path, dpi=160, bbox_inches='tight', pad_inches=.13,
                    metadata={'Creator': None, 'Date': None} if ext == 'svg' else None)
        outputs.append(path)
    plt.close(fig)

fig = plt.figure(figsize=(10.4, 3.4))
fig.text(.03, .92, 'Reciprocity at the same magnetic field', fontsize=15, weight='bold')
fig.text(.03, .79, r'$s=\mathrm{sign}(w_{z,\mathrm{out}}),\quad S_s=\mathrm{diag}(1,-1,-s)$', fontsize=15)
fig.text(.03, .65, r'$(\mathbf{r}^{*},\mathbf{w}^{*})=(S_s\mathbf{r}_{\mathrm{out}},-S_s\mathbf{w}_{\mathrm{out}})$', fontsize=15)
fig.text(.03, .53, 'Rotate the new transverse guiding center to +x; retain qB.', fontsize=11)
fig.text(.03, .39, r'$|\det D\mathcal{R}|=bup/(b^{*}u^{*}p^{*}),\qquad g=\exp[-U_{\mathrm{in}}/(m\theta)]$', fontsize=15)
fig.text(.03, .24, r'$C=D\cap\mathcal{R}(D),\qquad c\circ\mathcal{R}=-c,\qquad \int_C gc\,\mathrm{d}\mathcal{F}=0$', fontsize=15)
fig.text(.03, .08, 'Uniform field, identical particles and central potential; completed nongrazing first-plane encounters.', fontsize=10)
save(fig, 'reciprocal_equations')

fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), layout='constrained')
x = np.arange(len(fine))
for key, offset, color, label in [('total', -.23, '#374b70', 'full incoming band'),
                                  ('core', 0, '#16857b', 'reciprocal core'),
                                  ('escaping', .23, '#b96c38', 'inverse outside band')]:
    axes[0].bar(x + offset, [row[key][0] * 1e5 for row in fine], width=.22, color=color, label=label)
axes[0].axhline(0, color='.4', linewidth=.8)
axes[0].set(xticks=x, xticklabels=['4201', '4202', '4203', '4204', 'alternate'],
            xlabel='independent sample / phase partition', ylabel=r'signed integral per density $\times 10^5$',
            title='Saved Gibbs-weighted contributions')
axes[0].legend(loc='lower left')
comparison = balance['comparisons']
groups = [('sample', [k for k in comparison if k.startswith('sample_')]),
          ('replicas', [k for k in comparison if k.startswith('replica_')]),
          ('partition', [k for k in comparison if k.startswith('partition_')])]
for index, moment in enumerate(('signed mean', 'second', 'cross', 'spatial second')):
    maxima = [max(comparison[key][index] for key in keys) / inputs['quadrature_target'] for _, keys in groups]
    axes[1].plot(range(3), maxima, 'o-', label=moment)
axes[1].scatter([3], [max(row['core_zero_residual'] for row in fine) / inputs['quadrature_target']],
                marker='D', color='#16857b', label='core zero residual')
axes[1].axhline(1, color='#9e3d3d', linestyle='--', label='declared target')
axes[1].set(xticks=range(4), xticklabels=['nested\nsample', 'replica\npairs', 'alternate\npartition', 'exact-zero\ncomparison'],
            ylabel='largest scaled difference / 5% target', title='Core quadrature keeps its failed checks', ylim=(0, None))
axes[1].legend(fontsize=8, loc='upper right')
save(fig, 'core_balance')

fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), layout='constrained')
finest = [row for row in results['roundtrips'] if row['forward_step'] == .025 and row['reverse_step'] == .025]
measures = [('endpoint', [r.get('endpoint_error', np.nan) / inputs['endpoint_target'] for r in finest]),
            ('exit time', [r.get('time_error', np.nan) / inputs['endpoint_target'] for r in finest]),
            ('moment', [max(r.get('moment_changes', [np.nan])) / inputs['step_target'] for r in finest]),
            ('step', [max(r.get('changes', [np.nan]) + r.get('baseline_changes', [np.nan])) / inputs['step_target']
                      for r in results['step_checks'] + results['reverse_step_checks']])]
for index, (name, data) in enumerate(measures):
    vals = np.asarray(data)
    axes[0].scatter(np.full(len(vals), index), vals, s=10, alpha=.5)
axes[0].axhline(1, color='#9e3d3d', linestyle='--')
axes[0].set_yscale('symlog', linthresh=1e-6)
endpoint_failures = sum(row.get('endpoint_error', float('inf')) > inputs['endpoint_target'] for row in finest)
axes[0].set(xticks=range(4), xticklabels=[name for name, _ in measures],
            ylabel='error / its declared target', ylim=(0, max(3, max(measures[0][1]) * 1.2)),
            title=f'{endpoint_failures} of {len(finest)} endpoint controls fail')
for index, row in enumerate(results['finite_difference']):
    levels = row['levels']
    axes[1].plot([r['epsilon'] for r in levels], [r.get('relative_error', np.nan) for r in levels],
                 'o-', label=f'fixed node {index + 1}')
axes[1].axhline(inputs['fd_target'], color='#9e3d3d', linestyle='--', label='declared target')
axes[1].set(xscale='log', yscale='log', xlabel=r'central-difference scale $\epsilon$',
            ylabel=r'$\left||\det D\mathcal{R}|/(bup/b^{*}u^{*}p^{*})-1\right|$',
            title='Independent finite-difference Jacobian')
axes[1].invert_xaxis()
axes[1].legend()
save(fig, 'reciprocal_validation')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
receipt = dict(scope='Saved-data rendering only; no new solves, no reciprocal population added, no failed gate altered.',
               input_sha256={str(p.relative_to(ROOT)): sha(p) for p in (DATA / 'summary.json', DATA / 'saved_balance.json')},
               output_sha256={p.name: sha(p) for p in outputs}, source_sha256=sha(Path(__file__)),
               source_path=str(Path(__file__).resolve().relative_to(ROOT)),
               render_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
               trajectory_status=results['trajectory_status'], saved_core_status=results['saved_core_status'],
               core_gate_count=len(balance['required_gates']), failed_core_gates=[key for key, value in balance['required_gates'].items() if not value],
               fine_cases=fine, wall_s=perf_counter() - started,
               versions=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__))
(OUT / 'publication_render_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(dict(outputs=[str(p) for p in outputs], wall_s=receipt['wall_s']), indent=2))
