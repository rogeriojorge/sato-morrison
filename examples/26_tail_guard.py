"""Plot one saved line-search failure; no new trajectory or correction is solved."""
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sato_morrison.reference import run_metadata

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'results/nonuniform_spatial_blocks_full/dipole_dt16_replay'
SUMMARY = OUTPUT/'summary.json'
LEDGER = OUTPUT/'trial_ledger.jsonl'
RAY = OUTPUT/'final_ray.npz'
ZOOM_FIRST = 25
LINTHRESH = 1e-6
COLORS = {'log_range_guard': '#767676', 'population_guard': '#ba3c3c',
          'armijo_guard': '#d79528', 'accepted': '#398252'}
print('Saved-ray tail guard plot: checking immutable evidence; no nonlinear solve', flush=True)
start = perf_counter()
summary = json.loads(SUMMARY.read_text())
for path in (LEDGER, RAY):
    if hashlib.sha256(path.read_bytes()).hexdigest() != summary['files_sha256'][path.name]:
        raise RuntimeError(f'Saved evidence hash mismatch: {path.name}')
trials = [json.loads(line) for line in LEDGER.read_text().splitlines()]
iteration = summary['last_correction']['iteration']
last_trials = [row for row in trials if row['iteration'] == iteration]
if [row['backtrack'] for row in last_trials] != list(range(30)):
    raise RuntimeError('Expected exactly the original thirty final trials')
with np.load(RAY) as saved:
    g = saved['g'].copy()
    direction = saved['direction'].copy()
    residual = saved['R'].copy()
    descent = float(saved['descent'])
if (g.shape != direction.shape or residual.shape != g.shape
    or not np.all(np.isfinite([g, direction, residual]))):
    raise RuntimeError('Invalid saved final ray')
np.testing.assert_allclose(residual@direction, descent, rtol=1e-12, atol=0.)
normal = np.log(np.finfo(float).tiny)
subnormal = np.log(np.nextafter(0., 1.))
k = np.array([row['backtrack'] for row in last_trials])
y = np.array([row['min_log']-normal for row in last_trials])
for row in last_trials:
    if row['reason'] not in COLORS or row['alpha'] != 2.**(-row['backtrack']):
        raise RuntimeError('Unknown reason or changed original trial fraction')
    np.testing.assert_allclose(np.min(g+row['alpha']*direction), row['min_log'], rtol=0., atol=2e-12)
if not np.all(np.isfinite(y)) or any(row['reason'] == 'accepted' for row in last_trials):
    raise RuntimeError('This figure requires the failed final ray')
counts = {reason: sum(row['reason'] == reason for row in last_trials)
          for reason in COLORS}
if counts != {'log_range_guard': 4, 'population_guard': 26, 'armijo_guard': 0, 'accepted': 0}:
    raise RuntimeError('Saved failure reasons changed')
current = float(g.min()-normal)
if current <= 0 or y[-1] >= 0:
    raise RuntimeError('Saved normal-range boundary diagnosis changed')
inputs = {'summary': str(SUMMARY.relative_to(ROOT)), 'ledger': str(LEDGER.relative_to(ROOT)),
    'ray': str(RAY.relative_to(ROOT)), 'input_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (SUMMARY, LEDGER, RAY)}, 'final_newton_iteration': iteration,
    'backtracks': list(range(30)), 'zoom_backtracks': list(range(ZOOM_FIRST, 30)),
    'full_panel_scale': 'symlog', 'symlog_linear_threshold': LINTHRESH,
    'case_inputs': summary['inputs']}
metadata = run_metadata(inputs, model='Plot of saved failed ray only; no new correction, trial acceptance or evolution',
    boundary=summary['boundary'], units=summary['units'])
metadata.update({'replay_original_producer': summary['original_failure_producer'],
    'replay_source_contract': summary['replay_source_contract'],
    'plot_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'scope': 'This M4 logging-only replay fails at Newton19; the separate original Linux failure was at Newton18. No untried fraction is plotted. Bounds describe binary64 normal/subnormal ranges, not a theorem about all backends or root existence.',
    'saved_rejection_counts': counts, 'normal_log_bound': normal, 'subnormal_log_bound': subnormal,
    'current_minimum_above_normal': current, 'last_trial_minimum_above_normal': float(y[-1])})
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
for ax, mask in zip(axes, (np.ones(k.shape, dtype=bool), k >= ZOOM_FIRST)):
    for reason, color in COLORS.items():
        chosen = np.array([row['reason'] == reason for row in last_trials]) & mask
        if chosen.any():
            ax.scatter(k[chosen], y[chosen], color=color, s=35, zorder=3)
    ax.axhline(0, color='#222222', linewidth=1.1)
    ax.axhline(current, color='#3567a0', linestyle=':', linewidth=1.4)
    ax.set_xlabel(r'Backtrack index $k$ ($\alpha=2^{-k}$)')
    ax.grid(True, alpha=.18)
axes[0].set_yscale('symlog', linthresh=LINTHRESH)
axes[0].axhline(subnormal-normal, color='#888888', linestyle='--', linewidth=1)
axes[0].set_ylim(min(y.min()*1.2, -50), 3e-6)
axes[0].set_ylabel(r'$\min_i g_i^{\rm trial}-\log(\mathrm{tiny})$')
axes[0].set_title('All 30 original trials')
axes[1].set_ylim(y[k >= ZOOM_FIRST].min()*1.15, current*2.3)
axes[1].set_xticks(np.arange(ZOOM_FIRST, 30))
axes[1].set_title('Last five trials: normal-range boundary')
handles = [Line2D([], [], marker='o', linestyle='', color=COLORS['log_range_guard'], label='Log-range rejection (4)'),
    Line2D([], [], marker='o', linestyle='', color=COLORS['population_guard'], label='Population rejection (26)'),
    Line2D([], [], color='#222222', label='Smallest normal density'),
    Line2D([], [], color='#888888', linestyle='--', label='Smallest positive subnormal density'),
    Line2D([], [], color='#3567a0', linestyle=':', label='Current minimum (unchanged iterate)')]
fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=8.5)
fig.suptitle(r'Saved dipole ray: $f=\exp(g)$; finite positive weighted population required'+'\n'
    +'Final trial has positive NumPy density but JAX exp returns zero; no accepted second step', fontsize=11.5)
fig.tight_layout(rect=(0, .13, 1, .98))
fig.savefig(OUTPUT/'tail_guard.png', dpi=180)
fig.savefig(OUTPUT/'tail_guard.svg')
plt.close(fig)
metadata['checks_and_plot_wall_s'] = perf_counter()-start
metadata['output_sha256'] = {name: hashlib.sha256((OUTPUT/name).read_bytes()).hexdigest()
    for name in ('tail_guard.png', 'tail_guard.svg')}
(OUTPUT/'tail_guard_metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False))
print(f'Saved-ray figure complete in {metadata["checks_and_plot_wall_s"]:.3f}s', flush=True)
