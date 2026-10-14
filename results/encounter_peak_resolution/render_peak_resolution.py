"""Render a fixed-tick display variant from saved example-41 data only."""
from pathlib import Path
from time import perf_counter
import hashlib
import json
import platform
import numpy as np
started = perf_counter()
OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
source = ROOT/'examples/41_encounter_peak_resolution.py'
summary = json.loads((OUTPUT/'summary.json').read_text())
inputs, results = summary['inputs'], summary['results']
PHASE_TARGET, PLANE_TARGET, STEP_TARGET = inputs['phase_target'], inputs['plane_target'], inputs['step_target']
angles = 2*np.pi*np.arange(inputs['union_count'])/inputs['union_count']
status, checks = results['status'], results['comparisons']
cases = {c['case']:c for c in results['cases']}
fine = cases['phase4096_L24_union']
old = cases['phase256_L24_retained']
phase_cases = [cases[f'phase{n}_L24'] for n in inputs['phase_levels']]
with np.load(OUTPUT/'states.npz') as archive:
    attempts = [dict(moments=m.tolist(), exit=str(e)) for m,e in zip(archive['moments'],archive['exit'])]
with np.load(OUTPUT/'populations.npz') as archive:
    populations = {name:archive[name].copy() for name in archive.files}
if fine['complete']:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [attempts[i] for i in populations['phase4096_L24_union']]
    m2 = np.asarray([r['moments'][1] for r in rows])
    exit_mask = np.asarray([r['exit']=='reflected' for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    axes[0, 0].plot(angles, m2, lw=.8)
    axes[0, 0].scatter(angles[exit_mask], m2[exit_mask], s=14, color='orange', label='reflected; retained weight')
    axes[0, 0].set(xlabel='incoming zero-plane phase', ylabel='individual raw second moment', title='One fixed incident speed/impact; narrow scattering peak')
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].plot(angles, m2, lw=.8)
    axes[0, 1].scatter(angles[exit_mask], m2[exit_mask], s=14, color='orange')
    axes[0, 1].set(xlim=(1.45, 1.7), xlabel='incoming zero-plane phase', ylabel='individual raw second moment', title='Resolved transmission/reflection neighborhood')
    for j, name in enumerate(('mean', 'second', 'cross', 'spatial')):
        labels = ['phase1024_to2048', 'phase2048_to4096', 'halfshift', 'final_plane_phase', 'final_plane_halfshift', 'boundary_integral', 'boundary_weak_state', 'active_step']
        targets = [PHASE_TARGET]*5+[PLANE_TARGET]*2+[STEP_TARGET]
        axes[1, 0].plot(range(len(labels)), [checks[k][j]/target if checks[k] is not None else np.nan for k,target in zip(labels,targets)], 'o-', label=name)
    axes[1, 0].axhline(1, color='gray', ls=':', label='declared target')
    axes[1, 0].set_xticks(range(len(labels)), ['N1024', 'N2048', 'shift24', 'N48', 'shift48', 'planes', 'weak', 'step'], rotation=45)
    axes[1, 0].set(ylabel='change / own declared target', title=f'Fixed checks: {status}')
    axes[1, 0].legend(fontsize=7)
    for j, name in enumerate(('mean', 'second', 'cross')):
        axes[1, 1].plot([256, 512, 1024, 2048, 4096], [c['observables'][j] if c['complete'] else np.nan for c in [old,*phase_cases,fine]], 'o-', label=name)
    axes[1, 1].set(xscale='log', xlabel='uniform phases at L24', ylabel='angular moment average', title='Signed moments and second moment refined together')
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].set_xticks([256, 512, 1024, 2048, 4096], labels=['256', '512', '1024', '2048', '4096'])
    axes[1, 1].minorticks_off()
    fig.savefig(OUTPUT/'peak_resolution_readable.png', dpi=160)
    plt.close(fig)
import matplotlib
receipt = dict(scope='Display-only fixed-tick variant; no scattering solve or scientific data change.',
    operations=['Set x ticks of the angular-average refinement panel to 256, 512, 1024, 2048, 4096.', 'Disable minor ticks on that panel; all plotted arrays and other styling are retained.'],
    input_sha256={name:hashlib.sha256((OUTPUT/name).read_bytes()).hexdigest() for name in ('summary.json','states.npz','populations.npz','peak_resolution.png')},
    producer_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    render_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    output=dict(path='peak_resolution_readable.png',sha256=hashlib.sha256((OUTPUT/'peak_resolution_readable.png').read_bytes()).hexdigest()),
    matplotlib_version=matplotlib.__version__, numpy_version=np.__version__, python=platform.python_version(),
    wall_s=perf_counter()-started)
(OUTPUT/'render_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
