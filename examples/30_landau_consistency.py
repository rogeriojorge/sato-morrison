"""Compare the full initial Landau collision term with a continuum Gaussian oracle."""
from pathlib import Path
from hashlib import sha256
from time import perf_counter
import json
import resource
import sys
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sato_morrison.controls import landau_velocity_grid, landau_entropy_compiler
from sato_morrison.reference import run_metadata, progress

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results' / 'landau_consistency'
COVARIANCE = np.array([1.15, 1.15, .7])
ORDERS = [12, 16, 20, 24, 28]
EXTENT = 5.
DERIVATIVES = ['polynomial', 'local_quadratic']
SCALAR_ORDERS = [96, 192]
CHUNK = 128
TARGET = .01
# Tail and kernel scans keep the point count fixed; their changes are not pure errors.
EXTRA_CASES = [(20, 4., 0.), (20, 6., 0.), (20, 5., .2), (20, 5., .1)]
OUTPUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
    'axes.spines.right': False, 'svg.fonttype': 'path', 'figure.facecolor': 'white'})
print('Initial physical 3V Landau collision term: independent continuum formula versus finite weak derivatives.', flush=True)
print('Conservation and pressure rates do not bound full-distribution error. Compilation precedes each new grid.', flush=True)
start = perf_counter()
metadata = run_metadata({'density': 1., 'mass': 1., 'field_for_mu': 1., 'gamma': 1.,
    'covariance': COVARIANCE.tolist(), 'orders': ORDERS, 'extent': EXTENT,
    'derivatives': DERIVATIVES, 'scalar_orders': SCALAR_ORDERS,
    'extra_cases': EXTRA_CASES, 'chunk_size': CHUNK, 'relative_L1_target': TARGET},
    model='Instantaneous full physical 3V Coulomb Landau consistency',
    boundary='Continuum Gaussian on R3 versus natural weak no-flux velocity cube')
metadata.update(status='running', rows=[])
reference_path = ROOT / 'results' / 'operator_comparison' / 'physical_initial_rates.json'
metadata['independent_reference'] = {'path': str(reference_path.relative_to(ROOT)),
    'sha256': sha256(reference_path.read_bytes()).hexdigest()}
pressure_reference = np.array([-.07403925375622226, -.07403925375622226, .14807850751244445])
cumulant_reference = np.array([.1197406766201535, .1197406766201535, -.1166031877840004])
arrays = {}


def gaussian(v):
    return np.exp(-np.sum(v*v/COVARIANCE, axis=1)/2)/np.sqrt(np.prod(2*np.pi*COVARIANCE))


def continuum_relative_rate(v, order):
    """C[F]/F from the Laplace integral of the Gaussian Coulomb diffusion tensor.

    With score s=-A^-1 v, C/F = D:(ss^T-A^-1)+8*pi*F. Substituting
    z=q/(1-q) maps the positive Laplace variable to a finite Gauss interval.
    This is an instantaneous reference, not an evolving Gaussian closure.
    """
    q, weights = np.polynomial.legendre.leggauss(order)
    q = (q+1)/2
    z = q/(1-q)
    weights = weights/(2*(1-q)**2)
    d = 1+2*z[:, None]**2*COVARIANCE
    variance = COVARIANCE/d
    answer = np.empty(len(v))
    for first in range(0, len(v), 256):
        x = v[first:first+256]
        score = -x/COVARIANCE
        mean = x[:, None, :]/d
        factor = np.prod(d, axis=1)**-.5*np.exp(-z*z*np.sum(x[:, None, :]**2/d, axis=2))
        trace = np.sum(score*score-1/COVARIANCE, axis=1)[:, None]
        contraction = (np.sum(variance*(score[:, None, :]**2-1/COVARIANCE), axis=2)
            +np.sum(mean*score[:, None, :], axis=2)**2-np.sum(mean*mean/COVARIANCE, axis=2))
        answer[first:first+len(x)] = (2/np.sqrt(np.pi)*np.sum(
            weights*factor*(trace-2*z*z*contraction), axis=1)+8*np.pi*gaussian(x))
    return answer


def save():
    metadata['wall_s'] = perf_counter()-start
    metadata['process_peak_rss_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)
    (OUTPUT/'summary.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')


for order, extent, softening in [(n, EXTENT, 0.) for n in ORDERS]+EXTRA_CASES:
    base = landau_velocity_grid(order, extent)
    v, w = map(np.asarray, (base.velocity, base.weights))
    f = gaussian(v)
    population = w*f
    with progress(f'Continuum scalar quadrature at {order}³ nodes, L={extent}, epsilon={softening}'):
        refs = [continuum_relative_rate(v, q) for q in SCALAR_ORDERS]
    scalar_error = float(np.max(abs(refs[1]-refs[0])))
    if scalar_error > 1e-10:
        raise RuntimeError(f'Continuum scalar quadrature unresolved: {scalar_error}')
    exact = f*refs[-1]
    norm = w@abs(exact)
    core = np.linalg.norm(v, axis=1) <= 3.
    for method in DERIVATIVES:
        print(f'Compile {method}, n={order}, L={extent}, epsilon={softening}.', flush=True)
        grid = landau_velocity_grid(order, extent, derivative=method)
        compiler = landau_entropy_compiler(grid, softening=softening, chunk_size=CHUNK)
        logf = jnp.log(jnp.asarray(f))
        tick = perf_counter()
        residual = np.asarray(compiler.apply(logf, logf))
        first_s = perf_counter()-tick
        warm = []
        for repeat in range(3):
            tick = perf_counter()
            compiler.apply(logf, logf).block_until_ready()
            warm.append(perf_counter()-tick)
        computed = -residual/w
        error = computed-exact
        rate = -residual@(v*v)
        fourth = -residual@(v**4)-6*COVARIANCE*rate
        relative_error = computed/f-refs[-1]
        row = {'order': order, 'extent': extent, 'softening': softening, 'derivative': method,
            'nodes': grid.size, 'scalar_refinement_max_abs': scalar_error,
            'initial_number_error': abs(float(population.sum())-1.),
            'relative_L1_error': float(w@abs(error)/norm),
            'core_L1_error_over_full_reference': float(w[core]@abs(error[core])/norm),
            'tail_L1_error_over_full_reference': float(w[~core]@abs(error[~core])/norm),
            'population_weighted_relative_rate_RMS_error': float(np.sqrt(population@(relative_error**2))),
            'core_relative_rate_max_abs_error': float(np.max(abs(relative_error[core]))),
            'max_abs_relative_rate': float(np.max(abs(computed/f))),
            'max_abs_continuum_relative_rate': float(np.max(abs(refs[-1]))),
            'pressure_rate': rate.tolist(), 'fourth_cumulant_rate': fourth.tolist(),
            'pressure_relative_error': float(np.linalg.norm(rate-pressure_reference)/np.linalg.norm(pressure_reference)),
            'fourth_cumulant_relative_error': float(np.linalg.norm(fourth-cumulant_reference)/np.linalg.norm(cumulant_reference)),
            'number_rate': float(-residual.sum()), 'momentum_rate': (-residual@v).tolist(),
            'energy_rate': float(-residual@np.sum(v*v, axis=1)/2),
            'compile_first_action_s': first_s, 'warm_action_s': warm,
            'warm_action_median_s': float(np.median(warm))}
        metadata['rows'].append(row)
        key = f'{method}_n{order}_L{extent:g}_eps{softening:g}'
        arrays[key] = computed
        arrays[f'velocity_n{order}_L{extent:g}'] = v
        arrays[f'weights_n{order}_L{extent:g}'] = w
        arrays[f'exact_n{order}_L{extent:g}'] = exact
        save()
        print(f'  L1 RHS error={row["relative_L1_error"]:.2%}, pressure={row["pressure_relative_error"]:.2%}, fourth cumulant={row["fourth_cumulant_relative_error"]:.2%}; warm {np.median(warm):.3f}s.', flush=True)
        del compiler, grid
        jax.clear_caches()

finest = [r for r in metadata['rows'] if r['order']==ORDERS[-1] and r['extent']==EXTENT and r['softening']==0.]
metadata['status'] = 'passed_initial_L1_target' if all(r['relative_L1_error']<TARGET for r in finest) else 'unresolved'
metadata['scope'] = 'Instantaneous Gaussian collision term only. Finite-grid conservation and entropy do not imply continuum full-density accuracy. No finite-time trajectory, physical collision coefficient or magnetized validity claim.'
metadata['norms'] = {'L1': 'sum(w*abs(C_discrete-C_continuum))/sum(w*abs(C_continuum))',
    'RMS': 'sqrt(sum(w*F*(C_discrete/F-C_continuum/F)^2))',
    'core': '|v|<=3; core and tail L1 contributions share the full reference denominator',
    'fourth_cumulant': 'd<v_i^4>/dt-6*A_i*d<v_i^2>/dt; normalized continuum Gaussian inputs without finite-cube mass repair'}
np.savez_compressed(OUTPUT/'arrays.npz', **arrays)
fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
for method, color in zip(DERIVATIVES, ['#175f98', '#d46a28']):
    rows = [r for r in metadata['rows'] if r['derivative']==method and r['extent']==EXTENT and r['softening']==0.]
    for ax, key in zip(axes.ravel(), ['relative_L1_error', 'pressure_relative_error',
            'fourth_cumulant_relative_error', 'warm_action_median_s']):
        ax.semilogy([r['order'] for r in rows], [r[key] for r in rows], 'o-', color=color, label=method.replace('_', ' '))
for ax, title, ylabel in zip(axes.ravel(), ['Full collision term', 'Pressure rate', 'Fourth-cumulant rate', 'Warm matrix-free action'],
        ['Relative L¹ error', 'Relative error', 'Relative error', 'Seconds']):
    ax.set(xlabel='Gauss points per velocity axis', title=title, ylabel=ylabel)
    ax.grid(alpha=.2)
for ax in axes.ravel()[:3]:
    ax.axhline(TARGET, color='.5', ls=':', lw=1)
axes[0, 0].legend(fontsize=9)
fig.suptitle('Conservation alone does not establish Landau accuracy', fontweight='bold')
fig.savefig(OUTPUT/'convergence.png', dpi=180)
fig.savefig(OUTPUT/'convergence.svg', metadata={'Creator': None, 'Date': None})
plt.close(fig)

n = 20
v = arrays[f'velocity_n{n}_L5'].reshape(n, n, n, 3)
x = v[:, 0, 0, 0]
j = n//2
fields = [arrays[f'exact_n{n}_L5']]+[arrays[f'{m}_n{n}_L5_eps0'] for m in DERIVATIVES]
fields = [a.reshape(n, n, n)[:, j, :] for a in fields]
limit = max(float(np.max(abs(a))) for a in fields)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout='constrained')
for ax, field, title in zip(axes, fields, ['Continuum Gaussian oracle', 'Global polynomial derivative', 'Local quadratic derivative']):
    mesh = ax.pcolormesh(x, x, field.T, shading='nearest', cmap='RdBu_r', vmin=-limit, vmax=limit)
    ax.set(title=title, xlabel='vₓ', ylabel='v_z', aspect='equal')
fig.colorbar(mesh, ax=axes, shrink=.85, label='∂F/∂t: gain (+), loss (−)')
fig.suptitle(f'Initial redistribution in velocity space • vᵧ = {x[j]:.3f}, n = {n}', fontweight='bold')
fig.savefig(OUTPUT/'redistribution.png', dpi=180)
fig.savefig(OUTPUT/'redistribution.svg', metadata={'Creator': None, 'Date': None})
plt.close(fig)
metadata['outputs'] = {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(OUTPUT.iterdir()) if p.suffix in ('.npz', '.png', '.svg')}
save()
print(f'Initial full-F validation status={metadata["status"]}; elapsed={metadata["wall_s"]:.1f}s; results saved.', flush=True)
if metadata['status']=='unresolved':
    raise RuntimeError('Declared 1% full collision-term accuracy is unresolved; measurements and figures retained.')
