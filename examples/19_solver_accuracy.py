"""Separate nonlinear/linear tolerance scans on one fixed mirror grid.

The tightest solved endpoint is a numerical reference for this fixed method,
box, grid and timestep. This experiment does not estimate continuum error.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
from time import perf_counter
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sato_morrison.collisions as collision_module
import sato_morrison.geometry as geometry_module
import sato_morrison.reference as reference_module
import sato_morrison.solver as solver_module
from sato_morrison.collisions import cartesian_grid, mobility_action
from sato_morrison.geometry import Field
from sato_morrison.reference import gauss_interval, progress, run_metadata
from sato_morrison.solver import entropy, invariant_diagnostics, lagged_entropy_compiler, lagged_entropy_step

FIELD = Field('mirror', amplitude=.15)
BOUNDS = [(.8, 1.2), (-.2, .2), (.1, .5)]
NX, NU, NMU = 3, 25, 21
U_MAX, MU_MAX, D, FINAL_TIME, DT = 4., 20., .1, .02, .005
CHUNK, LINEAR_MAX_STEPS, MAX_NEWTON_STEPS = 1048576, 3000, 80
RUNS = [dict(name='outer_1e10', outer=1e-10, inner=1e-6),
        dict(name='outer_1e11', outer=1e-11, inner=1e-6),
        dict(name='baseline', outer=1e-12, inner=1e-6),
        dict(name='inner_1e4', outer=1e-12, inner=1e-4),
        dict(name='inner_1e5', outer=1e-12, inner=1e-5)]
SCANS = {'nonlinear': ['outer_1e10', 'outer_1e11', 'baseline'],
         'linear': ['inner_1e4', 'inner_1e5', 'baseline']}
OBSERVABLES = ['entropy_gain_per_particle', 'final_production_per_particle',
               'relaxation_moment_change_per_particle']
TARGETS = {'weighted_endpoint': 1e-8, 'relative_principal_observables': 1e-6,
           'accumulated_invariants': 1e-9}
OUTPUT = Path(__file__).resolve().parents[1] / 'results' / 'solver_accuracy'
OUTPUT.mkdir(parents=True, exist_ok=True)
inputs = {'field': FIELD.__dict__, 'bounds': BOUNDS, 'nx': NX, 'nu': NU, 'nmu': NMU,
    'u_domain': [-U_MAX, U_MAX], 'mu_domain': [0, MU_MAX], 'collision_strength': D,
    'final_time': FINAL_TIME, 'dt': DT, 'runs': RUNS, 'scans': SCANS, 'reference_run': 'baseline',
    'pair_chunk': CHUNK, 'linear_max_steps': LINEAR_MAX_STEPS, 'max_newton_steps': MAX_NEWTON_STEPS,
    'preconditioner_axis': 0, 'time_discretization': 'lagged-mobility backward Euler in entropy variables',
    'spatial_discretization': 'global Lagrange polynomial derivative on positive Gauss-Legendre quadrature',
    'velocity_discretization': 'local quadratic u derivative; positive Gauss-Legendre u/mu quadrature',
    'initial': 'exp(-E-.2mu+.1sin(pi*y/.4)*u)', 'targets': TARGETS, 'targets_applied_to': 'last adjacent pair of each scan',
    'scope': 'Conditional solver tolerance check on one fixed finite-dimensional method; the reference is not continuum truth', 'seed': None}
metadata = run_metadata(inputs, model='sm_local_nonlinear discrete-energy projector',
    boundary='natural no-flux collision-only Cartesian box')
dependencies = [Path(__file__), *[Path(module.__file__) for module in
    (collision_module, geometry_module, reference_module, solver_module)]]
metadata['experiment_dependency_sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in dependencies}
metadata['timing_scope'] = ('One process with a reused compiler. The first call for each distinct inner tolerance '
    'may compile because the PCG tolerance is static; costs retain compilation, evidence writes and concurrent machine load.')
result = {'metadata': metadata, 'rows': [], 'comparisons': {}, 'status': 'incomplete'}
snapshots = {}; completed_endpoints = {}


def save_evidence():
    temporary = OUTPUT/'summary.json.tmp'
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    temporary.replace(OUTPUT/'summary.json')
    if snapshots:
        arrays = {'initial': np.asarray(initial), 'weights': np.asarray(grid.weights),
            'energy': np.asarray(grid.energy), 'mu_index': np.asarray(grid.mu_index), 'shape': np.asarray(grid.shape)}
        arrays.update({name+'_accepted_states': np.stack(states) for name, states in snapshots.items()})
        arrays.update({name+'_endpoint': endpoint for name, endpoint in completed_endpoints.items()})
        np.savez(OUTPUT/'states.npz', **arrays)


def diagnostic_record(record):
    """Preserve nonfinite failed-trial evidence without illegal JSON numbers."""
    clean = dict(record)
    invalid = {key: repr(float(value)) for key, value in record.items()
               if isinstance(value, (float, np.floating)) and not np.isfinite(value)}
    if invalid:
        clean.update({key: None for key in invalid})
        clean['nonfinite_fields'] = invalid
    return clean


def endpoint_errors(first, second, weights):
    """Population and log differences in explicitly weighted endpoint norms."""
    first, second, weights = np.asarray(first), np.asarray(second), np.asarray(weights)
    if (first.shape != second.shape or first.shape != weights.shape
        or not all(np.all(np.isfinite(value)) for value in (first, second, weights))
        or np.any(first <= 0) or np.any(second <= 0) or np.any(weights <= 0)):
        raise ValueError('Finite positive matching endpoint arrays and weights required')
    population = float(np.sqrt(np.sum(weights*(first-second)**2)/np.sum(weights*second**2)))
    log = float(np.sqrt(np.sum(weights*second*(np.log(first)-np.log(second))**2)/np.sum(weights*second)))
    return {'population_weighted_relative_L2': population, 'entropy_scaled_log_L2': log,
            'maximum_absolute_population_difference': float(np.max(np.abs(first-second))),
            'maximum_absolute_log_difference': float(np.max(np.abs(np.log(first)-np.log(second))))}


def compare_pair(previous, last, endpoints, weights):
    errors = endpoint_errors(endpoints[previous['name']], endpoints[last['name']], weights)
    changes = {name: abs(last[name]-previous[name])/max(abs(last[name]), 1e-30) for name in OBSERVABLES}
    finite = all(np.isfinite(value) for value in [*errors.values(), *changes.values()])
    return {'previous_run': previous['name'], 'last_run': last['name'], 'endpoint_errors': errors,
        'relative_principal_observable_changes': changes,
        'status': 'passed' if finite and max(errors[key] for key in
            ('population_weighted_relative_L2', 'entropy_scaled_log_L2')) < TARGETS['weighted_endpoint']
            and max(changes.values()) < TARGETS['relative_principal_observables'] else 'unresolved'}


print(f'Fixed mirror solver accuracy: {NX}^3 x {NU} x {NMU}, D={D}, T={FINAL_TIME}, '
      f'dt={DT}; five unique runs, baseline reused; output={OUTPUT}', flush=True)
save_evidence()
with progress('Prepare the fixed mirror grid and initial diagnostics'):
    start = perf_counter()
    spatial = [gauss_interval(NX, a, b) for a, b in BOUNDS]
    axes = [pair[0] for pair in spatial]
    u, wu = gauss_interval(NU, -U_MAX, U_MAX);mu, wm = gauss_interval(NMU, 0, MU_MAX)
    grid = cartesian_grid(*axes, u, mu, FIELD, velocity_weights=(wu, wm),
        spatial_weights=[pair[1] for pair in spatial], compact=True, spatial_discretization='polynomial')
    x, y, z, uu, mm = np.meshgrid(*axes, u, mu, indexing='ij')
    log_equilibrium = -grid.energy-.2*mm.ravel()
    observable = jnp.asarray(np.sin(np.pi*y.ravel()/.4)*uu.ravel())
    initial = jnp.exp(log_equilibrium+.1*observable)
    number = float(jnp.sum(grid.weights*initial));initial_entropy = float(entropy(initial, grid.weights))
    moment0 = float(jnp.vdot(grid.weights*initial, observable))
    initial_marginal = np.bincount(np.asarray(grid.mu_index), weights=np.asarray(grid.weights*initial), minlength=NMU)
    physical_apply = jax.jit(lambda state: mobility_action(grid, state, jnp.log(state), collision_strength=D, chunk_size=CHUNK))
    print('Compile the physical mobility diagnostic on this fixed grid.', flush=True)
    initial_flux = physical_apply(initial);initial_flux.block_until_ready()
    result['initial'] = {'number': number, 'entropy': initial_entropy, 'relaxation_moment': moment0,
        'magnetic_moment_bin_populations': initial_marginal.tolist(),
        'physical_production_per_particle': float(jnp.vdot(jnp.log(initial), initial_flux))/number,
        'setup_and_initial_diagnostics_wall_s': perf_counter()-start}
    compiler = lagged_entropy_compiler(grid, collision_strength=D, chunk_size=CHUNK,
        linear_max_steps=LINEAR_MAX_STEPS, preconditioner_axis=0)
save_evidence()
seen_inner = set()
for configuration in RUNS:
    name = configuration['name'];start = perf_counter()
    row = {**configuration, 'status': 'running', 'history': [], 'load_start': os.getloadavg(),
           'first_use_of_inner_tolerance': configuration['inner'] not in seen_inner}
    result['rows'].append(row);snapshots[name] = [np.asarray(initial)];state = initial
    if row['first_use_of_inner_tolerance']:
        print(f"Compile/reuse Newton-PCG kernels for inner tolerance {configuration['inner']:g}.", flush=True)
    seen_inner.add(configuration['inner']);save_evidence()
    try:
        with progress(f"Solve {name}: outer={configuration['outer']:g}, inner={configuration['inner']:g}"):
            for index in range(round(FINAL_TIME/DT)):
                step = {'index': index+1, 'time': (index+1)*DT, 'status': 'running', 'newton_history': []}
                row['history'].append(step);save_evidence();begin = perf_counter()
                def observe(record):
                    k = record['iteration']-1;clean = diagnostic_record(record)
                    if k == len(step['newton_history']):step['newton_history'].append(clean)
                    else:step['newton_history'][k] = clean
                    save_evidence()
                    if 'nonfinite_fields' in clean:
                        print(f"  {name}, step {index+1}: nonfinite trial diagnostics {clean['nonfinite_fields']}", flush=True)
                    elif 'fraction' in record:
                        print(f"  {name}, step {index+1}, Newton {record['iteration']}: "
                            f"residual={record['relative_residual']:.3e}, PCG={record['linear_iterations']}, "
                            f"true residual={record['true_linear_relative_residual']:.3e}, fraction={record['fraction']:g}", flush=True)
                answer = lagged_entropy_step(grid, state, DT, collision_strength=D, compiled=compiler,
                    chunk_size=CHUNK, rtol=configuration['outer'], linear_rtol=configuration['inner'],
                    linear_max_steps=LINEAR_MAX_STEPS, max_steps=MAX_NEWTON_STEPS,
                    preconditioner_axis=0, iteration_callback=observe)
                state = answer.f;check = invariant_diagnostics(grid, state, initial)
                marginal = np.bincount(np.asarray(grid.mu_index), weights=np.asarray(grid.weights*state), minlength=NMU)
                snapshots[name].append(np.asarray(state))
                step.update({'status': 'passed', 'wall_s': perf_counter()-begin, 'newton_iterations': answer.iterations,
                    'pcg_iterations': answer.linear_iterations, 'nonlinear_relative_residual': answer.relative_residual,
                    'maximum_linear_relative_residual': answer.linear_relative_residual,
                    'entropy_change': answer.entropy_change, 'lagged_mobility_dissipation': answer.entropy_production,
                    'generalized_KL': answer.generalized_KL, 'entropy_identity_error': answer.entropy_identity_error,
                    'residual_entropy_defect': answer.residual_entropy_defect, 'entropy_defect_bound': answer.entropy_defect_bound,
                    'magnetic_moment_bin_populations': marginal.tolist(),
                    'relative_marginal_bin_errors': ((marginal-initial_marginal)/initial_marginal).tolist(), **check})
                save_evidence()
                if (not all(np.isfinite(check[key]) for key in ('number_error','energy_error','marginal_error','min_f'))
                    or max(check[key] for key in ('number_error','energy_error','marginal_error')) > TARGETS['accumulated_invariants']):
                    raise RuntimeError('Accumulated invariant diagnostic failed')
                print(f'  {name}: accepted t={(index+1)*DT:g}, {answer.iterations} Newton, '
                      f'{answer.linear_iterations} PCG, endpoint residual={answer.relative_residual:.3e}', flush=True)
        with progress(f'Compute final physical diagnostics for {name}'):
            flux = physical_apply(state);flux.block_until_ready()
            gain = float(entropy(state, grid.weights))-initial_entropy
            production = float(jnp.vdot(jnp.log(state), flux))
            moment_change = (float(jnp.vdot(grid.weights*state, observable))-moment0)/number
            if not np.all(np.isfinite([gain,production,moment_change])) or gain <= 0 or production < -1e-13:
                raise RuntimeError('Final physical diagnostic is nonfinite or violates its entropy sign check')
            row.update({'status': 'passed', 'entropy_gain_per_particle': gain/number,
                'final_production_per_particle': production/number, 'relaxation_moment_change_per_particle': moment_change,
                'entropy_telescoping_error': gain-sum(stage['entropy_change'] for stage in row['history'])})
            completed_endpoints[name] = np.asarray(state)
    except Exception as error:
        if row['history'] and row['history'][-1]['status'] == 'running':row['history'][-1]['status'] = 'failed'
        row.update({'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)})
        print(f'  FAILED {name}: {type(error).__name__}: {error}', flush=True)
    row.update({'wall_s': perf_counter()-start, 'load_end': os.getloadavg(),
        'total_recorded_pcg_iterations': sum(record['linear_iterations'] for stage in row['history'] for record in stage['newton_history']),
        'total_recorded_newton_corrections': sum(len(stage['newton_history']) for stage in row['history']),
        'accepted_steps': len(snapshots[name])-1,
        'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)})
    save_evidence()

passed = {row['name']: row for row in result['rows'] if row['status'] == 'passed'}
if 'baseline' in passed:
    for name, row in passed.items():row['errors_against_reference'] = endpoint_errors(completed_endpoints[name], completed_endpoints['baseline'], np.asarray(grid.weights))
for label, sequence in SCANS.items():
    pairs = [compare_pair(passed[a], passed[b], completed_endpoints, np.asarray(grid.weights)) if a in passed and b in passed
        else {'previous_run': a, 'last_run': b, 'status': 'not_run'} for a, b in zip(sequence, sequence[1:])]
    result['comparisons'][label] = {'adjacent_pairs': pairs, 'last_pair': pairs[-1],
        'status': pairs[-1]['status'], 'criterion_scope': 'Last adjacent pair; coarser runs retained separately'}
result['status'] = 'passed' if len(passed) == 5 and all(value['status'] == 'passed' for value in result['comparisons'].values()) else 'unresolved'
result['dependencies_unchanged'] = all(hashlib.sha256(path.read_bytes()).hexdigest() == metadata['experiment_dependency_sha256'][path.name] for path in dependencies)
if not result['dependencies_unchanged']:result['status'] = 'unresolved'
save_evidence()

order = ['outer_1e10','outer_1e11','baseline','inner_1e5','inner_1e4']
labels = [r'Outer $10^{-10}$',r'Outer $10^{-11}$','Baseline',r'Inner $10^{-5}$',r'Inner $10^{-4}$']
by_name = {row['name']: row for row in result['rows']}
fig, axes = plt.subplots(1,2,figsize=(10.5,4.3))
for key, label, color in [('population_weighted_relative_L2','Population norm','#2166ac'),('entropy_scaled_log_L2','Entropy-scaled log norm','#d95f02')]:
    values = [max(by_name[name].get('errors_against_reference',{}).get(key,np.nan),1e-18) for name in order]
    axes[0].plot(range(5), values, 'o-', label=label, color=color)
axes[0].axhline(TARGETS['weighted_endpoint'], color='#444444',linestyle='--',linewidth=1,label='Last-pair norm target')
axes[0].set(yscale='log',ylabel='Endpoint difference from tightest solved reference',title='Fixed-grid solver accuracy')
axes[0].legend(frameon=False,fontsize=8)
axes[0].text(.02,.02,'Zero differences displayed at $10^{-18}$',transform=axes[0].transAxes,fontsize=8)
counts = [by_name[name]['total_recorded_pcg_iterations'] for name in order]
axes[1].bar(range(5),counts,color='#6296b8')
for i,name in enumerate(order):
    row=by_name[name];label=f"{row['total_recorded_newton_corrections']} Newton" if row['status']=='passed' else 'FAILED'
    axes[1].annotate(label,(i,counts[i]),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
axes[1].set(ylabel='Recorded PCG iterations over four steps',title='Solver work; baseline used once')
for axis in axes:
    axis.set_xticks(range(5),labels,rotation=25,ha='right');axis.grid(axis='y',alpha=.2)
fig.suptitle('Mirror tolerance scans: fixed grid and timestep; reference is conditional',fontsize=11)
fig.tight_layout();fig.savefig(OUTPUT/'accuracy.png',dpi=180);plt.close(fig)
print(f"Saved {OUTPUT/'summary.json'}, states.npz and accuracy.png; status={result['status']}",flush=True)
if any(row['status']=='failed' for row in result['rows']):raise RuntimeError('One or more tolerance solves failed; accepted states and failure reasons are saved')
if result['status'] != 'passed':raise RuntimeError('Solver tolerance comparisons remain unresolved; evidence is saved')
