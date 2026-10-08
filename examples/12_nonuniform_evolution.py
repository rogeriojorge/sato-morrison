"""Positive finite-time nonlinear evolution with lagged-mobility backward Euler.

SM_EVOLUTION_FIELDS and SM_EVOLUTION_CASES select predefined independent jobs.
They do not change scientific inputs. With both unset the complete campaign runs.
Historical discrete-gradient and old-population-scaled results remain separate.
This campaign uses a fixed initial population to scale residuals and Newton systems.
"""
import hashlib
import json
import os
import resource
import sys
from pathlib import Path
from time import perf_counter
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sato_morrison.collisions import cartesian_grid, derivative_matrix, mobility_action
from sato_morrison.geometry import Field, field_vector
from sato_morrison.reference import gauss_interval, progress, run_metadata
from sato_morrison.solver import (lagged_entropy_compiler, lagged_entropy_step,
                                 entropy, invariant_diagnostics)

# Collision-only boxes have the natural boundary of the weak operator. No
# streaming is added across these non-tangent box walls. The state/pair measures
# use the same positive velocity quadrature, respectively wX B wu wmu and
# wX (B wu wmu)(B wu' wmu'). The discrete-energy projector is unmodified.
FIELDS = [Field('mirror', amplitude=.15), Field('dipole'),
          Field('nonaxisymmetric', amplitude=.03)]
BOUNDS = [(.8, 1.2), (-.2, .2), (.1, .5)]
NX, NU, NMU = 5, 25, 21
X_ORDERS, U_ORDERS, MU_ORDERS = [3, 5, 7], [17, 21, 25], [13, 17, 21]
U_MAX, MU_MAX = 4., 20.
D, FINAL_TIME, DT = .1, .02, .005
DT_VALUES = [.005, .0025, .00125, .000625, .0003125]
CHUNK, NEWTON_RTOL = 1048576, 1e-12
LINEAR_RTOL, LINEAR_MAX_STEPS, MAX_NEWTON_STEPS = 1e-6, 3000, 80
PRECONDITIONER_AXIS = 0
RELATIVE_TARGET = .01
RESIDUAL_METRIC = 'fixed_reference_population'
OUTPUT = Path(__file__).resolve().parents[1] / 'results' / 'nonuniform_fixed_reference'
OUTPUT.mkdir(parents=True, exist_ok=True)
print(f'Nonuniform positive nonlinear collision evolution; D={D}, T={FINAL_TIME}; '
      f'natural collision boundaries; fixed-initial-population Newton-PCG; output={OUTPUT}', flush=True)


def require_finite(value, path='result'):
    """Reject nonfinite numerical evidence before assigning a passing status."""
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite(item, f'{path}.{key}')
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite(item, f'{path}[{index}]')
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError(f'Nonfinite numerical evidence at {path}')


def scaled_residual_norm(residual, population):
    """Stable norm even when an obsolete tail scale makes components enormous."""
    scaled = np.asarray(residual)/np.sqrt(np.asarray(population))
    largest = float(np.max(np.abs(scaled)))
    return 0. if largest == 0 else float(largest*np.linalg.norm(scaled/largest)/np.sqrt(np.sum(population)))


def save_state(initial, previous, state, reference_population, time, field, case):
    """Retain the last accepted state for independent audits after a failure."""
    temporary = OUTPUT/'last_state.npz.tmp'
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, initial=np.asarray(initial), previous=np.asarray(previous),
            state=np.asarray(state), reference_population=np.asarray(reference_population),
            time=time, field=field, case=case, provenance_id=provenance_id)
    temporary.replace(OUTPUT/'last_state.npz')


def continuum_moments(field, xx, yy, zz, strength):
    """Continuous additional invariants; their drift is scheme error, not decay."""
    radius2 = xx**2 + yy**2
    if field.kind == 'mirror':
        psi = field.strength*radius2/2 + field.amplitude*radius2*zz**2/2 - field.amplitude*radius2**2/8
    else:
        psi = field.strength*radius2/(radius2+zz**2)**1.5
    if field.kind in ('mirror', 'dipole'):
        return {'psi': psi, 'psi_squared': psi**2, 'radius_squared': radius2,
                'z': zz, 'z_squared': zz**2, 'radius_squared_z': radius2*zz}
    chi = -field.strength*zz/(radius2+zz**2)**1.5 + field.amplitude*field.strength*xx*yy/field.length
    return {'B': strength, 'chi': chi, 'B_squared': strength**2, 'chi_squared': chi**2}


def evaluate(field, nx=NX, nu=NU, nmu=NMU, umax=U_MAX, mumax=MU_MAX, dt=DT, *, case_name=None):
    spatial = [gauss_interval(nx, a, b) for a, b in BOUNDS]
    axes = [pair[0] for pair in spatial]
    u, wu = gauss_interval(nu, -umax, umax)
    mu, wm = gauss_interval(nmu, 0, mumax)
    start = perf_counter()
    grid = cartesian_grid(*axes, u, mu, field, velocity_weights=(wu, wm), compact=True,
                          spatial_weights=[pair[1] for pair in spatial],spatial_discretization='polynomial')
    setup_s = perf_counter()-start
    xx, yy, zz = np.meshgrid(*axes, indexing='ij')
    positions = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)
    strength = np.linalg.norm(np.asarray(jax.vmap(lambda p: field_vector(p, field))(
        jnp.asarray(positions))), axis=1).reshape(xx.shape)
    x, y, z, uu, mm = np.meshgrid(*axes, u, mu, indexing='ij')
    log_equilibrium=-grid.energy-.2*mm.ravel()
    observable=jnp.asarray(np.sin(np.pi*y.ravel()/.4)*uu.ravel())
    initial = jnp.exp(log_equilibrium+.1*observable)
    reference_population = grid.weights*initial
    number = float(jnp.sum(reference_population))
    save_state(initial, initial, initial, reference_population, 0., field.kind, case_name)
    initial_marginal = np.bincount(np.asarray(grid.mu_index),
        weights=np.asarray(grid.weights*initial), minlength=nmu)
    initial_entropy = float(entropy(initial, grid.weights))
    reference=jnp.exp(log_equilibrium)
    reference=reference*number/jnp.sum(grid.weights*reference)
    log_reference=jnp.log(reference)
    initial_relative_entropy=float(jnp.sum(grid.weights*(initial*(jnp.log(initial)-log_reference)-initial+reference)))
    def perturbation_statistics(state):
        perturbation=jnp.log(state)-log_reference
        mean=float(jnp.sum(grid.weights*state*perturbation)/number)
        variance=float(jnp.sum(grid.weights*state*(perturbation-mean)**2)/number)
        return {'mean':mean,'variance':variance}
    initial_perturbation=perturbation_statistics(initial)
    moment0 = float(jnp.vdot(grid.weights*initial, observable))
    diagnostics = continuum_moments(field, xx, yy, zz, strength)
    diagnostics = {name: jnp.asarray(np.broadcast_to(value[..., None, None], grid.shape).ravel())
                   for name, value in diagnostics.items()}
    scales = {name: float(jnp.vdot(grid.weights*initial, jnp.abs(value)))
              for name, value in diagnostics.items()}
    moments0 = {name: float(jnp.vdot(grid.weights*initial, value)) for name, value in diagnostics.items()}
    discrete_grad_b = []
    for axis, nodes in enumerate(axes):
        value = np.tensordot(derivative_matrix(nodes,method='polynomial'), strength, axes=(1, axis))
        discrete_grad_b.append(np.moveaxis(value, 0, axis))
    minimum_grad_b = float(np.linalg.norm(np.stack(discrete_grad_b, axis=-1), axis=-1).min())
    # This establishes injectivity of the modified discrete energy flow on these
    # sampled nodes: u is recovered by b·V_X, then mu by the nonzero D_h B.
    reproduction = float(np.max(np.abs(derivative_matrix(u)@(u*u/2)-u)))
    if minimum_grad_b <= 0 or reproduction > 1e-10:
        raise RuntimeError('discrete energy-flow injectivity check failed')
    print(f'  Preparing/compiling {field.kind}: {grid.size} nodes, '
          f'{grid.pair_count} local pairs, x-line Newton-PCG', flush=True)
    compiler = lagged_entropy_compiler(grid, collision_strength=D, chunk_size=CHUNK,
        linear_max_steps=LINEAR_MAX_STEPS, preconditioner_axis=PRECONDITIONER_AXIS,
        reference_population=reference_population)
    apply = jax.jit(lambda f, h: mobility_action(grid, f, h, collision_strength=D, chunk_size=CHUNK))
    initial_flux = apply(initial, jnp.log(initial));initial_flux.block_until_ready()
    initial_production = float(jnp.vdot(jnp.log(initial), initial_flux))
    rates0 = {name: -float(jnp.vdot(value, initial_flux))/scales[name] for name, value in diagnostics.items()}
    steps = int(round(FINAL_TIME/dt))
    if abs(steps*dt-FINAL_TIME) > 1e-13:
        raise ValueError('time horizon must contain an integer number of timesteps')
    state = initial;history = [];load_start = os.getloadavg()
    for index in range(steps):
        start = perf_counter()
        previous_state = state
        def record_iteration(record):
            save_evidence(OUTPUT/'last_correction.json', {'field': field.kind,
                'case': case_name, 'step': index+1, 'provenance_id': provenance_id,
                'residual_metric': RESIDUAL_METRIC, **record})
            if 'fraction' not in record:
                print(f"    step {index+1}, Newton {record['iteration']}: "
                      f"PCG {record['linear_iterations']}, true residual "
                      f"{record['true_linear_relative_residual']:.3e}, "
                      f"{record['linear_wall_s']:.1f}s", flush=True)
        answer = lagged_entropy_step(grid, state, dt, collision_strength=D,
            compiled=compiler, chunk_size=CHUNK, rtol=NEWTON_RTOL,
            linear_rtol=LINEAR_RTOL, linear_max_steps=LINEAR_MAX_STEPS, max_steps=MAX_NEWTON_STEPS,
            preconditioner_axis=PRECONDITIONER_AXIS, iteration_callback=record_iteration)
        state = answer.f
        if answer.residual_metric != RESIDUAL_METRIC:
            raise RuntimeError('unexpected nonlinear residual metric')
        residual = np.asarray(compiler.evaluate(jnp.log(state), jnp.log(previous_state), dt))
        reference_residual = scaled_residual_norm(residual, reference_population)
        old_residual = scaled_residual_norm(residual, grid.weights*previous_state)
        if not np.isfinite(reference_residual) or reference_residual > 5*NEWTON_RTOL:
            raise RuntimeError('represented-state reference residual failed')
        check = invariant_diagnostics(grid, state, initial)
        if max(check[k] for k in ('number_error', 'energy_error', 'marginal_error')) > 1e-9:
            raise RuntimeError(f'accumulated invariant error failed: {check}')
        marginal = np.bincount(np.asarray(grid.mu_index),
            weights=np.asarray(grid.weights*state), minlength=nmu)
        history.append({'time': (index+1)*dt, 'wall_s': perf_counter()-start,
            'newton_iterations': answer.iterations, 'pcg_iterations': answer.linear_iterations,
            'nonlinear_relative_residual': float(answer.relative_residual),
            'residual_metric': answer.residual_metric,
            'recomputed_reference_relative_residual': reference_residual,
            'old_population_relative_residual': old_residual,
            'unscaled_population_residual_norm': float(np.linalg.norm(residual)),
            'maximum_linear_relative_residual': answer.linear_relative_residual,
            'entropy_change': answer.entropy_change, 'entropy_identity_error': answer.entropy_identity_error,
            'lagged_mobility_dissipation': answer.entropy_production, 'generalized_KL': answer.generalized_KL,
            'entropy_defect_bound': answer.entropy_defect_bound,
            'residual_entropy_defect': answer.residual_entropy_defect, 'newton_history': answer.iteration_history,
            'magnetic_moment_bin_populations': marginal.tolist(),
            'relative_marginal_bin_errors': ((marginal-initial_marginal)/initial_marginal).tolist(),
            **check})
        save_state(initial, previous_state, state, reference_population, (index+1)*dt, field.kind, case_name)
        save_evidence(OUTPUT/'partial_case.json', {'status': 'incomplete',
            'provenance_id': provenance_id, 'field': field.kind, 'case': case_name,
            'parameters': {'field': field.kind, 'nx': nx,
                'nu': nu, 'nmu': nmu, 'umax': umax, 'mumax': mumax, 'dt': dt},
            'initial_magnetic_moment_bin_populations': initial_marginal.tolist(), 'history': history})
        print(f'  {field.kind} {nx}^3 x {nu} x {nmu}, U={umax:g}, M={mumax:g}, dt={dt:g}: '
              f't={history[-1]["time"]:.6f}, Newton={answer.iterations}, PCG={answer.linear_iterations}, '
              f'dS={answer.entropy_change:.3e}, {history[-1]["wall_s"]:.2f}s', flush=True)
    final_flux = apply(state, jnp.log(state));final_flux.block_until_ready()
    final_production = float(jnp.vdot(jnp.log(state), final_flux))
    gain = float(entropy(state, grid.weights))-initial_entropy
    if (not np.all(np.isfinite([initial_production, final_production, gain]))
        or initial_production <= 0 or final_production < -1e-13 or gain <= 0):
        raise RuntimeError('entropy production/gain failed')
    continuous = {name: {'relative_signed_drift': (float(jnp.vdot(grid.weights*state, value))-moments0[name])/scales[name],
        'initial_relative_rate': rates0[name], 'final_relative_rate': -float(jnp.vdot(value, final_flux))/scales[name]}
        for name, value in diagnostics.items()}
    stored = [grid.coefficients, grid.energy_flow, grid.weights, grid.energy, grid.mu_index,
              *grid.local_quadrature, *[matrix for _, matrix in grid.derivatives]]
    row = {'field': field.kind, 'nx': nx, 'nu': nu, 'nmu': nmu, 'umax': umax, 'mumax': mumax, 'dt': dt,
        'nodes': grid.size, 'conceptual_unordered_pairs': grid.pair_count,
        'residual_metric': RESIDUAL_METRIC, 'reference_population_total': number,
        'minimum_reference_population': float(jnp.min(reference_population)),
        'stored_grid_bytes': sum(array.nbytes for array in stored),
        'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        'peak_rss_scope': 'Cumulative process high-water mark; includes setup and compiled runtime, not grid arrays alone',
        'pair_workspace_budget': CHUNK, 'minimum_discrete_grad_B': minimum_grad_b,
        'minimum_B':float(strength.min()),'maximum_B':float(strength.max()),
        'u_quadratic_derivative_error': reproduction, 'number': number,
        'initial_magnetic_moment_bin_populations': initial_marginal.tolist(),
        'entropy_gain_per_particle': gain/number,
        'initial_relative_entropy_per_particle':initial_relative_entropy/number,
        'relative_entropy_decrease_fraction':gain/initial_relative_entropy,
        'initial_perturbation':initial_perturbation,'final_perturbation':perturbation_statistics(state),
        'entropy_telescoping_error': gain-sum(step['entropy_change'] for step in history),
        'initial_production_per_particle': initial_production/number,
        'final_production_per_particle': final_production/number,
        'relaxation_moment_change_per_particle': (float(jnp.vdot(grid.weights*state, observable))-moment0)/number,
        'relaxation_moment_decrease_fraction':1-float(jnp.vdot(grid.weights*state,observable))/moment0,
        'continuum_invariant_errors': continuous, 'setup_s': setup_s, 'history': history,
        'load_average_start': load_start, 'load_average_end': os.getloadavg(), 'status': 'passed'}
    require_finite(row)
    jax.clear_caches()
    return row


def compare(sequence):
    observable_names = ['entropy_gain_per_particle', 'final_production_per_particle',
                        'relaxation_moment_change_per_particle']
    require_finite([{name: row[name] for name in observable_names} for row in sequence])
    def pair(previous, last):
        changes = {name: abs(last[name]-previous[name])/max(abs(last[name]), 1e-30)
                   for name in observable_names}
        return {'relative_changes': changes, 'target': RELATIVE_TARGET,
            'status': 'passed' if max(changes.values()) < RELATIVE_TARGET else 'unresolved',
            'previous_parameters': {k: previous[k] for k in ('nx', 'nu', 'nmu', 'umax', 'mumax', 'dt')},
            'last_parameters': {k: last[k] for k in ('nx', 'nu', 'nmu', 'umax', 'mumax', 'dt')}}
    return {**pair(*sequence[-2:]),
        'adjacent_comparisons': [pair(a,b) for a,b in zip(sequence,sequence[1:])],
        'coarsest_to_finest': pair(sequence[0],sequence[-1])}


inputs = {'fields': [field.__dict__ for field in FIELDS], 'bounds': BOUNDS,
    'nx': NX, 'nu': NU, 'nmu': NMU, 'x_orders': X_ORDERS, 'u_orders': U_ORDERS, 'mu_orders': MU_ORDERS,
    'u_domain': [-U_MAX, U_MAX], 'mu_domain': [0, MU_MAX], 'collision_strength': D,
    'final_time': FINAL_TIME, 'dt_values': DT_VALUES, 'newton_rtol': NEWTON_RTOL,
    'pair_chunk': CHUNK, 'time_discretization': 'lagged-mobility backward Euler in entropy variables',
    'preconditioner_axis': PRECONDITIONER_AXIS,
    'residual_metric': RESIDUAL_METRIC, 'reference_population': 'positive quadrature weights times the fixed initial distribution; same total particle number',
    'linear_rtol': LINEAR_RTOL, 'linear_max_steps': LINEAR_MAX_STEPS, 'max_newton_steps': MAX_NEWTON_STEPS, 'relative_refinement_target': RELATIVE_TARGET,
    'spatial_discretization':'global Lagrange polynomial derivative on positive Gauss-Legendre quadrature',
    'velocity_discretization':'local quadratic u derivative; positive Gauss-Legendre u/mu quadrature',
    'initial': 'exp(-E-.2mu+.1sin(pi*y/.4)*u)', 'seed': None}
metadata = run_metadata(inputs, model='sm_local_nonlinear discrete-energy projector',
    boundary='natural no-flux collision-only Cartesian boxes')
metadata['limitations'] = ('Independent finite-time refinement checks at fixed other parameters; '
    'no combined-streaming box claim and no E/Gmu-only equilibrium reachability claim. '
    'Finite spatial polynomial spaces may lift continuum invariants: flux/potential moments are measured '
    'as scheme errors, never projected or interpreted as physical relaxation. Tail-domain and '
    'wider-domain quadrature checks are separate. A fine timestep-pair pass does not certify '
    'the coarse baseline or jointly refine space and time; all adjacent and coarsest/fine differences '
    'are retained. Fixed-reference and old-population residual norms are different; no equality '
    'of their nominal tolerances is claimed. Timings include concurrent unrelated machine load.')
metadata['relaxation_diagnostic']='Relative entropy to mass-normalized exp(-E-.2mu) decreases by the entropy gain because its logarithm is a discrete collision invariant. This stationary reference is not asserted reachable under the additional continuum constraints. Mean/variance of log(f/reference) and the sin(pi*y/.4)*u moment are measured separately.'
# Fifteen distinct cases; the base case is reused in each independent scan.
CASES = {'spatial3': {'nx': 3}, 'base': {}, 'spatial7': {'nx': 7},
    'u17': {'nu': 17}, 'u21': {'nu': 21}, 'mu13': {'nmu': 13}, 'mu17': {'nmu': 17},
    'dt_half': {'dt': DT_VALUES[1]}, 'dt_quarter': {'dt': DT_VALUES[2]},
    'dt_eighth': {'dt': DT_VALUES[3]}, 'dt_sixteenth': {'dt': DT_VALUES[4]},
    'u_tail': {'umax': 5.}, 'mu_tail': {'mumax': 24.},
    'u_wide': {'umax': 5., 'nu': 29}, 'mu_wide': {'mumax': 24., 'nmu': 25}}
SEQUENCES = {'spatial': ['spatial3', 'base', 'spatial7'],
    'parallel_velocity': ['u17', 'u21', 'base'], 'magnetic_moment': ['mu13', 'mu17', 'base'],
    'timestep': ['base', 'dt_half', 'dt_quarter', 'dt_eighth', 'dt_sixteenth'], 'parallel_tail': ['base', 'u_tail'],
    'moment_tail': ['base', 'mu_tail'], 'wider_parallel_quadrature': ['u_tail', 'u_wide'],
    'wider_moment_quadrature': ['mu_tail', 'mu_wide']}


def select_names(value, allowed):
    names = list(allowed) if value is None else value.split(',')
    if not names or len(set(names)) != len(names) or any(name not in allowed for name in names):
        raise ValueError(f'Execution selection must contain distinct names from {list(allowed)}')
    return names


def refinement_checks(rows):
    checks = []
    for field in FIELDS:
        available = {row['case']: row for row in rows if row['field'] == field.kind}
        check = {'field': field.kind}
        for name, sequence in SEQUENCES.items():
            check[name] = compare([available[key] for key in sequence]) if all(key in available for key in sequence) else {'status': 'not_run'}
        if 'base' in available:
            base = available['base']
            check['appreciable_relaxation'] = {
                'relative_entropy_decrease_fraction': base['relative_entropy_decrease_fraction'],
                'moment_decrease_fraction': base['relaxation_moment_decrease_fraction'],
                'status': 'passed' if base['relative_entropy_decrease_fraction'] > .1 and base['relaxation_moment_decrease_fraction'] > .05 else 'unresolved'}
        else:
            check['appreciable_relaxation'] = {'status': 'not_run'}
        check['status'] = 'passed' if all(check[name]['status'] == 'passed' for name in [*SEQUENCES, 'appreciable_relaxation']) else 'unresolved'
        checks.append(check)
    return checks


if os.environ.get('SM_EVOLUTION_RESUME'):
    raise ValueError('This lagged-mobility campaign cannot resume historical discrete-gradient rows; use their recorded source commit.')
field_names = select_names(os.environ.get('SM_EVOLUTION_FIELDS'), [field.kind for field in FIELDS])
case_names = select_names(os.environ.get('SM_EVOLUTION_CASES'), CASES)
metadata['execution'] = {'selected_fields': field_names, 'selected_cases': case_names,
    'description': 'Independent predefined cases; each completed row retains its producing run.'}
provenance_id = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
rows = [];checks = []


def save_evidence(path, evidence):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(evidence, indent=2, allow_nan=False)+'\n');temporary.replace(path)


def checkpoint():
    global checks
    checks = refinement_checks(rows)
    evidence = {'metadata': metadata, 'provenance_runs': {provenance_id: metadata},
        'rows': rows, 'checks': checks,
        'completed_cases_by_field': {field.kind: sum(row['field'] == field.kind for row in rows) for field in FIELDS},
        'status': 'passed' if all(check['status'] == 'passed' for check in checks) else 'unresolved',
        'selected_execution_complete': len(rows) == len(field_names)*len(case_names)}
    save_evidence(OUTPUT/'summary.json', evidence)


print(f'Execution fields={field_names}, cases={case_names}; no historical method mixing.', flush=True)
checkpoint()
with progress('Evolve positive nonuniform boxes and compare independent refinements'):
    for field in FIELDS:
        if field.kind not in field_names:
            continue
        for name in case_names:
            print(f'  Starting {field.kind}, {name}: {CASES[name]}', flush=True)
            for stale in ('last_state.npz', 'last_state.npz.tmp', 'last_correction.json'):
                (OUTPUT/stale).unlink(missing_ok=True)
            save_evidence(OUTPUT/'partial_case.json', {'status': 'incomplete',
                'provenance_id': provenance_id, 'field': field.kind, 'case': name, 'history': []})
            try:
                row = evaluate(field, **CASES[name], case_name=name)
            except Exception as error:
                partial = json.loads((OUTPUT/'partial_case.json').read_text())
                partial.update(status='failed', error_type=type(error).__name__, error=str(error))
                save_evidence(OUTPUT/'partial_case.json', partial)
                raise
            row.update({'case': name, 'provenance_id': provenance_id})
            rows.append(row);checkpoint()
            (OUTPUT/'partial_case.json').unlink(missing_ok=True)
        print(f'  {field.kind}: all selected solves completed; full checks remain explicit.', flush=True)


def plot_evolution_evidence(rows,checks,output,final_time,planned_cases=None):
    fields=['mirror','dipole','nonaxisymmetric']
    colors={'mirror':'#2166ac','dipole':'#d95f02','nonaxisymmetric':'#27823b'}
    keys=['spatial','parallel_velocity','magnetic_moment','timestep','parallel_tail',
          'moment_tail','wider_parallel_quadrature','wider_moment_quadrature']
    labels=['Space',r'$u$',r'$\mu$',r'$\Delta t$',r'$u$ tail',r'$\mu$ tail',r'Wide $u$',r'Wide $\mu$']
    invariant_names=['number_error','energy_error','marginal_error']
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for kind in fields:
        actual=[row for row in rows if row['field']==kind]
        if not actual:
            continue
        check=next((item for item in checks if item['field']==kind),{})
        parameters=check.get('spatial',{}).get('last_parameters',actual[0])
        spatial=sorted([row for row in actual if all(row[key]==parameters[key]
            for key in ('nu','nmu','umax','mumax','dt'))],key=lambda row:row['nx'])
        color=colors[kind]
        axes[0,0].plot([row['nx'] for row in spatial],
            [100*row['relative_entropy_decrease_fraction'] for row in spatial],
            'o-',color=color,label=kind)
        available=[index for index,key in enumerate(keys) if key in check and 'relative_changes' in check[key]]
        if available:
            axes[0,1].plot(available,[100*max(check[keys[index]]['relative_changes'].values())
                for index in available],'o-',color=color,label=kind)
        axes[1,0].plot([row['nx'] for row in spatial],
            [max(max(abs(moment['relative_signed_drift']) for moment in
                row['continuum_invariant_errors'].values()),1e-18) for row in spatial],
            'o-',color=color,label=kind)
        axes[1,1].plot(range(3),[max(max(step[name] for row in actual for step in row['history']),1e-18)
            for name in invariant_names],'o-',color=color,label=kind)
    axes[0,0].set(xlabel='Nodes per spatial axis',ylabel='Relative entropy decrease (%)',
                  title='Appreciable measured relaxation')
    axes[0,0].legend(frameon=False)
    if not any('relative_changes' in check.get(key,{}) for check in checks for key in keys):
        axes[0,1].text(.5,.5,'Refinement comparisons not yet complete',
            transform=axes[0,1].transAxes,ha='center',va='center',fontsize=10,color='#555555')
    axes[0,1].set(yscale='log',ylabel='Maximum selected observable change (%)',
                  title='Eight independent refinements')
    axes[0,1].set_xticks(range(8),labels,rotation=30,ha='right')
    axes[0,1].axhline(1.,color='#333333',linestyle='--',linewidth=1,label='1% target')
    axes[0,1].annotate('1% target',(7,1.),xytext=(-2,5),textcoords='offset points',ha='right',fontsize=9)
    axes[1,0].set(yscale='log',xlabel='Nodes per spatial axis',
                  ylabel='Largest additional moment relative drift',
                  title='Continuum constraints: endpoint scheme error')
    axes[1,1].set(yscale='log',ylabel='Largest cumulative relative error',
                  title='Discrete conservation: all completed steps')
    axes[1,1].set_xticks(range(3),['Particle number','Energy',r'Full $\mu$ marginal'])
    axes[1,1].axhline(1e-9,color='#333333',linestyle='--',linewidth=1)
    axes[1,1].annotate('1e-9 check',(2,1e-9),xytext=(-2,-13),textcoords='offset points',ha='right',fontsize=9)
    for axis in axes.ravel():
        axis.grid(alpha=.2)
    completion='' if planned_cases is None else f'; {len(rows)}/{planned_cases} cases complete'
    fig.suptitle(f'Nonlinear collision-only boxes: T={final_time:g} (normalized){completion}',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.96))
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fig.savefig(output/'evolution.png',dpi=180);plt.close(fig)


plot_evolution_evidence(rows, checks, OUTPUT, FINAL_TIME, len(FIELDS)*len(CASES))
print(f'Saved {OUTPUT/"summary.json"}; each independent convergence status is explicit.', flush=True)

for check in checks:
    if check['field'] in field_names:
        print(f"{check['field']} refinement status: " + json.dumps({key:value['status']
            for key,value in check.items() if isinstance(value,dict) and 'status' in value}), flush=True)
if (set(field_names)=={field.kind for field in FIELDS} and set(case_names)==set(CASES)
    and any(check['status']!='passed' for check in checks)):
    raise RuntimeError('Full campaign completed but convergence targets remain unresolved; evidence saved')
