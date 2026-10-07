"""Finite-time nonlinear collision evolution and independent refinement checks."""
import json
import os
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
from sato_morrison.solver import (discrete_gradient_compiler, discrete_gradient_step,
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
DT_VALUES = [.01, .005, .0025]
CHUNK, NEWTON_RTOL = 1048576, 1e-12
RELATIVE_TARGET = .01
OUTPUT = Path(__file__).resolve().parents[1] / 'results' / 'nonuniform_evolution'
OUTPUT.mkdir(parents=True, exist_ok=True)
print(f'Nonuniform positive nonlinear collision evolution; D={D}, T={FINAL_TIME}; '
      f'natural collision boundaries; entropy-metric Newton-GMRES; output={OUTPUT}', flush=True)


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


def evaluate(field, nx=NX, nu=NU, nmu=NMU, umax=U_MAX, mumax=MU_MAX, dt=DT):
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
    number = float(jnp.sum(grid.weights*initial))
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
    compiler = discrete_gradient_compiler(grid, method='krylov', collision_strength=D, chunk_size=CHUNK)
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
        answer = discrete_gradient_step(grid, state, dt, collision_strength=D, method='krylov',
            compiled_residual=compiler, chunk_size=CHUNK, rtol=NEWTON_RTOL)
        state = answer.f
        check = invariant_diagnostics(grid, state, initial)
        if max(check[k] for k in ('number_error', 'energy_error', 'marginal_error')) > 1e-9:
            raise RuntimeError(f'accumulated invariant error failed: {check}')
        history.append({'time': (index+1)*dt, 'wall_s': perf_counter()-start,
            'newton_iterations': answer.iterations, 'gmres_iterations': answer.linear_iterations,
            'nonlinear_relative_residual': float(answer.relative_residual),
            'maximum_linear_relative_residual': answer.linear_relative_residual,
            'entropy_change': answer.entropy_change, 'entropy_identity_error': answer.entropy_identity_error,
            **check})
        print(f'  {field.kind} {nx}^3 x {nu} x {nmu}, U={umax:g}, M={mumax:g}, dt={dt:g}: '
              f't={history[-1]["time"]:.3f}, Newton={answer.iterations}, GMRES={answer.linear_iterations}, '
              f'dS={answer.entropy_change:.3e}, {history[-1]["wall_s"]:.2f}s', flush=True)
    final_flux = apply(state, jnp.log(state));final_flux.block_until_ready()
    final_production = float(jnp.vdot(jnp.log(state), final_flux))
    gain = float(entropy(state, grid.weights))-initial_entropy
    if initial_production <= 0 or final_production < -1e-13 or gain <= 0:
        raise RuntimeError('entropy production/gain failed')
    continuous = {name: {'relative_signed_drift': (float(jnp.vdot(grid.weights*state, value))-moments0[name])/scales[name],
        'initial_relative_rate': rates0[name], 'final_relative_rate': -float(jnp.vdot(value, final_flux))/scales[name]}
        for name, value in diagnostics.items()}
    stored = [grid.coefficients, grid.energy_flow, grid.weights, grid.energy, grid.mu_index,
              *grid.local_quadrature, *[matrix for _, matrix in grid.derivatives]]
    row = {'field': field.kind, 'nx': nx, 'nu': nu, 'nmu': nmu, 'umax': umax, 'mumax': mumax, 'dt': dt,
        'nodes': grid.size, 'conceptual_unordered_pairs': grid.pair_count,
        'stored_grid_bytes': sum(array.nbytes for array in stored),
        'pair_workspace_budget': CHUNK, 'minimum_discrete_grad_B': minimum_grad_b,
        'minimum_B':float(strength.min()),'maximum_B':float(strength.max()),
        'u_quadratic_derivative_error': reproduction, 'number': number,
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
    jax.clear_caches()
    return row


def compare(sequence):
    previous, last = sequence[-2:]
    observable_names = ['entropy_gain_per_particle', 'final_production_per_particle',
                        'relaxation_moment_change_per_particle']
    changes = {name: abs(last[name]-previous[name])/max(abs(last[name]), 1e-30)
               for name in observable_names}
    return {'relative_changes': changes, 'target': RELATIVE_TARGET,
        'status': 'passed' if max(changes.values()) < RELATIVE_TARGET else 'unresolved',
        'previous_parameters': {k: previous[k] for k in ('nx', 'nu', 'nmu', 'umax', 'mumax', 'dt')},
        'last_parameters': {k: last[k] for k in ('nx', 'nu', 'nmu', 'umax', 'mumax', 'dt')}}


def dense_reference_check():
    """Same nonlinear endpoint, independent pair storage and Newton routes."""
    field=FIELDS[0]
    spatial=[gauss_interval(3,a,b) for a,b in BOUNDS]
    axes=[item[0] for item in spatial]
    u,wu=gauss_interval(3,-2,2);mu,wm=gauss_interval(2,0,2)
    grids={method:cartesian_grid(*axes,u,mu,field,velocity_weights=(wu,wm),
        spatial_weights=[item[1] for item in spatial],spatial_discretization='polynomial',
        compact=method=='krylov') for method in ('dense','krylov')}
    x,y,z,uu,mm=np.meshgrid(*axes,u,mu,indexing='ij')
    initial=jnp.exp(-grids['dense'].energy-.2*mm.ravel()+.1*jnp.sin(np.pi*y.ravel()/.4)*uu.ravel())
    compilers={method:discrete_gradient_compiler(grid,method=method,collision_strength=D,chunk_size=64)
               for method,grid in grids.items()}
    endpoints={};timings={};compile_timings={};errors={}
    for method,grid in grids.items():
        samples=[]
        for repeat in range(4):
            state=initial;start=perf_counter()
            for _ in range(3):
                state=discrete_gradient_step(grid,state,FINAL_TIME/3,method=method,collision_strength=D,
                    compiled_residual=compilers[method],chunk_size=64,rtol=1e-12).f
            state.block_until_ready()
            if repeat>0:
                samples.append(perf_counter()-start)
            else:
                compile_timings[method]=perf_counter()-start
        endpoints[method]=state;timings[method]=samples
        errors[method]=invariant_diagnostics(grid,state,initial)
    mass=grids['dense'].weights*initial
    matched=float(jnp.sqrt(jnp.sum(mass*((endpoints['krylov']-endpoints['dense'])/initial)**2)/jnp.sum(mass)))
    if matched>1e-10 or max(errors[method][name] for method in errors
        for name in ('number_error','energy_error','marginal_error'))>1e-9:
        raise RuntimeError('tiny independent dense matched endpoint failed')
    jax.clear_caches()
    return {'nodes':grids['dense'].size,'dt':FINAL_TIME/3,'steps':3,'matched_relative_entropy_metric_error':matched,
            'compile_and_first_three_step_wall_s':compile_timings,
            'warm_three_step_wall_s':timings,'invariants':errors,'status':'passed',
            'description':'Explicit unordered pairs+denseNewton versus compact ordered targets+prepared Newton-GMRES; same initial state and discrete equation; compilation excluded after one warm traversal.'}


inputs = {'fields': [field.__dict__ for field in FIELDS], 'bounds': BOUNDS,
    'nx': NX, 'nu': NU, 'nmu': NMU, 'x_orders': X_ORDERS, 'u_orders': U_ORDERS, 'mu_orders': MU_ORDERS,
    'u_domain': [-U_MAX, U_MAX], 'mu_domain': [0, MU_MAX], 'collision_strength': D,
    'final_time': FINAL_TIME, 'dt_values': DT_VALUES, 'newton_rtol': NEWTON_RTOL,
    'pair_chunk': CHUNK, 'relative_refinement_target': RELATIVE_TARGET,
    'spatial_discretization':'global Lagrange polynomial derivative on positive Gauss-Legendre quadrature',
    'velocity_discretization':'local quadratic u derivative; positive Gauss-Legendre u/mu quadrature',
    'initial': 'exp(-E-.2mu+.1sin(pi*y/.4)*u)', 'seed': None}
metadata = run_metadata(inputs, model='sm_local_nonlinear discrete-energy projector',
    boundary='natural no-flux collision-only Cartesian boxes')
metadata['limitations'] = ('Independent finite-time refinement checks at fixed other parameters; '
    'no combined-streaming box claim and no E/Gmu-only equilibrium reachability claim. '
    'Finite spatial polynomial spaces may lift continuum invariants: flux/potential moments are measured '
    'as scheme errors, never projected or interpreted as physical relaxation. Tail-domain and '
    'wider-domain quadrature checks are separate. Timings include concurrent unrelated machine load.')
metadata['relaxation_diagnostic']='Relative entropy to mass-normalized exp(-E-.2mu) decreases by the entropy gain because its logarithm is a discrete collision invariant. This stationary reference is not asserted reachable under the additional continuum constraints. Mean/variance of log(f/reference) and the sin(pi*y/.4)*u moment are measured separately.'
rows = [];checks = [];reference_check=None


def checkpoint():
    (OUTPUT/'summary.json').write_text(json.dumps({'metadata': metadata, 'dense_reference':reference_check,'rows': rows, 'checks': checks,
        'status': 'passed' if len(checks)==len(FIELDS) and all(check['status']=='passed' for check in checks) else 'unresolved'},
        indent=2)+'\n')


with progress('Evolve nonuniform boxes and compare independent finite-time refinements'):
    reference_check=dense_reference_check();checkpoint()
    for field in FIELDS:
        cache = {}
        def get(**parameters):
            settings = {'nx': NX, 'nu': NU, 'nmu': NMU, 'umax': U_MAX, 'mumax': MU_MAX, 'dt': DT, **parameters}
            key = tuple(settings.values())
            if key not in cache:
                cache[key] = evaluate(field, **settings);rows.append(cache[key]);checkpoint()
            return cache[key]
        sequences = {'spatial': [get(nx=n) for n in X_ORDERS],
            'parallel_velocity': [get(nu=n) for n in U_ORDERS],
            'magnetic_moment': [get(nmu=n) for n in MU_ORDERS],
            'timestep': [get(dt=dt) for dt in DT_VALUES],
            'parallel_tail': [get(), get(umax=5.)],
            'moment_tail': [get(), get(mumax=24.)],
            'wider_parallel_quadrature': [get(umax=5.), get(umax=5., nu=29)],
            'wider_moment_quadrature': [get(mumax=24.), get(mumax=24., nmu=25)]}
        check = {'field': field.kind, **{name: compare(sequence) for name, sequence in sequences.items()}}
        base=get()
        check['appreciable_relaxation']={'relative_entropy_decrease_fraction':base['relative_entropy_decrease_fraction'],
            'moment_decrease_fraction':base['relaxation_moment_decrease_fraction'],
            'status':'passed' if base['relative_entropy_decrease_fraction']>.1 and base['relaxation_moment_decrease_fraction']>.05 else 'unresolved'}
        check['status'] = 'passed' if all(check[name]['status']=='passed' for name in
            [*sequences,'appreciable_relaxation']) else 'unresolved'
        checks.append(check);checkpoint()
        print(f'  {field.kind} refinement status: {check["status"]}; '
              f'{[(name, check[name]["relative_changes"]) for name in sequences]}', flush=True)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for field in FIELDS:
    selected = sorted([row for row in rows if row['field']==field.kind and row['nu']==NU and row['nmu']==NMU
        and row['umax']==U_MAX and row['mumax']==MU_MAX and row['dt']==DT], key=lambda row: row['nx'])
    axes[0].plot([row['nx'] for row in selected], [row['entropy_gain_per_particle'] for row in selected], 'o-', label=field.kind)
    for row in selected:
        name = 'psi' if field.kind!='nonaxisymmetric' else 'chi'
        axes[1].scatter(row['nx'], max(abs(row['continuum_invariant_errors'][name]['relative_signed_drift']), 1e-18))
    axes[1].plot([row['nx'] for row in selected], [max(abs(row['continuum_invariant_errors'][
        'psi' if field.kind!='nonaxisymmetric' else 'chi']['relative_signed_drift']), 1e-18) for row in selected], label=field.kind)
axes[0].set(xlabel='Nodes per spatial axis', ylabel='Finite-time entropy gain per particle')
axes[1].set(xlabel='Nodes per spatial axis', ylabel='Continuous flux/potential relative drift', yscale='log')
axes[0].legend();axes[1].legend();fig.suptitle(f'Collision-only boxes: T={FINAL_TIME}; scheme errors measured separately')
fig.tight_layout();fig.savefig(OUTPUT/'evolution.png', dpi=180);plt.close(fig)
print(f'Saved {OUTPUT/"summary.json"}; every independent convergence status remains explicit.', flush=True)
