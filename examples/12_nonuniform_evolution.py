"""Finite-time nonlinear collision evolution and independent refinement checks.

Execution only: SM_EVOLUTION_FIELDS is a comma-separated subset of mirror,
dipole,nonaxisymmetric (default all). SM_EVOLUTION_RESUME is a completed-row
summary.json from this canonical campaign. Neither changes scientific settings.
Resume requires the recorded Git commits to be available (fetch-depth: 0 in CI).
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
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


CANONICAL_COMMIT = 'fdc625d857cf69effe1debfdbd3b7d723cd17fad'
CANONICAL_INPUTS_SHA256 = '2513467a7332ac54752a4faeeccb2a8a281c2782ceaac29ec7f5cd7a93858428'
PARAMETER_NAMES = ('nx', 'nu', 'nmu', 'umax', 'mumax', 'dt')


def json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def selected_field_names(value):
    names = [field.kind for field in FIELDS] if not value else [name.strip() for name in value.split(',')]
    if not names or len(set(names)) != len(names) or set(names)-{field.kind for field in FIELDS}:
        raise ValueError('SM_EVOLUTION_FIELDS must contain distinct canonical field names')
    return names


def case_parameters():
    """The thirteen distinct cases already used by the eight canonical scans."""
    variations = ([{'nx': n} for n in X_ORDERS]+[{'nu': n} for n in U_ORDERS]
        +[{'nmu': n} for n in MU_ORDERS]+[{'dt': dt} for dt in DT_VALUES]
        +[{'umax': 5.}, {'mumax': 24.}, {'umax': 5., 'nu': 29}, {'mumax': 24., 'nmu': 25}])
    default = dict(zip(PARAMETER_NAMES, (NX, NU, NMU, U_MAX, MU_MAX, DT)))
    return {tuple({**default, **change}[name] for name in PARAMETER_NAMES) for change in variations}


def scientific_ast(source):
    """Permit driver changes, while freezing functions and scientific constants."""
    constants = {'FIELDS', 'BOUNDS', 'NX', 'NU', 'NMU', 'X_ORDERS', 'U_ORDERS', 'MU_ORDERS',
        'U_MAX', 'MU_MAX', 'D', 'FINAL_TIME', 'DT', 'DT_VALUES', 'CHUNK', 'NEWTON_RTOL',
        'RELATIVE_TARGET', 'inputs'}
    definitions = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            'continuum_moments', 'evaluate', 'compare', 'dense_reference_check'}:
            definitions.append(node)
        elif isinstance(node, ast.Assign):
            targets = [name for target in node.targets for name in
                (target.elts if isinstance(target, ast.Tuple) else [target])]
            if all(isinstance(name, ast.Name) and name.id in constants for name in targets):
                definitions.append(node)
    return ast.dump(ast.Module(body=definitions, type_ignores=[]), include_attributes=False)


def campaign_source_identity(root, commit, *, working=False):
    """Check physics against the anchor, and recover the recorded aggregate hash."""
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Resume needs an identifiable full Git commit')
    def git(*arguments):
        result = subprocess.run(['git', '-C', str(root), *arguments], capture_output=True)
        if result.returncode:
            raise ValueError(f'Cannot verify campaign Git source: {commit}; fetch its history')
        return result.stdout
    paths = git('ls-tree', '-r', '--name-only', commit).decode().splitlines()
    source_paths = sorted(path for path in paths if path == 'pyproject.toml'
        or path.startswith('src/') and path.endswith('.py')
        or path.startswith(('examples/', 'tests/')) and path.count('/') == 1 and path.endswith('.py'))
    physics_paths = [path for path in source_paths if path.startswith('src/') or path == 'pyproject.toml']
    anchor_paths = git('ls-tree', '-r', '--name-only', CANONICAL_COMMIT).decode().splitlines()
    if physics_paths != sorted(path for path in anchor_paths
        if path.startswith('src/') and path.endswith('.py') or path == 'pyproject.toml'):
        raise ValueError('Campaign physics source set differs from the canonical commit')
    hashes = {}
    digest = hashlib.sha256()
    for path in source_paths:
        recorded = git('show', f'{commit}:{path}')
        digest.update(path.encode());digest.update(recorded)
        if path in physics_paths:
            candidate = (root/path).read_bytes() if working else recorded
            if candidate != git('show', f'{CANONICAL_COMMIT}:{path}'):
                raise ValueError(f'Campaign physics differs from canonical source: {path}')
            hashes[path] = hashlib.sha256(candidate).hexdigest()
    example = 'examples/12_nonuniform_evolution.py'
    candidate = (root/example).read_text() if working else git('show', f'{commit}:{example}').decode()
    if scientific_ast(candidate) != scientific_ast(git('show', f'{CANONICAL_COMMIT}:{example}').decode()):
        raise ValueError('Campaign scientific functions or constants differ from the canonical source')
    return digest.hexdigest(), hashes


def require_finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f'Resume requires a finite numeric {label}')
    return value


def validate_completed_row(row):
    """Recheck measured completion and structural tolerances; ignore saved status."""
    if not isinstance(row, dict) or row.get('field') not in {field.kind for field in FIELDS}:
        raise ValueError('Invalid resume field row')
    try:
        key = tuple(require_finite(row[name], name) for name in PARAMETER_NAMES)
        if key not in case_parameters():
            raise ValueError('Resume row is not one of the thirteen canonical cases')
        for name in ('nodes', 'conceptual_unordered_pairs', 'stored_grid_bytes', 'pair_workspace_budget',
            'minimum_discrete_grad_B', 'minimum_B', 'maximum_B', 'number', 'initial_production_per_particle',
            'entropy_gain_per_particle', 'initial_relative_entropy_per_particle'):
            if require_finite(row[name], name) <= 0:
                raise ValueError(f'Resume row has nonpositive {name}')
        nx, nu, nmu, _, _, dt = key
        if row['nodes'] != nx**3*nu*nmu or row['conceptual_unordered_pairs'] != nx**3*(nu*nmu)*(nu*nmu-1)//2:
            raise ValueError('Resume row grid or pair count does not match its parameters')
        if row['pair_workspace_budget'] != CHUNK or row['minimum_B'] > row['maximum_B']:
            raise ValueError('Resume row grid diagnostics do not match the campaign')
        for name in ('setup_s', 'relative_entropy_decrease_fraction'):
            if require_finite(row[name], name) < 0:
                raise ValueError(f'Resume row has negative {name}')
        if require_finite(row['final_production_per_particle'], 'final production') < -1e-13/row['number']:
            raise ValueError('Resume row failed the original final-production tolerance')
        require_finite(row['relaxation_moment_decrease_fraction'], 'relaxation moment decrease')
        if abs(require_finite(row['u_quadratic_derivative_error'], 'u derivative error')) > 1e-10:
            raise ValueError('Resume row failed discrete energy-flow reproduction')
        for name in ('entropy_telescoping_error', 'relaxation_moment_change_per_particle'):
            require_finite(row[name], name)
        for name in ('initial_perturbation', 'final_perturbation'):
            require_finite(row[name]['mean'], name+' mean')
            if require_finite(row[name]['variance'], name+' variance') < 0:
                raise ValueError('Negative resume perturbation variance')
        expected = {'psi', 'psi_squared', 'radius_squared', 'z', 'z_squared', 'radius_squared_z'}
        if row['field'] == 'nonaxisymmetric':
            expected = {'B', 'chi', 'B_squared', 'chi_squared'}
        if set(row['continuum_invariant_errors']) != expected:
            raise ValueError('Resume row is missing continuum moment diagnostics')
        for moment in row['continuum_invariant_errors'].values():
            for name in ('relative_signed_drift', 'initial_relative_rate', 'final_relative_rate'):
                require_finite(moment[name], name)
        history = row['history']
        if not isinstance(history, list) or len(history) != int(round(FINAL_TIME/dt)):
            raise ValueError('Resume row has an incomplete time history')
        for index, step in enumerate(history):
            if abs(require_finite(step['time'], 'step time')-(index+1)*dt) > 1e-13:
                raise ValueError('Resume history does not reach the canonical time horizon')
            for name in ('number_error', 'energy_error', 'marginal_error'):
                if not 0 <= require_finite(step[name], name) <= 1e-9:
                    raise ValueError('Resume history failed an accumulated invariant tolerance')
            if not 0 <= require_finite(step['nonlinear_relative_residual'], 'Newton residual') <= NEWTON_RTOL:
                raise ValueError('Resume history failed the canonical Newton tolerance')
            # Per-Newton forcing is not retained. The original solver caps it
            # at .05 and checks its true residual against five times forcing.
            if not 0 <= require_finite(step['maximum_linear_relative_residual'], 'linear residual') <= .250000001:
                raise ValueError('Resume history failed the maximum Krylov forcing tolerance')
            if require_finite(step['min_f'], 'minimum population') <= 0:
                raise ValueError('Resume history is not nodally positive')
            # The gradient-dependent acceptance bound was checked by the
            # original solver, but its stages/bound are absent from the JSON.
            require_finite(step['entropy_change'], 'entropy change')
            require_finite(step['entropy_identity_error'], 'entropy identity error')
            for name in ('wall_s', 'newton_iterations', 'gmres_iterations'):
                if require_finite(step[name], name) < 0:
                    raise ValueError('Negative resume timing or iteration count')
        gain = row['entropy_gain_per_particle']*row['number']
        if abs(gain-sum(step['entropy_change'] for step in history)-row['entropy_telescoping_error']) > 1e-12*row['number']:
            raise ValueError('Resume entropy history does not match its endpoint')
        if abs(row['relative_entropy_decrease_fraction']-row['entropy_gain_per_particle']/row['initial_relative_entropy_per_particle']) > 1e-12:
            raise ValueError('Resume relative-entropy diagnostic is inconsistent')
    except (KeyError, TypeError) as error:
        raise ValueError('Resume row is missing required measured diagnostics') from error
    return {**row, 'status': 'passed'}


def load_completed_rows(path, root):
    """Accept original or resumed summaries, retaining every row's run context."""
    raw = Path(path).read_bytes()
    def invalid_constant(value):
        raise ValueError(f'Nonfinite JSON constant in resume: {value}')
    summary = json.loads(raw, parse_constant=invalid_constant)
    if not isinstance(summary, dict) or not isinstance(summary.get('rows'), list):
        raise ValueError('Resume requires a campaign summary with a rows list')
    runs = summary.get('provenance_runs')
    original = runs is None
    if original:
        context = summary.get('metadata')
        runs = {json_sha256(context): context}
    if not isinstance(runs, dict) or not runs:
        raise ValueError('Resume requires original per-run provenance')
    for identifier, context in runs.items():
        if not isinstance(context, dict) or identifier != json_sha256(context):
            raise ValueError('Resume provenance identifier does not match its metadata')
        if json_sha256(context.get('inputs')) != CANONICAL_INPUTS_SHA256:
            raise ValueError('Resume inputs differ from the full canonical configuration')
        digest, _ = campaign_source_identity(root, context.get('commit'))
        if context.get('source_sha256') != digest or context.get('source_changes') != '':
            raise ValueError('Resume metadata does not identify clean recorded source')
        if context.get('x64') is not True or context.get('units') != 'normalized' or context.get('model') != 'sm_local_nonlinear discrete-energy projector' or context.get('boundary') != 'natural no-flux collision-only Cartesian boxes':
            raise ValueError('Resume model, precision, units, or boundary differs from the campaign')
        for name in ('versions', 'logical_cpus', 'thread_environment', 'python', 'platform', 'processor', 'devices'):
            if name not in context:
                raise ValueError('Resume is missing original environment provenance')
    accepted = [];seen = set()
    for candidate in summary['rows']:
        row = validate_completed_row(candidate)
        identifier = next(iter(runs)) if original else row.get('provenance_id')
        if identifier not in runs:
            raise ValueError('Resume row has no verifiable original run context')
        row['provenance_id'] = identifier
        key = (row['field'], *[row[name] for name in PARAMETER_NAMES])
        if key in seen:
            raise ValueError('Duplicate completed case in resume')
        seen.add(key);accepted.append(row)
    # Dense control is cheap and is rerun in each new execution. Preserve any
    # old record verbatim in the resume segment, never use its saved status.
    return accepted, runs, {'path': str(Path(path).resolve()), 'sha256': hashlib.sha256(raw).hexdigest(),
        'original_dense_reference': summary.get('dense_reference'),
        'original_dense_reference_provenance_id': summary.get('dense_reference_provenance_id', next(iter(runs)) if original else None),
        'entropy_validation': 'Finite saved diagnostics and endpoint telescoping are rechecked. The original executed solver checked the gradient-dependent defect bound; saved summaries lack its stage states/bound, so resume cannot replay that check.'}


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
root = Path(__file__).resolve().parents[1]
names = selected_field_names(os.environ.get('SM_EVOLUTION_FIELDS'))
SELECTED_FIELDS = [field for field in FIELDS if field.kind in names]
rows = [];provenance_runs = {};resume_context = None
resume_path = os.environ.get('SM_EVOLUTION_RESUME')
physics_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted([*root.glob('src/**/*.py'), root/'pyproject.toml'])}
if resume_path:
    if json_sha256(inputs) != CANONICAL_INPUTS_SHA256:
        raise ValueError('Resume cannot change canonical scientific inputs')
    _, physics_hashes = campaign_source_identity(root, metadata['commit'], working=True)
    rows, provenance_runs, resume_context = load_completed_rows(resume_path, root)
metadata['execution'] = {'selected_fields': names, 'canonical_physics_commit': CANONICAL_COMMIT if resume_path else None,
    'physics_source_sha256': physics_hashes, 'resume': resume_context,
    'description': 'Completed rows retain their original provenance_id; new rows belong to this execution.'}
current_provenance = json_sha256(metadata)
provenance_runs[current_provenance] = metadata
checks = [];reference_check=None
print(f'Execution fields: {names}; accepted {len(rows)} completed canonical rows for resume.', flush=True)


def checkpoint():
    completed = {field.kind: sum(row['field']==field.kind for row in rows) for field in FIELDS}
    evidence = {'metadata': metadata, 'provenance_runs': provenance_runs,
        'dense_reference':reference_check,'dense_reference_provenance_id':current_provenance,
        'rows': rows, 'checks': checks, 'completed_cases_by_field': completed,
        'status': 'passed' if len(checks)==len(FIELDS) and all(check['status']=='passed' for check in checks) else 'unresolved',
        'selected_fields_status': 'passed' if len(checks)==len(SELECTED_FIELDS) and all(check['status']=='passed' for check in checks) else 'unresolved'}
    temporary = OUTPUT/'summary.json.tmp'
    temporary.write_text(json.dumps(evidence, indent=2)+'\n')
    temporary.replace(OUTPUT/'summary.json')


with progress('Evolve nonuniform boxes and compare independent finite-time refinements'):
    reference_check=dense_reference_check();checkpoint()
    for field in SELECTED_FIELDS:
        cache = {tuple(row[name] for name in PARAMETER_NAMES): row for row in rows if row['field']==field.kind}
        def get(**parameters):
            settings = {'nx': NX, 'nu': NU, 'nmu': NMU, 'umax': U_MAX, 'mumax': MU_MAX, 'dt': DT, **parameters}
            key = tuple(settings.values())
            if key not in cache:
                print(f'  Starting canonical case {field.kind}: {settings}', flush=True)
                cache[key] = evaluate(field, **settings)
                cache[key]['provenance_id'] = current_provenance
                rows.append(cache[key]);checkpoint()
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

def plot_evolution_evidence(rows,checks,output,final_time):
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
    fig.suptitle(f'Nonlinear collision-only field boxes: T={final_time:g} (normalized)',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.96))
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fig.savefig(output/'evolution.png',dpi=180);plt.close(fig)


plot_evolution_evidence(rows, checks, OUTPUT, FINAL_TIME)
print(f'Saved {OUTPUT/"summary.json"}; every independent convergence status remains explicit.', flush=True)
