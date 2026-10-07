"""Local continuous weak-form production and a checked geometry derivative."""
import json
from pathlib import Path
from time import perf_counter
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sato_morrison.geometry import Field, common_chart_action, energy_mu, field_data
from sato_morrison.reference import gauss_interval, progress, run_metadata

# Translate the local spatial sample in x, holding field parameters, normalized
# units, u/mu integration domains and the prescribed distribution function fixed.
# This is an initial weak-form observable, not a differentiated kinetic solve.
FIELDS = [Field('mirror', amplitude=.15), Field('dipole'),
          Field('nonaxisymmetric', amplitude=.03)]
POSITION = np.array([1., .12, .3])
MASS, CHARGE, D = 1.3, -.8, .1
NU, NMU, UMAX, MUMAX = 25, 21, 4., 20.
STEPS = np.array([.04, .02, .01, .005, .002, .001, .0005, .0002, .0001, .00005,
                  .00001, .000005, .000001, .0000002, .00000005, .00000001])
OUTPUT = Path(__file__).resolve().parents[1] / 'results' / 'geometry_sensitivity'
OUTPUT.mkdir(parents=True, exist_ok=True)
print('Continuous local entropy production and d/dx at fixed physical u, mu; '
      f'x={POSITION}, m={MASS}, q={CHARGE}, D={D}. Compiling float64 values and derivatives.',
      flush=True)


def make_observable(field, nu=NU, nmu=NMU, umax=UMAX, mumax=MUMAX):
    u, wu = gauss_interval(nu, -umax, umax)
    mu, wmu = gauss_interval(nmu, 0., mumax)
    velocity = jnp.asarray(np.stack(np.meshgrid(u, mu, indexing='ij'), axis=-1).reshape(-1, 2))
    weights = jnp.asarray(np.outer(wu, wmu).ravel())
    left, right = np.triu_indices(nu*nmu, 1)

    def log_density(state):
        return (-energy_mu(state, field, MASS, CHARGE) - .2*state[4]
                + .1*jnp.sin(state[0])*state[3] + .04*state[1]*state[4])

    def observable(displacement):
        position = jnp.asarray(POSITION).at[0].add(displacement)
        states = jnp.concatenate((jnp.broadcast_to(position, (len(velocity), 3)), velocity), axis=1)
        action = jax.vmap(lambda state: common_chart_action(
            state, jax.grad(log_density)(state), field, MASS, CHARGE))(states)
        flow = jax.vmap(lambda state: common_chart_action(state, jax.grad(
            lambda point: energy_mu(point, field, MASS, CHARGE))(state), field, MASS, CHARGE))(states)
        xi = flow[left]-flow[right]
        direction = xi/jnp.linalg.norm(xi, axis=1)[:, None]
        delta = action[left]-action[right]
        projected = delta-direction*jnp.sum(direction*delta, axis=1)[:, None]
        _, bmag, _, _, _ = field_data(position, field)
        population = jnp.exp(jax.vmap(log_density)(states))
        # Unordered pairs absorb the 1/2 in the ordered weak entropy form.
        measure = bmag**2*weights[left]*weights[right]*population[left]*population[right]
        return D*jnp.sum(measure*jnp.sum(projected[:, :3]**2, axis=1))

    return jax.jit(observable), jax.jit(jax.value_and_grad(observable))


def independent_value(field, displacement):
    """NumPy pair form with analytic vacuum energy flow and analytic log gradient.

    Field derivatives are shared geometry data; pair projection and covector
    actions are reconstructed here without common_chart_action or JAX AD of f.
    """
    position = POSITION + np.array([displacement, 0., 0.])
    _, bmag, b, gradb, _ = map(np.asarray, field_data(jnp.asarray(position), field))
    u, wu = gauss_interval(NU, -UMAX, UMAX)
    mu, wmu = gauss_interval(NMU, 0., MUMAX)
    uu, mm = map(np.ravel, np.meshgrid(u, mu, indexing='ij'))
    t = np.cross(b, gradb)
    parallel = b@gradb
    flow = np.column_stack((uu[:, None]*b+(MASS*uu**2+mm*bmag)[:, None]*t/(CHARGE*bmag**2),
                            -mm*parallel/MASS, mm*uu*parallel))
    # log f = -E-.2mu + p; energy and mu terms annihilate the pair projector.
    gx = np.column_stack((.1*np.cos(position[0])*uu, .04*mm, np.zeros_like(uu)))
    gu = .1*np.sin(position[0])
    bstar = bmag*b+MASS*uu[:, None]*t/(CHARGE*bmag)
    ax = np.cross(b, gx)/(CHARGE*bmag)+bstar*gu/(MASS*bmag)
    au = -np.sum(bstar*gx, axis=1)/(MASS*bmag)
    action = np.column_stack((ax, au, mm*(ax@gradb)))
    population = np.exp(-MASS*uu**2/2-mm*bmag-.2*mm+
                        .1*np.sin(position[0])*uu+.04*position[1]*mm)
    weights = np.outer(wu, wmu).ravel()*bmag
    left, right = np.triu_indices(len(uu), 1)
    xi = flow[left]-flow[right]
    delta = action[left]-action[right]
    projected = delta-xi*(np.sum(delta*xi, axis=1)/np.sum(xi*xi, axis=1))[:, None]
    return float(D*np.sum(weights[left]*weights[right]*population[left]*population[right]*
                         np.sum(projected[:, :3]**2, axis=1)))


rows = []
for field in FIELDS:
    with progress(f'{field.kind}: compile and check geometric sensitivity'):
        start = perf_counter()
        value, derivative = make_observable(field)
        production, gradient = map(float, derivative(0.))
        compile_s = perf_counter()-start
        fd = np.array([(independent_value(field, h)-independent_value(field, -h))/(2*h)
                       for h in STEPS])
        error = np.abs(fd-gradient)/abs(gradient)
        independent = independent_value(field, 0.)
        comparison = abs(independent-production)/production
        refinements = []
        for label, nu, nmu, umax, mumax in [('u17',17,NMU,UMAX,MUMAX),
                ('u21',21,NMU,UMAX,MUMAX), ('mu13',NU,13,UMAX,MUMAX),
                ('mu17',NU,17,UMAX,MUMAX), ('u_tail',29,NMU,5.,MUMAX),
                ('mu_tail',NU,25,UMAX,24.), ('u_tail_order',33,NMU,5.,MUMAX),
                ('mu_tail_order',NU,29,UMAX,24.)]:
            _, refined = make_observable(field, nu, nmu, umax, mumax)
            p, g = map(float, refined(0.))
            refinements.append({'case':label, 'nu':nu, 'nmu':nmu, 'umax':umax, 'mumax':mumax,
                                'production':p, 'derivative':g,
                                'production_change':abs(p-production)/production,
                                'derivative_change':abs(g-gradient)/abs(gradient)})
        warm = []
        for _ in range(5):
            start = perf_counter(); answer = derivative(0.); answer[1].block_until_ready()
            warm.append(perf_counter()-start)
        row = {'field':field.kind, 'production':production, 'derivative':gradient,
               'independent_pair_relative_error':comparison, 'fd_steps':STEPS.tolist(),
               'fd_derivatives':fd.tolist(), 'fd_relative_errors':error.tolist(),
               'refinements':refinements, 'compile_first_s':compile_s, 'warm_s':warm}
        rows.append(row)
        print(f'  P={production:.9g}; dP/dx={gradient:.9g}; independent={comparison:.2e}; '
              f'FD plateau={max(error[-3:]):.2e}', flush=True)
        if comparison > 1e-10 or max(error[-3:]) > 1e-6:
            raise RuntimeError(f'{field.kind}: independent value or derivative plateau failed')

metadata = run_metadata({'position':POSITION.tolist(), 'mass':MASS, 'charge':CHARGE,
    'D':D, 'nu':NU, 'nmu':NMU, 'umax':UMAX, 'mumax':MUMAX,
    'fields':[field.__dict__ for field in FIELDS], 'finite_difference_steps':STEPS.tolist(),
    'control':'translation of the sampled x coordinate; all other inputs held fixed'},
    model='sm_local_nonlinear continuous initial local weak entropy production',
    boundary='local volume density; truncated positive velocity quadrature; no evolution boundary')
metadata['rows'] = rows
metadata['scope'] = 'No implicit or steady-state geometry derivative, and no physical rate calibration.'
metadata['timing_scope'] = 'Compilation and synchronized warm calls under concurrent machine load; not isolated cost comparisons.'
checks = []
for row in rows:
    refinements = {r['case']:r for r in row['refinements']}
    changes = [refinements[name]['derivative_change'] for name in ['u21','mu17','u_tail','mu_tail']]
    changes += [abs(refinements[name+'_order']['derivative']-refinements[name]['derivative'])/
                abs(refinements[name+'_order']['derivative']) for name in ['u_tail','mu_tail']]
    checks.append({'field':row['field'], 'maximum_final_derivative_change':max(changes),
                   'target':.01, 'status':'passed' if max(changes)<.01 else 'unresolved'})
metadata['convergence_checks'] = checks
(OUTPUT/'summary.json').write_text(json.dumps(metadata, indent=2)+'\n')
fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
colors = ['#087e8b', '#c45b28', '#614e9e']
for row, color in zip(rows, colors):
    axes[0].loglog(STEPS, row['fd_relative_errors'], 'o-', color=color, label=row['field'])
    selected = [r for r in row['refinements'] if r['case'] in ['u21','mu17','u_tail','mu_tail']]
    axes[1].semilogy(range(4), [max(r['derivative_change'], 1e-16) for r in selected],
                     'o-', color=color, label=row['field'])
    axes[2].bar(row['field'], row['derivative']/row['production'], color=color)
axes[0].set(xlabel='Centered difference step', ylabel='Relative derivative error',
            title='A. Independent finite differences')
axes[0].legend(fontsize=8)
axes[1].axhline(.01, color='.45', linestyle='--', linewidth=1, label='1%')
axes[1].set(xticks=range(4), xticklabels=['u order','μ order','u tail','μ tail'],
            ylabel='Relative derivative change', title='B. Quadrature and tail checks')
axes[2].axhline(0, color='.5', linewidth=1)
axes[2].set(ylabel='∂ log P / ∂x', title='C. Geometry changes production')
axes[2].tick_params(axis='x', labelrotation=15)
for axis in axes[:2]:
    axis.grid(alpha=.2)
fig.savefig(OUTPUT/'sensitivity.png', dpi=180)
plt.close(fig)
print(f'Saved {OUTPUT}; geometry derivative checked for this local observable only.', flush=True)
if any(check['status'] != 'passed' for check in checks):
    raise RuntimeError('Geometry derivative quadrature/tail target unresolved; evidence saved.')
