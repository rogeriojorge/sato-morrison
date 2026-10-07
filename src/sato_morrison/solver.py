"""Checked implicit weak collision steps and tangent toroidal transport."""
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
from solvax import pcg_linear_solve
from .collisions import mobility_action


class StepFailure(RuntimeError):
    """A rejected step; the caller retains the previous state."""


def linear_step(mass, stiffness_action, state, dt, *, rtol=1e-11, max_steps=500,
                preconditioner=None):
    """JIT/AD-compatible SOLVAX solve (M+dt K)h+=M h.

    Returns PCGSolution diagnostics. Call checked_linear_step on the host when
    failure must raise. Never use M^-1 K directly as a Euclidean PCG operator.
    """
    mass, state = jnp.asarray(mass), jnp.asarray(state)
    if preconditioner is None:
        preconditioner = lambda value: value / mass
    operator = lambda value: mass * value + dt * stiffness_action(value)
    return pcg_linear_solve(operator, mass * state, x0=state,
                            precond=preconditioner, rtol=rtol, max_steps=max_steps)


def checked_linear_step(mass, stiffness_action, state, dt, **options):
    if not np.isfinite(dt) or dt < 0 or not np.all(np.isfinite(mass)) or np.any(np.asarray(mass) <= 0):
        raise ValueError('positive mass and nonnegative timestep required')
    solution = linear_step(mass, stiffness_action, state, dt, **options)
    solution.x.block_until_ready()
    residual = jnp.linalg.norm(mass*solution.x + dt*stiffness_action(solution.x) - mass*state)
    scale = jnp.maximum(jnp.linalg.norm(mass*state), 1e-30)
    if not bool(solution.converged) or not np.isfinite(float(residual)) or float(residual/scale) > 5*options.get('rtol', 1e-11):
        raise StepFailure(f'SOLVAX PCG rejected step: status={int(solution.status)}, '
                          f'iterations={int(solution.iterations)}, true relative residual={float(residual/scale):.3e}')
    return solution


def entropy(f, weights):
    return -jnp.sum(weights * f * jnp.log(f))


def entropy_discrete_gradient(log_old, log_new):
    """Stable exact component divided difference of -n log(n/w)."""
    difference = log_new - log_old
    small = jnp.abs(difference) < 1e-4
    safe = jnp.where(small, 1., difference)
    ratio = safe / (-jnp.expm1(-safe))
    series = 1 + difference/2 + difference**2/12 - difference**4/720
    return -log_old - jnp.where(small, series, ratio)


@dataclass(frozen=True)
class NonlinearStep:
    f: object
    iterations: int
    relative_residual: float
    entropy_change: float
    entropy_identity_error: float
    minimum: float


def discrete_gradient_step(grid, f, dt, *, collision_strength=1., rtol=1e-11,
                           max_steps=30, compiled_residual=None):
    """Positive nodal fixed-field entropy step with an arithmetic mean mobility.

    Newton's unknown is log(f+). A line search accepts residual decrease; no
    clipping or population/energy correction is applied. Nonconvergence raises.
    This guarantees nodal positivity, not positivity of a high-order interpolant.
    """
    f = np.asarray(f, dtype=float).reshape(-1)
    if np.any(f <= 0) or not np.all(np.isfinite(f)) or not np.isfinite(dt) or dt < 0:
        raise ValueError('finite strictly positive distribution and nonnegative dt required')
    old = jnp.log(jnp.asarray(f))
    population = grid.weights * f
    scale = max(float(jnp.linalg.norm(population)), 1e-30)

    def residual(new_log, old_log, timestep):
        new, previous = jnp.exp(new_log), jnp.exp(old_log)
        mean = (new + previous)/2
        gradient = entropy_discrete_gradient(old_log, new_log)
        return grid.weights*(new-previous) - timestep*mobility_action(
            grid, mean, gradient, collision_strength=collision_strength)

    if compiled_residual is None:
        compiled_residual = (jax.jit(residual), jax.jit(jax.jacfwd(residual, argnums=0)))
    evaluate, jacobian = compiled_residual
    new_log = np.array(old)
    norm = np.inf
    for iteration in range(max_steps+1):
        value = np.asarray(evaluate(new_log, old, dt))
        norm = np.linalg.norm(value)/scale
        if np.isfinite(norm) and norm <= rtol:
            break
        if iteration == max_steps or not np.isfinite(norm):
            raise StepFailure(f'discrete-gradient Newton rejected step: iteration={iteration}, residual={norm:.3e}')
        matrix = np.asarray(jacobian(new_log, old, dt))
        try:
            direction = np.linalg.solve(matrix, -value)
        except np.linalg.LinAlgError as error:
            raise StepFailure('singular discrete-gradient Newton system') from error
        accepted = False
        for backtrack in range(24):
            candidate = new_log + direction*2.**(-backtrack)
            candidate_norm = np.linalg.norm(np.asarray(evaluate(candidate, old, dt)))/scale
            if np.isfinite(candidate_norm) and candidate_norm < norm:
                new_log, accepted = candidate, True
                break
        if not accepted:
            raise StepFailure(f'discrete-gradient line search rejected step: residual={norm:.3e}')
    new = jnp.exp(jnp.asarray(new_log))
    gradient = entropy_discrete_gradient(old, jnp.asarray(new_log))
    change = float(entropy(new, grid.weights)-entropy(jnp.asarray(f), grid.weights))
    production = float(dt*jnp.vdot(gradient, mobility_action(grid, (new+f)/2, gradient,
                                 collision_strength=collision_strength)))
    identity_error = change-production
    bound = float(jnp.linalg.norm(gradient))*norm*scale + 100*np.finfo(float).eps*max(abs(change), 1.)
    if change < -bound or abs(identity_error) > 2*bound or not np.all(np.isfinite(new)) or np.any(np.asarray(new)<=0):
        raise StepFailure(f'entropy/finite check rejected step: deltaS={change:.3e}, identity residual={identity_error:.3e}')
    return NonlinearStep(new, iteration, norm, change, identity_error, float(new.min()))


def discrete_gradient_compiler(grid, *, collision_strength=1.):
    """Reusable residual and Jacobian compiled once across host Newton steps."""
    def residual(new_log, old_log, timestep):
        new, old = jnp.exp(new_log), jnp.exp(old_log)
        return grid.weights*(new-old) - timestep*mobility_action(grid, (new+old)/2,
            entropy_discrete_gradient(old_log, new_log), collision_strength=collision_strength)
    return jax.jit(residual), jax.jit(jax.jacfwd(residual, argnums=0))


def toroidal_stream(state, shape, radius, u, mu, dt, *, strength=1., mass=1., charge=1.,
                    theta_period=2*np.pi, z_period=2*np.pi):
    """Exact Fourier translation for vacuum toroidal Hamiltonian flow.

    theta_dot=u/R; z_dot=(m u²+mu B)/(q C). Both preserve R,u,mu;
    radial walls are tangent and theta,z periodic. Odd grids avoid the real
    Nyquist ambiguity. For finite amplitude callers must check reconstruction.
    """
    if shape[1] % 2 == 0 or shape[2] % 2 == 0:
        raise ValueError('use odd theta and z resolutions for exact real spectral transport')
    rr = jnp.asarray(radius)[:,None,None,None,None]
    uu = jnp.asarray(u)[None,None,None,:,None]
    mm = jnp.asarray(mu)[None,None,None,None,:]
    theta_speed = uu/rr
    z_speed = (mass*uu**2 + mm*strength/rr)/(charge*strength)
    kt = 2*jnp.pi*jnp.fft.fftfreq(shape[1], d=theta_period/shape[1])[None,:,None,None,None]
    kz = 2*jnp.pi*jnp.fft.fftfreq(shape[2], d=z_period/shape[2])[None,None,:,None,None]
    transformed = jnp.fft.fftn(jnp.asarray(state).reshape(shape), axes=(1,2))
    shifted = transformed*jnp.exp(-1j*dt*(kt*theta_speed+kz*z_speed))
    return jnp.fft.ifftn(shifted, axes=(1,2)).real.reshape(-1)


def invariant_diagnostics(grid, f, initial):
    f, initial = jnp.asarray(f).reshape(-1), jnp.asarray(initial).reshape(-1)
    delta = grid.weights*(f-initial)
    number_scale = jnp.sum(grid.weights*initial)
    energy_scale = jnp.sum(grid.weights*initial*jnp.abs(grid.energy))
    marginal = jnp.zeros(grid.shape[-1]).at[grid.mu_index].add(delta)
    return {'number_error': float(jnp.abs(jnp.sum(delta))/number_scale),
            'energy_error': float(jnp.abs(jnp.vdot(grid.energy, delta))/energy_scale),
            'marginal_error': float(jnp.max(jnp.abs(marginal))/number_scale),
            'min_f': float(f.min())}
