"""Checked implicit weak collision steps and tangent toroidal transport."""
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
from solvax import pcg_linear_solve, gmres
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
    linear_iterations: int = 0
    linear_relative_residual: float = 0.
    residual_metric: str = "population"


def _discrete_gradient_residual(grid,collision_strength,chunk_size):
    def residual(new_log,old_log,timestep):
        new,old=jnp.exp(new_log),jnp.exp(old_log)
        return grid.weights*(new-old)-timestep*mobility_action(grid,(new+old)/2,
            entropy_discrete_gradient(old_log,new_log),collision_strength=collision_strength,
            chunk_size=chunk_size)
    return residual


def discrete_gradient_compiler(grid, *, collision_strength=1., method='dense',
                               chunk_size=None,linear_restart=30,linear_max_restarts=10):
    """Reusable DG residual and either a dense Jacobian or Newton-GMRES solve.

    Matrix-free GMRES uses exact JAX Jacobian-vector products. Rows and log increments
    are scaled by the square root of the previous nodal populations, giving
    an entropy-metric Jacobian without forming a dense matrix. The two routes
    solve the same unscaled discrete-gradient equation.
    """
    if method not in ('dense','krylov'):
        raise ValueError(f'unsupported discrete-gradient method: {method}')
    if linear_restart<1 or linear_max_restarts<1:
        raise ValueError('positive Krylov iteration limits required')
    if method=='krylov' and chunk_size is None:
        chunk_size=65536
    residual=_discrete_gradient_residual(grid,collision_strength,chunk_size)
    evaluate=jax.jit(residual)
    if method=='dense':
        return evaluate,jax.jit(jax.jacfwd(residual,argnums=0))
    def correction(new_log,old_log,timestep,linear_rtol):
        mass_root=jnp.sqrt(grid.weights*jnp.exp(old_log))
        scaled=lambda value:residual(value,old_log,timestep)/mass_root
        value=scaled(new_log)
        # Recompute forward-mode products so linearization cannot retain all
        # primal pair blocks as a quadratic tape between GMRES iterations.
        operator=lambda vector:jax.jvp(scaled,(new_log,),(vector/mass_root,))[1]
        answer=gmres(operator,-value,restart=linear_restart,rtol=linear_rtol,
                     atol=0.,max_restarts=linear_max_restarts)
        relative=answer.residual_norm/jnp.maximum(jnp.linalg.norm(value),1e-300)
        return answer.x/mass_root,answer.iterations,relative,answer.converged
    return evaluate,jax.jit(correction)


def discrete_gradient_step(grid, f, dt, *, collision_strength=1., rtol=1e-11,
                           max_steps=30, compiled_residual=None,method='dense',
                           chunk_size=None,linear_restart=30,linear_max_restarts=10,
                           linear_rtol=None):
    """Positive conservative entropy step, using dense or matrix-free Newton.

    Newton's unknown is log(f+). A residual-decreasing line search rejects
    inadmissible/nonfinite candidates. There is no population clipping or
    invariant correction. Krylov stopping uses an entropy-weighted residual;
    dense stopping retains the population metric for the small reference.
    Neither route establishes positivity of an arbitrary high-order interpolant.
    """
    if method not in ('dense','krylov'):
        raise ValueError(f'unsupported discrete-gradient method: {method}')
    if not np.isfinite(rtol) or rtol<=0 or max_steps<0:
        raise ValueError('positive finite tolerance and nonnegative iteration limit required')
    if linear_rtol is not None and (not np.isfinite(linear_rtol) or not 0<linear_rtol<1):
        raise ValueError('linear_rtol must lie in (0,1)')
    f=np.asarray(f,dtype=float).reshape(-1)
    if np.any(f<=0) or not np.all(np.isfinite(f)) or not np.isfinite(dt) or dt<0:
        raise ValueError('finite strictly positive distribution and nonnegative dt required')
    old=jnp.log(jnp.asarray(f))
    population=np.asarray(grid.weights)*f
    if np.any(population<=0) or not np.all(np.isfinite(population)):
        raise ValueError('positive finite nodal populations required')
    mass_root=np.sqrt(population)
    if method=='krylov':
        metric=mass_root
        scale=np.sqrt(population.sum())
    else:
        metric=np.ones_like(population)
        scale=np.linalg.norm(population)
    scale=max(float(scale),1e-300)
    if compiled_residual is None:
        compiled_residual=discrete_gradient_compiler(grid,collision_strength=collision_strength,
            method=method,chunk_size=chunk_size,linear_restart=linear_restart,
            linear_max_restarts=linear_max_restarts)
    evaluate,solve_or_jacobian=compiled_residual
    new_log=np.array(old)
    norm=np.inf
    linear_iterations=0;maximum_linear_residual=0.
    for iteration in range(max_steps+1):
        value=np.asarray(evaluate(new_log,old,dt))
        norm=np.linalg.norm(value/metric)/scale
        if np.isfinite(norm) and norm<=rtol:
            break
        if iteration==max_steps or not np.isfinite(norm):
            raise StepFailure(f'discrete-gradient Newton rejected step: iteration={iteration}, residual={norm:.3e}')
        if method=='dense':
            matrix=np.asarray(solve_or_jacobian(new_log,old,dt))
            try:
                direction=np.linalg.solve(matrix,-value)
            except np.linalg.LinAlgError as error:
                raise StepFailure('singular discrete-gradient Newton system') from error
        else:
            forcing=linear_rtol if linear_rtol is not None else min(.05,max(1e-5,.5*np.sqrt(norm)))
            direction,count,linear_residual,converged=solve_or_jacobian(new_log,old,dt,forcing)
            direction=np.asarray(direction)
            linear_iterations+=int(count)
            maximum_linear_residual=max(maximum_linear_residual,float(linear_residual))
            if not bool(converged) or not np.isfinite(float(linear_residual)) or float(linear_residual)>5*forcing:
                raise StepFailure(f'SOLVAX GMRES rejected Newton correction: iterations={int(count)}, '
                                  f'true relative residual={float(linear_residual):.3e}, target={forcing:.3e}')
        if not np.all(np.isfinite(direction)):
            raise StepFailure('nonfinite discrete-gradient Newton correction')
        accepted=False
        for backtrack in range(24):
            candidate=new_log+direction*2.**(-backtrack)
            # Underflow/overflow cannot be hidden by a small population norm.
            if not np.all(np.isfinite(candidate)) or np.any(candidate>np.log(np.finfo(float).max)) or np.any(candidate<np.log(np.nextafter(0.,1.))):
                continue
            candidate_norm=np.linalg.norm(np.asarray(evaluate(candidate,old,dt))/metric)/scale
            if np.isfinite(candidate_norm) and candidate_norm<norm:
                new_log,accepted=candidate,True
                break
        if not accepted:
            raise StepFailure(f'discrete-gradient line search rejected step: residual={norm:.3e}')
    new=jnp.exp(jnp.asarray(new_log))
    gradient=entropy_discrete_gradient(old,jnp.asarray(new_log))
    change=float(entropy(new,grid.weights)-entropy(jnp.asarray(f),grid.weights))
    production=float(dt*jnp.vdot(gradient,mobility_action(grid,(new+f)/2,gradient,
                      collision_strength=collision_strength,chunk_size=chunk_size)))
    identity_error=change-production
    # The exact entropy identity defect is gradient dot unscaled root residual.
    bound=float(jnp.linalg.norm(gradient*jnp.asarray(metric)))*norm*scale+100*np.finfo(float).eps*max(abs(change),1.)
    if change < -bound or abs(identity_error)>2*bound or not np.all(np.isfinite(new)) or np.any(np.asarray(new)<=0):
        raise StepFailure(f'entropy/finite check rejected step: deltaS={change:.3e}, identity residual={identity_error:.3e}')
    return NonlinearStep(new,iteration,norm,change,identity_error,float(new.min()),
                         linear_iterations,maximum_linear_residual,
                         'entropy_population' if method=='krylov' else 'population')


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
