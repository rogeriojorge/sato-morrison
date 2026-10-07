"""Checked implicit weak collision steps and tangent toroidal transport."""
from dataclasses import dataclass
from typing import NamedTuple
from jax.scipy.linalg import solve_triangular
import jax
import jax.numpy as jnp
import numpy as np
from solvax import pcg_linear_solve, gmres, pcg
from .collisions import (mobility_action, compact_mobility_tangent,
                         mobility_channel_covariance, mobility_channel_diagonals)


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

    Matrix-free GMRES uses exact prepared pair tangents (and an independent
    JAX Jacobian-vector product on explicit-pair reference grids). Rows and log increments
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
        if grid.local_quadrature is None:
            # The explicit-pair route remains an independent AD reference.
            operator=lambda vector:jax.jvp(scaled,(new_log,),(vector/mass_root,))[1]
        else:
            new,old=jnp.exp(new_log),jnp.exp(old_log)
            midpoint=(new+old)/2
            gradient=entropy_discrete_gradient(old_log,new_log)
            gradient_prime=jax.jvp(lambda value:entropy_discrete_gradient(old_log,value),
                (new_log,),(jnp.ones_like(new_log),))[1]
            action_gradient=grid.action(gradient)
            def operator(vector):
                increment=vector/mass_root
                tangent=compact_mobility_tangent(grid,midpoint,action_gradient,
                    gradient_prime*increment,new/(new+old)*increment,
                    collision_strength=collision_strength,chunk_size=chunk_size)
                return (grid.weights*new*increment-timestep*tangent)/mass_root
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


@dataclass(frozen=True)
class LaggedEntropyStep(NonlinearStep):
    """Lagged BE: entropy change = production + KL - residual_entropy_defect."""
    generalized_KL: float = 0.
    entropy_production: float = 0.
    entropy_defect_bound: float = 0.
    iteration_history: tuple = ()
    residual_entropy_defect: float = 0.


def _population_difference(new_log,old_log,weights):
    delta=new_log-old_log
    old_mass=weights*jnp.exp(old_log)
    return jnp.where(jnp.abs(delta)<.5,old_mass*jnp.expm1(delta),
                     weights*(jnp.exp(new_log)-jnp.exp(old_log)))


def _expm1_minus_x(value):
    """Nonnegative exponential remainder, stable near zero."""
    series=value**2*(.5+value*(1/6+value*(1/24+value*(1/120+value/720))))
    return jnp.where(jnp.abs(value)<1e-3,series,jnp.expm1(value)-value)


def _check_preconditioner_axis(axis):
    if axis is not None and (type(axis) is not int or axis != 0):
        raise ValueError('preconditioner_axis must be None or Cartesian x axis0')


class LinePreconditionerData(NamedTuple):
    line_q: object
    other_diagonal: object


@dataclass(frozen=True,eq=False)
class LaggedEntropyCompiler:
    """Reusable kernels bound to one grid and explicit solver configuration."""
    evaluate: object
    correction: object
    objective_difference: object
    prepare_diagonal: object
    apply: object
    grid: object
    collision_strength: float
    chunk_size: int
    linear_max_steps: int
    preconditioner_axis: object


def lagged_entropy_compiler(grid, *, collision_strength=1.,chunk_size=65536,
                            linear_max_steps=1000,preconditioner_axis=None):
    """Reusable exact frozen-mobility residual and entropy-scaled SPD solve.

    All old-state arguments are dynamic. Positive covariance is prepared once
    per step. None selects lumped Jacobi; axis0 retains D_x.T diag(q_x) D_x
    correlations in independent SPD spatial lines and lumps the other channels.
    Both affect only preconditioning; all pair and derivative cross terms stay
    in the exact Hessian. PCG's true original residual is independently checked.
    SOLVAX requires linear_rtol to be static when tracing its input validation.
    """
    if grid.local_quadrature is None:
        raise ValueError('lagged entropy compiler requires a compact local grid')
    if not np.isfinite(collision_strength) or collision_strength<0:
        raise ValueError('finite nonnegative collision strength required')
    if not isinstance(linear_max_steps,int) or linear_max_steps<0:
        raise ValueError('nonnegative integer linear iteration limit required')
    if not isinstance(chunk_size,int) or chunk_size<1:
        raise ValueError('positive integer chunk size required')
    _check_preconditioner_axis(preconditioner_axis)
    if preconditioner_axis is not None:
        matches=[(component,matrix) for component,(axis,matrix) in enumerate(grid.derivatives) if axis==preconditioner_axis]
        if len(matches)!=1:raise ValueError('one derivative channel required on line axis')
        component,derivative=matches[0];length=grid.shape[preconditioner_axis]
        def lines(value):
            return jnp.moveaxis(value.reshape(grid.shape),preconditioner_axis,-1).reshape(-1,length)
        def unlines(value):
            shape=tuple(n for axis,n in enumerate(grid.shape) if axis!=preconditioner_axis)+(length,)
            return jnp.moveaxis(value.reshape(shape),-1,preconditioner_axis).reshape(-1)
    def apply(old_log,h):
        return mobility_action(grid,jnp.exp(old_log),h,
            collision_strength=collision_strength,chunk_size=chunk_size)
    def evaluate(new_log,old_log,dt):
        old_mass=grid.weights*jnp.exp(old_log)
        return _population_difference(new_log,old_log,grid.weights)+dt*apply(old_log,new_log)
    def prepare(old_log):
        q=mobility_channel_covariance(grid,jnp.exp(old_log),
            collision_strength=collision_strength,chunk_size=chunk_size)
        diags=mobility_channel_diagonals(grid,q)
        if preconditioner_axis is None:return jnp.sum(diags,axis=-1).reshape(-1)
        other_channels=[diags[...,d] for d in range(len(grid.derivatives)) if d!=component]
        other=(jnp.sum(jnp.stack(other_channels),axis=0) if other_channels
               else jnp.zeros(grid.shape,dtype=q.dtype))
        return LinePreconditionerData(lines(q[...,component]),lines(other))
    def correction(new_log,old_log,dt,diagonal,linear_rtol):
        old_mass=grid.weights*jnp.exp(old_log);root=jnp.sqrt(old_mass)
        ratio=jnp.exp(new_log-old_log);value=evaluate(new_log,old_log,dt)/root
        def operator(vector):
            return ratio*vector+dt*apply(old_log,vector/root)/root
        if preconditioner_axis is None:
            approximation=ratio+dt*diagonal/old_mass
            inverse=lambda vector:vector/approximation
        else:
            masses=lines(old_mass);mass_root=jnp.sqrt(masses)
            local=jnp.einsum('ki,lk,kj->lij',derivative,diagonal.line_q,derivative)
            block=dt*local/mass_root[:,:,None]/mass_root[:,None,:]
            diagonal_entries=lines(ratio)+dt*diagonal.other_diagonal/masses
            block=block+jnp.eye(length)[None,:,:]*diagonal_entries[:,:,None]
            factor=jnp.linalg.cholesky(block)
            def inverse(vector):
                intermediate=jax.vmap(lambda L,v:solve_triangular(L,v,lower=True))(factor,lines(vector))
                solved=jax.vmap(lambda L,v:solve_triangular(L.T,v,lower=False))(factor,intermediate)
                return unlines(solved)
        answer=pcg(operator,-value,precond=inverse,
            rtol=linear_rtol,atol=0.,max_steps=linear_max_steps)
        direction=answer.x/root
        true=jnp.linalg.norm(operator(answer.x)+value)/jnp.maximum(jnp.linalg.norm(value),1e-300)
        descent=jnp.vdot(value,answer.x).real
        curvature=jnp.vdot(direction,apply(old_log,direction)).real
        return direction,answer.iterations,true,answer.converged,descent,curvature,answer.status
    def objective_difference(new_log,old_log,dt,direction,fraction,descent,curvature):
        new_mass=grid.weights*jnp.exp(new_log);shift=fraction*direction
        return fraction*descent+jnp.sum(new_mass*_expm1_minus_x(shift))+.5*dt*fraction**2*curvature
    return LaggedEntropyCompiler(jax.jit(evaluate),jax.jit(correction,static_argnums=(4,)),
        jax.jit(objective_difference),jax.jit(prepare),jax.jit(apply),
        grid,float(collision_strength),chunk_size,linear_max_steps,preconditioner_axis)


def lagged_entropy_step(grid,f,dt,*,collision_strength=1.,rtol=1e-11,
                        max_steps=80,compiled=None,chunk_size=65536,
                        linear_rtol=1e-6,linear_max_steps=1000,
                        iteration_callback=None,preconditioner_axis=None):
    """First-order positive conservative lagged-mobility entropy step.

    The root w(exp(g)-f)+dt K(f)g=0 minimizes a strictly convex coercive
    objective for positive nodal populations. Frozen positive Gram weights
    preserve the same actual discrete nullspace. This is a first-order step
    for the original kinetic ODE; KL is numerical entropy dissipation. It does
    not assert a continuum convergence or positivity of polynomial density
    interpolation. Reuse a compiler built with matching grid/D/linear budget/axis.
    """
    from time import perf_counter
    _check_preconditioner_axis(preconditioner_axis)
    if not np.isfinite(rtol) or rtol<=0 or not isinstance(max_steps,int) or max_steps<0:
        raise ValueError('positive finite tolerance and nonnegative integer iteration limit required')
    if not np.isfinite(linear_rtol) or not 0<linear_rtol<1:
        raise ValueError('linear_rtol must lie in (0,1)')
    if not np.isfinite(collision_strength) or collision_strength<0:
        raise ValueError('finite nonnegative collision strength required')
    if not np.isfinite(dt) or dt<0:
        raise ValueError('finite nonnegative timestep required')
    if not isinstance(linear_max_steps,int) or linear_max_steps<0 or not isinstance(chunk_size,int) or chunk_size<1:
        raise ValueError('nonnegative integer linear limit and positive integer chunk size required')
    f=np.asarray(f,dtype=float).reshape(-1);weights=np.asarray(grid.weights)
    if weights.shape!=(grid.size,) or not np.all(np.isfinite(weights)) or np.any(weights<=0):
        raise ValueError('positive finite weights with grid size required')
    if f.size!=grid.size or not np.all(np.isfinite(f)) or np.any(f<=0):
        raise ValueError('finite strictly positive distribution with grid size required')
    population=weights*f
    if not np.all(np.isfinite(population)) or np.any(population<=0):
        raise ValueError('positive finite nodal populations required')
    if compiled is None:
        compiled=lagged_entropy_compiler(grid,collision_strength=collision_strength,
            chunk_size=chunk_size,linear_max_steps=linear_max_steps,preconditioner_axis=preconditioner_axis)
    if (not isinstance(compiled,LaggedEntropyCompiler) or compiled.grid is not grid
        or compiled.collision_strength!=float(collision_strength)
        or compiled.chunk_size!=chunk_size or compiled.linear_max_steps!=linear_max_steps
        or compiled.preconditioner_axis!=preconditioner_axis):
        raise ValueError('lagged compiler grid or configuration mismatch')
    evaluate=compiled.evaluate;correct=compiled.correction
    objective_difference=compiled.objective_difference
    prepare=compiled.prepare_diagonal;apply=compiled.apply
    total_mass=float(population.sum())
    if not np.isfinite(total_mass) or total_mass<=0:
        raise ValueError('positive finite total mass required')
    old=jnp.log(jnp.asarray(f));new_log=np.asarray(old).copy()
    root=np.sqrt(population);scale=np.sqrt(total_mass)
    diagonal=prepare(old);jax.tree.map(lambda value:value.block_until_ready(),diagonal)
    if any(not np.all(np.isfinite(value)) or np.any(np.asarray(value)<0) for value in jax.tree.leaves(diagonal)):
        raise StepFailure('lagged preconditioner is not finite nonnegative')
    history=[];linear_iterations=0;maximum_linear_residual=0.
    for iteration in range(max_steps+1):
        value=np.asarray(evaluate(new_log,old,dt));norm=float(np.linalg.norm(value/root)/scale)
        if np.isfinite(norm) and norm<=rtol:break
        if iteration==max_steps or not np.isfinite(norm):
            raise StepFailure(f'lagged entropy Newton rejected step: iteration={iteration}, residual={norm:.3e}')
        start=perf_counter();answer=correct(new_log,old,dt,diagonal,linear_rtol)
        answer[0].block_until_ready()
        direction=np.asarray(answer[0]);count=int(answer[1]);true=float(answer[2])
        descent=float(answer[4]);curvature=float(answer[5])
        linear_iterations+=count;maximum_linear_residual=max(maximum_linear_residual,true)
        record={'iteration':iteration+1,'relative_residual':norm,'linear_iterations':count,
            'true_linear_relative_residual':true,'linear_wall_s':perf_counter()-start,
            'maximum_log_direction':float(np.max(np.abs(direction))),
            'minimum_signed_log_direction':float(np.min(direction)),
            'maximum_signed_log_direction':float(np.max(direction)),
            'objective_descent':descent,'mobility_curvature':curvature}
        history.append(record)
        if iteration_callback is not None:iteration_callback(dict(record))
        if not bool(answer[3]) or not np.isfinite(true) or true>5*linear_rtol:
            raise StepFailure(f'SOLVAX PCG rejected lagged correction: status={int(answer[6])}, iterations={count}, true residual={true:.3e}')
        if not np.all(np.isfinite(direction)) or not np.isfinite(descent) or not np.isfinite(curvature) or descent>=0:
            raise StepFailure('lagged Newton direction is nonfinite or lacks objective descent')
        alpha0=1.
        for backtrack in range(30):
            fraction=alpha0*2.**(-backtrack);candidate=new_log+fraction*direction
            if not np.all(np.isfinite(candidate)) or np.any(candidate>np.log(np.finfo(float).max)) or np.any(candidate<np.log(np.nextafter(0.,1.))):continue
            candidate_population=weights*np.asarray(jnp.exp(jnp.asarray(candidate)))
            if not np.all(np.isfinite(candidate_population)) or np.any(candidate_population<=0):continue
            change=float(objective_difference(new_log,old,dt,direction,fraction,descent,curvature))
            if np.isfinite(change) and change<=1e-4*fraction*descent:
                new_log=candidate
                record.update({'fraction':fraction,'backtracks':backtrack,
                    'maximum_log_step':fraction*np.max(np.abs(direction)),
                    'objective_change':change})
                if iteration_callback is not None:iteration_callback(dict(record))
                break
        else:raise StepFailure(f'lagged convex objective line search rejected step: residual={norm:.3e}')
    new=jnp.exp(jnp.asarray(new_log));delta=jnp.asarray(new_log)-old
    production=float(dt*jnp.vdot(jnp.asarray(new_log),apply(old,jnp.asarray(new_log))).real)
    divergence=float(jnp.sum(jnp.asarray(population)*_expm1_minus_x(delta)))
    new_mass=grid.weights*new
    delta_mass=_population_difference(jnp.asarray(new_log),old,grid.weights)
    change=float(-jnp.sum(delta_mass*old+new_mass*delta))
    identity_error=change-production-divergence
    residual_defect=float(jnp.vdot(jnp.asarray(new_log)+1,jnp.asarray(value)).real)
    roundoff_scale=float(jnp.sum(jnp.abs(jnp.asarray(population)*old))+jnp.sum(jnp.abs(new_mass*jnp.asarray(new_log))))+abs(production)+abs(divergence)+abs(residual_defect)
    bound=float(np.linalg.norm(root*(new_log+1))*norm*scale)+100*np.finfo(float).eps*roundoff_scale
    if (not np.all(np.isfinite([change,production,divergence,identity_error,residual_defect,bound]))
        or production< -bound or divergence< -bound or change< -bound
        or abs(identity_error+residual_defect)>2*bound
        or not np.all(np.isfinite(new)) or np.any(np.asarray(new)<=0)):
        raise StepFailure(f'lagged entropy/finite check rejected step: deltaS={change:.3e}, identity={identity_error:.3e}')
    return LaggedEntropyStep(new,iteration,norm,change,identity_error,float(new.min()),
        linear_iterations,maximum_linear_residual,'entropy_population',
        divergence,production,bound,tuple(history),residual_defect)
