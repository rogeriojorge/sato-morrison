"""Independent checks of the lagged-mobility entropy-variable solver.

The tiny boxes verify a temporal numerical method, not continuum convergence.
The oracle uses explicit unordered-pair Gram assembly, dense Cholesky, and an
independent NumPy pair ODE. It contains no production Krylov/preconditioner code.
"""
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import pytest
from functools import lru_cache
from scipy.integrate import solve_ivp
from scipy.linalg import cho_factor,cho_solve
from sato_morrison.collisions import (cartesian_grid,dense_mobility,uniform_grid,
    mobility_channel_covariance,mobility_action)
from sato_morrison.geometry import Field
from sato_morrison.reference import gauss_interval
from sato_morrison.solver import (StepFailure,lagged_entropy_compiler,
    lagged_entropy_step)


def convex_dense_endpoint(weights,old,matrix,dt):
    """Independent dense root using exact objective increments."""
    population=weights*old;gold=np.log(old);g=gold.copy();root=np.sqrt(population)
    for _ in range(60):
        delta=g-gold;new=population*np.exp(delta)
        residual=population*np.expm1(delta)+dt*(matrix@g)
        if np.linalg.norm(residual/root)/np.sqrt(population.sum())<1e-12:
            return np.exp(g)
        hessian=(np.diag(new)+dt*matrix)/root[:,None]/root[None,:]
        direction=cho_solve(cho_factor(hessian,lower=True),-residual/root)/root
        slope=residual@direction
        assert slope<0
        fraction=min(1.,1/max(np.max(np.abs(direction)),1.))
        for _ in range(60):
            step=fraction*direction
            remainder=np.where(np.abs(step)<1e-4,
                step**2/2+step**3/6+step**4/24+step**5/120,np.expm1(step)-step)
            change=new@remainder+residual@step+.5*dt*(step@(matrix@step))
            if change<=1e-4*fraction*slope:
                g+=step
                break
            fraction/=2
        else:
            raise AssertionError('Dense convex oracle did not find a descent step')
    raise AssertionError('Dense convex oracle did not reach its root tolerance')


@lru_cache(maxsize=3)
def make_lagged_case(field):
    spatial=[gauss_interval(3,a,b) for a,b in ((.8,1.2),(-.2,.2),(.1,.5))]
    u,wu=gauss_interval(3,-2,2);mu,wm=gauss_interval(2,.1,1.1)
    arguments=(*[pair[0] for pair in spatial],u,mu,field)
    options=dict(spatial_weights=[pair[1] for pair in spatial],
        velocity_weights=(wu,wm),spatial_discretization='polynomial')
    explicit=cartesian_grid(*arguments,**options)
    compact=cartesian_grid(*arguments,compact=True,**options)
    _,y,_,uu,mm=np.meshgrid(*[pair[0] for pair in spatial],u,mu,indexing='ij')
    initial=np.exp(-np.asarray(explicit.energy)-.2*mm.ravel()+
        .1*np.sin(np.pi*y.ravel()/.4)*uu.ravel())
    matrix=dense_mobility(explicit,initial,collision_strength=.1)
    actions=np.asarray(jax.vmap(explicit.action)(jnp.eye(explicit.size))).transpose(1,2,0)
    left,right=np.asarray(explicit.left),np.asarray(explicit.right)
    delta=actions[left]-actions[right];direction=np.asarray(explicit.kernel_directions)
    projected=delta-direction[:,:,None]*np.einsum('pa,pai->pi',direction,delta)[:,None,:]
    factor=projected[:,:3,:].reshape(-1,explicit.size)
    _,singular,vh=np.linalg.svd(factor,full_matrices=False)
    nulls=vh[singular<1e-10*singular[0]].T
    compiler=lagged_entropy_compiler(compact,collision_strength=.1,chunk_size=64)
    return field.kind,explicit,compact,initial,matrix,factor,nulls,compiler


@pytest.fixture(scope='module',params=[Field('mirror',amplitude=.15),Field('dipole'),
    Field('nonaxisymmetric',amplitude=.03)],ids=lambda field:field.kind)
def lagged_case(request):return make_lagged_case(request.param)


@pytest.mark.parametrize('axis',[None,0,(0,1,2)],ids=['diagonal','xline','spatial_block'])
def test_lagged_endpoint_matches_explicit_gram_and_preserves_nulls(lagged_case,axis):
    _,explicit,grid,initial,matrix,_,nulls,compiler=lagged_case
    if axis is not None:
        compiler=lagged_entropy_compiler(grid,collision_strength=.1,chunk_size=64,
            preconditioner_axis=axis)
    weights=np.asarray(grid.weights);old=weights*initial
    for dt in (.005,.1,1.):
        expected=convex_dense_endpoint(weights,initial,matrix,dt)
        answer=lagged_entropy_step(grid,initial,dt,collision_strength=.1,
            compiled=compiler,chunk_size=64,rtol=1e-12,preconditioner_axis=axis)
        actual=np.asarray(answer.f);delta=weights*(actual-initial)
        error=np.linalg.norm(weights*(actual-expected)/np.sqrt(old))/np.sqrt(old.sum())
        assert error<2e-10 and actual.min()>0
        assert np.linalg.norm(nulls.T@delta)/np.linalg.norm(old)<2e-10
        assert abs(delta.sum())/old.sum()<2e-10
        assert abs(np.asarray(explicit.energy)@delta)/(np.abs(np.asarray(explicit.energy))@old)<2e-10
        marginal=np.zeros(grid.shape[-1]);np.add.at(marginal,np.asarray(grid.mu_index),delta)
        assert np.max(np.abs(marginal))/old.sum()<2e-10
        assert answer.entropy_change>=-answer.entropy_defect_bound
        assert answer.generalized_KL>=0
        assert abs(answer.entropy_identity_error+answer.residual_entropy_defect)<=2*answer.entropy_defect_bound
        assert abs(answer.entropy_change-answer.entropy_production-answer.generalized_KL)<=2*answer.entropy_defect_bound


def test_lagged_arbitrary_state_residual_entropy_sign(lagged_case):
    _,_,grid,initial,matrix,_,_,compiler=lagged_case
    weights=np.asarray(grid.weights);old=weights*initial;gold=np.log(initial)
    g=gold+.03*np.cos(.4*np.arange(grid.size));new=weights*np.exp(g);dt=.1
    residual=np.asarray(compiler.evaluate(jnp.asarray(g),jnp.asarray(gold),dt))
    np.testing.assert_allclose(residual,new-old+dt*(matrix@g),rtol=3e-12,atol=2e-13)
    entropy_gain=-new@g+old@gold
    divergence=np.sum(old*np.log(old/new)-old+new)
    budget=dt*(g@(matrix@g))+divergence-(g+1)@residual
    np.testing.assert_allclose(entropy_gain,budget,rtol=3e-12,atol=2e-13)


def test_lagged_failures_remain_visible(lagged_case):
    _,_,grid,initial,_,_,_,compiler=lagged_case
    with pytest.raises(StepFailure):
        lagged_entropy_step(grid,initial,.1,collision_strength=.1,compiled=compiler,
            chunk_size=64,max_steps=0,rtol=1e-12)
    with pytest.raises((ValueError,StepFailure)):
        lagged_entropy_step(grid,initial,.1,collision_strength=.1,chunk_size=64,
            linear_max_steps=0,rtol=1e-12)



@pytest.mark.parametrize('mismatch',['strength','grid','chunk','budget','axis','spatial_axis'])
def test_lagged_bound_compiler_rejects_configuration_mismatch(mismatch):
    from dataclasses import replace
    _,_,grid,initial,_,_,_,compiler=make_lagged_case(Field('dipole'))
    target=grid
    options=dict(collision_strength=.1,chunk_size=64,linear_max_steps=1000)
    if mismatch=='strength':options['collision_strength']=.2
    elif mismatch=='grid':target=replace(grid)
    elif mismatch=='chunk':options['chunk_size']=32
    elif mismatch=='budget':options['linear_max_steps']=999
    else:options['preconditioner_axis']=0 if mismatch=='axis' else (0,1,2)
    with pytest.raises(ValueError,match='configuration mismatch'):
        lagged_entropy_step(target,initial,.1,compiled=compiler,rtol=1e-12,**options)


def test_lagged_first_order_against_independent_pair_ODE():
    _,explicit,grid,initial,_,factor,_,compiler=make_lagged_case(Field('dipole'))
    weights=np.asarray(grid.weights);old=weights*initial
    left,right=np.asarray(explicit.left),np.asarray(explicit.right)
    pair_weights=.1*np.asarray(explicit.pair_weights)
    def rhs(t,n):
        assert np.all(n>0)
        f=n/weights
        pair=np.repeat(pair_weights*f[left]*f[right],3)
        return -factor.T@(pair*(factor@np.log(f)))
    reference=solve_ivp(rhs,(0,.02),old,method='DOP853',rtol=5e-13,atol=5e-15*old)
    assert reference.success
    errors=[]
    for dt in (.02,.01,.005,.0025,.00125):
        state=initial.copy()
        for _ in range(round(.02/dt)):
            state=np.asarray(lagged_entropy_step(grid,state,dt,collision_strength=.1,
                compiled=compiler,chunk_size=64,rtol=1e-12).f)
        errors.append(np.linalg.norm((weights*state-reference.y[:,-1])/np.sqrt(old))/np.sqrt(old.sum()))
    assert all(b<a for a,b in zip(errors,errors[1:]))
    assert .8<np.log2(errors[-2]/errors[-1])<1.2


@pytest.mark.parametrize('axis',[1,-1,.0,False,True,'0',np.int64(0),(),(0,),
    (0,1),(0,1,2,3),(0.,1,2),(False,1,2),(0,np.int64(1),2),(2,1,0),[0,1,2]])
def test_lagged_invalid_preconditioner_axis_rejected(axis):
    _,_,grid,initial,_,_,_,_=make_lagged_case(Field('dipole'))
    with pytest.raises(ValueError,match='preconditioner_axis'):
        lagged_entropy_compiler(grid,preconditioner_axis=axis)
    with pytest.raises(ValueError,match='preconditioner_axis'):
        lagged_entropy_step(grid,initial,.005,preconditioner_axis=axis)


def test_channel_covariance_matches_explicit_pair_projectors(lagged_case):
    _,explicit,compact,initial,_,_,_,_=lagged_case
    count=len(explicit.derivatives)
    coefficients=np.asarray(explicit.coefficients).reshape(explicit.size,5,count)
    expected=np.zeros((explicit.size,count))
    for left,right,weight,direction in zip(np.asarray(explicit.left),
            np.asarray(explicit.right),np.asarray(explicit.pair_weights),
            np.asarray(explicit.kernel_directions)):
        projector=np.eye(5)-np.outer(direction,direction)
        pair=.1*weight*initial[left]*initial[right]
        for node in (left,right):
            spatial=(projector@coefficients[node])[:3]
            expected[node]+=pair*np.sum(spatial**2,axis=0)
    assert np.all(expected>=0)
    for chunk in (1,64):
        actual=np.asarray(mobility_channel_covariance(compact,initial,
            collision_strength=.1,chunk_size=chunk)).reshape(explicit.size,count)
        np.testing.assert_allclose(actual,expected,rtol=4e-12,atol=2e-16)
    zero=np.asarray(mobility_channel_covariance(compact,initial,
        collision_strength=0.,chunk_size=64))
    np.testing.assert_array_equal(zero,0.)


def test_xline_single_active_channel_uniform_corner():
    # Uniform construction retains a u channel whose I_x P action is exactly
    # zero. Remove that inactive channel to exercise the empty-other sum.
    from dataclasses import replace
    x=2*np.pi*np.arange(8)/8;u=np.array([-1.,0.,1.]);mu=np.array([.1,.8])
    explicit=uniform_grid(x,u,mu)
    full=uniform_grid(x,u,mu,compact=True)
    grid=replace(full,derivatives=full.derivatives[:1],
        coefficients=full.coefficients[...,:1])
    xx,uu,_=np.meshgrid(x,u,mu,indexing='ij')
    initial=np.exp(-np.asarray(grid.energy))*(1+.1*np.cos(xx.ravel())*uu.ravel())
    matrix=dense_mobility(explicit,initial,collision_strength=.1)
    probe=np.cos(.3*np.arange(grid.size))
    np.testing.assert_allclose(np.asarray(mobility_action(grid,initial,probe,
        collision_strength=.1,chunk_size=64)),matrix@probe,rtol=3e-12,atol=2e-14)
    compiler=lagged_entropy_compiler(grid,collision_strength=.1,chunk_size=64,
        preconditioner_axis=0)
    actual=lagged_entropy_step(grid,initial,.1,collision_strength=.1,
        compiled=compiler,chunk_size=64,preconditioner_axis=0,rtol=1e-12)
    weights=np.asarray(grid.weights);old=weights*initial
    expected=convex_dense_endpoint(weights,initial,matrix,.1)
    error=np.linalg.norm(weights*(np.asarray(actual.f)-expected)/np.sqrt(old))/np.sqrt(old.sum())
    assert error<2e-10 and actual.minimum>0
    # The tuple means actual spatial axes. It must not silently treat this
    # reduced layout's u and mu axes as Cartesian y and z.
    with pytest.raises(ValueError,match='five-axis Cartesian'):
        lagged_entropy_compiler(grid,preconditioner_axis=(0,1,2))


@pytest.mark.parametrize('axis',[None,0,(0,1,2)],ids=['diagonal','xline','spatial_block'])
def test_fixed_reference_matches_independent_physical_root_and_budgets(axis):
    _,explicit,grid,initial,matrix,_,nulls,legacy=make_lagged_case(Field('dipole'))
    weights=np.asarray(grid.weights);old=weights*initial;gold=np.log(initial)
    # Five decades of relative reference variation exercise a genuine change
    # of congruence, rather than merely passing the old population twice.
    reference=old*np.exp(6*np.cos(np.arange(grid.size)*.31))
    reference*=old.sum()/reference.sum()
    assert np.max(reference/old)/np.min(reference/old)>1e5
    compiler=lagged_entropy_compiler(grid,collision_strength=.1,chunk_size=64,
        preconditioner_axis=axis,reference_population=reference)
    dt=.1
    expected=convex_dense_endpoint(weights,initial,matrix,dt)
    answer=lagged_entropy_step(grid,initial,dt,collision_strength=.1,compiled=compiler,
        chunk_size=64,preconditioner_axis=axis,rtol=1e-12,linear_rtol=1e-8)
    legacy_answer=lagged_entropy_step(grid,initial,dt,collision_strength=.1,
        compiled=legacy,chunk_size=64,rtol=1e-12)
    actual=np.asarray(answer.f);new=weights*actual;change=new-old;g=np.log(actual)
    for target in (expected,np.asarray(legacy_answer.f)):
        error=np.linalg.norm(weights*(actual-target)/np.sqrt(old))/np.sqrt(old.sum())
        assert error<2e-10
    assert answer.residual_metric=='fixed_reference_population'
    assert answer.minimum>0 and np.all(np.isfinite(new))
    # The expected residual and entropy budget use the independent explicit
    # Gram matrix, not the compiled kernels or channel preconditioner.
    residual=change+dt*(matrix@g)
    root=np.sqrt(reference);scale=np.sqrt(reference.sum())
    independent_norm=np.linalg.norm(residual/root)/scale
    assert independent_norm<2e-11
    assert abs(change.sum())/old.sum()<2e-10
    energy=np.asarray(grid.energy)
    assert abs(energy@change)/(np.abs(energy)@old)<2e-10
    marginal=np.bincount(np.asarray(grid.mu_index),weights=change)
    assert np.max(np.abs(marginal))/old.sum()<2e-10
    assert np.linalg.norm(nulls.T@change)/np.linalg.norm(old)<2e-10
    invariants=[np.ones(grid.size),energy]
    invariants.extend((np.asarray(grid.mu_index)==i).astype(float)
        for i in range(grid.shape[-1]))
    for invariant in invariants:
        contraction=invariant@residual
        dual_bound=np.linalg.norm(root*invariant)*np.linalg.norm(residual/root)
        assert abs(contraction)<=dual_bound*(1+1e-14)
        np.testing.assert_allclose(invariant@change,contraction,rtol=0.,atol=2e-15)
    entropy=-new@g+old@gold
    physical=dt*(g@(matrix@g))
    kl=np.sum(old*np.log(old/new)-old+new)
    defect=(g+1)@residual
    cauchy=np.linalg.norm(root*(g+1))*np.linalg.norm(residual/root)
    assert abs(defect)<=cauchy*(1+1e-14)
    np.testing.assert_allclose(entropy,physical+kl-defect,rtol=2e-12,atol=2e-15)
    assert abs(answer.entropy_identity_error+answer.residual_entropy_defect)<=2*answer.entropy_defect_bound
    # Independently verify the non-root scaled Hessian, where mass ratios
    # differ from one and the reference cannot accidentally be used as K(old).
    trial=gold+.03*np.cos(.4*np.arange(grid.size))
    prepared=compiler.prepare_diagonal(jnp.asarray(gold))
    direction,_,true,success,*_=compiler.correction(
        jnp.asarray(trial),jnp.asarray(gold),dt,prepared,1e-9)
    scaled=(np.diag(weights*np.exp(trial))+dt*matrix)/root[:,None]/root[None,:]
    rhs=-(weights*np.exp(trial)-old+dt*(matrix@trial))/root
    expected_direction=np.linalg.solve(scaled,rhs)/root
    assert bool(success) and float(true)<5e-9
    np.testing.assert_allclose(np.asarray(direction),expected_direction,rtol=2e-6,atol=2e-9)
    # Reuse the fixed metric while updating the physical mobility. An oracle
    # frozen at the initial distribution would give a different second root.
    next_matrix=dense_mobility(explicit,actual,collision_strength=.1)
    next_expected=convex_dense_endpoint(weights,actual,next_matrix,dt)
    next_answer=lagged_entropy_step(grid,actual,dt,collision_strength=.1,
        compiled=compiler,chunk_size=64,preconditioner_axis=axis,
        rtol=1e-12,linear_rtol=1e-8)
    error=np.linalg.norm(weights*(np.asarray(next_answer.f)-next_expected)/np.sqrt(old))/np.sqrt(old.sum())
    assert error<2e-10 and next_answer.residual_metric=='fixed_reference_population'


def test_fixed_references_are_copied_immutable_and_independent():
    from dataclasses import FrozenInstanceError
    _,_,grid,initial,_,_,_,_=make_lagged_case(Field('dipole'))
    population=np.asarray(grid.weights)*initial
    first=population.copy();second=population[::-1].copy()
    a=lagged_entropy_compiler(grid,reference_population=first)
    b=lagged_entropy_compiler(grid,reference_population=second)
    first[:]=0.;second[:]=np.nan
    np.testing.assert_array_equal(np.asarray(a.reference_population),population)
    np.testing.assert_array_equal(np.asarray(b.reference_population),population[::-1])
    assert a.reference_population is not b.reference_population
    with pytest.raises(FrozenInstanceError):a.reference_population=None
    with pytest.raises(TypeError):a.reference_population[0]=0.


@pytest.mark.parametrize('bad',['shape','length','zero','negative','nan','inf','total'])
def test_invalid_fixed_reference_rejected(bad):
    _,_,grid,initial,_,_,_,_=make_lagged_case(Field('dipole'))
    reference=(np.asarray(grid.weights)*initial).copy()
    if bad=='shape':reference=reference.reshape(grid.shape)
    elif bad=='length':reference=reference[:-1]
    elif bad=='total':reference[:]=np.finfo(float).max
    else:reference[0]={'zero':0.,'negative':-1.,'nan':np.nan,'inf':np.inf}[bad]
    with np.errstate(over='ignore'),pytest.raises(ValueError,match='reference'):
        lagged_entropy_compiler(grid,reference_population=reference)


def test_reference_mass_mismatch_and_numerical_failure_remain_visible():
    from dataclasses import replace
    _,_,grid,initial,_,_,_,_=make_lagged_case(Field('dipole'))
    population=np.asarray(grid.weights)*initial
    compiler=lagged_entropy_compiler(grid,collision_strength=.1,chunk_size=64,
        reference_population=population)
    options=dict(collision_strength=.1,compiled=compiler,chunk_size=64,rtol=1e-12)
    # Validation applies on every step, including a changed old state and dt=0.
    for state in (initial*(1+2e-9),initial*2):
        with pytest.raises(ValueError,match='reference total population'):
            lagged_entropy_step(grid,state,0.,**options)
    with pytest.raises(StepFailure,match='Newton rejected'):
        lagged_entropy_step(grid,initial,.1,max_steps=0,**options)
    def inaccurate(*args):
        return (jnp.ones(grid.size),1,1e-2,True,-1.,1.,1)
    options['compiled']=replace(compiler,correction=inaccurate)
    with pytest.raises(StepFailure,match='PCG rejected'):
        lagged_entropy_step(grid,initial,.1,**options)


def test_default_reference_none_retains_legacy_metric_and_result():
    _,_,grid,initial,_,_,_,default=make_lagged_case(Field('dipole'))
    explicit_none=lagged_entropy_compiler(grid,collision_strength=.1,chunk_size=64,
        reference_population=None)
    assert default.reference_population is None and explicit_none.reference_population is None
    answers=[lagged_entropy_step(grid,initial,.005,collision_strength=.1,
        compiled=compiler,chunk_size=64,rtol=1e-12) for compiler in (default,explicit_none)]
    np.testing.assert_array_equal(np.asarray(answers[0].f),np.asarray(answers[1].f))
    assert answers[0].relative_residual==answers[1].relative_residual
    assert answers[0].linear_iterations==answers[1].linear_iterations
    assert answers[0].residual_metric==answers[1].residual_metric=='entropy_population'


@pytest.mark.parametrize('fixed_reference',[False,True],ids=['old_population','fixed_reference'])
def test_spatial_block_matches_independent_noncubic_auxiliary(monkeypatch,fixed_reference):
    # Noncubic space and distinct derivative matrices expose transpose/Kronecker
    # mistakes that a cubic, equally spaced tensor could conceal.
    from types import SimpleNamespace
    import sato_morrison.solver as solver_module
    spatial=[gauss_interval(n,a,b) for n,(a,b) in zip((3,4,3),((.8,1.2),(-.2,.2),(.1,.5)))]
    u,wu=gauss_interval(3,-2,2);mu,wm=gauss_interval(2,.1,1.1)
    options=dict(spatial_weights=[p[1] for p in spatial],velocity_weights=(wu,wm),
        spatial_discretization='polynomial')
    args=(*[p[0] for p in spatial],u,mu,Field('dipole'))
    explicit=cartesian_grid(*args,**options);grid=cartesian_grid(*args,compact=True,**options)
    old=np.exp(-np.asarray(grid.energy)+.05*np.cos(.31*np.arange(grid.size)))
    gold=np.log(old);g=gold+.07*np.sin(.27*np.arange(grid.size))
    weights=np.asarray(grid.weights);population=weights*old;metric=population.copy()
    if fixed_reference:
        metric*=np.exp(2*np.cos(.19*np.arange(grid.size)))
        metric*=population.sum()/metric.sum()
    root=np.sqrt(metric);dt=.1
    # Direct unordered-pair self covariance, independent of both production
    # covariance and channel diagonal helpers.
    coefficients=np.asarray(explicit.coefficients).reshape(grid.size,5,-1)
    q=np.zeros((grid.size,len(grid.derivatives)))
    for i,j,w,direction in zip(np.asarray(explicit.left),np.asarray(explicit.right),
            np.asarray(explicit.pair_weights),np.asarray(explicit.kernel_directions)):
        projector=np.eye(5)-np.outer(direction,direction)
        for node in (i,j):
            channels=(projector@coefficients[node])[:3]
            q[node]+=.1*w*old[i]*old[j]*np.sum(channels**2,axis=0)
    auxiliary=np.zeros((grid.size,grid.size));expected_other=np.zeros(grid.size)
    for component,(axis,matrix) in enumerate(grid.derivatives):
        derivative=np.ones((1,1))
        for d,n in enumerate(grid.shape):
            derivative=np.kron(derivative,np.asarray(matrix) if d==axis else np.eye(n))
        if axis<3:auxiliary+=derivative.T@(q[:,component,None]*derivative)
        else:expected_other+=(derivative**2).T@q[:,component]
    auxiliary+=np.diag(expected_other)
    expected=np.diag(weights*np.exp(g)/metric)+dt*auxiliary/root[:,None]/root[None,:]
    matrix=dense_mobility(explicit,old,collision_strength=.1)
    scale=np.linalg.norm(24*auxiliary-matrix,2)
    assert np.linalg.eigvalsh(auxiliary).min()>-1e-12*max(scale,1e-30)
    assert np.linalg.eigvalsh(24*auxiliary-matrix).min()>-1e-11*max(scale,1e-30)
    assert np.linalg.eigvalsh(expected).min()>0
    compiler=lagged_entropy_compiler(grid,collision_strength=.1,chunk_size=64,
        preconditioner_axis=(0,1,2),reference_population=metric if fixed_reference else None)
    prepared=compiler.prepare_diagonal(jnp.asarray(gold))
    nx=int(np.prod(grid.spatial_shape));nv=int(np.prod(grid.velocity_shape))
    assert all(np.all(np.isfinite(a)) and np.all(np.asarray(a)>=0) for a in prepared)
    np.testing.assert_allclose(np.asarray(prepared.line_q),q[:,:3].reshape(nx,nv,3).transpose(1,0,2),rtol=4e-12,atol=2e-16)
    np.testing.assert_allclose(np.asarray(prepared.other_diagonal),expected_other.reshape(nx,nv).T,rtol=4e-12,atol=2e-16)
    captured={}
    def capture(operator,rhs,*,precond,**unused):
        captured.update(operator=operator,inverse=precond)
        return SimpleNamespace(x=jnp.zeros_like(rhs),iterations=0,converged=False,status=0)
    monkeypatch.setattr(solver_module,'pcg',capture)
    # Invoke the undecorated function so captured closures contain concrete
    # arrays rather than leaking tracers from a JIT trace. No Krylov oracle.
    compiler.correction.__wrapped__(jnp.asarray(g),jnp.asarray(gold),dt,prepared,1e-9)
    actual_inverse=np.asarray(jax.vmap(captured['inverse'])(jnp.eye(grid.size))).T
    np.testing.assert_allclose(actual_inverse,np.linalg.inv(expected),rtol=2e-11,atol=2e-13)
    exact=np.diag(weights*np.exp(g)/metric)+dt*matrix/root[:,None]/root[None,:]
    actual_hessian=np.asarray(jax.vmap(captured['operator'])(jnp.eye(grid.size))).T
    np.testing.assert_allclose(actual_hessian,exact,rtol=4e-12,atol=2e-13)


@pytest.mark.parametrize('bad',['spatial_shape','velocity_shape','quadrature','missing_channel','duplicate_channel','matrix_shape'])
def test_spatial_block_rejects_unsupported_layout(bad):
    from dataclasses import replace
    _,_,grid,_,_,_,_,_=make_lagged_case(Field('dipole'))
    if bad=='spatial_shape':grid=replace(grid,spatial_shape=(3,9))
    elif bad=='velocity_shape':grid=replace(grid,velocity_shape=(6,))
    elif bad=='quadrature':grid=replace(grid,local_quadrature=(grid.local_quadrature[0],grid.local_quadrature[1][:-1]))
    elif bad=='missing_channel':grid=replace(grid,derivatives=grid.derivatives[1:])
    elif bad=='duplicate_channel':grid=replace(grid,derivatives=grid.derivatives+(grid.derivatives[0],))
    else:grid=replace(grid,derivatives=((0,jnp.eye(2)),)+grid.derivatives[1:])
    with pytest.raises(ValueError,match='spatial block'):
        lagged_entropy_compiler(grid,preconditioner_axis=(0,1,2))


def test_spatial_block_true_residual_failure_remains_visible():
    _,_,grid,initial,_,_,_,_=make_lagged_case(Field('dipole'))
    compiler=lagged_entropy_compiler(grid,collision_strength=.1,chunk_size=64,
        linear_max_steps=0,preconditioner_axis=(0,1,2))
    with pytest.raises(StepFailure,match='PCG rejected'):
        lagged_entropy_step(grid,initial,.1,collision_strength=.1,compiled=compiler,
            chunk_size=64,linear_max_steps=0,preconditioner_axis=(0,1,2),rtol=1e-12)


def test_line_search_reports_population_underflow_without_accepting_state():
    # Isolate globalization with an exact one-coordinate strictly convex
    # Newton direction. This is a guard test, not a collision endpoint oracle.
    # Every declared trial has a positive NumPy density but crosses the normal
    # boundary where this JAX CPU backend flushes exp's subnormal output.
    from dataclasses import replace
    x=2*np.pi*np.arange(4)/4
    grid=uniform_grid(x,np.array([-1.,0.,1.]),np.array([.1,.8]),compact=True)
    initial=np.ones(grid.size)
    boundary=np.log(np.finfo(float).tiny)
    initial[0]=np.exp(boundary+2.**-35)
    old=initial.copy();weights=np.asarray(grid.weights)
    reference=np.full(grid.size,(weights*initial).sum()/grid.size)
    compiler=lagged_entropy_compiler(grid,collision_strength=0.,chunk_size=64,
        reference_population=reference)
    target=np.log(initial[0])-1.
    def residual(g,gold,dt):
        result=np.zeros(grid.size)
        result[0]=float(np.asarray(g)[0])-target
        return jnp.asarray(result)
    def exact_convex_direction(g,gold,dt,prepared,linear_rtol):
        # Hessian = I at this coordinate; slope and curvature are exact.
        direction=np.zeros(grid.size)
        direction[0]=target-float(np.asarray(g)[0])
        slope=-direction[0]**2
        return (jnp.asarray(direction),1,0.,True,slope,direction[0]**2,1)
    def unexpected_objective(*args):
        raise AssertionError('Population-rejected trials must not evaluate Armijo')
    compiler=replace(compiler,evaluate=residual,correction=exact_convex_direction,
        prepare_diagonal=lambda old:jnp.zeros(grid.size),
        objective_difference=unexpected_objective)
    smallest_trial=np.log(initial[0])-2.**-29
    assert smallest_trial>np.log(np.nextafter(0.,1.))
    assert np.exp(smallest_trial)>0
    # Explicitly verify the backend mechanism rather than assuming that all
    # floating implementations treat subnormals identically.
    if float(jnp.exp(jnp.asarray(smallest_trial)))>0:
        pytest.skip('Backend retains subnormal exponentials; CPU flush barrier absent')
    with pytest.raises(StepFailure,match='population') as captured:
        lagged_entropy_step(grid,initial,.1,compiled=compiler,
            collision_strength=0.,chunk_size=64,rtol=1e-12)
    diagnostic=captured.value.line_search_diagnostics
    assert diagnostic['rejection_counts']=={'log_range':0,'population':30,'armijo':0}
    assert diagnostic['last_fraction']==2.**-29
    assert diagnostic['last_backtrack']==29
    assert diagnostic['last_minimum_population']==0.
    assert diagnostic['last_minimum_log']<boundary
    assert diagnostic['last_objective_change'] is None
    assert diagnostic['last_armijo_rhs']<0
    np.testing.assert_array_equal(initial,old)
