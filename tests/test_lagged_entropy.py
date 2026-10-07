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
from sato_morrison.collisions import cartesian_grid,dense_mobility
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


def test_lagged_endpoint_matches_explicit_gram_and_preserves_nulls(lagged_case):
    _,explicit,grid,initial,matrix,_,nulls,compiler=lagged_case
    weights=np.asarray(grid.weights);old=weights*initial
    for dt in (.005,.1,1.):
        expected=convex_dense_endpoint(weights,initial,matrix,dt)
        answer=lagged_entropy_step(grid,initial,dt,collision_strength=.1,
            compiled=compiler,chunk_size=64,rtol=1e-12)
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



@pytest.mark.parametrize('mismatch',['strength','grid','chunk','budget'])
def test_lagged_bound_compiler_rejects_configuration_mismatch(mismatch):
    from dataclasses import replace
    _,_,grid,initial,_,_,_,compiler=make_lagged_case(Field('dipole'))
    target=grid
    options=dict(collision_strength=.1,chunk_size=64,linear_max_steps=1000)
    if mismatch=='strength':options['collision_strength']=.2
    elif mismatch=='grid':target=replace(grid)
    elif mismatch=='chunk':options['chunk_size']=32
    else:options['linear_max_steps']=999
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
