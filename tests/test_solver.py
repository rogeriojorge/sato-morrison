import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import pytest
from sato_morrison.collisions import uniform_grid,mobility_action,dense_mobility
from sato_morrison.solver import (linear_step,checked_linear_step,StepFailure,
    discrete_gradient_step,discrete_gradient_compiler,entropy_discrete_gradient,
    entropy,invariant_diagnostics,toroidal_stream)


def setup():
    grid=uniform_grid(np.arange(5)*2*np.pi/5,np.linspace(-2,2,3),np.linspace(.1,1.1,3))
    f=jnp.exp(-grid.energy)
    return grid,f


def test_solvax_dense_step_and_implicit_derivative():
    grid,f=setup(); mass=grid.weights*f; dense=jnp.asarray(dense_mobility(grid,f))
    h=jnp.asarray(np.random.default_rng(2).normal(size=grid.size));dt=.2
    stiffness=lambda value:mobility_action(grid,f,value)
    answer=checked_linear_step(mass,stiffness,h,dt)
    expected=jnp.linalg.solve(jnp.diag(mass)+dt*dense,mass*h)
    np.testing.assert_allclose(answer.x,expected,atol=2e-11)
    def solve_d(strength):
        return jnp.sum(linear_step(mass,lambda value:strength*stiffness(value),h,dt).x**2)
    def dense_d(strength):
        value=jnp.linalg.solve(jnp.diag(mass)+dt*strength*dense,mass*h)
        return jnp.sum(value**2)
    np.testing.assert_allclose(jax.grad(solve_d)(1.),jax.grad(dense_d)(1.),rtol=2e-9)
    for eps in (1e-3,1e-4,1e-5):
        fd=(solve_d(1+eps)-solve_d(1-eps))/(2*eps)
        np.testing.assert_allclose(fd,jax.grad(solve_d)(1.),rtol=2e-6)
    with pytest.raises(StepFailure,match='rejected'):
        checked_linear_step(mass,stiffness,h,dt,max_steps=0)


def test_positive_entropy_discrete_gradient_and_invariants():
    grid,f=setup()
    initial=f*jnp.exp(.25*jnp.asarray(np.random.default_rng(3).normal(size=grid.size)))
    compiler=discrete_gradient_compiler(grid)
    state=initial
    for _ in range(4):
        step=discrete_gradient_step(grid,state,.1,compiled_residual=compiler,rtol=2e-12)
        assert step.minimum>0 and step.entropy_change>=-1e-12
        assert abs(step.entropy_identity_error)<2e-11
        state=step.f
    diagnostic=invariant_diagnostics(grid,state,initial)
    assert max(diagnostic[k] for k in ('number_error','energy_error','marginal_error'))<1e-10
    with pytest.raises(StepFailure,match='rejected'):
        discrete_gradient_step(grid,initial,.1,max_steps=0)
    with pytest.raises(ValueError,match='positive'):
        discrete_gradient_step(grid,jnp.zeros(grid.size),.1)


def test_divided_difference_stability_and_chain_rule():
    old=jnp.array([-7.,-.2,2.]); new=old+jnp.array([1e-12,1e-7,.4])
    gradient=entropy_discrete_gradient(old,new)
    np.testing.assert_allclose(gradient[:2],(-old-1)[:2],atol=1e-7)
    population_old=jnp.exp(old);population_new=jnp.exp(new)
    exact=-jnp.sum(population_new*new-population_old*old)
    np.testing.assert_allclose(jnp.vdot(gradient,population_new-population_old),exact,atol=3e-15)


def test_toroidal_exact_streaming_translation_norm_and_invariants():
    shape=(3,7,5,3,3); radius=np.linspace(1,1.5,3);u=np.linspace(-2,2,3);mu=np.linspace(.1,1.1,3)
    theta=np.arange(7)*2*np.pi/7;z=np.arange(5)*2*np.pi/5
    rr,tt,zz,uu,mm=np.meshgrid(radius,theta,z,u,mu,indexing='ij')
    initial=np.cos(tt+zz);dt=.17
    state=toroidal_stream(initial,shape,radius,u,mu,dt)
    exact=np.cos(tt+zz-dt*(uu/rr+uu**2+mm/rr))
    np.testing.assert_allclose(np.asarray(state).reshape(shape),exact,atol=1e-14)
    np.testing.assert_allclose(jnp.vdot(state,state),np.vdot(initial,initial),rtol=1e-14)
    np.testing.assert_allclose(np.asarray(state).reshape(shape).sum(axis=(1,2)),initial.sum(axis=(1,2)),atol=1e-13)


def test_nonfinite_inputs_reject_without_repair():
    grid,f=setup();mass=grid.weights*f
    with pytest.raises(ValueError):
        checked_linear_step(mass,lambda v:v,jnp.ones(grid.size),float('nan'))
    with pytest.raises(ValueError):
        discrete_gradient_step(grid,f,float('nan'))
    with pytest.raises(ValueError,match='nonnegative'):
        mobility_action(grid,f,jnp.ones(grid.size),collision_strength=-1.)
