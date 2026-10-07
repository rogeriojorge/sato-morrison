import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import pytest
from sato_morrison.collisions import uniform_grid,cartesian_grid,mobility_action,dense_mobility
from sato_morrison.geometry import Field
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


@pytest.mark.parametrize('nonuniform',[False,True])
def test_matrix_free_nonlinear_step_matches_dense_reference(nonuniform):
    if nonuniform:
        grid=cartesian_grid(np.linspace(.8,1.2,3),np.linspace(-.2,.2,3),
            np.linspace(.1,.5,3),np.linspace(-2,2,3),np.array([.1,1.1]),
            Field('mirror',amplitude=.15),compact=True)
    else:
        grid=uniform_grid(np.arange(5)*2*np.pi/5,np.linspace(-2,2,3),
                          np.linspace(.1,1.1,3),compact=True)
    initial=jnp.exp(-grid.energy+.12*jnp.asarray(np.random.default_rng(7).normal(size=grid.size)))
    reference=discrete_gradient_step(grid,initial,.04,rtol=2e-12,chunk_size=64)
    answer=discrete_gradient_step(grid,initial,.04,method='krylov',rtol=2e-12,chunk_size=64)
    np.testing.assert_allclose(answer.f,reference.f,rtol=3e-10,atol=1e-12)
    assert answer.linear_iterations>0 and answer.residual_metric=='entropy_population'
    assert answer.entropy_change>0 and abs(answer.entropy_identity_error)<2e-12
    diagnostic=invariant_diagnostics(grid,answer.f,initial)
    assert max(diagnostic[k] for k in ('number_error','energy_error','marginal_error'))<1e-10


def test_matrix_free_direction_and_visible_linear_failure():
    grid,f=setup();initial=f*jnp.exp(.25*jnp.asarray(np.random.default_rng(3).normal(size=grid.size)))
    old=jnp.log(initial);new=old+.01*jnp.sin(jnp.arange(grid.size));dt=.1
    residual,jacobian=discrete_gradient_compiler(grid)
    _,correction=discrete_gradient_compiler(grid,method='krylov')
    direction,_,relative,converged=correction(new,old,dt,1e-11)
    expected=np.linalg.solve(np.asarray(jacobian(new,old,dt)),-np.asarray(residual(new,old,dt)))
    assert bool(converged) and float(relative)<1e-11
    np.testing.assert_allclose(direction,expected,rtol=2e-9,atol=1e-11)
    # The returned correction must solve the independently materialized Jacobian.
    np.testing.assert_allclose(jacobian(new,old,dt)@direction,-residual(new,old,dt),rtol=2e-9,atol=1e-12)
    with pytest.raises(StepFailure,match='GMRES rejected'):
        discrete_gradient_step(grid,initial,1.,method='krylov',linear_restart=1,
            linear_max_restarts=1,linear_rtol=1e-12)
    with pytest.raises(ValueError,match='unsupported'):
        discrete_gradient_step(grid,initial,.1,method='other')


def test_uniform_fixed_density_D_and_B_derivatives_against_rebuilt_grids():
    x=np.arange(5)*2*np.pi/5;u=np.linspace(-2,2,3);mu=np.linspace(.1,1.1,3)
    base=uniform_grid(x,u,mu,magnetic_field=1.)
    # At fixed physical n0, B f0 is constant; both M and the density marginal
    # remain fixed while the uniform weak stiffness scales as D/B**2.
    xx,uu,mm=np.meshgrid(x,u,mu,indexing='ij')
    shape=np.exp(-.5*uu.ravel()**2-.2*mm.ravel())
    density=float(jnp.sum(base.weights*jnp.asarray(shape))/(2*np.pi))
    f=jnp.asarray(shape/density);mass=base.weights*f
    h=jnp.asarray(np.cos(xx.ravel())*uu.ravel())
    stiffness=lambda value:mobility_action(base,f,value)
    def differentiable(parameters):
        D,B=parameters
        value=linear_step(mass,lambda v:D/B**2*stiffness(v),h,.2,rtol=1e-13).x
        return jnp.vdot(mass*h,value)
    def rebuilt(D,B):
        grid=uniform_grid(x,u,mu,magnetic_field=B)
        distribution=jnp.asarray(shape/(density*B))
        nodal_mass=grid.weights*distribution
        np.testing.assert_allclose(jnp.sum(nodal_mass)/(2*np.pi),1.,rtol=2e-15)
        value=checked_linear_step(nodal_mass,lambda v:mobility_action(grid,distribution,v,
            collision_strength=D),h,.2,rtol=1e-13).x
        return float(jnp.vdot(nodal_mass*h,value))
    point=jnp.array([.7,1.3]);ad=np.asarray(jax.grad(differentiable)(point))
    assert np.all(np.abs(ad)>1e-4)
    errors=[]
    for eps in (1e-2,3e-3,1e-3,3e-4,1e-4):
        fd=np.array([(rebuilt(.7+eps,1.3)-rebuilt(.7-eps,1.3))/(2*eps),
            (rebuilt(.7,1.3+eps)-rebuilt(.7,1.3-eps))/(2*eps)])
        errors.append(np.max(np.abs((fd-ad)/ad)))
    assert errors[-1]<2e-7 and errors[-1]<errors[0]/100


def test_matrix_free_thermal_tails_and_accumulated_conservation():
    nodes,weights=np.polynomial.legendre.leggauss(9)
    u,wu=4*nodes,4*weights;mu,wm=10*(nodes+1),10*weights
    axes=[np.linspace(.8,1.2,3),np.linspace(-.2,.2,3),np.linspace(.1,.5,3)]
    grid=cartesian_grid(*axes,u,mu,Field('mirror',amplitude=.15),
                        velocity_weights=(wu,wm),compact=True)
    x,y,z,uu,mm=np.meshgrid(*axes,u,mu,indexing='ij')
    initial=jnp.exp(-grid.energy-.2*mm.ravel()+.1*jnp.sin(x.ravel())*uu.ravel()+.04*y.ravel()*mm.ravel())
    assert float(initial.min())<1e-12
    compiler=discrete_gradient_compiler(grid,method='krylov',collision_strength=.1)
    state=initial
    for _ in range(3):
        step=discrete_gradient_step(grid,state,.02,collision_strength=.1,
            method='krylov',compiled_residual=compiler,rtol=1e-11)
        assert step.entropy_change>0 and step.minimum>0
        assert abs(step.entropy_identity_error)<1e-11
        state=step.f
    diagnostics=invariant_diagnostics(grid,state,initial)
    assert max(diagnostics[k] for k in ('number_error','energy_error','marginal_error'))<1e-10
