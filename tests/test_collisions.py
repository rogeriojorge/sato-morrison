import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest
from sato_morrison.collisions import (uniform_grid, toroidal_grid, mobility_action,
    dense_mobility, linear_rhs, nonlinear_rhs)


def small_grid(**options):
    return uniform_grid(np.arange(5)*2*np.pi/5, np.linspace(-2,2,3),
                        np.linspace(.15,1.3,3), **options)


def equilibrium(grid):
    return jnp.exp(-grid.energy-.3*jnp.tile(jnp.array([.15,.725,1.3]),grid.size//3))


def test_weak_uniform_exact_rate_nullspace_and_normalization():
    grid = small_grid(magnetic_field=1.7, charge=-1.3)
    f = jnp.exp(-grid.energy)
    velocity = np.asarray(f).reshape(grid.shape)[0]
    mass = np.asarray(grid.weights).reshape(grid.shape)[0]*velocity
    hvel = np.arange(9).reshape(3,3)
    hvel = hvel-np.sum(mass*hvel)/mass.sum()
    mode = np.cos(np.arange(5)*2*np.pi/5)[:,None,None]
    h = (mode*hvel).ravel()
    n0 = mass.sum()/(2*np.pi/5)
    rate = .7*n0/(1.3*1.7)**2
    np.testing.assert_allclose(linear_rhs(grid,f,h,collision_strength=.7),-rate*h,atol=2e-13)
    homogeneous = np.broadcast_to(hvel,grid.shape).ravel()
    np.testing.assert_allclose(linear_rhs(grid,f,homogeneous),0,atol=1e-14)
    density = np.broadcast_to(mode,grid.shape).ravel()
    np.testing.assert_allclose(linear_rhs(grid,f,density),0,atol=1e-14)
    np.testing.assert_allclose(mobility_action(grid,f,h,collision_strength=3),
                               3*mobility_action(grid,f,h),atol=1e-13)


def test_pair_kernel_and_dense_matrix_unmodified():
    grid = small_grid()
    f = equilibrium(grid)
    kernel = np.asarray(grid.kernels)
    np.testing.assert_allclose(kernel,kernel.swapaxes(1,2),atol=1e-15)
    assert np.linalg.eigvalsh(kernel).min() >= -1e-14
    flow = np.asarray(grid.action(grid.energy))
    xi = flow[np.asarray(grid.left)]-flow[np.asarray(grid.right)]
    np.testing.assert_allclose(np.einsum('pab,pb->pa',kernel,xi),0,atol=1e-14)
    dense = dense_mobility(grid,f)
    np.testing.assert_allclose(dense,dense.T,atol=1e-14)
    assert np.linalg.eigvalsh(dense).min() > -1e-12
    rng = np.random.default_rng(51)
    for _ in range(5):
        h = rng.normal(size=grid.size)
        np.testing.assert_allclose(mobility_action(grid,f,h),dense@h,atol=2e-13,rtol=1e-12)
    invariants = [np.ones(grid.size), np.asarray(grid.energy)]
    invariants += [(np.asarray(grid.mu_index)==i).astype(float) for i in range(3)]
    for invariant in invariants:
        np.testing.assert_allclose(dense@invariant,0,atol=2e-13)
    # Verify pair counting independently from the ordered scalar quadratic form.
    h = rng.normal(size=grid.size)
    ah = np.asarray(grid.action(h)); ff=np.asarray(f); total=0.
    for x in range(5):
        for i in range(9):
            for j in range(9):
                difference=ah[x*9+i]-ah[x*9+j]
                wv=np.asarray(grid.weights)[x*9:(x+1)*9]/(2*np.pi/5)
                total += .5*(2*np.pi/5)*wv[i]*wv[j]*ff[x*9+i]*ff[x*9+j]*np.dot(difference[:2],difference[:2])
    np.testing.assert_allclose(h@dense@h,total,rtol=2e-13)


def test_nonlinear_equilibrium_linearization_and_locality():
    grid=small_grid(); f=equilibrium(grid)
    np.testing.assert_allclose(nonlinear_rhs(grid,f),0,atol=1e-12)
    rng=np.random.default_rng(4); h=rng.normal(size=grid.size)
    expected=f*linear_rhs(grid,f,h)
    errors=[]
    for eps in (1e-2,1e-3,1e-4):
        fd=(nonlinear_rhs(grid,f*(1+eps*h))-nonlinear_rhs(grid,f*(1-eps*h)))/(2*eps)
        errors.append(float(jnp.linalg.norm(fd-expected)/jnp.linalg.norm(expected)))
    assert errors[-1] < 1e-7 and errors[-1] < errors[0]/1000
    assert np.all(np.asarray(grid.left)//9 == np.asarray(grid.right)//9)
    with pytest.raises(ValueError,match='unsupported'):
        small_grid(model='landau')


def test_finite_range_lifts_density_nullspace_and_retains_invariants():
    nx=5; x=np.arange(nx)*2*np.pi/nx
    distance=np.angle(np.exp(1j*(x[:,None]-x[None,:])))
    kernel=np.exp(-distance**2/(2*.8**2))
    kernel/=kernel.sum(axis=1)[0]*(2*np.pi/nx)
    grid=small_grid(model='sm_finite_range',spatial_kernel=kernel)
    f=jnp.exp(-grid.energy); h=np.broadcast_to(np.cos(x)[:,None,None],grid.shape).ravel()
    rhs=np.asarray(linear_rhs(grid,f,h))
    n0=float(jnp.sum(grid.weights*f)/(2*np.pi))
    khat=np.sum(kernel[0]*np.cos(x))*(2*np.pi/nx)
    np.testing.assert_allclose(rhs,-n0*(1-khat)*h,atol=2e-13)
    assert np.linalg.norm(rhs)>0
    for invariant in (np.ones(grid.size),grid.energy,(grid.mu_index==1).astype(float)):
        np.testing.assert_allclose(mobility_action(grid,f,invariant),0,atol=1e-13)


def test_toroidal_local_kernel_energy_mu_and_equilibrium():
    grid=toroidal_grid(np.linspace(1,1.5,3),np.arange(3)*2*np.pi/3,
        np.arange(3)*2*np.pi/3,np.linspace(-2,2,3),np.linspace(.1,1.1,3))
    f=jnp.exp(-grid.energy-.2*(grid.mu_index+1))
    np.testing.assert_allclose(nonlinear_rhs(grid,f),0,atol=3e-12)
    for invariant in (np.ones(grid.size),grid.energy,(grid.mu_index==1).astype(float)):
        np.testing.assert_allclose(mobility_action(grid,f,invariant),0,atol=2e-12)
    assert np.linalg.eigvalsh(np.asarray(grid.kernels)).min()>-1e-14


def test_pair_chunks_and_zero_strength_are_exact():
    grid=small_grid();f=equilibrium(grid)
    h=jnp.asarray(np.random.default_rng(17).normal(size=grid.size))
    expected=mobility_action(grid,f,h)
    for chunk in (1,7,128,1000):
        result=jax.jit(lambda value:mobility_action(grid,f,value,chunk_size=chunk))(h)
        np.testing.assert_allclose(result,expected,atol=3e-14,rtol=2e-13)
    np.testing.assert_array_equal(mobility_action(grid,f,h,collision_strength=0),np.zeros(grid.size))
    with pytest.raises(ValueError,match='positive pair'):
        mobility_action(grid,f,h,chunk_size=0)


def test_invalid_quadrature_nodes_fail():
    from sato_morrison.collisions import derivative_matrix,trapezoid_weights
    with pytest.raises(ValueError):
        derivative_matrix([0.,float('nan'),2.])
    with pytest.raises(ValueError):
        trapezoid_weights([1.])
    with pytest.raises(ValueError):
        trapezoid_weights([0.,0.])


def test_toroidal_extra_spatial_population_invariants_are_unmodified():
    grid=toroidal_grid(np.linspace(1,1.5,3),np.arange(3)*2*np.pi/3,
        np.arange(3)*2*np.pi/3,np.linspace(-2,2,3),np.linspace(.1,1.1,3))
    f=jnp.exp(-grid.energy)
    # Every function of R,theta has a local pair action independent of u,mu.
    rng=np.random.default_rng(7)
    phi=np.broadcast_to(rng.normal(size=(3,3,1,1,1)),grid.shape).ravel()
    action=np.asarray(grid.action(phi))
    pair_difference=action[np.asarray(grid.left)]-action[np.asarray(grid.right)]
    np.testing.assert_allclose(pair_difference,0,atol=1e-14)
    np.testing.assert_allclose(mobility_action(grid,f,phi),0,atol=1e-12)
    operator=jax.jit(lambda value:mobility_action(grid,f,value))
    for radial in range(3):
        for angular in range(3):
            for vertical in range(3):
                basis=np.zeros(grid.shape);basis[radial,angular,vertical]=1
                np.testing.assert_allclose(operator(basis.ravel()),0,atol=1e-12)


def test_direction_setup_rejects_exact_zero_and_matches_explicit_kernel():
    from sato_morrison.collisions import _complete_grid, _kernel, derivative_matrix
    derivative=derivative_matrix(np.linspace(-1,1,3))
    with pytest.raises(ValueError,match='zero pair energy direction'):
        _complete_grid((3,3,3),((0,derivative),(1,derivative)),
            np.zeros((3,3,3,5,2)),np.ones(3),np.ones(9),
            np.ones((3,3,3)),'zero energy flow')
    grid=toroidal_grid(np.linspace(1,1.5,3),np.arange(3)*2*np.pi/3,
        np.arange(3)*2*np.pi/3,np.linspace(-2,2,3),np.linspace(.1,1.1,3))
    flow=np.asarray(grid.action(grid.energy))
    xi=flow[np.asarray(grid.left)]-flow[np.asarray(grid.right)]
    np.testing.assert_allclose(np.asarray(grid.kernels),_kernel(xi),atol=2e-15,rtol=2e-14)
