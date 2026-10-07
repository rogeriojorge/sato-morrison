"""Independent orbit and equilibrium checks with analytic comparisons."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from sato_morrison.geometry import Field,poisson_mu,energy_mu
from sato_morrison.reference import constrained_equilibrium

jax.config.update('jax_enable_x64',True)


def test_harmonic_mirror_orbit_fourth_order():
    field=Field('mirror',strength=1.,amplitude=.4)
    initial=np.array([0.,0.,0.,.4,.5])
    omega=np.sqrt(.4)
    flow=jax.jit(lambda z:poisson_mu(z,field)@jax.grad(lambda zz:energy_mu(zz,field))(z))
    errors=[]
    for steps in (40,80,160):
        h=2*np.pi/omega/steps
        y=initial.copy()
        for _ in range(steps):
            a=np.asarray(flow(y));b=np.asarray(flow(y+h*a/2))
            c=np.asarray(flow(y+h*b/2));d=np.asarray(flow(y+h*c))
            y+=h*(a+2*b+2*c+d)/6
        errors.append(np.linalg.norm(y-initial))
        assert y[4]==initial[4]
        assert abs(float(energy_mu(y,field)-energy_mu(initial,field)))<2e-6
    assert errors[0]/errors[1]>15
    assert errors[1]/errors[2]>15


def test_full_marginal_equilibrium_multiplier():
    labels=np.tile(np.arange(3),8)
    e=np.repeat(np.linspace(.1,3.,8),3)+.4*labels
    w=np.linspace(.5,1.2,24)
    beta_exact=1.7
    f=np.exp(-beta_exact*e+.3*labels**2)
    beta, candidate=constrained_equilibrium(e,w,labels,f)
    np.testing.assert_allclose(beta,beta_exact,rtol=1e-12)
    np.testing.assert_allclose(candidate,f,rtol=1e-12)
    perturbed=f*np.exp(.15*np.sin(np.arange(24)))
    beta,candidate=constrained_equilibrium(e,w,labels,perturbed)
    np.testing.assert_allclose(np.bincount(labels,weights=w*candidate),np.bincount(labels,weights=w*perturbed),atol=1e-13)
    np.testing.assert_allclose(np.dot(w*candidate,e),np.dot(w*perturbed,e),atol=1e-12)
    assert -np.dot(w*candidate,np.log(candidate)) >= -np.dot(w*perturbed,np.log(perturbed))
    with pytest.raises(ValueError):
        constrained_equilibrium(e,w,labels,-f)


def test_nonuniform_zero_set_direction_dependence_and_weak_limit():
    from sato_morrison.geometry import common_chart_action
    from sato_morrison.collisions import _kernel
    field=Field('dipole')
    center=jnp.array([1.,.2,.3,.6,.4])
    flow=jax.jit(lambda z:common_chart_action(z,jax.grad(lambda p:energy_mu(p,field))(z),field))
    observable=jax.jit(lambda z:common_chart_action(z,jax.grad(lambda p:p[0]**2+p[3]**3+p[4]*p[0])(z),field))
    directional=[]
    weak=[]
    for eps in [1e-2,1e-3,1e-4]:
        kernels=[];quadratic=[]
        for axis in [3,4]:
            shifted=center.at[axis].add(eps)
            xi=np.asarray(flow(shifted)-flow(center))
            kernel=_kernel(xi[None])[0]
            delta=np.asarray(observable(shifted)-observable(center))
            kernels.append(kernel);quadratic.append(delta@kernel@delta)
            np.testing.assert_allclose(kernel@xi,0,atol=1e-14)
        directional.append(np.linalg.norm(kernels[0]-kernels[1]))
        weak.append(max(quadratic))
    assert min(directional)>.1  # no direction-independent projector extension
    assert weak[-1]<weak[0]*2e-4  # bounded pair weak integrand vanishes quadratically
    with pytest.raises(ValueError,match='zero pair energy direction'):
        _kernel(np.zeros((1,5)))


def test_uniform_field_scaling_at_fixed_physical_eta_and_zero_diffusion():
    from sato_morrison.collisions import uniform_grid,linear_rhs
    x=np.arange(5)*2*np.pi/5;u=np.linspace(-2,2,3);eta=np.linspace(.1,1.3,3)
    actions=[]
    h=np.cos(x)[:,None,None]*np.broadcast_to(u[None,:,None]**2,(5,3,3))
    for field_strength in [1.,2.,4.]:
        grid=uniform_grid(x,u,eta/field_strength,magnetic_field=field_strength)
        f=jnp.exp(-grid.energy)
        actions.append(np.asarray(linear_rhs(grid,f,h.ravel())))
        np.testing.assert_array_equal(linear_rhs(grid,f,h.ravel(),collision_strength=0.),np.zeros(grid.size))
    np.testing.assert_allclose(actions[1],actions[0]/4,atol=2e-14)
    np.testing.assert_allclose(actions[2],actions[0]/16,atol=2e-14)


def test_toroidal_angular_weighted_invariants_account_for_tiny_nullspace():
    from sato_morrison.collisions import toroidal_grid,mobility_action,dense_mobility
    r=np.linspace(1,1.5,3);theta=np.arange(3)*2*np.pi/3;z=theta
    u=np.linspace(-2,2,3);mu=np.linspace(.1,1.1,3)
    grid=toroidal_grid(r,theta,z,u,mu)
    f=jnp.exp(-grid.energy)
    rr,tt,zz,uu,mm=np.meshgrid(r,theta,z,u,mu,indexing='ij')
    labels=np.repeat(np.arange(27),9)
    vectors=[(labels==i).astype(float) for i in range(27)]
    for angle in theta:
        selector=(tt.ravel()==angle).astype(float)
        vectors.extend([selector*(np.asarray(grid.mu_index)==i) for i in range(3)])
        vectors.extend([selector*np.asarray(grid.energy),selector*(rr*uu).ravel()])
    known=np.array(vectors).T
    assert np.linalg.matrix_rank(known)==39
    action=jax.jit(lambda h:mobility_action(grid,f,h))
    for v in known.T:
        np.testing.assert_allclose(action(v),0.,atol=2e-13)
    gram=dense_mobility(grid,f)
    mass=np.asarray(grid.weights*f)
    scaled=gram/np.sqrt(mass[:,None]*mass[None,:])
    eigenvalues=np.linalg.eigvalsh(scaled)
    tolerance=1e-10*max(np.linalg.norm(scaled,2),1.)
    assert np.count_nonzero(abs(eigenvalues)<tolerance)==39


@pytest.mark.parametrize('charge', [-1.3, 1.7])
def test_toroidal_continuum_null_family_and_streaming_intersection(charge):
    from sato_morrison.geometry import common_chart_action
    from sato_morrison.collisions import _kernel
    field = Field('toroidal', strength=1.4)
    mass = 1.2

    def coordinates(state):
        return jnp.hypot(state[0], state[1]), jnp.arctan2(state[1], state[0])

    def collision_invariant(state):
        radius, theta = coordinates(state)
        energy = energy_mu(state, field, mass, charge)
        return (state[0]*jnp.sin(state[2]) + jnp.sin(theta)*state[4]**2
                + jnp.cos(theta)*energy + jnp.sin(2*theta)*mass*radius*state[3])

    def joint_invariant(state):
        radius, _ = coordinates(state)
        return radius**3 + state[4]**2 + .7*energy_mu(state, field, mass, charge) + .3*mass*radius*state[3]

    gradient = jax.grad(collision_invariant)
    energy_gradient = jax.grad(lambda state: energy_mu(state, field, mass, charge))
    for radius, theta, height in [(1.1,.3,.2), (1.4,1.7,-.4), (2.2,-.8,.7)]:
        states = [jnp.array([radius*np.cos(theta), radius*np.sin(theta), height, u, mu])
                  for u, mu in [(-.7,.2), (.3,.8), (1.1,.4)]]
        actions, flows = [], []
        for state in states:
            grad = gradient(state)
            action = common_chart_action(state, grad, field, mass, charge)
            h_r = grad[0]*np.cos(theta)+grad[1]*np.sin(theta)
            # Independent cylindrical spatial components of J grad(h).
            cylindrical = np.array([radius*grad[2]/(charge*1.4), grad[3]/mass,
                (-radius*h_r+state[3]*grad[3])/(charge*1.4)])
            rotated = np.array([np.cos(theta)*action[0]+np.sin(theta)*action[1],
                                -np.sin(theta)*action[0]+np.cos(theta)*action[1], action[2]])
            np.testing.assert_allclose(rotated, cylindrical, atol=2e-14)
            actions.append(np.asarray(action))
            flows.append(np.asarray(common_chart_action(state,energy_gradient(state),field,mass,charge)))
            stream = poisson_mu(state,field,mass,charge)@energy_gradient(state)
            np.testing.assert_allclose(jax.grad(joint_invariant)(state)@stream,0,atol=2e-14)
        for i,j in [(0,1),(0,2),(1,2)]:
            delta = actions[i]-actions[j]
            np.testing.assert_allclose(_kernel((flows[i]-flows[j])[None])[0]@delta,0,atol=2e-14)
        # An angularly weighted energy is a collision invariant but is transported.
        state = states[-1]
        weighted_energy = lambda point: jnp.sin(coordinates(point)[1])*energy_mu(point,field,mass,charge)
        stream = poisson_mu(state,field,mass,charge)@energy_gradient(state)
        assert abs(float(jax.grad(weighted_energy)(state)@stream)) > 1e-3


def test_mirror_bounce_integral_and_turning_point():
    from scipy.integrate import quad, solve_ivp
    from scipy.optimize import brentq
    mass, charge, mu, speed = 1.7, -1.1, .5, .4
    field=Field('mirror',strength=1.3,amplitude=.4)
    initial=np.array([0.,0.,0.,speed,mu])
    # Independent scalar energy and field on the symmetry axis.
    energy=.5*mass*speed**2+mu*field.strength
    potential=lambda z:mu*(field.strength+field.amplitude*z*z)
    turning=brentq(lambda z:potential(z)-energy,0.,2.,xtol=1e-14)
    # z=z_turn*(1-s^2) removes the integrable endpoint singularity.
    quarter,error=quad(lambda s:2*turning/np.sqrt(
        2*mu*field.amplitude*turning**2*(2-s*s)/mass),0.,1.,epsabs=1e-12,epsrel=1e-12)
    period=4*quarter
    flow=jax.jit(lambda z:poisson_mu(z,field,mass,charge)@jax.grad(
        lambda state:energy_mu(state,field,mass,charge))(z))
    def event(t,state):
        return state[3]
    deviations=[]
    for tolerance in (1e-8,1e-11):
        orbit=solve_ivp(lambda t,state:np.asarray(flow(jnp.asarray(state))),
            (0,2.1*period),initial,method='DOP853',events=event,
            rtol=tolerance,atol=tolerance*.01,dense_output=True)
        assert orbit.success and len(orbit.t_events[0])==4
        measured=2*(orbit.t_events[0][1]-orbit.t_events[0][0])
        measured_turn=orbit.y_events[0][0,2]
        deviations.append(max(abs(measured/period-1),abs(measured_turn/turning-1)))
        samples=orbit.sol(np.linspace(0,period,81)).T
        energies=np.asarray(jax.vmap(lambda z:energy_mu(z,field,mass,charge))(jnp.asarray(samples)))
        assert np.max(abs(energies-energy))/energy<20*tolerance
        np.testing.assert_array_equal(samples[:,4],np.full(81,mu))
    assert deviations[1]<2e-10 and deviations[1]<deviations[0]/30
    assert error<1e-11
