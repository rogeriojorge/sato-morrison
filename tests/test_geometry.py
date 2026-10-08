"""Independent identities for the continuous vacuum bracket and coordinate map."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sato_morrison.geometry import (
    Field, chart_transform, common_chart_action, energy_mu, field_data,
    field_vector, poisson_eta, poisson_mu, to_eta, to_mu, validate_geometry,
)

jax.config.update("jax_enable_x64", True)

FIELDS = [Field("uniform", direction=(1, 2, 3)), Field("mirror", amplitude=0.12),
          Field("toroidal"), Field("dipole"),
          Field("nonaxisymmetric", amplitude=0.07)]
POINTS = [jnp.array([1.2, 0.4, 0.8, 0.3, 0.6]),
          jnp.array([0.8, -0.5, 1.1, -0.7, 0.2])]


@pytest.mark.parametrize("field", FIELDS)
def test_vacuum_and_independent_finite_differences(field):
    for z in POINTS:
        x = z[:3]
        derivative = np.asarray(jax.jacfwd(lambda y: field_vector(y, field))(x))
        curl = np.array([derivative[2, 1] - derivative[1, 2],
                         derivative[0, 2] - derivative[2, 0],
                         derivative[1, 0] - derivative[0, 1]])
        assert abs(np.trace(derivative)) < 3e-14
        assert np.linalg.norm(curl) < 3e-14
        errors = []
        for step in (0.02, 0.01, 0.005):
            directions = np.eye(3) * step
            fd = np.stack([(np.asarray(field_vector(x + e, field))
                            - np.asarray(field_vector(x - e, field))) / (2 * step)
                           for e in directions], axis=1)
            errors.append(np.linalg.norm(fd - derivative))
        # Uniform/polynomial fields are already exact at rounding; rational ones converge.
        if errors[0] > 1e-10:
            assert errors[2] < 0.27 * errors[1] < 0.08 * errors[0]
        _, _, b, _, curlb = field_data(x, field)
        assert abs(float(jnp.dot(b, curlb))) < 3e-14


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("charge", [-1.4, 0.8])
def test_bracket_measure_jacobi_and_chart(field, charge):
    mass = 1.7
    for z in POINTS:
        validate_geometry(z, field, mass, charge)
        zeta = to_eta(z, field)
        np.testing.assert_allclose(to_mu(zeta, field), z, atol=1e-14)
        transform = chart_transform(z, field)
        np.testing.assert_allclose(jax.jacfwd(lambda w: to_eta(w, field))(z), transform,
                                   atol=2e-14)
        for chart, state, function in (("mu", z, poisson_mu), ("eta", zeta, poisson_eta)):
            tensor_fn = lambda w: function(w, field, mass, charge)
            tensor = tensor_fn(state)
            np.testing.assert_allclose(tensor + tensor.T, 0, atol=2e-14)
            if chart == "mu":
                casimir_grad = jnp.array([0., 0., 0., 0., 1.])
                rho = lambda w: jnp.linalg.norm(field_vector(w[:3], field))
            else:
                casimir_grad = jax.grad(lambda w: to_mu(w, field)[4])(state)
                rho = lambda w: jnp.array(1.)
            np.testing.assert_allclose(tensor @ casimir_grad, 0, atol=3e-14)
            density_derivative = jax.jacfwd(lambda w: rho(w) * tensor_fn(w))(state)
            liouville = jnp.einsum("iji->j", density_derivative)
            np.testing.assert_allclose(liouville, 0, atol=2e-13)
            derivative = jax.jacfwd(tensor_fn)(state)
            first = jnp.einsum("il,jkl->ijk", tensor, derivative)
            jacobi = first + jnp.transpose(first, (1, 2, 0)) + jnp.transpose(first, (2, 0, 1))
            np.testing.assert_allclose(jacobi, 0, atol=8e-13)
        grad_mu = jax.grad(lambda w: energy_mu(w, field, mass, charge))(z)
        grad_eta = jax.grad(lambda w: energy_mu(to_mu(w, field), field, mass, charge))(zeta)
        velocity_eta = common_chart_action(z, grad_mu, field, mass, charge)
        np.testing.assert_allclose(velocity_eta,
                                   poisson_eta(zeta, field, mass, charge) @ grad_eta,
                                   atol=3e-14)
        velocity_mu = poisson_mu(z, field, mass, charge) @ grad_mu
        np.testing.assert_allclose(velocity_eta[4],
                                   z[4] * jnp.dot(velocity_mu[:3], field_data(z[:3], field)[3]),
                                   atol=3e-14)


def test_toroidal_flow_and_uniform_electric_drift():
    field = Field("toroidal", strength=1.3)
    z = jnp.array([1.2, 0.7, 0.0, 0.6, 0.4])
    mass, charge = 1.4, -0.8
    velocity = poisson_mu(z, field, mass, charge) @ jax.grad(
        lambda w: energy_mu(w, field, mass, charge))(z)
    radius = jnp.linalg.norm(z[:2])
    np.testing.assert_allclose(jnp.dot(z[:2], velocity[:2]), 0, atol=1e-14)
    np.testing.assert_allclose(velocity[2],
        (mass * z[3]**2 + z[4] * field.strength / radius) / (charge * field.strength),
        atol=1e-14)
    np.testing.assert_allclose(velocity[3:], 0, atol=1e-14)
    uniform = Field("uniform", strength=2)
    potential = lambda x: -0.3 * x[0]
    gradient = jax.grad(lambda w: energy_mu(w, uniform, mass, charge, potential))(z)
    flow = poisson_mu(z, uniform, mass, charge) @ gradient
    np.testing.assert_allclose(flow[:3], [0, -0.15, z[3]], atol=1e-14)


def test_invalid_geometry_rejected():
    for kwargs in ({"kind": "current_carrying"}, {"strength": 0}, {"length": -1},
                   {"direction": (0, 0, 0)}, {"amplitude": np.nan}):
        with pytest.raises(ValueError):
            Field(**kwargs)
    invalid = [(np.zeros(3), Field("dipole")), (np.zeros(3), Field("toroidal")),
               ([np.nan, 1, 1], Field()), ([1, 1, 1, 0, -1], Field()),
               ([np.sqrt(2.), 0, 0], Field("mirror", amplitude=1))]
    for position, field in invalid:
        with pytest.raises(ValueError):
            validate_geometry(position, field)
    for kwargs in ({"mass": 0}, {"charge": 0}, {"min_field": 0}):
        with pytest.raises(ValueError):
            validate_geometry([1, 1, 1], Field(), **kwargs)
    with pytest.raises(ValueError):
        validate_geometry([1, 1, 1], lambda x: x)


def test_random_vacuum_samples_and_measure_quadrature():
    rng = np.random.default_rng(716)
    positions = rng.uniform([0.6, -0.6, 0.4], [1.4, 0.6, 1.2], size=(8, 3))
    nodes, weights = np.polynomial.legendre.leggauss(12)
    eta = 3 * (nodes + 1)
    eta_weights = 3 * weights
    for field in FIELDS:
        validate_geometry(positions, field)
        derivatives = np.asarray(jax.vmap(jax.jacfwd(lambda x: field_vector(x, field)))(
            jnp.asarray(positions)))
        np.testing.assert_allclose(np.trace(derivatives, axis1=1, axis2=2), 0, atol=4e-14)
        np.testing.assert_allclose(derivatives, derivatives.swapaxes(1, 2), atol=4e-14)
        for x in positions[:2]:
            magnitude = float(jnp.linalg.norm(field_vector(jnp.asarray(x), field)))
            mu = eta / magnitude
            mu_weights = eta_weights / magnitude
            np.testing.assert_allclose(np.dot(eta_weights, np.exp(-eta)),
                                       magnitude * np.dot(mu_weights, np.exp(-mu*magnitude)),
                                       atol=2e-15)
            states = [jnp.r_[x, 0.2, 0.3], jnp.r_[x + np.array([0.03, -0.02, 0.04]), -0.4, 0.8]]
            observable = lambda w: jnp.sin(w[0]) * w[3]**2 + w[1] * w[4]
            common = []
            transformed = []
            for state in states:
                grad = jax.grad(observable)(state)
                common.append(common_chart_action(state, grad, field))
                transformed.append(poisson_eta(to_eta(state, field), field) @ jax.grad(
                    lambda v: observable(to_mu(v, field)))(to_eta(state, field)))
            np.testing.assert_allclose(common[0] - common[1],
                                       transformed[0] - transformed[1], atol=1e-13)


def _mixed_dipole_candidate(z,mass,charge,strength):
    x,y,zz,u,_=z;radius2=x*x+y*y
    magnitude=strength*jnp.sqrt(radius2+4*zz*zz)/(radius2+zz*zz)**2
    return mass*u/magnitude-charge*jnp.arctan2(y,x)*zz*(3*radius2+8*zz*zz)/(radius2+2*zz*zz)


def _independent_mixed_dipole_data(position,charge,strength):
    x,y,z=np.asarray(position);radius2=x*x+y*y;r2=radius2+z*z;d=radius2+4*z*z
    magnitude=strength*np.sqrt(d)/r2**2
    unit=strength*np.array([3*x*z,3*y*z,2*z*z-radius2])/r2**2.5/magnitude
    grad=magnitude*(np.array([x,y,4*z])/d-4*np.array([x,y,z])/r2)
    den=radius2+2*z*z;theta=np.arctan2(y,x)
    coefficient=-z*(3*radius2+8*z*z)/den
    gradcoefficient=np.array([4*x*z**3/den**2,4*y*z**3/den**2,-3-6*z*z/den+8*z**4/den**2])
    gradphi=charge*(coefficient*np.array([-y/radius2,x/radius2,0.])+theta*gradcoefficient)
    return magnitude,unit,grad,gradphi,coefficient


@pytest.mark.parametrize('charge',[.7,-1.3])
def test_mixed_dipole_local_common_action_is_velocity_independent(charge):
    mass,strength=1.7,1.3;field=Field('dipole',strength=strength)
    for position in ([1.,.1,.3],[.85,-.15,.4],[1.15,.18,.15]):
        B,b,g,gradphi,_=_independent_mixed_dipole_data(position,charge,strength)
        np.testing.assert_allclose(np.cross(b,g)@gradphi,charge*np.dot(b,g),rtol=2e-14,atol=2e-14)
        expected=np.r_[b/B+np.cross(b,gradphi)/(charge*B),-b@gradphi/mass,0.]
        actions=[]
        for u,mu in [(-.8,.15),(.2,.6),(1.1,1.)]:
            state=jnp.array([*position,u,mu])
            gradient=jax.grad(lambda z:_mixed_dipole_candidate(z,mass,charge,strength))(state)
            np.testing.assert_allclose(gradient,np.r_[-mass*u*g/B**2+gradphi,mass/B,0.],rtol=2e-14,atol=2e-14)
            actual=np.asarray(common_chart_action(state,gradient,field,mass,charge));actions.append(actual)
            np.testing.assert_allclose(actual,expected,rtol=2e-14,atol=2e-14)
        np.testing.assert_allclose(np.asarray(actions)-actions[0],0.,atol=2e-14)


def test_mixed_dipole_moment_locality_square_and_ideal_controls():
    field=Field('dipole');left=jnp.array([1.,0.,.3,.4,.2])
    same=jnp.array([1.,0.,.3,-.6,.7]);separated=jnp.array([1.1,.05,.35,-.6,.7])
    def observable_action(z,power=1):
        gradient=jax.grad(lambda w:_mixed_dipole_candidate(w,1.,1.,1.)**power)(z)
        return np.asarray(common_chart_action(z,gradient,field))
    def energy_action(z):
        return np.asarray(common_chart_action(z,jax.grad(lambda w:energy_mu(w,field))(z),field))
    def projected_quadratic(right,power=1):
        delta=observable_action(left,power)-observable_action(right,power)
        xi=energy_action(left)-energy_action(right)
        assert xi@xi>0
        projected=delta-xi*(xi@delta)/(xi@xi)
        return projected[:3]@projected[:3]
    assert projected_quadratic(same)<1e-25
    # Form an independent nonlocal expected value from analytic common actions
    # and energy flow, including its eta component before projection.
    analytic_h=[];analytic_energy=[]
    for state in (left,separated):
        B,b,g,gradphi,_=_independent_mixed_dipole_data(state[:3],1.,1.)
        u,mu=map(float,state[3:]);t=np.cross(b,g);s=b@g
        analytic_h.append(np.r_[b/B+np.cross(b,gradphi)/B,-b@gradphi,0.])
        analytic_energy.append(np.r_[u*b+(u*u/B**2+mu/B)*t,-mu*s,mu*u*s])
    xi=analytic_energy[0]-analytic_energy[1];delta=analytic_h[0]-analytic_h[1]
    projected=delta-xi*(xi@delta)/(xi@xi);expected=projected[:3]@projected[:3]
    assert expected>.1
    np.testing.assert_allclose(projected_quadratic(separated),expected,rtol=3e-14,atol=3e-14)
    # A null moment does not certify its full marginal: this dipole h^2 is not
    # a local null. This is a concrete counterexample, not a universal F(h) claim.
    assert projected_quadratic(same,power=2)>1e-4
    gradient=jax.grad(lambda w:_mixed_dipole_candidate(w,1.,1.,1.))(separated)
    ideal=float(gradient@poisson_mu(separated,field)@jax.grad(lambda w:energy_mu(w,field))(separated))
    _,b,_,gradphi,_=_independent_mixed_dipole_data(separated[:3],1.,1.)
    np.testing.assert_allclose(ideal,float(separated[3])*b@gradphi,rtol=3e-14,atol=3e-14)
    assert abs(ideal)>.01


@pytest.mark.parametrize('charge',[.7,-1.3])
def test_mixed_dipole_branch_has_periodic_integrability_obstruction(charge):
    R,z,strength=1.,.3,1.3;angles,weights=np.polynomial.legendre.leggauss(20)
    angles=np.pi*(angles+1);weights=np.pi*weights;angular_derivatives=[]
    for theta in angles:
        position=jnp.array([R*np.cos(theta),R*np.sin(theta),z])
        _,_,b,g,_=field_data(position,Field('dipole',strength=strength))
        t=np.cross(np.asarray(b),np.asarray(g));etheta=np.array([-np.sin(theta),np.cos(theta),0.])
        angular=t@etheta
        assert abs(angular)>1e-3
        angular_derivatives.append(charge*R*float(b@g)/angular)
    coefficient=_independent_mixed_dipole_data([R,0.,z],charge,strength)[4]
    integral=weights@np.asarray(angular_derivatives)
    np.testing.assert_allclose(integral,2*np.pi*charge*coefficient,rtol=3e-14,atol=3e-14)
    # A periodic differentiable phi would have zero integral of dphi/dtheta.
    assert abs(integral)>1.
