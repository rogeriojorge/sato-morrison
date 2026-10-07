"""Continuous vacuum zero sets and local spatial-population constraints."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sato_morrison.geometry import (
    Field, common_chart_action, energy_mu, field_data, field_vector,
)

jax.config.update("jax_enable_x64", True)


def flow(state, field, mass, charge):
    gradient = jax.grad(lambda z: energy_mu(z, field, mass, charge))(state)
    return np.asarray(common_chart_action(state, gradient, field, mass, charge))


def explicit_flow(state, field, mass, charge):
    _, magnitude, b, gradient, _ = map(np.asarray, field_data(state[:3], field))
    u, mu = np.asarray(state[3:])
    transverse = np.cross(b, gradient)
    parallel = b @ gradient
    return np.r_[u*b + (mass*u*u + mu*magnitude)*transverse/(charge*magnitude**2),
                 -mu*parallel/mass, mu*u*parallel]


def action(state, observable, field, mass, charge):
    return np.asarray(common_chart_action(state, jax.grad(observable)(state),
                                         field, mass, charge))


def weak_factor(first, second, observable, field, mass, charge):
    direction = flow(first, field, mass, charge)-flow(second, field, mass, charge)
    assert direction @ direction > 0
    difference = action(first, observable, field, mass, charge)-action(
        second, observable, field, mass, charge)
    projected = difference-direction*(direction @ difference)/(direction @ direction)
    return projected[:3] @ projected[:3]


def scalar_potential(x, field):
    xx, yy, zz = x
    if field.kind == "mirror":
        return field.strength*zz+field.amplitude*(zz**3/3-zz*(xx**2+yy**2)/2)
    dipole = -field.strength*zz/jnp.dot(x, x)**1.5
    if field.kind == "nonaxisymmetric":
        dipole += field.amplitude*field.strength*xx*yy/field.length
    return dipole


def flux(x, field):
    radius_squared = x[0]**2+x[1]**2
    if field.kind == "dipole":
        return field.strength*radius_squared/jnp.dot(x, x)**1.5
    return (field.strength*radius_squared/2
            + field.amplitude*radius_squared*x[2]**2/2
            - field.amplitude*radius_squared**2/8)


@pytest.mark.parametrize("field", [Field("mirror", amplitude=.15), Field("dipole"),
                                   Field("nonaxisymmetric", amplitude=.03)])
@pytest.mark.parametrize("charge", [-.8, .8])
def test_vacuum_energy_flow_formula_and_velocity_immersion(field, charge):
    mass = 1.7
    points = [jnp.array([1., .2, .3]), jnp.array([.9, -.1, .4])]
    if field.kind == "dipole":
        points += [jnp.array([0., 0., 1.]), jnp.array([1., 0., 0.])]
    for x in points:
        z = jnp.r_[x, .6, .4]
        np.testing.assert_allclose(flow(z, field, mass, charge),
                                   explicit_flow(z, field, mass, charge), atol=2e-14)
        _, B, b, gradient, curl = map(np.asarray, field_data(x, field))
        t = np.cross(b, gradient);s = b @ gradient
        np.testing.assert_allclose(curl, t/B, atol=2e-14)
        # b dot V_X=u fixes the first velocity coordinate; the mu column is
        # nonzero and perpendicular to this row whenever grad B is nonzero.
        u, mu = np.asarray(z[3:])
        columns = np.stack([np.r_[b+2*mass*u*t/(charge*B**2), 0., mu*s],
                            np.r_[t/(charge*B), -s/mass, u*s]], axis=1)
        assert np.linalg.svd(columns, compute_uv=False)[-1] > 1e-3
        second = z.at[4].set(.9)
        assert np.linalg.norm(flow(z, field, mass, charge)-flow(second, field, mass, charge)) > 1e-3


def test_critical_mirror_distinct_state_weak_limit_is_not_unique():
    field = Field("mirror", strength=1.3, amplitude=.15)
    mass, charge = 1.7, -.8
    observable = lambda z: z[4]*z[0]
    expected = (.9-.3)**2/(charge*field.strength)**2
    for distance in [1e-2, 1e-3, 1e-4]:
        x_first = jnp.array([distance, 0., 0., .6, .3])
        z_first = jnp.array([0., 0., distance, .6, .3])
        x_limit = weak_factor(x_first, x_first.at[4].set(.9), observable, field, mass, charge)
        z_limit = weak_factor(z_first, z_first.at[4].set(.9), observable, field, mass, charge)
        assert x_limit < 1e-26
        np.testing.assert_allclose(z_limit,
            (.9-.3)**2/(charge*(field.strength+field.amplitude*distance**2))**2,
            rtol=2e-13)
    np.testing.assert_allclose(z_limit, expected, rtol=3e-9)
    critical = jnp.array([0., 0., 0., .6, .3])
    np.testing.assert_allclose(flow(critical, field, mass, charge),
                               flow(critical.at[4].set(.9), field, mass, charge), atol=0.)


@pytest.mark.parametrize("field", [Field("mirror", amplitude=.15), Field("dipole"),
                                   Field("nonaxisymmetric", amplitude=.03)])
def test_vacuum_scalar_potential_and_joint_spatial_invariants(field):
    mass, charge = 1.7, -.8
    x = jnp.array([1., .2, .3])
    np.testing.assert_allclose(jax.grad(lambda y: scalar_potential(y, field))(x),
                               field_vector(x, field), atol=2e-14)
    observable = lambda z: (jnp.sin(jnp.linalg.norm(field_vector(z[:3], field)))
                             + .2*scalar_potential(z[:3], field)**2)
    states = [jnp.r_[x, -.7, .2], jnp.r_[x, .4, .8], jnp.r_[x, 1.1, .5]]
    for state in states[1:]:
        np.testing.assert_allclose(action(state, observable, field, mass, charge),
                                   action(states[0], observable, field, mass, charge), atol=2e-14)
    # The complete criterion for spatial observables at a point with s!=0 is
    # (b cross grad B) dot grad phi=0. This Cartesian x function violates it.
    first = states[1];second = first.at[4].set(.2)
    assert weak_factor(first, second, lambda z: z[0], field, mass, charge) > 1e-9


@pytest.mark.parametrize("field", [Field("mirror", amplitude=.15), Field("dipole")])
@pytest.mark.parametrize("charge", [-.8, .8])
def test_poloidal_spatial_and_combined_flux_population_invariants(field, charge):
    mass = 1.7
    x = jnp.array([1., .2, .3])
    states = [jnp.r_[x, -.7, .2], jnp.r_[x, .4, .8], jnp.r_[x, 1.1, .5]]
    meridional = lambda z: .3*(z[0]**2+z[1]**2)*z[2]+jnp.sin(z[2])
    flux_observable = lambda z: jnp.sin(flux(z[:3], field))+.15*flux(z[:3], field)**2
    for observable in [meridional, flux_observable]:
        for state in states[1:]:
            np.testing.assert_allclose(action(state, observable, field, mass, charge),
                action(states[0], observable, field, mass, charge), atol=2e-14)
    for state in states:
        gradient = np.asarray(jax.grad(flux_observable)(state))
        assert abs(gradient[:3] @ flow(state, field, mass, charge)[:3]) < 2e-14
    # A joint (psi,mu) marginal is a stronger claim and fails for local pairs.
    joint = lambda z: z[4]*flux(z[:3], field)
    first = states[1];second = first.at[4].set(.2)
    assert weak_factor(first, second, joint, field, mass, charge) > 1e-8


def test_dipole_flux_coordinate_measure_independent_cartesian_jacobian():
    strength = 1.3
    field = Field("dipole", strength=strength)
    def cartesian(coordinates):
        psi, latitude, theta = coordinates
        radius = strength*jnp.cos(latitude)**2/psi
        return radius*jnp.array([jnp.cos(latitude)*jnp.cos(theta),
                                  jnp.cos(latitude)*jnp.sin(theta),
                                  jnp.sin(latitude)])
    for coordinates in [jnp.array([.7, .3, .4]), jnp.array([.9, -.4, 1.2])]:
        psi, latitude, _ = np.asarray(coordinates)
        x = cartesian(coordinates)
        determinant = abs(float(jnp.linalg.det(jax.jacfwd(cartesian)(coordinates))))
        np.testing.assert_allclose(determinant,
            strength**3*np.cos(latitude)**7/psi**4, rtol=2e-14)
        magnitude = float(jnp.linalg.norm(field_vector(x, field)))
        np.testing.assert_allclose(magnitude*determinant,
            strength*np.cos(latitude)*np.sqrt(1+3*np.sin(latitude)**2)/psi, rtol=2e-14)
        np.testing.assert_allclose(flux(x, field), psi, rtol=2e-14)
