"""Analytic limits and a sensitive screened-encounter trajectory regression."""
import numpy as np
import pytest

from sato_morrison.controls import (
    binary_encounter, encounter_relative, encounter_thermal_moments,
)


@pytest.mark.parametrize('field', [0., 2., -2.])
def test_zero_force_pair_follows_complete_free_helix(field):
    position = np.array([.7, -.3, -2.])
    velocity = np.array([.8, -.6, 1.3])
    mass, charge = 1.7, .6
    pair = binary_encounter(position, velocity, field=field, strength=0.,
                            mass=mass, charge=charge, duration=6.37,
                            max_step=.025, rtol=1e-12)
    time = pair['time']
    omega = charge*field/mass
    if omega:
        cosine, sine = np.cos(omega*time), np.sin(omega*time)
        exact_velocity = np.column_stack((
            velocity[0]*cosine+velocity[1]*sine,
            velocity[1]*cosine-velocity[0]*sine,
            np.full(len(time), velocity[2]),
        ))
        exact_position = np.column_stack((
            position[0]+velocity[0]*sine/omega+velocity[1]*(1-cosine)/omega,
            position[1]+velocity[1]*sine/omega-velocity[0]*(1-cosine)/omega,
            position[2]+velocity[2]*time,
        ))
    else:
        exact_velocity = np.tile(velocity, (len(time), 1))
        exact_position = position+time[:, None]*velocity
    actual_position = pair['positions'][:, 0]-pair['positions'][:, 1]
    actual_velocity = pair['velocities'][:, 0]-pair['velocities'][:, 1]
    np.testing.assert_allclose(actual_position, exact_position, rtol=0., atol=1e-10)
    np.testing.assert_allclose(actual_velocity, exact_velocity, rtol=0., atol=1e-10)
    if field:
        np.testing.assert_allclose(pair['delta_mu'], 0., rtol=0., atol=1e-10)


@pytest.mark.parametrize('strength,impact,distance', [(.03, .2, 20.), (.1, 1., 40.), (.5, .2, 40.)])
def test_coulomb_pair_matches_finite_radius_repulsive_hyperbola(strength, impact, distance):
    position = np.array([impact, 0., -distance])
    velocity = np.array([0., 0., 1.])
    # Equal masses give relative specific acceleration kappa*r/|r|^3.
    kappa = 2*strength
    radius = np.linalg.norm(position)
    angular_momentum = np.cross(position, velocity)
    energy = np.dot(velocity, velocity)/2+kappa/radius
    lenz = np.cross(velocity, angular_momentum)+kappa*position/radius
    axis = lenz/np.linalg.norm(lenz)
    eccentricity = np.linalg.norm(lenz)/kappa
    semiscale = kappa/(2*energy)
    anomaly = np.arccosh((radius/semiscale-1)/eccentricity)
    # Repulsive orbit r=a*(e*cosh(H)+1); time is measured from periapsis.
    duration = 2*np.sqrt(semiscale**3/kappa)*(eccentricity*np.sinh(anomaly)+anomaly)
    exact_position = 2*np.dot(position, axis)*axis-position
    exact_velocity = velocity-2*np.dot(velocity, axis)*axis
    pair = binary_encounter(position, velocity, field=0., strength=strength,
                            screening=np.inf, duration=duration, max_step=.025, rtol=1e-12)
    final_position = pair['positions'][-1, 0]-pair['positions'][-1, 1]
    final_velocity = pair['velocities'][-1, 0]-pair['velocities'][-1, 1]
    np.testing.assert_allclose(final_position, exact_position, rtol=0., atol=1e-8)
    np.testing.assert_allclose(final_velocity, exact_velocity, rtol=0., atol=1e-8)
    assert pair['energy_error'] < 1e-10


def test_sensitive_screened_phase_matches_independent_precision_reference():
    # Master phase2054 of8192, same public binary64 initial preparation.
    # Cartesian analytic Taylor integration at50/70 decimal digits, order28/36,
    # local coefficient targets1e-30/1e-44, agrees below1e-31 at this first exit.
    expected_time = 27.82149964382760842828873371436128667
    expected_position = [-.9432851953256336263552733594107928,
                         -.2006354880215104058803323493151745, -24.]
    expected_velocity = [1.390704188035333810852268416007950,
                         1.259953576513684060011466858579961,
                         -1.994609101679418799325685433200151]
    expected_moments = [-.02168182732196079925956920976869926,
                       .1421762612624866385436198520240196,
                       -.1412360579904479869349915087013718]
    relative = encounter_relative(
        .9938225266093732, 1.5753982691585535, 1.9056642462041216,
        1.9668376625283577, field=2., strength=.03, screening=3.,
        start_distance=24., max_step=.025, rtol=1e-12, flight_time_factor=12.,
    )
    assert relative['exit'] == 'reflected'
    assert abs(relative['time']-expected_time) <= 1e-8
    np.testing.assert_allclose(relative['final_relative_position'], expected_position, rtol=0., atol=1e-8)
    np.testing.assert_allclose(relative['final_relative_velocity'], expected_velocity, rtol=0., atol=1e-8)
    moments = encounter_thermal_moments(relative, .5, 1., 2.)
    np.testing.assert_allclose(moments, expected_moments, rtol=0., atol=1e-10)
    pair = binary_encounter(relative['initial_relative_position'], relative['initial_relative_velocity'],
                            field=2., strength=.03, screening=3., duration=relative['time'],
                            max_step=.025, rtol=1e-12)
    position = pair['positions'][-1, 0]-pair['positions'][-1, 1]
    velocity = pair['velocities'][-1, 0]-pair['velocities'][-1, 1]
    np.testing.assert_allclose(position, expected_position, rtol=0., atol=1e-8)
    np.testing.assert_allclose(velocity, expected_velocity, rtol=0., atol=1e-8)
    assert abs(pair['delta_mu'][0]-expected_moments[0]) <= 1e-10
