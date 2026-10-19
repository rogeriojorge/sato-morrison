"""Independent same-field inverse preparation and finite-plane balance checks."""
import numpy as np
import pytest

from sato_morrison.controls import (
    encounter_incoming, encounter_relative, encounter_same_field_reverse_incoming,
    encounter_thermal_moments,
)


def rotate_to_gauge(vector, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c*vector[0]+s*vector[1], -s*vector[0]+c*vector[1], vector[2]])


@pytest.mark.parametrize('mass,charge', [(1., 1.), (1.7, -.8)])
@pytest.mark.parametrize('exit_name,sign', [('transmitted', 1.), ('reflected', -1.)])
def test_same_field_inverse_reconstructs_actual_endpoint(mass, charge, exit_name, sign):
    # Arbitrary physical plane data; preparation must reproduce coordinates,
    # independently of whether an encounter integrator supplied the endpoint.
    position = np.array([1.3, -.27, sign*8.])
    velocity = np.array([.31, -.46, sign*.73])
    result = dict(exit=exit_name, final_relative_position=position,
                  final_relative_velocity=velocity)
    node, angle = encounter_same_field_reverse_incoming(
        result, field=2., mass=mass, charge=charge, start_distance=8.)
    b, u, p, phase = node
    rebuilt_r, rebuilt_w = encounter_incoming(b, phase, u, p, 8., charge*2/mass)
    reflection = np.array([1., -1., -sign])
    np.testing.assert_allclose(rebuilt_r, rotate_to_gauge(reflection*position, angle), atol=3e-14)
    np.testing.assert_allclose(rebuilt_w, rotate_to_gauge(-reflection*velocity, angle), atol=3e-14)
    assert b > 0 and u > 0 and p > 0 and 0 <= phase < 2*np.pi


@pytest.mark.parametrize('mass,charge', [(1., 1.), (1.7, -.8)])
@pytest.mark.parametrize('node,exit_name', [
    ((1.4, 1.1, .4, .6), 'transmitted'),
    ((.2, .1, .1, .4), 'reflected'),
])
def test_same_field_physical_inverse_both_exits_and_gibbs(mass, charge, node, exit_name):
    b, u, p, phase = node
    kwargs = dict(field=2., strength=.03, screening=3., start_distance=8.,
                  mass=mass, charge=charge, max_step=.05, rtol=2e-13,
                  flight_time_factor=48.)
    forward = encounter_relative(b, phase, u, p, **kwargs)
    assert forward['exit'] == exit_name
    inverse, angle = encounter_same_field_reverse_incoming(
        forward, field=2., mass=mass, charge=charge, start_distance=8.)
    bi, ui, pi, phii = inverse
    reverse = encounter_relative(bi, phii, ui, pi, **kwargs)
    assert reverse['exit'] == exit_name
    reflection = np.array([1., -1., -1. if exit_name == 'transmitted' else 1.])
    expected_r = rotate_to_gauge(reflection*forward['initial_relative_position'], angle)
    expected_w = rotate_to_gauge(-reflection*forward['initial_relative_velocity'], angle)
    np.testing.assert_allclose(reverse['final_relative_position'], expected_r, atol=1e-8, rtol=0)
    np.testing.assert_allclose(reverse['final_relative_velocity'], expected_w, atol=1e-8, rtol=0)
    np.testing.assert_allclose(reverse['time'], forward['time'], atol=1e-8, rtol=0)
    mf = encounter_thermal_moments(forward, .5, mass, 2.)
    mr = encounter_thermal_moments(reverse, .5, mass, 2.)
    np.testing.assert_allclose(mr, mf*np.array([-1., 1., 1.]), atol=1e-10, rtol=0)
    # The finite-plane invariant is total relative Gibbs energy, rather than
    # kinetic Maxwellian density alone. This is independent of moment helpers.
    def log_gibbs(position, velocity):
        radius = np.linalg.norm(position)
        potential = .03*np.exp(-radius/3)/radius
        return -(mass*np.dot(velocity, velocity)/4+potential)/(mass*.5)
    np.testing.assert_allclose(
        log_gibbs(forward['initial_relative_position'], forward['initial_relative_velocity']),
        log_gibbs(reverse['initial_relative_position'], reverse['initial_relative_velocity']),
        atol=1e-11, rtol=0)


@pytest.mark.parametrize('changes', [
    {'exit': 'missing'},
    {'final_relative_position': np.array([1., 0., 7.])},
    {'final_relative_velocity': np.array([.2, .3, -.5])},
    {'final_relative_velocity': np.array([.2, np.nan, .5])},
    # Under S and time reversal, this endpoint has exactly zero transverse GC.
    {'final_relative_position': np.array([-.15, .1, 8.])},
])
def test_same_field_inverse_rejects_incomplete_or_degenerate_endpoint(changes):
    result = dict(exit='transmitted', final_relative_position=np.array([1., 0., 8.]),
                  final_relative_velocity=np.array([.2, .3, .5]))
    result.update(changes)
    with pytest.raises(ValueError):
        encounter_same_field_reverse_incoming(result, field=2., start_distance=8.)


def test_same_field_inverse_zero_perpendicular_speed_is_a_valid_gauge_choice():
    result = dict(exit='reflected', final_relative_position=np.array([1., .2, -8.]),
                  final_relative_velocity=np.array([0., 0., -.5]))
    node, angle = encounter_same_field_reverse_incoming(result, field=2., start_distance=8.)
    b, u, p, phase = node
    r, w = encounter_incoming(b, phase, u, p, 8., 2.)
    np.testing.assert_allclose(r, rotate_to_gauge(np.array([1., -.2, -8.]), angle), atol=1e-14)
    np.testing.assert_array_equal(w, [0., 0., .5])


@pytest.mark.parametrize('charge', [-.8, 1.])
def test_free_inverse_is_exact_phase_reflection_and_unit_crossing_jacobian(charge):
    # The exact force-free inverse is (b,u,p,phi)->(b,u,p,pi-phi).
    # Thus its absolute Jacobian and bup flux ratio are both one. This also
    # checks the zero-plane phase shear independently of the incoming helper.
    node = np.array([1.7, .9, .6, 5.3])
    b, u, p, phase = node
    forward = encounter_relative(b, phase, u, p, field=2., strength=0.,
        screening=3., start_distance=8., mass=1.7, charge=charge,
        max_step=.05, rtol=2e-13)
    inverse, _ = encounter_same_field_reverse_incoming(
        forward, field=2., mass=1.7, charge=charge, start_distance=8.)
    np.testing.assert_allclose(inverse[:3], node[:3], atol=2e-13, rtol=0)
    phase_error = (inverse[3]-(np.pi-phase)+np.pi)%(2*np.pi)-np.pi
    assert abs(phase_error) < 3e-13
    assert abs(np.prod(inverse[:3])/np.prod(node[:3])-1) < 1e-12
