"""Signed kick interpolation across cancellations and independent COM averages."""
import numpy as np
import pytest

from sato_morrison.controls import encounter_kick_interpolator


def table_axes(phases=16):
    return [np.linspace(1., 2., 4), np.linspace(.6, 1.6, 4),
            np.linspace(.2, .8, 4), 2*np.pi*np.arange(phases)/phases]


def kicks_at(nodes):
    b, u, p, _ = np.moveaxis(np.asarray(nodes), -1, 0)
    return np.stack((.01*(b-1.5)*(u*u+p), .02*(b-1.5)**3), axis=-1)


def independent_com_moments(nodes, kicks, theta, mass, field):
    # Average actual individual squared velocities with Gaussian COM nodes.
    # This evaluates the observable without using the interpolator's formula.
    x, w = np.polynomial.hermite.hermgauss(5)
    center = np.sqrt(theta)*np.stack(np.meshgrid(x, x, indexing='ij'), axis=-1).reshape(-1, 2)
    weights = np.outer(w, w).ravel()/np.pi
    incoming = nodes[:, 2, None]*np.stack((np.cos(nodes[:, 3]), np.sin(nodes[:, 3])), axis=-1)
    first = []; second = []
    for sign, store in ((1, first), (-1, second)):
        vi = center[None]+sign*incoming[:, None]/2
        vf = center[None]+sign*(incoming+kicks)[:, None]/2
        store.extend(mass/(2*field)*(np.sum(vf*vf, axis=-1)-np.sum(vi*vi, axis=-1)))
    first, second = np.asarray(first), np.asarray(second)
    return np.stack((first@weights, first**2@weights, (first*second)@weights), axis=-1)


def test_signed_kicks_reconstruct_independent_thermal_averages_through_null():
    axes = table_axes()
    mesh = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
    theta, mass, field = .37, 1.7, 2.3
    interpolation = encounter_kick_interpolator(axes, kicks_at(mesh), theta, mass, field)
    nodes = np.array([[1.3, 1.1, .45, .23], [1.7, .8, .6, 5.5],
                      [1.5-1e-5, 1., .4, 3.], [1.5+1e-5, 1., .4, 3.]])
    expected = independent_com_moments(nodes, kicks_at(nodes), theta, mass, field)
    np.testing.assert_allclose(interpolation(nodes), expected, rtol=2e-7, atol=1e-20)
    null = interpolation([1.5, 1., .4, 1.])
    assert null[1] < 1e-25  # A physical zero needs no positive variance floor.
    assert null.shape == (3,)
    zero = encounter_kick_interpolator(axes, np.zeros(mesh.shape[:-1]+(2,)), theta)
    np.testing.assert_array_equal(zero(nodes), np.zeros((len(nodes), 3)))


def test_signed_kicks_periodic_phase_and_pair_constraint():
    axes = table_axes(32)
    mesh = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
    kicks = kicks_at(mesh)+.001*np.stack((np.sin(mesh[..., 3]), np.cos(mesh[..., 3])), axis=-1)
    interpolation = encounter_kick_interpolator(axes, kicks, .5)
    nodes = np.array([[1.3, 1.1, .45, .3], [1.7, .8, .6, 5.5]])
    predictions = interpolation(nodes)
    shifted = nodes.copy(); shifted[:, 3] += 4*np.pi
    np.testing.assert_allclose(interpolation(shifted), predictions, atol=1e-18, rtol=1e-12)
    assert np.all(predictions[:, 1] >= predictions[:, 0]**2)
    np.testing.assert_allclose(predictions[:, 1]+predictions[:, 2], 2*predictions[:, 0]**2, atol=1e-20)
    h = 1e-6
    seam = interpolation([[1.4, 1.1, .45, angle] for angle in (0., h, 2*h, -h, -2*h)])
    right = (-3*seam[0]+4*seam[1]-seam[2])/(2*h)
    left = (3*seam[0]-4*seam[3]+seam[4])/(2*h)
    np.testing.assert_allclose(left, right, atol=1e-11)


def test_signed_kick_table_rejects_invalid_coordinates_and_extrapolation():
    axes = table_axes()
    mesh = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
    kicks = kicks_at(mesh)
    interpolation = encounter_kick_interpolator(axes, kicks, .5)
    for nodes in (None, [1., 2., 3.], [1., 1., .4, np.nan], [.9, 1., .4, 0.]):
        with pytest.raises(ValueError):
            interpolation(nodes)
    for kwargs in ({'temperature': 0.}, {'temperature': np.nan}, {'temperature': .5, 'field': -1.}):
        with pytest.raises(ValueError):
            encounter_kick_interpolator(axes, kicks, **kwargs)
    shifted = axes[:3]+[axes[3]+.1]
    with pytest.raises(ValueError):
        encounter_kick_interpolator(shifted, kicks, .5)
    with pytest.raises(ValueError):
        encounter_kick_interpolator(axes, kicks[..., :1], .5)
