"""Small independent checks of the uniform-field oracle."""

import numpy as np
import pytest

from sato_morrison.reference import gauss_interval, pair_rhs


@pytest.mark.parametrize("charge", [1.3, -1.3])
@pytest.mark.parametrize("field", [[0., 0., 2.], [1., -2., 3.]])
def test_fourier_reduction(charge, field):
    u1, wu = gauss_interval(7, -4., 4.)
    eta1, we = gauss_interval(4, 0., 8.)
    u, eta = (a.ravel() for a in np.meshgrid(u1, eta1, indexing="ij"))
    w = (wu[:, None] * we[None, :]).ravel()
    f0 = np.exp(-u*u/2-eta)
    n0 = np.dot(w, f0)
    h = u*u + eta
    k = np.array([0.4, 0.7, -0.2])
    field = np.array(field)
    b = field / np.linalg.norm(field)
    coefficient = 0.8 * np.dot(k-np.dot(k, b)*b, k-np.dot(k, b)*b) / (charge*np.linalg.norm(field))**2
    args = dict(charge=charge, field=field, mass=1., diffusion=0.8)
    actual = pair_rhs(u, w, f0, h, 2*u, k=k, **args)
    expected = -coefficient * f0 * (n0*h - np.dot(w*f0, h))
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
    zero = pair_rhs(u, w, f0, h, 2*u, k=np.zeros(3), **args)
    np.testing.assert_allclose(zero, 0, atol=1e-14)
    density = pair_rhs(u, w, f0, np.ones_like(h), np.zeros_like(h), k=k, **args)
    np.testing.assert_allclose(density, 0, atol=1e-14)
    parallel = pair_rhs(u, w, f0, h, 2*u, k=b, **args)
    np.testing.assert_allclose(parallel, 0, atol=1e-13)


def test_quadrature_input():
    with pytest.raises(ValueError):
        gauss_interval(1, -1, 1)
    x, w = gauss_interval(5, -2, 3)
    np.testing.assert_allclose(np.sum(w), 5.)
    np.testing.assert_allclose(np.dot(w, x*x), 35./3)
