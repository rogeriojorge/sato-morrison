"""Finite-range density-nullspace lifting checked with independent pair quadrature."""

import numpy as np
import pytest

from sato_morrison.reference import range_fourier_rates, range_pair_matrix


def test_finite_range_exact_velocity_spectrum_and_local_limit():
    masses = np.array([0.2, 0.3, 0.4, 0.7, 0.1])
    density = np.sqrt(masses / masses.sum())
    for width in (0, 0.18, 0.5, 1.5):
        matrix, ratio = range_pair_matrix(64, masses, 2, width, diffusion=0.8,
                                          charge=-1.3, field=1.7)
        density_rate, neutral_rate = range_fourier_rates(
            masses, 2, width, diffusion=0.8, charge=-1.3, field=1.7)
        predicted = neutral_rate * (np.eye(masses.size) - ratio * np.outer(density, density))
        np.testing.assert_allclose(matrix, predicted, atol=5e-13)
        np.testing.assert_allclose(matrix @ density, density_rate * density, atol=5e-13)
        np.testing.assert_allclose(np.linalg.eigvalsh(matrix),
                                   [density_rate] + [neutral_rate] * (masses.size - 1),
                                   atol=5e-13)
        assert ratio == pytest.approx(np.exp(-2 * width**2), abs=3e-13)
    zero, _ = range_pair_matrix(8, masses, 0, 0.5)
    np.testing.assert_array_equal(zero, np.zeros((5, 5)))


def test_spatial_and_range_convergence():
    masses = np.array([0.3, 0.4, 0.7])
    width = 0.15
    exact = range_fourier_rates(masses, 1, width)[0]
    errors = []
    for nx in (16, 32, 64):
        matrix, _ = range_pair_matrix(nx, masses, 1, width)
        errors.append(abs(np.linalg.eigvalsh(matrix)[0] - exact))
    assert errors[1] < errors[0] / 20
    assert errors[2] < 1e-12
    for width in (0.2, 0.1, 0.05):
        density, neutral = range_fourier_rates(masses, 0.25, width)
        quartic = neutral * width**2 * 0.25**2 / 2
        assert abs(density / quartic - 1) < width**2 * 0.25**2 / 3


def test_invalid_range_parameters():
    for args in (([0, 1], 1, 0.2), ([1], 1, -1), ([np.nan], 1, 0.2)):
        with pytest.raises(ValueError):
            range_fourier_rates(*args)
    with pytest.raises(ValueError):
        range_pair_matrix(8, [1], 1, 0.2, length=0)
    for args in ((2, [1], 1, 0.2), (8, [1], 4, 0.2), (8, [1], 0.5, 0.2)):
        with pytest.raises(ValueError):
            range_pair_matrix(*args)
