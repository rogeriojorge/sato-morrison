"""NumPy oracle for Eq. (181) in a uniform field; not a general kinetic solver.

Coordinates are (x, y, z, u, eta). The equilibrium and potential are spatially
constant. The equal-parallel-velocity projector uses this geometry's continuous
limit. D is an input model coefficient, not a calibrated Coulomb rate.
"""

import numpy as np


def gauss_interval(n, lower, upper):
    """Return Gauss-Legendre nodes and weights on a finite interval."""
    if n < 2 or not lower < upper:
        raise ValueError("Need n >= 2 and an ordered interval.")
    x, w = np.polynomial.legendre.leggauss(n)
    return lower + (upper - lower) * (x + 1) / 2, w * (upper - lower) / 2


def pair_rhs(u, weights, f0, h, dh_du, *, charge, field, mass, diffusion, k):
    """Apply the five-coordinate pair tensor to f0*h for one Fourier mode.

    Velocity arrays are flat and share a shape. ``field`` and ``k`` are Cartesian
    three-vectors. The result is the collision contribution to d(delta f)/dt;
    Hamiltonian streaming is deliberately absent. The direct O(Nv**2) sum is a
    small-grid reference and should remain independent of the production kernel.
    """
    u, weights, f0, h, dh_du = map(np.asarray, (u, weights, f0, h, dh_du))
    field, k = np.asarray(field, dtype=float), np.asarray(k, dtype=float)
    if field.shape != (3,) or k.shape != (3,):
        raise ValueError("field and k must be three-vectors.")
    if u.ndim != 1 or not all(a.shape == u.shape for a in (weights, f0, h, dh_du)):
        raise ValueError("Velocity arrays must be flat and have equal shapes.")
    if not all(np.all(np.isfinite(a)) for a in (u, weights, f0, h, dh_du, field, k, charge, mass, diffusion)):
        raise ValueError("Inputs must be finite.")
    strength = np.linalg.norm(field)
    if charge == 0 or strength <= 0 or mass <= 0 or diffusion < 0:
        raise ValueError("Invalid physical parameters.")
    if np.any(weights <= 0) or np.any(f0 <= 0):
        raise ValueError("Weights and the reference distribution must be positive.")
    b = field / strength
    bx, by, bz = b
    poisson = np.zeros((5, 5))
    poisson[:3, :3] = np.array([[0, -bz, by], [bz, 0, -bx], [-by, bx, 0]]) / (charge * strength)
    poisson[:3, 3], poisson[3, :3] = b / mass, -b / mass
    ix = np.diag([1., 1., 1., 0., 0.])
    grad_energy = np.zeros((u.size, 5))
    grad_energy[:, 3], grad_energy[:, 4] = mass * u, 1.
    ideal_velocity = grad_energy @ poisson.T
    grad_h = np.zeros((u.size, 5), dtype=complex)
    grad_h[:, :3], grad_h[:, 3] = 1j * h[:, None] * k, dh_du
    j_grad_h = grad_h @ poisson.T
    rhs = np.empty(u.size, dtype=complex)
    for i in range(u.size):
        xi = ideal_velocity[i] - ideal_velocity
        norm = np.linalg.norm(xi, axis=1)
        direction = np.zeros_like(xi)
        nonzero = norm > 0
        direction[nonzero] = xi[nonzero] / norm[nonzero, None]
        direction[~nonzero, :3] = b
        projector = np.eye(5)[None] - np.einsum("ni,nj->nij", direction, direction)
        kernel = projector @ ix @ projector
        impulse = np.einsum("nij,nj->ni", kernel, j_grad_h - j_grad_h[i])
        flux = poisson @ np.sum((weights * f0)[:, None] * impulse, axis=0)
        rhs[i] = diffusion * f0[i] * 1j * np.dot(k, flux[:3])
    return rhs
