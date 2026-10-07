"""Vacuum guiding-center geometry in Cartesian mu and eta charts.

Numerical routines are pure single-point JAX functions, suitable for vmap/jit.
Call validate_geometry on host inputs before entering a compiled calculation.
The vacuum restriction is essential: the invariant mu-chart density is B.
"""

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class Field:
    """Analytic vacuum field; nonaxisymmetric adds grad(amplitude*strength*x*y/length)."""

    kind: str = "uniform"
    strength: float = 1.0
    amplitude: float = 0.0
    length: float = 1.0
    direction: tuple = (0.0, 0.0, 1.0)
    base: str = "dipole"

    def __post_init__(self):
        names = ("uniform", "mirror", "toroidal", "dipole", "nonaxisymmetric")
        if self.kind not in names or self.base not in names[:-1]:
            raise ValueError("Unsupported vacuum field.")
        scalars = (self.strength, self.amplitude, self.length)
        if not all(math.isfinite(float(v)) for v in scalars):
            raise ValueError("Field parameters must be finite.")
        if self.strength <= 0 or self.length <= 0:
            raise ValueError("Field strength and length must be positive.")
        direction = np.asarray(self.direction, dtype=float)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("Field direction must be a finite three-vector.")
        if np.linalg.norm(direction) == 0:
            raise ValueError("Field direction must be nonzero.")
        object.__setattr__(self, "direction", tuple(direction / np.linalg.norm(direction)))


def field_vector(x, field):
    """Evaluate a Cartesian vacuum field at one position (no domain clipping)."""
    if not isinstance(field, Field):
        raise ValueError("A validated analytic Field is required.")
    x = jnp.asarray(x)
    kind = field.base if field.kind == "nonaxisymmetric" else field.kind
    xx, yy, zz = x
    c, a = field.strength, field.amplitude
    if kind == "uniform":
        out = c * jnp.asarray(field.direction)
    elif kind == "mirror":
        # Here amplitude has units B / length**2, as in the stated harmonic field.
        out = jnp.array([-a * zz * xx, -a * zz * yy,
                         c + a * (zz * zz - (xx * xx + yy * yy) / 2)])
    elif kind == "toroidal":
        out = c * jnp.array([-yy, xx, 0.0]) / (xx * xx + yy * yy)
    else:
        r2 = jnp.dot(x, x)
        out = c * jnp.array([3 * xx * zz, 3 * yy * zz,
                            2 * zz * zz - xx * xx - yy * yy]) / r2**2.5
    if field.kind == "nonaxisymmetric":
        out = out + a * c / field.length * jnp.array([yy, xx, 0.0])
    return out


def field_data(x, field):
    """Return (B vector, B magnitude, b, grad B, curl b) from exact AD derivatives."""
    vector = field_vector(x, field)
    strength = jnp.linalg.norm(vector)
    unit = vector / strength
    grad = jax.grad(lambda y: jnp.linalg.norm(field_vector(y, field)))(x)
    derivative = jax.jacfwd(lambda y: field_vector(y, field)
                            / jnp.linalg.norm(field_vector(y, field)))(x)
    curl = jnp.array([derivative[2, 1] - derivative[1, 2],
                      derivative[0, 2] - derivative[2, 0],
                      derivative[1, 0] - derivative[0, 1]])
    return vector, strength, unit, grad, curl


def poisson_mu(z, field, mass=1.0, charge=1.0):
    """Poisson tensor at z=(x,y,z,u,mu), in the vacuum B_parallel_star=B model."""
    vector, strength, unit, _, curl = field_data(z[:3], field)
    bstar = vector + mass * z[3] / charge * curl
    bx, by, bz = unit
    cross = jnp.array([[0.0, -bz, by], [bz, 0.0, -bx], [-by, bx, 0.0]])
    matrix = jnp.zeros((5, 5), dtype=jnp.result_type(z, strength))
    matrix = matrix.at[:3, :3].set(cross / (charge * strength))
    matrix = matrix.at[:3, 3].set(bstar / (mass * strength))
    return matrix.at[3, :3].set(-bstar / (mass * strength))


def chart_transform(z, field):
    """Derivative d(X,u,eta)/d(X,u,mu), with eta=mu*B(X)."""
    _, strength, _, grad, _ = field_data(z[:3], field)
    transform = jnp.eye(5, dtype=jnp.result_type(z, strength))
    transform = transform.at[4, :3].set(z[4] * grad)
    return transform.at[4, 4].set(strength)


eta_transform = chart_transform


def to_eta(z, field):
    return z.at[4].set(z[4] * jnp.linalg.norm(field_vector(z[:3], field)))


def to_mu(zeta, field):
    return zeta.at[4].set(zeta[4] / jnp.linalg.norm(field_vector(zeta[:3], field)))


def poisson_eta(zeta, field, mass=1.0, charge=1.0):
    """Transform the tensor; do not recreate a Euclidean collision projector."""
    z = to_mu(zeta, field)
    transform = chart_transform(z, field)
    return transform @ poisson_mu(z, field, mass, charge) @ transform.T


def common_chart_action(z, gradient_mu, field, mass=1.0, charge=1.0):
    """Evaluate J grad(phi) in the common eta chart from a mu-chart covector."""
    return chart_transform(z, field) @ poisson_mu(z, field, mass, charge) @ gradient_mu


def energy_mu(z, field, mass=1.0, charge=1.0, potential=None):
    """Guiding-center energy; potential is a scalar callable Phi(X), if provided."""
    value = mass * z[3]**2 / 2 + z[4] * jnp.linalg.norm(field_vector(z[:3], field))
    return value if potential is None else value + charge * potential(z[:3])


def validate_geometry(points, field, mass=1.0, charge=1.0, min_field=1e-12):
    """Reject invalid host positions/states and fields outside the vacuum contract.

    Accept (N,3) positions or (N,5) mu-chart states; return minimum sampled B.
    Sampling verifies the inputs, and does not certify an entire intervening domain.
    """
    if not isinstance(field, Field):
        raise ValueError("Only the specified analytic vacuum fields are supported.")
    values = np.asarray(points, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] not in (3, 5) or not values.shape[0]:
        raise ValueError("Expected nonempty Cartesian positions or mu-chart states.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Coordinates must be finite.")
    if not all(math.isfinite(float(v)) for v in (mass, charge, min_field)):
        raise ValueError("Physical parameters must be finite.")
    if mass <= 0 or charge == 0 or min_field <= 0:
        raise ValueError("Require positive mass/minimum field and nonzero charge.")
    if values.shape[1] == 5 and np.any(values[:, 4] < 0):
        raise ValueError("Magnetic moment must be nonnegative.")
    kind = field.base if field.kind == "nonaxisymmetric" else field.kind
    positions = values[:, :3]
    if kind == "dipole" and np.any(np.linalg.norm(positions, axis=1) == 0):
        raise ValueError("Dipole source is excluded.")
    if kind == "toroidal" and np.any(np.linalg.norm(positions[:, :2], axis=1) == 0):
        raise ValueError("Toroidal axis is excluded.")
    magnitudes = np.asarray(jax.vmap(lambda x: jnp.linalg.norm(field_vector(x, field)))(
        jnp.asarray(positions)))
    if not np.all(np.isfinite(magnitudes)) or np.any(magnitudes <= min_field):
        raise ValueError("Nonzero, nondegenerate magnetic field required.")
    return float(magnitudes.min())
