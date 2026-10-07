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


def run_metadata(inputs, *, model, boundary, units="normalized"):
    """Reproduction context; source hashes also identify uncommitted calculations."""
    import hashlib
    import importlib.metadata
    import platform
    import os
    import subprocess
    from pathlib import Path
    import jax

    root = Path(__file__).resolve().parents[2]
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    sources = sorted([*root.glob("src/**/*.py"), *root.glob("examples/*.py"),
                      *root.glob("tests/*.py"), root / "pyproject.toml"])
    digest = hashlib.sha256()
    for path in sources:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    versions = {}
    for name in ("sato-morrison", "jax", "jaxlib", "solvax", "numpy", "scipy", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True) if platform.system() == "Darwin" else None
    return {"model": model, "boundary": boundary, "units": units, "inputs": inputs,
            "commit": git("rev-parse", "HEAD"), "working_tree": git("status", "--porcelain"),
            "source_sha256": digest.hexdigest(), "versions": versions,
            "source_changes": git("status", "--porcelain", "src", "examples", "tests", "pyproject.toml"),
            "logical_cpus": os.cpu_count(),
            "thread_environment": {key: os.environ.get(key, "unset") for key in
                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "XLA_FLAGS")},
            "python": platform.python_version(), "platform": platform.platform(),
            "processor": chip.stdout.strip() if chip else platform.processor(),
            "devices": [str(d) for d in jax.devices()], "x64": bool(jax.config.x64_enabled)}


def progress(message, interval=15):
    """Context manager reporting elapsed wall time during compilation or long work."""
    from contextlib import contextmanager
    from threading import Event, Thread
    from time import perf_counter

    @contextmanager
    def report():
        start, stopped = perf_counter(), Event()
        print(message, flush=True)
        def heartbeat():
            while not stopped.wait(interval):
                print(f"  {message}: {perf_counter()-start:.1f} s elapsed", flush=True)
        thread = Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stopped.set()
            thread.join()
            print(f"  {message}: returned after {perf_counter()-start:.3f} s", flush=True)
    return report()


def range_fourier_rates(masses, k, width, *, diffusion=1.0, charge=1.0, field=1.0):
    """Exact rates for a normalized periodized-Gaussian uniform-field surrogate.

    One perpendicular Fourier coordinate is retained, b=e_z, k=k_x.
    Returns (density rate, velocity-neutral rate). This is not a Coulomb tensor.
    """
    masses = np.asarray(masses, dtype=float)
    if masses.ndim != 1 or not masses.size or np.any(masses <= 0):
        raise ValueError("Positive flat equilibrium quadrature masses required.")
    scalars = np.asarray([k, width, diffusion, charge, field], dtype=float)
    if not np.all(np.isfinite(masses)) or not np.all(np.isfinite(scalars)):
        raise ValueError("Finite parameters required.")
    if width < 0 or diffusion < 0 or charge == 0 or field <= 0:
        raise ValueError("Invalid range or physical parameters.")
    rate = diffusion * masses.sum() * k**2 / (charge * field)**2
    return rate * (-np.expm1(-0.5 * width**2 * k**2)), rate


def range_pair_matrix(nx, masses, mode, width, *, length=2 * np.pi,
                      diffusion=1.0, charge=1.0, field=1.0):
    """Independent ordered spatial/velocity pair Gram assembly for one mode.

    Uniform periodic trapezoid quadrature and exact Fourier derivatives are used.
    The scalar normalized Gaussian spatial kernel is evaluated by image sums.
    All velocity pairs, including coincident nodes, retain the uniform directional
    projector limit. Return entropy-scaled positive matrix, sampled W_hat ratio.
    """
    if not isinstance(nx, (int, np.integer)) or nx < 3:
        raise ValueError("Need at least three periodic spatial nodes.")
    if not np.isfinite(length) or length <= 0 or not isinstance(mode, (int, np.integer)):
        raise ValueError("Positive period and integer Fourier mode required.")
    if abs(mode) >= nx / 2:
        raise ValueError("Fourier mode must lie strictly below the spatial Nyquist limit.")
    range_fourier_rates(masses, 2 * np.pi * mode / length, width,
                        diffusion=diffusion, charge=charge, field=field)
    masses = np.asarray(masses, dtype=float)
    dx = length / nx
    x = np.arange(nx) * dx
    k = 2 * np.pi * mode / length
    phase = np.exp(1j * k * x)
    distance = x[:, None] - x[None, :]
    if width == 0:
        kernel = np.eye(nx) / dx
    else:
        images = int(np.ceil(8 * width / length)) + 1
        kernel = sum(np.exp(-0.5 * ((distance + j * length) / width)**2)
                     for j in range(-images, images + 1))
        # Periodic uniform nodes give the same row mass; one scalar keeps symmetry.
        kernel = kernel / (dx * kernel[0].sum())
    nv = masses.size
    identity = np.eye(nv)
    pair_mass = masses[:, None] * masses[None, :]
    gram = np.zeros((nv, nv), dtype=complex)
    factor = diffusion * k**2 / (charge * field)**2
    for a in range(nx):
        for b in range(nx):
            # P J grad chi leaves i*k*(b cross e_x)/(qB)*chi in the y block.
            difference = phase[a] * identity[:, None, :] - phase[b] * identity[None, :, :]
            gram += 0.5 * dx**2 * factor * kernel[a, b] * np.einsum(
                "ij,ijp,ijq->pq", pair_mass, difference.conj(), difference)
    scaled = gram / (length * np.sqrt(masses[:, None] * masses[None, :]))
    ratio = float(np.real(dx * np.dot(kernel[0], phase)))
    return np.real_if_close(scaled), ratio


def constrained_equilibrium(energy, weights, mu_index, initial):
    """Finite-quadrature maximum entropy at fixed energy and full mu marginal.

    This independent reference computes a candidate state, not reachability.
    A positive inverse temperature must exist in the supplied finite domain.
    """
    from scipy.optimize import brentq
    e, w, labels, f = map(np.asarray, (energy, weights, mu_index, initial))
    if e.ndim != 1 or not all(a.shape == e.shape for a in (w, labels, f)):
        raise ValueError("Expected equal flat arrays.")
    if np.any(w <= 0) or np.any(f <= 0) or not all(np.all(np.isfinite(a)) for a in (e,w,f)):
        raise ValueError("Positive finite populations and weights required.")
    if labels.dtype.kind not in 'iu' or np.any(labels < 0):
        raise ValueError("Nonnegative integer mu-bin labels required.")
    marginal = np.bincount(labels, weights=w*f)
    target = np.dot(w*f,e)
    def population(beta):
        out = np.zeros_like(e,dtype=float)
        for label in np.unique(labels):
            mask = labels == label
            raw = np.exp(-beta*(e[mask]-e[mask].min()))
            out[mask] = marginal[label]*raw / np.dot(w[mask],raw)
        return out
    def mismatch(beta):
        return np.dot(w*population(beta),e)-target
    upper=1.
    while mismatch(upper)>0 and upper < 1e8:
        upper*=2
    if mismatch(0)<0 or mismatch(upper)>0:
        raise ValueError("No positive-temperature candidate on this quadrature.")
    beta=brentq(mismatch,0.,upper,xtol=1e-13,rtol=1e-13)
    return beta,population(beta)
