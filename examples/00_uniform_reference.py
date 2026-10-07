"""Check the uniform-field pair tensor and its entropy-weighted spectrum."""

import json
import platform
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from solvax import pcg_linear_solve

from sato_morrison.reference import gauss_interval, pair_rhs, run_metadata, progress

jax.config.update("jax_enable_x64", True)

# Inputs: normalized units, a uniform field and one perpendicular Fourier mode.
nu, neta = 18, 12
charge, mass, diffusion, n0 = 1.3, 0.8, 0.7, 1.4
field, k = np.array([0., 0., 2.]), np.array([0.9, 0., 0.])
output = Path(__file__).resolve().parents[1] / "results" / "uniform_reference"
output.mkdir(parents=True, exist_ok=True)
print(f"Uniform-field reference: {nu} x {neta} velocity nodes", flush=True)
u1, wu = gauss_interval(nu, -5., 5.)
e1, we = gauss_interval(neta, 0., 10.)
u, eta = (a.ravel() for a in np.meshgrid(u1, e1, indexing="ij"))
weights = (wu[:, None] * we[None, :]).ravel()
f0 = np.exp(-mass * u**2 / 2 - eta)
f0 *= n0 / np.dot(weights, f0)
h = u**2 - np.dot(weights * f0, u**2) / n0
params = dict(charge=charge, field=field, mass=mass, diffusion=diffusion)

# Solve the pair contraction independently of the reduced analytic expression.
start = perf_counter()
print("Evaluating three direct pair contractions...", flush=True)
rhs = pair_rhs(u, weights, f0, h, 2 * u, k=k, **params)
homogeneous = pair_rhs(u, weights, f0, h, 2 * u, k=np.zeros(3), **params)
density = pair_rhs(u, weights, f0, np.ones_like(u), np.zeros_like(u), k=k, **params)
b = field / np.linalg.norm(field)
kperp2 = np.dot(k, k) - np.dot(k, b)**2
rate = diffusion * n0 * kperp2 / (charge * np.linalg.norm(field))**2
expected = -rate * f0 * h
error = np.linalg.norm(rhs - expected) / np.linalg.norm(expected)
print(f"Pair versus analytic relative error: {error:.3e}", flush=True)
measure = weights * f0
operator = rate * (np.outer(np.ones(u.size), measure) / n0 - np.eye(u.size))
root = np.sqrt(measure)
weighted = root[:, None] * operator / root[None, :]
print("Solving the reduced entropy-weighted eigenproblem...", flush=True)
eigenvalues = np.linalg.eigvalsh((weighted + weighted.T) / 2)
report = {
    "scope": "Eq. 181 uniform-field algebra only; no general PDE evolution",
    "velocity_points": int(u.size), "analytic_decay_rate": float(rate),
    "pair_relative_error": float(error),
    "homogeneous_rhs_max": float(np.max(np.abs(homogeneous))),
    "density_rhs_max": float(np.max(np.abs(density))),
    "weighted_symmetry_error": float(np.linalg.norm(weighted-weighted.T)/np.linalg.norm(weighted)),
    "largest_eigenvalue": float(eigenvalues[-1]),
    "nonzero_spectrum_error": float(np.max(np.abs(eigenvalues[:-1]+rate))),
    "inputs": {"u_nodes": nu, "eta_nodes": neta, "u_interval": [-5., 5.],
               "eta_interval": [0., 10.], "charge": charge, "mass": mass,
               "diffusion": diffusion, "density": n0, "field": field.tolist(),
               "wave_vector": k.tolist(), "units": "normalized"},
    "dtype": str(f0.dtype), "python_version": platform.python_version(),
    "platform": platform.platform(), "numpy_version": np.__version__,
    "elapsed_seconds": perf_counter()-start,
}
report["metadata"] = run_metadata(report["inputs"], model="sm181_local", boundary="periodic Fourier mode; finite velocity quadrature")

assert error < 1e-12
assert max(report["homogeneous_rhs_max"], report["density_rhs_max"]) < 1e-12
assert abs(eigenvalues[-1]) < 1e-12
assert report["nonzero_spectrum_error"] < 1e-12
(output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
fig, ax = plt.subplots(figsize=(6.2, 3.7))
ax.plot(np.arange(eigenvalues.size), eigenvalues, ".")
ax.axhline(-rate, linestyle="--", label="Analytic nonzero eigenvalue")
ax.set(xlabel="Sorted mode index", ylabel="Collision eigenvalue")
ax.legend()
fig.tight_layout()
fig.savefig(output / "spectrum.png", dpi=180)
plt.close(fig)
print(json.dumps(report, indent=2), flush=True)
print(f"PASS. Saved summary.json and spectrum.png in {output}", flush=True)

# Time evolution: entropy-scaled backward Euler solved by SOLVAX PCG.
final_time = 1.0
step_counts = [20, 40, 80, 160]
sj = jnp.asarray(root)
mj = jnp.asarray(measure)
y0 = sj * jnp.asarray(h + 0.2)
time_rows = []

def evolve(dt, steps):
    def step(y, unused):
        def action(v):
            return v + dt * rate * (v - sj * jnp.vdot(sj, v) / n0)
        solution = pcg_linear_solve(action, y, rtol=1e-12, atol=1e-14, max_steps=20)
        residual = jnp.linalg.norm(action(solution.x)-y) / jnp.linalg.norm(y)
        return solution.x, (solution.converged, residual)
    return jax.lax.scan(step, y0, None, length=steps)

for steps in step_counts:
    dt = final_time / steps
    compiled = jax.jit(lambda: evolve(dt, steps))
    with progress(f"Compiling uniform time evolution: {steps} steps"):
        evolved, (converged, residuals) = compiled()
        evolved.block_until_ready()
    if not bool(jnp.all(converged)) or float(jnp.max(residuals)) > 1e-11:
        raise RuntimeError(f"Uniform implicit solve failed at dt={dt}")
    numerical_h = np.asarray(evolved / sj)
    neutral = numerical_h - np.dot(measure, numerical_h) / n0
    amplitude = np.dot(measure*h, neutral) / np.dot(measure*h, h)
    fitted_rate = -np.log(amplitude) / final_time
    rate_error = abs(fitted_rate/rate-1)
    time_rows.append(dict(dt=dt, steps=steps, fitted_rate=float(fitted_rate),
                          relative_rate_error=float(rate_error),
                          density_error=float(abs(np.dot(measure, numerical_h)/n0-0.2)),
                          linear_residual=float(jnp.max(residuals))))
assert time_rows[-1]["relative_rate_error"] < 1e-3
assert all(time_rows[i]["relative_rate_error"] > 1.9*time_rows[i+1]["relative_rate_error"] for i in range(3))
report["time_evolution"] = time_rows
report["scope"] = "Eq.181 uniform pair algebra, entropy spectrum and backward-Euler time convergence"
(output / "summary.json").write_text(json.dumps(report, indent=2)+"\n")
fig, ax = plt.subplots(figsize=(6.2,3.7))
ax.loglog([r["dt"] for r in time_rows], [r["relative_rate_error"] for r in time_rows], "o-", label="SOLVAX backward Euler")
ax.set(xlabel="Timestep (normalized)", ylabel="Relative fitted-rate error")
ax.legend()
fig.tight_layout()
fig.savefig(output / "time_convergence.png", dpi=180)
plt.close(fig)
print(json.dumps(time_rows, indent=2), flush=True)
print("PASS: pair formula, unmodified spectrum and first-order time convergence", flush=True)
