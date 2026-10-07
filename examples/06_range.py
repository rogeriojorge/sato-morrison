"""Uniform finite-range surrogate: exact Fourier branches and independent weak sum."""

from pathlib import Path
import csv
import json
import time

import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sato_morrison.reference import progress, range_fourier_rates, range_pair_matrix, run_metadata

jax.config.update("jax_enable_x64", True)
spatial_grids = [16, 32, 64]
velocity_nodes = [3, 5, 9]
widths = [0.15, 0.3, 0.6]
mode = 1
length = 2 * np.pi
field_strength = 1.7
charge = -1.3
diffusion = 0.8
number_density = 1.0
show_figures = False
output = Path(__file__).resolve().parents[1] / "results" / "finite_range"
output.mkdir(parents=True, exist_ok=True)

print("Model: sm_finite_range; normalized periodized Gaussian, uniform B parallel z", flush=True)
print(f"Periodic perpendicular x, exact Fourier derivative; nx={spatial_grids}, Nv={velocity_nodes}")
print("Units: fixed normalized coordinates; collision only; all inputs fixed during refinement")
print("Comparison: pair Gram spectrum against density and velocity-neutral analytic rates")
print(f"Destination: {output}", flush=True)
rows = []
started = time.perf_counter()
with progress("Assembling independent ordered spatial and velocity pair weak sums"):
    for nv in velocity_nodes:
        # Positive resolved velocity weights; normalize once at setup to the fixed n0.
        masses = np.exp(-np.linspace(-2, 2, nv)**2 / 2)
        masses = number_density * masses / masses.sum()
        for width in widths:
            for nx in spatial_grids:
                begin = time.perf_counter()
                matrix, ratio = range_pair_matrix(nx, masses, mode, width, length=length,
                    diffusion=diffusion, charge=charge, field=field_strength)
                spectrum = np.linalg.eigvalsh(matrix)
                density_rate, neutral_rate = range_fourier_rates(masses, mode, width,
                    diffusion=diffusion, charge=charge, field=field_strength)
                root_mass = np.sqrt(masses / masses.sum())
                predicted = neutral_rate * (np.eye(nv) - ratio * np.outer(root_mass, root_mass))
                residual = np.linalg.norm(matrix - predicted) / neutral_rate
                error = abs(spectrum[0] - density_rate) / density_rate
                rows.append({"nx": nx, "nv": nv, "width": width,
                    "density_rate": float(spectrum[0]), "density_exact": density_rate,
                    "neutral_rate": float(spectrum[-1]), "neutral_exact": neutral_rate,
                    "density_relative_error": error, "pair_matrix_relative_residual": residual,
                    "sampled_kernel_ratio": ratio, "seconds": time.perf_counter() - begin,
                    "status": ("failed" if residual >= 1e-11 else
                               "passed" if error < 1e-6 else "unresolved")})
with (output / "summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
inputs = {"spatial_grids": spatial_grids, "velocity_nodes": velocity_nodes,
          "widths": widths, "mode": mode, "length": length, "field_strength": field_strength,
          "charge": charge, "diffusion": diffusion, "number_density": number_density,
          "velocity_weights": "positive normalized sampled Gaussian; exact finite-quadrature theorem"}
metadata = run_metadata(inputs, model="sm_finite_range", boundary="periodic perpendicular x")
metadata.update({"rows": rows, "elapsed_s": time.perf_counter() - started,
    "novelty": "derived and independently checked surrogate spectrum; publication priority unresolved",
    "limitations": "No Coulomb tensor calibration, nonuniform finite-range convergence, or physical spectral gap."})
(output / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
kwidth = np.geomspace(0.015, 4, 150)
axes[0].loglog(kwidth, -np.expm1(-kwidth**2 / 2), label="density / neutral rate")
axes[0].loglog(kwidth[:85], kwidth[:85]**2 / 2, "--", label="long-wave prediction")
axes[0].set(xlabel="range × transverse wave number", ylabel="rate ratio",
            title="Density nullspace lifts at finite range")
axes[0].legend()
for width in widths:
    selected = [r for r in rows if r["nv"] == 5 and r["width"] == width]
    axes[1].semilogy(spatial_grids, [max(r["density_relative_error"], 1e-15) for r in selected],
                     "o-", label=f"range={width}")
axes[1].set(xlabel="periodic spatial nodes", ylabel="density rate relative error",
            title="Resolve the chosen range")
axes[1].legend()
k = np.geomspace(0.01, 2, 100)
rate = diffusion * number_density * k**2 / (charge * field_strength)**2
width = widths[1]
axes[2].loglog(k, rate * (-np.expm1(-width**2 * k**2 / 2)), label="Gaussian exact")
axes[2].loglog(k, rate * width**2 * k**2 / 2, "--", label="quartic prediction")
axes[2].set(xlabel="transverse wave number", ylabel="density decay rate",
            title="No long-wave continuum gap")
axes[2].legend()
fig.savefig(output / "range.png", dpi=160)
if show_figures:
    plt.show()
plt.close(fig)
finest_error = max(r["density_relative_error"] for r in rows if r["nx"] == spatial_grids[-1])
print(f"Complete in {metadata['elapsed_s']:.3f} s; finest spatial relative error={finest_error:.3e}")
if any(row["status"] == "failed" for row in rows) or finest_error > 1e-10:
    raise RuntimeError("Finite-range validation failed; inspect saved evidence.")
