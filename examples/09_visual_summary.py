"""Reproduce the README animation and evidence panels from checked operators/data."""
from pathlib import Path
from hashlib import sha256
import json
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter, writers
from sato_morrison.reference import range_pair_matrix, run_metadata, progress

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "visual_summary"
OUTPUT.mkdir(parents=True, exist_ok=True)
NX, NV, WIDTH = 64, 9, .6
D, B, CHARGE, N0, MODE = .8, 1.7, -1.3, 1., 1
DENSITY_AMPLITUDE, NEUTRAL_AMPLITUDE = .25, .20
TAU = np.linspace(0, 8, 65)
FPS = 8
TEAL, ORANGE, INK = "#147D92", "#C66B28", "#253447"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.labelcolor": INK, "text.color": INK,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
print("Uniform linear surrogate; fixed dimensionless coefficients; no physical clock.", flush=True)
print("Assembling the ordered-pair matrix before animation; no compilation required.", flush=True)
started = perf_counter()
velocity_nodes = np.linspace(-2, 2, NV)
masses = np.exp(-velocity_nodes**2/2)
masses *= N0/masses.sum()
root_mass = np.sqrt(masses)
neutral = velocity_nodes**2 - np.dot(masses, velocity_nodes**2)/N0
neutral /= np.max(np.abs(neutral))
initial = DENSITY_AMPLITUDE + NEUTRAL_AMPLITUDE*neutral
rate = D*N0*MODE**2/(CHARGE*B)**2
matrix, sampled_ratio = range_pair_matrix(NX, masses, MODE, WIDTH,
    diffusion=D, charge=CHARGE, field=B)
local_matrix = rate*(np.eye(NV)-np.outer(root_mass, root_mass)/N0)
exact_ratio = np.exp(-WIDTH**2*MODE**2/2)
with progress("Evolve the two independently assembled Fourier matrices"):
    local = np.array([expm(-local_matrix*t/rate) @ (root_mass*initial)/root_mass for t in TAU])
    finite = np.array([expm(-matrix*t/rate) @ (root_mass*initial)/root_mass for t in TAU])
expected_local = DENSITY_AMPLITUDE + NEUTRAL_AMPLITUDE*np.exp(-TAU[:,None])*neutral
expected_finite = (DENSITY_AMPLITUDE*np.exp(-(1-exact_ratio)*TAU[:,None])
                   + NEUTRAL_AMPLITUDE*np.exp(-TAU[:,None])*neutral)
error = max(np.max(np.abs(local-expected_local)), np.max(np.abs(finite-expected_finite)))
if error > 1e-11 or max(np.max(np.abs(local)), np.max(np.abs(finite))) >= 1:
    raise RuntimeError(f"Animation operator/positivity verification failed: error={error}")
x = np.linspace(0, 2*np.pi, 193)
local_frames = local[:,:,None]*np.cos(MODE*x)
finite_frames = finite[:,:,None]*np.cos(MODE*x)
np.savez_compressed(OUTPUT/"evolution.npz", tau=TAU, x=x, velocity_nodes=velocity_nodes,
                    masses=masses, local=local_frames, finite_range=finite_frames)

fig = plt.figure(figsize=(10.4, 6.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1, .8], width_ratios=[1, 1, .055],
                     left=.09, right=.94, bottom=.10, top=.84, hspace=.58, wspace=.25)
axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
images = []
for ax, values, title in zip(axes, [local_frames, finite_frames],
                           ["Local: density structure survives", "Finite range: density also decays"]):
    images.append(ax.imshow(values[0], origin="lower", aspect="auto", cmap="RdBu_r",
                            vmin=-.45, vmax=.45, extent=(0, 2*np.pi, -.5, NV-.5)))
    ax.set(title=title, xlabel="perpendicular position x", ylabel="velocity-node index",
           xticks=[0,np.pi,2*np.pi], xticklabels=["0", "π", "2π"], yticks=[0,4,8])
fig.colorbar(images[0], cax=fig.add_subplot(gs[0,2]), label="relative perturbation δf / f₀")
trace = fig.add_subplot(gs[1,:2])
trace.plot(TAU, np.ones_like(TAU), color=INK, lw=2, label="local density")
trace.plot(TAU, np.exp(-(1-exact_ratio)*TAU), color=ORANGE, lw=2, label="finite-range density")
trace.plot(TAU, np.exp(-TAU), color=TEAL, lw=2, label="velocity-neutral (both)")
cursor = trace.axvline(0, color=INK, lw=1, ls=":")
trace.set(xlim=(0,8), ylim=(0,1.1), xlabel="normalized time τ = λt", ylabel="mode amplitude / initial")
trace.legend(loc="upper center", bbox_to_anchor=(.5,1.29), ncol=3, frameon=False, fontsize=9)
fig.suptitle("What changes when collisions have finite spatial range?", y=.98, fontsize=15, weight="bold")
clock_text = fig.text(.5, .89, "", ha="center", fontsize=11)
fig.text(.5,.015,"Uniform linear surrogate • Gaussian range ℓ = 0.6 • fixed color scale • finite velocity quadrature",
         ha="center", fontsize=9, color=INK)

def draw(frame):
    for image, values in zip(images, [local_frames, finite_frames]):
        image.set_data(values[frame])
    cursor.set_xdata([TAU[frame], TAU[frame]])
    clock_text.set_text(f"τ = {TAU[frame]:.1f}     |     density decay rate / λ = {1-exact_ratio:.3f}")
    return *images, cursor, clock_text

draw(30)
fig.savefig(OUTPUT/"range_poster.png", dpi=160)
animation = FuncAnimation(fig, draw, frames=len(TAU), interval=1000/FPS, blit=False)
with progress("Render 65 checked evolution frames to GIF"):
    animation.save(OUTPUT/"range_evolution.gif", writer=PillowWriter(fps=FPS), dpi=90)
video_written = writers.is_available("ffmpeg")
if video_written:
    with progress("Encode the same frames as MP4"):
        animation.save(OUTPUT/"range_evolution.mp4", writer=FFMpegWriter(fps=FPS,
            codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]), dpi=140)
plt.close(fig)

sources = {}
for name in ["relaxation.json", "geometry.json", "benchmarks/summary.json"]:
    path = ROOT/"results"/name
    raw = path.read_bytes()
    result = json.loads(raw)
    sources[name] = {"sha256":sha256(raw).hexdigest(),
                     "commit":result.get("commit", result.get("metadata",{}).get("commit"))}
    if name == "relaxation.json": relaxation = result
    elif name == "geometry.json": geometry = result
    else: benchmarks = result

fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.5), layout="constrained")
ax = axes[0,0]
ax.plot(relaxation["times"], np.asarray(relaxation["entropy"])-relaxation["entropy"][0], "o-", color=TEAL)
ax.set(title="A  Nonlinear entropy increases", xlabel="normalized time", ylabel="S(t) − S(0)")
ax.text(.04,.86,"Number, energy, full μ marginal\nerrors < 10⁻¹⁵", transform=ax.transAxes, fontsize=10)
ax = axes[0,1]
for control, color, label in zip(geometry["controls"], [ORANGE, INK, TEAL],
                                ["collision only", "ideal only", "combined"]):
    ax.plot(control["times"], np.array(control["Q"])/control["Q"][0], "o-", ms=3, label=label, color=color)
ax.set(title="B  Toroidal decay depends on dynamics", xlabel="normalized time", ylabel="quadratic perturbation norm Q / Q₀")
ax.legend(frameon=False, fontsize=9)
ax = axes[1,0]
keys = ["nr","ntheta","nz","nu","nmu","dt","umax","mumax"]
values = [100*geometry["convergence"][key]["relative_dissipation_change"] for key in keys]
ax.barh(["R grid", "θ grid", "z grid", "u grid", "μ grid", "timestep", "u tail", "μ tail"], values, color=TEAL)
ax.axvline(1, ls="--", color=ORANGE, label="1% target")
ax.invert_yaxis()
ax.set(title="C  Independent toroidal refinements", xlabel="change in dissipated fraction (%)", xlim=(0,1.18))
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax = axes[1,1]
eigenvalues = np.asarray(geometry["spectrum"]["eigenvalues"])
ax.plot(np.arange(1,len(eigenvalues)+1),eigenvalues,".",color=TEAL,ms=4)
ax.set_yscale("symlog",linthresh=1e-14)
ax.axhspan(-1e-14,1e-14,color=INK,alpha=.07)
ax.annotate("39 raw near-zero eigenvalues\n39 independent known invariants", xy=(23,0), xytext=(60,1e-7),
            fontsize=9, arrowprops={"arrowstyle":"->", "color":INK})
ax.set(title="D  Null modes are retained", xlabel="sorted eigenvalue index (243-node grid)", ylabel="entropy-weighted mobility eigenvalue")
fig.savefig(OUTPUT/"validation.png",dpi=180)
plt.close(fig)

selected = [r for r in benchmarks["rows"] if r["nx"]==4 and r["nv"]==128]
fig, axes = plt.subplots(1,2,figsize=(10.4,3.5),layout="constrained")
labels = ["Dense", "PCG", "Diagonal PCG"]
for ax, key, scale, ylabel in [(axes[0],"warm_median_s",1e6,"warm time (μs; lower is better)"),
                              (axes[1],"compiled_memory",1/1024,"XLA temporary buffers (KiB)")]:
    values = [r[key]*scale if key!="compiled_memory" else r[key]["temp_size_in_bytes"]*scale for r in selected]
    ax.bar(labels,values,color=[INK,ORANGE,TEAL],width=.55)
    if key=="warm_median_s":
        bounds = np.array([[r["warm_median_s"]-r["warm_min_s"] for r in selected],
                           [r["warm_max_s"]-r["warm_median_s"] for r in selected]])*scale
        ax.errorbar(labels,values,yerr=bounds,fmt="none",color=INK,capsize=4)
    for i,value in enumerate(values):
        ax.annotate(f"{value:.1f}",(i,value),xytext=((25 if key=="warm_median_s" else 0),5),textcoords="offset points",ha="center",fontsize=9)
    ax.set(ylabel=ylabel)
fig.suptitle("Same implicit step, same error: 4 × 128 nodes on Apple M4",weight="bold")
fig.savefig(OUTPUT/"cost.png",dpi=180)
plt.close(fig)
metadata = run_metadata({"nx":NX,"nv":NV,"width":WIDTH,"D":D,"B":B,"charge":CHARGE,
    "n0":N0,"mode":MODE,"density_amplitude":DENSITY_AMPLITUDE,"neutral_amplitude":NEUTRAL_AMPLITUDE,
    "velocity_weights":"normalized exp(-v²/2) on [-2,2]; finite quadrature, not a physical 3V grid",
    "tau":TAU.tolist(),"fps":FPS,"time_normalization":"tau = D n0 k² t / (q B)²"},
    model="uniform linear local and finite-range surrogate",boundary="periodic perpendicular x")
metadata.update({"status":"passed","matrix_exponential_vs_exact_max_error":float(error),
    "sampled_kernel_ratio":sampled_ratio,"continuum_kernel_ratio":float(exact_ratio),
    "minimum_relative_distribution":float(1+min(local_frames.min(),finite_frames.min())),
    "source_results":sources,"mp4_written":video_written,"elapsed_s":perf_counter()-started,
    "limitations":"Exact uniform linear branches at finite velocity quadrature; no nonlinear or physical-rate claim."})
(OUTPUT/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
print(f"Saved {OUTPUT}; operator evolution max error={error:.3e}; elapsed={metadata['elapsed_s']:.1f}s",flush=True)
