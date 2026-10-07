"""Matched-error implicit solves for the exact uniform Fourier reduction."""

import json
import resource
import platform
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from solvax import pcg_linear_solve

from sato_morrison.reference import gauss_interval, progress, run_metadata

jax.config.update("jax_enable_x64", True)

# Inputs: independent spatial Fourier blocks; normalized Eq.181 coefficient.
velocity_sizes = [32, 64, 128]
spatial_blocks = [1, 4]
repetitions = 5
rate, dt, rtol = 0.3, 0.025, 1e-12
output = Path(__file__).resolve().parents[1] / "results" / "benchmarks"
output.mkdir(parents=True, exist_ok=True)
print("sm181_local uniform; independent periodic Fourier blocks; float64; normalized units", flush=True)
print(f"Matched solve residual <1e-10, {repetitions} synchronized warm repeats; destination {output}", flush=True)
rows = []
for nv in velocity_sizes:
    u, w = gauss_interval(nv, -3., 3.)
    measure = w * np.exp(-u*u/2)
    measure /= measure.sum()
    m = jnp.asarray(measure)
    for nx in spatial_blocks:
        h = jnp.broadcast_to(jnp.asarray(0.2 + u*u - np.dot(measure, u*u)), (nx, nv))
        rhs = h*m
        def action(v):
            return m*v + dt*rate*m*(v-jnp.sum(m*v, axis=-1, keepdims=True))
        diagonal = m + dt*rate*m*(1-m)
        matrix = jnp.diag(m) + dt*rate*(jnp.diag(m)-jnp.outer(m,m))
        expected = jnp.sum(m*h,axis=-1,keepdims=True)+(h-jnp.sum(m*h,axis=-1,keepdims=True))/(1+dt*rate)
        exact = jnp.sum(m*h,axis=-1,keepdims=True)+(h-jnp.sum(m*h,axis=-1,keepdims=True))*jnp.exp(-dt*rate)
        for route in ["dense", "pcg", "diagonal_pcg"]:
            def solve(b):
                if route == "dense":
                    return jnp.linalg.solve(matrix, b.T).T, jnp.array(True), jnp.array(0)
                precond = (lambda v: v/diagonal) if route == "diagonal_pcg" else None
                result = pcg_linear_solve(action,b,precond=precond,rtol=rtol,atol=1e-14,max_steps=2000)
                return result.x,result.converged,result.iterations
            compiled = jax.jit(solve)
            with progress(f"Compiling {route}: Nx={nx}, Nv={nv}"):
                start=perf_counter()
                solved,success,iterations=compiled(rhs)
                solved.block_until_ready()
                compile_s=perf_counter()-start
            times=[]
            for repeat in range(repetitions):
                start=perf_counter()
                solved,success,iterations=compiled(rhs)
                solved.block_until_ready()
                times.append(perf_counter()-start)
            residual=float(jnp.linalg.norm(action(solved)-rhs)/jnp.linalg.norm(rhs))
            error=float(jnp.linalg.norm(solved-expected)/jnp.linalg.norm(expected))
            physical_error=float(jnp.sqrt(jnp.sum(m*(solved-exact)**2)/jnp.sum(m*exact**2)))
            if not bool(success) or residual>1e-10 or error>1e-9:
                raise RuntimeError(f"{route} failed: residual={residual:g}, solve error={error:g}")
            analysis=compiled.lower(rhs).compile().memory_analysis()
            memory={name:int(getattr(analysis,name,0)) for name in
                    ["argument_size_in_bytes","output_size_in_bytes","temp_size_in_bytes","alias_size_in_bytes"]}
            row=dict(route=route,nx=nx,nv=nv,dt=dt,rtol=rtol,solve_error=error,
                     residual=residual,time_discretization_error=physical_error,compile_and_first_s=compile_s,
                     warm_median_s=float(np.median(times)),warm_min_s=min(times),warm_max_s=max(times),
                     iterations=int(iterations),state_bytes=nx*nv*8,compiled_memory=memory,
                     process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if platform.system()=="Darwin" else 1024),status="passed")
            rows.append(row)
            print(row,flush=True)
metadata=run_metadata(dict(velocity_sizes=velocity_sizes,spatial_blocks=spatial_blocks,
                          repetitions=repetitions,rate=rate,dt=dt,rtol=rtol),
                      model="sm181_local exact separable uniform reduction",boundary="periodic Fourier blocks")
metadata["cost_scope"]="MacOS process RSS is cumulative high-water mark in bytes; compiled memory is XLA buffer accounting. Compile includes first execution. No pair tensor is stored or evaluated in this separable limit. No general-kernel speedup claim."
(output/"summary.json").write_text(json.dumps(dict(metadata=metadata,rows=rows),indent=2)+"\n")
fig,ax=plt.subplots(figsize=(6.2,3.7))
for route in ["dense","pcg","diagonal_pcg"]:
    data=[r for r in rows if r["route"]==route and r["nx"]==4]
    ax.loglog([r["nv"] for r in data],[r["warm_median_s"] for r in data],"o-",label=route)
ax.set(xlabel="Velocity nodes per block (four Fourier blocks)",ylabel="Warm median solve time (s)")
ax.legend();fig.tight_layout();fig.savefig(output/"cost.png",dpi=180);plt.close(fig)
print("PASS: all routes meet the same implicit-step error target",flush=True)

# General weak pair machinery at matched action error, including chunk scaling.
from sato_morrison.collisions import uniform_grid,mobility_action,dense_mobility

pair_rows=[]
for nx in [5,9]:
    for nu in [3,5]:
        grid=uniform_grid(np.arange(nx)*2*np.pi/nx,np.linspace(-2.,2.,nu),np.linspace(.1,1.5,3))
        f=jnp.exp(-grid.energy)
        h=jnp.asarray(np.cos(np.arange(grid.size)*.17))
        with progress(f'Assembling independent dense weak reference Nx={nx}, Nv={nu*3}'):
            dense=jnp.asarray(dense_mobility(grid,f))
            expected=dense@h
        for chunk in [None,16,128]:
            compiled=jax.jit(lambda state:mobility_action(grid,f,state,chunk_size=chunk))
            with progress(f'Compiling pair weak action Nx={nx}, Nv={nu*3}, chunk={chunk}'):
                start=perf_counter();result=compiled(h);result.block_until_ready();compile_s=perf_counter()-start
            times=[]
            for repeat in range(repetitions):
                start=perf_counter();result=compiled(h);result.block_until_ready();times.append(perf_counter()-start)
            error=float(jnp.linalg.norm(result-expected)/jnp.linalg.norm(expected))
            if error>1e-11:
                raise RuntimeError(f'Chunk action failed independent dense comparison: {error}')
            analysis=compiled.lower(h).compile().memory_analysis()
            pair_rows.append(dict(nx=nx,nv=nu*3,chunk=chunk,pairs=len(grid.left),
                                  action_error=error,compile_and_first_s=compile_s,
                                  warm_median_s=float(np.median(times)),warm_min_s=min(times),warm_max_s=max(times),
                                  state_bytes=grid.size*8,compiled_temporary_bytes=int(analysis.temp_size_in_bytes),
                                  process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if platform.system()=="Darwin" else 1024),status='passed'))
metadata['inputs']['weak_pairs']=dict(nx=[5,9],nu=[3,5],nmu=3,u_extent=2.,mu_domain=[.1,1.5],chunks=[None,16,128],test='cos(0.17*flat_node_index)',f='exp(-E)')
(output/'summary.json').write_text(json.dumps(dict(metadata=metadata,rows=rows,pair_actions=pair_rows),indent=2)+'\n')
print('PASS: 12 weak pair benchmarks meet independent dense action error <1e-11',flush=True)
