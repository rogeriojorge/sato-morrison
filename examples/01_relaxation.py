"""Finite-amplitude relaxation of the specified local nonlinear surrogate."""
from pathlib import Path
import json
from time import perf_counter
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sato_morrison.collisions import uniform_grid, mobility_action
from sato_morrison.solver import (discrete_gradient_step, discrete_gradient_compiler,
                                entropy, invariant_diagnostics)
from sato_morrison.reference import constrained_equilibrium, progress, run_metadata

# Fixed dimensionless scales; D is a prescribed surrogate coefficient.
NX, NU, NMU = 7, 5, 3
U_MAX, MU_MAX = 3., 2.5
D, B, Q, MASS = .3, 1., 1., 1.
DT, STEPS, AMPLITUDE = .08, 10, .2
SHOW_FIGURES = False
OUTPUT = Path(__file__).resolve().parents[1] / 'results'
OUTPUT.mkdir(exist_ok=True)
(OUTPUT / 'figures').mkdir(exist_ok=True)
print(f'Model=sm_local_nonlinear; uniform B={B}; grid={NX}x{NU}x{NMU}; normalized units; '
      f'periodic x, natural no-flux velocity; entropy and full mu marginal checks; output={OUTPUT}', flush=True)

x = np.arange(NX)*2*np.pi/NX
u = np.linspace(-U_MAX,U_MAX,NU)
mu = np.linspace(.1,MU_MAX,NMU)
grid = uniform_grid(x,u,mu,magnetic_field=B,mass=MASS,charge=Q,model='sm_local_nonlinear')
base = jnp.exp(-grid.energy-.3*jnp.tile(jnp.asarray(mu),NX*NU))
pattern = np.cos(x)[:,None,None]*(u[None,:,None]**2/4-.5)
initial = base*jnp.exp(AMPLITUDE*jnp.asarray(np.broadcast_to(pattern,grid.shape)).ravel())
# Match only the overall chosen population once, never repair a timestep.
initial *= jnp.sum(grid.weights*base)/jnp.sum(grid.weights*initial)
beta, target = constrained_equilibrium(grid.energy,grid.weights,grid.mu_index,initial)
print(f'Candidate constrained maximum entropy beta={beta:.8g}; local extra null modes can prevent attraction.', flush=True)
compile_start = perf_counter()
with progress('Compile nonlinear residual and Jacobian'):
    compiler = discrete_gradient_compiler(grid,collision_strength=D)
    compiler[0](jnp.log(initial),jnp.log(initial),DT).block_until_ready()
    compiler[1](jnp.log(initial),jnp.log(initial),DT).block_until_ready()

compile_s = perf_counter()-compile_start
state = initial
times, entropies, qvalues, residuals, identity_errors, minima = [0.], [float(entropy(state,grid.weights))], [], [], [], [float(state.min())]
qvalues.append(float(jnp.sum(grid.weights*(state-target)**2/target)))
start = perf_counter()
with progress('Solve positive nonlinear discrete-gradient steps'):
    for step_number in range(STEPS):
        solution = discrete_gradient_step(grid,state,DT,collision_strength=D,rtol=2e-12,
                                         compiled_residual=compiler)
        state = solution.f
        times.append((step_number+1)*DT)
        entropies.append(float(entropy(state,grid.weights)))
        qvalues.append(float(jnp.sum(grid.weights*(state-target)**2/target)))
        residuals.append(solution.relative_residual)
        identity_errors.append(solution.entropy_identity_error)
        minima.append(solution.minimum)
        print(f'  step={step_number+1}/{STEPS}; residual={solution.relative_residual:.2e}; '
              f'deltaS={solution.entropy_change:.3e}; min f={solution.minimum:.3e}', flush=True)
elapsed = perf_counter()-start
checks = invariant_diagnostics(grid,state,initial)
if max(checks[key] for key in ('number_error','energy_error','marginal_error'))>1e-9:
    raise RuntimeError(f'Conservation budget failed: {checks}')
if np.diff(entropies).min() < -2e-11:
    raise RuntimeError('Entropy decrease exceeded the residual-based allowance')
# Independently vary the timestep at fixed final time and quadrature.
refinements = []
refinement_states = []
with progress('Check nonlinear timestep refinement'):
    for timestep in (DT, DT/2, DT/4):
        refined = initial
        refined_start = perf_counter()
        for _ in range(int(round(STEPS*DT/timestep))):
            refined = discrete_gradient_step(grid,refined,timestep,collision_strength=D,
                         rtol=2e-12,compiled_residual=compiler).f
        refinements.append({'dt':timestep,'invariants':invariant_diagnostics(grid,refined,initial),
                            'entropy':float(entropy(refined,grid.weights)),
                            'warm_s':perf_counter()-refined_start})
        refinement_states.append(np.asarray(refined))
    weighted = np.asarray(grid.weights/target)
    timestep_errors = [float(np.sqrt(np.sum(weighted*(refinement_states[i]-refinement_states[i+1])**2)))
                       for i in range(2)]
    timestep_order = float(np.log2(timestep_errors[0]/timestep_errors[1]))
    if timestep_errors[-1]/np.sqrt(float(jnp.sum(grid.weights*target)))>1e-3:
        raise RuntimeError('Nonlinear timestep refinement did not reach 1e-3 distribution error')
    print(f'  consecutive weighted errors={timestep_errors}; observed order={timestep_order:.4f}',flush=True)

# Fourier reconstruction is checked at 8x the represented spatial grid.
coefficients = np.fft.fft(np.asarray(state).reshape(grid.shape),axis=0)/NX
fine_x = np.arange(8*NX)*2*np.pi/(8*NX)
reconstruction = np.einsum('ak,kij->aij',np.exp(1j*np.outer(fine_x,np.fft.fftfreq(NX,d=1/NX))),coefficients).real
reconstruction_min = float(reconstruction.min())
# Triangle inequality certifies all x, rather than only sampled x values.
positivity_bound = float(np.min(coefficients[0].real-np.sum(np.abs(coefficients[1:]),axis=0)))
if positivity_bound<=0:
    raise RuntimeError(f'Continuous spatial positivity certificate failed: {positivity_bound}')
if reconstruction_min <= 0:
    raise RuntimeError(f'Spatial Fourier reconstruction loses positivity: {reconstruction_min}')
report = {**run_metadata({'nx':NX,'nu':NU,'nmu':NMU,'u_domain':[-U_MAX,U_MAX],
          'mu_domain':[.1,MU_MAX],'dt':DT,'steps':STEPS,'D':D,'B':B,'q':Q,'m':MASS,
          'amplitude':AMPLITUDE,'initial':'normalized exp(-E-.3mu+.2cos(x)(u²/4-.5))',
          'seed':None,'quadrature':'periodic x trapezoid; u and mu trapezoid'},
          model='sm_local_nonlinear',boundary='periodic x; natural zero collision flux u'),
          'status':'passed','equation':'fixed-field nonlinear weak SM local surrogate, P Ix P',
          'normalization':'dimensionless constant D; no physical rate calibration',
          'invariants':checks,'candidate_beta':beta,'newton_max_residual':max(residuals),
          'entropy_identity_max_error':max(map(abs,identity_errors)),
          'entropy_min_step':float(np.diff(entropies).min()),'reconstruction_min':reconstruction_min,'continuous_spatial_positivity_bound':positivity_bound,
          'state_reconstruction':'Fourier in x; positive piecewise-linear interpolation in u,mu',
          'times':times,'entropy':entropies,'candidate_distance':qvalues,
          'timestep_refinements':refinements,'consecutive_timestep_errors':timestep_errors,
          'observed_timestep_order':timestep_order,'compile_s':compile_s,'warm_s':elapsed}
(OUTPUT/'relaxation.json').write_text(json.dumps(report,indent=2)+'\n')
fig,axes = plt.subplots(1,2,figsize=(9,3.4))
axes[0].plot(times,np.array(entropies)-entropies[0],'o-');axes[0].set(xlabel='time',ylabel='entropy increase')
axes[1].semilogy(times,np.array(qvalues)/qvalues[0],'o-');axes[1].set(xlabel='time',ylabel='distance to candidate / initial')
fig.suptitle('Specified local nonlinear surrogate; fixed uniform field');fig.tight_layout()
fig.savefig(OUTPUT/'figures/relaxation.png',dpi=170)
if SHOW_FIGURES:
    plt.show()
plt.close(fig)
print(f'Completed nonlinear run in {elapsed:.3f}s; invariant errors={checks}; entropy identity '
      f'error={report["entropy_identity_max_error"]:.3e}; saved {OUTPUT/"relaxation.json"}',flush=True)
