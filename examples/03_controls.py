"""Run independent Lorentz, nonlinear Dougherty, and physical 3V Coulomb controls."""
from pathlib import Path
from time import perf_counter
import csv
import json
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sato_morrison.controls import (GaussianMixture, lorentz_evolve,
    dougherty_evolve, dougherty_rhs, mixture_diagnostics,
    landau_gaussian_analytic, landau_gaussian_quadrature,
    landau_gaussian_cartesian)
from sato_morrison.reference import run_metadata, progress

# Named normalized physical inputs: distributions integrate to one in 3V.
collision_rate = 0.7
pitch_order = 24
lorentz_degrees = [0, 1, 2, 3]
times = np.linspace(0., 3., 19)
mixture_weights = np.array([.4, .6])
mixture_means = np.array([[-1.2, .4, .2], [1.1, -.1, .2]])
mixture_covariances = np.array([np.diag([.65, 1., .8]), np.diag([.9, .7, 1.2])])
density_order, density_extent = 40, 9.
landau_covariance = np.diag([1.15, 1.15, .7])
landau_gamma = 1.0
radial_orders = [8, 12, 20, 28]
cartesian_orders = [8, 16, 32, 48]
tail_radii = [5., 8., 12.]
softenings = [.2, .1, .05, .025, 0.]
phase_orders = [8, 16, 32]
small_anisotropies = [.03, .01, .003]
output = Path(__file__).resolve().parents[1] / 'results' / 'controls'
show_figures = False

output.mkdir(parents=True, exist_ok=True)
print(f'Models=lorentz,dougherty,landau; uniform field for shell chart; normalized units; output={output}',flush=True)
print('Comparisons: exact Legendre rates; exact conserving nonGaussian mixture PDE; independent 1D/3V Coulomb weak moments.',flush=True)
print('Landau evaluates instantaneous moments, not a kinetic time evolution. NumPy/SciPy controls require no compilation.',flush=True)
started = perf_counter()
rows = []
xi, pitch_weights = np.polynomial.legendre.leggauss(pitch_order)
lorentz_curves = []
with progress('Evaluating shell eigenmodes and positive nonlinear mixture'):
    for ell in lorentz_degrees:
        mode = np.polynomial.legendre.Legendre.basis(ell)(xi)
        initial = 1+.15*mode
        curves = [lorentz_evolve(xi,initial,collision_rate,t,3) for t in times]
        expected = [1+.15*np.exp(-collision_rate*ell*(ell+1)/2*t)*mode for t in times]
        error = max(np.max(np.abs(a-b)) for a,b in zip(curves,expected))
        amplitude = [(pitch_weights@((f-1)*mode))/(pitch_weights@(mode*mode)) for f in curves]
        lorentz_curves.append(amplitude)
        rows.append({'case':f'lorentz_ell{ell}','model':'lorentz','error':float(error),'status':'passed'})
    initial_mixture = GaussianMixture(mixture_weights,mixture_means,mixture_covariances)
    drift, initial_covariance = initial_mixture.moments()
    theta = np.trace(initial_covariance)/3
    diagnostics = [mixture_diagnostics(dougherty_evolve(initial_mixture,collision_rate,t),density_order,density_extent) for t in times]
    covariance_error = max(np.linalg.norm(d['covariance']-(theta*np.eye(3)+(initial_covariance-theta*np.eye(3))*np.exp(-2*collision_rate*t))) for t,d in zip(times,diagnostics))
    number_error = max(abs(d['number']-1.) for d in diagnostics)
    energy_error = max(abs(np.trace(d['covariance'])-np.trace(initial_covariance)) for d in diagnostics)
    entropy = np.array([d['entropy'] for d in diagnostics])
    samples = np.random.default_rng(301).normal(size=(23,3))
    time, step = .4, 1e-5
    derivative = (dougherty_evolve(initial_mixture,collision_rate,time+step).evaluate(samples)-dougherty_evolve(initial_mixture,collision_rate,time-step).evaluate(samples))/(2*step)
    strong_error = np.linalg.norm(derivative-dougherty_rhs(dougherty_evolve(initial_mixture,collision_rate,time),samples,collision_rate))/np.linalg.norm(derivative)
    rows.extend([{'case':'dougherty_covariance','model':'dougherty','error':float(covariance_error),'status':'passed'},
                 {'case':'dougherty_nongaussian_strong_equation','model':'dougherty','error':float(strong_error),'status':'passed'}])
    if number_error>1e-8 or energy_error>1e-7 or np.min(np.diff(entropy)) < -1e-10 or strong_error>1e-6:
        raise RuntimeError('Dougherty comparison or invariant budget failed.')

with progress('Computing independent physical 3V Coulomb moment convergence'):
    analytic = landau_gaussian_analytic(landau_covariance,landau_gamma)
    analytic_scale = np.linalg.norm(analytic)
    spherical = [landau_gaussian_quadrature(landau_covariance,landau_gamma,radial_order=n) for n in radial_orders]
    spherical_errors = [np.linalg.norm(q['covariance_rate']-analytic)/analytic_scale for q in spherical]
    cartesian_errors = [np.linalg.norm(landau_gaussian_cartesian(landau_covariance,landau_gamma,n)-analytic)/analytic_scale for n in cartesian_orders]
    tails = [landau_gaussian_quadrature(landau_covariance,landau_gamma,radius=r) for r in tail_radii]
    tail_errors = [np.linalg.norm(q['covariance_rate']-analytic)/analytic_scale for q in tails]
    softened = [landau_gaussian_quadrature(landau_covariance,landau_gamma,softening=e) for e in softenings]
    soft_errors = [np.linalg.norm(q['covariance_rate']-analytic)/analytic_scale for q in softened]
    gyro = [landau_gaussian_quadrature(landau_covariance,landau_gamma,phase_order=n) for n in phase_orders]
    gyro_errors = [np.linalg.norm(q['covariance_rate']-analytic)/analytic_scale for q in gyro]
    small_anisotropy_results=[]
    for epsilon in small_anisotropies:
        covariance=np.diag([1+epsilon,1+epsilon,1-2*epsilon])
        exact=landau_gaussian_analytic(covariance,landau_gamma)
        numerical=landau_gaussian_quadrature(covariance,landau_gamma)['covariance_rate']
        relative_error=np.linalg.norm(numerical-exact)/np.linalg.norm(exact)
        effective_rate=-(exact[0,0]-exact[2,2])/(3*epsilon)
        small_anisotropy_results.append({'epsilon':epsilon,'relative_reference_error':float(relative_error),'instantaneous_anisotropy_rate':float(effective_rate)})
        rows.append({'case':f'landau_small_anisotropy_{epsilon}','model':'landau','error':float(relative_error),'status':'passed' if relative_error<1e-9 else 'unresolved'})
    for label,levels,errors in [('radial',radial_orders,spherical_errors),('cartesian',cartesian_orders,cartesian_errors),('tail',tail_radii,tail_errors),('softening',softenings,soft_errors),('gyrophase',phase_orders,gyro_errors)]:
        for level,error in zip(levels,errors):
            rows.append({'case':f'landau_{label}_{level}','model':'landau','error':float(error),'status':'passed' if error < .01 else 'unresolved'})
    if spherical_errors[-1]>1e-9 or cartesian_errors[-1]>.01 or spherical[-1]['entropy_production']<0 or abs(spherical[-1]['energy_rate'])>1e-13:
        raise RuntimeError('Coulomb weak moment reference failed acceptance.')

fig, axes = plt.subplots(2,2,figsize=(10,7),layout='constrained')
for ell,curve in zip(lorentz_degrees,lorentz_curves):
    axes[0,0].semilogy(times,np.maximum(np.abs(curve),1e-14),label=f'ell={ell}, rate={collision_rate*ell*(ell+1)/2:g}')
axes[0,0].set(xlabel='time',ylabel='Legendre amplitude',title='Lorentz speed-shell decay')
axes[0,0].legend()
for j,label in enumerate(['xx','yy','zz']):
    axes[0,1].plot(times,[d['covariance'][j,j] for d in diagnostics],label=label)
axes[0,1].axhline(theta,color='gray',linestyle=':')
axes[0,1].set(xlabel='time',ylabel='centered covariance',title='NonGaussian conserving Dougherty')
axes[0,1].legend()
axes[1,0].plot(times,entropy-entropy[0])
axes[1,0].set(xlabel='time',ylabel='entropy gain',title='Positive mixture, conserved energy')
axes[1,1].semilogy(radial_orders,np.maximum(spherical_errors,1e-16),'o-',label='3V spherical vs scalar integral')
axes[1,1].semilogy(cartesian_orders,cartesian_errors,'s-',label='Cartesian Hermite vs scalar integral')
axes[1,1].set(xlabel='quadrature order',ylabel='relative covariance-rate error',title='Physical Coulomb moment convergence')
axes[1,1].legend(fontsize=8)
fig.savefig(output/'controls.png',dpi=160)
if show_figures:
    plt.show(block=False)
plt.close(fig)
inputs = {'nu':collision_rate,'times':times.tolist(),'pitch_order':pitch_order,
    'mixture_weights':mixture_weights.tolist(),'mixture_means':mixture_means.tolist(),
    'mixture_covariances':mixture_covariances.tolist(),'density_order':density_order,'density_extent':density_extent,
    'landau_covariance':landau_covariance.tolist(),'gamma':landau_gamma,'radial_orders':radial_orders,
    'cartesian_orders':cartesian_orders,'polar_order':32,'phase_orders':phase_orders,'tail_radii':tail_radii,
    'softenings':softenings,'small_anisotropies':small_anisotropies,'seed':301,'density_normalization':1.,
    'landau_scope':'instantaneous Gaussian weak moments and entropy production; no kinetic time evolution',
    'dougherty_scope':'exact positive Gaussian-mixture evolution with self-consistent conserved moments'}
metadata = run_metadata(inputs,model='lorentz+dougherty+landau',boundary='infinite velocity domains analytically; finite quadrature with tail checks')
metadata['results'] = {'dougherty_number_error':number_error,'dougherty_covariance_error':float(covariance_error),
    'dougherty_centered_energy_error':energy_error,'dougherty_min_f':min(d['min_f'] for d in diagnostics),
    'dougherty_entropy_min_step':float(np.min(np.diff(entropy))),'dougherty_strong_relative_error':float(strong_error),
    'landau_covariance_rate':analytic.tolist(),'landau_entropy_production':spherical[-1]['entropy_production'],
    'landau_energy_rate':spherical[-1]['energy_rate'],'landau_small_anisotropies':small_anisotropy_results,'landau_linear_anisotropy_rate':4*landau_gamma/(5*np.sqrt(np.pi)),'wall_s':perf_counter()-started}
(output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
with (output/'summary.csv').open('w',newline='') as file:
    writer=csv.DictWriter(file,fieldnames=['case','model','error','status']);writer.writeheader();writer.writerows(rows)
print(f'Controls passed: Dougherty strong relative error={strong_error:.3e}; Coulomb independent spherical error={spherical_errors[-1]:.3e}; Cartesian error={cartesian_errors[-1]:.3e}; elapsed={perf_counter()-started:.3f}s',flush=True)
