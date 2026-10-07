"""Bounded Maxwellian incoming-flux quadrature and held-out encounter checks."""
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
from scipy.interpolate import RegularGridInterpolator
from sato_morrison.controls import (encounter_flux_quadrature, encounter_bounded_flux,
    encounter_relative, encounter_thermal_moments)
from sato_morrison.reference import progress, run_metadata

# Normalized equal-particle, prescribed screened pair model; all cutoffs explicit.
mass, charge, field = 1., 1., 2.
# temperature denotes theta=k_B*T/m, the single-particle velocity variance.
interaction_strength, screening_length, temperature = .03, 3., .5
impact_bounds = (1.2, 2.4)
parallel_bounds = (.7, 1.5)
perpendicular_bounds = (.2, .8)
accepted_orders = (4, 4, 4, 16)
refinement_orders = [3, 4, 5]
phase_orders = [8, 16, 32]
start_distance = 24.
start_distances = [18., 24., 32.]
max_step, relative_tolerance = .35, 1e-10
held_out_count, held_out_seed = 64, 701
interpolation_rms_target = .10
output = Path(__file__).resolve().parents[1] / 'results' / 'encounter_ensemble'
show_figures = False

output.mkdir(parents=True,exist_ok=True)
print(f'Model=encounter; uniform Bz={field}; equal thermal particles; bounded incoming-plane flux; output={output}',flush=True)
print(f'Inputs: m={mass}, q={charge}, kappa={interaction_strength}, screening={screening_length}, theta={temperature}; GC impact={impact_bounds}, relative v_parallel={parallel_bounds}, v_perp={perpendicular_bounds}',flush=True)
print('Measure: two incoming signs, 2*pi*b db, |v_parallel| F_relative d^3v; COM Maxwellian moments integrated analytically.',flush=True)
print('Conditional flux and moment integrals are per unit partner density. Omitted impact/speed bands and correlated encounters prevent a full plasma-rate or lifetime interpretation.',flush=True)
print('DOP853 event integration requires no compilation; reflected exits are retained, nonexit/solver failure rejects a case.',flush=True)
started=perf_counter()
cache={};summaries=[];rows=[]

# Cache shared deterministic quadrature nodes; keep all progress outside kernels.
def evaluate(nodes, distance, step=max_step):
    moment=[];details=[]
    for impact,parallel,perpendicular,phase in nodes:
        key=tuple(np.round([impact,parallel,perpendicular,phase,distance,step],14))
        if key not in cache:
            encounter=encounter_relative(impact,phase,parallel,perpendicular,field=field,strength=interaction_strength,screening=screening_length,start_distance=distance,mass=mass,charge=charge,max_step=step,rtol=relative_tolerance)
            cache[key]=(encounter_thermal_moments(encounter,temperature,mass,field),encounter)
        value,detail=cache[key]
        moment.append(value);details.append(detail)
    return np.array(moment),details


def integrate(case,orders,distance=start_distance,phase_offset=0.):
    nodes,weights=encounter_flux_quadrature(orders,impact_bounds,parallel_bounds,perpendicular_bounds,temperature,phase_offset)
    moments,details=evaluate(nodes,distance)
    flux=weights.sum()
    mean=weights@moments/flux
    integrated=weights@moments
    relative_perp=nodes[:,2]
    initial_mu=mass/(2*field)*(temperature+relative_perp**2/4)
    result={'case':case,'orders':list(orders),'start_distance':distance,'phase_offset':phase_offset,
        'nodes':len(nodes),'crossing_flux_per_density':float(flux),'conditional_mean_delta_mu':float(mean[0]),
        'conditional_second_moment_delta_mu':float(mean[1]),'conditional_cross_moment_mu1_mu2':float(mean[2]),
        'conditional_initial_mu':float(weights@initial_mu/flux),
        'conditional_drift_per_density':float(integrated[0]),'conditional_diffusion_per_density':float(integrated[1]/2),
        'max_energy_error':max(d['energy_error'] for d in details),'max_end_force':max(d['end_force'] for d in details),
        'min_separation':min(d['min_separation'] for d in details),
        'reflected_nodes':sum(d['exit']=='reflected' for d in details),'parallel_turns':sum(d['parallel_turns'] for d in details)}
    result['conditional_rms_mu_fraction']=float(np.sqrt(mean[1])/result['conditional_initial_mu'])
    summaries.append(result)
    rows.append({k:result[k] for k in ['case','nodes','crossing_flux_per_density','conditional_mean_delta_mu','conditional_second_moment_delta_mu','conditional_drift_per_density','conditional_diffusion_per_density','max_energy_error']})
    print(f"  {case}: nodes={len(nodes)}, flux/n={flux:.8g}, mean(delta_mu)={mean[0]:.6g}, E(delta_mu^2)={mean[1]:.6g}, energy={result['max_energy_error']:.3e}",flush=True)
    return result,nodes,weights,moments

with progress('Integrating positive bounded incident flux and independent quadrature refinements'):
    accepted,nodes,weights,moments=integrate('accepted',accepted_orders)
    for axis,label in enumerate(['impact','parallel','perpendicular']):
        for order in refinement_orders:
            orders=list(accepted_orders);orders[axis]=order
            integrate(f'{label}_{order}',tuple(orders))
    for count in phase_orders:
        integrate(f'phase_{count}',accepted_orders[:3]+(count,))
    shifted,_,_,_=integrate('held_out_shifted_phase',accepted_orders,phase_offset=np.pi/accepted_orders[-1])
    for distance in start_distances:
        integrate(f'boundary_{distance}',accepted_orders,distance)

with progress('Checking independent held-out scattering states against the tabulated moment interpolant'):
    axes=[np.unique(nodes[:,i]) for i in range(4)]
    table=moments.reshape(*accepted_orders,3)
    phase_axis=np.concatenate((axes[3],[2*np.pi]))
    periodic_table=np.concatenate((table,table[:,:,:,:1,:]),axis=3)
    interpolator=RegularGridInterpolator((*axes[:3],phase_axis),periodic_table,bounds_error=True)
    rng=np.random.default_rng(held_out_seed)
    test_nodes=np.empty((held_out_count,4))
    for i in range(3):
        test_nodes[:,i]=rng.uniform(axes[i][0],axes[i][-1],held_out_count)
    test_nodes[:,3]=rng.uniform(0.,2*np.pi,held_out_count)
    predictions=interpolator(test_nodes)
    direct,_=evaluate(test_nodes,start_distance)
    interpolation_errors=np.sqrt(np.mean((predictions-direct)**2,axis=0))/np.sqrt(np.mean(direct**2,axis=0))
    print(f'  Held-out normalized RMS errors (mean,second moment,cross moment)={interpolation_errors}',flush=True)
    # The predeclared target concerns the positive second moment used for diffusion.
    interpolation_status='passed' if interpolation_errors[1]<interpolation_rms_target else 'unresolved'

analytic_flux=encounter_bounded_flux(impact_bounds,parallel_bounds,perpendicular_bounds,temperature)
flux_relative_error=abs(accepted['crossing_flux_per_density']-analytic_flux)/analytic_flux
checks={}
for label in ['impact','parallel','perpendicular']:
    group=[s for s in summaries if s['case'].startswith(label+'_')]
    checks[label]=max(abs(group[-1][key]-group[-2][key])/abs(group[-1][key]) for key in ['conditional_mean_delta_mu','conditional_second_moment_delta_mu'])
phase_group=[s for s in summaries if s['case'].startswith('phase_')]
checks['phase']=max(abs(phase_group[-1][key]-phase_group[-2][key])/abs(phase_group[-1][key]) for key in ['conditional_mean_delta_mu','conditional_second_moment_delta_mu'])
boundary_group=[s for s in summaries if s['case'].startswith('boundary_')]
checks['boundary']=max(abs(boundary_group[-1][key]-boundary_group[-2][key])/abs(boundary_group[-1][key]) for key in ['conditional_mean_delta_mu','conditional_second_moment_delta_mu'])
checks['held_out_shifted_phase']=max(abs(shifted[key]-accepted[key])/abs(accepted[key]) for key in ['conditional_mean_delta_mu','conditional_second_moment_delta_mu'])
# Save every unresolved result before rejecting acceptance at the end.
numerical_status='passed' if max(checks.values())<=.01 and flux_relative_error<=1e-7 and accepted['max_energy_error']<=1e-8 else 'unresolved'
timestep_results=[]
with progress('Independent timestep refinement on eight held-out scattering states'):
    for step in [.6,.35,.2]:
        checked,details=evaluate(test_nodes[:8],start_distance,step)
        timestep_results.append({'max_step':step,'moments':checked.tolist(),'max_energy_error':max(d['energy_error'] for d in details)})
    fine=np.asarray(timestep_results[-1]['moments'])
    medium=np.asarray(timestep_results[-2]['moments'])
    timestep_relative_change=np.linalg.norm(fine[:,1]-medium[:,1])/np.linalg.norm(fine[:,1])
    if timestep_relative_change>.001 or timestep_results[-1]['max_energy_error']>1e-8:
        numerical_status='unresolved'

fig,axes_plot=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for label in ['impact','parallel','perpendicular']:
    group=[s for s in summaries if s['case'].startswith(label+'_')]
    errors=[abs(s['conditional_second_moment_delta_mu']/accepted['conditional_second_moment_delta_mu']-1) for s in group]
    axes_plot[0,0].semilogy(refinement_orders,np.maximum(errors,1e-14),'o-',label=label)
axes_plot[0,0].set(xlabel='Gauss order in one coordinate',ylabel='relative second-moment change',title='Refine each physical integration separately');axes_plot[0,0].legend()
phase_errors=[abs(s['conditional_second_moment_delta_mu']/accepted['conditional_second_moment_delta_mu']-1) for s in phase_group]
axes_plot[0,1].semilogy(phase_orders,np.maximum(phase_errors,1e-14),'o-')
axes_plot[0,1].set(xlabel='gyrophases',ylabel='relative second-moment change',title='Explicit 3V phase geometry')
base_weights=weights.reshape(*accepted_orders)
base_moments=moments.reshape(*accepted_orders,3)
impact_nodes=np.unique(nodes[:,0])
weight_by_b=base_weights.sum(axis=(1,2,3))
full_second=(base_weights*base_moments[...,1]).sum(axis=(1,2,3))/weight_by_b
zero_center_second=(base_weights*base_moments[...,0]**2).sum(axis=(1,2,3))/weight_by_b
axes_plot[1,0].semilogy(impact_nodes,full_second,'o-',label='thermal COM included')
axes_plot[1,0].semilogy(impact_nodes,zero_center_second,'s-',label='zero COM only')
axes_plot[1,0].set(xlabel='guiding-center impact',ylabel='conditional E(delta_mu squared)',title='Individual moments need center fluctuations');axes_plot[1,0].legend()
axes_plot[1,1].scatter(direct[:,1],predictions[:,1],s=14)
minimum,maximum=min(direct[:,1].min(),predictions[:,1].min()),max(direct[:,1].max(),predictions[:,1].max())
axes_plot[1,1].plot([minimum,maximum],[minimum,maximum],color='gray',linestyle=':')
axes_plot[1,1].set(xlabel='independent direct held-out encounter',ylabel='tabulated linear interpolation',title=f'Held-out second moment: RMS {interpolation_errors[1]:.1%}')
fig.savefig(output/'ensemble.png',dpi=160)
if show_figures:
    plt.show(block=False)
plt.close(fig)
inputs={'mass':mass,'charge':charge,'field':field,'interaction_strength':interaction_strength,'screening_length':screening_length,
    'temperature':temperature,'temperature_definition':'theta=k_B*T/m, single-particle velocity variance','impact_bounds':impact_bounds,'parallel_bounds':parallel_bounds,'perpendicular_bounds':perpendicular_bounds,
    'accepted_orders':accepted_orders,'refinement_orders':refinement_orders,'phase_orders':phase_orders,
    'start_distance':start_distance,'start_distances':start_distances,'max_step':max_step,'rtol':relative_tolerance,
    'held_out_count':held_out_count,'held_out_seed':held_out_seed,'interpolation_rms_target':interpolation_rms_target,
    'relative_distribution':'Gaussian covariance 2*theta I; independent center covariance theta/2 I',
    'flux_measure':'2*(2*pi*b db)*v_parallel*F_relative*v_perp dv_perp dv_parallel dphi',
    'pair_potential':'kappa exp(-r/screening_length)/r','exit_policy':'first outgoing +/-start_distance plane; reflected retained; no exit by 6L/vpar rejected',
    'rate_scope':'bounded crossing flux/moment integrals per unit partner density; no plasma independent-encounter validation or SM D inference'}
metadata=run_metadata(inputs,model='encounter',boundary='bounded incoming flux; first outgoing plane; finite screening-tail convergence')
full_band_flux=2*np.pi*(impact_bounds[1]**2-impact_bounds[0]**2)*np.sqrt(temperature/np.pi)
metadata['results']={'accepted':accepted,'runs':summaries,'analytic_bounded_flux':analytic_flux,'flux_relative_error':flux_relative_error,
    'velocity_band_fraction_of_flux_in_same_impact_annulus':accepted['crossing_flux_per_density']/full_band_flux,
    'last_relative_changes':checks,'held_out_interpolation_rms_errors':interpolation_errors.tolist(),'held_out_interpolation_status':interpolation_status,
    'timestep_results':timestep_results,'timestep_second_moment_relative_change':float(timestep_relative_change),'numerical_status':numerical_status,'unique_integrated_encounters':len(cache),'wall_s':perf_counter()-started,
    'limits':['incoming impact/speed domain is conditional, not converged to complete plasma statistics',
        'screening is prescribed rather than derived from density/temperature',
        'successive encounters/correlations and kinetic closure have not been validated',
        'bounded cutoffs need not preserve the detailed-balance reverse ensemble; conditional drift is not full Maxwellian drift',
        'no independent SM mixing rate or physical constrained-state lifetime']}
(output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
with (output/'summary.csv').open('w',newline='') as file:
    writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
np.savez_compressed(output/'held_out.npz',nodes=test_nodes,direct=direct,interpolated=predictions)
if numerical_status!='passed':
    raise RuntimeError(f'Bounded encounter numerical convergence unresolved; evidence saved in {output}: {checks}, flux_error={flux_relative_error}, timestep_change={timestep_relative_change}')
print(f'Bounded flux integration passed: analytic flux error={flux_relative_error:.3e}, largest last relative change={max(checks.values()):.3e}; held-out interpolation={interpolation_status}; elapsed={perf_counter()-started:.3f}s',flush=True)
