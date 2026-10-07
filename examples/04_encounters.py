"""Bounded direct encounter evidence; no plasma rate or lifetime calibration."""
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
from sato_morrison.controls import binary_encounter, encounter_incoming
from sato_morrison.reference import progress, run_metadata

# Normalized Lorentz-force equations, mass=charge=1, potential kappa exp(-r/lambda)/r.
mass, charge, interaction_strength = 1., 1., .05
screening_length = 4.
parallel_speed, perpendicular_speed, gc_impact = 1., .35, 1.2
fields = [.5, 1., 2.]
phase_orders = [8, 16, 32, 64]
start_distance = 24.
start_distances = [12., 16., 24., 32.]
max_step, relative_tolerance = .2, 1e-10
rutherford_distances = [12., 24., 48., 96.]
energy_timesteps = [1.6, .8, .4, .2]
output = Path(__file__).resolve().parents[1] / 'results' / 'encounters'
show_figures = False

output.mkdir(parents=True,exist_ok=True)
print(f'Model=encounter; uniform Bz; equal masses/charges; screened repulsive pair; normalized units; output={output}',flush=True)
print(f'Inputs: kappa={interaction_strength}, screening={screening_length}, GC impact={gc_impact}, relative speeds=({perpendicular_speed},{parallel_speed}), fields={fields}',flush=True)
print('Reference checks: zero force, Rutherford, energy/timestep, fixed incoming helix, start/end distance, phase mean/second moment.',flush=True)
print('DOP853 integration has no compilation. This finite incoming ensemble has no assigned flux measure, density, collision frequency, or lifetime.',flush=True)
started=perf_counter()
rows=[]
with progress('Checking free helices, zero-field Rutherford and timestep convergence'):
    incoming,velocity=encounter_incoming(gc_impact,.7,parallel_speed,perpendicular_speed,start_distance,fields[-1])
    free=binary_encounter(incoming,velocity,field=fields[-1],strength=0.,duration=2*start_distance+2,max_step=max_step,rtol=1e-12)
    if np.max(np.abs(free['delta_mu']))>1e-12 or np.max(np.abs(free['delta_gc_perpendicular']))>1e-12:
        raise RuntimeError('Zero-interaction control failed.')
    analytic_angle=2*np.arctan(2*interaction_strength/(mass*parallel_speed**2*gc_impact))
    rutherford_angles=[]
    rutherford_energies=[]
    for distance in rutherford_distances:
        result=binary_encounter([gc_impact,0.,-distance],[0.,0.,parallel_speed],field=0.,strength=interaction_strength,mass=mass,charge=charge,duration=2*distance/parallel_speed+4,max_step=.3,rtol=relative_tolerance)
        relative=result['velocities'][-1,0]-result['velocities'][-1,1]
        angle=np.arctan2(np.linalg.norm(relative[:2]),relative[2])
        rutherford_angles.append(float(angle));rutherford_energies.append(result['energy_error'])
        rows.append({'case':f'rutherford_L{distance}','field':0.,'resolution':distance,'mean_delta_mu':'','second_moment':'','error':abs(angle-analytic_angle),'status':'passed' if abs(angle-analytic_angle)<.001 else 'unresolved'})
    timestep_errors=[]
    timestep_moments=[]
    for step in energy_timesteps:
        result=binary_encounter([.5,0.,-12.],[.35,0.,1.],field=.5,strength=.1,screening=screening_length,duration=26.,max_step=step,rtol=1e-3)
        timestep_errors.append(result['energy_error']);timestep_moments.append(float(result['delta_mu'][0]))
        rows.append({'case':f'energy_dt{step}','field':.5,'resolution':step,'mean_delta_mu':result['delta_mu'][0],'second_moment':'','error':result['energy_error'],'status':'passed' if result['energy_error']<1e-8 else 'unresolved'})
    if not np.all(np.diff(timestep_errors)<0) or timestep_errors[-1]>1e-8:
        raise RuntimeError('Encounter energy convergence failed.')

ensemble_results=[]
with progress('Integrating gyrophases at fixed incoming guiding-center impact'):
    for field in fields:
        for count in phase_orders:
            increments=[];gc_increments=[];energy_errors=[];minimum_separations=[];velocity_changes=[]
            for phase in 2*np.pi*np.arange(count)/count:
                incoming,velocity=encounter_incoming(gc_impact,phase,parallel_speed,perpendicular_speed,start_distance,charge*field/mass)
                result=binary_encounter(incoming,velocity,field=field,strength=interaction_strength,mass=mass,charge=charge,screening=screening_length,duration=2*start_distance/parallel_speed+2,max_step=max_step,rtol=relative_tolerance)
                increments.append(result['delta_mu']);gc_increments.append(result['delta_gc_perpendicular'][0])
                energy_errors.append(result['energy_error']);minimum_separations.append(result['min_separation'])
                velocity_changes.append(2*np.linalg.norm(result['delta_velocity_from_free'][0])/np.linalg.norm(velocity))
            increments=np.array(increments);gc_increments=np.array(gc_increments)
            joint=np.column_stack((increments,gc_increments))
            mu_reference=mass*(perpendicular_speed/2)**2/(2*field)
            statistics={'field':field,'phase_order':count,'mean_delta_mu':float(increments[:,0].mean()),
                'second_moment_delta_mu':float(np.mean(increments[:,0]**2)),
                'variance_delta_mu':float(np.var(increments[:,0])),
                'mu_reference':mu_reference,'rms_fraction':float(np.sqrt(np.mean(increments[:,0]**2))/mu_reference),
                'joint_covariance_columns':['delta_mu_1','delta_mu_2','delta_gc_x_1','delta_gc_y_1'],
                'joint_covariance':np.cov(joint,rowvar=False,bias=True).tolist(),
                'max_energy_error':max(energy_errors),'min_separation':min(minimum_separations),
                'max_relative_velocity_change_from_free':max(velocity_changes),
                'rho_over_screening':mass*(perpendicular_speed/2)/(charge*field*screening_length),
                'rho_over_b90':mass*(perpendicular_speed/2)/(charge*field)/(2*interaction_strength/(mass*(parallel_speed**2+perpendicular_speed**2))),
                'omega_screening_transit':charge*field/mass*screening_length/np.sqrt(parallel_speed**2+perpendicular_speed**2)}
            ensemble_results.append(statistics)
            rows.append({'case':f'phase_B{field}_N{count}','field':field,'resolution':count,'mean_delta_mu':statistics['mean_delta_mu'],'second_moment':statistics['second_moment_delta_mu'],'error':statistics['max_energy_error'],'status':'passed'})
            print(f"  B={field}, phases={count}: mean delta_mu={statistics['mean_delta_mu']:.6g}, RMS/mu_ref={statistics['rms_fraction']:.6g}, max energy error={statistics['max_energy_error']:.3e}",flush=True)
        previous,current=ensemble_results[-2:]
        for key in ['mean_delta_mu','second_moment_delta_mu']:
            if abs(current[key]-previous[key])/max(abs(current[key]),1e-20)>1e-3:
                raise RuntimeError('Gyrophase ensemble has not converged.')
        if current['max_energy_error']>1e-8:
            raise RuntimeError('Phase ensemble energy budget failed.')

boundary_statistics=[]
with progress('Receding start/end planes with the same incoming free helix'):
    field=fields[-1]
    for distance in start_distances:
        increments=[]
        for phase in 2*np.pi*np.arange(32)/32:
            incoming,velocity=encounter_incoming(gc_impact,phase,parallel_speed,perpendicular_speed,distance,charge*field/mass)
            result=binary_encounter(incoming,velocity,field=field,strength=interaction_strength,mass=mass,charge=charge,screening=screening_length,duration=2*distance/parallel_speed+2,max_step=max_step,rtol=relative_tolerance)
            increments.append(float(result['delta_mu'][0]))
        boundary_statistics.append({'distance':distance,'mean_delta_mu':float(np.mean(increments)),'second_moment':float(np.mean(np.array(increments)**2))})
        rows.append({'case':f'boundary_L{distance}','field':field,'resolution':distance,'mean_delta_mu':boundary_statistics[-1]['mean_delta_mu'],'second_moment':boundary_statistics[-1]['second_moment'],'error':'','status':'passed'})
    boundary_relative_change=max(abs(boundary_statistics[-1][key]-boundary_statistics[-2][key])/abs(boundary_statistics[-1][key]) for key in ['mean_delta_mu','second_moment'])
    if boundary_relative_change>1e-3:
        raise RuntimeError('Encounter start/end distance has not converged.')

fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
axes[0,0].plot(rutherford_distances,rutherford_angles,'o-',label='direct finite endpoints')
axes[0,0].axhline(analytic_angle,color='gray',linestyle=':',label='asymptotic Rutherford')
axes[0,0].set(xlabel='start distance',ylabel='deflection angle (rad)',title='Zero-field Coulomb limit');axes[0,0].legend()
axes[0,1].loglog(energy_timesteps,timestep_errors,'o-');axes[0,1].set_xticks(energy_timesteps,labels=[str(x) for x in energy_timesteps]);axes[0,1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter());axes[0,1].set(xlabel='maximum timestep',ylabel='max relative total energy error',title='DOP853 step refinement')
for field in fields:
    group=[s for s in ensemble_results if s['field']==field]
    axes[1,0].plot(phase_orders,[s['rms_fraction'] for s in group],'o-',label=f'B={field}')
axes[1,0].set(xlabel='gyrophases',ylabel='RMS delta_mu / initial mu',title='Energy conserved, magnetic moment changes');axes[1,0].legend()
axes[1,1].plot(start_distances,[s['second_moment'] for s in boundary_statistics],'o-')
axes[1,1].set(xlabel='start distance',ylabel='mean(delta_mu squared)',title='Fixed incoming helix, B=2')
fig.savefig(output/'encounters.png',dpi=160)
if show_figures:
    plt.show(block=False)
plt.close(fig)
inputs={'mass':mass,'charge':charge,'interaction_strength':interaction_strength,'screening_length':screening_length,
    'parallel_speed':parallel_speed,'perpendicular_speed':perpendicular_speed,'gc_impact':gc_impact,'fields':fields,
    'phase_orders':phase_orders,'start_distance':start_distance,'start_distances':start_distances,
    'max_step':max_step,'rtol':relative_tolerance,'duration':'2*start_distance/parallel_speed+2',
    'rutherford_distances':rutherford_distances,'rutherford_max_step':.3,'rutherford_screening':'infinite',
    'energy_timesteps':energy_timesteps,'free_control_rtol':1e-12,'energy_check':{'relative_position':[.5,0.,-12.],'relative_velocity':[.35,0.,1.],'field':.5,'strength':.1,'duration':26.,'rtol':1e-3},
    'ensemble_scope':'deterministic uniform gyrophase at fixed relative guiding-center impact and velocities; not plasma incident-flux distribution',
    'encounter_rate':'not specified','rho_over_field_gradient_scale':0.,'nu_over_omega':'not inferred','random_seed':'not applicable'}
metadata=run_metadata(inputs,model='encounter',boundary='finite incoming/outgoing free-helical planes; screened force tail checked')
metadata['results']={'ensemble':ensemble_results,'boundary':boundary_statistics,'boundary_relative_change':boundary_relative_change,
    'rutherford_analytic_angle':analytic_angle,'rutherford_angles':rutherford_angles,'rutherford_energy_errors':rutherford_energies,
    'energy_timestep_errors':timestep_errors,'energy_timestep_delta_mu':timestep_moments,'wall_s':perf_counter()-started,
    'conclusion':'conserving pair encounters produce magnetic-moment mean and variance; no measured plasma lifetime or SM coefficient'}
(output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
with (output/'summary.csv').open('w',newline='') as file:
    writer=csv.DictWriter(file,fieldnames=['case','field','resolution','mean_delta_mu','second_moment','error','status']);writer.writeheader();writer.writerows(rows)
print(f'Encounter checks passed; start/end last relative change={boundary_relative_change:.3e}; elapsed={perf_counter()-started:.3f}s',flush=True)
