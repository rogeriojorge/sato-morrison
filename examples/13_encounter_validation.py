"""Independent positive scattering-table validation and stratified cutoff audit."""
from pathlib import Path
from time import perf_counter
import json
import hashlib
import csv
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import qmc
from sato_morrison.controls import (encounter_relative,encounter_thermal_moments,
    encounter_flux_quadrature,encounter_flux_samples,encounter_bounded_flux,
    encounter_moment_interpolator,encounter_reverse_incoming)
from sato_morrison.reference import progress,run_metadata

# All predictions and numerical targets are declared before the fresh holdout.
mass,charge,field=1.,1.,2.
strength,screening,theta=.03,3.,.5  # theta=k_B*T/m, single-particle velocity variance.
training_bounds=((1.2,2.4),(.7,1.5),(.2,.8))
training_orders=(4,4,4,16)
holdout_count,holdout_seed=256,1301
second_moment_rms_target,second_moment_p95_target=.02,.05
impact_annuli=[(.15,.4),(.4,.8),(.8,1.2),(1.2,2.4),(2.4,4.8),(4.8,8.)]
parallel_bounds=(.35,3.)
perpendicular_bounds=(0.,2.5)
sobol_levels=[32,64,128,256]
sobol_seed,replica_seed=1307,1308
phase_order=16
relative_diffusion_target=.05
start_distance,max_step,rtol=24.,.2,1e-10
reverse_checks=16
boundary_distances=[18.,24.,32.]
output=Path(__file__).resolve().parents[1]/'results'/'encounter_validation'
show_figures=False

inputs={'mass':mass,'charge':charge,'field':field,'strength':strength,'screening':screening,'theta':theta,
    'theta_definition':'single-particle velocity variance k_B*T/m','training_bounds':training_bounds,
    'training_orders':training_orders,'holdout_count':holdout_count,'holdout_seed':holdout_seed,
    'second_moment_rms_target':second_moment_rms_target,'second_moment_p95_target':second_moment_p95_target,
    'impact_annuli':impact_annuli,'parallel_bounds':parallel_bounds,'perpendicular_bounds':perpendicular_bounds,
    'sobol_levels':sobol_levels,'sobol_seed':sobol_seed,'replica_seed':replica_seed,'phase_order':phase_order,
    'relative_diffusion_target':relative_diffusion_target,'start_distance':start_distance,'max_step':max_step,
    'rtol':rtol,'reverse_checks':reverse_checks,'boundary_distances':boundary_distances,
    'table_method':'periodic cubic c and logarithmic positive COM variance; reconstruct second and cross moments',
    'coverage_method':'stratified incoming-flux inverse-CDF scrambled Sobol, explicit phase quadrature, independent replica',
    'rate_scope':'conditional crossing moments per partner density; screening specified, no SM D inference or plasma lifetime'}
output.mkdir(parents=True,exist_ok=True)
print(f'Model=encounter; uniform B={field}; screened repulsive equal particles; output={output}',flush=True)
print(f'Fresh holdout seed={holdout_seed}, states={holdout_count}; targets RMS={second_moment_rms_target:g}, p95={second_moment_p95_target:g}',flush=True)
print(f'Coverage: impacts {impact_annuli}, parallel {parallel_bounds}, perpendicular {perpendicular_bounds}, Sobol up to {sobol_levels[-1]}, phases={phase_order}',flush=True)
print('Explicit density normalization comes from bounded incoming flux. Complete plasma statistics, correlations and SM mixing/lifetime remain outside this calculation.',flush=True)
print('No compilation; direct DOP853 solves have first-outgoing-plane events and measured step sizes.',flush=True)
metadata=run_metadata(inputs,model='encounter',boundary='bounded incoming plane flux with transmitted/reflected first exits')
repository=Path(__file__).resolve().parents[1]
dependencies=[repository/'src/sato_morrison/controls.py',repository/'src/sato_morrison/reference.py',Path(__file__).resolve()]
metadata['experiment_dependency_sha256']={str(path.relative_to(repository)):hashlib.sha256(path.read_bytes()).hexdigest() for path in dependencies}
(output/'declared_inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
started=perf_counter();cache={};failures=[]


def evaluate(node,distance=start_distance,q=charge,step=max_step):
    b,parallel,perpendicular,phase=node
    key=tuple(np.round([b,parallel,perpendicular,phase%(2*np.pi),distance,q,step],14))
    if key not in cache:
        try:
            encounter=encounter_relative(b,phase,parallel,perpendicular,field=field,strength=strength,screening=screening,start_distance=distance,mass=mass,charge=q,max_step=step,rtol=rtol)
            cache[key]=(encounter_thermal_moments(encounter,theta,mass,field),encounter)
        except RuntimeError as exception:
            cache[key]=(np.full(3,np.nan),{'failure':str(exception)})
            failures.append({'node':list(map(float,node)),'distance':distance,'charge':q,'reason':str(exception)})
    return cache[key]

with progress('Building the physically constrained scattering table and fresh independent holdout'):
    training_nodes,training_weights=encounter_flux_quadrature(training_orders,*training_bounds,theta)
    training=[]
    for node in training_nodes:
        moment,detail=evaluate(node)
        if 'failure' in detail:
            raise RuntimeError('Training table encounter failed; no interpolation assembled.')
        training.append(moment)
    training=np.asarray(training)
    axes=[np.unique(training_nodes[:,i]) for i in range(4)]
    interpolator=encounter_moment_interpolator(axes,training.reshape(*training_orders,3))
    rng=np.random.default_rng(holdout_seed)
    heldout=np.column_stack([rng.uniform(axis[0],axis[-1],holdout_count) for axis in axes[:3]]+[rng.uniform(0.,2*np.pi,holdout_count)])
    predicted=interpolator(heldout)
    direct=np.array([evaluate(node)[0] for node in heldout])
    rms=np.sqrt(np.mean((predicted-direct)**2,axis=0))/np.sqrt(np.mean(direct**2,axis=0))
    relative=np.abs(predicted[:,1]-direct[:,1])/direct[:,1]
    p95=float(np.quantile(relative,.95))
    table_status='passed' if rms[1]<=second_moment_rms_target and p95<=second_moment_p95_target else 'unresolved'
    print(f'  Fresh heldout RMS(mean,second,cross)={rms}; second-moment p95={p95:.4%}; status={table_status}',flush=True)

with progress('Testing inverse charged-sign encounters and incoming cutoff closure'):
    reverse_errors=[];inverse_cutoff_exits=[]
    for node in heldout[:reverse_checks]:
        moment,forward=evaluate(node)
        inverse,angle=encounter_reverse_incoming(forward,field=field,mass=mass,charge=charge,start_distance=start_distance)
        reversed_moment,reverse=evaluate(inverse,q=-charge)
        error=np.abs(reversed_moment-moment*np.array([-1,1,1]))
        reverse_errors.append(error)
        inverse_cutoff_exits.append(any(not bounds[0]<=value<=bounds[1] for value,bounds in zip(inverse[:3],training_bounds)))
    reverse_errors=np.asarray(reverse_errors)
    print(f'  Max inverse absolute errors(mean,second,cross)={reverse_errors.max(axis=0)}',flush=True)

coverage_data=[];coverage_rows=[]
with progress('Integrating stratified broad incoming flux and independent coverage refinements'):
    units=qmc.Sobol(3,scramble=True,seed=sobol_seed).random_base2(int(np.log2(sobol_levels[-1])))
    replica_units=qmc.Sobol(3,scramble=True,seed=replica_seed).random_base2(int(np.log2(sobol_levels[-2])))
    phases=2*np.pi*np.arange(phase_order)/phase_order
    for annulus in impact_annuli:
        flux=encounter_bounded_flux(annulus,parallel_bounds,perpendicular_bounds,theta)
        samples=encounter_flux_samples(units,annulus,parallel_bounds,perpendicular_bounds,theta)
        replica_samples=encounter_flux_samples(replica_units,annulus,parallel_bounds,perpendicular_bounds,theta)
        moments=[];details=[]
        for sample in samples:
            state=[];state_details=[]
            for phase in phases:
                moment,detail=evaluate((*sample,phase))
                state.append(moment);state_details.append(detail)
            moments.append(state);details.append(state_details)
        moments=np.asarray(moments)
        replica=[]
        for sample in replica_samples:
            replica.append([evaluate((*sample,phase))[0] for phase in phases])
        replica=np.asarray(replica)
        finite=np.all(np.isfinite(moments),axis=-1)
        missing_fraction=float(1-finite.mean())
        # Nonexit states never disappear into a silently renormalized encounter rate.
        complete=bool(finite.all() and np.all(np.isfinite(replica)))
        rates=[]
        for count in sobol_levels:
            rates.append(flux*np.nansum(moments[:count],axis=(0,1))/(count*phase_order))
        replica_rate=flux*np.nansum(replica,axis=(0,1))/(len(replica)*phase_order)
        fine=rates[-1];previous=rates[-2]
        phase8_rate=flux*np.nansum(moments[:,::2],axis=(0,1))/(len(samples)*(phase_order//2))
        error=max(abs(fine[1]-previous[1]),abs(fine[1]-replica_rate[1]),abs(fine[1]-phase8_rate[1]))/abs(fine[1])
        valid_details=[d for state in details for d in state if 'failure' not in d]
        energy=max(d['energy_error'] for d in valid_details) if valid_details else np.nan
        status='passed' if complete and error<=relative_diffusion_target and energy<=1e-8 else 'unresolved'
        row={'impact_lower':annulus[0],'impact_upper':annulus[1],'flux_per_density':flux,
            'drift_per_density':float(fine[0]) if complete else None,
            'diffusion_per_density':float(fine[1]/2) if complete else None,
            'partial_diffusion_lower_bound_per_density':float(flux*np.nansum(moments[...,1])/(len(samples)*phase_order)/2),
            'last_diffusion_relative_uncertainty':float(error),'drift_replica_difference':float(abs(fine[0]-replica_rate[0])),
            'phase_diffusion_relative_change':float(abs(fine[1]-phase8_rate[1])/abs(fine[1])),
            'missing_fraction':missing_fraction,'reflected_fraction':float(sum(d['exit']=='reflected' for d in valid_details)/(len(samples)*phase_order)),
            'max_parallel_turns':max(d['parallel_turns'] for d in valid_details) if valid_details else None,
            'max_energy_error':float(energy),'max_actual_step':max(d['max_actual_step'] for d in valid_details) if valid_details else None,
            'status':status}
        coverage_rows.append(row)
        coverage_data.append({'annulus':annulus,'samples':samples,'moments':moments,'rates':rates,'replica_rate':replica_rate})
        print(f"  impacts={annulus}: A/n={row['drift_per_density']}, D/n={row['diffusion_per_density']}, last spread={error:.3%}, reflected={row['reflected_fraction']:.3%}, status={status}",flush=True)
        # Preserve partial evidence if a later expensive band fails.
        (output/'coverage_partial.json').write_text(json.dumps(coverage_rows,indent=2)+'\n')

with progress('Checking broad-state endpoint and active timestep sensitivity'):
    audit_nodes=[]
    for data in coverage_data:
        for sample in data['samples'][:2]:
            audit_nodes.append((*sample,.37))
    audits={}
    for distance in boundary_distances:
        audits[str(distance)]=np.array([evaluate(node,distance)[0] for node in audit_nodes])
    step_audits={}
    for step in [.2,.1,.05]:
        step_audits[str(step)]=np.array([evaluate(node,step=step)[0] for node in audit_nodes])
    boundary_change=float(np.linalg.norm(audits[str(boundary_distances[-1])][:,1]-audits[str(boundary_distances[-2])][:,1])/np.linalg.norm(audits[str(boundary_distances[-1])][:,1]))
    timestep_change=float(np.linalg.norm(step_audits['0.05'][:,1]-step_audits['0.1'][:,1])/np.linalg.norm(step_audits['0.05'][:,1]))

complete=all(row['drift_per_density'] is not None for row in coverage_rows)
total_drift=sum(row['drift_per_density'] for row in coverage_rows) if complete else None
total_diffusion=sum(row['diffusion_per_density'] for row in coverage_rows) if complete else None
uncertainty=sum(row['partial_diffusion_lower_bound_per_density']*row['last_diffusion_relative_uncertainty'] for row in coverage_rows)
normalizer=encounter_bounded_flux((impact_annuli[0][0],impact_annuli[-1][1]),parallel_bounds,perpendicular_bounds,theta)
area=np.pi*(impact_annuli[-1][1]**2-impact_annuli[0][0]**2)
full_speed_flux=2*area*np.sqrt(theta/np.pi)
velocity_fraction=normalizer/full_speed_flux
old_rate=(training_weights@training)[1]/2
status='passed' if complete and total_diffusion is not None and uncertainty/total_diffusion<=relative_diffusion_target and boundary_change<=.005 and timestep_change<=.001 else 'unresolved'
fig,axes_plot=plt.subplots(2,2,figsize=(11,8),layout='constrained')
axes_plot[0,0].scatter(direct[:,1],predicted[:,1],s=9)
lo,hi=direct[:,1].min(),direct[:,1].max();axes_plot[0,0].plot([lo,hi],[lo,hi],':',color='gray')
axes_plot[0,0].set(xlabel='new direct encounters',ylabel='positive cubic table',title=f'Independent second moment: RMS {rms[1]:.2%}, {table_status}')
centers=[np.sqrt(a*b) for a,b in impact_annuli]
diffusions=[r['partial_diffusion_lower_bound_per_density'] for r in coverage_rows]
axes_plot[0,1].bar(np.arange(len(centers)),diffusions)
axes_plot[0,1].set_xticks(np.arange(len(centers)),labels=[f'{a:g}–{b:g}' for a,b in impact_annuli],rotation=30)
axes_plot[0,1].set(xlabel='guiding-center impact annulus',ylabel='conditional diffusion integral / density',title='Broad velocities reveal cutoff sensitivity')
for data in coverage_data:
    axes_plot[1,0].plot(sobol_levels,[rate[1]/2 for rate in data['rates']],'-o',label=f"{data['annulus'][0]:g}–{data['annulus'][1]:g}")
axes_plot[1,0].set(xlabel='Sobol points ×16 phases',ylabel='diffusion integral / density',title='Explicit phase average; nonexits yield partial lower bounds');axes_plot[1,0].legend(fontsize=8,ncol=2)
axes_plot[1,1].bar(['old narrow bands','broad velocities + impacts'],[old_rate,total_diffusion or sum(diffusions)])
axes_plot[1,1].set(ylabel='conditional diffusion integral / density',title=f'Broad speed flux retained: {velocity_fraction:.1%}')
fig.savefig(output/'validation.png',dpi=160)
if show_figures:
    plt.show(block=False)
plt.close(fig)
metadata['experiment_dependency_sha256_end']={str(path.relative_to(repository)):hashlib.sha256(path.read_bytes()).hexdigest() for path in dependencies}
metadata['experiment_dependencies_unchanged']=metadata['experiment_dependency_sha256']==metadata['experiment_dependency_sha256_end']
metadata['results']={'table_status':table_status,'fresh_holdout_rms':rms.tolist(),'fresh_holdout_second_p95':p95,
    'reverse_max_absolute_errors':reverse_errors.max(axis=0).tolist(),'reverse_heldout_outside_training_band':int(sum(inverse_cutoff_exits)),
    'coverage_status':status,'coverage':coverage_rows,'total_conditional_drift_per_density':total_drift,
    'total_conditional_diffusion_per_density':total_diffusion,'empirical_diffusion_spread':uncertainty,
    'velocity_flux_retained_fraction':velocity_fraction,'old_narrow_diffusion_per_density':float(old_rate),
    'broad_state_boundary_relative_change':boundary_change,'broad_state_timestep_relative_change':timestep_change,
    'failures':failures,'unique_encounters':len(cache),'wall_s':perf_counter()-started,
    'limits':['conditional initial domain omits reverse states; its drift is not full Maxwellian drift',
        'b<.15,b>8,v_parallel<.35 or>3,v_perp>2.5 are omitted and are not proved negligible',
        'two independent Sobol scrambles measure a finite numerical spread, not a statistical confidence interval',
        'no independent successive-encounter correlations or kinetic closure validation',
        'no physical SM coefficient, full plasma diffusion rate, constrained mixing time, or dipole lifetime']}
(output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
with (output/'summary.csv').open('w',newline='') as file:
    writer=csv.DictWriter(file,fieldnames=list(coverage_rows[0]));writer.writeheader();writer.writerows(coverage_rows)
np.savez_compressed(output/'holdout.npz',training_nodes=training_nodes,training_moments=training,heldout=heldout,direct=direct,predicted=predicted)
print(f'Finished: new table={table_status}, broad bounded coverage={status}, unique direct encounters={len(cache)}, elapsed={perf_counter()-started:.1f}s',flush=True)
