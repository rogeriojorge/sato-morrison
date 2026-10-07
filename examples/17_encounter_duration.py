"""Audit one prior numerical nonexit without replacing its quadrature evidence."""
from pathlib import Path
from time import perf_counter
import hashlib
import json
import csv
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import qmc
from sato_morrison.controls import (encounter_relative,encounter_thermal_moments,
    encounter_flux_samples,binary_encounter)
from sato_morrison.reference import progress,run_metadata

# Preserve the failed campaign, then independently vary its finite flight budget.
repository=Path(__file__).resolve().parents[1]
prior_directory=repository/'results'/'encounter_validation'
prior_metadata_sha256='c8f2626a09aa81e33d3826f5a4a6b80a892fe973b0d5cd5e33347997da193810'
flight_time_factors=[6.,12.,24.,48.]
active_max_steps=[.2,.1,.05]
endpoint_distances=[24.,32.,40.]
energy_target=1e-8
timestep_moment_target=.001
endpoint_moment_target=.005
independent_pair_velocity_absolute_target=1e-8
independent_pair_position_absolute_target=1e-8
independent_pair_max_steps=[.05,.025]
output=repository/'results'/'encounter_duration'
show_figures=False
inputs={'prior_metadata_sha256':prior_metadata_sha256,'flight_time_factors':flight_time_factors,
    'active_max_steps':active_max_steps,'endpoint_distances':endpoint_distances,
    'energy_target':energy_target,'timestep_moment_target':timestep_moment_target,
    'endpoint_moment_target':endpoint_moment_target,
    'independent_pair_velocity_absolute_target':independent_pair_velocity_absolute_target,
    'independent_pair_position_absolute_target':independent_pair_position_absolute_target,
    'independent_pair_max_steps':independent_pair_max_steps,
    'scope':'finite flight-budget diagnosis of exact previously failed nodes; original quadrature and interpolation failures retained'}
output.mkdir(parents=True,exist_ok=True)
print(f'Model=screened repulsive equal-particle encounter; output={output}',flush=True)
print(f'Prior evidence SHA={prior_metadata_sha256}; budgets={flight_time_factors}; maxsteps={active_max_steps}; endpoint distances={endpoint_distances}',flush=True)
print('The original broad conditional integral remains unresolved. Successful extended flight cannot repair its independent quadrature error.',flush=True)
print('No compilation; direct DOP853 events plus independently integrated 12D Cartesian pair path.',flush=True)
started=perf_counter()
prior_path=prior_directory/'metadata.json'
if hashlib.sha256(prior_path.read_bytes()).hexdigest()!=prior_metadata_sha256:
    raise ValueError('Prior evidence changed; do not silently replace the predeclared failed-node campaign')
prior=json.loads(prior_path.read_text());physical=prior['inputs'];previous=prior['results']
inputs['physical_inputs']=physical
inputs['exact_failed_nodes']=[failure['node'] for failure in previous['failures']]
metadata=run_metadata(inputs,model='encounter',boundary='same outgoing-plane events with independently extended flight budgets')
dependencies=[repository/'src/sato_morrison/controls.py',repository/'src/sato_morrison/reference.py',Path(__file__).resolve()]
metadata['experiment_dependency_sha256']={str(path.relative_to(repository)):hashlib.sha256(path.read_bytes()).hexdigest() for path in dependencies}
metadata['prior_evidence']={'producer_commit':prior['commit'],'dependency_sha256':prior['experiment_dependency_sha256'],
    'metadata_sha256':prior_metadata_sha256,'physical_inputs':physical,'prior_failure_count':len(previous['failures'])}
(output/'declared_inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')

# Reconstruct both quadrature populations to expose replica omissions separately.
units=qmc.Sobol(3,scramble=True,seed=physical['sobol_seed']).random_base2(int(np.log2(physical['sobol_levels'][-1])))
replica_units=qmc.Sobol(3,scramble=True,seed=physical['replica_seed']).random_base2(int(np.log2(physical['sobol_levels'][-2])))
phases=2*np.pi*np.arange(physical['phase_order'])/physical['phase_order']
missing=[]
for annulus in physical['impact_annuli']:
    def quadrature_keys(unit):
        samples=encounter_flux_samples(unit,annulus,physical['parallel_bounds'],physical['perpendicular_bounds'],physical['theta'])
        return {tuple(np.round((*sample,phase),12)) for sample in samples for phase in phases}
    base_keys=quadrature_keys(units);replica_keys=quadrature_keys(replica_units)
    failed_keys={tuple(np.round(failure['node'],12)) for failure in previous['failures'] if failure['distance']==physical['start_distance'] and failure['charge']==physical['charge']}
    base_count=len(failed_keys&base_keys);replica_count=len(failed_keys&replica_keys)
    missing.append({'annulus':annulus,'base_missing_count':base_count,'base_total':len(base_keys),
        'base_missing_fraction':base_count/len(base_keys),'replica_missing_count':replica_count,
        'replica_total':len(replica_keys),'replica_missing_fraction':replica_count/len(replica_keys)})


def solve(node,distance,budget,step):
    b,parallel,perpendicular,phase=node
    return encounter_relative(b,phase,parallel,perpendicular,field=physical['field'],strength=physical['strength'],
        screening=physical['screening'],start_distance=distance,mass=physical['mass'],charge=physical['charge'],
        max_step=step,rtol=physical['rtol'],flight_time_factor=budget)


def attempt(node,distance,budget,step):
    record={'node':node,'distance':distance,'flight_time_factor':budget,'max_step':step}
    try:
        result=solve(node,distance,budget,step)
        moment=encounter_thermal_moments(result,physical['theta'],physical['mass'],physical['field'])
        record.update(status='outgoing_event',time=result['time'],exit=result['exit'],energy_error=result['energy_error'],
            parallel_turns=result['parallel_turns'],final_parallel_velocity=float(result['final_relative_velocity'][2]),
            max_actual_step=result['max_actual_step'],step_count=result['step_count'],evaluations=result['evaluations'],
            moments=moment.tolist(),final_position=result['final_relative_position'].tolist(),final_velocity=result['final_relative_velocity'].tolist())
        return record,result
    except RuntimeError as error:
        record.update(status='unresolved',reason=str(error),integration_diagnostics=getattr(error,'integration_diagnostics',{}))
        return record,None

campaign=[];pair_paths={}
with progress('Auditing exact nonexit nodes: budget, active timestep, endpoint, independent Cartesian pair'):
    for index,failure in enumerate(previous['failures']):
        node=failure['node'];distance=failure['distance']
        budgets=[];successful=[]
        for budget in flight_time_factors:
            record,result=attempt(node,distance,budget,active_max_steps[0]);budgets.append(record)
            if result is not None:
                successful.append((budget,result))
            print(f'  Failednode {index}: budgetfactor={budget:g}, outcome={record["status"]}, time={record.get("time",record.get("integration_diagnostics",{}).get("final_time"))}',flush=True)
        report={'original_failure':failure,'budget_audit':budgets,'outcome':'unresolved at all predeclared budgets'}
        if successful:
            budget,first=successful[0]
            timesteps=[attempt(node,distance,budget,step)[0] for step in active_max_steps]
            endpoints=[attempt(node,L,flight_time_factors[-1],active_max_steps[-1])[0] for L in endpoint_distances]
            steps_complete=all(row['status']=='outgoing_event' for row in timesteps)
            endpoints_complete=all(row['status']=='outgoing_event' for row in endpoints)
            step_change=abs(timesteps[-1]['moments'][1]-timesteps[-2]['moments'][1])/abs(timesteps[-1]['moments'][1]) if steps_complete else None
            endpoint_change=abs(endpoints[-1]['moments'][1]-endpoints[-2]['moments'][1])/abs(endpoints[-1]['moments'][1]) if endpoints_complete else None
            # Independent 12D integration contains individual positions and velocities.
            pair=binary_encounter(first['initial_relative_position'],first['initial_relative_velocity'],
                field=physical['field'],strength=physical['strength'],screening=physical['screening'],duration=first['time'],
                mass=physical['mass'],charge=physical['charge'],max_step=active_max_steps[-1],rtol=physical['rtol'])
            pair_final=pair['velocities'][-1,0]-pair['velocities'][-1,1]
            pair_velocity_error=float(np.max(np.abs(pair_final-first['final_relative_velocity'])))
            pair_mu_error=float(abs(pair['delta_mu'][0]-first['conditional_mu_mean']))
            # Keep the original coarse comparison and compare both independently refined
            # Cartesian paths to the finest already declared relative solve.
            finest=solve(node,distance,budget,active_max_steps[-1])
            pair_refinement=[];pair_refinement_paths=[]
            for pair_step in independent_pair_max_steps:
                fine_pair=binary_encounter(finest['initial_relative_position'],finest['initial_relative_velocity'],
                    field=physical['field'],strength=physical['strength'],screening=physical['screening'],duration=finest['time'],
                    mass=physical['mass'],charge=physical['charge'],max_step=pair_step,rtol=physical['rtol'])
                pair_position=fine_pair['positions'][-1,0]-fine_pair['positions'][-1,1]
                pair_velocity=fine_pair['velocities'][-1,0]-fine_pair['velocities'][-1,1]
                pair_refinement.append({'max_step':pair_step,'event_time':finest['time'],
                    'velocity_max_absolute_error':float(np.max(np.abs(pair_velocity-finest['final_relative_velocity']))),
                    'position_max_absolute_error':float(np.max(np.abs(pair_position-finest['final_relative_position']))),
                    'mu_absolute_error':float(abs(fine_pair['delta_mu'][0]-finest['conditional_mu_mean'])),
                    'energy_error':fine_pair['energy_error']})
                pair_refinement_paths.append(fine_pair)
            pair_plateau=float(np.max(np.abs((pair_refinement_paths[-1]['velocities'][-1,0]-pair_refinement_paths[-1]['velocities'][-1,1])-(pair_refinement_paths[-2]['velocities'][-1,0]-pair_refinement_paths[-2]['velocities'][-1,1]))))
            pair_passed=all(item['velocity_max_absolute_error']<=independent_pair_velocity_absolute_target and item['position_max_absolute_error']<=independent_pair_position_absolute_target and item['energy_error']<=energy_target for item in pair_refinement) and pair_plateau<=independent_pair_velocity_absolute_target
            pair_paths[index]=pair_refinement_paths[-1]
            report.update(outcome='original flight budget insufficient; outgoing event reached with extended budget' if budgets[0]['status']=='unresolved' else 'original numerical nonexit not reproduced',
                minimum_successful_declared_factor=budget,timestep_audit=timesteps,endpoint_audit=endpoints,
                timestep_second_moment_relative_change=step_change,endpoint_second_moment_relative_change=endpoint_change,
                independent_pair_velocity_max_absolute_error=pair_velocity_error,independent_pair_mu_absolute_error=pair_mu_error,
                independent_pair_energy_error=pair['energy_error'],
                original_coarse_comparison_status='passed' if pair_velocity_error<=independent_pair_velocity_absolute_target else 'unresolved',
                independent_pair_refinement=pair_refinement,independent_pair_refinement_velocity_plateau=pair_plateau,
                finest_relative_max_step=active_max_steps[-1],
                numerical_status='passed' if steps_complete and endpoints_complete and step_change<=timestep_moment_target and endpoint_change<=endpoint_moment_target and pair_passed and finest['energy_error']<=energy_target else 'unresolved')
        if 'independent_pair_refinement' in report:
            print(f"  Independent pair: original coarse velocity error={pair_velocity_error:.3e}; finest velocity error={pair_refinement[-1]['velocity_max_absolute_error']:.3e}; position error={pair_refinement[-1]['position_max_absolute_error']:.3e}; status={report['numerical_status']}",flush=True)
        campaign.append(report)
        (output/'audit_partial.json').write_text(json.dumps(campaign,indent=2)+'\n')

# Replot saved campaign data with precise finite-quadrature wording, no rerun.
with np.load(prior_directory/'holdout.npz',allow_pickle=False) as saved:
    direct=saved['direct'];predicted=saved['predicted']
rows=previous['coverage'];band_values=[row['partial_diffusion_lower_bound_per_density'] for row in rows]
labels=[f'{a:g}–{b:g}' for a,b in physical['impact_annuli']]
fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
axes[0,0].scatter(direct[:,1],predicted[:,1],s=8)
lo,hi=direct[:,1].min(),direct[:,1].max();axes[0,0].plot([lo,hi],[lo,hi],':',color='gray')
axes[0,0].set(xlabel='direct second moment',ylabel='original cubic table',title=f'Original table: 95th-percentile error {previous["fresh_holdout_second_p95"]:.1%}\nTarget 5% failed')
axes[0,1].bar(np.arange(len(labels)),band_values)
axes[0,1].set_xticks(np.arange(len(labels)),labels=labels,rotation=25)
axes[0,1].set(xlabel='guiding-center impact annulus',ylabel='positive finite-quadrature contribution / density',title='Broad campaign; total integral remains unresolved')
axes[1,0].bar(np.arange(len(labels)),[100*row['last_diffusion_relative_uncertainty'] for row in rows],color=['tab:green' if row['status']=='passed' else 'tab:orange' for row in rows])
axes[1,0].axhline(100*physical['relative_diffusion_target'],color='gray',linestyle=':')
axes[1,0].set_xticks(np.arange(len(labels)),labels=labels,rotation=25)
axes[1,0].set(xlabel='guiding-center impact annulus',ylabel='empirical quadrature spread (%)',title='Two scrambles and phase refinement; no confidence bound')
same_annulus=next(row for row in rows if (row['impact_lower'],row['impact_upper'])==tuple(physical['training_bounds'][0]))
axes[1,1].bar(['original narrow speeds','same impact annulus\nbroader speeds'],[previous['old_narrow_diffusion_per_density'],same_annulus['diffusion_per_density']])
axes[1,1].set(ylabel='conditional second-moment integral / (2 density)',title='Cutoff dependence: broader speed band, same impacts')
fig.savefig(output/'coverage_summary.png',dpi=160);plt.close(fig)

if pair_paths:
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    paths={}
    for index,pair in pair_paths.items():
        z=pair['positions'][:,0,2]-pair['positions'][:,1,2]
        wz=pair['velocities'][:,0,2]-pair['velocities'][:,1,2]
        axes[0].plot(pair['time'],z,label=f'previously unresolved node {index}')
        axes[1].plot(pair['time'],wz,label=f'node {index}')
        original_budget=6*physical['start_distance']/previous['failures'][index]['node'][1]
        axes[0].axvline(original_budget,color='gray',linestyle=':',label='original flight limit')
        paths.update({f'time_{index}':pair['time'],f'positions_{index}':pair['positions'],f'velocities_{index}':pair['velocities']})
    axes[0].set(xlabel='time',ylabel='relative parallel position',title='Outgoing event after original timeout');axes[0].legend(fontsize=8)
    axes[1].set(xlabel='time',ylabel='relative parallel velocity',title='Repulsive z acceleration has sign(z)');axes[1].axhline(0.,color='gray',linestyle=':')
    fig.savefig(output/'duration.png',dpi=160);plt.close(fig)
    np.savez_compressed(output/'pair_paths.npz',**paths)
metadata['experiment_dependency_sha256_end']={str(path.relative_to(repository)):hashlib.sha256(path.read_bytes()).hexdigest() for path in dependencies}
metadata['experiment_dependencies_unchanged']=metadata['experiment_dependency_sha256']==metadata['experiment_dependency_sha256_end']
metadata['results']={'missing_populations':missing,'failure_audits':campaign,'wall_s':perf_counter()-started,
    'prior_coverage_status_preserved':previous['coverage_status'],'original_conditional_integral_not_repaired':True,
    'interpretation':['repulsive z acceleration has sign(z), allowing at most one parallel turn; finite nonexit proves neither trapping nor inter-encounter correlation',
        'positive partial contributions concern chosen finite quadrature sums; no rigorous lower bound on the continuous integral',
        'the additional flight audit addresses numerical event termination, not omitted-band or kinetic-closure validation']}
(output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(f'Originalfailures={len(previous["failures"])}; resolved outgoingevents={sum("minimum_successful_declared_factor" in row for row in campaign)}; elapsed={perf_counter()-started:.1f}s; prior bounded integral={previous["coverage_status"]}',flush=True)
