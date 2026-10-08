"""Resolve a fixed close encounter's positive gyrophase second-moment integral."""
from pathlib import Path
from time import perf_counter
import hashlib,json,csv
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sato_morrison.controls import encounter_relative,encounter_thermal_moments,binary_encounter
from sato_morrison.reference import run_metadata,progress

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'results'/'encounter_phase'
PRIOR=ROOT/'results'/'encounter_duration'/'metadata.json'
PRIOR_SHA='d2b411a3213d5becb1504ba7d654802240505159d52a7b7dcba936cce852f5ab'
MASS,CHARGE,FIELD,STRENGTH,SCREENING,THETA=1.,1.,2.,.03,3.,.5
IMPACT,PARALLEL,PERPENDICULAR=.9938225266093732,1.9056642462041216,1.9668376625283577
DISTANCE,FLIGHT_FACTOR,MAX_STEP,RTOL=24.,12.,.05,1e-10
ORDERS=(32,64,128,256,512)
PHASE_TARGET,TIMESTEP_TARGET,ENERGY_TARGET=.01,.001,1e-8
SUCCESSIVE_CHECKS=2
REFINED_STEP=.025
CONTROL_INDICES=(0,128,256)
PAIR_STEPS=(.05,.025)
PAIR_POSITION_TARGET,PAIR_VELOCITY_TARGET,PAIR_MU_TARGET=1e-8,1e-8,1e-10
ANCHOR_TARGET=1e-8


def phase_attempt(index,step):
    phase=2*np.pi*index/ORDERS[-1]
    row={'master_phase_index':int(index),'phase':float(phase),'max_step':float(step)}
    begin=perf_counter()
    try:
        result=encounter_relative(IMPACT,phase,PARALLEL,PERPENDICULAR,
            field=FIELD,strength=STRENGTH,screening=SCREENING,start_distance=DISTANCE,
            mass=MASS,charge=CHARGE,max_step=step,rtol=RTOL,flight_time_factor=FLIGHT_FACTOR)
        moments=encounter_thermal_moments(result,THETA,MASS,FIELD)
        if not np.all(np.isfinite(moments)) or moments[1]<0 or moments[1]<abs(moments[2])-1e-14:
            raise RuntimeError('Nonfinite or inconsistent center-of-mass moments')
        row.update(status='outgoing_event',moments=moments.tolist(),
            **{key:result[key] for key in ('time','exit','energy_error','parallel_turns',
                'step_count','evaluations','min_step','max_actual_step','min_separation',
                'flight_budget','end_force')},
            **{key:np.asarray(result[key]).tolist() for key in ('initial_relative_position',
                'initial_relative_velocity','final_relative_position','final_relative_velocity',
                'delta_relative_perpendicular')})
    except (RuntimeError,ValueError,FloatingPointError) as error:
        row.update(status='unresolved',reason=str(error),error_type=type(error).__name__,
            integration_diagnostics=getattr(error,'integration_diagnostics',{}))
    row['wall_s']=perf_counter()-begin
    return row


def phase_summary(records,order):
    complete=all(r['status']=='outgoing_event' for r in records)
    valid=[r for r in records if r['status']=='outgoing_event']
    known=np.sum([r['moments'] for r in valid],axis=0)/order if valid else np.zeros(3)
    return {'phase_order':order,'complete':complete,'missing_count':order-len(valid),
        'missing_phase_weight':(order-len(valid))/order,
        'phase_averaged_moments':known.tolist() if complete else None,
        'known_positive_second_moment_contribution':float(known[1]),
        'max_energy_error':max((r['energy_error'] for r in valid),default=None),
        'reflected_count':sum(r['exit']=='reflected' for r in valid),
        'scope':'Each phase has weight 1/N. Unknown phases are never renormalized; partial positive sum is not a rigorous continuous-integral bound.'}


def second_change(current,previous):
    if not current['complete'] or not previous['complete']:
        return None
    a,b=current['phase_averaged_moments'][1],previous['phase_averaged_moments'][1]
    return float(abs(a-b)/a) if a>0 else None


inputs={'mass':MASS,'charge':CHARGE,'field':FIELD,'strength':STRENGTH,'screening':SCREENING,
    'theta':THETA,'theta_definition':'single-particle velocity variance k_B*T/m',
    'fixed_incoming_triple':[IMPACT,PARALLEL,PERPENDICULAR],
    'start_distance':DISTANCE,'flight_time_factor':FLIGHT_FACTOR,'max_step':MAX_STEP,'rtol':RTOL,
    'phase_orders':ORDERS,'master_phase_count':ORDERS[-1],'phase_measure':'dphi/(2pi)',
    'phase_relative_target':PHASE_TARGET,'required_successive_checks':SUCCESSIVE_CHECKS,
    'refined_max_step':REFINED_STEP,'timestep_relative_target':TIMESTEP_TARGET,
    'energy_target':ENERGY_TARGET,'control_master_indices':CONTROL_INDICES,
    'independent_pair_max_steps':PAIR_STEPS,'pair_position_absolute_target':PAIR_POSITION_TARGET,
    'pair_velocity_absolute_target':PAIR_VELOCITY_TARGET,'pair_mu_absolute_target':PAIR_MU_TARGET,
    'anchor_absolute_moment_target':ANCHOR_TARGET,
    'prior_metadata_sha256':PRIOR_SHA,'seed':None,
    'moment_definition':'analytic independent Maxwellian COM average of individual delta_mu_1, its square, and delta_mu_1*delta_mu_2',
    'stop_rule':'Stop after two successive complete 1% second-moment phase checks; otherwise retain all declared levels through512.'}
print('Fixed-triple gyrophase integral; no impact/speed integral, density rate or source D.',flush=True)
if hashlib.sha256(PRIOR.read_bytes()).hexdigest()!=PRIOR_SHA:
    raise ValueError('Saved original duration receipt changed')
prior=json.loads(PRIOR.read_text());original=prior['results']['failure_audits'][0]
if original['original_failure']['node'][:3]!=inputs['fixed_incoming_triple']:
    raise ValueError('The fixed triple differs from the predeclared saved encounter')
OUTPUT.mkdir(parents=True,exist_ok=True)
metadata=run_metadata(inputs,model='fixed-triple screened encounter gyrophase convergence',
    boundary='first outgoing relative z=+/-24 plane; transmitted and reflected exits; fixed finite flight budget',units='normalized')
dependencies=[Path(__file__).resolve(),ROOT/'src/sato_morrison/controls.py',ROOT/'src/sato_morrison/reference.py']
metadata['experiment_dependency_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
metadata['prior_evidence']={'path':str(PRIOR.relative_to(ROOT)),'sha256':PRIOR_SHA,'producer_commit':prior['commit']}
(OUTPUT/'declared_inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
started=perf_counter();cache={};levels=[];consecutive=0


def evaluate(index,step):
    key=(int(index),float(step))
    if key not in cache:
        cache[key]=phase_attempt(*key)
        (OUTPUT/'nodes_partial.json').write_text(json.dumps(list(cache.values()),indent=2,allow_nan=False)+'\n')
    return cache[key]


with progress('Integrate nested phases, retaining every numerical nonexit and its measure'):
    for order in ORDERS:
        records=[evaluate(index,MAX_STEP) for index in range(0,ORDERS[-1],ORDERS[-1]//order)]
        summary=phase_summary(records,order)
        change=second_change(summary,levels[-1]) if levels else None
        summary['second_moment_relative_change']=change
        check=change is not None and change<=PHASE_TARGET and summary['max_energy_error']<=ENERGY_TARGET
        consecutive=consecutive+1 if check else 0
        summary['successive_passes']=consecutive;levels.append(summary)
        print(f'  N={order}: complete={summary["complete"]}, second={summary["known_positive_second_moment_contribution"]:.9g}, change={change}, consecutive={consecutive}',flush=True)
        (OUTPUT/'levels_partial.json').write_text(json.dumps(levels,indent=2,allow_nan=False)+'\n')
        if consecutive>=SUCCESSIVE_CHECKS:break
final_order=levels[-1]['phase_order'];indices=list(range(0,ORDERS[-1],ORDERS[-1]//final_order))
with progress('Timestep refinement of the entire final gyrophase integral'):
    fine_records=[evaluate(index,REFINED_STEP) for index in indices]
fine=phase_summary(fine_records,final_order)
step_change=second_change(fine,levels[-1]);fine['second_moment_relative_change_from_primary_step']=step_change

controls=[];paths={}
with progress('Independent individual-particle Cartesian equations at three fixed phases'):
    for index in CONTROL_INDICES:
        row=evaluate(index,REFINED_STEP);control={'master_phase_index':index,'phase':2*np.pi*index/ORDERS[-1],
            'relative_status':row['status'],'pair_refinements':[]}
        if row['status']=='outgoing_event':
            for step in PAIR_STEPS:
                begin=perf_counter()
                try:
                    pair=binary_encounter(row['initial_relative_position'],row['initial_relative_velocity'],
                        field=FIELD,strength=STRENGTH,mass=MASS,charge=CHARGE,screening=SCREENING,
                        duration=row['time'],max_step=step,rtol=RTOL)
                    position=pair['positions'][-1,0]-pair['positions'][-1,1]
                    velocity=pair['velocities'][-1,0]-pair['velocities'][-1,1]
                    angle=CHARGE*FIELD/MASS*row['time'];c,s=np.cos(angle),np.sin(angle)
                    kick=np.array([c*velocity[0]-s*velocity[1],s*velocity[0]+c*velocity[1]])-row['initial_relative_velocity'][:2]
                    mean=float(pair['delta_mu'][0]);variance=(MASS/(2*FIELD))**2*THETA/2*np.dot(kick,kick)
                    pair_moments=np.array([mean,mean**2+variance,mean**2-variance])
                    check={'max_step':step,'position_max_absolute_error':float(np.max(abs(position-row['final_relative_position']))),
                        'velocity_max_absolute_error':float(np.max(abs(velocity-row['final_relative_velocity']))),
                        'mu_absolute_error':float(abs(pair['delta_mu'][0]-row['moments'][0])),
                        'independent_COM_moments':pair_moments.tolist(),
                        'COM_moments_max_absolute_error':float(np.max(abs(pair_moments-row['moments']))),
                        'energy_error':pair['energy_error'],'evaluations':pair['evaluations']}
                    check['status']='passed' if check['position_max_absolute_error']<=PAIR_POSITION_TARGET and check['velocity_max_absolute_error']<=PAIR_VELOCITY_TARGET and check['mu_absolute_error']<=PAIR_MU_TARGET and check['COM_moments_max_absolute_error']<=PAIR_MU_TARGET and check['energy_error']<=ENERGY_TARGET else 'unresolved'
                    suffix=f'{index}_{step:g}'
                    paths.update({f'time_{suffix}':pair['time'],f'positions_{suffix}':pair['positions'],f'velocities_{suffix}':pair['velocities']})
                except (RuntimeError,ValueError,FloatingPointError) as error:
                    check={'max_step':step,'status':'unresolved','reason':str(error)}
                check['wall_s']=perf_counter()-begin;control['pair_refinements'].append(check)
        control['status']='passed' if len(control['pair_refinements'])==len(PAIR_STEPS) and all(c['status']=='passed' for c in control['pair_refinements']) else 'unresolved'
        controls.append(control)
np.savez_compressed(OUTPUT/'independent_pair_paths.npz',**paths)
anchor=evaluate(ORDERS[-1]//4,REFINED_STEP)
prior_fine=next(r for r in original['timestep_audit'] if r['max_step']==REFINED_STEP) if any(r['max_step']==REFINED_STEP for r in original['timestep_audit']) else original['timestep_audit'][-1]
anchor_error=float(np.max(abs(np.asarray(anchor['moments'])-prior_fine['moments']))) if anchor['status']=='outgoing_event' else None
phase_passed=consecutive>=SUCCESSIVE_CHECKS
step_passed=step_change is not None and step_change<=TIMESTEP_TARGET and fine['max_energy_error']<=ENERGY_TARGET
metadata['experiment_dependency_sha256_end']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
metadata['experiment_dependencies_unchanged']=metadata['experiment_dependency_sha256']==metadata['experiment_dependency_sha256_end']
passed=phase_passed and step_passed and all(c['status']=='passed' for c in controls) and anchor_error is not None and anchor_error<ANCHOR_TARGET and metadata['experiment_dependencies_unchanged']
records=[evaluate(index,MAX_STEP) for index in indices]
second=np.array([r['moments'][1] if r['status']=='outgoing_event' else np.nan for r in records])
known=np.where(np.isfinite(second),second,0.);total=known.sum()
ranked=np.sort(known)[::-1]
mu_initial=MASS/(2*FIELD)*(THETA+PERPENDICULAR**2/4)
results={'phase_levels':levels,'timestep_check':fine,'controls':controls,
    'phase_status':'passed' if phase_passed else 'unresolved','timestep_status':'passed' if step_passed else 'unresolved',
    'status':'passed_conditional_phase_checks' if passed else 'unresolved',
    'anchor_absolute_moment_difference_from_prior':anchor_error,
    'prior_anchor_max_step':prior_fine['max_step'],
    'mean_initial_mu':mu_initial,
    'aggregate_jump_proxy':float(np.sqrt(fine['phase_averaged_moments'][1])/mu_initial) if fine['complete'] else None,
    'finite_sum_concentration':[{'requested_largest_node_fraction':fraction,
        'actual_node_fraction':max(1,int(np.ceil(final_order*fraction)))/final_order,
        'node_count':max(1,int(np.ceil(final_order*fraction))),
        'known_second_moment_contribution_fraction':float(ranked[:max(1,int(np.ceil(final_order*fraction)))].sum()/total)} for fraction in (.01,.05,.1,.25)] if total>0 else [],
    'unique_relative_solves':len(cache),'wall_s':perf_counter()-started,
    'wall_s_scope':'Numerical solves/checks and intermediate checkpoint/path writes; excludes imports, initial metadata, plots and final output writing.',
    'limits':['Fixed incoming triple, not an annular or speed-integrated rate.','Finite phase differences are convergence diagnostics, not rigorous quadrature error bounds.',
        'All phases retain weight1/N, including unknown nonexits. Partial positive sums are not continuous-integral bounds.',
        'Closest separations are sampled. Jump normalized by mean initial moment is not a pointwise relative jump or an asymptotic ordering proof.',
        'Independent12D controls use separate individual equations/coordinates but the same DOP853 integrator; COM=0 checks the mean, with full COM second moment supplied analytically.',
        'No sourceD calibration, independent-encounter plasma closure or lifetime.']}
metadata['results']=results
metadata['independent_pair_paths_sha256']=hashlib.sha256((OUTPUT/'independent_pair_paths.npz').read_bytes()).hexdigest()
(OUTPUT/'nodes.json').write_text(json.dumps(list(cache.values()),indent=2,allow_nan=False)+'\n')
metadata['nodes_sha256']=hashlib.sha256((OUTPUT/'nodes.json').read_bytes()).hexdigest()
(OUTPUT/'summary.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
with (OUTPUT/'phase_levels.csv').open('w',newline='') as handle:
    columns=['phase_order','complete','missing_count','missing_phase_weight','known_positive_second_moment_contribution','second_moment_relative_change','successive_passes']
    writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader();writer.writerows({key:r[key] for key in columns} for r in levels)

fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
phase=2*np.pi*np.asarray(indices)/ORDERS[-1]
axes[0].semilogy(phase,second,'.-',color='#147D92',ms=3)
axes[0].set(xlabel='incoming gyrophase φ',ylabel='individual COM-averaged second moment',title=f'Fixed triple; {final_order} phases')
axes[0].axvline(np.pi/2,color='0.5',ls=':',lw=1)
axes[1].plot(np.arange(1,final_order+1)/final_order,np.cumsum(ranked)/total if total>0 else np.zeros(final_order),color='#C66B28')
axes[1].set(xlabel='fraction of largest contributing phase nodes',ylabel='fraction of known positive finite sum',title='Contribution concentration',ylim=(0,1.03))
axes[2].plot([r['phase_order'] for r in levels],[r['known_positive_second_moment_contribution'] for r in levels],'o-',label='step .05')
axes[2].plot(final_order,fine['known_positive_second_moment_contribution'],'x',ms=8,label='step .025')
axes[2].set(xlabel='phase nodes',ylabel='second-moment integral / (2π)',title='Nested phase and timestep checks');axes[2].set_xscale('log',base=2);axes[2].legend(fontsize=9)
fig.suptitle(f'Conditional gyrophase convergence: {results["status"]}\nAll phases retain their weight; missing final weight = {levels[-1]["missing_phase_weight"]:.4g}',fontsize=11)
fig.savefig(OUTPUT/'phase_convergence.png',dpi=180);plt.close(fig)
print(f'Finished: {results["status"]}, phase checks={consecutive}, timestep change={step_change}, elapsed={results["wall_s"]:.1f}s',flush=True)
if not passed:raise RuntimeError('Conditional phase checks unresolved; all measurements and failures retained')
