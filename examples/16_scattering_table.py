"""Fresh validation of a refined, positive, periodic scattering-moment table."""
from pathlib import Path
from time import perf_counter
import hashlib
import json
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binom
from sato_morrison.controls import (encounter_relative,encounter_thermal_moments,
    encounter_flux_quadrature,encounter_moment_interpolator)
from sato_morrison.reference import progress,run_metadata

# Refine resolution, keep physical model and accuracy targets fixed.
mass,charge,field=1.,1.,2.
strength,screening,theta=.03,3.,.5
training_bounds=((1.2,2.4),(.7,1.5),(.2,.8))
training_orders=(6,6,6,32)
holdout_count,holdout_seed=256,1311
second_moment_rms_target,second_moment_p95_target=.02,.05
bootstrap_count,bootstrap_seed=2000,1312
start_distance,max_step,rtol=24.,.2,1e-10
impact_derivative_states=8
impact_fd_steps=[.001,.0005,.00025]
derivative_rms_target=.05
table_derivative_step=1e-5
output=Path(__file__).resolve().parents[1]/'results'/'scattering_table'
show_figures=False

inputs={'mass':mass,'charge':charge,'field':field,'strength':strength,'screening':screening,'theta':theta,
    'theta_definition':'single-particle velocity variance k_B*T/m','training_bounds':training_bounds,
    'training_orders':training_orders,'holdout_count':holdout_count,'holdout_seed':holdout_seed,
    'second_moment_rms_target':second_moment_rms_target,'second_moment_p95_target':second_moment_p95_target,
    'bootstrap_count':bootstrap_count,'bootstrap_seed':bootstrap_seed,'start_distance':start_distance,
    'max_step':max_step,'rtol':rtol,'impact_derivative_states':impact_derivative_states,
    'impact_fd_steps':impact_fd_steps,'derivative_rms_target':derivative_rms_target,
    'table_derivative_step':table_derivative_step,
    'method':'spatial cubic of signed c and log(COMvariance); truly periodic cubic phase; exact moment reconstruction',
    'validation_domain':'uniform independent states inside the new training-node inner hull; no extrapolation',
    'previous_test_reuse':'none; seeds701 and1301 are excluded from this test and fitting',
    'scope':'single-encounter conditional moment interpolation, not a plasma or SM collision coefficient'}
output.mkdir(parents=True,exist_ok=True)
print(f'Model=encounter; equal screened thermal particles; uniform B={field}; output={output}',flush=True)
print(f'Training={training_orders}, fresh independent holdout={holdout_count} seed={holdout_seed}; targets RMS={second_moment_rms_target}, p95={second_moment_p95_target}',flush=True)
print('Interpolation preserves positive variance and pair moments. Test interpretation is limited to uniformly sampled states in the stated inner hull.',flush=True)
print('No compilation; direct DOP853 event solves; all comparison inputs are declared before the fresh test.',flush=True)
repository=Path(__file__).resolve().parents[1]
dependencies=[repository/'src/sato_morrison/controls.py',repository/'src/sato_morrison/reference.py',Path(__file__).resolve()]
metadata=run_metadata(inputs,model='encounter',boundary='same finite incoming/outgoing planes, L=24, transmitted/reflected first exit')
metadata['experiment_dependency_sha256']={str(path.relative_to(repository)):hashlib.sha256(path.read_bytes()).hexdigest() for path in dependencies}
(output/'declared_inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
started=perf_counter()


def direct_moment(node):
    impact,parallel,perpendicular,phase=node
    result=encounter_relative(impact,phase,parallel,perpendicular,field=field,strength=strength,screening=screening,start_distance=start_distance,mass=mass,charge=charge,max_step=max_step,rtol=rtol)
    return encounter_thermal_moments(result,theta,mass,field),result

with progress('Building the refined periodic positive table'):
    nodes,weights=encounter_flux_quadrature(training_orders,*training_bounds,theta)
    training=[];energies=[];reflected=0
    for node in nodes:
        moments,result=direct_moment(node)
        training.append(moments);energies.append(result['energy_error'])
        reflected+=int(result['exit']=='reflected')
    training=np.asarray(training)
    axes=[np.unique(nodes[:,i]) for i in range(4)]
    interpolation=encounter_moment_interpolator(axes,training.reshape(*training_orders,3))
    inner_hull=[(float(a[0]),float(a[-1])) for a in axes[:3]]
    print(f'  Training inner hull (impact,parallel,perpendicular)={inner_hull}',flush=True)

with progress('Evaluating the predeclared fresh independent test'):
    rng=np.random.default_rng(holdout_seed)
    heldout=np.column_stack([rng.uniform(a[0],a[-1],holdout_count) for a in axes[:3]]+[rng.uniform(0.,2*np.pi,holdout_count)])
    direct=[];test_energies=[]
    for node in heldout:
        moments,result=direct_moment(node)
        direct.append(moments);test_energies.append(result['energy_error'])
    direct=np.asarray(direct);predicted=interpolation(heldout)
    differences=predicted-direct
    rms=np.sqrt(np.mean(differences**2,axis=0))/np.sqrt(np.mean(direct**2,axis=0))
    relative=np.abs(differences[:,1])/direct[:,1]
    p95=float(np.quantile(relative,.95))
    rank_low=int(binom.ppf(.025,holdout_count,.95));rank_high=int(binom.ppf(.975,holdout_count,.95))
    sorted_relative=np.sort(relative)
    p95_interval=[float(sorted_relative[max(0,rank_low-1)]),float(sorted_relative[min(holdout_count-1,rank_high)])]
    bootstrap_rng=np.random.default_rng(bootstrap_seed)
    resamples=bootstrap_rng.integers(0,holdout_count,size=(bootstrap_count,holdout_count))
    bootstrap_rms=np.sqrt(np.mean(differences[resamples,1]**2,axis=1))/np.sqrt(np.mean(direct[resamples,1]**2,axis=1))
    rms_interval=np.quantile(bootstrap_rms,[.025,.975]).tolist()
    status='passed' if rms[1]<=second_moment_rms_target and p95<=second_moment_p95_target and max(energies+test_energies)<=1e-8 else 'unresolved'
    print(f'  RMS(mean,second,cross)={rms}; p95={p95:.3%}, rank interval={p95_interval}; status={status}',flush=True)

with progress('Independent impact derivative and finite-difference plateau check'):
    derivative_nodes=heldout[:impact_derivative_states]
    table_plus=derivative_nodes.copy();table_plus[:,0]+=table_derivative_step
    table_minus=derivative_nodes.copy();table_minus[:,0]-=table_derivative_step
    table_derivative=(interpolation(table_plus)[:,1]-interpolation(table_minus)[:,1])/(2*table_derivative_step)
    direct_derivatives=[]
    for step in impact_fd_steps:
        derivatives=[]
        for node in derivative_nodes:
            plus=node.copy();minus=node.copy();plus[0]+=step;minus[0]-=step
            positive,_=direct_moment(plus);negative,_=direct_moment(minus)
            derivatives.append((positive[1]-negative[1])/(2*step))
        direct_derivatives.append(derivatives)
    direct_derivatives=np.asarray(direct_derivatives)
    fine=direct_derivatives[-1]
    derivative_plateau=float(np.linalg.norm(fine-direct_derivatives[-2])/np.linalg.norm(fine))
    derivative_error=float(np.linalg.norm(table_derivative-fine)/np.linalg.norm(fine))
    derivative_status='passed' if derivative_error<=derivative_rms_target and derivative_plateau<=.001 else 'unresolved'
    print(f'  Impact derivative relative RMS={derivative_error:.3%}, FD plateau={derivative_plateau:.3e}; status={derivative_status}',flush=True)

fig,axes_plot=plt.subplots(1,3,figsize=(13,4),layout='constrained')
axes_plot[0].scatter(direct[:,1],predicted[:,1],s=10)
lo,hi=direct[:,1].min(),direct[:,1].max();axes_plot[0].plot([lo,hi],[lo,hi],':',color='gray')
axes_plot[0].set(xlabel='fresh direct scattering moment',ylabel='positive periodic interpolation',title=f'Second-moment RMS {rms[1]:.2%}')
axes_plot[1].hist(100*relative,bins=30)
axes_plot[1].axvline(100*second_moment_p95_target,color='gray',linestyle=':',label='p95 target')
axes_plot[1].set(xlabel='absolute relative second-moment error (%)',ylabel='independent states',title=f'p95 {p95:.2%}; {status}');axes_plot[1].legend()
axes_plot[2].scatter(fine,table_derivative,s=24)
lo,hi=fine.min(),fine.max();axes_plot[2].plot([lo,hi],[lo,hi],':',color='gray')
axes_plot[2].set(xlabel='independent direct impact derivative',ylabel='interpolant impact derivative',title=f'Derivative RMS {derivative_error:.2%}')
fig.savefig(output/'table.png',dpi=160)
if show_figures:
    plt.show(block=False)
plt.close(fig)
metadata['experiment_dependency_sha256_end']={str(path.relative_to(repository)):hashlib.sha256(path.read_bytes()).hexdigest() for path in dependencies}
metadata['experiment_dependencies_unchanged']=metadata['experiment_dependency_sha256']==metadata['experiment_dependency_sha256_end']
metadata['results']={'status':status,'rms_errors':rms.tolist(),'second_moment_p95':p95,
    'p95_order_statistic_95_interval':p95_interval,'p95_interval_order_ranks':[rank_low,rank_high],
    'second_moment_rms_bootstrap_95_interval':rms_interval,'one_state_fraction':1/holdout_count,
    'training_inner_hull':inner_hull,'training_states':len(nodes),'max_training_energy_error':max(energies),
    'max_test_energy_error':max(test_energies),'training_reflected_states':reflected,
    'derivative_status':derivative_status,'impact_derivative_relative_rms':derivative_error,
    'direct_fd_plateau_relative_change':derivative_plateau,'wall_s':perf_counter()-started,
    'limitations':['finite independent uniform-state sampling intervals describe this inner-hull error distribution, not a bound over all phase space',
        'table does not extrapolate beyond its training inner hull','derivative test concerns one conditional impact direction at eight independent states',
        'no kinetic closure, inter-encounter correlations, SM coefficient or dipole lifetime is validated']}
(output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
np.savez_compressed(output/'table_data.npz',training_nodes=nodes,training_moments=training,heldout=heldout,direct=direct,predicted=predicted,
    derivative_nodes=derivative_nodes,direct_derivatives=direct_derivatives,table_derivative=table_derivative)
print(f'Refined scattering table finished: interpolation={status}; derivative={derivative_status}; elapsed={perf_counter()-started:.1f}s',flush=True)
