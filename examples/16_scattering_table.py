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
# The original 256-state evidence is immutable. This new stage tests the same table.
robust_holdout_count,robust_holdout_seed=4096,1321
robust_bootstrap_seed=1322
p95_confidence_upper_target=.05
reuse_training_cache=True
original_output=Path(__file__).resolve().parents[1]/'results'/'scattering_table'
cache_data_sha256='7636ce6c8bb55c7106b62693f571dfef30dd69eb729130880030c178c3f9f100'
cache_metadata_sha256='7dc4d6a6173662ad9952ef319383f9b379fa121d8f4e813c71619f8936ff6385'
cache_producer_commit='6d1c83f4ae8cd78e3d66c1f0a929b3a79515f0fb'
output=original_output/'validation_4096'
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
    'robust_holdout_count':robust_holdout_count,'robust_holdout_seed':robust_holdout_seed,
    'robust_bootstrap_seed':robust_bootstrap_seed,'p95_confidence_upper_target':p95_confidence_upper_target,
    'reuse_training_cache':reuse_training_cache,'cache_data_sha256':cache_data_sha256,
    'cache_metadata_sha256':cache_metadata_sha256,'cache_producer_commit':cache_producer_commit,
    'robust_acceptance':'second-moment RMS <=2%, observed p95 <=5%, and two-sided 95% order-statistic p95 upper limit <=5%; no retuning',
    'scope':'single-encounter conditional moment interpolation, not a plasma or SM collision coefficient'}
output.mkdir(parents=True,exist_ok=True)
print(f'Model=encounter; equal screened thermal particles; uniform B={field}; output={output}',flush=True)
print(f'Training={training_orders}, fresh independent holdout={holdout_count} seed={holdout_seed}; targets RMS={second_moment_rms_target}, p95={second_moment_p95_target}',flush=True)
print('Interpolation preserves positive variance and pair moments. Test interpretation is limited to uniformly sampled states in the stated inner hull.',flush=True)
print(f'Fresh robustness validation: n={robust_holdout_count}, seed={robust_holdout_seed}, p95 confidence upper target={p95_confidence_upper_target:.1%}; original evidence preserved.',flush=True)
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

cache_available=reuse_training_cache and (original_output/'table_data.npz').exists() and (original_output/'metadata.json').exists()
cache_controls_compatible=False
if cache_available:
    cached_metadata=json.loads((original_output/'metadata.json').read_text())
    cache_controls_compatible=cached_metadata['experiment_dependency_sha256']['src/sato_morrison/controls.py']==metadata['experiment_dependency_sha256']['src/sato_morrison/controls.py']
cache_used=cache_available and cache_controls_compatible
if cache_available and not cache_controls_compatible:
    print('Controls SHA changed since original cache: explicitly regenerating training and both independent validation stages.',flush=True)
if cache_used:
    with progress('Verifying immutable training cache and original independent test'):
        data_path=original_output/'table_data.npz';producer_path=original_output/'metadata.json'
        if hashlib.sha256(data_path.read_bytes()).hexdigest()!=cache_data_sha256 or hashlib.sha256(producer_path.read_bytes()).hexdigest()!=cache_metadata_sha256:
            raise ValueError('Training cache SHA differs from the predeclared original evidence')
        producer=json.loads(producer_path.read_text())
        if producer['commit']!=cache_producer_commit or not producer['experiment_dependencies_unchanged']:
            raise ValueError('Training producer commit or dependency provenance differs')
        control_key='src/sato_morrison/controls.py'
        if producer['experiment_dependency_sha256'][control_key]!=metadata['experiment_dependency_sha256'][control_key]:
            raise ValueError('Direct-encounter/interpolation controls changed since cache production')
        for key in ['mass','charge','field','strength','screening','theta','training_bounds','training_orders','holdout_count','holdout_seed','start_distance','max_step','rtol','method']:
            if json.dumps(producer['inputs'][key])!=json.dumps(inputs[key]):
                raise ValueError(f'Training/test input changed: {key}')
        with np.load(data_path,allow_pickle=False) as cache:
            nodes=cache['training_nodes'];training=cache['training_moments']
            heldout=cache['heldout'];direct=cache['direct'];predicted=cache['predicted']
            derivative_nodes=cache['derivative_nodes'];direct_derivatives=cache['direct_derivatives'];table_derivative=cache['table_derivative']
        declared_nodes,_=encounter_flux_quadrature(training_orders,*training_bounds,theta)
        if not np.array_equal(nodes,declared_nodes) or training.shape!=(len(nodes),3) or not np.all(np.isfinite(training)):
            raise ValueError('Cached training nodes/moments violate declared table specification')
        axes=[np.unique(nodes[:,i]) for i in range(4)]
        interpolation=encounter_moment_interpolator(axes,training.reshape(*training_orders,3))
        original_rng=np.random.default_rng(holdout_seed)
        declared_heldout=np.column_stack([original_rng.uniform(a[0],a[-1],holdout_count) for a in axes[:3]]+[original_rng.uniform(0.,2*np.pi,holdout_count)])
        if not np.array_equal(heldout,declared_heldout) or not np.allclose(interpolation(heldout),predicted,rtol=1e-12,atol=1e-18):
            raise ValueError('Cached independent test or reconstructed interpolant differs')
        original=producer['results']
        inner_hull=original['training_inner_hull'];energies=[original['max_training_energy_error']]
        test_energies=[original['max_test_energy_error']];reflected=original['training_reflected_states']
        differences=predicted-direct;rms=np.asarray(original['rms_errors'])
        relative=np.abs(differences[:,1])/direct[:,1];p95=original['second_moment_p95']
        p95_interval=original['p95_order_statistic_95_interval'];rank_low,rank_high=original['p95_interval_order_ranks']
        rms_interval=original['second_moment_rms_bootstrap_95_interval'];status=original['status']
        fine=direct_derivatives[-1];derivative_error=original['impact_derivative_relative_rms']
        derivative_plateau=original['direct_fd_plateau_relative_change'];derivative_status=original['derivative_status']
        metadata['training_cache']={'used':True,'data_sha256':cache_data_sha256,'metadata_sha256':cache_metadata_sha256,
            'producer_commit':producer['commit'],'producer_dependency_sha256':producer['experiment_dependency_sha256'],
            'original_256_evidence_preserved':str(original_output.relative_to(repository)),'original_results':original}
else:
    metadata['training_cache']={'used':False,'reason':'disabled, absent, or controls SHA changed; regenerating the same training table and original 256-state stage',
        'cache_available':cache_available,'cache_controls_compatible':cache_controls_compatible}
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

with progress('Testing the same table on 4096 fresh independent states, without retraining'):
    robust_rng=np.random.default_rng(robust_holdout_seed)
    robust_nodes=np.column_stack([robust_rng.uniform(a[0],a[-1],robust_holdout_count) for a in axes[:3]]+[robust_rng.uniform(0.,2*np.pi,robust_holdout_count)])
    robust_direct=np.full((robust_holdout_count,3),np.nan);robust_energies=[];robust_failures=[]
    for i,node in enumerate(robust_nodes):
        try:
            moments,result=direct_moment(node)
            robust_direct[i]=moments;robust_energies.append(result['energy_error'])
        except RuntimeError as error:
            robust_failures.append({'index':i,'node':node.tolist(),'reason':str(error)})
        if (i+1)%512==0:
            print(f'  Fresh states {i+1}/{robust_holdout_count}; unresolved encounters={len(robust_failures)}',flush=True)
    robust_predicted=interpolation(robust_nodes)
    robust_results={'status':'unresolved','states':robust_holdout_count,'seed':robust_holdout_seed,
        'failure_count':len(robust_failures),'failures':robust_failures,'one_state_fraction':1/robust_holdout_count,
        'max_energy_error':max(robust_energies,default=None),'sampling':'independent uniform states in identical inner hull; no training or parameter changes'}
    robust_relative=np.abs(robust_predicted[:,1]-robust_direct[:,1])/robust_direct[:,1]
    if not robust_failures:
        robust_differences=robust_predicted-robust_direct
        robust_rms=np.sqrt(np.mean(robust_differences**2,axis=0))/np.sqrt(np.mean(robust_direct**2,axis=0))
        robust_p95=float(np.quantile(robust_relative,.95))
        robust_low=int(binom.ppf(.025,robust_holdout_count,.95));robust_high=int(binom.ppf(.975,robust_holdout_count,.95))
        sorted_robust=np.sort(robust_relative)
        robust_interval=[float(sorted_robust[max(0,robust_low-1)]),float(sorted_robust[min(robust_holdout_count-1,robust_high)])]
        robust_bootstrap_rng=np.random.default_rng(robust_bootstrap_seed)
        robust_bootstrap=[]
        for batch in range(0,bootstrap_count,100):
            indices=robust_bootstrap_rng.integers(0,robust_holdout_count,size=(min(100,bootstrap_count-batch),robust_holdout_count))
            robust_bootstrap.extend(np.sqrt(np.mean(robust_differences[indices,1]**2,axis=1))/np.sqrt(np.mean(robust_direct[indices,1]**2,axis=1)))
        robust_status='passed' if robust_rms[1]<=second_moment_rms_target and robust_p95<=second_moment_p95_target and robust_interval[1]<=p95_confidence_upper_target and max(robust_energies)<=1e-8 else 'unresolved'
        robust_results.update(status=robust_status,rms_errors=robust_rms.tolist(),second_moment_p95=robust_p95,
            p95_order_statistic_95_interval=robust_interval,p95_interval_order_ranks=[max(1,robust_low),min(robust_holdout_count,robust_high+1)],
            p95_interval_binomial_count_cutoffs=[robust_low,robust_high],
            second_moment_rms_bootstrap_95_interval=np.quantile(robust_bootstrap,[.025,.975]).tolist(),
            fraction_exceeding_five_percent=float(np.mean(robust_relative>second_moment_p95_target)),
            fraction_exceeding_one=float(np.mean(robust_relative>1.)))
        print(f'  Fresh4096 RMS={robust_rms}; p95={robust_p95:.3%}; 95% p95 interval={robust_interval}; status={robust_status}',flush=True)
    else:
        print(f'  Fresh4096 unresolved: {len(robust_failures)} direct encounters failed; no valid-only statistics.',flush=True)

fig_robust,robust_axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
robust_axes[0].scatter(robust_direct[:,1],robust_predicted[:,1],s=3,alpha=.4)
lo,hi=np.nanmin(robust_direct[:,1]),np.nanmax(robust_direct[:,1]);robust_axes[0].plot([lo,hi],[lo,hi],':',color='gray')
robust_axes[0].set(xlabel='fresh direct second moment',ylabel='same interpolated table',title=f'4096 independent states: {robust_results["status"]}')
positive_relative=np.sort(robust_relative[np.isfinite(robust_relative)])
robust_axes[1].plot(positive_relative,np.arange(1,len(positive_relative)+1)/robust_holdout_count)
robust_axes[1].axvline(.05,color='gray',linestyle=':');robust_axes[1].axhline(.95,color='gray',linestyle=':')
robust_axes[1].set(xscale='log',xlabel='absolute relative second-moment error',ylabel='fraction of all 4096 states',title='Empirical error distribution; no retraining')
fig_robust.savefig(output/'robust_validation.png',dpi=160);plt.close(fig_robust)

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
metadata['fresh_4096_results']=robust_results
metadata['results']={'status':status,'rms_errors':rms.tolist(),'second_moment_p95':p95,
    'p95_order_statistic_95_interval':p95_interval,'p95_interval_binomial_count_cutoffs':[rank_low,rank_high],
    'p95_interval_order_ranks':[max(1,rank_low),min(holdout_count,rank_high+1)],
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
    derivative_nodes=derivative_nodes,direct_derivatives=direct_derivatives,table_derivative=table_derivative,
    robust_nodes=robust_nodes,robust_direct=robust_direct,robust_predicted=robust_predicted)
print(f'Refined scattering table finished: original256={status}; fresh4096={robust_results["status"]}; derivative={derivative_status}; elapsed={perf_counter()-started:.1f}s',flush=True)
