"""Independent saved-state audit; no encounter helper or ODE integration imported."""
from pathlib import Path
from itertools import combinations
from time import perf_counter
import hashlib
import json
import os
import subprocess
import resource
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/reciprocal_encounter_core'
RECEIPT=Path(os.environ.get('SM_RECIPROCAL_AUDIT_OUTPUT',str(OUT/'independent_audit.json'))).resolve()
commit_run=subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],capture_output=True,text=True)
audit_commit=commit_run.stdout.strip() if commit_run.returncode==0 else None
status_run=subprocess.run(['git','-C',str(ROOT),'status','--porcelain','--',str(Path(__file__).resolve())],capture_output=True,text=True)
source_uncommitted=bool(status_run.stdout.strip()) if status_run.returncode==0 else None
began=perf_counter();checks=[];metrics={}
def failed_receipt(exception_type, exception, traceback):
    receipt=dict(status='failed', checks=len(checks), failed_checks=sum(not c['passed'] for c in checks), fatal_error=dict(type=exception_type.__name__, message=str(exception)), failed_check_details=[c['name'] for c in checks if not c['passed']], scope='Independent saved-array audit failed; zero ODE calls.',audit_commit=audit_commit,source_uncommitted=source_uncommitted)
    temporary=RECEIPT.with_name(RECEIPT.name+'.tmp')
    temporary.write_text(json.dumps(receipt,indent=2)+'\n')
    temporary.replace(RECEIPT)
    sys.__excepthook__(exception_type,exception,traceback)
sys.excepthook=failed_receipt
def check(name,ok):
    checks.append(dict(name=name,passed=bool(ok)))
def close(name,x,y,atol=3e-13,rtol=1e-10):
    x,y=np.asarray(x),np.asarray(y)
    check(name,x.shape==y.shape and np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.allclose(x,y,atol=atol,rtol=rtol))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wrap(x):return (x+np.pi)%(2*np.pi)-np.pi
def inverse_state(position,velocity,exit_names,distances):
    # Same-B coordinate reflection plus time reversal, independently vectorized.
    signs=np.where(np.asarray(exit_names)=='transmitted',1.,-1.)
    reflection=np.column_stack((np.ones(len(signs)),-np.ones(len(signs)),-signs))
    r=reflection*position;v=-reflection*velocity
    gc=r[:,:2]+np.column_stack((v[:,1],-v[:,0]))/2
    angles=np.arctan2(gc[:,1],gc[:,0]);c,s=np.cos(angles),np.sin(angles)
    vr=np.column_stack((c*v[:,0]+s*v[:,1],-s*v[:,0]+c*v[:,1],v[:,2]))
    rr=np.column_stack((c*r[:,0]+s*r[:,1],-s*r[:,0]+c*r[:,1],r[:,2]))
    u=vr[:,2];p=np.linalg.norm(vr[:,:2],axis=1)
    phi=(np.arctan2(vr[:,1],vr[:,0])-2*distances/u)%(2*np.pi)
    return np.column_stack((np.linalg.norm(gc,axis=1),u,p,phi)),angles,rr,vr

def reconstructed_moments(ri,ro,vi,vo,times,incident_p=None):
    initial_square=np.sum(vi[:,:2]**2,axis=1) if incident_p is None else np.asarray(incident_p)**2
    c=(np.sum(vo[:,:2]**2,axis=1)-initial_square)/16
    angles=2*times;cs,sn=np.cos(angles),np.sin(angles)
    delta=np.column_stack((cs*vo[:,0]-sn*vo[:,1],sn*vo[:,0]+cs*vo[:,1]))-vi[:,:2]
    variance=np.sum(delta**2,axis=1)/64
    initial_gc=ri[:,:2]+np.column_stack((vi[:,1],-vi[:,0]))/2
    final_gc=ro[:,:2]+np.column_stack((vo[:,1],-vo[:,0]))/2
    spatial=np.sum((final_gc-initial_gc)**2,axis=1)/4
    return np.column_stack((c,c*c+variance,c*c-variance,spatial))

def potential(r):
    radius=np.linalg.norm(r,axis=1)
    return .03*np.exp(-radius/3)/radius

def change(x,y):
    return abs(x-y)/np.array([x[1]/.25,x[1],x[1],x[3]])

def read(path):return json.loads(path.read_text())
summary=read(OUT/'summary.json');result=summary['results'];inputs=summary['inputs']
for group in ('input_sha256','experiment_dependency_sha256'):
    for name,wanted in summary[group].items():check(group+':'+name,sha(ROOT/name)==wanted)
check('scientific_scales_frozen',inputs['quadrature_target']==.05 and inputs['step_target']==.001 and inputs['endpoint_target']==1e-8 and inputs['fd_target']==1e-5)
check('independent_physical_constants',all(inputs[k]==v for k,v in dict(mass=1.,charge=1.,field=2.,strength=.03,screening=3.,theta=.5,distance=24.).items()) and np.array_equal(inputs['bounds'],[[1.2,2.4],[.35,3.],[0.,2.5]]))
check('retrospective_disclosed','inspected before implementation' in inputs['retrospective'])
keys=('nodes','initial_relative_position','final_relative_position','initial_relative_velocity','final_relative_velocity','moments','individual_gc_second','time','distance','step','exit','status','max_actual_step','energy_error','min_separation')
print('Loading immutable saved physical arrays for independent audit',flush=True)
with np.load(ROOT/'results/stratified_encounter_flux/states.npz') as b,np.load(ROOT/'results/antithetic_encounter_flux/states.npz') as n:
    a={k:np.concatenate((b[k],n[k])) for k in keys}
with np.load(ROOT/'results/antithetic_encounter_flux/populations.npz') as f:pop={k:f[k] for k in f.files}
prior=read(ROOT/'results/antithetic_encounter_flux/summary.json')
check('all_saved_physical_states_complete',np.all(a['status']=='passed'))
ri,ro=a['initial_relative_position'],a['final_relative_position'];vi,vo=a['initial_relative_velocity'],a['final_relative_velocity']
reciprocal,angles,prepared_r,prepared_v=inverse_state(ro,vo,a['exit'],a['distance'])
phase=reciprocal[:,3]+2*a['distance']/reciprocal[:,1]
rebuilt_v=np.column_stack((reciprocal[:,2]*np.cos(phase),reciprocal[:,2]*np.sin(phase),reciprocal[:,1]))
rebuilt_r=np.column_stack((reciprocal[:,0]-rebuilt_v[:,1]/2,rebuilt_v[:,0]/2,-a['distance']))
close('all_saved_reciprocal_endpoint_positions',rebuilt_r,prepared_r,atol=1e-8,rtol=0)
close('all_saved_reciprocal_endpoint_velocities',rebuilt_v,prepared_v,atol=1e-8,rtol=0)
# Build the reverse endpoint from the known original incoming data, rather than
# calling the producer's helper a second time; the second map must restore z.
signs=np.where(a['exit']=='transmitted',1.,-1.)
reflection=np.column_stack((np.ones(len(signs)),-np.ones(len(signs)),-signs))
def gauge(v):
    q=reflection*v;c,s=np.cos(angles),np.sin(angles)
    return np.column_stack((c*q[:,0]+s*q[:,1],-s*q[:,0]+c*q[:,1],q[:,2]))
involution,_,_,_=inverse_state(gauge(ri),-gauge(vi),a['exit'],a['distance'])
close('all_saved_reciprocal_involution_magnitudes',involution[:,:3],a['nodes'][:,:3],atol=1e-10,rtol=0)
check('all_saved_reciprocal_involution_phases',np.max(abs(wrap(involution[:,3]-a['nodes'][:,3])))<=1e-8)
values=reconstructed_moments(ri,ro,vi,vo,a['time'],a['nodes'][:,2])
close('independent_all_saved_Cartesian_moments',values,np.column_stack((a['moments'],a['individual_gc_second'])),atol=3e-14,rtol=2e-10)
core=(reciprocal[:,0]>=1.2)&(reciprocal[:,0]<=2.4)&(reciprocal[:,1]>=.35)&(reciprocal[:,1]<=3)&(reciprocal[:,2]<=2.5)
g=np.exp(-potential(ri)/.5)
log_error=abs(-(np.sum(vi**2,axis=1)/4+potential(ri))/.5+(np.sum(vo**2,axis=1)/4+potential(ro))/.5)
bound=1-np.exp(-.03*np.exp(-a['distance']/3)/(a['distance']*.5))
check('finite_plane_Gibbs_factor_analytic_bound',np.all(1-g<=bound+3e-16))
with np.load(OUT/'saved_population_diagnostics.npz') as f:
    close('saved_inverse_magnitudes',reciprocal[:,:3],f['inverse_coordinates'],atol=1e-12,rtol=0)
    check('saved_reciprocal_core_membership',np.array_equal(core,f['inverse_inside']))
    close('saved_Gibbs_factors',g,f['gibbs_factor'],atol=2e-16,rtol=0)
    close('saved_endpoint_Gibbs_identity_residuals',log_error,f['endpoint_log_gibbs_error'],atol=5e-15,rtol=0)
    selected=f['selected_references'].copy()
balance=read(OUT/'saved_balance.json');saved_cases={c['case']:c for c in balance['cases']};core_integrals={};rebuilt_gates={}
for c in prior['results']['cases']:
    label=c['case'];ids=np.r_[pop[label+'_original'],pop[label+'_partner']];w=np.tile(pop[label+'_pair_weights']/2,2)
    mask=core[ids];weighted=w*g[ids]
    vectors={name:np.sum(weighted[m,None]*values[ids[m]],axis=0) for name,m in [('total',np.ones(len(ids),bool)),('core',mask),('escaping',~mask)]}
    old=saved_cases[label]
    for name,vector in vectors.items():close(label+':'+name,vector,old[name])
    close(label+':bare_total',np.sum(w[:,None]*values[ids],axis=0),old['bare_total'])
    close(label+':original_bare_target_preserved',old['bare_total'],c['integrals'])
    for name,m in [('core',mask),('escaping',~mask)]:close(label+':'+name+'_flux',np.sum(weighted[m]),old[name+'_flux'])
    close(label+':partition',vectors['total'],vectors['core']+vectors['escaping'])
    zero=abs(vectors['core'][0])/(vectors['core'][1]/.25)
    close(label+':zero_residual',zero,old['core_zero_residual'])
    core_integrals[label]=vectors['core']
    if label.endswith('N2048') or label=='alternate_partition':rebuilt_gates[label+'_zero']=zero<=.05
comparisons={}
for s in (4201,4202,4203,4204):
    comparisons[f'sample_{s}']=change(core_integrals[f'seed{s}_N2048'],core_integrals[f'seed{s}_N1024'])
    comparisons[f'partition_{s}']=change(core_integrals[f'seed{s}_N2048'],core_integrals['alternate_partition'])
for x,y in combinations((4201,4202,4203,4204),2):comparisons[f'replica_{x}_{y}']=change(core_integrals[f'seed{x}_N2048'],core_integrals[f'seed{y}_N2048'])
for name,v in comparisons.items():
    close('core_comparison:'+name,v,balance['comparisons'][name]);rebuilt_gates[name]=bool(np.max(v)<=.05)
check('all_core_gates_faithfully_reproduced',rebuilt_gates==balance['required_gates']==result['saved_core_gates'])
check('core_failure_preserved',balance['status']=='unresolved' and result['saved_core_status']=='unresolved' and not all(rebuilt_gates.values()))
selection=read(OUT/'selection.json');check('fixed128_distinct_controls',len(selected)==len(set(selected))==128)
check('selection_references_match',np.array_equal(selected,[r['reference'] for r in selection['controls']]))
close('selection_nodes_match',a['nodes'][selected],selection['nodes'],atol=0,rtol=0)
attempts=read(OUT/'attempts.json');cache={(tuple(r['node']),r['step']):r for r in attempts}
check('unique_attempt_cache',len(cache)==len(attempts)==result['new_encounters'])
check('all_fresh_attempts_complete',all(r['status']=='passed' for r in attempts))
for index,r in enumerate(attempts):
    t=r['trajectory'];r0,r1,v0,v1=(np.asarray(t[k])[None,:] for k in ('initial_relative_position','final_relative_position','initial_relative_velocity','final_relative_velocity'))
    rv,angle,pr,pv=inverse_state(r1,v1,[t['exit']],np.array([24.]))
    close(f'fresh_{index}:reciprocal_node',np.r_[rv[0,:3],wrap(rv[0,3]-r['reverse_node'][3])],np.r_[r['reverse_node'][:3],0.],atol=1e-10,rtol=0)
    close(f'fresh_{index}:gauge_angle',wrap(angle[0]-r['angle']),0,atol=1e-13,rtol=0)
    close(f'fresh_{index}:Cartesian_moments',reconstructed_moments(r0,r1,v0,v1,np.array([t['time']]),np.array([r['node'][2]]))[0],r['moments'],atol=3e-14,rtol=2e-10)
    fresh_log_error=float(abs(-(np.sum(v0*v0)/4+potential(r0)[0])/.5+(np.sum(v1*v1)/4+potential(r1)[0])/.5))
    close(f'fresh_{index}:Gibbs_identity',fresh_log_error,r['endpoint_log_gibbs_error'],atol=5e-15,rtol=0)
    # Input order is tested independently: the ODE starts at the declared b,u,p,phi.
    b,u,p,phi=r['node'];phase=phi+48/u;v=np.array([p*np.cos(phase),p*np.sin(phase),u]);pos=np.array([b-v[1]/2,v[0]/2,-24.])
    close(f'fresh_{index}:declared_input_velocity',v,v0[0],atol=2e-13,rtol=0)
    close(f'fresh_{index}:declared_input_position',pos,r0[0],atol=2e-13,rtol=0)
forward=[cache[(tuple(a['nodes'][i]),step)] for i in selected for step in (.05,.025)]
links=[]
for index,f in enumerate(forward):
    for step in ((.2,.05,.025) if f['step']==.025 else (.05,)):links.append((index,step))
check('roundtrip_population_cardinality',len(links)==len(result['roundtrips'])==512)
finest=[]
for index,((fi,step),saved) in enumerate(zip(links,result['roundtrips'])):
    f=forward[fi];back=cache[(tuple(f['reverse_node']),step)];t=f['trajectory'];actual=back['trajectory']
    z=1 if t['exit']=='transmitted' else -1;S=np.array([1,-1,-z]);c,s=np.cos(f['angle']),np.sin(f['angle'])
    def rg(v):
        x,y,z=S*np.asarray(v);return np.array([c*x+s*y,-s*x+c*y,z])
    err=max(np.max(abs(np.asarray(actual['final_relative_position'])-rg(t['initial_relative_position']))),np.max(abs(np.asarray(actual['final_relative_velocity'])+rg(t['initial_relative_velocity']))))
    time_error=abs(actual['time']-t['time']);mom=change(np.asarray(f['moments']),np.asarray(back['moments'])*np.array([-1,1,1,1]))
    close(f'roundtrip_{index}:endpoint',err,saved['endpoint_error']);close(f'roundtrip_{index}:time',time_error,saved['time_error'])
    close(f'roundtrip_{index}:moments',mom,saved['moment_changes'],atol=1e-10,rtol=1e-8)
    check(f'roundtrip_{index}:exit',saved['same_exit']==(actual['exit']==t['exit']))
    if f['step']==step==.025:finest.append((err,time_error,mom,actual['exit']==t['exit']))
for side,saved_rows in [('forward',result['step_checks']),('reverse',result['reverse_step_checks'])]:
    for i,saved in enumerate(saved_rows):
        if side=='forward':
            fine=forward[2*i+1];coarse=forward[2*i];baseline=values[selected[i]]
        else:
            f=forward[2*i+1];node=tuple(f['reverse_node']);fine=cache[(node,.025)];coarse=cache[(node,.05)];baseline=cache[(node,.2)]['moments']
        close(f'{side}_step_{i}:change',change(np.asarray(fine['moments']),np.asarray(coarse['moments'])),saved['changes'])
        close(f'{side}_step_{i}:baseline',change(np.asarray(fine['moments']),np.asarray(baseline)),saved['baseline_changes'])
        close(f'{side}_step_{i}:actual_ratio',fine['trajectory']['max_actual_step']/coarse['trajectory']['max_actual_step'],saved['actual_step_ratio'])
fd_gates=[]
for n,saved in enumerate(result['finite_difference']):
    node=np.asarray(saved['node']);center=cache[(tuple(node),.025)];determinants=[];branches=[]
    for j,epsilon in enumerate(inputs['fd_levels']):
        jac=np.empty((4,4));same=True
        for axis in range(4):
            h=epsilon*max(1,abs(node[axis]));pair=[]
            for sign in (-1,1):
                x=node.copy();x[axis]+=sign*h;r=cache[(tuple(x),.025)];pair.append(np.asarray(r['reverse_node']));same &= r['trajectory']['exit']==center['trajectory']['exit']
            delta=pair[1]-pair[0];delta[3]=wrap(delta[3]);jac[:,axis]=delta/(2*h)
        determinant=abs(np.linalg.det(jac));expected=np.prod(node[:3])/np.prod(center['reverse_node'][:3]);relative=abs(determinant/expected-1);level=saved['levels'][j]
        close(f'FD_{n}_{j}:matrix',jac,level['jacobian'],atol=2e-11,rtol=1e-8)
        close(f'FD_{n}_{j}:determinant',determinant,level['absolute_determinant']);close(f'FD_{n}_{j}:expected',expected,level['expected_determinant']);close(f'FD_{n}_{j}:error',relative,level['relative_error'],atol=1e-12,rtol=1e-8)
        check(f'FD_{n}_{j}:branch',same==level['exit_branch_stable']);determinants.append(determinant);branches.append(same)
    last=abs(determinants[-1]/determinants[-2]-1);close(f'FD_{n}:last_change',last,saved['last_relative_change'],atol=1e-12,rtol=1e-8)
    fd_gates.append(all(branches) and last<=1e-5 and relative<=1e-5)
gates=dict(complete=not result['cap_exhausted'] and all(r['status']=='passed' for r in attempts) and len(finest)==128,
    endpoints=all(r[0]<=1e-8 for r in finest),reciprocal_times=all(r[1]<=1e-8 for r in finest),reciprocal_moments=all(max(r[2])<=.001 and r[3] for r in finest),
    active_step=all(max(r['changes'])<=.001 and max(r['baseline_changes'])<=.001 for r in result['step_checks']+result['reverse_step_checks']),
    actual_step_sizes=all(r['actual_step_ratio']<=.6 for r in result['step_checks']+result['reverse_step_checks']),
    energy=all(r['trajectory']['energy_error']<=1e-9 for r in attempts),saved_gibbs_identity=float(max(log_error))<=1e-9,fresh_gibbs_identity=all(r['endpoint_log_gibbs_error']<=1e-9 for r in attempts),jacobian=all(fd_gates),input_integrity=True,source_integrity=True)
check('all_trajectory_gates_faithfully_reproduced',gates==result['trajectory_gates'])
check('trajectory_status_faithfully_reproduced',result['trajectory_status']==('passed' if all(gates.values()) else 'unresolved'))
metrics.update(accessible_saved_states=len(a['nodes']),fresh_states=len(attempts),selected_controls=len(selected),
    max_endpoint_log_Gibbs_error=float(max(log_error)),max_Gibbs_correction=float(max(1-g)),
    max_finest_roundtrip_endpoint=float(max(x[0] for x in finest)),max_finest_roundtrip_time=float(max(x[1] for x in finest)),
    failed_core_gates=[k for k,v in rebuilt_gates.items() if not v],failed_trajectory_gates=[k for k,v in gates.items() if not v],
    max_core_replica_comparison=float(max(max(v) for k,v in comparisons.items() if k.startswith('replica_'))))
failed=[c['name'] for c in checks if not c['passed']]
receipt=dict(status='failed' if failed else 'passed',checks=len(checks),failed_checks=len(failed),failed_check_details=failed,metrics=metrics,wall_s=perf_counter()-began,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
    scope='Independent saved-array and endpoint algebra audit; faithful reproduction of failed scientific gates is an audit success; zero ODE calls.',
    source_sha256=sha(Path(__file__)),summary_sha256=sha(OUT/'summary.json'),audit_commit=audit_commit,source_uncommitted=source_uncommitted)
temporary=RECEIPT.with_name(RECEIPT.name+'.tmp')
temporary.write_text(json.dumps(receipt,indent=2)+'\n')
temporary.replace(RECEIPT)
print(json.dumps(receipt,indent=2))
sys.exit(1 if failed else 0)
