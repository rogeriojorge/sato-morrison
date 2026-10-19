"""Same-field reciprocal encounters and retrospective finite-plane core balance.

The saved incoming population is used directly: constructed inverse partners
are never averaged into it. Fixed selected fresh trajectories verify the map.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from time import perf_counter
import hashlib
import json
import multiprocessing as mp
import os
import resource
import sys
import numpy as np
from sato_morrison.controls import (encounter_relative, encounter_thermal_moments,
    encounter_same_field_reverse_incoming)
from sato_morrison.reference import run_metadata

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'results/antithetic_encounter_flux'
OUTPUT=ROOT/os.environ.get('SM_RECIPROCAL_OUTPUT','results/reciprocal_encounter_core')
M,Q,B,K,LAMBDA,THETA=1.,1.,2.,.03,3.,.5
OMEGA=Q*B/M
BOUNDS=((1.2,2.4),(.35,3.),(0.,2.5))
GIBBS_TARGET=1e-9
DISTANCE,RTOL,FLIGHT=24.,2e-13,48.
STEPS=(.2,.05,.025)
WORKERS,BATCH,CAP=4,32,600.
QUADRATURE_TARGET,ENDPOINT_TARGET,STEP_TARGET,ENERGY_TARGET=.05,1e-8,.001,1e-9
FD_TARGET=1e-5
FD_LEVELS=(1e-4,5e-5,2.5e-5)
FD_NODES=((1.7,1.1,.4,.6),(2.,1.4,2.7,2.2),(1.4,.9,.6,5.3))
MU=M*THETA/B
SOURCE=Path(__file__).resolve()
DEPENDENCIES=(SOURCE,ROOT/'src/sato_morrison/controls.py',ROOT/'src/sato_morrison/reference.py')
INPUT_PATHS=(PRIOR/'summary.json',PRIOR/'states.npz',PRIOR/'populations.npz',ROOT/'results/stratified_encounter_flux/states.npz')

def native(value):
    if isinstance(value,np.ndarray):return native(value.tolist())
    if isinstance(value,np.generic):return native(value.item())
    if isinstance(value,dict):return {str(k):native(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [native(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):raise ValueError('Nonfinite receipt value')
    return value

def write(name,value):
    path=OUTPUT/name
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(native(value),indent=2,allow_nan=False)+'\n')
    temporary.replace(path)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rss():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
def rotate(vector,angle):
    x,y,z=vector;c,s=np.cos(angle),np.sin(angle)
    return np.array([c*x+s*y,-s*x+c*y,z])
def potential(position):
    radius=np.linalg.norm(position,axis=-1)
    return K*np.exp(-radius/LAMBDA)/radius

def solve(spec):
    node,step=spec;began=perf_counter()
    record=dict(node=node,step=step)
    try:
        impact,parallel,perpendicular,phase=node
        r=encounter_relative(impact,phase,parallel,perpendicular,field=B,strength=K,screening=LAMBDA,start_distance=DISTANCE,
            mass=M,charge=Q,max_step=step,rtol=RTOL,flight_time_factor=FLIGHT)
        reverse,angle=encounter_same_field_reverse_incoming(r,field=B,mass=M,charge=Q,start_distance=DISTANCE)
        vi=r['initial_relative_velocity'];vo=r['final_relative_velocity'];ri=r['initial_relative_position'];ro=r['final_relative_position']
        kick=(ro[:2]+np.array([vo[1],-vo[0]])/OMEGA)-(ri[:2]+np.array([vi[1],-vi[0]])/OMEGA)
        record.update(status='passed',trajectory=r,reverse_node=reverse,angle=angle,
            moments=np.r_[encounter_thermal_moments(r,THETA,M,B),kick@kick/4],
            endpoint_log_gibbs_error=float(abs((M*(vi@vi-vo@vo)/4+potential(ri)-potential(ro))/(M*THETA))),
            worker_peak_rss_bytes=rss())
    except Exception as error:
        record.update(status='unresolved',error_type=type(error).__name__,reason=str(error),
            diagnostics=getattr(error,'integration_diagnostics',{}))
    record['wall_s']=perf_counter()-began
    return native(record)

if OUTPUT.exists():raise FileExistsError(f'Preserve existing evidence: {OUTPUT}')
OUTPUT.mkdir(parents=True)
started=perf_counter()
source_hashes={str(p.relative_to(ROOT)):sha(p) for p in DEPENDENCIES}
input_hashes={str(p.relative_to(ROOT)):sha(p) for p in INPUT_PATHS}
prior=json.loads((PRIOR/'summary.json').read_text())
for key,wanted in dict(mass=M,charge=Q,field=B,strength=K,screening=LAMBDA,theta=THETA).items():
    if prior['inputs'][key]!=wanted:raise ValueError(f'Archived physical input differs: {key}')
inputs=dict(mass=M,charge=Q,field=B,strength=K,screening=LAMBDA,theta=THETA,distance=DISTANCE,bounds=BOUNDS,gibbs_log_identity_target=GIBBS_TARGET,
    steps=STEPS,rtol=RTOL,flight_time_factor=FLIGHT,workers=WORKERS,batch_size=BATCH,runtime_cap_s=CAP,
    control_count=128,endpoint_target=ENDPOINT_TARGET,step_target=STEP_TARGET,energy_target=ENERGY_TARGET,
    quadrature_target=QUADRATURE_TARGET,fd_target=FD_TARGET,fd_levels=FD_LEVELS,fd_nodes=FD_NODES,
    fd_rule='central difference of four incoming coordinates; h=epsilon*max(1,abs(coordinate)); wrapped phase differences; maxstep .025',
    selection='unique fine L24 physical members; up to16 largest weighted M2 in each exit/cutoff class,16 closest,16 longest reflected,16 nearest cutoff,then weighted M2 fill; no selection replacement after execution',
    reciprocity='S=diag(1,-1,-s); reciprocal endpoint is (S*rout,-S*wout); rotate guiding center to+x at unchanged qB',
    gibbs='g=exp(-U_in/(m*theta)); surface flux g*b*u*p*exp(-(u^2+p^2)/(4theta)) invariant',
    core='C=D intersect R(D); saved original and antithetic members keep original half-weights; inverse partners are never added to this population',
    retrospective='Core residuals were inspected before implementation, including approximately5.50% core replica failure; no preregistered or independent quadrature claim',
    stop_rule='fixed controls and FD nodes; runtime cap between batches; preserve all failed/missing weights; no adaptive rescue',
    scope='Finite-plane reversible scattering and finite-box core balance only; no global plasma coefficient, screening closure, mixing time or lifetime')
write('declared_inputs.json',inputs)
print('Preparing saved population and fixed 128 controls before fresh trajectories.',flush=True)
# Start worker processes before run_metadata initializes the JAX backend.
pool=ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('fork'))
list(pool.map(float,range(WORKERS)))
import jax
jax.config.update('jax_enable_x64',True)
context=run_metadata(inputs,model='same-field reciprocal finite-plane core diagnostic',boundary='first outgoing +/-L plane; both exits retained',units='normalized')
context.update(experiment_dependency_sha256=source_hashes,input_sha256=input_hashes)
write('run_context.json',context)
with np.load(ROOT/'results/stratified_encounter_flux/states.npz') as f:base={k:f[k].copy() for k in f.files}
with np.load(PRIOR/'states.npz') as f:new={k:f[k].copy() for k in f.files}
with np.load(PRIOR/'populations.npz') as f:pop={k:f[k].copy() for k in f.files}
a={k:np.concatenate((base[k],new[k])) for k in base}
prior=json.loads((PRIOR/'summary.json').read_text())
if not np.all(a['status']=='passed'):raise ValueError('Incomplete original archive')
wi,wo=a['initial_relative_velocity'],a['final_relative_velocity']
ri,ro=a['initial_relative_position'],a['final_relative_position']
outgc=ro[:,:2]+np.column_stack((wo[:,1],-wo[:,0]))/OMEGA
inverse=np.column_stack((np.linalg.norm(outgc,axis=1),abs(wo[:,2]),np.linalg.norm(wo[:,:2],axis=1)))
inside=np.all((inverse>=np.asarray(BOUNDS)[:,0])&(inverse<=np.asarray(BOUNDS)[:,1]),axis=1)
g=np.exp(-potential(ri)/(M*THETA))
gout=np.exp(-potential(ro)/(M*THETA))
log_gibbs_error=abs(-potential(ri)/(M*THETA)-np.sum(wi*wi,axis=1)/(4*THETA)+potential(ro)/(M*THETA)+np.sum(wo*wo,axis=1)/(4*THETA))
values=np.column_stack((a['moments'],a['individual_gc_second']))
cases=[];core_by_label={};fine_ids=[];weighted=np.zeros(len(inside))
for c in prior['results']['cases']:
    label=c['case'];ids=np.r_[pop[label+'_original'],pop[label+'_partner']];weights=np.tile(pop[label+'_pair_weights']/2,2)
    corrected=weights*g[ids];mask=inside[ids]
    total=np.sum(corrected[:,None]*values[ids],axis=0)
    core=np.sum(corrected[mask,None]*values[ids[mask]],axis=0)
    escaping=np.sum(corrected[~mask,None]*values[ids[~mask]],axis=0)
    row=dict(case=label,core=core,total=total,escaping=escaping,
        core_flux=float(corrected[mask].sum()),escaping_flux=float(corrected[~mask].sum()),
        core_zero_residual=float(abs(core[0])/(core[1]/MU)),
        partition_error=float(np.max(abs(total-core-escaping))),
        bare_total=np.sum(weights[:,None]*values[ids],axis=0))
    cases.append(native(row));core_by_label[label]=core
    if label.endswith('N2048') or label=='alternate_partition':
        fine_ids.extend(ids.tolist());np.add.at(weighted,ids,weights*values[ids,1])
ids=np.unique(fine_ids)
rank=lambda pool,key:sorted((int(i) for i in pool),key=lambda i:(-float(key[i]),i))
selected=[];selection=[]
def take(candidates,count,reason):
    added=0
    for i in candidates:
        if i in selected:continue
        selected.append(i);selection.append(dict(reference=i,reason=reason,exit=str(a['exit'][i]),inverse_inside=bool(inside[i])))
        added+=1
        if added==count:return
for exit_kind in ('transmitted','reflected'):
    for flag in (True,False):
        take(rank(ids[(a['exit'][ids]==exit_kind)&(inside[ids]==flag)],weighted),16,f'{exit_kind}_inverse_{"inside" if flag else "outside"}')
take(rank(ids,-a['min_separation']),16,'closest')
take(rank(ids[a['exit'][ids]=='reflected'],a['time']),16,'longest_reflected')
margin=np.min(abs(inverse[:,None,:]-np.asarray(BOUNDS).T[None,:,:]),axis=(1,2))
take(rank(ids,-margin),16,'nearest_inverse_cutoff')
take(rank(ids,weighted),128-len(selected),'largest_weighted_second_fill')
if len(selected)!=128:raise ValueError('Fixed128 controls unavailable')
write('selection.json',dict(controls=selection,nodes=a['nodes'][selected],input_sha256=input_hashes))
np.savez_compressed(OUTPUT/'saved_population_diagnostics.npz',inverse_coordinates=inverse,inverse_inside=inside,gibbs_factor=g,endpoint_log_gibbs_error=log_gibbs_error,selected_references=selected)
comparisons={}
def change(x,y):return (abs(x-y)/np.array([x[1]/MU,x[1],x[1],x[3]])).tolist()
for seed in (4201,4202,4203,4204):
    comparisons[f'sample_{seed}']=change(core_by_label[f'seed{seed}_N2048'],core_by_label[f'seed{seed}_N1024'])
    comparisons[f'partition_{seed}']=change(core_by_label[f'seed{seed}_N2048'],core_by_label['alternate_partition'])
for x,y in combinations((4201,4202,4203,4204),2):comparisons[f'replica_{x}_{y}']=change(core_by_label[f'seed{x}_N2048'],core_by_label[f'seed{y}_N2048'])
core_gates={k:max(v)<=QUADRATURE_TARGET for k,v in comparisons.items()}
core_gates.update({c['case']+'_zero':c['core_zero_residual']<=QUADRATURE_TARGET for c in cases if c['case'].endswith('N2048') or c['case']=='alternate_partition'})
write('saved_balance.json',dict(cases=cases,comparisons=comparisons,required_gates=core_gates,
    status='passed' if all(core_gates.values()) else 'unresolved',max_endpoint_log_gibbs_error=float(max(log_gibbs_error)),
    max_gibbs_correction=float(max(abs(g-1))),scope=inputs['retrospective']))
cache={};attempts=[];cap_exhausted=False

def evaluate(specs):
    global cap_exhausted
    normalized=[(tuple(map(float,node)),float(step)) for node,step in specs]
    pending=list(dict.fromkeys(s for s in normalized if s not in cache))
    for start in range(0,len(pending),BATCH):
        if perf_counter()-started>CAP:
            cap_exhausted=True;break
        batch=pending[start:start+BATCH]
        records=list(pool.map(solve,batch))
        for key,record in zip(batch,records):cache[key]=record;attempts.append(record)
        write(f'checkpoint_{len(attempts):04d}.json',records)
        write('progress.json',dict(elapsed_s=perf_counter()-started,attempts=len(attempts),failures=sum(r['status']!='passed' for r in attempts),parent_peak_rss_bytes=rss()))
        print(f'Saved {len(attempts)} fresh states; elapsed {perf_counter()-started:.1f}s',flush=True)
    return [cache.get(s,dict(status='not_run',node=s[0],step=s[1])) for s in normalized]

print('Launching fixed fresh forward and reciprocal trajectories; declared 600-second cap.',flush=True)
forward=evaluate([(a['nodes'][i],step) for i in selected for step in STEPS[1:]])
reverse_specs=[];reverse_links=[]
for index,f in enumerate(forward):
    if f['status']=='passed':
        for step in (STEPS if f['step']==STEPS[-1] else (f['step'],)):
            reverse_specs.append((f['reverse_node'],step));reverse_links.append((index,step))
reverse=evaluate(reverse_specs)
roundtrips=[]
for (index,step),back in zip(reverse_links,reverse):
    f=forward[index];entry=dict(forward_index=index,forward_step=f['step'],reverse_step=step,complete=back['status']=='passed')
    if entry['complete']:
        r=f['trajectory'];z=1. if r['exit']=='transmitted' else -1.;S=np.array([1.,-1.,-z]);angle=f['angle']
        expected_r=rotate(S*np.asarray(r['initial_relative_position']),angle)
        expected_v=rotate(-S*np.asarray(r['initial_relative_velocity']),angle)
        actual=back['trajectory']
        endpoint=max(np.max(abs(np.asarray(actual['final_relative_position'])-expected_r)),np.max(abs(np.asarray(actual['final_relative_velocity'])-expected_v)))
        fm,bm=np.asarray(f['moments']),np.asarray(back['moments']);expected_m=fm*np.array([-1.,1.,1.,1.])
        scale=np.array([fm[1]/MU,fm[1],fm[1],fm[3]])
        entry.update(endpoint_error=float(endpoint),time_error=abs(actual['time']-r['time']),moment_changes=(abs(bm-expected_m)/scale).tolist(),same_exit=actual['exit']==r['exit'])
    roundtrips.append(entry)
step_checks=[]
for index,i in enumerate(selected):
    coarse,fine=forward[2*index:2*index+2]
    entry=dict(reference=i,complete=coarse['status']==fine['status']=='passed')
    if entry['complete']:
        baseline=values[i];f=np.asarray(fine['moments']);c=np.asarray(coarse['moments'])
        entry.update(changes=change(f,c),baseline_changes=change(f,baseline),actual_step_ratio=fine['trajectory']['max_actual_step']/coarse['trajectory']['max_actual_step'])
    step_checks.append(entry)
reverse_step_checks=[]
for index in range(1,len(forward),2):
    group={step:r for (fi,step),r in zip(reverse_links,reverse) if fi==index}
    entry=dict(forward_index=index,complete=len(group)==3 and all(r['status']=='passed' for r in group.values()))
    if entry['complete']:
        fine=group[.025];coarse=group[.05];baseline=group[.2]
        entry.update(changes=change(np.asarray(fine['moments']),np.asarray(coarse['moments'])),
            baseline_changes=change(np.asarray(fine['moments']),np.asarray(baseline['moments'])),
            actual_step_ratio=fine['trajectory']['max_actual_step']/coarse['trajectory']['max_actual_step'])
    reverse_step_checks.append(entry)
fd=[]
for node in FD_NODES:
    center=evaluate([(node,.025)])[0];levels=[]
    for epsilon in FD_LEVELS:
        specs=[]
        for axis in range(4):
            h=epsilon*max(1.,abs(node[axis]))
            for sign in (-1,1):
                perturbed=np.array(node);perturbed[axis]+=sign*h;specs.append((perturbed,.025))
        records=evaluate(specs)
        entry=dict(epsilon=epsilon,complete=center['status']=='passed' and all(r['status']=='passed' for r in records))
        entry['exit_branch_stable']=entry['complete'] and all(r['trajectory']['exit']==center['trajectory']['exit'] for r in records)
        if entry['complete']:
            jac=np.empty((4,4))
            for axis in range(4):
                delta=np.asarray(records[2*axis+1]['reverse_node'])-np.asarray(records[2*axis]['reverse_node']);delta[3]=(delta[3]+np.pi)%(2*np.pi)-np.pi
                jac[:,axis]=delta/(2*epsilon*max(1.,abs(node[axis])))
            target=float(np.prod(node[:3])/np.prod(center['reverse_node'][:3]));det=float(abs(np.linalg.det(jac)))
            entry.update(jacobian=jac,absolute_determinant=det,expected_determinant=target,relative_error=abs(det/target-1))
        levels.append(native(entry))
    last_change=abs(levels[-1]['absolute_determinant']/levels[-2]['absolute_determinant']-1) if all(l['complete'] for l in levels) else None
    fd.append(dict(node=node,levels=levels,last_relative_change=last_change))
pool.shutdown()
paired_finest=[r for r in roundtrips if r['forward_step']==.025 and r['reverse_step']==.025]
gates=dict(complete=not cap_exhausted and all(r['status']=='passed' for r in attempts) and all(r['complete'] for r in step_checks+reverse_step_checks+roundtrips) and len(paired_finest)==128,
    endpoints=all(r.get('endpoint_error',1)<=ENDPOINT_TARGET for r in paired_finest),
    reciprocal_times=all(r.get('time_error',1)<=ENDPOINT_TARGET for r in paired_finest),
    reciprocal_moments=all(max(r.get('moment_changes',[1]))<=STEP_TARGET and r.get('same_exit',False) for r in paired_finest),
    active_step=all(max(r.get('changes',[1]))<=STEP_TARGET and max(r.get('baseline_changes',[1]))<=STEP_TARGET for r in step_checks+reverse_step_checks),
    actual_step_sizes=all(r.get('actual_step_ratio',1)<=.6 for r in step_checks+reverse_step_checks),
    energy=all(r.get('trajectory',{}).get('energy_error',1)<=ENERGY_TARGET for r in attempts),
    saved_gibbs_identity=float(max(log_gibbs_error))<=GIBBS_TARGET,
    fresh_gibbs_identity=all(r.get('endpoint_log_gibbs_error',1)<=GIBBS_TARGET for r in attempts),
    jacobian=all(all(level['exit_branch_stable'] for level in r['levels']) and r['last_relative_change'] is not None and r['last_relative_change']<=FD_TARGET and r['levels'][-1].get('relative_error',1)<=FD_TARGET for r in fd),
    input_integrity=all(sha(ROOT/name)==value for name,value in input_hashes.items()),
    source_integrity=all(sha(ROOT/name)==value for name,value in source_hashes.items()))
write('attempts.json',attempts)
result=dict(trajectory_status='passed' if all(gates.values()) else 'unresolved',saved_core_status='passed' if all(core_gates.values()) else 'unresolved',
    trajectory_gates=gates,saved_core_gates=core_gates,roundtrips=roundtrips,step_checks=step_checks,reverse_step_checks=reverse_step_checks,finite_difference=fd,
    new_encounters=len(attempts),failures=sum(r['status']!='passed' for r in attempts),cap_exhausted=cap_exhausted,
    wall_s=perf_counter()-started,parent_peak_rss_bytes=rss(),max_worker_peak_rss_bytes=max((r.get('worker_peak_rss_bytes',0) for r in attempts),default=0),
    memory_scope='Separate lifetime process peaks; no simultaneous aggregate memory measurement.',
    limitations=['Existing core quadrature is retrospective and may fail unchanged5% gates.','Selected trajectory verification does not establish whole-domain integration accuracy.','No constructed reciprocal population is used to force integral cancellation.'])
context['results']=result
write('summary.json',context)
print(f"Finished trajectory {result['trajectory_status']}; saved core {result['saved_core_status']}; {result['wall_s']:.1f}s",flush=True)
sys.exit(0 if all(gates.values()) and all(core_gates.values()) else 1)
