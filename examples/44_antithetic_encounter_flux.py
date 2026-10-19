"""Exact physical antithetic extension of a saved bounded flux quadrature.

Each old incident node keeps half its original mass and gains an equally
weighted physical partner at phase pi-phi. No physical state is discarded.
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
from sato_morrison.controls import encounter_relative, encounter_thermal_moments
from sato_morrison.reference import run_metadata

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT/'results/stratified_encounter_flux'
OUTPUT = ROOT/os.environ.get('SM_ANTITHETIC_FLUX_OUTPUT','results/antithetic_encounter_flux')
REPLAY = os.environ.get('SM_ANTITHETIC_RECEIPT_REPLAY','0')=='1'
MASS,CHARGE,FIELD,STRENGTH,SCREENING,THETA = 1.,1.,2.,.03,3.,.5
BASE_STEP,RTOL,FLIGHT_FACTOR = .2,1e-12,48.
STEPS,DISTANCES = (.1,.05,.025),(40.,48.)
WORKERS,CHECKPOINT,CAP = 4,512,3000.
QUADRATURE_TARGET,PLANE_TARGET,STEP_TARGET,ENERGY_TARGET = .05,.005,.001,1e-9
MU_REF=MASS*THETA/FIELD
prior=json.loads((PRIOR/'summary.json').read_text())
INPUTS=dict(mass=MASS,charge=CHARGE,field=FIELD,strength=STRENGTH,screening=SCREENING,
    theta=THETA,base_step=BASE_STEP,rtol=RTOL,flight_time_factor=FLIGHT_FACTOR,
    steps=STEPS,distances=DISTANCES,workers=WORKERS,runtime_cap_s=CAP,
    quadrature_target=QUADRATURE_TARGET,plane_target=PLANE_TARGET,step_target=STEP_TARGET,
    energy_target=ENERGY_TARGET,phase_transform_tolerance=1e-13,units=prior['units'],
    base_levels=prior['inputs']['base_levels'],seeds=prior['inputs']['seeds'],
    alternate_seed=prior['inputs']['alternate_seed'],bounded_flux_per_density=prior['inputs']['bounded_flux_per_density'],
    main_strata=prior['inputs']['main_strata'],alternate_strata=prior['inputs']['alternate_strata'],
    transform='T(b,u,p,phi)=(b,u,p,(pi-phi) modulo2pi)',
    measure='T preserves every phase cell and uniform phase mass; each original/partner receives Ws/(2Ns), pair mean receives Ws/Ns; all reflected exits and failed weights retained',
    theory='For the symmetric free helix the first-Born magnetic-moment work is odd under T with z->-z, so pairing cancels that component. Actual nonlinear observables are solved for both members; no variance or QMC convergence guarantee follows.',
    controls='top four weighted paired second moments from each of four main scrambles and alternate; closest four and longest four reflected new partners; sixteen predetermined fresh controls and historical weak boundary control from base experiment; both members at every active step and farther plane',
    acceptance='same four sample, six independent-replica, four alternate-partition, two full matched boundary, selected paired and memberwise boundary, paired and memberwise active-step, energy and integrity gates as base; no failed criterion relaxed',
    stop_rule='one fixed physical antithetic extension; all original inputs and weights frozen; cap between batches; no outcome-dependent continuation or failure renormalization',
    archive='states.npz stores new partners and missing refined original/control states only; global reference ids <base_state_count address immutable base states, ids >=base_state_count address new states at id-base_state_count',
    scope='bounded incident-domain empirical quadrature; no omitted-domain moment bound, reverse-complete incident population, physical screening closure, plasma coefficient or lifetime')
for name in ('mass','charge','field','strength','screening','theta','rtol','flight_time_factor'):
    if prior['inputs'][name]!=INPUTS[name]:
        raise ValueError(f'Prior physical/numerical input differs: {name}')
INPUTS['prior_artifact_sha256']={str(PRIOR.relative_to(ROOT)/name):hashlib.sha256((PRIOR/name).read_bytes()).hexdigest() for name in ('summary.json','states.npz','populations.npz')}
INPUTS['preserved_born_control_summary_sha256']=hashlib.sha256((ROOT/'results/encounter_born_control/summary.json').read_bytes()).hexdigest()
for name in ('src/sato_morrison/controls.py','src/sato_morrison/reference.py'):
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=prior['experiment_dependency_sha256'][name]:
        raise ValueError(f'Prior helper differs: {name}')


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)


def solve(spec):
    node, distance, step = spec
    started = perf_counter()
    row = dict(node=list(node), distance=distance, step=step)
    try:
        b, u, p, phase = node
        r = encounter_relative(b, phase, u, p, field=FIELD, strength=STRENGTH,
            screening=SCREENING, start_distance=distance, mass=MASS, charge=CHARGE,
            max_step=step, rtol=RTOL, flight_time_factor=FLIGHT_FACTOR)
        def gc(position, velocity):
            return position[:2]+np.array([velocity[1], -velocity[0]])/(CHARGE*FIELD/MASS)
        kick = gc(r['final_relative_position'], r['final_relative_velocity'])-gc(r['initial_relative_position'], r['initial_relative_velocity'])
        row.update(status='passed', moments=encounter_thermal_moments(r, THETA, MASS, FIELD).tolist(),
            individual_gc_second=float(kick@kick/4), gc_relative_kick=kick.tolist(),
            **{k:r[k].tolist() for k in ('initial_relative_position', 'final_relative_position', 'initial_relative_velocity', 'final_relative_velocity', 'delta_relative_perpendicular')},
            **{k:r[k] for k in ('time', 'exit', 'energy_error', 'min_separation', 'max_actual_step', 'evaluations', 'parallel_turns')})
        if not np.all(np.isfinite(row['moments'])) or not np.isfinite(kick@kick):
            raise RuntimeError('Nonfinite encounter observable')
    except Exception as error:
        row.update(status='unresolved', reason=str(error), error_type=type(error).__name__,
            diagnostics=getattr(error, 'integration_diagnostics', {}))
    row.update(wall_s=perf_counter()-started, worker_peak_rss_bytes=rss())
    return row


def save_states(path, rows):
    arrays = dict(nodes=np.asarray([r['node'] for r in rows]), distance=np.asarray([r['distance'] for r in rows]),
        step=np.asarray([r['step'] for r in rows]), status=np.asarray([r['status'] for r in rows]))
    for name, width in (('moments', 3), ('initial_relative_position', 3), ('final_relative_position', 3),
            ('initial_relative_velocity', 3), ('final_relative_velocity', 3),
            ('delta_relative_perpendicular', 2), ('gc_relative_kick', 2)):
        arrays[name] = np.asarray([r.get(name, [np.nan]*width) for r in rows])
    for name in ('time', 'energy_error', 'min_separation', 'max_actual_step', 'evaluations',
            'parallel_turns', 'individual_gc_second', 'wall_s', 'worker_peak_rss_bytes'):
        arrays[name] = np.asarray([r.get(name, np.nan) for r in rows])
    arrays['exit'] = np.asarray([r.get('exit', 'missing') for r in rows])
    np.savez_compressed(path, **arrays)


if REPLAY:
    if (OUTPUT/'summary.json').exists():
        raise FileExistsError('Do not overwrite an existing recovered receipt')
    original_attempt=json.loads((OUTPUT/'original_attempt.json').read_text())
    preserved_hashes=original_attempt['saved_artifact_sha256']
    original_manifest_hash=hashlib.sha256((OUTPUT/'original_attempt.json').read_bytes()).hexdigest()
    def verify_original_artifacts():
        absent=[]
        for name,h in preserved_hashes.items():
            path=OUTPUT/name
            if not path.exists():
                if name.startswith('checkpoint_') and name.endswith('.npz'):
                    absent.append(name)
                    continue
                raise FileNotFoundError(f'Required original artifact missing: {name}')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=h:
                raise ValueError(f'Original artifact changed: {name}')
        for name,h in ((original_attempt['original_source_copy'],original_attempt['source_copy_sha256']),
                (original_attempt['progress_log'],original_attempt['progress_log_sha256'])):
            if hashlib.sha256((OUTPUT/name).read_bytes()).hexdigest()!=h:
                raise ValueError(f'Original source or log changed: {name}')
        if hashlib.sha256((OUTPUT/'original_attempt.json').read_bytes()).hexdigest()!=original_manifest_hash:
            raise ValueError('Original artifact manifest changed')
        return dict(verified_manifest_artifact_count=len(preserved_hashes)-len(absent),
            absent_optional_checkpoint_count=len(absent),absent_optional_checkpoints=absent,
            verified_original_source_and_log_count=2)
    preservation_check=verify_original_artifacts()
else:
    if OUTPUT.exists():
        raise FileExistsError(f'Preserve existing evidence: {OUTPUT}; select a fresh destination')
    OUTPUT.mkdir(parents=True)
with np.load(PRIOR/'states.npz') as saved:
    base={name:saved[name].copy() for name in saved.files}
with np.load(PRIOR/'populations.npz') as saved:
    original_populations={name:saved[name].copy() for name in saved.files}
base_count=len(base['nodes'])
INPUTS['base_state_count']=base_count
if not np.all(base['status']=='passed') or not all(c['complete'] for c in prior['results']['cases']):
    raise ValueError('Base contains missing or failed physical states; do not discard their weights')
if REPLAY:
    if json.loads((OUTPUT/'declared_inputs.json').read_text())!=json.loads(json.dumps(INPUTS)):
        raise ValueError('Receipt replay changed the original declared scientific inputs')
else:
    (OUTPUT/'declared_inputs.json').write_text(json.dumps(INPUTS,indent=2)+'\n')
from jax._src import xla_bridge
if xla_bridge.backends_are_initialized():
    raise RuntimeError('Run in a fresh process before JAX backend initialization')
pool=None if REPLAY else ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('fork'))
if pool is not None:
    list(pool.map(float,range(WORKERS)))
import jax
jax.config.update('jax_enable_x64',True)
metadata=run_metadata(INPUTS,model='exact physical antithetic bounded encounter-flux extension',
    boundary=prior['boundary'],units=INPUTS['units'])
dependencies=[Path(__file__).resolve(),ROOT/'src/sato_morrison/controls.py',ROOT/'src/sato_morrison/reference.py']
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
metadata['experiment_dependency_sha256']=hashes
metadata['execution_mode']='saved-state receipt replay; no ODE solves' if REPLAY else 'fresh physical partner integration'
(OUTPUT/('recovery_run_context.json' if REPLAY else 'run_context.json')).write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
started=perf_counter()
attempts,cache,populations,cases,failures=[],{},{},[],[]
for i,node in enumerate(base['nodes']):
    cache[(tuple(node),float(base['distance'][i]),float(base['step'][i]))]=i
if REPLAY:
    with np.load(OUTPUT/'states.npz') as saved:
        for i in range(len(saved['nodes'])):
            record={('node' if k=='nodes' else k):saved[k][i].tolist() for k in saved.files}
            key=(tuple(record['node']),float(record['distance']),float(record['step']))
            if key in cache:
                raise ValueError('Saved new-state reference duplicates a base reference')
            cache[key]=base_count+len(attempts)
            attempts.append(record)
cap_exhausted=False
base_references_used=set()


def partner(nodes):
    result=np.asarray(nodes,dtype=float).copy()
    result[:,3]=(np.pi-result[:,3])%(2*np.pi)
    return result


def row(reference):
    if reference<0:
        return {'status':'not_run'}
    if reference>=base_count:
        return attempts[reference-base_count]
    return {('node' if k=='nodes' else k):v[reference].tolist() for k,v in base.items()}


def vector(record):
    return record['moments']+[record['individual_gc_second']]


def evaluate(nodes,distance=24.,step=BASE_STEP):
    global cap_exhausted
    specs=[(tuple(map(float,n)),float(distance),float(step)) for n in nodes]
    pending=list(dict.fromkeys(s for s in specs if s not in cache))
    if REPLAY and pending:
        raise RuntimeError('Receipt replay requires a missing solve; no ODE call is allowed')
    for offset in range(0,len(pending),CHECKPOINT):
        if perf_counter()-started>CAP:
            cap_exhausted=True
            break
        batch=pending[offset:offset+CHECKPOINT]
        records=list(pool.map(solve,batch))
        for spec,record in zip(batch,records):
            cache[spec]=base_count+len(attempts)
            attempts.append(record)
            if record['status']!='passed':
                failures.append(record)
                print('FAILED:',record,flush=True)
        save_states(OUTPUT/f'checkpoint_{len(attempts):06d}.npz',records)
        (OUTPUT/'failures.json').write_text(json.dumps(failures,indent=2,allow_nan=False)+'\n')
        print(f'Saved {len(attempts)} new states; wall {perf_counter()-started:.1f}s',flush=True)
    ids=np.asarray([cache.get(s,-1) for s in specs],dtype=int)
    base_references_used.update(int(i) for i in ids if 0<=i<base_count)
    return ids


def integrate(original):
    label=original['case']
    ids=original_populations[label]
    base_references_used.update(ids.tolist())
    weights=original_populations[label+'_weights']
    new_ids=evaluate(partner(base['nodes'][ids]),original['distance'])
    populations[label+'_original']=ids
    populations[label+'_partner']=new_ids
    populations[label+'_pair_weights']=weights
    populations[label+'_stratum']=original_populations[label+'_stratum']
    records=[row(i) for i in new_ids]
    complete=all(r['status']=='passed' for r in records)
    case={k:original[k] for k in ('case','seed','base_count','distance','node_count')}
    case.update(complete=complete,missing_count=sum(r['status']!='passed' for r in records),
        physical_member_count=2*len(ids),total_flux_weight=float(weights.sum()))
    if complete:
        old=np.column_stack((base['moments'][ids],base['individual_gc_second'][ids]))
        new=np.asarray([vector(r) for r in records])
        values=(old+new)/2
        integral=np.sum(weights[:,None]*values,axis=0)
        strata=populations[label+'_stratum']
        case.update(integrals=integral.tolist(),original_integrals=original['integrals'],
            max_energy_error=max(float(base['energy_error'][ids].max()),max(r['energy_error'] for r in records)),
            reflected_flux_fraction=float(weights@(np.asarray(base['exit'][ids]=='reflected',dtype=float)+np.asarray([r['exit']=='reflected' for r in records]))/(2*INPUTS['bounded_flux_per_density'])),
            strata=[dict(stratum=int(k),count=int(np.sum(strata==k)),flux_per_density=float(weights[strata==k].sum()),integrals=np.sum(weights[strata==k,None]*values[strata==k],axis=0).tolist()) for k in np.unique(strata)])
    cases.append(case)
    (OUTPUT/('reconstructed_cases.json' if REPLAY else 'cases_partial.json')).write_text(json.dumps(cases,indent=2)+'\n')
    print(label,case.get('integrals'),flush=True)
    return case


def change(a,b):
    if not a['complete'] or not b['complete']:
        return None
    x,y=np.asarray(a['integrals']),np.asarray(b['integrals'])
    scale=np.array([x[1]/MU_REF,x[1],x[1],x[3]])
    return (abs(x-y)/scale).tolist()


transform_checks=[]
for original in prior['results']['cases']:
    label=original['case']
    ids=original_populations[label]
    old_nodes=base['nodes'][ids]
    new_nodes=partner(old_nodes)
    definitions=INPUTS['alternate_strata'] if original['seed']==INPUTS['alternate_seed'] else INPUTS['main_strata']
    cell=original_populations[label+'_stratum']
    tol=INPUTS['phase_transform_tolerance']
    membership=all(np.all((new_nodes[cell==d['index'],3]-d['phase'][0])%(2*np.pi)<=d['phase'][1]-d['phase'][0]+tol) for d in definitions)
    inverse_error=float(np.max(abs((partner(new_nodes)[:,3]-old_nodes[:,3]+np.pi)%(2*np.pi)-np.pi)))
    transform_checks.append(dict(case=label,same_cell=membership,involution_max_phase_error=inverse_error))
    integrate(original)
by_label={c['case']:c for c in cases}
seeds,levels=INPUTS['seeds'],INPUTS['base_levels']
checks={f'sample_{s}':change(by_label[f'seed{s}_N{levels[-1]}'],by_label[f'seed{s}_N{levels[-2]}']) for s in seeds}
for a,b in combinations(seeds,2):
    checks[f'replica_{a}_{b}']=change(by_label[f'seed{a}_N{levels[-1]}'],by_label[f'seed{b}_N{levels[-1]}'])
for seed in seeds:
    checks[f'partition_{seed}']=change(by_label[f'seed{seed}_N{levels[-1]}'],by_label['alternate_partition'])
checks['boundary_ensemble']=change(by_label['boundary_L48'],by_label['boundary_L40'])
checks['boundary_L24_to40']=change(by_label['boundary_L40'],by_label[f'seed{seeds[0]}_N{levels[-1]}'])
control_nodes=[]
for seed in (*seeds,INPUTS['alternate_seed']):
    label='alternate_partition' if seed==INPUTS['alternate_seed'] else f'seed{seed}_N{levels[-1]}'
    if by_label[label]['complete']:
        original_ids=populations[label+'_original']
        partner_ids=populations[label+'_partner']
        second=np.asarray([(base['moments'][a,1]+row(b)['moments'][1])/2 for a,b in zip(original_ids,partner_ids)])
        top=np.argsort(populations[label+'_pair_weights']*second)[-4:]
        control_nodes.extend(base['nodes'][original_ids[top]].tolist())
fine_labels=[f'seed{seed}_N{levels[-1]}' for seed in seeds]+['alternate_partition']
partner_to_original={int(b):int(a) for label in fine_labels for a,b in zip(populations[label+'_original'],populations[label+'_partner']) if b>=0}
valid=[i for i in partner_to_original if row(i)['status']=='passed']
closest=sorted(valid,key=lambda i:row(i)['min_separation'])[:4]
reflected=sorted((i for i in valid if row(i)['exit']=='reflected'),key=lambda i:row(i)['time'])[-4:]
control_nodes.extend(base['nodes'][[partner_to_original[i] for i in closest+reflected]].tolist())
control_nodes.extend(prior['results']['controls']['nodes'][-17:])
control_nodes=np.asarray(list(dict.fromkeys(tuple(n) for n in control_nodes)))
controls=dict(nodes=control_nodes.tolist(),ids={})


def control_vectors(distance=24.,step=BASE_STEP):
    original_ids=evaluate(control_nodes,distance,step)
    partner_ids=evaluate(partner(control_nodes),distance,step)
    values=np.asarray([(np.asarray(vector(row(a)))+np.asarray(vector(row(b))))/2 if row(a)['status']==row(b)['status']=='passed' else [np.nan]*4 for a,b in zip(original_ids,partner_ids)])
    return values,dict(original=original_ids.tolist(),partner=partner_ids.tolist())


def max_changes(a,b):
    scale=np.column_stack((a[:,1]/MU_REF,a[:,1],a[:,1],a[:,3]))
    return np.max(abs(a-b)/scale,axis=0).tolist() if np.all(np.isfinite(a+b)) and np.all(scale>0) else None


step_vectors=[]
for step in STEPS:
    values,ids=control_vectors(step=step)
    controls['ids'][str(step)]=ids
    step_vectors.append(values)
checks['active_step']=max_changes(step_vectors[-1],step_vectors[-2])
def member_vectors(ids):
    return np.asarray([vector(row(i)) if row(i)['status']=='passed' else [np.nan]*4 for member in ('original','partner') for i in ids[member]])
checks['active_step_members']=max_changes(member_vectors(controls['ids']['0.025']),member_vectors(controls['ids']['0.05']))
fine_ids,coarse_ids=controls['ids']['0.025'],controls['ids']['0.05']
controls['actual_step_ratios']=[row(i)['max_actual_step']/row(j)['max_actual_step'] if i>=0 and j>=0 and row(i)['status']==row(j)['status']=='passed' else None for member in ('original','partner') for i,j in zip(fine_ids[member],coarse_ids[member])]
plane_vectors=[]
for distance in DISTANCES:
    values,ids=control_vectors(distance=distance)
    controls['ids'][f'L{distance:g}']=ids
    plane_vectors.append(values)
checks['boundary_controls']=max_changes(plane_vectors[-1],plane_vectors[-2])
checks['boundary_controls_members']=max_changes(member_vectors(controls['ids']['L48']),member_vectors(controls['ids']['L40']))
if pool is not None:
    pool.shutdown()


def gate(values,target):
    return values is not None and np.all(np.isfinite(values)) and max(values)<=target


gates={k:gate(v,PLANE_TARGET if k.startswith('boundary_') else STEP_TARGET if k.startswith('active_step') else QUADRATURE_TARGET) for k,v in checks.items()}
gates.update(phase_transform=all(c['same_cell'] and c['involution_max_phase_error']<=INPUTS['phase_transform_tolerance'] for c in transform_checks),
    preserved_born_control=hashlib.sha256((ROOT/'results/encounter_born_control/summary.json').read_bytes()).hexdigest()==INPUTS['preserved_born_control_summary_sha256'],
    complete=all(c['complete'] for c in cases) and not failures and not cap_exhausted,
    exact_flux_mass=all(np.isclose(c['total_flux_weight'],INPUTS['bounded_flux_per_density'],rtol=5e-13,atol=0) for c in cases),
    energy=bool(attempts) and all(r['status']=='passed' and r['energy_error']<=ENERGY_TARGET for r in attempts) and np.all(base['energy_error']<=ENERGY_TARGET),
    active_step_sizes=bool(controls['actual_step_ratios']) and all(r is not None and r<=.6 for r in controls['actual_step_ratios']),
    prior_integrity=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==wanted for name,wanted in INPUTS['prior_artifact_sha256'].items()),
    prior_prior_integrity=prior['results']['required_gates']['prior_integrity'],
    prior_source_integrity=prior['results']['required_gates']['source_integrity'])
ending={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
metadata['experiment_dependency_sha256_end']=ending
gates['source_integrity']=hashes==ending
gates={k:bool(v) for k,v in gates.items()}
status='passed' if all(gates.values()) else 'unresolved'
if REPLAY:
    with np.load(OUTPUT/'populations.npz') as saved:
        if set(saved.files)!=set(populations) or not all(np.array_equal(saved[k],populations[k]) for k in saved.files):
            raise ValueError('Reconstructed population references or weights differ')
    preservation_check=verify_original_artifacts()
else:
    save_states(OUTPUT/'states.npz',attempts)
    np.savez_compressed(OUTPUT/'populations.npz',**populations)
metadata['results']=dict(status=status,required_gates=gates,comparisons=checks,cases=cases,controls=controls,transform_checks=transform_checks,
    raw_campaign_status=prior['results']['status'],raw_campaign_required_gates=prior['results']['required_gates'],
    raw_campaign_comparisons=prior['results']['comparisons'],base_state_count=base_count,
    new_encounters=len(attempts),reused_encounters=len(base_references_used),unique_accessible_encounters=base_count+len(attempts),
    failures=len(failures),cap_exhausted=cap_exhausted,wall_s=perf_counter()-started,
    parent_process_peak_rss_bytes=rss(),max_worker_process_peak_rss_bytes=max((r['worker_peak_rss_bytes'] for r in attempts),default=0),
    memory_scope='separate lifetime process peaks; not simultaneous aggregate',
    timing_scope='new physical partner and refined control solves; old solves and figure rendering excluded',
    limits=['Pairing preserves the exact incoming measure but provides no rigorous error bound or QMC variance guarantee.',
        'Original raw and first-Born control failures remain intact; this is a distinct estimator using new physical trajectories.',
        'No omitted-domain, reverse-complete population, screening closure, physical coefficient or lifetime claim.'])
if REPLAY:
    result=metadata['results']
    result.update(recovery_wall_s=result['wall_s'],recovery_parent_process_peak_rss_bytes=result['parent_process_peak_rss_bytes'],
        wall_s=None,parent_process_peak_rss_bytes=None,new_solves_this_execution=0,
        original_last_logged_elapsed_s=original_attempt['last_logged_elapsed_s'],
        timing_scope='Original exact wall endpoint unavailable; last logged elapsed is a rounded lower bound. Recovery wall is separately measured.',
        memory_scope='Original parent peak unavailable; original worker peaks persisted per state. Recovery parent peak is separately measured.')
    metadata['original_producer']=original_attempt
    metadata['receipt_scope']='Numerical gates reconstructed from immutable saved arrays; top-level code/commit/environment identify receipt recovery, not the original ODE execution.'
    metadata['recovery_preserved_artifact_sha256']=preserved_hashes
    metadata['original_manifest_sha256']=original_manifest_hash
    metadata['recovery_preservation_check']=preservation_check
(OUTPUT/'summary.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
for seed in seeds:
    cc=[c for c in cases if c['seed']==seed and c['distance']==24. and c['complete']]
    axes[0].plot([c['physical_member_count'] for c in cc],[c['integrals'][0] for c in cc],'o-',label=f'seed {seed}')
axes[0].set(xlabel='physical encounter members including reused originals',ylabel='signed mean integral per density (normalized)',title='Exact physical antithetic pairs')
groups=[('sample',[k for k in gates if k.startswith('sample_')],QUADRATURE_TARGET),('replicas',[k for k in gates if k.startswith('replica_')],QUADRATURE_TARGET),('partition',[k for k in gates if k.startswith('partition_')],QUADRATURE_TARGET),('L24 to40',['boundary_L24_to40'],PLANE_TARGET),('L40 to48',['boundary_ensemble'],PLANE_TARGET),('selected planes',['boundary_controls','boundary_controls_members'],PLANE_TARGET),('active step',['active_step','active_step_members'],STEP_TARGET)]
for column,name in enumerate(('signed mean','second','cross','spatial second')):
    ratios=[max(checks[k][column] for k in keys)/target if all(checks[k] is not None for k in keys) else np.nan for _,keys,target in groups]
    axes[1].plot(range(len(groups)),ratios,'o-',ms=4,label=name)
axes[1].axhline(1,color='gray',ls=':')
axes[1].set_xticks(range(len(groups)),[g[0] for g in groups],rotation=35,ha='right',fontsize=8)
axes[1].set(ylabel='worst change in group / fixed target',title=status)
for ax in axes:
    ax.legend(fontsize=8)
fig.savefig(OUTPUT/'convergence.png',dpi=160)
plt.close(fig)
with np.load(OUTPUT/'states.npz') as saved:
    if len(saved['nodes'])!=len(attempts):
        raise RuntimeError('Final new-state archive incomplete; retain checkpoints')
if not REPLAY:
    for checkpoint in OUTPUT.glob('checkpoint_*.npz'):
        checkpoint.unlink()
print(f'Finished {status}; gates={gates}; elapsed {perf_counter()-started:.1f}s',flush=True)
sys.exit(0 if status=='passed' else 1)
