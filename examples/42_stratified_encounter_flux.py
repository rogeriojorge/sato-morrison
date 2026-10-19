"""Fresh stratified four-dimensional bounded encounter-flux quadrature.

Exact rectangular flux masses concentrate work near the free-orbit tangent
corner. Every stratum retains positive coverage and all reflected exits count.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from itertools import product, combinations
from time import perf_counter
import hashlib
import json
import multiprocessing as mp
import os
import resource
import sys
import numpy as np
from scipy.stats import qmc
from sato_morrison.controls import (encounter_relative, encounter_thermal_moments,
    encounter_flux_samples, encounter_bounded_flux)
from sato_morrison.reference import run_metadata

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT/'results/broad_speed_extended'
PEAK = ROOT/'results/encounter_peak_resolution'
OUTPUT = ROOT/os.environ.get('SM_STRATIFIED_FLUX_OUTPUT', 'results/stratified_encounter_flux')
MASS, CHARGE, FIELD, STRENGTH, SCREENING, THETA = 1., 1., 2., .03, 3., .5
BOUNDS = ((1.2,2.4),(.35,3.),(0.,2.5))
IMPACT_BINS, PERP_BINS = ((1.2,1.5),(1.5,2.4)), ((0.,2.),(2.,2.5))
LEVELS, SEEDS, ALTERNATE_SEED = (512,1024,2048), (4201,4202,4203,4204), 4205
PHASE_HALF_WIDTH, ALTERNATE_HALF_WIDTH = .25, .15
RISK_MULTIPLIER = 8
DISTANCES, STEPS = (40.,48.), (.1,.05,.025)
RTOL, FLIGHT_FACTOR, BASE_STEP = 1e-12, 48., .2
WORKERS, CHECKPOINT, CAP = 4, 512, 3000.
QUADRATURE_TARGET, PLANE_TARGET, STEP_TARGET, ENERGY_TARGET = .05, .005, .001, 1e-9
mu_ref = MASS*THETA/FIELD
flux = encounter_bounded_flux(*BOUNDS,THETA)
INPUTS = dict(mass=MASS,charge=CHARGE,field=FIELD,strength=STRENGTH,screening=SCREENING,
    theta=THETA,bounds=BOUNDS,impact_bins=IMPACT_BINS,perpendicular_bins=PERP_BINS,
    base_levels=LEVELS,seeds=SEEDS,alternate_seed=ALTERNATE_SEED,
    phase_half_width=PHASE_HALF_WIDTH,alternate_half_width=ALTERNATE_HALF_WIDTH,
    risk_multiplier=RISK_MULTIPLIER,risk_stratum='low impact, high perpendicular speed, central phase interval',
    phase_intervals='central [pi/2-h,pi/2+h]; complement [pi/2+h,pi/2-h+2pi] modulo2pi',
    stratum_seed_rule='100*replica_seed+stratum_index; independent scrambled Sobol4 engines',
    distances=DISTANCES,steps=STEPS,rtol=RTOL,flight_time_factor=FLIGHT_FACTOR,
    base_step=BASE_STEP,workers=WORKERS,runtime_cap_s=CAP,
    quadrature_target=QUADRATURE_TARGET,plane_target=PLANE_TARGET,step_target=STEP_TARGET,
    energy_target=ENERGY_TARGET,units='normalized; theta=kBT/m; U=kappa exp(-r/lambda)/r',
    measure='two-sided bounded tagged plane-crossing flux; exact stratum flux times uniform phase fraction; each node has stratum_flux/Nstratum; reflected exits retain weights',
    design='rectangular bins and phase window fixed from archived example39/41 diagnosis before fresh independent samples; no result-dependent allocations',
    normalization='mean: fine second/mu_reference; second and cross: fine second; spatial: fine individual GC second',
    controls='top four weighted M2 nodes from each fresh replica; closest four separations and longest four reflected exits from primary; sixteen fresh predetermined flux/phase nodes; historical weak signed-mean boundary node',
    acceptance='each replica final nested sample comparison, all six replica pairs, alternate phase partition versus every main replica, full matched primary L24/L40/L48 boundary, selected boundary and active-step/energy/integrity gates',
    stop_rule='fixed levels, seeds, allocations, boundary and controls; cap between batches; missing states invalidate integrals; no failed-weight renormalization',
    scope='one finite bounded incident domain; empirical quadrature checks only; no physical screening, reverse-complete population, collision closure or lifetime')
prior=json.loads((PRIOR/'summary.json').read_text())
for name in ('mass','charge','field','strength','screening','theta','rtol','flight_time_factor'):
    if prior['inputs'][name]!=INPUTS[name]:
        raise ValueError(f'Prior input differs: {name}')
INPUTS['prior_artifact_sha256']={str(folder.relative_to(ROOT)/name):hashlib.sha256((folder/name).read_bytes()).hexdigest() for folder in (PRIOR,PEAK) for name in ('summary.json','states.npz')}
for name in ('src/sato_morrison/controls.py','src/sato_morrison/reference.py'):
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=prior['experiment_dependency_sha256'][name]:
        raise ValueError(f'Prior helper differs: {name}')


def strata(half_width):
    phase_intervals=((np.pi/2-half_width,np.pi/2+half_width),
        (np.pi/2+half_width,np.pi/2-half_width+2*np.pi))
    rows=[]
    for index,(ib,ip,ia) in enumerate(product(range(2),repeat=3)):
        bb,pb,ab=IMPACT_BINS[ib],PERP_BINS[ip],phase_intervals[ia]
        weight=encounter_bounded_flux(bb,BOUNDS[1],pb,THETA)*(ab[1]-ab[0])/(2*np.pi)
        rows.append(dict(index=index,impact=bb,parallel=BOUNDS[1],perpendicular=pb,
            phase=ab,flux_per_density=weight,multiplier=RISK_MULTIPLIER if (ib,ip,ia)==(0,1,0) else 1))
    if not np.isclose(sum(s['flux_per_density'] for s in rows),flux,rtol=2e-15,atol=0):
        raise ValueError('Stratum masses do not cover the original bounded flux')
    return rows


INPUTS['main_strata']=strata(PHASE_HALF_WIDTH)
INPUTS['alternate_strata']=strata(ALTERNATE_HALF_WIDTH)
INPUTS['bounded_flux_per_density']=flux
INPUTS['fine_paths_per_replica']=sum(LEVELS[-1]*s['multiplier'] for s in INPUTS['main_strata'])
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



if OUTPUT.exists():
    raise FileExistsError(f'Preserve existing evidence: {OUTPUT}; choose a fresh destination')
OUTPUT.mkdir(parents=True)
(OUTPUT/'declared_inputs.json').write_text(json.dumps(INPUTS,indent=2)+'\n')
from jax._src import xla_bridge
if xla_bridge.backends_are_initialized():
    raise RuntimeError('Run in a fresh process before JAX backend initialization')
pool=ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('fork'))
list(pool.map(float,range(WORKERS)))
import jax
jax.config.update('jax_enable_x64',True)
metadata=run_metadata(INPUTS,model='fresh stratified four-dimensional bounded encounter flux',
    boundary='first outgoing relative z=+/-L plane; reflected retained',units=INPUTS['units'])
dependencies=[Path(__file__).resolve(),ROOT/'src/sato_morrison/controls.py',ROOT/'src/sato_morrison/reference.py']
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
metadata['experiment_dependency_sha256']=hashes
started=perf_counter()
attempts,cache,populations,cases,failures=[],{},{},[],[]
cap_exhausted=False


def evaluate(nodes,distance=24.,step=BASE_STEP):
    global cap_exhausted
    specs=[(tuple(map(float,n)),float(distance),float(step)) for n in nodes]
    pending=list(dict.fromkeys(s for s in specs if s not in cache))
    for offset in range(0,len(pending),CHECKPOINT):
        if perf_counter()-started>CAP:
            cap_exhausted=True
            break
        batch=pending[offset:offset+CHECKPOINT]
        rows=list(pool.map(solve,batch))
        for spec,row in zip(batch,rows):
            cache[spec]=len(attempts)
            attempts.append(row)
            if row['status']!='passed':
                failures.append(row)
                print('FAILED:',row,flush=True)
        save_states(OUTPUT/f'checkpoint_{len(attempts):06d}.npz',rows)
        (OUTPUT/'failures.json').write_text(json.dumps(failures,indent=2,allow_nan=False)+'\n')
        print(f'Saved {len(attempts)} states; wall {perf_counter()-started:.1f}s',flush=True)
    ids=np.asarray([cache.get(s,-1) for s in specs],dtype=int)
    return [attempts[i] if i>=0 else {'status':'not_run'} for i in ids],ids


def vector(row):
    return row['moments']+[row['individual_gc_second']]


sample_sets={}
for seed in (*SEEDS,ALTERNATE_SEED):
    definitions=INPUTS['alternate_strata'] if seed==ALTERNATE_SEED else INPUTS['main_strata']
    sample_sets[seed]=[]
    for s in definitions:
        n=LEVELS[-1]*s['multiplier']
        cube=qmc.Sobol(4,scramble=True,seed=100*seed+s['index']).random_base2(int(np.log2(n)))
        speeds=encounter_flux_samples(cube[:,:3],s['impact'],s['parallel'],s['perpendicular'],THETA)
        phase=(s['phase'][0]+(s['phase'][1]-s['phase'][0])*cube[:,3])%(2*np.pi)
        sample_sets[seed].append(np.column_stack((speeds,phase)))


def integrate(label,seed,level=LEVELS[-1],distance=24.):
    definitions=INPUTS['alternate_strata'] if seed==ALTERNATE_SEED else INPUTS['main_strata']
    components=[]
    all_ids=[]
    all_weights=[]
    strata_ids=[]
    complete=True
    for s,nodes in zip(definitions,sample_sets[seed]):
        n=level*s['multiplier']
        rows,ids=evaluate(nodes[:n],distance)
        good=all(r['status']=='passed' for r in rows)
        complete=complete and good
        all_ids.extend(ids.tolist())
        all_weights.extend([s['flux_per_density']/n]*n)
        strata_ids.extend([s['index']]*n)
        components.append(dict(stratum=s['index'],count=n,flux_per_density=s['flux_per_density'],complete=good,
            integrals=(s['flux_per_density']*np.asarray([vector(r) for r in rows]).mean(0)).tolist() if good else None))
    populations[label]=np.asarray(all_ids,dtype=int)
    populations[label+'_weights']=np.asarray(all_weights)
    populations[label+'_stratum']=np.asarray(strata_ids,dtype=int)
    case=dict(case=label,seed=seed,base_count=level,distance=distance,complete=complete,
        missing_count=sum(i<0 or attempts[i]['status']!='passed' for i in all_ids),
        node_count=len(all_ids),total_flux_weight=float(sum(all_weights)),strata=components)
    if complete:
        values=np.asarray([vector(attempts[i]) for i in all_ids])
        weights=np.asarray(all_weights)
        integral=np.sum(weights[:,None]*values,axis=0)
        case.update(integrals=integral.tolist(),drift_per_density=float(integral[0]),
            diffusion_per_density=float(integral[1]/2),cross_per_density=float(integral[2]),
            spatial_second_per_density=float(integral[3]),
            max_energy_error=max(attempts[i]['energy_error'] for i in all_ids),
            reflected_flux_fraction=float(np.sum(weights*np.asarray([attempts[i]['exit']=='reflected' for i in all_ids]))/flux))
    cases.append(case)
    (OUTPUT/'cases_partial.json').write_text(json.dumps(cases,indent=2)+'\n')
    print(label,case.get('integrals'),flush=True)
    return case


def changes(a,b,reference=None):
    if not a['complete'] or not b['complete'] or (reference is not None and not reference['complete']):
        return None
    x,y=np.asarray(a['integrals']),np.asarray(b['integrals'])
    r=np.asarray((reference or a)['integrals'])
    scales=np.array([r[1]/mu_ref,r[1],r[1],r[3]])
    return (abs(x-y)/scales).tolist() if np.all(scales>0) else None


checks={}
fine_cases=[]
history={}
for seed in SEEDS:
    levels=[integrate(f'seed{seed}_N{n}',seed,n) for n in LEVELS]
    fine_cases.append(levels[-1])
    checks[f'sample_{seed}']=changes(levels[-1],levels[-2])
    history[f'sample_{seed}_early']=changes(levels[-2],levels[-3],levels[-1])
for i,j in combinations(range(len(SEEDS)),2):
    checks[f'replica_{SEEDS[i]}_{SEEDS[j]}']=changes(fine_cases[i],fine_cases[j])
alternate=integrate('alternate_partition',ALTERNATE_SEED)
for seed,case in zip(SEEDS,fine_cases):
    checks[f'partition_{seed}']=changes(case,alternate)
boundary=[integrate(f'boundary_L{d:g}',SEEDS[0],distance=d) for d in DISTANCES]
checks['boundary_ensemble']=changes(boundary[-1],boundary[-2])
# Paired boundary differences remove sample changes from this endpoint test.
checks['boundary_L24_to40']=changes(boundary[0],fine_cases[0])

control_nodes=[]
for seed,case in zip(SEEDS,fine_cases):
    if case['complete']:
        ids=populations[f'seed{seed}_N{LEVELS[-1]}']
        weights=populations[f'seed{seed}_N{LEVELS[-1]}_weights']
        top=np.argsort(weights*np.asarray([attempts[i]['moments'][1] for i in ids]))[-4:]
        control_nodes.extend([attempts[ids[j]]['node'] for j in top])
if fine_cases[0]['complete']:
    ids=populations[f'seed{SEEDS[0]}_N{LEVELS[-1]}']
    closest=sorted(ids,key=lambda i:attempts[i]['min_separation'])[:4]
    reflected=sorted((i for i in ids if attempts[i]['exit']=='reflected'),key=lambda i:attempts[i]['time'])[-4:]
    control_nodes.extend([attempts[i]['node'] for i in closest+reflected])
fresh=encounter_flux_samples(qmc.Sobol(3,scramble=True,seed=4210).random_base2(4),*BOUNDS,THETA)
fresh_phase=2*np.pi*qmc.Sobol(1,scramble=True,seed=4211).random_base2(4).ravel()
control_nodes.extend(np.column_stack((fresh,fresh_phase)).tolist())
control_nodes.append(prior['results']['controls']['nodes'][25])
control_nodes=list(dict.fromkeys(tuple(n) for n in control_nodes))
controls=dict(nodes=control_nodes,ids={})
step_vectors=[]
for step in STEPS:
    rows,ids=evaluate(control_nodes,step=step)
    controls['ids'][str(step)]=ids.tolist()
    step_vectors.append(np.asarray([vector(r) if r['status']=='passed' else [np.nan]*4 for r in rows]))
a,b=step_vectors[-1],step_vectors[-2]
scale=np.column_stack((a[:,1]/mu_ref,a[:,1],a[:,1],a[:,3]))
checks['active_step']=np.max(abs(a-b)/scale,axis=0).tolist() if np.all(np.isfinite(a+b)) and np.all(scale>0) else None
controls['actual_step_ratios']=[attempts[i]['max_actual_step']/attempts[j]['max_actual_step'] if i>=0 and j>=0 and attempts[i]['status']=='passed' and attempts[j]['status']=='passed' else None for i,j in zip(controls['ids'][str(STEPS[-1])],controls['ids'][str(STEPS[-2])])]
plane_vectors=[]
for distance in DISTANCES:
    rows,ids=evaluate(control_nodes,distance=distance)
    controls['ids'][f'L{distance:g}']=ids.tolist()
    plane_vectors.append(np.asarray([vector(r) if r['status']=='passed' else [np.nan]*4 for r in rows]))
a,b=plane_vectors[-1],plane_vectors[-2]
scale=np.column_stack((a[:,1]/mu_ref,a[:,1],a[:,1],a[:,3]))
checks['boundary_controls']=np.max(abs(a-b)/scale,axis=0).tolist() if np.all(np.isfinite(a+b)) and np.all(scale>0) else None
pool.shutdown()
ending={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
metadata['experiment_dependency_sha256_end']=ending


def gate(values,target):
    return values is not None and np.all(np.isfinite(values)) and max(values)<=target


gates={k:gate(v,PLANE_TARGET if k.startswith('boundary_') else STEP_TARGET if k=='active_step' else QUADRATURE_TARGET) for k,v in checks.items() }
gates.update(complete=all(c['complete'] for c in cases) and not failures and not cap_exhausted,
    exact_flux_mass=all(np.isclose(c['total_flux_weight'],flux,rtol=5e-13,atol=0) for c in cases),
    energy=bool(attempts) and all(r['status']=='passed' and r['energy_error']<=ENERGY_TARGET for r in attempts),
    active_step_sizes=bool(controls['actual_step_ratios']) and all(r is not None and r<=.6 for r in controls['actual_step_ratios']),
    source_integrity=hashes==ending,
    prior_integrity=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==wanted for name,wanted in INPUTS['prior_artifact_sha256'].items()))
status='passed' if all(gates.values()) else 'unresolved'
save_states(OUTPUT/'states.npz',attempts)
np.savez_compressed(OUTPUT/'populations.npz',**populations)
metadata['results']=dict(status=status,required_gates=gates,comparisons=checks,history_diagnostics=history,
    cases=cases,controls=controls,new_encounters=len(attempts),unique_encounters=len(attempts),
    reused_encounters=0,failures=len(failures),cap_exhausted=cap_exhausted,wall_s=perf_counter()-started,
    parent_process_peak_rss_bytes=rss(),max_worker_process_peak_rss_bytes=max((r['worker_peak_rss_bytes'] for r in attempts),default=0),
    memory_scope='separate process lifetime peaks; not simultaneous aggregate',
    timing_scope='this execution; includes all fresh solves and excludes figure rendering',
    limits=['Nested Sobol, scramble and partition differences are empirical; no confidence interval or rigorous quadrature error bound.',
        'Matched full boundary ensemble uses first fresh scramble; other scrambles are evaluated at L24.',
        'Original example39 and41 evidence remains intact; this is a fresh estimator.',
        'No domain-tail, reverse-complete population, screening closure, correlation or lifetime claim.'])
(OUTPUT/'summary.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
if all(c['complete'] for c in fine_cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for seed in SEEDS:
        cc=[c for c in cases if c['seed']==seed and c['distance']==24.]
        axes[0,0].plot([c['node_count'] for c in cc],[c['integrals'][1] for c in cc],'o-',label=f'seed {seed}')
        axes[0,1].plot([c['node_count'] for c in cc],[c['integrals'][0] for c in cc],'o-',label=f'seed {seed}')
    axes[0,0].set(xlabel='fresh four-dimensional nodes',ylabel='bounded second integral per density (normalized)')
    axes[0,1].set(xlabel='fresh four-dimensional nodes',ylabel='signed mean integral per density (normalized)')
    flux_fraction=np.asarray([s['flux_per_density'] for s in INPUTS['main_strata']])/flux
    axes[1,0].bar(np.arange(8),flux_fraction,color='lightgray',label='exact incoming flux fraction')
    for seed in SEEDS:
        cc=next(c for c in fine_cases if c['seed']==seed)
        axes[1,0].plot(range(8),np.asarray([s['integrals'][1] for s in cc['strata']])/cc['integrals'][1],'o',ms=4,label=f'second fraction, {seed}')
    labels=('LLC','LLW','LHC','LHW','HLC','HLW','HHC','HHW')
    axes[1,0].set_xticks(range(8),labels)
    axes[1,0].set(xlabel='impact L/H; perpendicular speed L/H; phase C=central/W=wrapped',ylabel='fraction of incoming flux or second integral',yscale='log')
    groups=[('sample',[k for k in gates if k.startswith('sample_')],QUADRATURE_TARGET),
        ('replicas',[k for k in gates if k.startswith('replica_')],QUADRATURE_TARGET),
        ('partition',[k for k in gates if k.startswith('partition_')],QUADRATURE_TARGET),
        ('L24 to40',['boundary_L24_to40'],PLANE_TARGET),
        ('L40 to48',['boundary_ensemble'],PLANE_TARGET),
        ('selected planes',['boundary_controls'],PLANE_TARGET),
        ('active step',['active_step'],STEP_TARGET)]
    for column,name in enumerate(('signed mean','second','cross','spatial second')):
        ratios=[max(checks[k][column] for k in keys)/target if all(checks[k] is not None for k in keys) else np.nan for _,keys,target in groups]
        axes[1,1].plot(range(len(groups)),ratios,'o-',ms=4,label=name)
    axes[1,1].axhline(1,color='gray',ls=':')
    axes[1,1].set_xticks(range(len(groups)),[g[0] for g in groups],rotation=35,ha='right',fontsize=8)
    axes[1,1].set(ylabel='worst change in group / fixed target',title=status)
    for ax in axes.ravel():
        ax.legend(fontsize=7)
    fig.savefig(OUTPUT/'convergence.png',dpi=160)
    plt.close(fig)
with np.load(OUTPUT/'states.npz') as saved:
    if len(saved['nodes'])!=len(attempts):
        raise RuntimeError('Final archive incomplete; keep checkpoints')
for checkpoint in OUTPUT.glob('checkpoint_*.npz'):
    checkpoint.unlink()
print(f'Finished {status}; gates={gates}; elapsed {perf_counter()-started:.1f}s',flush=True)
sys.exit(0 if status=='passed' else 1)
