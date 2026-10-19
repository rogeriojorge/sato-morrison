"""Known-zero first-Born control for a saved bounded encounter ensemble.

The physical trajectories and nonlinear moments remain unchanged. This tests
subtracting a term with a known zero phase average to reduce estimator noise.
"""
from pathlib import Path
from itertools import combinations
from time import perf_counter
import hashlib
import json
import os
import resource
import sys
import numpy as np
from scipy.special import roots_legendre
from sato_morrison.reference import run_metadata

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT/'results/stratified_encounter_flux'
PILOT = ROOT/'results/encounter_born_control_pilot'
OUTPUT = ROOT/os.environ.get('SM_BORN_CONTROL_OUTPUT','results/encounter_born_control')
MASS, CHARGE, FIELD, STRENGTH, SCREENING, THETA = 1., 1., 2., .03, 3., .5
COEFFICIENT, MIN_GAP, CONTROL_DISTANCE = 1., .3, 24.
ORDERS, BATCH = (1024,2048), 256
QUADRATURE_TARGET, PLANE_TARGET, STEP_TARGET, STABILITY_TARGET = .05,.005,.001,.01
SYMMETRY_TARGET, CAP = 1e-12, 600.
OMEGA, MU_REF = CHARGE*FIELD/MASS, MASS*THETA/FIELD
INPUTS = dict(mass=MASS,charge=CHARGE,field=FIELD,strength=STRENGTH,screening=SCREENING,
    theta=THETA,coefficient=COEFFICIENT,min_free_orbit_gap=MIN_GAP,
    control_distance=CONTROL_DISTANCE,quadrature_orders=ORDERS,batch_size=BATCH,
    quadrature_target=QUADRATURE_TARGET,plane_target=PLANE_TARGET,
    step_target=STEP_TARGET,control_stability_target=STABILITY_TARGET,
    symmetry_absolute_target=SYMMETRY_TARGET,runtime_cap_s=CAP,
    units='normalized; magnetic moment m*v^2/B; theta=kBT/m; U=kappa exp(-r/lambda)/r',
    formula='c_B=kappa*b*p/(2*B*u) integral[-Lcontrol,Lcontrol] cos(phi-Omega*z/u)*exp(-r/lambda)*(1+r/lambda)/r^3 dz; r^2=b^2+(p/Omega)^2-2*b*(p/Omega)*sin(phi-Omega*z/u)+z^2',
    zero_expectation_proof='For fixed b,u,p, transform phi to pi-phi modulo2pi and z to -z. Radius and phase sine are unchanged while cosine changes sign. Symmetric Gauss-Legendre nodes preserve this oddness exactly apart from floating-point error. Both central phase interval around pi/2 and its wrapped complement are invariant under the phase involution, so each stratum control expectation is exactly zero. The phase-independent gap mask preserves zero expectation.',
    estimator='signed mean=sum physical incoming weights*(nonlinear mean-c_B); second, cross and spatial moments unchanged',
    mask_scope='g=b-p/Omega>=.3 uses the fixed first-Born control; all other physical states keep original nonlinear mean and their original probability',
    design='coefficient one, gap .3, control L24 and order1024 frozen from historical-only pilot; order2048 is fixed estimator sensitivity, not outcome-dependent tuning',
    stability_scope='Both finite Gauss-Legendre controls have exactly zero phase expectation; their difference measures empirical estimator sensitivity, not physical quadrature bias.',
    boundary_scope='Identical control with Lcontrol24 used for every trajectory endpoint plane; paired endpoint differences are unchanged.',
    variance_scope='Within-cell variance ratios and four-replica spreads are descriptive empirical comparisons, not confidence intervals.',
    acceptance='recompute all original fixed full-domain sample, replica, partition, boundary, trajectory and integrity gates with controlled signed mean; all case order1024-to2048 scaled mean changes <=1%; preserve raw original failures',
    stop_rule='fixed saved states, controls and orders; no new ODE, adaptive coefficient, failure clipping or probability reweighting',
    scope='one bounded incident domain; analytically equivalent integrated signed mean; no complete plasma coefficient, collision closure or lifetime')
prior = json.loads((PRIOR/'summary.json').read_text())
for name in ('mass','charge','field','strength','screening','theta'):
    if prior['inputs'][name]!=INPUTS[name]:
        raise ValueError(f'Prior physical input differs: {name}')
INPUTS['prior_artifact_sha256']={str(PRIOR.relative_to(ROOT)/name):hashlib.sha256((PRIOR/name).read_bytes()).hexdigest() for name in ('summary.json','states.npz','populations.npz')}
INPUTS['historical_pilot_sha256']={str(PILOT.relative_to(ROOT)/name):hashlib.sha256((PILOT/name).read_bytes()).hexdigest() for name in ('summary.json','pilot_script.py','declared_inputs.json')}


def born_control(nodes,order):
    """Fixed free-helix potential-action phase derivative; zero phase average."""
    nodes=np.asarray(nodes)
    result=np.zeros(len(nodes))
    z,weights=roots_legendre(order)
    z=z*CONTROL_DISTANCE
    weights=weights*CONTROL_DISTANCE
    mask=nodes[:,0]-nodes[:,2]/OMEGA>=MIN_GAP
    for start in range(0,len(nodes),BATCH):
        indices=np.arange(start,min(start+BATCH,len(nodes)))
        indices=indices[mask[indices]]
        if not len(indices):
            continue
        b,u,p,phase=nodes[indices].T
        rho=p/OMEGA
        psi=phase[:,None]-OMEGA*z[None,:]/u[:,None]
        radius=np.sqrt(b[:,None]**2+rho[:,None]**2-2*b[:,None]*rho[:,None]*np.sin(psi)+z[None,:]**2)
        integrand=np.cos(psi)*np.exp(-radius/SCREENING)*(1+radius/SCREENING)/radius**3
        result[indices]=STRENGTH*b*p/(2*FIELD*u)*(integrand@weights)
    return COEFFICIENT*result


if OUTPUT.exists():
    raise FileExistsError(f'Preserve existing evidence: {OUTPUT}; select a fresh destination')
OUTPUT.mkdir(parents=True)
(OUTPUT/'declared_inputs.json').write_text(json.dumps(INPUTS,indent=2)+'\n')
import jax
jax.config.update('jax_enable_x64',True)
metadata=run_metadata(INPUTS,model='known-zero first-Born control of the saved bounded signed-mean integral',
    boundary=prior['boundary'],units=INPUTS['units'])
dependencies=[Path(__file__).resolve(),ROOT/'src/sato_morrison/reference.py']
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
metadata['experiment_dependency_sha256']=hashes
started=perf_counter()
with np.load(PRIOR/'states.npz') as saved:
    states={name:saved[name].copy() for name in saved.files}
with np.load(PRIOR/'populations.npz') as saved:
    populations={name:saved[name].copy() for name in saved.files}
if not np.all(states['status']=='passed') or not all(c['complete'] for c in prior['results']['cases']):
    raise ValueError('Cannot discard any failed or missing incoming weight')
unique_nodes,inverse=np.unique(states['nodes'],axis=0,return_inverse=True)
controls={}
for order in ORDERS:
    if perf_counter()-started>CAP:
        raise RuntimeError('Fixed control postprocessing cap exceeded; no partial integral accepted')
    unique_control=born_control(unique_nodes,order)
    controls[order]=unique_control[inverse]
    np.save(OUTPUT/f'control_GL{order}.npy',controls[order])
    print(f'GL{order}: {len(unique_nodes)} unique incoming nodes, {len(inverse)} saved states; wall {perf_counter()-started:.2f}s',flush=True)
probe_indices=np.linspace(0,len(unique_nodes)-1,128,dtype=int)
probe=unique_nodes[probe_indices].copy()
antithetic=probe.copy()
antithetic[:,3]=(np.pi-probe[:,3])%(2*np.pi)
symmetry={str(order):float(np.max(abs(born_control(probe,order)+born_control(antithetic,order)))) for order in ORDERS}
raw_cases=prior['results']['cases']
case_sets={}
variance_diagnostics=[]
for order in ORDERS:
    cases=[]
    for raw in raw_cases:
        label=raw['case']
        ids=populations[label]
        weights=populations[label+'_weights']
        values=np.column_stack((states['moments'][ids],states['individual_gc_second'][ids]))
        cv=values.copy()
        cv[:,0]-=controls[order][ids]
        integral=np.sum(weights[:,None]*cv,axis=0)
        case=dict(case=label,seed=raw['seed'],base_count=raw['base_count'],distance=raw['distance'],
            complete=True,integrals=integral.tolist(),raw_integrals=raw['integrals'],
            control_integral=float(weights@controls[order][ids]),node_count=len(ids),
            exact_total_flux_weight=float(weights.sum()))
        cases.append(case)
        if order==ORDERS[0] and raw['distance']==24. and raw['base_count']==max(prior['inputs']['base_levels']):
            strata=populations[label+'_stratum']
            entries=[]
            for k in np.unique(strata):
                m=strata==k
                raw_variance=float(np.var(values[m,0],ddof=1))
                controlled_variance=float(np.var(cv[m,0],ddof=1))
                cell_weight=float(weights[m].sum())
                entries.append(dict(stratum=int(k),count=int(m.sum()),flux_per_density=cell_weight,
                    raw_mean_variance=raw_variance,controlled_mean_variance=controlled_variance,
                    variance_ratio=controlled_variance/raw_variance if raw_variance>0 else None,
                    iid_weighted_variance_proxy_raw=cell_weight**2*raw_variance/m.sum(),
                    iid_weighted_variance_proxy_controlled=cell_weight**2*controlled_variance/m.sum()))
            variance_diagnostics.append(dict(case=label,strata=entries,
                scope='descriptive within-cell sample variances and IID allocation proxy; Sobol dependence precludes a confidence interval'))
    case_sets[order]=cases


def change(a,b):
    x,y=np.asarray(a['integrals']),np.asarray(b['integrals'])
    scale=np.array([x[1]/MU_REF,x[1],x[1],x[3]])
    return (abs(x-y)/scale).tolist()


def control_comparison(ids_a,ids_b,order):
    a=np.column_stack((states['moments'][ids_a],states['individual_gc_second'][ids_a]))
    b=np.column_stack((states['moments'][ids_b],states['individual_gc_second'][ids_b]))
    a[:,0]-=controls[order][ids_a]
    b[:,0]-=controls[order][ids_b]
    scale=np.column_stack((a[:,1]/MU_REF,a[:,1],a[:,1],a[:,3]))
    return np.max(abs(a-b)/scale,axis=0).tolist()


comparison_sets={}
for order,cases in case_sets.items():
    by_label={c['case']:c for c in cases}
    seeds=prior['inputs']['seeds']
    levels=prior['inputs']['base_levels']
    checks={}
    for seed in seeds:
        checks[f'sample_{seed}']=change(by_label[f'seed{seed}_N{levels[-1]}'],by_label[f'seed{seed}_N{levels[-2]}'])
    for a,b in combinations(seeds,2):
        checks[f'replica_{a}_{b}']=change(by_label[f'seed{a}_N{levels[-1]}'],by_label[f'seed{b}_N{levels[-1]}'])
    for seed in seeds:
        checks[f'partition_{seed}']=change(by_label[f'seed{seed}_N{levels[-1]}'],by_label['alternate_partition'])
    checks['boundary_ensemble']=change(by_label['boundary_L48'],by_label['boundary_L40'])
    checks['boundary_L24_to40']=change(by_label['boundary_L40'],by_label[f'seed{seeds[0]}_N{levels[-1]}'])
    ids=prior['results']['controls']['ids']
    checks['active_step']=control_comparison(np.asarray(ids['0.025']),np.asarray(ids['0.05']),order)
    checks['boundary_controls']=control_comparison(np.asarray(ids['L48']),np.asarray(ids['L40']),order)
    comparison_sets[order]=checks
stability={a['case']:abs(a['integrals'][0]-b['integrals'][0])/(a['integrals'][1]/MU_REF) for a,b in zip(case_sets[ORDERS[0]],case_sets[ORDERS[-1]])}


def gate(values,target):
    return np.all(np.isfinite(values)) and max(values)<=target


gates={k:gate(v,PLANE_TARGET if k.startswith('boundary_') else STEP_TARGET if k=='active_step' else QUADRATURE_TARGET) for k,v in comparison_sets[ORDERS[0]].items()}
for key in ('complete','exact_flux_mass','energy','active_step_sizes'):
    gates[key]=prior['results']['required_gates'][key]
gates.update(control_order_stability=all(np.isfinite(v) and v<=STABILITY_TARGET for v in stability.values()),
    control_symmetry=all(v<=SYMMETRY_TARGET for v in symmetry.values()),
    finite_controls=all(np.all(np.isfinite(v)) for v in controls.values()),
    prior_source_integrity=prior['results']['required_gates']['source_integrity'],
    prior_prior_integrity=prior['results']['required_gates']['prior_integrity'])
ending={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
metadata['experiment_dependency_sha256_end']=ending
gates['source_integrity']=hashes==ending
gates['prior_integrity']=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==wanted for name,wanted in INPUTS['prior_artifact_sha256'].items())
gates['historical_pilot_integrity']=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==wanted for name,wanted in INPUTS['historical_pilot_sha256'].items())
max_level=max(prior['inputs']['base_levels'])
replica_labels=[f'seed{seed}_N{max_level}' for seed in prior['inputs']['seeds']]
raw_replica=np.asarray([next(c['integrals'] for c in raw_cases if c['case']==label) for label in replica_labels])
controlled_replica=np.asarray([next(c['integrals'] for c in case_sets[ORDERS[0]] if c['case']==label) for label in replica_labels])
replica_spread=dict(raw_signed_mean_sample_std=float(np.std(raw_replica[:,0],ddof=1)),
    controlled_signed_mean_sample_std=float(np.std(controlled_replica[:,0],ddof=1)),
    ratio=float(np.std(controlled_replica[:,0],ddof=1)/np.std(raw_replica[:,0],ddof=1)),
    scope='descriptive spread of four independent Sobol scrambles; no confidence interval')
status='passed' if all(gates.values()) else 'unresolved'
metadata['results']=dict(status=status,required_gates=gates,comparisons=comparison_sets[ORDERS[0]],
    sensitivity_comparisons=comparison_sets[ORDERS[-1]],cases=case_sets[ORDERS[0]],
    sensitivity_cases=case_sets[ORDERS[-1]],control_order_stability=stability,control_symmetry_residual=symmetry,
    raw_campaign_status=prior['results']['status'],raw_campaign_required_gates=prior['results']['required_gates'],
    raw_campaign_comparisons=prior['results']['comparisons'],variance_diagnostics=variance_diagnostics,replica_spread=replica_spread,
    reused_encounters=len(states['nodes']),unique_control_nodes=len(unique_nodes),new_encounters=0,
    wall_s=perf_counter()-started,parent_process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
    timing_scope='postprocessing saved states only; all historical and fresh ODE solves excluded; plotting excluded',
    memory_scope='single process lifetime peak',
    limits=['The original raw campaign failures remain preserved; this is a distinct analytically equivalent mean estimator.',
        'Empirical sample, scramble, partition and control-order changes are not rigorous error bounds or confidence intervals.',
        'Known-zero control does not repair unresolved second/cross/spatial, boundary, active-step or physical-model errors.',
        'No omitted-domain moment bound, plasma screening closure, physical coefficient or lifetime.'])
(OUTPUT/'summary.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
for seed in prior['inputs']['seeds']:
    raw=[c for c in raw_cases if c['seed']==seed and c['distance']==24.]
    cv=[c for c in case_sets[ORDERS[0]] if c['seed']==seed and c['distance']==24.]
    line=axes[0].plot([c['node_count'] for c in raw],[c['integrals'][0] for c in raw],':o',ms=3,alpha=.5)[0]
    axes[0].plot([c['node_count'] for c in cv],[c['integrals'][0] for c in cv],'-o',ms=4,color=line.get_color(),label=f'controlled seed {seed}')
axes[0].set(xlabel='fresh four-dimensional encounter nodes',ylabel='signed mean integral per density (normalized)',title='Dotted: raw; solid: known-zero control')
axes[0].legend(fontsize=8)
groups=[('sample',[k for k in gates if k.startswith('sample_')],QUADRATURE_TARGET),
    ('replicas',[k for k in gates if k.startswith('replica_')],QUADRATURE_TARGET),
    ('partition',[k for k in gates if k.startswith('partition_')],QUADRATURE_TARGET),
    ('L24 to40',['boundary_L24_to40'],PLANE_TARGET),('L40 to48',['boundary_ensemble'],PLANE_TARGET),
    ('selected planes',['boundary_controls'],PLANE_TARGET),('active step',['active_step'],STEP_TARGET)]
for column,name in enumerate(('signed mean','second','cross','spatial second')):
    values=[max(comparison_sets[ORDERS[0]][k][column] for k in keys)/target for _,keys,target in groups]
    axes[1].plot(range(len(groups)),values,'o-',ms=4,label=name)
axes[1].axhline(1,color='gray',ls=':')
axes[1].set_xticks(range(len(groups)),[g[0] for g in groups],rotation=35,ha='right',fontsize=8)
axes[1].set(ylabel='worst change in group / fixed target',title=f'Controlled estimator: {status}')
axes[1].legend(fontsize=8)
fig.savefig(OUTPUT/'convergence.png',dpi=160)
plt.close(fig)
print(f'Finished {status}; gates={gates}; elapsed {perf_counter()-started:.2f}s',flush=True)
sys.exit(0 if status=='passed' else 1)
