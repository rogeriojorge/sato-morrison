"""Independent physical member, positive-weight, and antithetic gate audit."""
from pathlib import Path
from itertools import combinations
from time import perf_counter
from importlib.metadata import version
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import numpy as np

ROOT=Path(os.environ.get('SM_AUDIT_ROOT',Path(__file__).resolve().parents[2]))
D=ROOT/os.environ.get('SM_ANTITHETIC_AUDIT_INPUT','results/antithetic_encounter_flux')
P=ROOT/'results/stratified_encounter_flux'
OUTPUT=Path(os.environ.get('SM_ANTITHETIC_AUDIT_OUTPUT',D/'independent_audit.json'))
if OUTPUT.exists():raise FileExistsError(OUTPUT)
started=perf_counter()
j=json.loads((D/'summary.json').read_text());inp=j['inputs'];res=j['results']
raw=json.loads((P/'summary.json').read_text())
with np.load(P/'states.npz') as f:base={k:f[k] for k in f.files}
with np.load(D/'states.npz') as f:new={k:f[k] for k in f.files}
with np.load(D/'populations.npz') as f:p={k:f[k] for k in f.files}
with np.load(P/'populations.npz') as f:oldpop={k:f[k] for k in f.files}
a={k:np.concatenate((base[k],new[k])) for k in base};nbase=len(base['nodes'])
m,q,B,T=(inp[k] for k in ('mass','charge','field','theta'));omega=q*B/m;mu=m*T/B
checks=[]
def check(name,ok,error=None):
    r=dict(name=name,passed=bool(ok))
    if error is not None:r['maximum_absolute_error']=float(error) if np.isfinite(error) else 'nonfinite'
    checks.append(r)
def close(name,x,y,atol=2e-13,rtol=2e-12):
    x,y=np.asarray(x),np.asarray(y)
    check(name,x.shape==y.shape and np.allclose(x,y,atol=atol,rtol=rtol),np.max(abs(x-y)))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
if not np.all(a['status']=='passed') or not all(c['complete'] for c in res['cases']):
    receipt=dict(status='failed',reason='Missing or failed physical member invalidates the fixed-weight integral; no valid-only renormalization or partial moment reconstruction is allowed.',
        producer_status=res['status'],failed_attempts=res['failures'],cap_exhausted=res['cap_exhausted'],
        incomplete_cases=[c['case'] for c in res['cases'] if not c['complete']],
        source_sha256=sha(Path(__file__)),input_sha256={str(f.relative_to(ROOT)):sha(f) for f in (D/'summary.json',D/'states.npz',D/'populations.npz',P/'states.npz',P/'populations.npz')})
    OUTPUT.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2));sys.exit(1)
check('immutable base reference count',nbase==inp['base_state_count']==res['base_state_count'])
check('all accessible physical states complete',np.all(a['status']=='passed'))
keys=np.column_stack((a['nodes'],a['distance'],a['step']))
check('new archive contains no duplicated old or new solver keys',len(np.unique(keys,axis=0))==len(keys))
wi,wo=a['initial_relative_velocity'],a['final_relative_velocity'];angle=omega*a['time']
unrotated=np.column_stack((np.cos(angle)*wo[:,0]-np.sin(angle)*wo[:,1],np.sin(angle)*wo[:,0]+np.cos(angle)*wo[:,1]))
delta=unrotated-wi[:,:2]
c=m*(np.sum(wo[:,:2]**2,axis=1)-np.sum(wi[:,:2]**2,axis=1))/(8*B)
variance=m*m*T*np.sum(delta**2,axis=1)/(8*B*B)
moments=np.column_stack((c,c*c+variance,c*c-variance))
quarter=np.array([[0.,1.],[-1.,0.]])
gi=a['initial_relative_position'][:,:2]+wi[:,:2]@quarter.T/omega
go=a['final_relative_position'][:,:2]+wo[:,:2]@quarter.T/omega
kick=go-gi;spatial=np.sum(kick**2,axis=1)/4;values=np.column_stack((moments,spatial))
close('all member COM moments before pairing',moments,a['moments'],atol=2e-15)
close('all member free-gyro kicks',delta,a['delta_relative_perpendicular'],atol=3e-13)
close('all actual guiding-center kicks',kick,a['gc_relative_kick'],atol=3e-13)
close('all individual spatial seconds before pairing',spatial,a['individual_gc_second'],atol=3e-15)
phase=a['nodes'][:,3]+omega*a['distance']/a['nodes'][:,1]
vi=np.column_stack((a['nodes'][:,2]*np.cos(phase),a['nodes'][:,2]*np.sin(phase),a['nodes'][:,1]))
ri=np.column_stack((a['nodes'][:,0]-vi[:,1]/omega,vi[:,0]/omega,-a['distance']))
close('all incoming free-helix velocities',vi,wi,atol=3e-13)
close('all incoming free-helix positions',ri,a['initial_relative_position'],atol=3e-13)
check('all outgoing planes and directions',np.allclose(abs(a['final_relative_position'][:,2]),a['distance'],atol=3e-12,rtol=0) and np.all(np.sign(wo[:,2])==np.sign(a['final_relative_position'][:,2])) and np.all((a['exit']=='reflected')==(wo[:,2]<0)))
def energy(r,v):
    radius=np.linalg.norm(r,axis=1)
    return m*np.sum(v*v,axis=1)/4+inp['strength']*np.exp(-radius/inp['screening'])/radius
ee=abs(energy(a['final_relative_position'],wo)/energy(a['initial_relative_position'],wi)-1)
check('all endpoint energies bounded by saved path errors',np.all(ee<=a['energy_error']+3e-14))
def canonical(r,v):return r[:,0]*v[:,1]-r[:,1]*v[:,0]+omega*np.sum(r[:,:2]**2,axis=1)/2
cs=abs(omega)*a['nodes'][:,0]**2/2+a['nodes'][:,2]**2/(2*abs(omega))
ce=abs(canonical(a['final_relative_position'],wo)-canonical(a['initial_relative_position'],wi))/cs
check('all axial canonical invariants',np.max(ce)<=1e-9,np.max(ce))
rs=a['nodes'][:,0]**2+a['nodes'][:,2]**2/omega**2
re=abs(np.sum(go**2,axis=1)-np.sum(gi**2,axis=1)-(np.sum(wo[:,:2]**2,axis=1)-np.sum(wi[:,:2]**2,axis=1))/omega**2)/rs
check('all canonical radius identities',np.max(re)<=1e-9,np.max(re))
def flux(bb,ub,pb):
    f=16*np.pi**2*T*T*(bb[1]**2-bb[0]**2)/(4*np.pi*T)**1.5
    for lo,hi in (ub,pb):f*=np.exp(-lo*lo/(4*T))-np.exp(-hi*hi/(4*T))
    return f
F=flux(*raw['inputs']['bounds']);cv={};transforms=[];used=set();inverse_diagnostics={}
for case in res['cases']:
    label=case['case'];old=p[label+'_original'];partner=p[label+'_partner'];weights=p[label+'_pair_weights'];cell=p[label+'_stratum']
    close(label+' original selector bits',old,oldpop[label],atol=0,rtol=0)
    close(label+' unchanged strata',cell,oldpop[label+'_stratum'],atol=0,rtol=0)
    close(label+' unchanged pair masses',weights,oldpop[label+'_weights'],atol=0,rtol=0)
    check(label+' valid unique member references',np.all((old>=0)&(old<nbase)) and np.all((partner>=0)&(partner<len(a['nodes']))) and len(np.unique(partner))==len(partner))
    used.update(int(i) for ids in (old,partner) for i in ids if i<nbase)
    expected=a['nodes'][old].copy();expected[:,3]=np.remainder(np.pi-expected[:,3],2*np.pi)
    close(label+' exact physical partner transform',a['nodes'][partner],expected,atol=0,rtol=0)
    close(label+' matched member planes',a['distance'][partner],a['distance'][old],atol=0,rtol=0)
    close(label+' all bulk member steps',a['step'][partner],np.full(len(partner),inp['base_step']),atol=0,rtol=0)
    definitions=inp['alternate_strata'] if case['seed']==inp['alternate_seed'] else inp['main_strata']
    member_contribution=(values[old]+values[partner])/2
    integral=weights@member_contribution;cv[label]=integral
    close(label+' reconstructed physical pair integral',integral,case['integrals'],atol=2e-14)
    close(label+' original failed estimator integral retained',case['original_integrals'],next(x['integrals'] for x in raw['results']['cases'] if x['case']==label),atol=0,rtol=0)
    same_cell=True
    for definition in definitions:
        s=definition['index'];mask=cell==s;count=case['base_count']*definition['multiplier']
        mass=flux(definition['impact'],definition['parallel'],definition['perpendicular'])*(definition['phase'][1]-definition['phase'][0])/(2*np.pi)
        check(label+f' cell{s} unchanged allocation',np.sum(mask)==count)
        check(label+f' cell{s} recorded count',case['strata'][s]['count']==count)
        close(label+f' cell{s} recorded mass',case['strata'][s]['flux_per_density'],mass,atol=2e-14)
        close(label+f' cell{s} positive half-weights',weights[mask]/2,np.full(count,mass/(2*count)),atol=1e-18,rtol=2e-14)
        local=(a['nodes'][partner[mask],3]-definition['phase'][0])%(2*np.pi)
        same_cell=same_cell and bool(np.all(local<=definition['phase'][1]-definition['phase'][0]+inp['phase_transform_tolerance']))
        close(label+f' cell{s} paired component integral',weights[mask]@member_contribution[mask],case['strata'][s]['integrals'],atol=2e-14)
    twice=(np.pi-a['nodes'][partner,3])%(2*np.pi)
    inverse_error=float(np.max(abs((twice-a['nodes'][old,3]+np.pi)%(2*np.pi)-np.pi)))
    transforms.append(dict(case=label,same_cell=same_cell,involution_max_phase_error=inverse_error))
    close(label+' complete member mass',np.sum(weights),F,atol=2e-14)
    close(label+' recorded member mass',case['total_flux_weight'],np.sum(weights),atol=2e-14)
    close(label+' reflected incoming half-masses',np.sum(weights*((a['exit'][old]=='reflected').astype(float)+(a['exit'][partner]=='reflected')))/2/F,case['reflected_flux_fraction'],atol=2e-14)
    check(label+' physical member counts',case['physical_member_count']==2*len(old) and case['node_count']==len(old) and case['complete'] and case['missing_count']==0)
    diagnostics={}
    for name,axis in (('impact',0),('parallel_speed',1),('perpendicular_speed',2),('any_band',None)):
        outside=[]
        for ids in (old,partner):
            out=np.column_stack((np.linalg.norm(go[ids],axis=1),abs(wo[ids,2]),np.linalg.norm(wo[ids,:2],axis=1)))
            bounds=np.asarray(raw['inputs']['bounds']);flags=(out<bounds[:,0])|(out>bounds[:,1])
            outside.append(np.any(flags,axis=1) if axis is None else flags[:,axis])
        diagnostics[name]=dict(incoming_flux_fraction=float(np.sum(weights*(outside[0].astype(float)+outside[1]))/(2*F)),
            incoming_second_integral_fraction=float(np.sum(weights*(moments[old,1]*outside[0]+moments[partner,1]*outside[1]))/(2*integral[1])))
    inverse_diagnostics[label]=diagnostics
for actual,recorded in zip(transforms,res['transform_checks']):
    check(actual['case']+' transform receipt',actual['case']==recorded['case'] and actual['same_cell']==recorded['same_cell'])
    close(actual['case']+' involution roundoff receipt',actual['involution_max_phase_error'],recorded['involution_max_phase_error'],atol=0,rtol=0)
def change(x,y):
    v,w=cv[x],cv[y]
    return abs(v-w)/np.array([v[1]/mu,v[1],v[1],v[3]])
levels,seeds=inp['base_levels'],inp['seeds'];cmp={}
for seed in seeds:cmp[f'sample_{seed}']=change(f'seed{seed}_N{levels[-1]}',f'seed{seed}_N{levels[-2]}')
for x,y in combinations(seeds,2):cmp[f'replica_{x}_{y}']=change(f'seed{x}_N{levels[-1]}',f'seed{y}_N{levels[-1]}')
for seed in seeds:cmp[f'partition_{seed}']=change(f'seed{seed}_N{levels[-1]}','alternate_partition')
cmp['boundary_ensemble']=change('boundary_L48','boundary_L40')
cmp['boundary_L24_to40']=change('boundary_L40',f'seed{seeds[0]}_N{levels[-1]}')
controls=res['controls'];control_values={};member_values={}
for label,selectors in controls['ids'].items():
    old=np.asarray(selectors['original']);partner=np.asarray(selectors['partner'])
    used.update(int(i) for ids in (old,partner) for i in ids if i<nbase)
    close('control '+label+' original nodes',a['nodes'][old],controls['nodes'],atol=0,rtol=0)
    expected=a['nodes'][old].copy();expected[:,3]=(np.pi-expected[:,3])%(2*np.pi)
    close('control '+label+' actual partner nodes',a['nodes'][partner],expected,atol=0,rtol=0)
    expected_distance=float(label[1:]) if label.startswith('L') else 24.
    expected_step=inp['base_step'] if label.startswith('L') else float(label)
    for name,ids in (('original',old),('partner',partner)):
        close('control '+label+' '+name+' boundary',a['distance'][ids],np.full(len(ids),expected_distance),atol=0,rtol=0)
        close('control '+label+' '+name+' step',a['step'][ids],np.full(len(ids),expected_step),atol=0,rtol=0)
    control_values[label]=(values[old]+values[partner])/2;member_values[label]=np.vstack((values[old],values[partner]))
def maxchange(v,w):return np.max(abs(v-w)/np.column_stack((v[:,1]/mu,v[:,1],v[:,1],v[:,3])),axis=0)
for label,x,y in (('active_step','0.025','0.05'),('boundary_controls','L48','L40')):
    cmp[label]=maxchange(control_values[x],control_values[y]);cmp[label+'_members']=maxchange(member_values[x],member_values[y])
fine,coarse=controls['ids']['0.025'],controls['ids']['0.05']
ratios=np.concatenate([a['max_actual_step'][fine[k]]/a['max_actual_step'][coarse[k]] for k in ('original','partner')])
close('both member active-step ratios',ratios,controls['actual_step_ratios'],atol=2e-14)
for key,v in cmp.items():close(key+' fixed comparison',v,res['comparisons'][key],atol=3e-7,rtol=3e-7)
gates={k:np.max(v)<=(inp['plane_target'] if k.startswith('boundary_') else inp['step_target'] if k.startswith('active_step') else inp['quadrature_target']) for k,v in cmp.items()}
gates.update(phase_transform=all(r['same_cell'] and r['involution_max_phase_error']<=inp['phase_transform_tolerance'] for r in transforms),
    preserved_born_control=sha(ROOT/'results/encounter_born_control/summary.json')==inp['preserved_born_control_summary_sha256'],
    complete=all(c['complete'] for c in res['cases']) and not res['failures'] and not res['cap_exhausted'],
    exact_flux_mass=all(np.isclose(np.sum(p[c['case']+'_pair_weights']),F,rtol=5e-13,atol=0) for c in res['cases']),
    energy=bool(len(new['nodes'])) and np.all(a['status']=='passed') and np.all(a['energy_error']<=inp['energy_target']),
    active_step_sizes=bool(len(ratios)) and np.all(ratios<=.6),
    prior_integrity=all(sha(ROOT/k)==v for k,v in inp['prior_artifact_sha256'].items()),
    prior_prior_integrity=raw['results']['required_gates']['prior_integrity'],prior_source_integrity=raw['results']['required_gates']['source_integrity'],
    source_integrity=j['experiment_dependency_sha256']==j['experiment_dependency_sha256_end'])
check('all declared gate names reconstructed',set(gates)==set(res['required_gates']))
for key,v in gates.items():check(key+' decision reconstructed',bool(v)==res['required_gates'][key])
check('old failed-gate evidence unchanged',res['raw_campaign_required_gates']==raw['results']['required_gates'] and res['raw_campaign_comparisons']==raw['results']['comparisons'] and res['raw_campaign_status']==raw['results']['status'])
check('overall status reconstructed',res['status']==('passed' if all(gates.values()) else 'unresolved'))
check('new-only archive and actual reference counts',len(new['nodes'])==res['new_encounters'] and len(used)==res['reused_encounters'] and len(a['nodes'])==res['unique_accessible_encounters'])
original=j['original_producer']
check('original producer exact timing and parent memory unavailable',res['wall_s'] is None and res['parent_process_peak_rss_bytes'] is None and original['original_dependency_end_sha256'] is None)
check('receipt replay performed no new solves',res['new_solves_this_execution']==0)
check('original source copy immutable',sha(D/original['original_source_copy'])==original['source_copy_sha256'])
check('original failure log immutable',sha(D/original['progress_log'])==original['progress_log_sha256'])
for name,wanted in original['experiment_dependency_sha256'].items():
    blob=subprocess.run(['git','-C',str(ROOT),'show',original['producer_commit']+':'+name],capture_output=True,check=True).stdout
    check('original physics producer '+name,hashlib.sha256(blob).hexdigest()==wanted)
for name,wanted in original['saved_artifact_sha256'].items():
    path=D/name
    if path.exists():check('immutable original artifact '+name,sha(path)==wanted)
    else:check('only archived optional checkpoints absent '+name,name.startswith('checkpoint_') and name.endswith('.npz'))
check('recovery scientific cases reproduce saved cases',json.loads((D/'cases_partial.json').read_text())==res['cases'])
producer={}
for name,wanted in j['experiment_dependency_sha256'].items():
    blob=subprocess.run(['git','-C',str(ROOT),'show',j['commit']+':'+name],capture_output=True,check=True).stdout
    actual=hashlib.sha256(blob).hexdigest();check('historical receipt writer '+name,actual==wanted)
    producer[name]=dict(commit=j['commit'],expected_sha256=wanted,actual_sha256=actual)
ratio={label:float(inp['screening']**2*v[1]/(mu*mu*v[3])) for label,v in cv.items()}
fine_labels=[f'seed{s}_N{levels[-1]}' for s in seeds]+['alternate_partition']
script=Path(__file__).resolve();relative=str(script.relative_to(ROOT))
receipt=dict(status='passed' if all(c['passed'] for c in checks) else 'failed',checks=checks,
    scope='Independent all-member Cartesian reconstruction, exact positive half-weights, physical partner mapping, no duplicated old archive, and all fixed gate decisions. No producer physics imports or fresh ODE solves.',
    comparison_normalization=dict(formula='abs(I_fine-I_other)/[I2_fine/mu_reference,I2_fine,I2_fine,I_R_fine]',
        interpretation='All scales are positive. A 5% gate does not mean 5% relative accuracy in a small signed mean or cross moment. Fixed sample, scramble and partition comparisons are empirical; no confidence interval or continuous-domain bound follows.'),
    reconstructed_gates={k:bool(v) for k,v in gates.items()},reconstructed_comparisons={k:v.tolist() for k,v in cmp.items()},
    transform_checks=transforms,normalized_kick_ratio=dict(formula='lambda^2 I2/(mu_reference^2 I_R)',cases=ratio,fine_range=[min(ratio[k] for k in fine_labels),max(ratio[k] for k in fine_labels)],scope='Finite-band normalized squared-kick ratio only; no relaxation/mixing time or plasma closure.'),
    inverse_cutoff_diagnostic=dict(scope='Each physical original and partner retains half its exact incoming mass, including reflected exits; no extra outgoing flux factor or detailed-balance claim.',cases=inverse_diagnostics),
    invariant_scales=dict(canonical='|Omega|b_in^2/2+p_in^2/(2|Omega|), length*velocity',radius='b_in^2+(p_in/Omega)^2, length^2',target=1e-9),
    original_physics_producer_commit=original['producer_commit'],receipt_writer_commit=j['commit'],receipt_writer_source_identities=producer,source_path=relative,source_sha256=sha(script),
    audit_git_identity=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
    source_uncommitted=bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--',relative],text=True).strip()),
    input_sha256={str(f.relative_to(ROOT)):sha(f) for f in (D/'summary.json',D/'states.npz',D/'populations.npz',P/'states.npz',P/'populations.npz')},
    wall_s=perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
    versions=dict(numpy=version('numpy'),python=sys.version.split()[0]),platform=platform.platform())
OUTPUT.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(status=receipt['status'],checks=len(checks),failed=[c for c in checks if not c['passed']]),indent=2))
