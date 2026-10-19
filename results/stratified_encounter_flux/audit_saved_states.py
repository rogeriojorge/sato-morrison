"""Independent saved Cartesian states, incident measure, and gate audit."""
from pathlib import Path
from itertools import product, combinations
import hashlib
import json
import os
import subprocess
import platform
from importlib.metadata import version
import resource
import sys
from time import perf_counter
import numpy as np
from scipy.stats import qmc

ROOT = Path(os.environ.get('SM_AUDIT_ROOT',Path(__file__).resolve().parents[2]))
started=perf_counter()
D = ROOT/os.environ.get('SM_STRATIFIED_AUDIT_INPUT','results/stratified_encounter_flux')
OUTPUT=Path(os.environ.get('SM_STRATIFIED_AUDIT_OUTPUT',D/'independent_audit.json'))
if OUTPUT.exists():raise FileExistsError(f'Preserve existing audit receipt: {OUTPUT}')
summary = json.loads((D/'summary.json').read_text())
inp, result = summary['inputs'], summary['results']
with np.load(D/'states.npz') as loaded:a={key:loaded[key] for key in loaded.files}
with np.load(D/'populations.npz') as loaded:pop={key:loaded[key] for key in loaded.files}
m,B,q,T = (inp[k] for k in ('mass','field','charge','theta'))
omega,mu = q*B/m,m*T/B
checks=[]
def check(name,ok,error=None):
    row=dict(name=name,passed=bool(ok))
    if error is not None:row['maximum_absolute_error']=float(error)
    checks.append(row)
def close(name,x,y,atol=2e-13,rtol=2e-12):
    x,y=np.asarray(x),np.asarray(y)
    check(name,x.shape==y.shape and np.allclose(x,y,atol=atol,rtol=rtol),np.max(abs(x-y)))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

wi,wo=a['initial_relative_velocity'],a['final_relative_velocity']
t=a['time'];ang=omega*t
unwound=np.column_stack((np.cos(ang)*wo[:,0]-np.sin(ang)*wo[:,1],np.sin(ang)*wo[:,0]+np.cos(ang)*wo[:,1]))
delta=unwound-wi[:,:2]
c=m*(np.sum(wo[:,:2]**2,axis=1)-np.sum(wi[:,:2]**2,axis=1))/(8*B)
variance=m*m*T*np.sum(delta**2,axis=1)/(8*B*B)
moments=np.column_stack((c,c*c+variance,c*c-variance))
quarter=np.array([[0.,1.],[-1.,0.]])
gc=a['final_relative_position'][:,:2]-a['initial_relative_position'][:,:2]+(wo[:,:2]-wi[:,:2])@quarter.T/omega
spatial=np.sum(gc**2,axis=1)/4
values=np.column_stack((moments,spatial))
check('all attempted states complete',np.all(a['status']=='passed'))
close('COM moments from Cartesian velocities',moments,a['moments'],atol=2e-15)
close('free gyro unwinding',delta,a['delta_relative_perpendicular'],atol=2e-13)
close('actual endpoint guiding-center kick',gc,a['gc_relative_kick'],atol=5e-14)
close('individual spatial second',spatial,a['individual_gc_second'],atol=2e-15)
phase=a['nodes'][:,3]+omega*a['distance']/a['nodes'][:,1]
expected=np.column_stack((a['nodes'][:,2]*np.cos(phase),a['nodes'][:,2]*np.sin(phase),a['nodes'][:,1]))
close('incoming free-helix velocity',expected,wi,atol=2e-13)
position=np.column_stack((a['nodes'][:,0]-wi[:,1]/omega,wi[:,0]/omega,-a['distance']))
close('incoming free-helix position',position,a['initial_relative_position'],atol=2e-13)
check('first exit plane and direction',np.allclose(abs(a['final_relative_position'][:,2]),a['distance'],atol=2e-12,rtol=0) and np.all((a['exit']=='reflected')==(wo[:,2]<0)) and np.all(np.sign(wo[:,2])==np.sign(a['final_relative_position'][:,2])))
def energy(r,v):
    radius=np.linalg.norm(r,axis=1)
    return m*np.sum(v*v,axis=1)/4+inp['strength']*np.exp(-radius/inp['screening'])/radius
ee=abs(energy(a['final_relative_position'],wo)/energy(a['initial_relative_position'],wi)-1)
check('endpoint energy bounded by path maximum',np.all(ee<=a['energy_error']+3e-14))
def canonical(r,v):
    return r[:,0]*v[:,1]-r[:,1]*v[:,0]+omega*np.sum(r[:,:2]**2,axis=1)/2
canonical_initial=canonical(a['initial_relative_position'],wi)
canonical_final=canonical(a['final_relative_position'],wo)
canonical_scale=abs(omega)*a['nodes'][:,0]**2/2+a['nodes'][:,2]**2/(2*abs(omega))
canonical_error=abs(canonical_final-canonical_initial)/canonical_scale
check('endpoint axial canonical angular momentum',np.max(canonical_error)<=1e-9,np.max(canonical_error))
initial_gc=a['initial_relative_position'][:,:2]+wi[:,:2]@quarter.T/omega
final_gc=a['final_relative_position'][:,:2]+wo[:,:2]@quarter.T/omega
gc_radius_change=np.sum(final_gc**2,axis=1)-np.sum(initial_gc**2,axis=1)
speed_change=(np.sum(wo[:,:2]**2,axis=1)-np.sum(wi[:,:2]**2,axis=1))/omega**2
radius_scale=a['nodes'][:,0]**2+a['nodes'][:,2]**2/omega**2
check('canonical guiding-center radius identity',np.max(abs(gc_radius_change-speed_change)/radius_scale)<=1e-9,np.max(abs(gc_radius_change-speed_change)/radius_scale))

def flux(bb,ub,pb):
    z=16*np.pi**2*T*T*(bb[1]**2-bb[0]**2)/(4*np.pi*T)**1.5
    for lo,hi in (ub,pb):z*=np.exp(-lo*lo/(4*T))-np.exp(-hi*hi/(4*T))
    return z
F=flux(*inp['bounds'])
close('analytic original flux',F,inp['bounded_flux_per_density'],atol=1e-14)
case_values={};records={r['case']:r for r in result['cases']};inverse_diagnostics={}
for label,row in records.items():
    seed=row['seed'];h=inp['alternate_half_width'] if seed==inp['alternate_seed'] else inp['phase_half_width']
    intervals=((np.pi/2-h,np.pi/2+h),(np.pi/2+h,np.pi/2-h+2*np.pi))
    ids=pop[label];weight=pop[label+'_weights'];stratum=pop[label+'_stratum']
    check(label+' complete unique selectors',row['complete'] and row['missing_count']==0 and len(np.unique(ids))==len(ids) and np.all((ids>=0)&(ids<len(wi))))
    close(label+' matched boundary plane',a['distance'][ids],np.full(len(ids),row['distance']),atol=0,rtol=0)
    close(label+' base integration step',a['step'][ids],np.full(len(ids),inp['base_step']),atol=0,rtol=0)
    totalmass=0.
    for s,(ib,ip,ia) in enumerate(product(range(2),repeat=3)):
        bb=inp['impact_bins'][ib];ub=inp['bounds'][1];pb=inp['perpendicular_bins'][ip];ab=intervals[ia]
        n=row['base_count']*(inp['risk_multiplier'] if (ib,ip,ia)==(0,1,0) else 1)
        mass=flux(bb,ub,pb)*(ab[1]-ab[0])/(2*np.pi);totalmass+=mass
        mask=stratum==s;ii=ids[mask]
        check(label+f' stratum{s} allocation',len(ii)==n)
        close(label+f' stratum{s} exact weights',weight[mask],np.full(n,mass/n),atol=1e-18,rtol=2e-14)
        cube=qmc.Sobol(4,scramble=True,seed=100*seed+s).random_base2(int(np.log2(n)))
        coords=[np.sqrt(bb[0]**2+cube[:,0]*(bb[1]**2-bb[0]**2))]
        for k,(lo,hi) in enumerate((ub,pb),1):
            v=np.exp(-lo*lo/(4*T))-cube[:,k]*(np.exp(-lo*lo/(4*T))-np.exp(-hi*hi/(4*T)))
            coords.append(np.sqrt(-4*T*np.log(v)))
        coords.append((ab[0]+(ab[1]-ab[0])*cube[:,3])%(2*np.pi))
        close(label+f' stratum{s} independent exact incoming map',np.column_stack(coords),a['nodes'][ii],atol=2e-14,rtol=2e-14)
        component=weight[mask]@values[ii]
        close(label+f' stratum{s} weighted integral',component,row['strata'][s]['integrals'],atol=2e-15)
    close(label+' exact partition mass',totalmass,F,atol=2e-14)
    close(label+' retained full incoming weight',np.sum(weight),F,atol=2e-14)
    close(label+' recorded incoming mass',np.sum(weight),row['total_flux_weight'],atol=1e-14,rtol=5e-13)
    v=weight@values[ids];case_values[label]=v
    close(label+' complete weighted integral',v,row['integrals'],atol=2e-14)
    close(label+' reported moment prefactors',[v[0],v[1]/2,v[2],v[3]],[row['drift_per_density'],row['diffusion_per_density'],row['cross_per_density'],row['spatial_second_per_density']],atol=2e-14)
    close(label+' reflected flux retained',np.sum(weight[a['exit'][ids]=='reflected'])/F,row['reflected_flux_fraction'],atol=2e-15)
    outgoing=np.column_stack((np.linalg.norm(final_gc[ids],axis=1),abs(wo[ids,2]),np.linalg.norm(wo[ids,:2],axis=1)))
    masks={name:(outgoing[:,axis]<inp['bounds'][axis][0])|(outgoing[:,axis]>inp['bounds'][axis][1]) for axis,name in enumerate(('impact','parallel_speed','perpendicular_speed'))}
    masks['any_band']=np.logical_or.reduce(list(masks.values()))
    inverse_diagnostics[label]={name:dict(incoming_flux_fraction=float(np.sum(weight[mask])/F),
        incoming_second_integral_fraction=float(np.sum(weight[mask]*moments[ids[mask],1])/v[1])) for name,mask in masks.items()}

def change(x,y,ref=None):
    x,y=case_values[x],case_values[y]
    r=x if ref is None else case_values[ref]
    return abs(x-y)/np.array([r[1]/mu,r[1],r[1],r[3]])
levels=inp['base_levels'];seeds=inp['seeds'];cmp={};hist={}
for seed in seeds:
    labels=[f'seed{seed}_N{n}' for n in levels]
    cmp[f'sample_{seed}']=change(labels[-1],labels[-2])
    hist[f'sample_{seed}_early']=change(labels[-2],labels[-3],labels[-1])
for x,y in combinations(seeds,2):cmp[f'replica_{x}_{y}']=change(f'seed{x}_N{levels[-1]}',f'seed{y}_N{levels[-1]}')
for seed in seeds:cmp[f'partition_{seed}']=change(f'seed{seed}_N{levels[-1]}','alternate_partition')
cmp['boundary_ensemble']=change('boundary_L48','boundary_L40')
cmp['boundary_L24_to40']=change('boundary_L40',f'seed{seeds[0]}_N{levels[-1]}')
controls=result['controls'];ids=controls['ids']
for key,selectors in ids.items():
    close('control '+key+' exact incident nodes',a['nodes'][selectors],controls['nodes'],atol=0,rtol=0)
    expected_distance=float(key[1:]) if key.startswith('L') else 24.
    expected_step=inp['base_step'] if key.startswith('L') else float(key)
    close('control '+key+' exact boundary',a['distance'][selectors],np.full(len(selectors),expected_distance),atol=0,rtol=0)
    close('control '+key+' exact step',a['step'][selectors],np.full(len(selectors),expected_step),atol=0,rtol=0)
def control_change(x,y):
    v,w=values[ids[x]],values[ids[y]]
    return np.max(abs(v-w)/np.column_stack((v[:,1]/mu,v[:,1],v[:,1],v[:,3])),axis=0)
cmp['active_step']=control_change(str(inp['steps'][-1]),str(inp['steps'][-2]))
cmp['boundary_controls']=control_change('L48','L40')
ratios=a['max_actual_step'][ids[str(inp['steps'][-1])]]/a['max_actual_step'][ids[str(inp['steps'][-2])]]
close('active maximum step ratios',ratios,controls['actual_step_ratios'],atol=2e-14)
for k,v in cmp.items():close(k+' gate comparison',v,result['comparisons'][k],atol=2e-7,rtol=2e-7)
for k,v in hist.items():close(k+' history diagnostic',v,result['history_diagnostics'][k],atol=2e-7,rtol=2e-7)
gates={k:bool(np.max(v)<=(inp['plane_target'] if k.startswith('boundary_') else inp['step_target'] if k=='active_step' else inp['quadrature_target'])) for k,v in cmp.items()}
gates.update(complete=all(r['complete'] for r in records.values()) and not result['failures'] and not result['cap_exhausted'],
 exact_flux_mass=all(np.isclose(np.sum(pop[k+'_weights']),F,rtol=5e-13,atol=0) for k in records),
 energy=bool(len(wi)) and np.all(a['status']=='passed') and np.all(a['energy_error']<=inp['energy_target']),
 active_step_sizes=bool(len(ratios)) and np.all(ratios<=.6),
 source_integrity=summary['experiment_dependency_sha256']==summary['experiment_dependency_sha256_end'],
 prior_integrity=all(sha(ROOT/k)==v for k,v in inp['prior_artifact_sha256'].items()))
check('all declared gates present',set(gates)==set(result['required_gates']))
for k,v in gates.items():check(k+' decision reproduced',v==result['required_gates'][k])
check('overall producer status reproduced',result['status']==('passed' if all(gates.values()) else 'unresolved'))
kick_ratios={label:float(inp['screening']**2*v[1]/(mu*mu*v[3])) for label,v in case_values.items()}
ratio_changes={}
for seed in seeds:
    x,y=(f'seed{seed}_N{n}' for n in levels[-2:])
    ratio_changes[f'sample_{seed}']=abs(kick_ratios[y]-kick_ratios[x])/kick_ratios[y]
for x,y in combinations(seeds,2):
    vx,vy=kick_ratios[f'seed{x}_N{levels[-1]}'],kick_ratios[f'seed{y}_N{levels[-1]}']
    ratio_changes[f'replica_{x}_{y}']=abs(vx-vy)/vx
for seed in seeds:
    v=kick_ratios[f'seed{seed}_N{levels[-1]}']
    ratio_changes[f'partition_{seed}']=abs(v-kick_ratios['alternate_partition'])/v
old_path=ROOT/'results/physical_final_campaign/cases.json'
old_summary=json.loads((old_path.parent/'summary.json').read_text())
old=next(row for row in json.loads(old_path.read_text()) if row['case']=='B2_final')
check('archived narrow comparison same physical parameters',old['field']==inp['field'] and all(old_summary['inputs'][k]==inp[k] for k in ('mass','charge','theta','screening','strength')))
old_ratio=inp['screening']**2*(2*old['diffusion_mu_per_density'])/(mu*mu*(2*old['gc_trace_diffusion_per_density']))
close('archived narrow same-field normalized kick ratio',old_ratio,old['moment_to_spatial_rate_ratio'],atol=1e-13)
fine_ratio_values=[kick_ratios[f'seed{seed}_N{levels[-1]}'] for seed in seeds]+[kick_ratios['alternate_partition']]
check('archive attempt count',len(wi)==result['new_encounters']==result['unique_encounters'] and result['reused_encounters']==0)
producer_identities={}
for name,wanted in summary['experiment_dependency_sha256'].items():
    blob=subprocess.run(['git','-C',str(ROOT),'show',summary['commit']+':'+name],capture_output=True,check=True).stdout
    actual=hashlib.sha256(blob).hexdigest()
    check('historical committed source '+name,actual==wanted)
    producer_identities[name]=dict(commit=summary['commit'],expected_sha256=wanted,actual_sha256=actual)
script=Path(__file__).resolve();script_relative=str(script.relative_to(ROOT))
receipt=dict(status='passed' if all(r['passed'] for r in checks) else 'failed',checks=checks,
 wall_s=perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
 scope='Independent saved-state and exact-measure reconstruction; no producer imports or new trajectory solves. Scramble comparisons are empirical, not rigorous error bounds.',
 invariant_formulas=dict(canonical='J=(r cross w)_z+Omega*|r_perp|^2/2, units length*velocity',
 canonical_scale='|Omega|*b_in^2/2+p_in^2/(2*|Omega|), positive on bounded domain',
 radius='|Rgc_out|^2-|Rgc_in|^2=(p_out^2-p_in^2)/Omega^2=8B*c/(m*Omega^2)',
 radius_scale='b_in^2+(p_in/Omega)^2, units length^2',residual_target=1e-9),
 producer_commit=summary['commit'],producer_source_identities=producer_identities,
 versions=dict(numpy=version('numpy'),scipy=version('scipy'),python=sys.version.split()[0]),platform=platform.platform(),
 audit_git_identity=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
 source_path=script_relative,source_uncommitted=bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--',script_relative],text=True).strip()),
 inverse_cutoff_diagnostic=dict(scope='Kinematic cutoff-closure diagnostic: magnitudes of inverse incident data at outgoing finite planes; includes transmitted and reflected exits with exact original incoming flux weights. No detailed-balance or physical-rate claim.',
    outgoing_coordinates='(|r_perp+(w_y,-w_x)/Omega|, |w_z|, |w_perp|)',cases=inverse_diagnostics),
 normalized_kick_ratio=dict(formula='Q=lambda^2*I2/(mu_reference^2*I_R)=r_mu/r_R; partner density cancels',
    scope='Ratio of normalized accumulated squared kicks on the bounded incoming band; not an actual relaxation/mixing-time ratio or a kinetic closure. Born mean control leaves this ratio unchanged.',
    cases=kick_ratios,fine_four_replica_and_partition_range=[min(fine_ratio_values),max(fine_ratio_values)],
    empirical_relative_changes=ratio_changes,
    narrow_same_B2_reference=dict(source='results/physical_final_campaign/cases.json',source_sha256=sha(old_path),case='B2_final',bounds=old['bounds'],field=old['field'],orders=old['orders'],ratio=old_ratio,
        scope='Same B2 and impact band with narrower parallel/perpendicular speed bands; this is a domain comparison, not a stronger-field comparison.')),
 source_sha256=sha(Path(__file__)),input_sha256={p.name:sha(p) for p in (D/'summary.json',D/'states.npz',D/'populations.npz')},
 reconstructed_gates={k:bool(v) for k,v in gates.items()},reconstructed_comparisons={k:v.tolist() for k,v in cmp.items()})
OUTPUT.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(status=receipt['status'],checks=len(checks),failed=[r for r in checks if not r['passed']]),indent=2))
