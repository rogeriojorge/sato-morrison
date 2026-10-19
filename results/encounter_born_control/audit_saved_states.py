"""Independent Cartesian force-work reconstruction of saved Born controls."""
from pathlib import Path
from itertools import combinations
from time import perf_counter
import hashlib
import json
import os
import subprocess
import platform
from importlib.metadata import version
import resource
import sys
import numpy as np
from scipy.special import roots_legendre

ROOT=Path(os.environ.get('SM_AUDIT_ROOT',Path(__file__).resolve().parents[2]))
D=ROOT/os.environ.get('SM_BORN_AUDIT_INPUT','results/encounter_born_control')
P=ROOT/'results/stratified_encounter_flux'
OUTPUT=Path(os.environ.get('SM_BORN_AUDIT_OUTPUT',D/'independent_audit.json'))
if OUTPUT.exists():raise FileExistsError(OUTPUT)
started=perf_counter()
j=json.loads((D/'summary.json').read_text());inp=j['inputs'];res=j['results']
raw=json.loads((P/'summary.json').read_text())
with np.load(P/'states.npz') as loaded:a={k:loaded[k] for k in loaded.files}
with np.load(P/'populations.npz') as loaded:p={k:loaded[k] for k in loaded.files}
m,q,B,kappa,screen,T=(inp[k] for k in ('mass','charge','field','strength','screening','theta'))
omega=q*B/m;mu=m*T/B;checks=[]
def check(name,ok,error=None):
    r=dict(name=name,passed=bool(ok))
    if error is not None:r['maximum_absolute_error']=float(error)
    checks.append(r)
def close(name,x,y,atol=2e-14,rtol=3e-11):
    x,y=np.asarray(x),np.asarray(y)
    check(name,x.shape==y.shape and np.allclose(x,y,atol=atol,rtol=rtol),np.max(abs(x-y)))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def force_work(nodes,order):
    z,w=roots_legendre(order);z*=inp['control_distance'];w*=inp['control_distance']
    out=np.zeros(len(nodes))
    for start in range(0,len(nodes),256):
        ids=np.arange(start,min(start+256,len(nodes)))
        ids=ids[nodes[ids,0]-nodes[ids,2]/omega>=inp['min_free_orbit_gap']]
        if not len(ids):continue
        b,u,v,phase=nodes[ids].T
        angle=phase[:,None]-omega*z[None,:]/u[:,None]
        vx=v[:,None]*np.cos(angle);vy=v[:,None]*np.sin(angle)
        x=b[:,None]-vy/omega;y=vx/omega
        radius=np.sqrt(x*x+y*y+z[None,:]*z[None,:])
        f=kappa*np.exp(-radius/screen)*(1+radius/screen)/radius**3
        work=f*x*vx+f*y*vy
        out[ids]=inp['coefficient']*(work@w)/(2*B*u)
    return out

unique,inverse=np.unique(a['nodes'],axis=0,return_inverse=True)
controls={};symmetry={}
probe=unique[np.linspace(0,len(unique)-1,128,dtype=int)].copy()
opposite=probe.copy();opposite[:,3]=(np.pi-probe[:,3])%(2*np.pi)
for order in inp['quadrature_orders']:
    defined=force_work(unique,order)[inverse]
    saved=np.load(D/f'control_GL{order}.npy')
    close(f'GL{order} independent Cartesian force-work definition',defined,saved,atol=3e-14,rtol=3e-11)
    controls[order]=defined
    residual=float(np.max(abs(force_work(probe,order)+force_work(opposite,order))))
    symmetry[str(order)]=residual
    check(f'GL{order} antithetic symmetry',residual<=inp['symmetry_absolute_target'],residual)
    close(f'GL{order} saved symmetry diagnostic',residual,res['control_symmetry_residual'][str(order)],atol=1e-13)

case_sets={};comp_sets={};variance_rows=[]
for order,cv in controls.items():
    cases={}
    recorded=res['cases'] if order==inp['quadrature_orders'][0] else res['sensitivity_cases']
    for row in recorded:
        label=row['case'];ids=p[label];weights=p[label+'_weights']
        values=np.column_stack((a['moments'][ids],a['individual_gc_second'][ids]))
        values[:,0]-=cv[ids]
        v=weights@values;cases[label]=v
        close(f'GL{order} {label} corrected integral',v,row['integrals'])
        close(f'GL{order} {label} control weighted sum',weights@cv[ids],row['control_integral'])
        close(f'GL{order} {label} raw receipt retained',row['raw_integrals'],next(r['integrals'] for r in raw['results']['cases'] if r['case']==label),atol=0,rtol=0)
        close(f'GL{order} {label} nonmean moments unchanged',v[1:],row['raw_integrals'][1:])
    case_sets[order]=cases
    def change(x,y):
        v,w=cases[x],cases[y]
        return abs(v-w)/np.array([v[1]/mu,v[1],v[1],v[3]])
    cmp={};levels=raw['inputs']['base_levels'];seeds=raw['inputs']['seeds']
    for seed in seeds:cmp[f'sample_{seed}']=change(f'seed{seed}_N{levels[-1]}',f'seed{seed}_N{levels[-2]}')
    for x,y in combinations(seeds,2):cmp[f'replica_{x}_{y}']=change(f'seed{x}_N{levels[-1]}',f'seed{y}_N{levels[-1]}')
    for seed in seeds:cmp[f'partition_{seed}']=change(f'seed{seed}_N{levels[-1]}','alternate_partition')
    cmp['boundary_ensemble']=change('boundary_L48','boundary_L40')
    cmp['boundary_L24_to40']=change('boundary_L40',f'seed{seeds[0]}_N{levels[-1]}')
    ids=raw['results']['controls']['ids']
    for label,x,y in (('active_step','0.025','0.05'),('boundary_controls','L48','L40')):
        v=np.column_stack((a['moments'][ids[x]],a['individual_gc_second'][ids[x]]))
        w=np.column_stack((a['moments'][ids[y]],a['individual_gc_second'][ids[y]]))
        v[:,0]-=cv[ids[x]];w[:,0]-=cv[ids[y]]
        cmp[label]=np.max(abs(v-w)/np.column_stack((v[:,1]/mu,v[:,1],v[:,1],v[:,3])),axis=0)
    recorded_cmp=res['comparisons'] if order==inp['quadrature_orders'][0] else res['sensitivity_comparisons']
    for key,v in cmp.items():close(f'GL{order} {key} comparison',v,recorded_cmp[key],atol=2e-7,rtol=2e-7)
    comp_sets[order]=cmp

lo,hi=inp['quadrature_orders'];stability={}
for row in res['variance_diagnostics']:
    label=row['case'];ids=p[label];weights=p[label+'_weights'];strata=p[label+'_stratum']
    for cell in row['strata']:
        mask=strata==cell['stratum'];v=a['moments'][ids[mask],0];cv=v-controls[lo][ids[mask]]
        before=np.var(v,ddof=1);after=np.var(cv,ddof=1);mass=np.sum(weights[mask]);n=np.sum(mask)
        close(label+f' stratum{cell["stratum"]} descriptive variances',
              [before,after,after/before,mass*mass*before/n,mass*mass*after/n],
              [cell['raw_mean_variance'],cell['controlled_mean_variance'],cell['variance_ratio'],cell['iid_weighted_variance_proxy_raw'],cell['iid_weighted_variance_proxy_controlled']],atol=2e-12)
fine_labels=[f'seed{seed}_N{raw["inputs"]["base_levels"][-1]}' for seed in raw['inputs']['seeds']]
raw_spread=np.std([next(r['integrals'][0] for r in raw['results']['cases'] if r['case']==label) for label in fine_labels],ddof=1)
cv_spread=np.std([case_sets[lo][label][0] for label in fine_labels],ddof=1)
close('descriptive four-replica signed-mean spread',[raw_spread,cv_spread,cv_spread/raw_spread],
      [res['replica_spread']['raw_signed_mean_sample_std'],res['replica_spread']['controlled_signed_mean_sample_std'],res['replica_spread']['ratio']],atol=2e-12)
for label in case_sets[lo]:
    v,w=case_sets[lo][label],case_sets[hi][label]
    stability[label]=abs(v[0]-w[0])/(v[1]/mu)
    close(label+' control-definition sensitivity',stability[label],res['control_order_stability'][label],atol=2e-8)
cmp=comp_sets[lo]
gates={k:np.max(v)<=(inp['plane_target'] if k.startswith('boundary_') else inp['step_target'] if k=='active_step' else inp['quadrature_target']) for k,v in cmp.items()}
for key in ('complete','exact_flux_mass','energy','active_step_sizes'):gates[key]=raw['results']['required_gates'][key]
gates.update(control_order_stability=all(v<=inp['control_stability_target'] for v in stability.values()),
    control_symmetry=all(v<=inp['symmetry_absolute_target'] for v in symmetry.values()),
    finite_controls=all(np.all(np.isfinite(v)) for v in controls.values()),
    prior_source_integrity=raw['results']['required_gates']['source_integrity'],
    prior_prior_integrity=raw['results']['required_gates']['prior_integrity'],
    source_integrity=j['experiment_dependency_sha256']==j['experiment_dependency_sha256_end'],
    prior_integrity=all(sha(ROOT/k)==v for k,v in inp['prior_artifact_sha256'].items()),
    historical_pilot_integrity=all(sha(ROOT/k)==v for k,v in inp['historical_pilot_sha256'].items()))
check('all declared gate names reconstructed',set(gates)==set(res['required_gates']))
for k,v in gates.items():check(k+' decision reconstructed',bool(v)==res['required_gates'][k])
check('raw failed-gate receipt unchanged',res['raw_campaign_required_gates']==raw['results']['required_gates'] and res['raw_campaign_comparisons']==raw['results']['comparisons'] and res['raw_campaign_status']==raw['results']['status'])
check('overall producer status reconstructed',res['status']==('passed' if all(gates.values()) else 'unresolved'))
producer_identities={}
for name,wanted in j['experiment_dependency_sha256'].items():
    blob=subprocess.run(['git','-C',str(ROOT),'show',j['commit']+':'+name],capture_output=True,check=True).stdout
    actual=hashlib.sha256(blob).hexdigest()
    check('historical committed producer '+name,actual==wanted)
    producer_identities[name]=dict(commit=j['commit'],expected_sha256=wanted,actual_sha256=actual)
script=Path(__file__).resolve();script_relative=str(script.relative_to(ROOT))
receipt=dict(status='passed' if all(r['passed'] for r in checks) else 'failed',checks=checks,
    producer_commit=j['commit'],producer_source_identities=producer_identities,
    versions=dict(numpy=version('numpy'),scipy=version('scipy'),python=sys.version.split()[0]),platform=platform.platform(),
    audit_git_identity=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
    source_path=script_relative,source_uncommitted=bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--',script_relative],text=True).strip()),
    scope='Independent Cartesian force-work control calculation, all saved weighted integrals and fixed-gate reconstruction; no producer physics imports or new ODE solves.',
    formulas=dict(control='(coefficient/(2B*u)) integral F(r_free(z)) dot w_perp_free(z) dz; same fixed symmetric Gauss-Legendre rule as the declared control definition',
        force='kappa exp(-r/lambda)(1+r/lambda) r/r^3',zero_expectation='Each fixed symmetric phase cell is invariant under phi->pi-phi, while symmetric z quadrature makes the control odd.'),
    reconstructed_gates={k:bool(v) for k,v in gates.items()},wall_s=perf_counter()-started,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
    source_sha256=sha(Path(__file__)),input_sha256={str(f.relative_to(ROOT)):sha(f) for f in (D/'summary.json',D/'control_GL1024.npy',D/'control_GL2048.npy',P/'states.npz',P/'populations.npz')})
OUTPUT.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(status=receipt['status'],checks=len(checks),failed=[r for r in checks if not r['passed']]),indent=2))
sys.exit(0 if receipt['status']=='passed' else 1)
