"""Independent bin-index reconstruction of physical member contribution maps."""
from pathlib import Path
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
from time import perf_counter
import numpy as np
ROOT=Path(os.environ.get('SM_AUDIT_ROOT',Path(__file__).resolve().parents[2]))
D=ROOT/os.environ.get('SM_ANTITHETIC_AUDIT_INPUT','results/antithetic_encounter_flux')
OUT=Path(os.environ.get('SM_ANTITHETIC_MAP_AUDIT_OUTPUT',D/'contribution_map_audit.json'))
if OUT.exists():raise FileExistsError(OUT)
started=perf_counter()
s=json.loads((D/'summary.json').read_text());inp=s['inputs']
with np.load(ROOT/'results/stratified_encounter_flux/states.npz') as f:b={k:f[k] for k in ('nodes','initial_relative_velocity','final_relative_velocity','time')}
with np.load(D/'states.npz') as f:a={k:np.concatenate((b[k],f[k])) for k in b}
with np.load(D/'populations.npz') as f:p={k:f[k] for k in f.files}
with np.load(D/'contribution_maps.npz') as f:maps={k:f[k] for k in f.files}
m,B,T=(inp[k] for k in ('mass','field','theta'));omega=inp['charge']*B/m
wi,wo=a['initial_relative_velocity'],a['final_relative_velocity'];phi=omega*a['time']
delta=np.column_stack((np.cos(phi)*wo[:,0]-np.sin(phi)*wo[:,1],np.sin(phi)*wo[:,0]+np.cos(phi)*wo[:,1]))-wi[:,:2]
c=m*(np.sum(wo[:,:2]**2,axis=1)-np.sum(wi[:,:2]**2,axis=1))/(8*B)
second=c*c+m*m*T*np.sum(delta**2,axis=1)/(8*B*B)
checks=[];cases=[];hb=np.zeros((32,32));hp=np.zeros((48,32))
for seed in inp['seeds']:
    label=f'seed{seed}_N{inp["base_levels"][-1]}';ids=np.r_[p[label+'_original'],p[label+'_partner']]
    w=np.tile(p[label+'_pair_weights']/2,2);nodes=a['nodes'][ids];v=w*second[ids]
    ratio=nodes[:,2]/(abs(omega)*nodes[:,0]);h=[]
    for x,xe in ((nodes[:,0],maps['b_edges']),(nodes[:,3]%(2*np.pi),maps['phase_edges'])):
        ix=np.searchsorted(xe,x,side='right')-1;iy=np.searchsorted(maps['ratio_edges'],ratio,side='right')-1
        ix=np.minimum(ix,len(xe)-2);iy=np.minimum(iy,len(maps['ratio_edges'])-2)
        valid=(ix>=0)&(iy>=0)&(ix<len(xe)-1)&(iy<len(maps['ratio_edges'])-1)
        bins=np.zeros((len(xe)-1,len(maps['ratio_edges'])-1));np.add.at(bins,(ix[valid],iy[valid]),v[valid]);h.append(bins)
        checks.append(bool(np.all(valid) and np.isclose(bins.sum(),v.sum(),rtol=2e-12,atol=0)))
    hb+=h[0]/4;hp+=h[1]/4
    case=next(z for z in s['results']['cases'] if z['case']==label)
    checks.append(bool(np.isclose(v.sum(),case['integrals'][1],rtol=2e-12,atol=0)))
    cases.append(dict(case=label,second_integral=float(v.sum()),
        rho_over_b_at_least_point8_incoming_flux_fraction=float(w[ratio>=.8].sum()/w.sum()),
        rho_over_b_at_least_point8_second_share=float(v[ratio>=.8].sum()/v.sum())))
errors={k:float(np.max(abs(x-maps[k]))) for k,x in (('b_ratio_mean_second',hb),('phase_ratio_mean_second',hp))}
checks.extend(np.allclose(x,maps[k],rtol=2e-12,atol=1e-20) for k,x in (('b_ratio_mean_second',hb),('phase_ratio_mean_second',hp)))
mean=np.mean([r['second_integral'] for r in cases]);checks.append(bool(np.isclose(mean,maps['arithmetic_mean_second'],rtol=2e-12,atol=0)))
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
receipt=dict(status='passed' if all(checks) else 'failed',scope='Independent endpoint reconstruction of second moments and explicit integer-bin accumulation, with actual original and partner locations and positive halfweights. Fixed rho/b threshold .8 is descriptive concentration, not a convergence gate.',
    checks_passed=sum(checks),check_count=len(checks),maximum_bin_absolute_errors=errors,cases=cases,
    input_sha256={f.name:sha(f) for f in (D/'summary.json',D/'states.npz',D/'populations.npz',D/'contribution_maps.npz')},source_sha256=sha(Path(__file__)),source_uncommitted=bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--',str(Path(__file__).relative_to(ROOT))],text=True).strip()))
receipt.update(audit_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
    versions=dict(numpy=np.__version__,python=platform.python_version()),platform=platform.platform(),processor=s.get('processor'),
    wall_s=perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
OUT.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
sys.exit(0 if receipt['status']=='passed' else 1)
