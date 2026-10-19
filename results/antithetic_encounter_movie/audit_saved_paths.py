"""Independent saved physical path and movie association audit."""
from pathlib import Path
from time import perf_counter
from importlib.metadata import version
import hashlib
import json
import sys
import os
import subprocess
import numpy as np
from PIL import Image

ROOT=Path(os.environ.get('SM_AUDIT_ROOT',Path(__file__).resolve().parents[2]))
D=ROOT/os.environ.get('SM_PHASE_MOVIE_AUDIT_INPUT','results/antithetic_encounter_movie')
OUTPUT=Path(os.environ.get('SM_PHASE_MOVIE_AUDIT_OUTPUT',D/'independent_audit.json'))
if OUTPUT.exists():raise FileExistsError(OUTPUT)
started=perf_counter();j=json.loads((D/'summary.json').read_text());inp=j['inputs'];res=j['results']
with np.load(D/'paths.npz') as loaded:a={k:loaded[k] for k in loaded.files}
m,q,B,kappa,screen,L=(inp[k] for k in ('mass','charge','field','strength','screening','start_distance'))
omega=q*B/m;checks=[]
def check(name,ok,error=None):
    row=dict(name=name,passed=bool(ok))
    if error is not None:row['maximum_absolute_error']=float(error)
    checks.append(row)
def close(name,x,y,atol=2e-13,rtol=2e-12):
    x,y=np.asarray(x),np.asarray(y)
    check(name,x.shape==y.shape and np.allclose(x,y,atol=atol,rtol=rtol),np.max(abs(x-y)))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
close('fixed phase involution',inp['phases'][1],(np.pi-inp['phases'][0])%(2*np.pi),atol=0,rtol=0)
archive=ROOT/'results/encounter_peak_resolution/summary.json'
old=json.loads(archive.read_text())
close('fixed archived impact and speeds',[inp['impact'],inp['parallel_speed'],inp['perpendicular_speed']],old['inputs']['speed'],atol=0,rtol=0)
energy_errors={};canonical_errors={}
for record in res['records']:
    i=record['index'];t=a[f'time_{i}'];r=a[f'positions_{i}'];v=a[f'velocities_{i}']
    check(f'path{i} finite ordered stored samples',t[0]==0 and np.all(np.diff(t)>0) and np.all(np.isfinite(r)) and np.all(np.isfinite(v)))
    phase=inp['phases'][i]+omega*L/inp['parallel_speed']
    wi=np.array([inp['perpendicular_speed']*np.cos(phase),inp['perpendicular_speed']*np.sin(phase),inp['parallel_speed']])
    ri=np.array([inp['impact']-wi[1]/omega,wi[0]/omega,-L])
    close(f'path{i} exact physical incoming velocities',v[0],np.array([wi/2,-wi/2]),atol=2e-13)
    close(f'path{i} exact physical incoming positions',r[0],np.array([ri/2,-ri/2]),atol=2e-13)
    close(f'path{i} zero position COM',np.sum(r,axis=1),np.zeros((len(t),3)),atol=inp['endpoint_target'],rtol=0)
    close(f'path{i} zero velocity COM',np.sum(v,axis=1),np.zeros((len(t),3)),atol=inp['endpoint_target'],rtol=0)
    mu=m*np.sum(v[:,:,:2]**2,axis=2)/(2*B)
    close(f'path{i} stored individual moment',mu[:,0],a[f'mu_{i}'],atol=2e-14)
    close(f'path{i} symmetric zero-COM individual moments',mu[:,0],mu[:,1],atol=2e-13)
    close(f'path{i} recorded moment values',[mu[0,0],mu[-1,0],mu[-1,0]-mu[0,0]],[record['initial_mu'],record['final_mu'],record['delta_mu']],atol=2e-14)
    rel=r[:,0]-r[:,1];vel=v[:,0]-v[:,1];radius=np.linalg.norm(rel,axis=1)
    E=m*np.sum(v*v,axis=(1,2))/2+kappa*np.exp(-radius/screen)/radius
    ee=float(np.max(abs(E/E[0]-1)));energy_errors[str(i)]=ee
    check(f'path{i} Cartesian energy target',ee<=inp['energy_target'],ee)
    close(f'path{i} saved Cartesian energy diagnostic',ee,record['cartesian_energy_error'],atol=3e-14)
    J=rel[:,0]*vel[:,1]-rel[:,1]*vel[:,0]+omega*np.sum(rel[:,:2]**2,axis=1)/2
    scale=abs(omega)*inp['impact']**2/2+inp['perpendicular_speed']**2/(2*abs(omega))
    ce=float(np.max(abs(J-J[0]))/scale);canonical_errors[str(i)]=ce
    check(f'path{i} canonical angular momentum target',ce<=1e-9,ce)
    close(f'path{i} actual integration stop time',t[-1],record['exit_time'],atol=0,rtol=0)
    check(f'path{i} outgoing plane and direction',abs(abs(rel[-1,2])-L)<=inp['endpoint_target'] and np.sign(vel[-1,2])==np.sign(rel[-1,2]) and (record['exit']=='reflected')==(rel[-1,2]<0))
    check(f'path{i} closest saved sample',record['closest_sample_index']==int(np.argmin(radius)))
    close(f'path{i} closest sample quantities',[radius.min(),t[np.argmin(radius)]],[record['sampled_minimum_separation'],record['closest_sample_time']],atol=2e-13)
    check(f'path{i} original trajectory checks retained',all(record['checks'].values()))
frames=np.asarray(res['frame_times'])
end=max(r['exit_time'] for r in res['records'])
low=max(0,min(r['closest_sample_time'] for r in res['records'])-.5)
high=min(end,max(r['closest_sample_time'] for r in res['records'])+.5)
expected=np.r_[np.linspace(0,low,18,endpoint=False),np.linspace(low,high,84,endpoint=False),np.linspace(high,end,18)]
close('shared physical movie clock',frames,expected,atol=0,rtol=0)
check('no movie clock beyond final integration',frames[0]==0 and frames[-1]==end and np.all(np.diff(frames)>0))
check('canvas frame hashes cover declared clock',set(res['canvas_rgba_sha256'])==set(map(str,range(inp['frames']))))
decoded=[]
with Image.open(D/'phase_pair.gif') as gif:
    for i in range(gif.n_frames):
        gif.seek(i);decoded.append(hashlib.sha256(gif.convert('RGBA').tobytes()).hexdigest())
check('decoded GIF frames match recorded bytes',decoded==res['gif_decoded_frame_sha256'] and len(decoded)==res['encoded_frames'])
for name,wanted in j['output_sha256'].items():check('saved output hash '+name,sha(ROOT/name)==wanted)
producer={}
for name,wanted in j['experiment_dependency_sha256'].items():
    blob=subprocess.run(['git','-C',str(ROOT),'show',j['commit']+':'+name],capture_output=True,check=True).stdout
    actual=hashlib.sha256(blob).hexdigest();check('historical committed producer '+name,actual==wanted)
    producer[name]=dict(commit=j['commit'],expected_sha256=wanted,actual_sha256=actual)
check('producer unchanged while integrating',j['experiment_dependency_sha256']==j['experiment_dependency_sha256_end'])
check('producer physical outcome status',res['status']=='passed' and not res['failures'])
script=Path(__file__).resolve();relative=str(script.relative_to(ROOT))
receipt=dict(status='passed' if all(c['passed'] for c in checks) else 'failed',checks=checks,
    scope='Saved physical zero-COM Cartesian arrays and media-clock/hash associations only. No fresh ODE, thermal COM averaging, ensemble convergence or physical rate claim.',
    energy_errors=energy_errors,canonical_scaled_errors=canonical_errors,
    producer_commit=j['commit'],producer_source_identities=producer,source_path=relative,source_sha256=sha(script),
    audit_git_identity=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
    source_uncommitted=bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--',relative],text=True).strip()),
    input_sha256={str(f.relative_to(ROOT)):sha(f) for f in (D/'summary.json',D/'paths.npz',D/'phase_pair.gif',archive)},
    versions=dict(numpy=version('numpy'),pillow=version('Pillow')),wall_s=perf_counter()-started)
OUTPUT.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(status=receipt['status'],checks=len(checks),failed=[c for c in checks if not c['passed']]),indent=2))
sys.exit(0 if receipt['status']=='passed' else 1)
