"""Independent twelve-coordinate replay of declared fresh encounter controls."""
from pathlib import Path
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
from scipy.integrate import solve_ivp

ROOT=Path(os.environ.get('SM_AUDIT_ROOT',Path(__file__).resolve().parents[2]))
D=ROOT/os.environ.get('SM_STRATIFIED_AUDIT_INPUT','results/stratified_encounter_flux')
OUTPUT=Path(os.environ.get('SM_STRATIFIED_CARTESIAN_OUTPUT',D/'cartesian_audit.json'))
if OUTPUT.exists():raise FileExistsError(f'Preserve existing audit receipt: {OUTPUT}')
audit_started=perf_counter()
j=json.loads((D/'summary.json').read_text());inp=j['inputs']
with np.load(D/'states.npz') as loaded:a={key:loaded[key] for key in loaded.files}
with np.load(D/'populations.npz') as loaded:p={key:loaded[key] for key in loaded.files}
m,q,B,kappa,screen=(inp[k] for k in ('mass','charge','field','strength','screening'))
omega=q*B/m
labels=[f'seed{seed}_N{inp["base_levels"][-1]}' for seed in inp['seeds']]+['alternate_partition']
ii=np.concatenate([p[label] for label in labels]);ww=np.concatenate([p[label+'_weights'] for label in labels])
reflected=ii[a['exit'][ii]=='reflected']
selected=[('largest_weighted_second',int(ii[np.argmax(ww*a['moments'][ii,1])])),
          ('closest_saved_separation',int(ii[np.argmin(a['min_separation'][ii])]))]
if len(reflected):selected.append(('longest_reflected',int(reflected[np.argmax(a['time'][reflected])])))
selected.append(('predetermined_primary_first_node',int(p[labels[0]][0])))
selection_groups={}
for reason,index in selected:selection_groups.setdefault(index,[]).append(reason)
receipts=[]
for index,reasons in selection_groups.items():
    reason=', '.join(reasons)
    started=perf_counter()
    ri,vi=a['initial_relative_position'][index],a['initial_relative_velocity'][index]
    initial=np.r_[ri/2,vi/2,-ri/2,-vi/2]
    distance=a['distance'][index]
    def rhs(t,y):
        r=y[:3]-y[6:9];radius=np.linalg.norm(r)
        acceleration=kappa*np.exp(-radius/screen)*(1+radius/screen)*r/(m*radius**3)
        v,w=y[3:6],y[9:12]
        return np.r_[v,q*B/m*np.cross(v,[0.,0.,1.])+acceleration,
                     w,q*B/m*np.cross(w,[0.,0.,1.])-acceleration]
    def plus(t,y):return y[2]-y[8]-distance
    def minus(t,y):return y[2]-y[8]+distance
    plus.terminal=True;plus.direction=1
    minus.terminal=True;minus.direction=-1
    solved=solve_ivp(rhs,(0.,inp['flight_time_factor']*distance/a['nodes'][index,1]),initial,
        method='DOP853',rtol=2e-13,atol=2e-15,max_step=.0125,events=(plus,minus))
    row=dict(selection=reason,selections=reasons,index=index,node=a['nodes'][index].tolist(),
        initial_state=initial.tolist(),rtol=2e-13,atol=2e-15,max_step=.0125,
        solver_success=solved.success,solver_status=solved.status,evaluations=solved.nfev,
        wall_s=perf_counter()-started)
    if solved.success and solved.status==1:
        y=solved.y[:,-1];r=y[:3]-y[6:9];v=y[3:6]-y[9:12]
        rr=solved.y[:3]-solved.y[6:9];vv=solved.y[3:6]-solved.y[9:12]
        rad=np.linalg.norm(rr,axis=0)
        E=m/4*np.sum(vv**2,axis=0)+kappa*np.exp(-rad/screen)/rad
        err=float(np.max(abs(E/E[0]-1)))
        dr=float(np.max(abs(r-a['final_relative_position'][index])))
        dv=float(np.max(abs(v-a['final_relative_velocity'][index])))
        dt=float(abs(solved.t[-1]-a['time'][index]))
        exit='transmitted' if len(solved.t_events[0]) else 'reflected'
        row.update(final_position=r.tolist(),final_velocity=v.tolist(),time=float(solved.t[-1]),exit=exit,
                   position_error=dr,velocity_error=dv,time_error=dt,energy_error=err,
                   passed=max(dr,dv,dt)<=1e-8 and err<=1e-9 and exit==a['exit'][index])
        # Compare any pre-existing smaller-step solves without replacing base failure.
        matches=np.where(np.all(a['nodes']==a['nodes'][index],axis=1)&(a['distance']==distance))[0]
        row['stored_step_comparisons']=[dict(index=int(z),step=float(a['step'][z]),
            position_error=float(np.max(abs(r-a['final_relative_position'][z]))),
            velocity_error=float(np.max(abs(v-a['final_relative_velocity'][z]))),
            time_error=float(abs(solved.t[-1]-a['time'][z]))) for z in matches]
    else:row.update(passed=False,message=solved.message)
    receipts.append(row)
    print(reason,index,'passed=',row['passed'],'wall_s=',row['wall_s'],flush=True)
producer_identities={}
for name,wanted in j['experiment_dependency_sha256'].items():
    blob=subprocess.run(['git','-C',str(ROOT),'show',j['commit']+':'+name],capture_output=True,check=True).stdout
    actual=hashlib.sha256(blob).hexdigest()
    producer_identities[name]=dict(commit=j['commit'],expected_sha256=wanted,actual_sha256=actual)
identity_pass=all(row['expected_sha256']==row['actual_sha256'] for row in producer_identities.values())
script=Path(__file__).resolve();script_relative=str(script.relative_to(ROOT))
OUTPUT.write_text(json.dumps(dict(status='passed' if all(r['passed'] for r in receipts) and identity_pass else 'failed',
    scope='Fresh twelve-coordinate solves with exact saved incoming bits; each original coarse comparison retained.',
    targets=dict(position=1e-8,velocity=1e-8,time=1e-8,energy=1e-9),
    producer_commit=j['commit'],producer_source_identities=producer_identities,producer_source_integrity=identity_pass,
    versions=dict(numpy=version('numpy'),scipy=version('scipy'),python=sys.version.split()[0]),platform=platform.platform(),
    wall_s=perf_counter()-audit_started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
    audit_git_identity=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
    source_path=script_relative,source_uncommitted=bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--',script_relative],text=True).strip()),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    input_sha256={name:hashlib.sha256((D/name).read_bytes()).hexdigest() for name in ('summary.json','states.npz','populations.npz')},
    replays=receipts),indent=2)+'\n')
