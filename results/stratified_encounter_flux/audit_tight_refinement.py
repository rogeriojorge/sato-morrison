"""Fixed smaller-step relative replays preserving failed coarse comparisons."""
from pathlib import Path
from time import perf_counter
import hashlib
import json
import os
import subprocess
import numpy as np
from scipy.integrate import solve_ivp

ROOT=Path(__file__).resolve().parents[2]
D=ROOT/'results/stratified_encounter_flux'
OUTPUT=Path(os.environ.get('SM_STRATIFIED_TIGHT_OUTPUT',D/'cartesian_tight_refinement.json'))
if OUTPUT.exists():raise FileExistsError(OUTPUT)
started=perf_counter()
j=json.loads((D/'summary.json').read_text());inp=j['inputs']
original=json.loads((D/'cartesian_audit.json').read_text())
with np.load(D/'states.npz') as saved:a={key:saved[key] for key in saved.files}
with np.load(D/'populations.npz') as saved:p={key:saved[key] for key in saved.files}
m,q,B,kappa,screen,T=(inp[key] for key in ('mass','charge','field','strength','screening','theta'))
omega=q*B/m;mu=m*T/B
targets=dict(position=1e-8,velocity=1e-8,time=1e-8,energy=1e-9,moments=.001)
levels=((.025,1e-12),(.0125,2e-13))
rows=[];replacement={}
def observable(r,v,t,ri,vi):
    angle=omega*t
    unrotated=np.array([np.cos(angle)*v[0]-np.sin(angle)*v[1],np.sin(angle)*v[0]+np.cos(angle)*v[1]])
    delta=unrotated-vi[:2]
    c=m*(v[:2]@v[:2]-vi[:2]@vi[:2])/(8*B)
    variance=m*m*T*(delta@delta)/(8*B*B)
    kick=r[:2]-ri[:2]+np.array([v[1]-vi[1],-v[0]+vi[0]])/omega
    return np.array([c,c*c+variance,c*c-variance,kick@kick/4])
for coarse in original['replays']:
    if coarse['passed']:continue
    index=coarse['index'];ri=a['initial_relative_position'][index];vi=a['initial_relative_velocity'][index]
    initial=np.r_[ri,vi];L=a['distance'][index];replays=[]
    for step,rtol in levels:
        def rhs(t,y):
            r=y[:3];v=y[3:];radius=np.linalg.norm(r)
            f=2*kappa/m*np.exp(-radius/screen)*(1+radius/screen)/radius**3
            return np.r_[v,omega*np.cross(v,[0.,0.,1.])+f*r]
        def plus(t,y):return y[2]-L
        def minus(t,y):return y[2]+L
        plus.terminal=True;plus.direction=1
        minus.terminal=True;minus.direction=-1
        solved=solve_ivp(rhs,(0.,inp['flight_time_factor']*L/a['nodes'][index,1]),initial,
            method='DOP853',rtol=rtol,atol=rtol*.01,max_step=step,events=(plus,minus))
        if not solved.success or solved.status!=1:raise RuntimeError(solved.message)
        r,v=solved.y[:3,-1],solved.y[3:,-1];t=float(solved.t[-1])
        radius=np.linalg.norm(solved.y[:3],axis=0)
        E=m*np.sum(solved.y[3:]**2,axis=0)/4+kappa*np.exp(-radius/screen)/radius
        energy=float(np.max(abs(E/E[0]-1)))
        dr=float(np.max(abs(r-np.array(coarse['final_position']))));dv=float(np.max(abs(v-np.array(coarse['final_velocity']))));dt=abs(t-coarse['time'])
        exit='transmitted' if len(solved.t_events[0]) else 'reflected'
        value=observable(r,v,t,ri,vi)
        replays.append(dict(step=step,rtol=rtol,atol=rtol*.01,initial_state=initial.tolist(),
            final_position=r.tolist(),final_velocity=v.tolist(),time=t,exit=exit,energy_error=energy,
            cartesian_position_error=dr,cartesian_velocity_error=dv,cartesian_time_error=dt,
            passed=max(dr,dv,dt)<=1e-8 and energy<=1e-9 and exit==coarse['exit'],
            observables=value.tolist(),max_actual_step=float(np.max(np.diff(solved.t))),evaluations=solved.nfev))
    x,y=(np.array(row['observables']) for row in replays)
    scaled=abs(y-x)/np.array([y[1]/mu,y[1],y[1],y[3]])
    original_values=np.r_[a['moments'][index],a['individual_gc_second'][index]]
    replacement[index]=y-original_values
    row=dict(index=index,node=a['nodes'][index].tolist(),original_failed_comparison=coarse,
             replays=replays,scaled_tight_level_change=scaled.tolist(),
             active_maximum_step_ratio=replays[1]['max_actual_step']/replays[0]['max_actual_step'])
    rows.append(row)
    print(index,[r['passed'] for r in replays],flush=True)
effects={}
for case in j['results']['cases']:
    label=case['case'];ids=p[label];weights=p[label+'_weights'];delta=np.zeros(4)
    for index,change in replacement.items():delta+=np.sum(weights[ids==index])*change
    fine=np.asarray(case['integrals'])+delta
    effects[label]=dict(integral_change=delta.tolist(),scaled_integral_change=(abs(delta)/np.array([fine[1]/mu,fine[1],fine[1],fine[3]])).tolist())
script=Path(__file__).resolve();relative=str(script.relative_to(ROOT))
passed=bool(rows) and all(all(r['passed'] for r in row['replays']) and max(row['scaled_tight_level_change'])<=.001 and row['active_maximum_step_ratio']<=.6 for row in rows)
receipt=dict(status='passed' if passed else 'failed',
    scope='Selected fixed smaller-step follow-up only. The original coarse Cartesian failures are retained verbatim and remain failed. No ensemble probabilities or original results are replaced.',
    levels=levels,targets=targets,replays=rows,hypothetical_two_state_replacement_effects=effects,
    wall_s=perf_counter()-started,input_sha256={name:hashlib.sha256((D/name).read_bytes()).hexdigest() for name in ('summary.json','states.npz','populations.npz','cartesian_audit.json')},
    source_path=relative,source_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
    audit_git_identity=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
    source_uncommitted=bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--',relative],text=True).strip()))
OUTPUT.write_text(json.dumps(receipt,indent=2)+'\n')
