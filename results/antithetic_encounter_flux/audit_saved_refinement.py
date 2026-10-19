"""Quantify an independently observed coarse endpoint failure using saved refinements."""
from pathlib import Path
import hashlib
import json
import os
import sys
import subprocess
import numpy as np
ROOT=Path(os.environ.get('SM_AUDIT_ROOT',Path(__file__).resolve().parents[2]))
D=ROOT/os.environ.get('SM_ANTITHETIC_AUDIT_INPUT','results/antithetic_encounter_flux')
OUT=Path(os.environ.get('SM_ANTITHETIC_REFINEMENT_AUDIT_OUTPUT',D/'saved_refinement_audit.json'))
if OUT.exists():raise FileExistsError(OUT)
with np.load(D/'states.npz') as f:a={k:f[k] for k in f.files}
with np.load(D/'populations.npz') as f:p={k:f[k] for k in f.files}
s=json.loads((D/'summary.json').read_text());cart=json.loads((D/'cartesian_audit.json').read_text())
m,B,T=(s['inputs'][k] for k in ('mass','field','theta'));omega=s['inputs']['charge']*B/m;mu=m*T/B
nbase=s['results']['base_state_count'];rows=[]
def values(i):
    wi=a['initial_relative_velocity'][i];wo=a['final_relative_velocity'][i];phi=omega*a['time'][i]
    delta=np.array([np.cos(phi)*wo[0]-np.sin(phi)*wo[1],np.sin(phi)*wo[0]+np.cos(phi)*wo[1]])-wi[:2]
    c=m*(wo[:2]@wo[:2]-wi[:2]@wi[:2])/(8*B);v=m*m*T*(delta@delta)/(8*B*B)
    def gc(r,w):return r[:2]+np.array([w[1],-w[0]])/omega
    kick=gc(a['final_relative_position'][i],wo)-gc(a['initial_relative_position'][i],wi)
    return np.array([c,c*c+v,c*c-v,kick@kick/4])
for replay in cart['replays']:
    if replay['passed']:continue
    coarse=replay['index'];fine=replay['finest_available_control_comparison']['index']
    x,y=values(fine-nbase),values(coarse-nbase);scale=np.array([x[1]/mu,x[1],x[1],x[3]])
    effects={}
    for case in s['results']['cases']:
        label=case['case'];ids=p[label+'_partner'];selected=ids==coarse
        if np.any(selected):
            # Exactly one physical partner's incoming mass; the other member is unchanged.
            shift=np.sum(p[label+'_pair_weights'][selected])/2*(x-y)
            v=np.asarray(case['integrals']);norm=np.array([v[1]/mu,v[1],v[1],v[3]])
            effects[label]=dict(hypothetical_integral_shift=shift.tolist(),positive_scaled_absolute_change=(abs(shift)/norm).tolist())
    rows.append(dict(coarse_global_index=coarse,refined_global_index=fine,coarse_step=float(a['step'][coarse-nbase]),refined_step=float(a['step'][fine-nbase]),
        original_cartesian_passed=False,refined_cartesian_passed=replay['finest_available_control_comparison']['passed'],
        refined_moments=x.tolist(),coarse_moments=y.tolist(),positive_scaled_member_change=(abs(x-y)/scale).tolist(),hypothetical_case_replacements=effects))
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
receipt=dict(status='passed' if rows and all(r['refined_cartesian_passed'] for r in rows) else 'failed',
    scope='Separate saved-refinement receipt; the original coarse Cartesian failure remains unchanged. Hypothetical single-member replacement is an error-size diagnostic, not a replacement of production results or a whole-domain trajectory bound.',
    normalization='Absolute moment change divided by [I2/mu_reference,I2,I2,I_R], using positive individual refined moments or original case integrals as appropriate.',
    original_cartesian_status=cart['status'],refinements=rows,
    maximum_hypothetical_scaled_case_change=max(v for r in rows for e in r['hypothetical_case_replacements'].values() for v in e['positive_scaled_absolute_change']),
    input_sha256={f.name:sha(f) for f in (D/'summary.json',D/'states.npz',D/'populations.npz',D/'cartesian_audit.json')},
    source_sha256=sha(Path(__file__)),source_uncommitted=bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--',str(Path(__file__).relative_to(ROOT))],text=True).strip()))
OUT.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
sys.exit(0 if receipt['status']=='passed' else 1)
