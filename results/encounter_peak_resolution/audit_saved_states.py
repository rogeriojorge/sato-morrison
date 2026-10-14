"""Saved-array audit of the fixed peak diagnosis; no producer imports or solves."""
from pathlib import Path
from time import perf_counter
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import numpy as np

ROOT = Path(os.environ.get('SM_AUDIT_ROOT', Path(__file__).resolve().parents[2])).resolve()
INPUT = ROOT/os.environ.get('SM_PEAK_AUDIT_INPUT', 'results/encounter_peak_resolution')
OUTPUT = ROOT/os.environ.get('SM_PEAK_AUDIT_OUTPUT', 'results/encounter_peak_resolution/independent_audit.json')
if OUTPUT.exists():
    raise FileExistsError(f'Preserve existing receipt: {OUTPUT}')
started = perf_counter()
summary = json.loads((INPUT/'summary.json').read_text())
inputs, results = summary['inputs'], summary['results']
states = np.load(INPUT/'states.npz')
populations = np.load(INPUT/'populations.npz')
m, q, B, theta = [inputs[k] for k in ('mass', 'charge', 'field', 'theta')]
omega, mu = q*B/m, m*theta/B
checks = []

def check(name, passed, error=None):
    row = dict(name=name, passed=bool(passed))
    if error is not None:
        row['maximum_absolute_error'] = float(error) if np.isfinite(error) else 'nonfinite'
    checks.append(row)

def close(name, a, b, atol=3e-10, rtol=1e-11):
    a, b = np.asarray(a), np.asarray(b)
    check(name, a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=rtol), np.max(abs(a-b)))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

print(f'Independent saved-array audit: {INPUT}', flush=True)
N = len(states['nodes'])
check('all stored rows completed', np.all(states['status'] == 'passed'))
check('finite Cartesian states', all(np.all(np.isfinite(states[k])) for k in
    ('nodes','initial_relative_position','final_relative_position','initial_relative_velocity','final_relative_velocity','time')))
wi, wf = states['initial_relative_velocity'], states['final_relative_velocity']
angles = omega*states['time']
# Independent Cartesian rotation matrix, R(+Omega*t), undoes free Lorentz motion.
rotation = np.empty((N, 2, 2))
rotation[:, 0, 0] = rotation[:, 1, 1] = np.cos(angles)
rotation[:, 0, 1] = -np.sin(angles)
rotation[:, 1, 0] = np.sin(angles)
delta = np.einsum('nij,nj->ni', rotation, wf[:, :2])-wi[:, :2]
c = m*(np.sum(wf[:, :2]**2, axis=1)-np.sum(wi[:, :2]**2, axis=1))/(8*B)
variance = m*m*theta*np.sum(delta*delta, axis=1)/(8*B*B)
moments = np.column_stack((c, c*c+variance, c*c-variance))
# GC coordinates are evaluated at actual endpoints, not the unrotated velocities.
initial_gc = states['initial_relative_position'][:, :2]+np.column_stack((wi[:, 1], -wi[:, 0]))/omega
final_gc = states['final_relative_position'][:, :2]+np.column_stack((wf[:, 1], -wf[:, 0]))/omega
gc_kick = final_gc-initial_gc
spatial = np.sum(gc_kick**2, axis=1)/4
values = np.column_stack((moments, spatial))
scale = np.column_stack((np.sqrt(moments[:, 1]), moments[:, 1], moments[:, 1]))
close('Cartesian COM moments / individual signal scales', moments/scale, states['moments']/scale, atol=3e-7)
close('free-gyro velocity kick', delta, states['delta_relative_perpendicular'], atol=2e-12)
close('actual GC coordinate kick', gc_kick, states['gc_relative_kick'], atol=2e-12)
close('individual GC squared kick', spatial, states['individual_gc_second'], atol=2e-14)
check('nonnegative COM variance and second moment', np.all(variance>=0) and np.all(moments[:, 1]>0))
phase = states['nodes'][:, 3]+omega*states['distance']/states['nodes'][:, 1]
expected_velocity = np.column_stack((states['nodes'][:, 2]*np.cos(phase), states['nodes'][:, 2]*np.sin(phase), states['nodes'][:, 1]))
expected_position = np.column_stack((states['nodes'][:, 0]-expected_velocity[:, 1]/omega, expected_velocity[:, 0]/omega, -states['distance']))
close('same free incoming helix velocity', expected_velocity, wi, atol=2e-13)
close('same free incoming helix position', expected_position, states['initial_relative_position'], atol=2e-13)
check('exit planes and signs', np.allclose(abs(states['final_relative_position'][:, 2]), states['distance'], atol=2e-12, rtol=0)
    and np.all((states['exit']=='reflected') == (states['final_relative_position'][:, 2]<0))
    and np.all(np.sign(wf[:, 2]) == np.sign(states['final_relative_position'][:, 2])))

def energy(position, velocity):
    radius = np.linalg.norm(position, axis=-1)
    return m*np.sum(velocity*velocity, axis=-1)/4+inputs['strength']*np.exp(-radius/inputs['screening'])/radius
endpoint_energy = abs(energy(states['final_relative_position'], wf)/energy(states['initial_relative_position'], wi)-1)
check('endpoint energy bounded by saved path maximum', np.all(endpoint_energy <= states['energy_error']+3e-14))
check('energy acceptance', np.all(endpoint_energy<=inputs['energy_target']) and np.all(states['energy_error']<=inputs['energy_target']))

case_values = {}
case_records = {row['case']:row for row in results['cases']}
canonical = 2*np.pi*np.arange(inputs['union_count'])/inputs['union_count']
for label, row in case_records.items():
    ids = populations[label]
    check(label+' valid unique state selectors', len(ids)==row['phase_count'] and len(set(ids.tolist()))==len(ids) and np.all((ids>=0)&(ids<N)))
    expected_angles = canonical[1::2] if row['shifted'] else canonical[::inputs['union_count']//row['phase_count']]
    expected_nodes = np.column_stack((np.repeat(np.asarray(inputs['speed'])[None], len(ids), axis=0), expected_angles))
    close(label+' exact incoming angular grid', states['nodes'][ids], expected_nodes, atol=0, rtol=0)
    close(label+' matched endpoint and integration step', states['distance'][ids], np.full(len(ids),row['distance']), atol=0, rtol=0)
    close(label+' bulk maximum step', states['step'][ids], np.full(len(ids),inputs['base_step']), atol=0, rtol=0)
    check(label+' all original phase weights retained', row['complete'] and row['missing_count']==0 and np.all(states['status'][ids]=='passed'))
    v = values[ids].sum(axis=0)/len(ids)
    case_values[label] = v
    close(label+' uniform weighted average', v, row['observables'], atol=3e-13)
    check(label+' reflection count', np.count_nonzero(states['exit'][ids]=='reflected')==row['reflected_count'])
    close(label+' maximum integration time', states['time'][ids].max(), row['maximum_time'], atol=1e-12)
    close(label+' maximum path energy error', states['energy_error'][ids].max(), row['max_energy_error'], atol=1e-15)

close('regular and half-shift union average L24', case_values['phase4096_L24_union'],
      (case_values['phase2048_L24']+case_values['phase2048_L24_shifted'])/2, atol=1e-15)
close('regular and half-shift union average L48', case_values['phase4096_L48'],
      (case_values['phase2048_L48']+case_values['phase2048_L48_shifted'])/2, atol=1e-15)

def changes(a, b, reference=None):
    a,b=np.asarray(a),np.asarray(b)
    reference=a if reference is None else np.asarray(reference)
    return abs(a-b)/np.array([reference[1]/mu,reference[1],reference[1],reference[3]])

cv=case_values
computed = dict(phase1024_to2048=changes(cv['phase2048_L24'],cv['phase1024_L24'],cv['phase4096_L24_union']),
    phase2048_to4096=changes(cv['phase4096_L24_union'],cv['phase2048_L24']),
    halfshift=changes(cv['phase2048_L24'],cv['phase2048_L24_shifted'],cv['phase4096_L24_union']),
    final_plane_phase=changes(cv['phase4096_L48'],cv['phase2048_L48']),
    final_plane_halfshift=changes(cv['phase2048_L48'],cv['phase2048_L48_shifted'],cv['phase4096_L48']),
    boundary_integral=changes(cv['phase4096_L48'],cv['phase4096_L40']))
weak_values=[]
for row,distance in zip(results['weak_boundary'],inputs['distances']):
    ids=np.asarray(row['ids']); check(row['case']+' one retained state',len(ids)==1 and row['complete'])
    close(row['case']+' exact free incoming node',states['nodes'][ids[0]],inputs['weak_node'],atol=0,rtol=0)
    close(row['case']+' endpoint distance',states['distance'][ids[0]],distance,atol=0,rtol=0)
    weak_values.append(values[ids[0]])
    close(row['case']+' independent moments',weak_values[-1],row['observables'],atol=1e-14)
computed['boundary_weak_state']=changes(weak_values[-1],weak_values[-2])
control=results['controls'];control_ids=control['ids'];steps=inputs['steps']
for step in steps:
    ids=np.asarray(control_ids[str(step)])
    close(f'controls step{step} exact incoming nodes',states['nodes'][ids],control['nodes'],atol=0,rtol=0)
    close(f'controls step{step} exact maximum step',states['step'][ids],np.full(len(ids),step),atol=0,rtol=0)
    close(f'controls step{step} endpoint distance',states['distance'][ids],np.full(len(ids),24.),atol=0,rtol=0)
finest=np.asarray(control_ids[str(steps[-1])]);middle=np.asarray(control_ids[str(steps[-2])])
a,b=values[finest],values[middle]
computed['active_step']=np.max(abs(a-b)/np.column_stack((a[:,1]/mu,a[:,1],a[:,1],a[:,3])),axis=0)
ratios=states['max_actual_step'][finest]/states['max_actual_step'][middle]
close('independent active maximum-step ratios',ratios,control['actual_step_ratios'],atol=2e-14)
for key,v in computed.items(): close(key+' independently reconstructed comparison',v,results['comparisons'][key],atol=2e-8)

cartesian_good=[]
for row in results['cartesian']:
    i=row['relative_id'];close('Cartesian replay same input',states['nodes'][i],row['node'],atol=0,rtol=0)
    errors=np.array([max(abs(np.asarray(row['final_position'])-states['final_relative_position'][i])),max(abs(np.asarray(row['final_velocity'])-wf[i]))])
    close('Cartesian raw endpoint discrepancies',errors,[row['position_error'],row['velocity_error']],atol=0,rtol=0)
    cartesian_good.append(row['status']=='passed' and max(errors)<=inputs['cartesian_absolute_target'] and row['energy_error']<=inputs['energy_target'])
tight_good=[]
for row in results['tight_relative']:
    selected_index=next(j for j,n in enumerate(control['nodes']) if np.array_equal(n,row['node']));i=finest[selected_index]
    close('mandatory tight initial bits',row['initial_state'],np.r_[states['initial_relative_position'][i],wi[i]],atol=0,rtol=0)
    errors=np.array([max(abs(np.asarray(row['final_position'])-states['final_relative_position'][i])),max(abs(np.asarray(row['final_velocity'])-wf[i])),abs(row['time']-states['time'][i])])
    close('mandatory tight raw endpoint/time discrepancies',errors,[row['position_error'],row['velocity_error'],row['time_error']],atol=0,rtol=0)
    check('mandatory tight reflected first exit',row['exit']==states['exit'][i]=='reflected')
    tight_good.append(row['status']=='passed' and max(errors)<=inputs['cartesian_absolute_target'] and row['energy_error']<=inputs['energy_target'])

prior=ROOT/'results/broad_speed_extended';pj=json.loads((prior/'summary.json').read_text())
selection=next(p for p in pj['results']['phase_controls'] if p['seed']==inputs['seed'] and p['sample_index']==inputs['sample_index'])
with np.load(prior/'states.npz') as old:
    old_ids=selection['phase256_ids']
    for key in states.files:close('reused prior256 bits '+key,states[key][:len(old_ids)],old[key][old_ids],atol=0,rtol=0) if states[key].dtype.kind not in 'US' else check('reused prior256 bits '+key,np.array_equal(states[key][:len(old_ids)],old[key][old_ids]))
check('stored rows and unique cache keys',N==results['unique_encounters'] and len(set((tuple(n),float(d),float(st)) for n,d,st in zip(states['nodes'],states['distance'],states['step'])))==N)
check('reuse/new counts',results['reused_encounters']==len(old_ids)==256 and results['new_encounters']==N-256)
check('separate replay counts',results['separate_cartesian_solves']==len(results['cartesian']) and results['separate_tight_relative_solves']==len(results['tight_relative']))

producer_identities={}
for name,wanted in summary['experiment_dependency_sha256'].items():
    blob=subprocess.run(['git','-C',str(ROOT),'show',summary['commit']+':'+name],capture_output=True,check=True).stdout
    actual=hashlib.sha256(blob).hexdigest();check('historical committed producer '+name,actual==wanted)
    producer_identities[name]=dict(git_commit=summary['commit'],sha256=actual)
source_gate=summary['experiment_dependency_sha256']==summary['experiment_dependency_sha256_end']
prior_gate=all(sha(prior/name)==wanted for name,wanted in inputs['prior_artifact_sha256'].items())
original=ROOT/inputs['original_cartesian_audit']['path'];original_gate=sha(original)==inputs['original_cartesian_audit']['sha256']
original_record=json.loads(original.read_text())
original_failure=next(r for r in original_record['broad_speed']['selected_cartesian_replays'] if r['index']==inputs['required_reflected_prior_state_id'])
check('original failed Cartesian comparison retained',not original_failure['passed'] and original_failure==inputs['original_cartesian_audit']['preserved_failed_comparison'])
gates={k:max(computed[k])<=inputs['phase_target'] for k in ('phase1024_to2048','phase2048_to4096','halfshift','final_plane_phase','final_plane_halfshift')}
gates.update(boundary_integral=max(computed['boundary_integral'])<=inputs['plane_target'],
    boundary_weak_state=max(computed['boundary_weak_state'])<=inputs['plane_target'],
    active_step=max(computed['active_step'])<=inputs['step_target'],active_step_sizes=len(ratios)>0 and np.all(ratios<=.6),
    cartesian=bool(cartesian_good) and all(cartesian_good),required_reflected_tight=bool(tight_good) and all(tight_good),
    complete=all(c['complete'] for c in results['cases']) and all(w['complete'] for w in results['weak_boundary']) and not results['failures'] and not results['cap_exhausted'],
    energy=np.all(states['status']=='passed') and np.all(states['energy_error']<=inputs['energy_target']),
    source_integrity=source_gate,prior_integrity=prior_gate,original_failure_integrity=original_gate)
for key,v in gates.items():check(key+' producer gate reproduced',bool(v)==results['required_gates'][key])
check('all declared producer gates reconstructed',set(gates)==set(results['required_gates']))
check('producer overall status reproduced',results['status']==('passed' if all(gates.values()) else 'unresolved'))
head=subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
script_path=str(Path(__file__).resolve().relative_to(ROOT))
script_status=subprocess.run(['git','-C',str(ROOT),'status','--porcelain','--untracked-files=all','--',script_path],capture_output=True,text=True,check=True).stdout.strip()
receipt=dict(status='passed' if all(c['passed'] for c in checks) else 'failed',producer_status=results['status'],
    producer_commit=summary['commit'],producer_source_identities=producer_identities,
    audit_git_identity=head,audit_script_path=script_path,audit_script_sha256=sha(Path(__file__).resolve()),
    audit_source_uncommitted=bool(script_status),audit_script_git_status=script_status,
    reproduction_command='SM_PEAK_AUDIT_OUTPUT=results/encounter_peak_resolution/audit_reproduction.json .venv/bin/python '+script_path,
    input_sha256={p.name:sha(p) for p in [INPUT/'summary.json',INPUT/'states.npz',INPUT/'populations.npz',INPUT/'declared_inputs.json',INPUT/'failures.json']},
    formulas=dict(mean='m*(|w_out_perp|^2-|w_in_perp|^2)/(8B)',variance='m^2*theta*|R(+Omega*t)w_out_perp-w_in_perp|^2/(8B^2)',
        second='mean^2+variance',cross='mean^2-variance',spatial='|GC_out-GC_in|^2/4',phase='sum(all selected values)/N; no valid-only renormalization',
        comparison='abs(a-b)/[fine_M2/mu_reference,fine_M2,fine_M2,fine_spatial]'),
    array_selectors={k:dict(population_key=k,phase_count=v['phase_count'],distance=v['distance'],shifted=v['shifted']) for k,v in case_records.items()},
    reconstructed_gates={k:bool(v) for k,v in gates.items()},reconstructed_comparisons={k:v.tolist() for k,v in computed.items()},
    all_check_count=len(checks),checks=checks,wall_s=perf_counter()-started,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
    python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
    scope='Saved-array algebra, weights, grid hierarchy, source identities and acceptance replay. No new encounter integration, full-domain bound or plasma closure conclusion.')
OUTPUT.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
print(f'Audit {receipt["status"]}: {len(checks)} checks; {receipt["wall_s"]:.2f}s; receipt={OUTPUT}',flush=True)
if receipt['status']!='passed':raise RuntimeError('Independent audit failed: '+str([c for c in checks if not c['passed']]))
