"""Reconstruct retained encounter observables and declared acceptance gates.

Run from any directory with Python, NumPy, SciPy and mpmath. Inputs are the
explicit paths below. Scientific helpers from the producer are never imported.
The original summaries and failures are read only; only audit JSONs in a fresh separate destination are written.
"""
from pathlib import Path
from time import perf_counter
import gzip
import hashlib
import itertools
import json
import platform
import os
import resource
import subprocess
import sys
import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from scipy.sparse.linalg import spsolve
from scipy.special import k0, k1

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'results' / 'physical_final_campaign'
PRECISION = ROOT / 'results' / 'encounter_precision' / 'saved_binary64'
REFINEMENT = ROOT / 'results' / 'physical_refinement'
OUTPUT = ROOT / os.environ.get('SM_ENCOUNTER_AUDIT_OUTPUT', 'results/encounter_audit')
if OUTPUT.exists():
    raise FileExistsError(f'Preserve existing evidence: {OUTPUT}; choose a fresh SM_ENCOUNTER_AUDIT_OUTPUT')
OUTPUT.mkdir(parents=True)
AUDIT_RTOL, AUDIT_ATOL = 2e-7, 2e-13
# Dimensionless cancellation checks have binary64 drift subtraction noise.
AUDIT_CHANGE_ATOL = 2e-9
source_revisions = {}
started = perf_counter()
errors = []


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(label, actual, recorded, rtol=AUDIT_RTOL, atol=AUDIT_ATOL):
    a, b = np.asarray(actual, float), np.asarray(recorded, float)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.allclose(a, b, rtol=rtol, atol=atol):
        errors.append(label)
    return float(np.max(np.abs(a-b))) if a.shape == b.shape and a.size else None


def status(label, actual, recorded):
    if actual != recorded:
        errors.append(label)


def source_hash_matches(filename, wanted, producer_commit):
    if sha(ROOT/filename)==wanted:
        source_revisions[filename]='current matching bytes'
        return True
    # An I/O-only later revision may differ from the actual producer bytes.
    # Untracked producer sources have no blob in producer_commit; locate the
    # subsequent committed exact byte content rather than relabeling its origin.
    candidates=[producer_commit]
    candidates+=subprocess.check_output(['git','log','-100','--format=%H','--',filename],cwd=ROOT,text=True).splitlines()
    for revision in dict.fromkeys(candidates):
        result=subprocess.run(['git','show',revision+':'+filename],cwd=ROOT,capture_output=True)
        if result.returncode==0 and hashlib.sha256(result.stdout).hexdigest()==wanted:
            source_revisions[filename]=revision
            return True
    return False


def key(node, field, strength, distance, step):
    return tuple(map(float, (*node, field, strength, distance, step)))


def thermal(row, theta, mass, charge):
    # Rotation is undone before averaging an independent Gaussian COM velocity.
    wi = np.asarray(row['initial_relative_velocity'])[:2]
    wo = np.asarray(row['final_relative_velocity'])[:2]
    rotated = complex(*wo) * np.exp(1j*charge*row['field']/mass*row['time'])
    dw2 = abs(rotated-complex(*wi))**2
    c = mass*(wo@wo-wi@wi)/(8*row['field'])
    variance = mass**2*theta*dw2/(8*row['field']**2)
    return np.array([c, c*c+variance, c*c-variance])


def quadrature(case, theta):
    axes, qs = [], []
    for n, (lo, hi) in zip(case['orders'][:3], case['bounds']):
        x, q = np.polynomial.legendre.leggauss(n)
        axes.append(lo+(hi-lo)*(x+1)/2)
        qs.append(q*(hi-lo)/2)
    n = case['orders'][3]
    offset = np.pi/n if case['case'].endswith('_shifted') else 0.
    axes.append((2*np.pi*np.arange(n)/n+offset) % (2*np.pi))
    qs.append(np.full(n, 2*np.pi/n))
    nodes = np.array(list(itertools.product(*axes)))
    volume = np.prod(np.array(list(itertools.product(*qs))), axis=1)
    b, u, p, _ = nodes.T
    weights = 4*np.pi*b*u*p*np.exp(-(u*u+p*p)/(4*theta))/(4*np.pi*theta)**1.5*volume
    (bl, bh), (ul, uh), (pl, ph) = case['bounds']
    flux = 16*np.pi**2*theta**2*(bh*bh-bl*bl)*(np.exp(-ul*ul/(4*theta))-np.exp(-uh*uh/(4*theta)))*(np.exp(-pl*pl/(4*theta))-np.exp(-ph*ph/(4*theta)))/(4*np.pi*theta)**1.5
    return nodes, weights, float(flux), axes


def changes(a, b):
    if not a['complete'] or not b['complete']:
        return None
    va, vb = np.array(a['integral']), np.array(b['integral'])
    scale = np.array([abs(va[1])/a['mu_reference'], abs(va[1]), abs(va[1]), abs(va[3])])
    return (abs(va-vb)/scale).tolist() if np.all(scale>0) else None


def within(values, target):
    return values is not None and all(v is not None and np.isfinite(v) and v<=target for v in values)


def born_values(nodes, field, strength, distance, order, inputs):
    x, q = np.polynomial.legendre.leggauss(order)
    omega = inputs['charge']*field/inputs['mass']
    values = []
    for b, u, p, phase in nodes:
        t = x*distance/u
        free_velocity = p*np.exp(1j*(phase-omega*t))
        free_position = b+1j*free_velocity/omega
        radius = np.sqrt(abs(free_position)**2+(u*t)**2)
        force = 2*strength/inputs['mass']*np.exp(-radius/inputs['screening'])*(1+radius/inputs['screening'])/radius**3
        kick = np.sum(q*distance/u*force*free_position*np.exp(1j*omega*t))
        c = inputs['mass']/(4*field)*np.real(np.conj(p*np.exp(1j*phase))*kick)
        v = inputs['mass']**2*inputs['theta']*abs(kick)**2/(8*field**2)
        values.append([c, c*c+v, c*c-v])
    return np.array(values)


print('Audit: reading original campaign; reconstructing positive flux and COM moments', flush=True)
s = read(CAMPAIGN/'summary.json'); inputs = s['inputs']; saved = s['results']
attempts = read(CAMPAIGN/'attempts.json'); cases = read(CAMPAIGN/'cases.json')
lookup = {key(r['node'], r['field'], r['strength'], r['distance'], r['max_step']): r for r in attempts}
if len(lookup) != len(attempts): errors.append('duplicate encounter keys')
row_errors, gc_errors = [], []
for row in attempts:
    if row['status'] == 'passed':
        row_errors.append(compare('COM moment '+str(row['node']), thermal(row, inputs['theta'], inputs['mass'], inputs['charge']), row['moments']))
        gc_errors.append(compare('individual GC second '+str(row['node']), np.dot(row['gc_relative_kick'], row['gc_relative_kick'])/4, row['individual_gc_second']))
regenerated, case_reports = {}, []
for case in cases:
    nodes, weights, flux, axes = quadrature(case, inputs['theta'])
    records = [lookup.get(key(node, case['field'], case['strength'], case['distance'], inputs['max_step'])) for node in nodes]
    complete = all(r is not None and r['status']=='passed' for r in records)
    status('case completeness '+case['case'], complete, case['complete'])
    compare('exact flux '+case['case'], flux, case['flux_per_density'])
    flux_error = abs(weights.sum()/flux-1)
    compare('flux quadrature '+case['case'], flux_error, case['flux_quadrature_relative_error'])
    entry = dict(case=case['case'], complete=complete, flux_per_density=flux, flux_quadrature_relative_error=float(flux_error), mu_reference=inputs['mass']*inputs['theta']/case['field'])
    if complete:
        moments = np.array([thermal(r, inputs['theta'], inputs['mass'], inputs['charge']) for r in records])
        gc = np.array([np.dot(r['gc_relative_kick'], r['gc_relative_kick'])/4 for r in records])
        integral = np.r_[weights@moments, weights@gc]
        recorded = [case['drift_per_density'], 2*case['diffusion_mu_per_density'], case['cross_per_density'], 2*case['gc_trace_diffusion_per_density']]
        error = compare('case integral '+case['case'], integral, recorded)
        normalized = [integral[1]/entry['mu_reference']**2, integral[3]/inputs['screening']**2]
        compare('normalized rates '+case['case'], normalized, [case['normalized_mu_rate_per_density'], case['normalized_spatial_rate_per_density']])
        entry.update(integral=integral.tolist(), normalized_rates=normalized, maximum_integral_absolute_error=error, max_energy_error=max(r['energy_error'] for r in records))
    regenerated[case['case']] = entry
    case_reports.append(entry)

print('Audit: reconstructing coordinate, angular, boundary, step and heldout gates', flush=True)
scan_checks = []
for field in inputs['fields']:
    label = f'B{field:g}'; base = regenerated[label]
    row = dict(field=field, coordinate_changes=changes(base, regenerated[label+'_coarse']), phase_changes=changes(base, regenerated[label+'_phase8']), shifted_phase_changes=changes(base, regenerated[label+'_shifted']))
    row['status'] = 'passed' if all(within(row[k], inputs['moment_target']) for k in ('coordinate_changes','phase_changes','shifted_phase_changes')) and base.get('max_energy_error', float('inf'))<=inputs['energy_target'] else 'unresolved'
    original = next(r for r in saved['scan_checks'] if r['field']==field)
    for k in ('coordinate_changes','phase_changes','shifted_phase_changes'): compare('scan '+label+' '+k, row[k], original[k], atol=AUDIT_CHANGE_ATOL)
    status('scan status '+label, row['status'], original['status']); scan_checks.append(row)
held = np.load(CAMPAIGN/'heldout.npz'); nodes = held['nodes']
held_direct = np.array([thermal(lookup[key(node, 2., inputs['strength'], inputs['start_distance'], inputs['max_step'])], inputs['theta'], inputs['mass'], inputs['charge']) for node in nodes])
compare('heldout direct', held_direct, held['direct'])
final_case = next(c for c in cases if c['case']=='B2_final')
fn, fw, ff, axes = quadrature(final_case, inputs['theta'])
fm = np.array([thermal(lookup[key(node, 2., inputs['strength'], inputs['start_distance'], inputs['max_step'])], inputs['theta'], inputs['mass'], inputs['charge']) for node in fn]).reshape(*final_case['orders'], 3)
fields = np.stack((fm[...,0], np.log((fm[...,1]-fm[...,2])/2)), axis=-1)
spatial = RegularGridInterpolator(axes[:3], fields, method='cubic', solver=spsolve)
raw = spatial(nodes[:,:3]); closed = np.concatenate((raw, raw[:,:1]), axis=1)
phase_spline = CubicSpline(np.r_[axes[3], 2*np.pi], closed, axis=1, bc_type='periodic')
cv = np.array([phase_spline(nodes[j,3])[j] for j in range(len(nodes))])
c, variance = cv[:,0], np.exp(cv[:,1]); predicted = np.column_stack((c, c*c+variance, c*c-variance))
compare('heldout table predictions', predicted, held['predicted'], rtol=3e-6)
rms = np.sqrt(np.mean((held_direct-predicted)**2, axis=0)/np.mean(held_direct**2, axis=0))
p95 = float(np.quantile(abs(predicted[:,1]/held_direct[:,1]-1), .95))
table_check = dict(rms=rms.tolist(), second_p95=p95, status='passed' if rms[1]<=inputs['table_rms_target'] and p95<=inputs['table_p95_target'] else 'unresolved')
compare('table RMS', rms, saved['heldout_table']['rms']); compare('table p95', p95, saved['heldout_table']['second_p95']); status('table status', table_check['status'], saved['heldout_table']['status'])
steps = {step: [lookup[key(node, 2., inputs['strength'], inputs['start_distance'], step)] for node in nodes[:16]] for step in inputs['step_levels']}
a = np.array([thermal(r, inputs['theta'], inputs['mass'], inputs['charge']) for r in steps[.05]])
b = np.array([thermal(r, inputs['theta'], inputs['mass'], inputs['charge']) for r in steps[.1]])
scale = np.sqrt(np.mean(a[:,1]**2))*np.array([1/(inputs['mass']*inputs['theta']/2), 1, 1])
step_error = (np.sqrt(np.mean((a-b)**2, axis=0))/scale).tolist()
ratios = [f['max_actual_step']/c['max_actual_step'] for f,c in zip(steps[.05],steps[.1])]
compare('step moments', step_error, saved['step_changes_01_to_005']); compare('actual step ratios', ratios, saved['step_activity']['fine_over_coarse_actual_step_ratios'])
born_checks = []
for strength in inputs['weak_strengths']:
    data = np.load(CAMPAIGN/f'born_{strength:g}.npz')
    direct = np.array([thermal(lookup[key(node,2.,strength,inputs['start_distance'],inputs['max_step'])],inputs['theta'],inputs['mass'],inputs['charge']) for node in data['nodes']])
    fine = born_values(data['nodes'],2.,strength,inputs['start_distance'],inputs['born_orders'][-1],inputs)
    coarse = born_values(data['nodes'],2.,strength,inputs['start_distance'],inputs['born_orders'][0],inputs)
    for label, actual in [('direct',direct),('predicted',fine),('coarse',coarse)]: compare(f'Born {strength} {label}',actual,data[label])
    norm = np.sqrt(np.mean(direct**2,axis=0)); brms = np.sqrt(np.mean((fine-direct)**2,axis=0))/norm; change = np.sqrt(np.mean((fine-coarse)**2,axis=0))/norm
    check = dict(strength=strength,rms=brms.tolist(),born_quadrature_change=change.tolist(),status='passed' if max(brms[1:])<=inputs['born_target'] and max(change)<=.001 else 'unresolved')
    original = next(r for r in saved['born_checks'] if r['strength']==strength)
    compare('Born RMS',brms,original['rms']);compare('Born refinement',change,original['born_quadrature_change']);status('Born status',check['status'],original['status']);born_checks.append(check)

print('Audit: regenerating equal-individual-moment ring and Cartesian path controls', flush=True)
ring_rows = []
for old in read(CAMPAIGN/'ring.json'):
    node = (inputs['ring_impact'],inputs['ring_parallel_speed'],2*inputs['ring_speed']*np.sin(old['delta']/2),old['phase'])
    row = lookup[key(node,old['field'],inputs['strength'],inputs['start_distance'],old['max_step'])]
    if row['status']!='passed': ring_rows.append(dict(old)); continue
    wi = complex(*row['initial_relative_velocity'][:2]); wo = complex(*row['final_relative_velocity'][:2])
    center = .5j*wi/np.tan(old['delta']/2)
    rotated_center = center*np.exp(-1j*inputs['charge']*old['field']/inputs['mass']*row['time'])
    initial = inputs['mass']/(2*old['field'])*np.array([abs(center+wi/2)**2,abs(center-wi/2)**2])
    final = inputs['mass']/(2*old['field'])*np.array([abs(rotated_center+wo/2)**2,abs(rotated_center-wo/2)**2])
    increment = final-initial; mu=inputs['mass']*inputs['theta']/old['field']
    broadening = np.sum(final**2-initial**2)/mu**2
    compare('ring initial',initial,old['initial_mu']);compare('ring final',final,old['final_mu']);compare('ring increments',increment,old['increments']);compare('ring broadening',broadening,old['normalized_marginal_broadening'],atol=2e-14)
    ring_rows.append(dict(field=old['field'],delta=old['delta'],phase=old['phase'],max_step=old['max_step'],status=row['status'],increments=increment.tolist(),mu_reference=mu,normalized_marginal_broadening=float(broadening),normalized_total_mu_change=float(increment.sum()/mu),normalized_increment_second=float(increment@increment/mu**2),normalized_total_correction=float(2*increment.sum()/mu)))
ring_checks=[]
for fine in [r for r in ring_rows if r['max_step']==inputs['ring_refined_max_step']]:
    coarse=next(r for r in ring_rows if r['field']==fine['field'] and r['delta']==fine['delta'] and r['phase']==fine['phase'] and r['max_step']==inputs['max_step'])
    error=float(np.max(abs(np.array(coarse['increments'])-fine['increments']))/fine['mu_reference']); delta=abs(coarse['normalized_marginal_broadening']-fine['normalized_marginal_broadening']);signal=fine['normalized_marginal_broadening']
    check=dict(field=fine['field'],delta=fine['delta'],phase=fine['phase'],increment_refinement_error=error,normalized_broadening=signal,broadening_refinement_error=delta,signal_to_refinement=signal/delta if delta else None,status='passed' if error<=inputs['ring_increment_normalized_target'] and signal>inputs['ring_broadening_signal_to_refinement_minimum']*delta and signal>0 else 'unresolved')
    old=next(r for r in saved['ring_checks'] if all(r[k]==check[k] for k in ('field','delta','phase')))
    status('ring gate',check['status'],old['status']);ring_checks.append(check)
trajectory_checks=[]
paths=np.load(CAMPAIGN/'trajectories.npz')
for i in range(3):
    t=paths[f'time{i}'];position=paths[f'positions{i}'];velocity=paths[f'velocities{i}'];r=position[:,0]-position[:,1];w=velocity[:,0]-velocity[:,1]
    def rhs(time,y):
        radius=np.linalg.norm(y[:3]);f=2*inputs['strength']/inputs['mass']*np.exp(-radius/inputs['screening'])*(1+radius/inputs['screening'])/radius**3*y[:3]
        f+=np.array([2*y[4],-2*y[3],0.]);return np.r_[y[3:],f]
    independent=solve_ivp(rhs,(float(t[0]),float(t[-1])),np.r_[r[0],w[0]],method='DOP853',rtol=1e-13,atol=1e-14,max_step=.0125)
    if not independent.success:errors.append('Cartesian solve '+str(i))
    pe=float(max(abs(independent.y[:3,-1]-r[-1])));ve=float(max(abs(independent.y[3:,-1]-w[-1])))
    energy=inputs['mass']*np.sum(w*w,axis=1)/4+inputs['strength']*np.exp(-np.linalg.norm(r,axis=1)/inputs['screening'])/np.linalg.norm(r,axis=1)
    ee=float(max(abs(energy/energy[0]-1)))
    gc=r[:,:2]+np.column_stack((w[:,1],-w[:,0]))/2;gc_kick=gc[-1]-gc[0]
    check=dict(path=i,position_absolute_error=pe,velocity_absolute_error=ve,energy_error=ee,gc_relative_kick=gc_kick.tolist(),individual_gc_second=float(gc_kick@gc_kick/4),com_position_max=float(np.max(abs(position.mean(axis=1)))),com_velocity_max=float(np.max(abs(velocity.mean(axis=1)))))
    trajectory_checks.append(check)
    if pe>inputs['cartesian_position_absolute_target'] or ve>inputs['cartesian_velocity_absolute_target'] or ee>inputs['energy_target']:errors.append('Cartesian check '+str(i))
base=[regenerated[f'B{field:g}'] for field in inputs['fields']];final=regenerated['B2_final']
final_change=changes(final,base[0]);boundary_change=changes(regenerated['boundary_32'],regenerated['boundary_24'])
compare('final B2 refinement',final_change,saved['base_final_refinement_changes']);compare('boundary refinement',boundary_change,saved['boundary_changes_24_to_32'])
gates=dict(complete_base_scan=all(r['complete'] for r in base+[final]),energy=all(r.get('max_energy_error',float('inf'))<=inputs['energy_target'] for r in base+[final]),phase_and_coordinate_moments=all(r['status']=='passed' for r in scan_checks),final_B2_refinement=within(final_change,inputs['moment_target']),incoming_flux=all(r['flux_quadrature_relative_error']<=inputs['flux_target'] for r in base+[final]),receding_planes=within(boundary_change,inputs['moment_target']),active_timestep=within(step_error,inputs['step_target']) and all(v<=inputs['fine_to_coarse_actual_step_ratio_target'] for v in ratios),heldout_table=table_check['status']=='passed',weak_Born=len(born_checks)==len(inputs['weak_strengths']) and all(r['status']=='passed' for r in born_checks),ring_refinement_and_positive_broadening=len(ring_checks)==24 and all(r['status']=='passed' for r in ring_checks),cartesian_trajectory=len(trajectory_checks)==3 and all(r['position_absolute_error']<=inputs['cartesian_position_absolute_target'] and r['velocity_absolute_error']<=inputs['cartesian_velocity_absolute_target'] and r['energy_error']<=inputs['energy_target'] for r in trajectory_checks),full_moment_consistency=all(v[1]>=abs(v[2]) and np.all(np.isfinite(v)) for v in (thermal(r,inputs['theta'],inputs['mass'],inputs['charge']) for r in attempts if r['status']=='passed')),source_integrity=s['experiment_dependency_sha256']==s['experiment_dependency_sha256_end'])
for k,v in gates.items(): status('required gate '+k,v,saved['required_gates'][k])
for filename,wanted in s['output_sha256'].items(): status('campaign artifact hash '+filename,sha(CAMPAIGN/filename),wanted)
for filename,wanted in s['experiment_dependency_sha256'].items(): status('campaign dependency hash '+filename,source_hash_matches(filename,wanted,s['commit']),True)
physical_errors=list(errors)
physical=dict(audit_passed=not physical_errors,errors=physical_errors,producer_overall_status=saved['overall_status'],reconstructed_overall_status='passed' if all(gates.values()) and not saved['budget_exhausted'] else 'unresolved',required_gates=gates,failed_gates=[k for k,v in gates.items() if not v],maximum_row_moment_absolute_error=max(row_errors),maximum_gc_second_absolute_error=max(gc_errors),audit_numeric_comparison_tolerances=dict(relative=AUDIT_RTOL,absolute=AUDIT_ATOL,dimensionless_cancellation_absolute=AUDIT_CHANGE_ATOL),case_integrals=case_reports,scan_checks=scan_checks,heldout_table=table_check,born_checks=born_checks,step_changes=step_error,actual_step_ratios=ratios,ring_rows=ring_rows,ring_checks=ring_checks,cartesian_checks=trajectory_checks,original_failures_preserved=True,scope='Independent retained-state algebra, positive incoming flux, COM moments, kick coefficients, ring marginal changes, all declared gates, and three independent Cartesian ODE controls. Encounter rows omit positions: their stored GC kicks are primitive inputs, not independently recomputed trajectories. Energy/max-step row diagnostics and source-integrity gate use retained producer records. Finite conditional incoming model only.')

print('Audit: checking full precision paths, exact binary64 inputs and precision refinement',flush=True)
errors=[];p=read(PRECISION/'summary.json');mp.mp.dps=90;packed=np.load(PRECISION/'paths_binary64.npz');path_checks=[]
for filename,wanted in p['output_sha256'].items():status('precision artifact hash '+filename,sha(PRECISION/filename),wanted)
for filename,wanted in p['source_sha256'].items():status('precision dependency hash '+filename,source_hash_matches(filename,wanted,p['commit']),True)
for run in p['results']['runs']:
    with gzip.open(PRECISION/(run['label']+'_full_precision_path.json.gz'),'rt') as stream:path=json.load(stream)
    case=next(c for c in p['inputs']['cases'] if run['label'].startswith(c['label']+'_'))
    states=[[mp.mpf(v) for v in row] for row in path['state']];times=[mp.mpf(v) for v in path['time']]
    plane=mp.mpf(float(p['inputs']['start_distance']));crossings=[]
    for j,(before,after) in enumerate(zip(states,states[1:])):
        if before[2]<plane<=after[2] and after[5]>0:crossings.append((j,'transmitted'))
        if before[2]>-plane>=after[2] and after[5]<0:crossings.append((j,'reflected'))
    event=run['event'];event_time=mp.mpf(event['time']);event_state=[mp.mpf(v) for v in event['state']]
    if not crossings:errors.append('retained first outgoing crossing '+run['label'])
    else:
        j,exit_label=crossings[0]
        if exit_label!=event['exit'] or not times[j]<=event_time<=times[j+1]:errors.append('first outgoing event bracket '+run['label'])
    if abs(abs(event_state[2])-plane)>mp.mpf(10)**(-run['dps']+8):errors.append('event plane '+run['label'])
    serialization=max(abs(v-mp.mpf(float(w))) for v,w in zip(states[0],case['initial']))
    status('exact binary64 initial '+run['label'],[float(v).hex() for v in states[0]],case['initial_binary64_hex'])
    if serialization>mp.mpf(10)**(-run['dps']+3):errors.append('decimal serialization '+run['label'])
    status('packed states '+run['label'],bool(np.array_equal(np.asarray(path['state'],float),packed[run['label']+'_state'])),True)
    status('packed times '+run['label'],bool(np.array_equal(np.asarray(path['time'],float),packed[run['label']+'_time'])),True)
    status('endpoint states '+run['label'],path['state'][-1],run['state']);status('endpoint time '+run['label'],path['time'][-1],run['time'])
    if not all(a<b for a,b in zip(times,times[1:])):errors.append('monotonic time '+run['label'])
    omega=mp.mpf(float(p['inputs']['charge']))*mp.mpf(float(p['inputs']['field']))/mp.mpf(float(p['inputs']['mass']))
    canonical=[y[0]*y[4]-y[1]*y[3]+omega*(y[0]**2+y[1]**2)/2 for y in states]
    strength=mp.mpf(float(p['inputs']['strength']));screening=mp.mpf(float(p['inputs']['screening']));mass=mp.mpf(float(p['inputs']['mass']))
    energies=[]
    for y in states:
        radius=mp.sqrt(sum(v*v for v in y[:3]));energies.append(mass*sum(v*v for v in y[3:])/4+strength*mp.exp(-radius/screening)/radius)
    drift=max(abs(e/energies[0]-1) for e in energies)
    path_checks.append(dict(run=run['label'],saved_states=len(states),initial_decimal_serialization_difference=str(serialization),canonical_angular_momentum_absolute_drift=str(max(abs(v-canonical[0]) for v in canonical)),relative_energy_drift=str(drift)))
comparisons=[]
for case in p['inputs']['cases']:
    pair=[r for r in p['results']['runs'] if r['label'].startswith(case['label']+'_')];low,high=pair
    difference=[abs(mp.mpf(a)-mp.mpf(b)) for a,b in zip(low['state'],high['state'])]
    time_difference=abs(mp.mpf(low['event']['time'])-mp.mpf(high['event']['time']))
    check=dict(case=case['label'],position_difference=str(max(difference[:3])),velocity_difference=str(max(difference[3:])),first_exit_time_difference=str(time_difference),matching_first_exit=low['event']['exit']==high['event']['exit'])
    check['passed']=max(difference[:3])<=p['inputs']['position_absolute_target'] and max(difference[3:])<=p['inputs']['velocity_absolute_target'] and time_difference<=p['inputs']['time_absolute_target'] and check['matching_first_exit']
    old=next(r for r in p['results']['precision_comparisons'] if r['case']==case['label'])
    for quantity in ('position_difference','velocity_difference','first_exit_time_difference'):
        if not mp.almosteq(mp.mpf(check[quantity]),mp.mpf(old[quantity]),rel_eps=mp.mpf('1e-25'),abs_eps=mp.mpf('1e-75')):errors.append('precision difference '+case['label']+' '+quantity)
    status('precision refinement '+case['label'],check['passed'],old['passed']);comparisons.append(check)
old_phase=read(ROOT/'results/encounter_phase/all_moments8192/summary.json')['results']['phase_status']
precision=dict(audit_passed=not errors,errors=list(errors),path_checks=path_checks,precision_comparisons=comparisons,producer_status=p['results']['status'],original_phase_integral_status=old_phase,original_censoring_status=p['results']['original_precision_checked_status'],original_failures_preserved=True,scope='Independent retained high-precision path serialization, exact binary64 input, canonical-angular-momentum and energy checks, and recomputed fixed-state 50/70-digit refinement. No new arbitrary-precision solve or continuous integral certification.')


def receipt(source,directory,source_paths):
    dirty=source.get('source_changes')
    return dict(producer_commit=source.get('commit'),producer_scientific_source_changes=dirty,producer_dirty_source_diff_sha256=None,producer_source_sha256=source_paths,matching_source_revision={k:source_revisions.get(k) for k in source_paths},producer_wall_s=source.get('wall_s',source.get('results',{}).get('wall_s')),producer_parent_process_peak_rss_bytes=source.get('parent_process_peak_rss_bytes'),producer_worker_process_peak_rss_bytes_max=source.get('worker_process_peak_rss_bytes_max'),producer_memory_scope=source.get('memory_scope'),producer_python=source.get('python'),producer_platform=source.get('platform'),audit_source_sha256=sha(Path(__file__)),audit_input_sha256={f.name:sha(f) for f in directory.iterdir() if f.is_file() and not f.name.startswith('independent_audit')},audit_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),audit_python=sys.version,audit_platform=platform.platform(),audit_wall_s=perf_counter()-started,audit_process_lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if sys.platform=='darwin' else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,missing_provenance='Null means absent in producer metadata. Exact dirty diff bytes were not retained, so no diff hash is asserted. Producer source byte hashes identify actual scientific source content; historical matching revisions do not relabel the producer commit.')

outputs = [(CAMPAIGN,physical,s,s['experiment_dependency_sha256']),(PRECISION,precision,p,p['source_sha256'])]
refinement_errors=[]
if (REFINEMENT/'summary.json').exists():
    print('Audit: reconstructing focused refinement, fresh heldouts and weak B32 amplitudes',flush=True)
    errors=[];rs=read(REFINEMENT/'summary.json');ri=rs['inputs'];rr=rs['results'];ra=read(REFINEMENT/'attempts.json');rc=read(REFINEMENT/'cases.json')
    rl={key(r['node'],r['field'],r['strength'],ri['start_distance'],ri['max_step']):r for r in ra}
    row_moment_errors=[];row_kick_errors=[];refcases={}
    for row in ra:
        if row['status']!='passed':continue
        row_moment_errors.append(compare('refinement moment',thermal(row,ri['theta'],ri['mass'],ri['charge']),row['moments']))
        omega=ri['charge']*row['field']/ri['mass']
        pos0=np.asarray(row['initial_relative_position']);pos1=np.asarray(row['final_relative_position']);vel0=np.asarray(row['initial_relative_velocity']);vel1=np.asarray(row['final_relative_velocity'])
        kick=pos1[:2]-pos0[:2]+np.array([vel1[1]-vel0[1],vel0[0]-vel1[0]])/omega
        row_kick_errors.append(compare('refinement GC kick',kick,row['gc_relative_kick']))
        compare('refinement GC second',kick@kick/4,row['individual_gc_second'])
    for case in rc:
        declared=dict(case,bounds=ri['bounds'],distance=ri['start_distance'])
        cn,cw,cf,caxes=quadrature(declared,ri['theta'])
        rows=[rl.get(key(node,case['field'],case['strength'],ri['start_distance'],ri['max_step'])) for node in cn]
        complete=all(r is not None and r['status']=='passed' for r in rows)
        status('refinement case completeness',complete,case['complete'])
        reconstructed=dict(case=case['case'],complete=complete,mu_reference=ri['mass']*ri['theta']/case['field'],field=case['field'])
        if complete:
            moments=np.array([thermal(r,ri['theta'],ri['mass'],ri['charge']) for r in rows])
            spatial=np.array([np.dot(r['gc_relative_kick'],r['gc_relative_kick'])/4 for r in rows])
            integral=np.r_[cw@moments,cw@spatial]
            compare('refinement case integral',integral,case['integrals']+[case['spatial_second_integral']])
            flux_error=float(abs(cw.sum()/cf-1));compare('refinement flux',cw.sum(),case['flux_per_density']);compare('refinement flux error',flux_error,case['flux_relative_error'])
            normalized=[integral[1]/reconstructed['mu_reference']**2,integral[3]/ri['screening']**2]
            compare('refinement normalized rates',normalized,[case['normalized_mu_rate_per_density'],case['normalized_spatial_rate_per_density']])
            reconstructed.update(integral=integral.tolist(),normalized_rates=normalized,flux_relative_error=flux_error,max_energy_error=max(r['energy_error'] for r in rows))
        refcases[case['case']]=reconstructed
    table_case=next(c for c in rc if c['case']=='B2_table7');tn,tw,tf,taxes=quadrature(dict(table_case,bounds=ri['bounds']),ri['theta'])
    table_checks=[]
    if table_case['complete']:
        tm=np.array([thermal(rl[key(node,2.,ri['strength'],ri['start_distance'],ri['max_step'])],ri['theta'],ri['mass'],ri['charge']) for node in tn]).reshape(*table_case['orders'],3)
        values=np.stack((tm[...,0],np.log((tm[...,1]-tm[...,2])/2)),axis=-1)
        interpolation=RegularGridInterpolator(taxes[:3],values,method='cubic',solver=spsolve)
        generator=np.random.default_rng(ri['fresh_seed'])
        fresh=np.column_stack([generator.uniform(axis[0],axis[-1],ri['fresh_count']) for axis in taxes[:3]]+[generator.uniform(0,2*np.pi,ri['fresh_count'])])
        for label in ('original128','fresh128'):
            data=np.load(REFINEMENT/f'heldout_{label}.npz');nodes=data['nodes']
            compare('heldout sampling '+label,nodes,held['nodes'] if label=='original128' else fresh,rtol=0,atol=0)
            direct=np.array([thermal(rl[key(node,2.,ri['strength'],ri['start_distance'],ri['max_step'])],ri['theta'],ri['mass'],ri['charge']) for node in nodes])
            raw=interpolation(nodes[:,:3]);spline=CubicSpline(np.r_[taxes[3],2*np.pi],np.concatenate((raw,raw[:,:1]),axis=1),axis=1,bc_type='periodic')
            cv=np.array([spline(nodes[j,3])[j] for j in range(len(nodes))]);c=cv[:,0];v=np.exp(cv[:,1]);predicted=np.column_stack((c,c*c+v,c*c-v))
            compare('refinement heldout direct',direct,data['direct']);compare('refinement table predictions',predicted,data['predicted'],rtol=3e-6)
            rms=np.sqrt(np.mean((direct-predicted)**2,axis=0)/np.mean(direct**2,axis=0));p95=float(np.quantile(abs(predicted[:,1]/direct[:,1]-1),.95))
            check=dict(sample=label,rms=rms.tolist(),second_p95=p95,status='passed' if rms[1]<=ri['table_rms_target'] and p95<=ri['table_p95_target'] else 'unresolved')
            old=next(r for r in rr['table_checks'] if r['sample']==label);compare('refinement table RMS',rms,old['rms']);compare('refinement table p95',p95,old['second_p95']);status('refinement heldout status',check['status'],old['status']);table_checks.append(check)
    weak_checks=[]
    for old in rr['weak_B32']:
        case=refcases[old['case']]
        if not case['complete']:weak_checks.append(dict(case,status='unresolved'));continue
        cn,cw,_,_=quadrature(dict(old,bounds=ri['bounds']),ri['theta']);b,u,p,_=cn.T;omega=ri['charge']*old['field']/ri['mass']
        alpha=2*old['strength']*k0(b/ri['screening'])/(ri['mass']*omega*u*ri['screening']**2)
        reference=np.array([cw@(alpha**2*p**2/(8*ri['theta'])),cw@(4*old['strength']**2*k1(b/ri['screening'])**2/(ri['mass']**2*omega**2*u**2*ri['screening']**4))])
        change=abs(np.array(case['normalized_rates'])/reference-1)
        compare('weak B32 asymptotic reference',reference,[old['asymptotic_mu_rate'],old['asymptotic_spatial_rate']]);compare('weak B32 amplitude error',change,old['asymptotic_relative_errors'])
        weak_checks.append(dict(case=old['case'],strength=old['strength'],asymptotic_rates=reference.tolist(),relative_errors=change.tolist(),status='passed' if max(change)<=ri['weak_asymptotic_target'] else 'unresolved'))
    b4_change=changes(refcases['B4_order6'],refcases['B4_order5']);compare('B4 last refinement',b4_change,rr['B4_last_refinement_changes'],atol=AUDIT_CHANGE_ATOL)
    gates=dict(B4_coordinate_refinement=within(b4_change,ri['moment_target']),table_both_heldouts=len(table_checks)==2 and all(r['status']=='passed' for r in table_checks),energy=all(c.get('max_energy_error',float('inf'))<=ri['energy_target'] for c in refcases.values()),flux=all(c.get('flux_relative_error',float('inf'))<=1e-5 for c in refcases.values()),weak_B32_asymptotic=len(weak_checks)==len(ri['weak_B32_strengths']) and all(r['status']=='passed' for r in weak_checks),source_integrity=rs['experiment_dependency_sha256']==rs['experiment_dependency_sha256_end'])
    for k,v in gates.items():status('refinement required gate '+k,v,rr['required_gates'][k])
    overall='passed' if all(gates.values()) else 'unresolved';status('refinement overall status',overall,rr['overall_status'])
    for filename,wanted in rs['output_sha256'].items():status('refinement artifact hash '+filename,sha(REFINEMENT/filename),wanted)
    for filename,wanted in rs['source_artifact_sha256'].items():status('original campaign retained hash '+filename,sha(CAMPAIGN/filename),wanted)
    for filename,wanted in rs['experiment_dependency_sha256'].items():status('refinement source hash '+filename,source_hash_matches(filename,wanted,rs['commit']),True)
    refinement_errors=list(errors)
    report=dict(audit_passed=not refinement_errors,errors=refinement_errors,producer_overall_status=rr['overall_status'],reconstructed_overall_status=overall,required_gates=gates,maximum_row_moment_absolute_error=max(row_moment_errors,default=None),maximum_gc_kick_absolute_error=max(row_kick_errors,default=None),case_integrals=list(refcases.values()),B4_last_refinement_changes=b4_change,table_checks=table_checks,weak_B32=weak_checks,original_campaign_status=physical['producer_overall_status'],original_failures_preserved=True,scope='Independent saved position/velocity COM moments and GC kicks, flux quadrature, all coefficients, spatial and periodic phase table reconstruction using a direct spline solve, old/fresh seeded heldouts, weak B32 leading references and all declared followup gates. Original campaign failure remains unchanged. No plasma closure or kinetic mixing-time inference.')
    outputs.append((REFINEMENT,report,rs,rs['experiment_dependency_sha256']))
else:
    print('Audit: focused refinement summary absent; optional followup audit not run',flush=True)

for directory,report,source,source_paths in outputs:
    destination=OUTPUT/directory.name
    destination.mkdir()
    (destination/'independent_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (destination/'independent_audit_receipt.json').write_text(json.dumps(receipt(source,directory,source_paths),indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(physical_audit_passed=physical['audit_passed'],precision_audit_passed=precision['audit_passed'],campaign_status=physical['reconstructed_overall_status'],failed_gates=physical['failed_gates'],physical_errors=physical_errors,precision_errors=precision['errors'],refinement_errors=refinement_errors,wall_s=perf_counter()-started),indent=2),flush=True)
if physical_errors or precision['errors'] or refinement_errors:raise RuntimeError('Independent audit mismatch; see saved independent_audit.json files')
