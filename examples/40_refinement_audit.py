"""Independently reconstruct new interpolation and broad-speed evidence.

Only NumPy/SciPy and saved Cartesian states are used. No producer physics,
interpolation, flux, or metadata helpers are imported. Existing evidence is
preserved; select a fresh SM_REFINEMENT_AUDIT_OUTPUT for each replay.
"""
from pathlib import Path
from time import perf_counter
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import make_interp_spline
from scipy.stats import binom, qmc

ROOT = Path(__file__).resolve().parents[1]
INTERPOLATION = ROOT / os.environ.get('SM_INTERPOLATION_AUDIT_INPUT', 'results/scattering_interpolation')
BROAD = ROOT / os.environ.get('SM_BROAD_AUDIT_INPUT', 'results/broad_speed_refinement')
OUTPUT = ROOT / os.environ.get('SM_REFINEMENT_AUDIT_OUTPUT', 'results/refinement_audit')
CARTESIAN_POSITION_TARGET = CARTESIAN_VELOCITY_TARGET = CARTESIAN_TIME_TARGET = 1e-8


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def historical_source_identity(directory, summary, identity):
    """Check the producer's actual Git blobs, retaining its original revision."""
    revisions = {}
    producer = summary['commit']
    for name, expected in summary['experiment_dependency_sha256'].items():
        candidates = [producer]
        candidates.extend(subprocess.check_output(['git', 'log', '-100', '--format=%H', '--', name], cwd=ROOT, text=True).splitlines())
        matched = None
        for commit in dict.fromkeys(candidates):
            blob = subprocess.run(['git', 'show', commit+':'+name], cwd=ROOT, capture_output=True)
            if blob.returncode==0 and hashlib.sha256(blob.stdout).hexdigest()==expected:
                matched = commit
                break
        identity(directory.name+' historical producer source '+name, matched is not None)
        revisions[name] = {'sha256':expected, 'matching_git_revision':matched,
            'matches_declared_producer_commit':matched==producer}
    for key in ('output_sha256',):
        for name, expected in summary.get(key, {}).items():
            identity(directory.name+' immutable output '+name, digest(directory/name)==expected)
    return revisions


def rotation(angle):
    """Standard mathematical rotation; free positive-charge motion is clockwise."""
    c, s = np.cos(angle), np.sin(angle)
    answer = np.empty(np.shape(angle)+(2, 2))
    answer[..., 0, 0] = c
    answer[..., 0, 1] = -s
    answer[..., 1, 0] = s
    answer[..., 1, 1] = c
    return answer


def reconstruct(arrays, theta, mass, charge, field, distance):
    """COM average from Cartesian velocities and independent rotation matrices."""
    wi = np.asarray(arrays['initial_relative_velocity'])
    wo = np.asarray(arrays['final_relative_velocity'])
    time = np.asarray(arrays['time'])
    omega = charge*field/mass
    incoming_kick = np.einsum('...ij,...j->...i', rotation(omega*time), wo[..., :2])-wi[..., :2]
    c = mass*(np.sum(wo[..., :2]**2, axis=-1)-np.sum(wi[..., :2]**2, axis=-1))/(8*field)
    variance = mass**2*theta*np.sum(incoming_kick**2, axis=-1)/(8*field**2)
    moments = np.stack((c, c*c+variance, c*c-variance), axis=-1)
    zero_kick = np.einsum('...ij,...j->...i', rotation(-omega*np.asarray(distance)/np.asarray(arrays['nodes'])[..., 1]), incoming_kick)
    ri, ro = np.asarray(arrays['initial_relative_position']), np.asarray(arrays['final_relative_position'])
    quarter_turn = np.array([[0., 1.], [-1., 0.]])
    gc_kick = ro[..., :2]-ri[..., :2]+np.einsum('ij,...j->...i', quarter_turn, wo[..., :2]-wi[..., :2])/omega
    spatial = np.sum(gc_kick**2, axis=-1)/4
    return moments, zero_kick, incoming_kick, gc_kick, spatial


def check_incoming_frame(label, arrays, mass, charge, field, distance, compare):
    nodes = np.asarray(arrays['nodes'])
    ri, wi = np.asarray(arrays['initial_relative_position']), np.asarray(arrays['initial_relative_velocity'])
    omega = charge*field/mass
    zero_velocity = np.einsum('...ij,...j->...i', rotation(-omega*np.asarray(distance)/nodes[:, 1]), wi[:, :2])
    wanted = nodes[:, 2, None]*np.column_stack((np.cos(nodes[:, 3]), np.sin(nodes[:, 3])))
    compare(label+' independently checked zero-plane incoming phase', zero_velocity, wanted, atol=2e-13)
    compare(label+' incoming parallel speed', wi[:, 2], nodes[:, 1], atol=2e-13)
    incoming_gc = ri[:, :2]+np.column_stack((wi[:, 1], -wi[:, 0]))/omega
    compare(label+' guiding-center impact', incoming_gc, np.column_stack((nodes[:, 0], np.zeros(len(nodes)))), atol=2e-13)
    compare(label+' incoming event plane', ri[:, 2], -np.broadcast_to(distance, len(nodes)), atol=2e-13)


def bounded_flux(bounds, theta):
    (bl, bh), (ul, uh), (pl, ph) = bounds
    # Analytically integrate the area and the two independent squared speeds.
    return (16*np.pi**2*theta**2*(bh*bh-bl*bl)
        *(np.exp(-ul*ul/(4*theta))-np.exp(-uh*uh/(4*theta)))
        *(np.exp(-pl*pl/(4*theta))-np.exp(-ph*ph/(4*theta)))
        /(4*np.pi*theta)**1.5)


def inverse_flux(unit, bounds, theta):
    coordinates = [np.sqrt(bounds[0][0]**2+unit[:, 0]*(bounds[0][1]**2-bounds[0][0]**2))]
    for j, (lo, hi) in enumerate(bounds[1:], 1):
        survival = np.exp(-lo*lo/(4*theta))-unit[:, j]*(np.exp(-lo*lo/(4*theta))-np.exp(-hi*hi/(4*theta)))
        coordinates.append(np.sqrt(-4*theta*np.log(survival)))
    return np.column_stack(coordinates)


def tensor_spline(axes, fields, queries):
    """Sequential not-a-knot splines then periodic phase; no RGI implementation."""
    answers = []
    phase = np.r_[axes[3], 2*np.pi]
    for point in queries:
        value = fields
        for axis, coordinate in zip(axes[:3], point[:3]):
            value = make_interp_spline(axis, value, k=3, axis=0)(coordinate)
        closed = np.concatenate((value, value[:1]), axis=0)
        answers.append(make_interp_spline(phase, closed, k=3, axis=0, bc_type='periodic')(point[3] % (2*np.pi)))
    return np.asarray(answers)


def scaled_changes(reference, other, mu_reference):
    scale = np.array([reference[1]/mu_reference, reference[1], reference[1], reference[3]])
    return abs(reference-other)/scale


def fresh_cartesian_replay(states, index, inputs):
    """Twelve Cartesian particle coordinates, independently assembled force."""
    mass, charge, field = (inputs[k] for k in ('mass', 'charge', 'field'))
    strength, screening = (inputs[k] for k in ('strength', 'screening'))
    distance = float(states['distance'][index])
    ri, wi = states['initial_relative_position'][index], states['initial_relative_velocity'][index]
    initial = np.r_[ri/2, wi/2, -ri/2, -wi/2]
    omega = charge*field/mass
    def rhs(time, y):
        separation = y[:3]-y[6:9]
        radius = np.linalg.norm(separation)
        force = strength*np.exp(-radius/screening)*(1+radius/screening)*separation/(mass*radius**3)
        first, second = y[3:6], y[9:12]
        return np.r_[first, omega*np.cross(first, [0., 0., 1.])+force,
            second, omega*np.cross(second, [0., 0., 1.])-force]
    def positive(time, y):
        return y[2]-y[8]-distance
    def negative(time, y):
        return y[2]-y[8]+distance
    positive.terminal, positive.direction = True, 1
    negative.terminal, negative.direction = True, -1
    budget = inputs['flight_time_factor']*distance/states['nodes'][index, 1]
    solve = solve_ivp(rhs, (0., budget), initial, method='DOP853', rtol=2e-13,
        atol=2e-15, max_step=.025, events=(positive, negative))
    if not solve.success or solve.status != 1:
        return {'index':int(index), 'passed':False, 'reason':solve.message}
    final = solve.y[:, -1]
    position_error = float(np.max(abs(final[:3]-final[6:9]-states['final_relative_position'][index])))
    velocity_error = float(np.max(abs(final[3:6]-final[9:12]-states['final_relative_velocity'][index])))
    time_error = float(abs(solve.t[-1]-states['time'][index]))
    exit_name = 'transmitted' if len(solve.t_events[0]) else 'reflected'
    return {'index':int(index), 'passed':position_error<=CARTESIAN_POSITION_TARGET and velocity_error<=CARTESIAN_VELOCITY_TARGET
        and time_error<=CARTESIAN_TIME_TARGET and exit_name==states['exit'][index],
        'position_max_absolute_error':position_error, 'velocity_max_absolute_error':velocity_error,
        'exit_time_absolute_error':time_error, 'exit':exit_name, 'evaluations':solve.nfev,
        'acceptance_targets':{'position':CARTESIAN_POSITION_TARGET, 'velocity':CARTESIAN_VELOCITY_TARGET, 'time':CARTESIAN_TIME_TARGET}}


def select_cartesian_replays(states, primary_ids, values, count=3):
    """Include a reflected accepted state and avoid repeat incident-node copies."""
    passed = np.asarray(states['status'])=='passed'
    primary = np.asarray(primary_ids)
    primary = primary[primary>=0]
    primary = primary[passed[primary]]
    candidates = []
    if len(primary):
        candidates.extend((primary[np.argmin(states['min_separation'][primary])],
            primary[np.argmax(values[primary, 1])]))
    reflected = np.flatnonzero(passed & (np.asarray(states['exit'])=='reflected'))
    if len(reflected):
        candidates.append(reflected[0])
    candidates.extend(primary[:count])
    selected, seen = [], set()
    for index in candidates:
        node = tuple(states['nodes'][index])
        if node not in seen:
            seen.add(node)
            selected.append(int(index))
        if len(selected)==count:
            break
    return selected




def audit_interpolation(directory, compare, identity):
    summary = load(directory/'summary.json')
    inputs, results = summary['inputs'], summary['results']
    table, tests = np.load(directory/'table.npz'), np.load(directory/'tests.npz')
    attempts = load(directory/'attempts.json')
    theta, mass, charge, field, distance = (inputs[k] for k in ('theta', 'mass', 'charge', 'field', 'start_distance'))
    orders, axes = inputs['table_orders'], [np.asarray(a) for a in inputs['table_axes']]
    moments, kicks, _, _, _ = reconstruct(table, theta, mass, charge, field, distance)
    check_incoming_frame('table', table, mass, charge, field, distance, compare)
    table_scales = np.column_stack((np.sqrt(moments[:, 1]), moments[:, 1], moments[:, 1]))
    compare('table Cartesian COM reconstruction / signal scales', moments/table_scales, table['moments']/table_scales, atol=2e-7)
    compare('table zero-plane signed kicks', kicks, table['kicks'], atol=2e-12)
    expected_nodes = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 4)
    compare('table tensor-node order', table['nodes'], expected_nodes, atol=0)
    identity('table size', len(expected_nodes)==results['table_states']==int(np.prod(orders)))
    for j, (lo, hi) in enumerate(inputs['table_bounds']):
        x = np.polynomial.legendre.leggauss(orders[j])[0]
        compare('table independently rebuilt coordinate '+str(j), axes[j], lo+(hi-lo)*(x+1)/2, atol=2e-14)
    compare('table uniform periodic phase', axes[3], 2*np.pi*np.arange(orders[3])/orders[3], atol=2e-14)
    source = ROOT/'results/physical_refinement'
    archived = load(source/'attempts.json')
    source_rows = {tuple(r['node']):r for r in archived if r['field']==field and r['strength']==inputs['strength']}
    archive_arrays = {}
    for label in ('original128', 'fresh128'):
        archive_arrays[label] = np.load(source/f'heldout_{label}.npz')
        compare(label+' unchanged archived nodes', tests[label+'_nodes'], archive_arrays[label]['nodes'], atol=0)
        compare(label+' unchanged direct moments', tests[label+'_direct'], archive_arrays[label]['direct'], atol=0)
    rng = np.random.default_rng(inputs['fresh_seed'])
    new = np.column_stack([rng.uniform(a[0], a[-1], inputs['fresh_count']) for a in axes[:3]]+[rng.uniform(0, 2*np.pi, inputs['fresh_count'])])
    compare('new independent seed and nodes', new, tests['new128_nodes'], atol=0)
    neighborhoods = []
    for label, index in inputs['weak_anchors']:
        anchor = tests[label+'_nodes'][index]
        neighborhood = np.tile(anchor, (inputs['weak_phase_count'], 1))
        neighborhood[:, 3] += np.linspace(-inputs['weak_phase_half_width'], inputs['weak_phase_half_width'], inputs['weak_phase_count'])
        neighborhoods.append(neighborhood)
    compare('weak fixed anchor neighborhoods', np.concatenate(neighborhoods), tests['weak98_nodes'], atol=0)
    kick_table = kicks.reshape(*orders, 2)
    variance_table = (table['moments'][:, 1]-table['moments'][:, 2])/2
    legacy_fields = np.column_stack((table['moments'][:, 0], np.log(variance_table))).reshape(*orders, 2)
    def algebra(nodes, vector):
        wi = nodes[:, 2, None]*np.column_stack((np.cos(nodes[:, 3]), np.sin(nodes[:, 3])))
        c = mass*(2*np.sum(wi*vector, axis=1)+np.sum(vector**2, axis=1))/(8*field)
        variance = mass**2*theta*np.sum(vector**2, axis=1)/(8*field**2)
        return np.column_stack((c, c*c+variance, c*c-variance))
    table_algebra = algebra(table['nodes'], kicks)
    table_second_error = float(np.max(abs(table_algebra[:, 1]/table['moments'][:, 1]-1)))
    compare('table algebra maximum relative second-moment discrepancy', table_second_error, results['table_second_reconstruction_max_relative'], atol=2e-12)
    comparisons, legacy_reproduction = {}, []
    attempt_by_sample = {(r['sample'], r['index']):r for r in attempts}
    for label in ('original128', 'fresh128', 'new128', 'weak98'):
        nodes, direct = tests[label+'_nodes'], tests[label+'_direct']
        if label in archive_arrays:
            rows = [source_rows[tuple(n)] for n in nodes]
        else:
            rows = [attempt_by_sample[(label, j)] for j in range(len(nodes))]
        good = np.asarray([r['status']=='passed' for r in rows])
        identity(label+' every planned row retained', len(rows)==results['comparisons'][label]['states'])
        if np.any(good):
            raw = {k:np.asarray([r[k] for r, flag in zip(rows, good) if flag]) for k in
                ('node', 'initial_relative_position', 'final_relative_position', 'initial_relative_velocity', 'final_relative_velocity', 'time')}
            raw['nodes'] = raw.pop('node')
            check_incoming_frame(label, raw, mass, charge, field, distance, compare)
            actual, direct_kicks, _, _, _ = reconstruct(raw, theta, mass, charge, field, distance)
            scale = np.column_stack((np.sqrt(actual[:, 1]), actual[:, 1], actual[:, 1]))
            compare(label+' direct Cartesian moments / signal scales', actual/scale, direct[good]/scale, atol=2e-7)
            compare(label+' direct zero-plane kicks', direct_kicks, tests[label+'_direct_kicks'][good], atol=2e-12)
        signed_predicted = algebra(nodes, tensor_spline(axes, kick_table, nodes))
        raw_legacy = tensor_spline(axes, legacy_fields, nodes)
        c, variance = raw_legacy[:, 0], np.exp(raw_legacy[:, 1])
        legacy_predicted = np.column_stack((c, c*c+variance, c*c-variance))
        for name, predicted in (('signed', signed_predicted), ('legacy', legacy_predicted)):
            scale = np.column_stack((np.sqrt(np.maximum(predicted[:, 1], np.finfo(float).tiny)), predicted[:, 1], predicted[:, 1]))
            compare(label+' '+name+' separate tensor spline / signal scales', predicted/scale, tests[label+'_'+name]/scale, atol=3e-7)
            stored = results['comparisons'][label][name]
            if not np.all(good):
                identity(label+' '+name+' failed direct states remain unresolved', stored['status']=='unresolved')
                comparisons.setdefault(label, {})[name] = {'status':'unresolved'}
                continue
            relative = abs(predicted[:, 1]/direct[:, 1]-1)
            metric = {'rms':np.linalg.norm(predicted-direct, axis=0)/np.linalg.norm(direct, axis=0),
                'second_p95':float(np.quantile(relative, .95)), 'second_max':float(max(relative)),
                'worst_index':int(np.argmax(relative)), 'minimum_direct_second':float(min(direct[:, 1])),
                'fraction_above_five_percent':float(np.mean(relative>inputs['second_p95_target']))}
            for key, value in metric.items():
                if key=='worst_index':
                    identity(label+' '+name+' '+key, value==stored[key])
                else:
                    compare(label+' '+name+' '+key, value, stored[key], atol=3e-7)
            metric['status'] = 'passed' if metric['rms'][1]<=inputs['second_rms_target'] and metric['second_p95']<=inputs['second_p95_target'] else 'unresolved'
            identity(label+' '+name+' acceptance status', metric['status']==stored['status'])
            if label=='new128':
                ranks = binom.ppf([.025, .975], len(relative), .95).astype(int)
                ordered = np.sort(relative)
                interval = [ordered[max(0, ranks[0]-1)], ordered[min(len(relative)-1, ranks[1])]]
                compare(name+' fresh percentile order-statistic interval', interval, stored['p95_order_statistic_95_interval'], atol=3e-7)
                identity(name+' fresh interval order ranks', [max(1, int(ranks[0])), min(len(relative), int(ranks[1])+1)]==stored['p95_interval_order_ranks'])
                metric['p95_order_statistic_95_interval'] = interval
            comparisons.setdefault(label, {})[name] = metric
        if label in archive_arrays:
            legacy_reproduction.append(float(np.linalg.norm(tests[label+'_legacy']-archive_arrays[label]['predicted'])/np.linalg.norm(archive_arrays[label]['predicted'])))
    compare('legacy unchanged archived predictions', legacy_reproduction, results['legacy_reproduction_relative_norm'], atol=2e-13)
    ordinary = dict(source_rows)
    ordinary.update({tuple(r['node']):r for r in attempts if r['sample']!='solver_refinement'})
    tight = [r for r in attempts if r['sample']=='solver_refinement']
    refinements = []
    for row, saved in zip(tight, results['solver_refinements']):
        identity('tight refinement node selection', row['node']==saved['node'])
        if row['status']=='passed':
            old = ordinary[tuple(row['node'])]['moments']
            change = abs(old[1]/row['moments'][1]-1)
            compare('tight second-moment relative change', change, saved['second_relative_change'], atol=2e-12)
            status = 'passed' if change<=inputs['solver_refinement_target'] else 'unresolved'
        else:
            status = 'unresolved'
        identity('tight refinement status', status==saved['status'])
        refinements.append(status)
    identity('tight refinement planned count', len(tight)==len(results['solver_refinements']))
    complete = all(r['status']=='passed' for r in attempts)
    gates = dict(complete_new_states=complete,
        signed_original_targets=all(comparisons[label]['signed']['status']=='passed' for label in comparisons),
        fresh_p95_confidence=complete and comparisons['new128']['signed']['p95_order_statistic_95_interval'][1]<=inputs['second_p95_target'],
        sampled_max=complete and all(comparisons[label]['signed']['second_max']<=inputs['sampled_max_target'] for label in comparisons),
        solver_refinement=all(s=='passed' for s in refinements),
        energy=complete and max(r['energy_error'] for r in attempts)<=inputs['energy_target'],
        table_algebra=table_second_error<=1e-8,
        legacy_reproduction=max(legacy_reproduction)<=1e-10,
        source_integrity=all(digest(ROOT/name)==wanted for name, wanted in summary['source_artifact_sha256'].items()),
        dependency_integrity=summary['experiment_dependency_sha256']==summary['experiment_dependency_sha256_end'])
    gates = {k:bool(v) for k, v in gates.items()}
    identity('interpolation all independently rebuilt gates', gates==results['required_gates'])
    identity('interpolation overall status retains every gate', results['overall_status']==('passed' if all(gates.values()) else 'unresolved'))
    identity('interpolation unique new encounter count', len(attempts)==results['new_encounters'])
    return {'producer_status':results['overall_status'], 'reconstructed_gates':gates,
        'table_states':len(expected_nodes), 'new_states':len(attempts),
        'independent_interpolation_route':'Sequential SciPy make_interp_spline, not-a-knot coordinates and periodic phase',
        'sampled_maximum_signed_errors':{k:float(v['signed']['second_max']) if 'second_max' in v['signed'] else None for k, v in comparisons.items()}}


def audit_broad(directory, compare, identity):
    summary = load(directory/'summary.json')
    inputs, saved = summary['inputs'], summary['results']
    arrays = np.load(directory/'states.npz')
    populations = np.load(directory/'populations.npz')
    theta, mass, charge, field = (inputs[k] for k in ('theta', 'mass', 'charge', 'field'))
    passed = arrays['status']=='passed'
    retained = {k:arrays[k][passed] for k in arrays.files}
    check_incoming_frame('broad', retained, mass, charge, field, retained['distance'], compare)
    moments, _, incoming, gc, spatial = reconstruct(retained, theta, mass, charge, field, retained['distance'])
    # Signed means use a positive signal scale; a tiny absolute tolerance must
    # not swallow the second moment in a weak trajectory.
    scales = np.column_stack((np.sqrt(moments[:, 1]), moments[:, 1], moments[:, 1]))
    compare('broad Cartesian COM moments / positive scales', moments/scales, retained['moments']/scales, atol=2e-7)
    compare('broad incoming-frame kick', incoming, retained['delta_relative_perpendicular'], atol=2e-12)
    compare('broad GC displacement', gc, retained['gc_relative_kick'], atol=2e-12)
    compare('broad individual GC second moment', spatial, retained['individual_gc_second'], atol=2e-13)
    identity('broad no nonfinite accepted state', np.all(np.isfinite(moments)) and np.all(moments[:, 1]>0))
    all_values = np.full((len(passed), 4), np.nan)
    all_values[passed] = np.column_stack((moments, spatial))
    flux, mu_ref = bounded_flux(inputs['bounds'], theta), mass*theta/field
    nmax, pmax = max(inputs['sample_levels']), max(inputs['phase_levels'])
    extended = inputs.get('mode', 'base')=='extend'
    previous_count = sorted(inputs['sample_levels'])[-2]
    fine_label, previous_label = f'primary{nmax}_phase{pmax}', f'primary{previous_count}_phase{pmax}'
    phase_label, replica_label = f'primary{nmax}_phase{min(inputs["phase_levels"])}', f'replica{nmax}_phase{pmax}'
    phase = 2*np.pi*np.arange(pmax)/pmax
    sample_nodes = []
    for seed in inputs['seeds']:
        unit = qmc.Sobol(3, scramble=True, seed=seed).random_base2(int(np.log2(nmax)))
        sample = inverse_flux(unit, inputs['bounds'], theta)
        sample_nodes.append(np.column_stack((np.repeat(sample, pmax, axis=0), np.tile(phase, nmax))))
    specs = [(case['case'], inputs['seeds'].index(case['seed']), case['sample_count'],
        pmax//case['phase_count'], case['distance']) for case in saved['cases']]
    recomputed, completeness = {}, {}
    for label, seed_index, count, stride, distance in specs:
        ids = populations[label]
        wanted = sample_nodes[seed_index].reshape(nmax, pmax, 4)[:count, ::stride].reshape(-1, 4)
        identity(label+' expected population length', len(ids)==len(wanted))
        present = ids>=0
        identity(label+' ID range', np.all(ids[present]<len(passed)))
        compare(label+' Sobol and phase nodes', arrays['nodes'][ids[present]], wanted[present], atol=2e-14)
        compare(label+' event distances', arrays['distance'][ids[present]], np.full(sum(present), distance), atol=0)
        # The ensemble uses the producer's declared default max_step=.2;
        # step_levels is the distinct (.1,.05,.025) control hierarchy.
        compare(label+' active step limit', arrays['step'][ids[present]], np.full(sum(present), .2), atol=0)
        complete = bool(np.all(present) and np.all(passed[ids[present]]))
        completeness[label] = complete
        case = next(r for r in saved['cases'] if r['case']==label)
        identity(label+' completeness status', complete==case['complete'])
        identity(label+' missing weights retained', int(np.count_nonzero(~present))+int(np.count_nonzero(~passed[ids[present]]))==case['missing_count'])
        compare(label+' exact flux', flux, case['flux_per_density'], atol=2e-14)
        if complete:
            integral = flux*np.mean(all_values[ids], axis=0)
            recomputed[label] = integral
            compare(label+' integral', integral, case['integrals'], atol=2e-12)
            compare(label+' diffusion half factor', integral[1]/2, case['diffusion_per_density'], atol=2e-12)
            compare(label+' reflected fraction', np.mean(arrays['exit'][ids]=='reflected'), case['reflected_fraction'], atol=0)
    compare('broad thermal crossing flux retained fraction', flux/(2*np.pi*(inputs['bounds'][0][1]**2-inputs['bounds'][0][0]**2)*np.sqrt(theta/np.pi)), saved['velocity_flux_retained_fraction'])
    relationships = {'sample':(fine_label, previous_label),
        'phase':(fine_label, phase_label), 'replica':(fine_label, replica_label),
        'boundary_ensemble':('primary512_phase64_L32', 'primary512_phase64' if extended else fine_label)}
    if extended:
        relationships.update(sample_replica=(replica_label, f'replica{previous_count}_phase{pmax}'),
            history_primary512_to1024=(previous_label, 'primary512_phase64'),
            history_replica512_to1024=(f'replica{previous_count}_phase{pmax}', 'replica512_phase64'))
    computed_checks = {}
    for name, (a, b) in relationships.items():
        computed_checks[name] = scaled_changes(recomputed[a], recomputed[b], mu_ref) if a in recomputed and b in recomputed else None
        if computed_checks[name] is None:
            identity(name+' missing comparison remains unresolved', saved['comparisons'][name] is None)
        else:
            compare('broad '+name+' scaled changes', computed_checks[name], saved['comparisons'][name], atol=3e-10)
    control = saved['controls']
    primary = populations[fine_label]
    if completeness[fine_label]:
        eligible = np.arange(len(primary)) >= (512*pmax if extended else 0)
        top = np.flatnonzero(eligible)[np.argsort(all_values[primary[eligible], 1])[-inputs['control_count_each']:]]
        fresh_unit = qmc.Sobol(3, scramble=True, seed=3903).random_base2(4)
        fresh = inverse_flux(fresh_unit, inputs['bounds'], theta)
        fresh_phase = 2*np.pi*qmc.Sobol(1, scramble=True, seed=3904).random_base2(4).ravel()
        wanted = np.vstack((sample_nodes[0][top], np.column_stack((fresh, fresh_phase))))
        compare('broad largest-contributor and fresh controls', wanted, control['nodes'], atol=2e-14)
        identity('broad control selection labels', control['selection']==['largest_new_primary_second' if extended else 'largest_primary_second']*16+['fresh_predetermined_flux']*16)
        if 'primary_indices' in control:
            compare('broad contributor control indices', top, control['primary_indices'], atol=0)
        if extended:
            plane_values = []
            for distance in inputs['distances']:
                ids = np.asarray(control[f'plane_{distance:g}'])
                valid = ids>=0
                compare(f'broad new plane {distance} nodes', arrays['nodes'][ids[valid]], wanted[valid], atol=2e-14)
                compare(f'broad new plane {distance} distance', arrays['distance'][ids[valid]], np.full(sum(valid), distance), atol=0)
                compare(f'broad new plane {distance} step', arrays['step'][ids[valid]], np.full(sum(valid), .2), atol=0)
                plane_values.append(all_values[ids] if np.all(valid) and np.all(passed[ids]) else None)
            a, b = plane_values[-1], plane_values[-2]
            if a is not None and b is not None:
                scales = np.column_stack((a[:, 1]/mu_ref, a[:, 1], a[:, 1], a[:, 3]))
                computed_checks['boundary_new_states'] = np.max(abs(a-b)/scales, axis=0)
                compare('broad new contributor plane changes', computed_checks['boundary_new_states'], saved['comparisons']['boundary_new_states'], atol=3e-10)
            else:
                computed_checks['boundary_new_states'] = None
                identity('broad missing new-plane comparison unresolved', saved['comparisons']['boundary_new_states'] is None)
        for step in inputs['step_levels']:
            ids = np.asarray(control[str(step)])
            valid = ids>=0
            compare(f'broad control nodes step {step}', arrays['nodes'][ids[valid]], wanted[valid], atol=2e-14)
            compare(f'broad control step {step}', arrays['step'][ids[valid]], np.full(sum(valid), step), atol=0)
        fine_ids, mid_ids = map(lambda k:np.asarray(control[str(k)]), inputs['step_levels'][-2:][::-1])
        if np.all(fine_ids>=0) and np.all(mid_ids>=0) and np.all(passed[fine_ids]) and np.all(passed[mid_ids]):
            a, b = all_values[fine_ids], all_values[mid_ids]
            scale = np.column_stack((a[:, 1]/mu_ref, a[:, 1], a[:, 1], a[:, 3]))
            computed_checks['active_step'] = np.max(abs(a-b)/scale, axis=0)
            ratios = arrays['max_actual_step'][fine_ids]/arrays['max_actual_step'][mid_ids]
            compare('broad active step changes', computed_checks['active_step'], saved['comparisons']['active_step'], atol=3e-9)
            compare('broad active actual-step ratios', ratios, control['actual_step_ratios'], atol=2e-14)
        else:
            computed_checks['active_step'], ratios = None, None
    else:
        computed_checks['active_step'], ratios = None, None
        if extended:
            computed_checks['boundary_new_states'] = None
    if extended:
        phase_changes, expected_phase_controls = [], []
        for seed_index, label in enumerate((fine_label, replica_label)):
            ids = populations[label]
            if not completeness[label]:
                continue
            averaged = all_values[ids, 1].reshape(nmax, pmax).mean(axis=1)
            selected = np.argsort(averaged)[-inputs['influential_per_seed']:]
            expected_phase_controls.extend((inputs['seeds'][seed_index], int(i)) for i in selected)
        identity('broad influential speed selection and order', expected_phase_controls==[(r['seed'], r['sample_index']) for r in saved['phase_controls']])
        for row in saved['phase_controls']:
            seed_index = inputs['seeds'].index(row['seed'])
            point = sample_nodes[seed_index].reshape(nmax, pmax, 4)[row['sample_index'], 0, :3]
            compare('broad influential incoming speed', point, row['point'], atol=2e-14)
            primary_or_replica = populations[fine_label if seed_index==0 else replica_label]
            average = np.mean(all_values[primary_or_replica, 1].reshape(nmax, pmax)[row['sample_index']])
            compare('broad influential original phase average', average, row['phase_average_second'], atol=2e-13)
            values = []
            for count in [pmax]+inputs['influential_phase_levels']:
                ids = np.asarray(row[f'phase{count}_ids'])
                if count==pmax:
                    compare('broad influential original population IDs', ids,
                        primary_or_replica.reshape(nmax, pmax)[row['sample_index']], atol=0)
                wanted = np.column_stack((np.repeat(point[None], count, axis=0), 2*np.pi*np.arange(count)/count))
                valid = ids>=0
                compare(f'broad influential {count} phases', arrays['nodes'][ids[valid]], wanted[valid], atol=2e-14)
                compare(f'broad influential {count} plane', arrays['distance'][ids[valid]], np.full(sum(valid), inputs['distances'][0]), atol=0)
                compare(f'broad influential {count} step', arrays['step'][ids[valid]], np.full(sum(valid), .2), atol=0)
                vector = np.mean(all_values[ids], axis=0) if np.all(valid) and np.all(passed[ids]) else None
                if vector is None:
                    identity(f'broad influential {count} missing vector unresolved', row[f'phase{count}_observables'] is None)
                else:
                    compare(f'broad influential {count} reconstructed observable', vector, row[f'phase{count}_observables'], atol=2e-13)
                values.append(vector)
            if all(v is not None for v in values):
                final_scale = np.array([values[-1][1]/mu_ref, values[-1][1], values[-1][1], values[-1][3]])
                change = abs(values[-1]-values[-2])/final_scale
                early_change = abs(values[-2]-values[-3])/final_scale
            else:
                change, early_change = None, None
            if change is None:
                identity('broad influential missing change unresolved', row['changes'] is None)
            else:
                compare('broad influential per-speed convergence', change, row['changes'], atol=3e-10)
            if early_change is None:
                identity('broad influential missing earlier change unresolved', row['changes64_to128'] is None)
            else:
                compare('broad influential per-speed earlier convergence', early_change, row['changes64_to128'], atol=3e-10)
            phase_changes.extend((early_change, change))
        computed_checks['phase_influential'] = np.max(phase_changes, axis=0) if phase_changes and all(c is not None for c in phase_changes) and completeness[fine_label] and completeness[replica_label] else None
        if computed_checks['phase_influential'] is None:
            identity('broad influential unresolved global comparison', saved['comparisons']['phase_influential'] is None)
        else:
            compare('broad worst influential phase changes', computed_checks['phase_influential'], saved['comparisons']['phase_influential'], atol=3e-10)
        prior = ROOT/'results/broad_speed_refinement'
        prior_arrays = np.load(prior/'states.npz')
        identity('broad retained state count', saved['reused_encounters']==len(prior_arrays['nodes']))
        for key in prior_arrays.files:
            identity('broad bit-identical prior state '+key, np.array_equal(arrays[key][:len(prior_arrays[key])], prior_arrays[key]))
        identity('broad new-state count', saved['new_encounters']==len(passed)-saved['reused_encounters'])
    def gate(values, target):
        return values is not None and bool(np.all(np.isfinite(values)) and max(values)<=target)
    gates = {name:gate(computed_checks[name], inputs['moment_target']) for name in ('sample', 'phase', 'replica')}
    gates.update(boundary_ensemble=gate(computed_checks['boundary_ensemble'], inputs['boundary_target']),
        active_step=gate(computed_checks['active_step'], inputs['step_target']),
        active_step_sizes=ratios is not None and bool(len(ratios)) and bool(np.all(np.isfinite(ratios)) and np.all(ratios<=.6)),
        complete=all(completeness.values()) and np.all(passed) and not saved['cap_exhausted'],
        energy=bool(len(passed)) and bool(np.all(passed) and np.all(arrays['energy_error']<=inputs['energy_target'])),
        source_integrity=summary['experiment_dependency_sha256']==summary['experiment_dependency_sha256_end'])
    if extended:
        gates.update(sample_replica=gate(computed_checks['sample_replica'], inputs['moment_target']),
            phase_influential=gate(computed_checks['phase_influential'], inputs['moment_target']),
            boundary_new_states=gate(computed_checks['boundary_new_states'], inputs['boundary_target']),
            prior_integrity=all(digest(ROOT/'results/broad_speed_refinement'/name)==wanted for name, wanted in inputs['prior_artifact_sha256'].items()))
    gates = {k:bool(v) for k, v in gates.items()}
    identity('broad independently rebuilt gates', gates==saved['required_gates'])
    identity('broad overall status retains every gate', saved['status']==('passed' if all(gates.values()) else 'unresolved'))
    identity('broad unique encounter count', len(passed)==saved['unique_encounters'])
    identity('broad failure count', int(np.count_nonzero(~passed))==saved['failures'])
    candidates = select_cartesian_replays(arrays, primary, all_values)
    replays = [fresh_cartesian_replay(arrays, i, inputs) for i in candidates]
    for row in replays:
        identity('broad independent Cartesian path '+str(row['index']), row['passed'])
    return {'producer_status':saved['status'], 'reconstructed_gates':gates,
        'accepted_states':int(sum(passed)), 'missing_or_failed_states':int(sum(~passed)),
        'exact_bounded_flux':float(flux), 'selected_cartesian_replays':replays,
        'comparisons':{k:v.tolist() if v is not None else None for k, v in computed_checks.items()}}


if OUTPUT.exists():
    raise FileExistsError(f'Preserve existing audit: {OUTPUT}; choose a fresh SM_REFINEMENT_AUDIT_OUTPUT')
if not (INTERPOLATION/'summary.json').exists() or not (BROAD/'summary.json').exists():
    raise FileNotFoundError('Both declared refinement summaries must exist before the joint audit')
OUTPUT.mkdir(parents=True)
start = perf_counter()
failures, checks = [], []
def compare(name, actual, saved, rtol=2e-7, atol=2e-13):
    a, b = np.asarray(actual), np.asarray(saved)
    good = a.shape==b.shape and np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.allclose(a, b, rtol=rtol, atol=atol)
    if not good:
        failures.append(name)
    checks.append({'name':name, 'passed':bool(good), 'maximum_absolute_error':float(np.max(abs(a-b))) if a.shape==b.shape and a.size else None})
def identity(name, condition):
    if not condition:
        failures.append(name)
    checks.append({'name':name, 'passed':bool(condition)})
# Audit sections below are independent and preserve unresolved producer gates.
interpolation = audit_interpolation(INTERPOLATION, compare, identity)
broad = audit_broad(BROAD, compare, identity)
producer_source_identities = {directory.name:historical_source_identity(directory, load(directory/'summary.json'), identity)
    for directory in (INTERPOLATION, BROAD)}
sources = {str(p.relative_to(ROOT)):digest(p) for directory in (INTERPOLATION, BROAD)
    for p in directory.iterdir() if p.is_file() and p.name in ('summary.json', 'declared_inputs.json', 'states.npz', 'populations.npz', 'table.npz', 'tests.npz', 'attempts.json')}
report = {'status':'passed' if not failures else 'failed', 'failures':failures,
    'checks':checks, 'interpolation':interpolation, 'broad_speed':broad,
    'producer_source_identities':producer_source_identities,
    'input_sha256':sources, 'audit_commit':subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'audit_working_tree':subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True).strip(),
    'audit_source_sha256':digest(Path(__file__).resolve()), 'python':sys.version,
    'platform':platform.platform(), 'wall_s':perf_counter()-start,
    'versions':{name:importlib.metadata.version(name) for name in ('numpy', 'scipy')},
    'parent_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
    'scope':'Saved Cartesian-state and sampling reconstruction; selected independent twelve-coordinate paths. No infinite-domain or plasma calibration inference.'}
(OUTPUT/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
print(f'Audit {report["status"]}: {len(checks)} checks, {len(failures)} failures; {report["wall_s"]:.2f} s', flush=True)
if failures:
    raise RuntimeError(f'Independent refinement audit failed: {failures}; saved {OUTPUT}')
