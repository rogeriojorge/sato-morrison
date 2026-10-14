"""Fixed angular and endpoint diagnosis of one unresolved close encounter.

This is a single incoming-speed state, not a repaired broad-band integral.
Uniform phase weights retain reflected trajectories and every failed weight.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
import hashlib
import json
import multiprocessing as mp
import os
import resource
import sys
import numpy as np
from sato_morrison.controls import encounter_relative, encounter_thermal_moments, binary_encounter
from sato_morrison.reference import run_metadata

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT/'results/broad_speed_extended'
ORIGINAL_AUDIT = ROOT/'results/refinement_audit_extended_complete/summary.json'
OUTPUT = ROOT/os.environ.get('SM_ENCOUNTER_PEAK_OUTPUT', 'results/encounter_peak_resolution')
SEED, SAMPLE_INDEX, REFLECTED_STATE_ID = 3902, 1155, 237872
MASS, CHARGE, FIELD, STRENGTH, SCREENING, THETA = 1., 1., 2., .03, 3., .5
PHASE_LEVELS, UNION_COUNT = (512, 1024, 2048), 4096
DISTANCES, STEPS = (32., 40., 48.), (.1, .05, .025)
RTOL, FLIGHT_FACTOR, BASE_STEP = 1e-12, 48., .2
WORKERS, CHECKPOINT, CAP = 4, 512, 900.
PHASE_TARGET, PLANE_TARGET, STEP_TARGET, ENERGY_TARGET = .05, .005, .001, 1e-9
CARTESIAN_TARGET, CARTESIAN_STEP, CARTESIAN_RTOL = 1e-8, .0125, 2e-13
TIGHT_RELATIVE_STEP, TIGHT_RELATIVE_RTOL = .0125, 2e-13
prior = json.loads((PRIOR/'summary.json').read_text())
original_audit = json.loads(ORIGINAL_AUDIT.read_text())
original_cartesian_failure = next(r for r in original_audit['broad_speed']['selected_cartesian_replays'] if r['index']==REFLECTED_STATE_ID)
selected = next(p for p in prior['results']['phase_controls'] if p['seed']==SEED and p['sample_index']==SAMPLE_INDEX)
SPEED = np.asarray(selected['point'])
WEAK_NODE = np.asarray(prior['results']['controls']['nodes'][25])
mu_ref = MASS*THETA/FIELD
INPUTS = dict(seed=SEED, sample_index=SAMPLE_INDEX, speed=SPEED.tolist(), weak_node=WEAK_NODE.tolist(),
    required_reflected_prior_state_id=REFLECTED_STATE_ID,
    mass=MASS, charge=CHARGE, field=FIELD, strength=STRENGTH, screening=SCREENING, theta=THETA,
    phase_levels=PHASE_LEVELS, halfshift_count=2048, union_count=UNION_COUNT,
    distances=DISTANCES, steps=STEPS, rtol=RTOL, flight_time_factor=FLIGHT_FACTOR, base_step=BASE_STEP,
    workers=WORKERS, runtime_cap_s=CAP, phase_target=PHASE_TARGET, plane_target=PLANE_TARGET,
    step_target=STEP_TARGET, energy_target=ENERGY_TARGET, cartesian_absolute_target=CARTESIAN_TARGET,
    cartesian_max_step=CARTESIAN_STEP, cartesian_rtol=CARTESIAN_RTOL,
    required_reflected_tight_relative_step=TIGHT_RELATIVE_STEP,
    required_reflected_tight_relative_rtol=TIGHT_RELATIVE_RTOL,
    controls='largest four M2 phases, closest four sampled separations, longest two reflected exits if present, exact prior reflected state 237872 at pi/2, four fresh phases seed4101; deduplicated; independent Cartesian replay of largest M2, closest separation, longest reflected and required prior reflected control',
    normalization='mean: fine M2/mu_reference; second and cross: fine M2; spatial: fine individual GC second',
    phase_measure='uniform incoming gyrophase; each state has weight 1/N, reflected exits retain their original weights',
    halfshift_scope='regular 2048 and half-step-shifted 2048 union equals regular 4096; a deterministic refinement, not independent statistical evidence',
    acceptance='1024 to 2048,2048 to 4096 and regular-versus-shifted 2048 angular gates; same angular checks at L48; final L40 to L48 integral and weak-node gates; all active-step, energy, Cartesian and integrity gates',
    stop_rule='fixed levels, distances and controls; no outcome-dependent refinement; cap between batches; missing weights invalidate their integral',
    scope='conditional angular average at one fixed speed/impact and separate weak-state endpoint probe; preserves failed example 39 integral; no plasma rate, closure or lifetime',
    units='normalized; theta=kBT/m; U=kappa exp(-r/lambda)/r',
    prior_artifact_sha256={name: hashlib.sha256((PRIOR/name).read_bytes()).hexdigest() for name in ('summary.json', 'states.npz')})
INPUTS['original_cartesian_audit'] = dict(path=str(ORIGINAL_AUDIT.relative_to(ROOT)),
    sha256=hashlib.sha256(ORIGINAL_AUDIT.read_bytes()).hexdigest(), preserved_failed_comparison=original_cartesian_failure)
for name in ('mass', 'charge', 'field', 'strength', 'screening', 'theta', 'rtol', 'flight_time_factor'):
    if prior['inputs'][name] != INPUTS[name]:
        raise ValueError(f'Prior physical/numerical input differs: {name}')
for name in ('src/sato_morrison/controls.py', 'src/sato_morrison/reference.py'):
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=prior['experiment_dependency_sha256'][name]:
        raise ValueError(f'Prior helper differs: {name}')


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)


def solve(spec):
    node, distance, step = spec
    started = perf_counter()
    row = dict(node=list(node), distance=distance, step=step)
    try:
        b, u, p, phase = node
        r = encounter_relative(b, phase, u, p, field=FIELD, strength=STRENGTH,
            screening=SCREENING, start_distance=distance, mass=MASS, charge=CHARGE,
            max_step=step, rtol=RTOL, flight_time_factor=FLIGHT_FACTOR)
        def gc(position, velocity):
            return position[:2]+np.array([velocity[1], -velocity[0]])/(CHARGE*FIELD/MASS)
        kick = gc(r['final_relative_position'], r['final_relative_velocity'])-gc(r['initial_relative_position'], r['initial_relative_velocity'])
        row.update(status='passed', moments=encounter_thermal_moments(r, THETA, MASS, FIELD).tolist(),
            individual_gc_second=float(kick@kick/4), gc_relative_kick=kick.tolist(),
            **{k:r[k].tolist() for k in ('initial_relative_position', 'final_relative_position', 'initial_relative_velocity', 'final_relative_velocity', 'delta_relative_perpendicular')},
            **{k:r[k] for k in ('time', 'exit', 'energy_error', 'min_separation', 'max_actual_step', 'evaluations', 'parallel_turns')})
        if not np.all(np.isfinite(row['moments'])) or not np.isfinite(kick@kick):
            raise RuntimeError('Nonfinite encounter observable')
    except Exception as error:
        row.update(status='unresolved', reason=str(error), error_type=type(error).__name__,
            diagnostics=getattr(error, 'integration_diagnostics', {}))
    row.update(wall_s=perf_counter()-started, worker_peak_rss_bytes=rss())
    return row


def save_states(path, rows):
    arrays = dict(nodes=np.asarray([r['node'] for r in rows]), distance=np.asarray([r['distance'] for r in rows]),
        step=np.asarray([r['step'] for r in rows]), status=np.asarray([r['status'] for r in rows]))
    for name, width in (('moments', 3), ('initial_relative_position', 3), ('final_relative_position', 3),
            ('initial_relative_velocity', 3), ('final_relative_velocity', 3),
            ('delta_relative_perpendicular', 2), ('gc_relative_kick', 2)):
        arrays[name] = np.asarray([r.get(name, [np.nan]*width) for r in rows])
    for name in ('time', 'energy_error', 'min_separation', 'max_actual_step', 'evaluations',
            'parallel_turns', 'individual_gc_second', 'wall_s', 'worker_peak_rss_bytes'):
        arrays[name] = np.asarray([r.get(name, np.nan) for r in rows])
    arrays['exit'] = np.asarray([r.get('exit', 'missing') for r in rows])
    np.savez_compressed(path, **arrays)


if OUTPUT.exists():
    raise FileExistsError(f'Preserve existing evidence: {OUTPUT}; choose a fresh destination')
OUTPUT.mkdir(parents=True)
(OUTPUT/'declared_inputs.json').write_text(json.dumps(INPUTS, indent=2)+'\n')
from jax._src import xla_bridge
if xla_bridge.backends_are_initialized():
    raise RuntimeError('Run in a fresh process before the JAX backend initializes')
pool = ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context('fork'))
list(pool.map(float, range(WORKERS)))
import jax
jax.config.update('jax_enable_x64', True)
metadata = run_metadata(INPUTS, model='fixed-state close encounter phase and endpoint diagnosis',
    boundary='first outgoing relative z=+/-L planes; reflected retained', units=INPUTS['units'])
dependencies = [Path(__file__).resolve(), ROOT/'src/sato_morrison/controls.py', ROOT/'src/sato_morrison/reference.py']
hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
metadata['experiment_dependency_sha256'] = hashes
started = perf_counter()
attempts, cache, populations, cases, failures = [], {}, {}, [], []
with np.load(PRIOR/'states.npz') as archive:
    old_ids = np.asarray(selected['phase256_ids'])
    subset = {name:archive[name][old_ids].copy() for name in archive.files}
    for index in range(len(old_ids)):
        row = {name:values[index].tolist() for name,values in subset.items()}
        row['node'] = row.pop('nodes')
        if row['status']!='passed':
            raise ValueError('Prior selected phase state is missing')
        cache[(tuple(row['node']), row['distance'], row['step'])] = len(attempts)
        attempts.append(row)
    required_initial = np.concatenate((archive['initial_relative_position'][REFLECTED_STATE_ID], archive['initial_relative_velocity'][REFLECTED_STATE_ID]))
    required_node = archive['nodes'][REFLECTED_STATE_ID].copy()
    if not np.array_equal(required_node, np.r_[SPEED, np.pi/2]):
        raise ValueError('Required reflected node differs from the declared one')
reused_count = len(attempts)
cap_exhausted = False
angles = 2*np.pi*np.arange(UNION_COUNT)/UNION_COUNT
if not np.array_equal(np.asarray([r['node'][3] for r in attempts]), angles[::UNION_COUNT//256]):
    raise ValueError('Retained phase256 prefix is not bit-identical')


def evaluate(nodes, distance=24., step=BASE_STEP):
    global cap_exhausted
    specs = [(tuple(map(float, n)), float(distance), float(step)) for n in nodes]
    pending = list(dict.fromkeys(s for s in specs if s not in cache))
    for offset in range(0, len(pending), CHECKPOINT):
        if perf_counter()-started>CAP:
            cap_exhausted = True
            break
        batch = pending[offset:offset+CHECKPOINT]
        rows = list(pool.map(solve, batch))
        for spec, row in zip(batch, rows):
            cache[spec] = len(attempts)
            attempts.append(row)
            if row['status']!='passed':
                failures.append(row)
                print('FAILED:', row, flush=True)
        save_states(OUTPUT/f'checkpoint_{len(attempts):06d}.npz', rows)
        (OUTPUT/'failures.json').write_text(json.dumps(failures, indent=2, allow_nan=False)+'\n')
        print(f'Saved {len(attempts)-reused_count} new states; wall {perf_counter()-started:.1f}s', flush=True)
    ids = np.asarray([cache.get(s, -1) for s in specs], dtype=int)
    rows = [attempts[i] if i>=0 else {'status':'not_run'} for i in ids]
    return rows, ids


def vector(row):
    return row['moments']+[row['individual_gc_second']]


def integrate(label, count, distance=24., shifted=False):
    phase = angles[1::2] if shifted else angles[::UNION_COUNT//count]
    nodes = np.column_stack((np.repeat(SPEED[None], len(phase), axis=0), phase))
    rows, ids = evaluate(nodes, distance)
    populations[label] = ids
    complete = all(r['status']=='passed' for r in rows)
    case = dict(case=label, phase_count=len(phase), distance=distance, shifted=shifted,
        complete=complete, missing_count=sum(r['status']!='passed' for r in rows))
    if complete:
        case.update(observables=np.asarray([vector(r) for r in rows]).mean(axis=0).tolist(),
            reflected_count=sum(r['exit']=='reflected' for r in rows),
            max_energy_error=max(r['energy_error'] for r in rows),
            maximum_time=max(r['time'] for r in rows))
    cases.append(case)
    (OUTPUT/'cases_partial.json').write_text(json.dumps(cases, indent=2)+'\n')
    print(label, case.get('observables'), 'reflected', case.get('reflected_count'), flush=True)
    return case


old = integrate('phase256_L24_retained', 256)
phase_cases = [integrate(f'phase{n}_L24', n) for n in PHASE_LEVELS]
shifted = integrate('phase2048_L24_shifted', 2048, shifted=True)
fine = integrate('phase4096_L24_union', UNION_COUNT)
planes = [integrate(f'phase4096_L{d:g}', UNION_COUNT, d) for d in DISTANCES]
plane_coarse = integrate('phase2048_L48', 2048, DISTANCES[-1])
plane_shifted = integrate('phase2048_L48_shifted', 2048, DISTANCES[-1], shifted=True)


def change(a, b, reference=None):
    if not a['complete'] or not b['complete'] or (reference is not None and not reference['complete']):
        return None
    x, y = np.asarray(a['observables']), np.asarray(b['observables'])
    r = np.asarray((reference or a)['observables'])
    scale = np.array([r[1]/mu_ref, r[1], r[1], r[3]])
    return (abs(x-y)/scale).tolist() if np.all(scale>0) else None


checks = dict(phase1024_to2048=change(phase_cases[-1], phase_cases[-2], fine),
    phase2048_to4096=change(fine, phase_cases[-1]),
    halfshift=change(phase_cases[-1], shifted, fine),
    final_plane_phase=change(planes[-1], plane_coarse),
    final_plane_halfshift=change(plane_coarse, plane_shifted, planes[-1]),
    boundary_integral=change(planes[-1], planes[-2]))
weak = []
for distance in DISTANCES:
    rows, ids = evaluate(WEAK_NODE[None], distance)
    weak.append(dict(case=f'weak_L{distance:g}', complete=rows[0]['status']=='passed',
        ids=ids.tolist(), observables=vector(rows[0]) if rows[0]['status']=='passed' else None))
checks['boundary_weak_state'] = change(weak[-1], weak[-2])
controls = dict(nodes=[], ids={})
cartesian = []
tight_relative = []
if fine['complete']:
    ids = populations['phase4096_L24_union']
    rows = [attempts[i] for i in ids]
    m2 = np.asarray([r['moments'][1] for r in rows])
    separation = np.asarray([r['min_separation'] for r in rows])
    reflected = np.asarray([i for i, r in enumerate(rows) if r['exit']=='reflected'], dtype=int)
    chosen = list(np.argsort(m2)[-4:])+list(np.argsort(separation)[:4])
    required_phase_index = UNION_COUNT//4
    if not np.array_equal(np.concatenate((rows[required_phase_index]['initial_relative_position'], rows[required_phase_index]['initial_relative_velocity'])), required_initial):
        raise ValueError('Required reflected Cartesian initial state changed bits')
    chosen.append(required_phase_index)
    if len(reflected):
        chosen += list(reflected[np.argsort([rows[i]['time'] for i in reflected])[-2:]])
    chosen = list(dict.fromkeys(map(int, chosen)))
    fresh = np.random.default_rng(4101).uniform(0, 2*np.pi, 4)
    control_nodes = np.vstack((np.asarray([rows[i]['node'] for i in chosen]),
        np.column_stack((np.repeat(SPEED[None], len(fresh), axis=0), fresh))))
    controls.update(nodes=control_nodes.tolist(), selected_phase_indices=chosen)
    step_vectors = []
    for step in STEPS:
        states, control_ids = evaluate(control_nodes, step=step)
        controls['ids'][str(step)] = control_ids.tolist()
        step_vectors.append(np.asarray([vector(r) if r['status']=='passed' else [np.nan]*4 for r in states]))
    a, b = step_vectors[-1], step_vectors[-2]
    scale = np.column_stack((a[:, 1]/mu_ref, a[:, 1], a[:, 1], a[:, 3]))
    checks['active_step'] = np.max(abs(a-b)/scale, axis=0).tolist() if np.all(np.isfinite(a+b)) and np.all(scale>0) else None
    controls['actual_step_ratios'] = [attempts[i]['max_actual_step']/attempts[j]['max_actual_step'] if i>=0 and j>=0 and attempts[i]['status']=='passed' and attempts[j]['status']=='passed' else None for i,j in zip(controls['ids'][str(STEPS[-1])], controls['ids'][str(STEPS[-2])])]
    replay_indices = [int(np.argmax(m2)), int(np.argmin(separation)), required_phase_index]
    if len(reflected):
        replay_indices.append(int(reflected[np.argmax([rows[i]['time'] for i in reflected])]))
    for phase_index in dict.fromkeys(replay_indices):
        node = rows[phase_index]['node']
        finest, relative_ids = evaluate([node], step=STEPS[-1])
        r = finest[0]
        record = dict(node=node, relative_id=int(relative_ids[0]))
        if r['status']=='passed' and perf_counter()-started<CAP:
            try:
                pair = binary_encounter(r['initial_relative_position'], r['initial_relative_velocity'],
                    field=FIELD, strength=STRENGTH, screening=SCREENING, mass=MASS, charge=CHARGE,
                    duration=r['time'], max_step=CARTESIAN_STEP, rtol=CARTESIAN_RTOL)
                position = pair['positions'][-1, 0]-pair['positions'][-1, 1]
                velocity = pair['velocities'][-1, 0]-pair['velocities'][-1, 1]
                record.update(status='passed', final_position=position.tolist(), final_velocity=velocity.tolist(),
                    position_error=float(max(abs(position-np.asarray(r['final_relative_position'])))),
                    velocity_error=float(max(abs(velocity-np.asarray(r['final_relative_velocity'])))),
                    energy_error=pair['energy_error'])
            except Exception as error:
                record.update(status='unresolved', reason=str(error))
        else:
            record.update(status='not_run', reason='missing relative state or fixed cap')
        cartesian.append(record)
    required_ids = controls['ids'][str(STEPS[-1])]
    required_index = next(i for i,n in enumerate(control_nodes) if np.array_equal(n, required_node))
    required_fine = attempts[required_ids[required_index]] if required_ids[required_index]>=0 else None
    record = dict(node=required_node.tolist(), original_coarse_prior_id=REFLECTED_STATE_ID,
        max_step=TIGHT_RELATIVE_STEP, rtol=TIGHT_RELATIVE_RTOL)
    if required_fine is not None and required_fine['status']=='passed' and perf_counter()-started<CAP:
        try:
            b,u,p,phase = required_node
            r = encounter_relative(b,phase,u,p,field=FIELD,strength=STRENGTH,screening=SCREENING,
                start_distance=24.,mass=MASS,charge=CHARGE,rtol=TIGHT_RELATIVE_RTOL,
                max_step=TIGHT_RELATIVE_STEP,flight_time_factor=FLIGHT_FACTOR)
            incoming = np.r_[r['initial_relative_position'],r['initial_relative_velocity']]
            if not np.array_equal(incoming,required_initial):
                raise ValueError('Tighter reflected solve changed exact initial bits')
            record.update(status='passed', initial_state=incoming.tolist(),
                final_position=r['final_relative_position'].tolist(), final_velocity=r['final_relative_velocity'].tolist(),
                exit=r['exit'], time=r['time'], energy_error=r['energy_error'],
                position_error=float(max(abs(r['final_relative_position']-required_fine['final_relative_position']))),
                velocity_error=float(max(abs(r['final_relative_velocity']-required_fine['final_relative_velocity']))),
                time_error=abs(r['time']-required_fine['time']))
        except Exception as error:
            record.update(status='unresolved', reason=str(error))
    else:
        record.update(status='not_run', reason='missing fine required state or fixed cap')
    tight_relative.append(record)
else:
    checks['active_step'] = None
pool.shutdown()
ending = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
metadata['experiment_dependency_sha256_end'] = ending


def gate(values, target):
    return values is not None and np.all(np.isfinite(values)) and max(values)<=target


gates = {k:gate(checks[k], PHASE_TARGET) for k in ('phase1024_to2048', 'phase2048_to4096', 'halfshift', 'final_plane_phase', 'final_plane_halfshift')}
gates.update(boundary_integral=gate(checks['boundary_integral'], PLANE_TARGET),
    boundary_weak_state=gate(checks['boundary_weak_state'], PLANE_TARGET),
    active_step=gate(checks['active_step'], STEP_TARGET),
    active_step_sizes=bool(controls.get('actual_step_ratios')) and all(x is not None and x<=.6 for x in controls.get('actual_step_ratios', [])),
    cartesian=bool(cartesian) and all(r['status']=='passed' and max(r['position_error'], r['velocity_error'])<=CARTESIAN_TARGET and r['energy_error']<=ENERGY_TARGET for r in cartesian),
    required_reflected_tight=bool(tight_relative) and all(r['status']=='passed' and max(r['position_error'],r['velocity_error'],r['time_error'])<=CARTESIAN_TARGET and r['energy_error']<=ENERGY_TARGET for r in tight_relative),
    complete=all(c['complete'] for c in cases) and all(w['complete'] for w in weak) and not failures and not cap_exhausted,
    energy=all(r['status']=='passed' and r['energy_error']<=ENERGY_TARGET for r in attempts),
    source_integrity=hashes==ending,
    prior_integrity=all(hashlib.sha256((PRIOR/name).read_bytes()).hexdigest()==wanted for name,wanted in INPUTS['prior_artifact_sha256'].items()))
gates['original_failure_integrity'] = hashlib.sha256(ORIGINAL_AUDIT.read_bytes()).hexdigest()==INPUTS['original_cartesian_audit']['sha256']
status = 'passed' if all(gates.values()) else 'unresolved'
save_states(OUTPUT/'states.npz', attempts)
np.savez_compressed(OUTPUT/'populations.npz', **populations)
metadata['results'] = dict(status=status, required_gates=gates, comparisons=checks, cases=cases,
    weak_boundary=weak, controls=controls, cartesian=cartesian, tight_relative=tight_relative,
    separate_cartesian_solves=len(cartesian), separate_tight_relative_solves=len(tight_relative),
    reused_encounters=reused_count,
    new_encounters=len(attempts)-reused_count, unique_encounters=len(attempts), failures=len(failures),
    cap_exhausted=cap_exhausted, wall_s=perf_counter()-started, parent_process_peak_rss_bytes=rss(),
    max_worker_process_peak_rss_bytes=max((r['worker_peak_rss_bytes'] for r in attempts[reused_count:]), default=0),
    memory_scope='separate process lifetime peaks; not simultaneous aggregate', timing_scope='this execution; old solves and figure rendering excluded')
(OUTPUT/'summary.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
if fine['complete']:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [attempts[i] for i in populations['phase4096_L24_union']]
    m2 = np.asarray([r['moments'][1] for r in rows])
    exit_mask = np.asarray([r['exit']=='reflected' for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    axes[0, 0].plot(angles, m2, lw=.8)
    axes[0, 0].scatter(angles[exit_mask], m2[exit_mask], s=14, color='orange', label='reflected; retained weight')
    axes[0, 0].set(xlabel='incoming zero-plane phase', ylabel='individual raw second moment', title='One fixed incident speed/impact; narrow scattering peak')
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].plot(angles, m2, lw=.8)
    axes[0, 1].scatter(angles[exit_mask], m2[exit_mask], s=14, color='orange')
    axes[0, 1].set(xlim=(1.45, 1.7), xlabel='incoming zero-plane phase', ylabel='individual raw second moment', title='Resolved transmission/reflection neighborhood')
    for j, name in enumerate(('mean', 'second', 'cross', 'spatial')):
        labels = ['phase1024_to2048', 'phase2048_to4096', 'halfshift', 'final_plane_phase', 'final_plane_halfshift', 'boundary_integral', 'boundary_weak_state', 'active_step']
        targets = [PHASE_TARGET]*5+[PLANE_TARGET]*2+[STEP_TARGET]
        axes[1, 0].plot(range(len(labels)), [checks[k][j]/target if checks[k] is not None else np.nan for k,target in zip(labels,targets)], 'o-', label=name)
    axes[1, 0].axhline(1, color='gray', ls=':', label='declared target')
    axes[1, 0].set_xticks(range(len(labels)), ['N1024', 'N2048', 'shift24', 'N48', 'shift48', 'planes', 'weak', 'step'], rotation=45)
    axes[1, 0].set(ylabel='change / own declared target', title=f'Fixed checks: {status}')
    axes[1, 0].legend(fontsize=7)
    for j, name in enumerate(('mean', 'second', 'cross')):
        axes[1, 1].plot([256, 512, 1024, 2048, 4096], [c['observables'][j] if c['complete'] else np.nan for c in [old,*phase_cases,fine]], 'o-', label=name)
    axes[1, 1].set(xscale='log', xlabel='uniform phases at L24', ylabel='angular moment average', title='Signed moments and second moment refined together')
    axes[1, 1].legend(fontsize=7)
    fig.savefig(OUTPUT/'peak_resolution.png', dpi=160)
    plt.close(fig)
with np.load(OUTPUT/'states.npz') as saved:
    if len(saved['nodes'])!=len(attempts):
        raise RuntimeError('Incomplete final archive; keep checkpoints')
for path in OUTPUT.glob('checkpoint_*.npz'):
    path.unlink()
print(f'Finished {status}: gates={gates}; wall={perf_counter()-started:.1f}s', flush=True)
sys.exit(0 if status=='passed' else 1)
