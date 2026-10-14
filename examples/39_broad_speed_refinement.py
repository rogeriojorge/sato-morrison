"""Refine one previously unresolved broad-speed screened encounter annulus.

The experiment preserves the old campaign and retains failed incident weights.
It reports a conditional plane-crossing integral, never a complete plasma rate.
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
from scipy.stats import qmc
from sato_morrison.controls import (
    encounter_relative, encounter_thermal_moments, encounter_flux_samples,
    encounter_bounded_flux,
)
from sato_morrison.reference import run_metadata

ROOT = Path(__file__).resolve().parents[1]
MODE = os.environ.get('SM_BROAD_SPEED_MODE', 'base')
if MODE not in ('base', 'extend'):
    raise ValueError('SM_BROAD_SPEED_MODE must be base or extend')
PRIOR = ROOT/'results/broad_speed_refinement'
OUTPUT = ROOT / os.environ.get('SM_BROAD_SPEED_OUTPUT',
    'results/broad_speed_extended' if MODE == 'extend' else 'results/broad_speed_refinement')
MASS, CHARGE, FIELD, STRENGTH, SCREENING, THETA = 1., 1., 2., .03, 3., .5
BOUNDS = ((1.2, 2.4), (.35, 3.), (0., 2.5))
SAMPLE_LEVELS = (512, 1024, 2048) if MODE == 'extend' else (256, 512)
PHASE_LEVELS = (32, 64)
INFLUENTIAL_PHASE_LEVELS, INFLUENTIAL_PER_SEED = (128, 256), 8
SEEDS = (3901, 3902)
DISTANCES, STEP_LEVELS = (24., 32.), (.1, .05, .025)
RTOL, FLIGHT_FACTOR = 1e-12, 48.
WORKERS, CHECKPOINT = 4, 1024
RUNTIME_CAP = 2400. if MODE == 'extend' else 1800.
MOMENT_TARGET, ENERGY_TARGET = .05, 1e-9
BOUNDARY_TARGET, STEP_TARGET = .005, .001
CONTROL_COUNT = 16

INPUTS = dict(mode=MODE, mass=MASS, charge=CHARGE, field=FIELD, strength=STRENGTH,
    screening=SCREENING, theta=THETA, bounds=BOUNDS, sample_levels=SAMPLE_LEVELS,
    phase_levels=PHASE_LEVELS, seeds=SEEDS, distances=DISTANCES,
    step_levels=STEP_LEVELS, rtol=RTOL, flight_time_factor=FLIGHT_FACTOR,
    workers=WORKERS, runtime_cap_s=RUNTIME_CAP,
    expected_new_bulk_paths=(2*(SAMPLE_LEVELS[-1]-512)*PHASE_LEVELS[-1]
        if MODE == 'extend' else 3*SAMPLE_LEVELS[-1]*PHASE_LEVELS[-1]),
    influential_phase_levels=INFLUENTIAL_PHASE_LEVELS,
    influential_per_seed=INFLUENTIAL_PER_SEED, moment_target=MOMENT_TARGET,
    energy_target=ENERGY_TARGET, boundary_target=BOUNDARY_TARGET,
    step_target=STEP_TARGET, control_count_each=CONTROL_COUNT,
    units='normalized; theta=kBT/m; U=kappa exp(-r/lambda)/r',
    measure='two-sided tagged plane crossing flux; b^2 uniform; squared speeds truncated exponential; periodic phase',
    error_scale='mean: fine second/mu_reference; second and cross: fine second; GC kick: fine spatial second',
    controls=('retained full 512-state plane comparison; 16 largest new primary individual second moments and 16 predetermined flux samples at both planes and active steps; top eight phase-averaged contributors per seed at 128/256 phases'
        if MODE == 'extend' else 'full primary ensemble at both planes; 16 largest second-moment contributors and 16 fresh predetermined flux samples at all active steps'),
    acceptance=('final 1024 to 2048 sample comparison for each seed plus independent final replica and phase checks; 512 to 1024 history diagnostic only; influential 64 to 128 and 128 to 256 per-speed comparisons required'
        if MODE == 'extend' else '256 to 512 sample, replica and phase comparisons'),
    stop_rule='fixed domains, levels, seeds and flight budget; cap checked between batches; no missing weights renormalized; failed gates exit nonzero',
    scope='one bounded broad-speed annulus from example13; inner impacts, slow/fast and perpendicular tails, inverse-complete incident surface, screening closure, correlations and mixing time unresolved')


def peak_rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)


def solve(spec):
    node, distance, step = spec
    started = perf_counter()
    row = dict(node=list(node), distance=distance, step=step)
    try:
        b, u, p, phase = node
        r = encounter_relative(b, phase, u, p, field=FIELD, strength=STRENGTH,
            screening=SCREENING, start_distance=distance, mass=MASS,
            charge=CHARGE, rtol=RTOL, max_step=step,
            flight_time_factor=FLIGHT_FACTOR)
        def gc(position, velocity):
            return position[:2]+np.array([velocity[1], -velocity[0]])/(CHARGE*FIELD/MASS)
        kick = gc(r['final_relative_position'], r['final_relative_velocity'])-gc(r['initial_relative_position'], r['initial_relative_velocity'])
        row.update(status='passed', moments=encounter_thermal_moments(r, THETA, MASS, FIELD).tolist(),
            gc_relative_kick=kick.tolist(), individual_gc_second=float(kick@kick/4),
            **{key: r[key].tolist() for key in ('initial_relative_position', 'final_relative_position', 'initial_relative_velocity', 'final_relative_velocity', 'delta_relative_perpendicular')},
            **{key: r[key] for key in ('time', 'exit', 'energy_error', 'min_separation', 'max_actual_step', 'evaluations', 'parallel_turns')})
        if not np.all(np.isfinite(row['moments'])) or not np.isfinite(kick@kick):
            raise RuntimeError('Nonfinite scattering observable')
    except Exception as error:
        row.update(status='unresolved', reason=str(error), error_type=type(error).__name__,
            diagnostics=getattr(error, 'integration_diagnostics', {}))
    row.update(wall_s=perf_counter()-started, worker_peak_rss_bytes=peak_rss())
    return row


def save_states(path, rows):
    """Retain every state needed to reconstruct moments independently."""
    arrays = {'nodes': np.asarray([r['node'] for r in rows]),
        'distance': np.asarray([r['distance'] for r in rows]),
        'step': np.asarray([r['step'] for r in rows]),
        'status': np.asarray([r['status'] for r in rows])}
    for name, width in (('moments', 3), ('initial_relative_position', 3),
            ('final_relative_position', 3), ('initial_relative_velocity', 3),
            ('final_relative_velocity', 3), ('delta_relative_perpendicular', 2),
            ('gc_relative_kick', 2)):
        arrays[name] = np.asarray([r.get(name, [np.nan]*width) for r in rows])
    for name in ('time', 'energy_error', 'min_separation', 'max_actual_step',
            'evaluations', 'parallel_turns', 'individual_gc_second', 'wall_s', 'worker_peak_rss_bytes'):
        arrays[name] = np.asarray([r.get(name, np.nan) for r in rows])
    arrays['exit'] = np.asarray([r.get('exit', 'missing') for r in rows])
    np.savez_compressed(path, **arrays)


prior_states = None
prior_summary = None
if MODE == 'extend':
    INPUTS['prior_artifact_sha256'] = {name: hashlib.sha256((PRIOR/name).read_bytes()).hexdigest()
        for name in ('summary.json', 'states.npz', 'populations.npz')}
    prior_summary = json.loads((PRIOR/'summary.json').read_text())
    for name in ('mass', 'charge', 'field', 'strength', 'screening', 'theta',
            'bounds', 'distances', 'step_levels', 'rtol', 'flight_time_factor'):
        if prior_summary['inputs'][name] != json.loads(json.dumps(INPUTS[name])):
            raise ValueError(f'Prior physical or numerical input differs: {name}')
    for name in ('src/sato_morrison/controls.py', 'src/sato_morrison/reference.py'):
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != prior_summary['experiment_dependency_sha256'][name]:
            raise ValueError(f'Prior scientific helper differs: {name}')
    with np.load(PRIOR/'states.npz') as saved:
        prior_states = {name: saved[name].copy() for name in saved.files}
    if not np.all(prior_states['status'] == 'passed'):
        raise ValueError('Extension requires the retained complete prior states; do not silently fill failures')

if OUTPUT.exists():
    raise FileExistsError(f'Preserve existing evidence: {OUTPUT}; select a fresh output')
OUTPUT.mkdir(parents=True)
(OUTPUT/'declared_inputs.json').write_text(json.dumps(INPUTS, indent=2)+'\n')
from jax._src import xla_bridge
if xla_bridge.backends_are_initialized():
    raise RuntimeError('Run in a fresh process before the JAX backend initializes')
pool = ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context('fork'))
list(pool.map(float, range(WORKERS)))
import jax
jax.config.update('jax_enable_x64', True)
metadata = run_metadata(INPUTS, model='bounded broad-speed screened encounter refinement',
    boundary='first outgoing relative z=+/-L plane; reflected retained', units=INPUTS['units'])
dependencies = [Path(__file__).resolve(), ROOT/'src/sato_morrison/controls.py', ROOT/'src/sato_morrison/reference.py']
hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
metadata['experiment_dependency_sha256'] = hashes
prior = ROOT/'results/encounter_validation/metadata.json'
metadata['prior_evidence_sha256'] = hashlib.sha256(prior.read_bytes()).hexdigest()
started, cache, attempts, failures, cases = perf_counter(), {}, [], [], []
cap_exhausted = False
reused_count = 0
if prior_states is not None:
    reused_count = len(prior_states['nodes'])
    for index in range(reused_count):
        row = {name: values[index].tolist() for name, values in prior_states.items()}
        row['node'] = row.pop('nodes')
        row['status'] = 'passed'
        key = (tuple(row['node']), float(row['distance']), float(row['step']))
        if key in cache:
            raise ValueError('Prior state keys are not unique')
        cache[key] = index
        attempts.append(row)
    print(f'Extended mode: reusing {reused_count} bit-identical archived states; no old solve repeated', flush=True)

def evaluate(nodes, distance=24., step=.2):
    global cap_exhausted
    specs = [(tuple(map(float, node)), float(distance), float(step)) for node in nodes]
    pending = list(dict.fromkeys(spec for spec in specs if spec not in cache))
    for offset in range(0, len(pending), CHECKPOINT):
        if perf_counter()-started > RUNTIME_CAP:
            cap_exhausted = True
            break
        batch = pending[offset:offset+CHECKPOINT]
        new = list(pool.map(solve, batch))
        for spec, row in zip(batch, new):
            cache[spec] = len(attempts)
            attempts.append(row)
            if row['status'] != 'passed':
                failures.append(row)
                print('FAILED:', row, flush=True)
        save_states(OUTPUT/f'checkpoint_{len(attempts):06d}.npz', new)
        (OUTPUT/'failures.json').write_text(json.dumps(failures, indent=2, allow_nan=False)+'\n')
        print(f'Saved {len(attempts)-reused_count} new encounters ({len(attempts)} total); wall {perf_counter()-started:.1f} s', flush=True)
    ids = np.asarray([cache.get(spec, -1) for spec in specs], dtype=int)
    rows = [attempts[i] if i >= 0 else {'status': 'not_run'} for i in ids]
    return rows, ids

flux = encounter_bounded_flux(*BOUNDS, THETA)
mu_ref = MASS*THETA/FIELD
phases = 2*np.pi*np.arange(PHASE_LEVELS[-1])/PHASE_LEVELS[-1]
samples = [encounter_flux_samples(qmc.Sobol(3, scramble=True, seed=seed).random_base2(int(np.log2(SAMPLE_LEVELS[-1]))), *BOUNDS, THETA) for seed in SEEDS]
nodes = [np.column_stack((np.repeat(s, len(phases), axis=0), np.tile(phases, len(s)))) for s in samples]
populations = {}
if MODE == 'extend':
    with np.load(PRIOR/'populations.npz') as saved:
        for seed, label in zip(SEEDS, ('primary512_phase64', 'replica512_phase64')):
            index = SEEDS.index(seed)
            retained = prior_states['nodes'][saved[label]]
            if not np.array_equal(nodes[index][:len(retained)], retained):
                raise ValueError('Extended Sobol prefix differs from original incoming bits')

def integrate(label, index, count=SAMPLE_LEVELS[-1], phase_stride=1, distance=DISTANCES[0]):
    query = nodes[index].reshape(SAMPLE_LEVELS[-1], PHASE_LEVELS[-1], 4)[:count, ::phase_stride].reshape(-1, 4)
    records, ids = evaluate(query, distance)
    complete = all(r['status'] == 'passed' for r in records)
    row = dict(case=label, seed=SEEDS[index], sample_count=count,
        phase_count=PHASE_LEVELS[-1]//phase_stride, distance=distance, complete=complete,
        missing_count=sum(r['status'] != 'passed' for r in records), flux_per_density=flux)
    populations[label] = ids
    if complete:
        values = np.asarray([r['moments']+[r['individual_gc_second']] for r in records])
        integral = flux*values.mean(axis=0)
        row.update(integrals=integral.tolist(), drift_per_density=float(integral[0]),
            diffusion_per_density=float(integral[1]/2), cross_per_density=float(integral[2]),
            spatial_second_per_density=float(integral[3]),
            max_energy_error=max(r['energy_error'] for r in records),
            reflected_fraction=float(np.mean([r['exit']=='reflected' for r in records])))
    cases.append(row)
    (OUTPUT/'cases_partial.json').write_text(json.dumps(cases, indent=2)+'\n')
    print(f'{label}: complete={complete}; D/n={row.get("diffusion_per_density")}', flush=True)
    return row

fine_label = f'primary{SAMPLE_LEVELS[-1]}_phase64'
fine = integrate(fine_label, 0)
previous = integrate(f'primary{SAMPLE_LEVELS[-2]}_phase64', 0, SAMPLE_LEVELS[-2])
phase = integrate(f'primary{SAMPLE_LEVELS[-1]}_phase32', 0, phase_stride=PHASE_LEVELS[-1]//PHASE_LEVELS[0])
replica_label = f'replica{SAMPLE_LEVELS[-1]}_phase64'
replica = integrate(replica_label, 1)
if MODE == 'extend':
    replica_previous = integrate('replica1024_phase64', 1, SAMPLE_LEVELS[-2])
    primary_history = integrate('primary512_phase64', 0, SAMPLE_LEVELS[0])
    replica_history = integrate('replica512_phase64', 1, SAMPLE_LEVELS[0])
    boundary = integrate('primary512_phase64_L32', 0, SAMPLE_LEVELS[0], distance=DISTANCES[-1])
    boundary_reference = primary_history
else:
    boundary = integrate('primary512_phase64_L32', 0, distance=DISTANCES[-1])
    boundary_reference = fine

def changes(reference, other):
    if not reference['complete'] or not other['complete']:
        return None
    a, b = np.asarray(reference['integrals']), np.asarray(other['integrals'])
    scales = np.array([a[1]/mu_ref, a[1], a[1], a[3]])
    return (abs(a-b)/scales).tolist() if np.all(scales > 0) else None

checks = {'sample': changes(fine, previous), 'phase': changes(fine, phase),
    'replica': changes(fine, replica), 'boundary_ensemble': changes(boundary, boundary_reference)}
if MODE == 'extend':
    checks['sample_replica'] = changes(replica, replica_previous)
    checks['history_primary512_to1024'] = changes(previous, primary_history)
    checks['history_replica512_to1024'] = changes(replica_previous, replica_history)
controls = {}
if fine['complete']:
    ids = populations[fine_label]
    contributions = np.asarray([attempts[i]['moments'][1] for i in ids])
    candidates = np.arange(len(ids)) >= (512*len(phases) if MODE == 'extend' else 0)
    top_indices = np.flatnonzero(candidates)[np.argsort(contributions[candidates])[-CONTROL_COUNT:]]
    fresh = encounter_flux_samples(qmc.Sobol(3, scramble=True, seed=3903).random_base2(int(np.log2(CONTROL_COUNT))), *BOUNDS, THETA)
    fresh_phase = 2*np.pi*qmc.Sobol(1, scramble=True, seed=3904).random_base2(int(np.log2(CONTROL_COUNT))).ravel()
    control_nodes = np.vstack((nodes[0][top_indices], np.column_stack((fresh, fresh_phase))))
    controls['nodes'] = control_nodes.tolist()
    controls['selection'] = [('largest_new_primary_second' if MODE == 'extend' else 'largest_primary_second')]*CONTROL_COUNT+['fresh_predetermined_flux']*CONTROL_COUNT
    controls['primary_indices'] = top_indices.tolist()
    if MODE == 'extend':
        boundary_arrays = []
        for distance in DISTANCES:
            records, ids = evaluate(control_nodes, distance=distance)
            controls[f'plane_{distance:g}'] = ids.tolist()
            boundary_arrays.append(np.asarray([r['moments']+[r['individual_gc_second']] if r['status']=='passed' else [np.nan]*4 for r in records]))
        a, b = boundary_arrays[-1], boundary_arrays[-2]
        if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
            scales = np.column_stack((a[:, 1]/mu_ref, a[:, 1], a[:, 1], a[:, 3]))
            checks['boundary_new_states'] = np.max(abs(a-b)/scales, axis=0).tolist() if np.all(scales > 0) else None
        else:
            checks['boundary_new_states'] = None
    arrays = []
    for step in STEP_LEVELS:
        records, ids = evaluate(control_nodes, step=step)
        controls[str(step)] = ids.tolist()
        arrays.append(np.asarray([r['moments']+[r['individual_gc_second']] if r['status']=='passed' else [np.nan]*4 for r in records]))
    a, b = arrays[-1], arrays[-2]
    if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
        scales = np.column_stack((a[:, 1]/mu_ref, a[:, 1], a[:, 1], a[:, 3]))
        checks['active_step'] = np.max(abs(a-b)/scales, axis=0).tolist() if np.all(scales > 0) else None
        controls['actual_step_ratios'] = [attempts[i]['max_actual_step']/attempts[j]['max_actual_step'] if i>=0 and j>=0 and attempts[i]['status']=='passed' and attempts[j]['status']=='passed' else None for i, j in zip(controls[str(STEP_LEVELS[-1])], controls[str(STEP_LEVELS[-2])])]
    else:
        checks['active_step'] = None
else:
    checks['active_step'] = None
phase_controls = []
if MODE == 'extend':
    phase_changes = []
    for index, label in enumerate((fine_label, replica_label)):
        ids = populations[label]
        if np.any(ids < 0) or not all(attempts[i]['status']=='passed' for i in ids):
            checks['phase_influential'] = None
            break
        second = np.asarray([attempts[i]['moments'][1] for i in ids]).reshape(SAMPLE_LEVELS[-1], len(phases)).mean(axis=1)
        selected = np.argsort(second)[-INFLUENTIAL_PER_SEED:]
        for sample_index in selected:
            report = dict(seed=SEEDS[index], sample_index=int(sample_index),
                point=samples[index][sample_index].tolist(), phase_average_second=float(second[sample_index]))
            selected_ids = ids.reshape(SAMPLE_LEVELS[-1], len(phases))[sample_index]
            report['phase64_ids'] = selected_ids.tolist()
            original_vector = np.asarray([attempts[i]['moments']+[attempts[i]['individual_gc_second']] for i in selected_ids]).mean(axis=0)
            report['phase64_observables'] = original_vector.tolist()
            vectors = []
            for count in INFLUENTIAL_PHASE_LEVELS:
                angles = 2*np.pi*np.arange(count)/count
                query = np.column_stack((np.repeat(samples[index][sample_index][None], count, axis=0), angles))
                records, phase_ids = evaluate(query)
                report[f'phase{count}_ids'] = phase_ids.tolist()
                complete = all(r['status']=='passed' for r in records)
                vector = np.asarray([r['moments']+[r['individual_gc_second']] for r in records]).mean(axis=0) if complete else None
                report[f'phase{count}_observables'] = vector.tolist() if vector is not None else None
                vectors.append(vector)
            a, b = vectors[-1], vectors[-2]
            if a is not None and b is not None:
                scales = np.array([a[1]/mu_ref, a[1], a[1], a[3]])
                change = (abs(a-b)/scales).tolist() if np.all(scales > 0) else None
                early_change = (abs(b-original_vector)/scales).tolist() if np.all(scales > 0) else None
            else:
                change = None
                early_change = None
            report['changes'] = change
            report['changes64_to128'] = early_change
            phase_changes.extend((early_change, change))
            phase_controls.append(report)
        checks['phase_influential'] = np.max(phase_changes, axis=0).tolist() if all(c is not None for c in phase_changes) else None
pool.shutdown()
ending = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
metadata['experiment_dependency_sha256_end'] = ending
def gate(values, target):
    return values is not None and np.all(np.isfinite(values)) and max(values) <= target
gates = {key: gate(checks[key], MOMENT_TARGET) for key in ('sample', 'phase', 'replica')}
gates.update(boundary_ensemble=gate(checks['boundary_ensemble'], BOUNDARY_TARGET),
    active_step=gate(checks['active_step'], STEP_TARGET),
    active_step_sizes=bool(controls.get('actual_step_ratios')) and all(r is not None and r <= .6 for r in controls.get('actual_step_ratios', [])),
    complete=all(row['complete'] for row in cases) and not failures and not cap_exhausted,
    energy=bool(attempts) and all(r['status']=='passed' and r['energy_error']<=ENERGY_TARGET for r in attempts),
    source_integrity=hashes==ending)
if MODE == 'extend':
    gates.update(sample_replica=gate(checks['sample_replica'], MOMENT_TARGET),
        phase_influential=gate(checks.get('phase_influential'), MOMENT_TARGET),
        boundary_new_states=gate(checks.get('boundary_new_states'), BOUNDARY_TARGET),
        prior_integrity=all(hashlib.sha256((PRIOR/name).read_bytes()).hexdigest()==wanted
            for name, wanted in INPUTS['prior_artifact_sha256'].items()))
status = 'passed' if all(gates.values()) else 'unresolved'
save_states(OUTPUT/'states.npz', attempts)
np.savez_compressed(OUTPUT/'populations.npz', **populations)
metadata['results'] = dict(status=status, required_gates=gates, comparisons=checks,
    cases=cases, controls=controls, phase_controls=phase_controls,
    unique_encounters=len(attempts), reused_encounters=reused_count,
    new_encounters=len(attempts)-reused_count, failures=len(failures),
    cap_exhausted=cap_exhausted, wall_s=perf_counter()-started,
    parent_process_peak_rss_bytes=peak_rss(),
    max_worker_process_peak_rss_bytes=max((r['worker_peak_rss_bytes'] for r in attempts[reused_count:]), default=0),
    memory_scope='separate lifetime process peaks; not a simultaneous aggregate',
    timing_scope='this execution only; previously saved solves are reused and their costs are not added; figure rendering excluded',
    velocity_flux_retained_fraction=flux/(2*np.pi*(BOUNDS[0][1]**2-BOUNDS[0][0]**2)*np.sqrt(THETA/np.pi)))
(OUTPUT/'summary.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
if fine['complete']:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    contribution = np.asarray([attempts[i]['moments'][1] for i in populations[fine_label]]).reshape(SAMPLE_LEVELS[-1], PHASE_LEVELS[-1])
    if MODE == 'extend':
        fig, grid = plt.subplots(2, 2, figsize=(14, 8), layout='constrained')
        axes = [grid[0, 1], grid[1, 0], grid[1, 1]]
        impact, rho_relative = samples[0][:, 0], samples[0][:, 2]/(CHARGE*FIELD/MASS)
        sc = grid[0, 0].scatter(impact, rho_relative, c=contribution.mean(axis=1), s=10)
        fig.colorbar(sc, ax=grid[0, 0], label='phase-averaged individual second moment')
        line = np.linspace(*BOUNDS[0], 100)
        grid[0, 0].plot(line, line, '--', color='gray', label='relative radius = impact')
        grid[0, 0].plot(line, .8*line, ':', color='gray', label='relative radius = 0.8 impact')
        grid[0, 0].set(xlim=BOUNDS[0], ylim=(0., BOUNDS[2][1]/(CHARGE*FIELD/MASS)*1.05),
            xlabel='relative guiding-center impact b', ylabel='relative gyroradius p / Omega',
            title='Close orbit approaches carry rare large kicks')
        grid[0, 0].legend(fontsize=7)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    sc = axes[0].scatter(samples[0][:, 1], samples[0][:, 2], c=contribution.mean(axis=1), s=14)
    fig.colorbar(sc, ax=axes[0], label='phase-averaged individual second moment')
    axes[0].set(xlabel='relative parallel speed', ylabel='relative perpendicular speed', title='Broad incident speed band; fixed annulus')
    order = np.sort(contribution.ravel())[::-1]
    axes[1].plot(np.arange(1, len(order)+1)/len(order), np.cumsum(order)/order.sum())
    axes[1].set(xlabel='fraction of incident nodes, largest first', ylabel='fraction of second-moment integral', title='Where the moment contribution lives')
    labels = (['sample', 'sample_replica', 'phase', 'replica', 'phase_influential', 'boundary_ensemble', 'boundary_new_states', 'active_step']
        if MODE == 'extend' else ['sample', 'phase', 'replica', 'boundary_ensemble', 'active_step'])
    for column, name in enumerate(('mean', 'second', 'cross', 'spatial')):
        targets = ([MOMENT_TARGET]*5+[BOUNDARY_TARGET]*2+[STEP_TARGET]
            if MODE == 'extend' else [MOMENT_TARGET]*3+[BOUNDARY_TARGET, STEP_TARGET])
        axes[2].plot(np.arange(len(labels)), [checks[k][column]/target if checks[k] is not None else np.nan for k, target in zip(labels, targets)], 'o-', label=name)
    axes[2].axhline(1., color='gray', ls=':', label='declared target')
    axes[2].set_xticks(range(len(labels)), labels=(['primary', 'replica N', 'phase', 'replica', 'top phase', 'old planes', 'new planes', 'step']
        if MODE == 'extend' else ['sample', 'phase', 'replica', 'planes', 'step']), rotation=50)
    axes[2].set(ylabel='change / own target (5%, 0.5%, 0.1%)', title=f'Finite comparisons: {status}')
    axes[2].legend(fontsize=7)
    fig.savefig(OUTPUT/'coverage.png', dpi=160)
    plt.close(fig)
# Final archive supersedes only checkpoints created by this run.
with np.load(OUTPUT/'states.npz') as saved:
    if len(saved['nodes']) != len(attempts):
        raise RuntimeError('Final archive incomplete; keep checkpoints')
for checkpoint in OUTPUT.glob('checkpoint_*.npz'):
    checkpoint.unlink()
print(f'Finished: {status}; gates={gates}; elapsed {perf_counter()-started:.1f}s', flush=True)
sys.exit(0 if status == 'passed' else 1)

