"""Compare signed kicks with logarithmic COM variance on the same saved table.

No new table nodes or fitted corrections. Fresh and weak-kick trajectories check
the representation change; the old heldout failures remain unchanged. All point
errors refer to the stated finite samples, not a supremum over the domain.
"""
from pathlib import Path
from time import perf_counter
import hashlib
import json
import os
import resource
import sys
import numpy as np
from scipy.stats import binom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sato_morrison.controls import (encounter_relative, encounter_thermal_moments,
    encounter_flux_quadrature, encounter_moment_interpolator, encounter_kick_interpolator)
from sato_morrison.reference import run_metadata

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results/physical_refinement'
OUTPUT = ROOT/os.environ.get('SM_INTERPOLATION_OUTPUT', 'results/scattering_interpolation')
MASS, CHARGE, FIELD, THETA = 1., 1., 2., .5
STRENGTH, SCREENING, DISTANCE = .03, 3., 24.
BOUNDS, ORDERS = ((1.2, 2.4), (.7, 1.5), (.2, .8)), (7, 7, 7, 32)
FRESH_COUNT, FRESH_SEED = 128, 3811
WEAK_ANCHORS = (('original128', 66), ('fresh128', 118))
WEAK_PHASE_COUNT, WEAK_PHASE_WIDTH = 49, .3
RTOL, MAX_STEP, TIGHT_RTOL, TIGHT_MAX_STEP = 1e-12, .2, 2e-13, .1
RMS_TARGET, P95_TARGET, SAMPLE_MAX_TARGET = .02, .05, .01
SOLVER_TARGET, ENERGY_TARGET, RUNTIME_CAP = .001, 1e-9, 180.


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def peak_rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)


def rotate(vectors, angles):
    """Standard Cartesian rotation R(angle), with arbitrary leading dimensions."""
    vectors, angles = np.asarray(vectors), np.asarray(angles)
    return np.stack((vectors[..., 0]*np.cos(angles)-vectors[..., 1]*np.sin(angles),
        vectors[..., 0]*np.sin(angles)+vectors[..., 1]*np.cos(angles)), axis=-1)


def saved_kicks(rows):
    nodes = np.array([row['node'] for row in rows])
    incoming = np.array([row['initial_relative_velocity'] for row in rows])
    final = np.array([row['final_relative_velocity'] for row in rows])
    times = np.array([row['time'] for row in rows])
    omega = CHARGE*FIELD/MASS
    # Free Lorentz motion is R(-Omega*t). Undo it, then move the incoming
    # -L preparation frame back to the zero-plane free-helix phase frame.
    delta = rotate(final[:, :2], omega*times)-incoming[:, :2]
    return rotate(delta, -omega*DISTANCE/nodes[:, 1])


def algebra(nodes, kicks):
    incoming = nodes[:, 2, None]*np.stack((np.cos(nodes[:, 3]), np.sin(nodes[:, 3])), axis=-1)
    square = np.sum(kicks*kicks, axis=-1)
    mean = MASS/(8*FIELD)*(2*np.sum(incoming*kicks, axis=-1)+square)
    variance = MASS**2*THETA/(8*FIELD**2)*square
    return np.stack((mean, mean**2+variance, mean**2-variance), axis=-1)


def metrics(predicted, direct, independent=False):
    if not np.all(np.isfinite(direct)) or np.any(direct[:, 1] <= 0):
        return dict(status='unresolved', reason='All planned direct states must have resolved positive second moments.')
    rms = np.sqrt(np.mean((predicted-direct)**2, axis=0)/np.mean(direct**2, axis=0))
    relative = abs(predicted[:, 1]/direct[:, 1]-1)
    check = dict(rms=rms.tolist(), second_p95=float(np.quantile(relative, .95)),
        second_max=float(relative.max()), worst_index=int(relative.argmax()),
        minimum_direct_second=float(direct[:, 1].min()),
        fraction_above_five_percent=float(np.mean(relative > P95_TARGET)),
        status='passed' if rms[1] <= RMS_TARGET and np.quantile(relative, .95) <= P95_TARGET else 'unresolved')
    if independent:
        lower, upper = binom.ppf([.025, .975], len(relative), .95).astype(int)
        ordered = np.sort(relative)
        check['p95_order_statistic_95_interval'] = [float(ordered[max(0, lower-1)]), float(ordered[min(len(relative)-1, upper)])]
        check['p95_interval_order_ranks'] = [max(1, int(lower)), min(len(relative), int(upper)+1)]
    return check


if OUTPUT.exists():
    raise FileExistsError(f'Preserve existing evidence: {OUTPUT}; choose a fresh SM_INTERPOLATION_OUTPUT')
OUTPUT.mkdir(parents=True)
started = perf_counter()
dependencies = [Path(__file__).resolve(), ROOT/'src/sato_morrison/controls.py', ROOT/'src/sato_morrison/reference.py']
source_paths = [SOURCE/name for name in ('summary.json', 'attempts.json', 'heldout_original128.npz', 'heldout_fresh128.npz')]
source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
dependency_hashes = {str(path.relative_to(ROOT)): digest(path) for path in dependencies}
archived = json.loads((SOURCE/'attempts.json').read_text())
cache = {tuple(row['node']): row for row in archived if row['field'] == FIELD and row['strength'] == STRENGTH}
table_nodes, _ = encounter_flux_quadrature(ORDERS, *BOUNDS, THETA)
table_rows = [cache[tuple(node)] for node in table_nodes]
axes = [np.unique(table_nodes[:, index]) for index in range(4)]
inputs = dict(mass=MASS, charge=CHARGE, field=FIELD, theta=THETA, strength=STRENGTH,
    screening=SCREENING, start_distance=DISTANCE, table_bounds=BOUNDS, table_orders=ORDERS,
    table_axes=[axis.tolist() for axis in axes], fresh_count=FRESH_COUNT, fresh_seed=FRESH_SEED,
    weak_anchors=WEAK_ANCHORS, weak_phase_count=WEAK_PHASE_COUNT, weak_phase_half_width=WEAK_PHASE_WIDTH,
    rtol=RTOL, max_step=MAX_STEP, tight_rtol=TIGHT_RTOL, tight_max_step=TIGHT_MAX_STEP,
    second_rms_target=RMS_TARGET, second_p95_target=P95_TARGET, sampled_max_target=SAMPLE_MAX_TARGET,
    solver_refinement_target=SOLVER_TARGET, energy_target=ENERGY_TARGET, runtime_cap_s=RUNTIME_CAP,
    frame='delta_w0=R(-Omega*L/u)[R(+Omega*t)w_out-w_in]; w_in0=p(cos(phi),sin(phi))',
    methods=['signed c and log(COM variance)', 'signed Cartesian zero-plane velocity kicks'],
    scope='Same finite B2 screened table; no new training solves, correction fit, physical rate, or supremum bound.',
    units='m,q,velocity,length normalized; theta=kBT/m; U=kappa exp(-r/lambda)/r',
    stop_rule='Fixed seed, anchors, neighborhoods and targets; retain failures and all planned sample counts.')
(OUTPUT/'declared_inputs.json').write_text(json.dumps(inputs, indent=2)+'\n')
metadata = run_metadata(inputs, model='conditional screened encounter interpolation representation',
    boundary='first outgoing relative z=+/-24 planes', units=inputs['units'])
metadata.update(experiment_dependency_sha256=dependency_hashes, source_artifact_sha256=source_hashes,
    source_campaign_commit=json.loads((SOURCE/'summary.json').read_text())['commit'], load_average_start=os.getloadavg())
table_moments = np.array([row['moments'] for row in table_rows])
table_kicks = saved_kicks(table_rows)
legacy = encounter_moment_interpolator(axes, table_moments.reshape(*ORDERS, 3))
signed = encounter_kick_interpolator(axes, table_kicks.reshape(*ORDERS, 2), THETA, MASS, FIELD)
table_payload = dict(nodes=table_nodes, moments=table_moments, kicks=table_kicks)
for key in ('initial_relative_position', 'final_relative_position', 'initial_relative_velocity', 'final_relative_velocity', 'time'):
    table_payload[key] = np.array([row[key] for row in table_rows])
np.savez_compressed(OUTPUT/'table.npz', **table_payload)
print(f'Loaded {len(table_rows)} archived table states; fresh={FRESH_COUNT}, weak={2*WEAK_PHASE_COUNT}; no new training solves.', flush=True)
samples, tests, attempts = {}, {}, []
legacy_reproduction = []
for label in ('original128', 'fresh128'):
    with np.load(SOURCE/f'heldout_{label}.npz') as archive:
        nodes, direct = archive['nodes'], archive['direct']
        reproduced = legacy(nodes)
        legacy_reproduction.append(float(np.linalg.norm(reproduced-archive['predicted'])/np.linalg.norm(archive['predicted'])))
    rows = [cache[tuple(node)] for node in nodes]
    samples[label] = (nodes, direct)
    tests[f'{label}_direct_kicks'] = saved_kicks(rows)


def evaluate(nodes, label, rtol=RTOL, max_step=MAX_STEP):
    rows = []
    for index, node in enumerate(nodes):
        row = dict(sample=label, index=index, node=node.tolist(), rtol=rtol, max_step=max_step)
        tick = perf_counter()
        try:
            if tick-started > RUNTIME_CAP:
                raise RuntimeError('Declared runtime cap reached; all remaining planned states are unresolved.')
            b, u, p, phase = node
            result = encounter_relative(b, phase, u, p, field=FIELD, strength=STRENGTH,
                screening=SCREENING, start_distance=DISTANCE, mass=MASS, charge=CHARGE, rtol=rtol, max_step=max_step)
            row.update(status='passed', moments=encounter_thermal_moments(result, THETA, MASS, FIELD).tolist(),
                **{key: np.asarray(result[key]).tolist() for key in ('initial_relative_position', 'final_relative_position', 'initial_relative_velocity', 'final_relative_velocity')},
                **{key: result[key] for key in ('time', 'exit', 'energy_error', 'max_actual_step', 'step_count', 'evaluations')})
            if not np.all(np.isfinite(row['moments'])) or row['moments'][1] <= 0:
                raise RuntimeError('Nonfinite or unresolved zero second moment; relative error cannot be reported.')
            row['kick'] = saved_kicks([row])[0].tolist()
        except Exception as error:
            row.update(status='unresolved', error_type=type(error).__name__, reason=str(error))
            print('FAILED:', row, flush=True)
        row['wall_s'] = perf_counter()-tick
        rows.append(row); attempts.append(row)
        if (index+1) % 32 == 0 or index+1 == len(nodes):
            (OUTPUT/'attempts_partial.json').write_text(json.dumps(attempts, allow_nan=False)+'\n')
            print(f'{label}: {index+1}/{len(nodes)} states; elapsed {perf_counter()-started:.1f} s', flush=True)
    return rows


rng = np.random.default_rng(FRESH_SEED)
new_nodes = np.column_stack([rng.uniform(axis[0], axis[-1], FRESH_COUNT) for axis in axes[:3]]+
    [rng.uniform(0, 2*np.pi, FRESH_COUNT)])
weak_nodes = []
for label, index in WEAK_ANCHORS:
    anchor = samples[label][0][index]
    neighborhood = np.tile(anchor, (WEAK_PHASE_COUNT, 1))
    neighborhood[:, 3] += np.linspace(-WEAK_PHASE_WIDTH, WEAK_PHASE_WIDTH, WEAK_PHASE_COUNT)
    weak_nodes.append(neighborhood)
weak_nodes = np.concatenate(weak_nodes)
new_rows = {}
for label, nodes in (('new128', new_nodes), ('weak98', weak_nodes)):
    rows = evaluate(nodes, label)
    new_rows[label] = rows
    direct = np.array([row['moments'] if row['status'] == 'passed' else [np.nan]*3 for row in rows])
    samples[label] = (nodes, direct)
    tests[f'{label}_direct_kicks'] = np.array([row['kick'] if row['status'] == 'passed' else [np.nan]*2 for row in rows])

comparisons = {}
for label, (nodes, direct) in samples.items():
    predicted_legacy, predicted_signed = legacy(nodes), signed(nodes)
    independent = label == 'new128'
    comparisons[label] = dict(legacy=metrics(predicted_legacy, direct, independent),
        signed=metrics(predicted_signed, direct, independent), states=len(nodes))
    tests.update({f'{label}_{name}': array for name, array in
        (('nodes', nodes), ('direct', direct), ('legacy', predicted_legacy), ('signed', predicted_signed))})
    print(f'{label}: {json.dumps(comparisons[label])}', flush=True)

# Refinements check selected worst signed errors and smallest actual increments.
# These are diagnostic trajectory solves, never new training or correction data.
selected = [cache[tuple(samples[label][0][index])] for label, index in WEAK_ANCHORS]
for label in ('new128', 'weak98'):
    rows = new_rows[label]; direct = samples[label][1]
    if all(row['status'] == 'passed' for row in rows):
        relative = abs(tests[f'{label}_signed'][:, 1]/direct[:, 1]-1)
        indices = set(np.argsort(relative)[-3:].tolist()+np.argsort(direct[:, 1])[:3].tolist())
        selected.extend(rows[index] for index in sorted(indices))
selected = list({tuple(row['node']): row for row in selected}.values())
tight_rows = evaluate(np.array([row['node'] for row in selected]), 'solver_refinement', TIGHT_RTOL, TIGHT_MAX_STEP)
refinements = []
for ordinary, tight in zip(selected, tight_rows):
    check = dict(node=ordinary['node'], status='unresolved')
    if tight['status'] == 'passed':
        old, new = np.array(ordinary['moments']), np.array(tight['moments'])
        error = float(abs(old[1]/new[1]-1))
        check.update(second_relative_change=error, mean_change_over_second_scale=float(abs(old[0]-new[0])/np.sqrt(new[1])),
            ordinary_second=float(old[1]), tight_second=float(new[1]),
            status='passed' if error <= SOLVER_TARGET else 'unresolved')
    refinements.append(check)
np.savez_compressed(OUTPUT/'tests.npz', **tests)
(OUTPUT/'attempts.json').write_text(json.dumps(attempts, allow_nan=False)+'\n')
table_reconstruction = float(np.max(abs(algebra(table_nodes, table_kicks)[:, 1]/table_moments[:, 1]-1)))
finite_new = all(row['status'] == 'passed' for row in attempts)
all_signed = all(row['signed']['status'] == 'passed' for row in comparisons.values())
fresh_check = comparisons['new128']['signed']
gates = dict(complete_new_states=finite_new, signed_original_targets=all_signed,
    fresh_p95_confidence=finite_new and fresh_check['p95_order_statistic_95_interval'][1] <= P95_TARGET,
    sampled_max=finite_new and all(row['signed']['second_max'] <= SAMPLE_MAX_TARGET for row in comparisons.values()),
    solver_refinement=all(row['status'] == 'passed' for row in refinements),
    energy=finite_new and max(row['energy_error'] for row in attempts) <= ENERGY_TARGET,
    table_algebra=table_reconstruction <= 1e-8, legacy_reproduction=max(legacy_reproduction) <= 1e-10,
    source_integrity=source_hashes == {str(path.relative_to(ROOT)): digest(path) for path in source_paths},
    dependency_integrity=dependency_hashes == {str(path.relative_to(ROOT)): digest(path) for path in dependencies})
results = dict(overall_status='passed' if all(gates.values()) else 'unresolved', required_gates=gates,
    comparisons=comparisons, solver_refinements=refinements, table_second_reconstruction_max_relative=table_reconstruction,
    legacy_reproduction_relative_norm=legacy_reproduction, new_encounters=len(attempts), table_states=len(table_rows),
    limitations=['Finite samples and two phase neighborhoods; no supremum interpolation bound.',
        'Reuses archived trajectory/table accuracy; tighter solves check selected states only.',
        'No independent-encounter plasma closure, kinetic rate, or lifetime conclusion.'])
if finite_new:
    fig, panels = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    for method, color in (('legacy', '#bd5c28'), ('signed', '#146f8a')):
        direct = tests['new128_direct'][:, 1]
        relative = abs(tests[f'new128_{method}'][:, 1]/direct-1)
        panels[0, 0].scatter(direct, relative, s=12, alpha=.7, label=method, color=color)
        panels[0, 1].plot(np.sort(relative), np.arange(1, len(relative)+1)/len(relative), label=method, color=color)
    panels[0, 0].set(xscale='log', yscale='log', xlabel='Direct individual second moment', ylabel='Absolute relative error', title='128 new independent states; seed 3811')
    panels[0, 1].set(xscale='log', xlabel='Absolute relative second-moment error', ylabel='Fraction of planned states', title='Fresh error distribution')
    panels[0, 1].axvline(P95_TARGET, color='gray', linestyle=':'); panels[0, 1].axhline(.95, color='gray', linestyle=':')
    for index, (label, anchor_index) in enumerate(WEAK_ANCHORS):
        block = slice(index*WEAK_PHASE_COUNT, (index+1)*WEAK_PHASE_COUNT)
        phase = tests['weak98_nodes'][block, 3]
        panel = panels[1, index]
        panel.plot(phase, tests['weak98_direct'][block, 1], color='black', label='direct')
        panel.plot(phase, tests['weak98_legacy'][block, 1], '--', color='#bd5c28', label='legacy')
        panel.plot(phase, tests['weak98_signed'][block, 1], ':', color='#146f8a', label='signed')
        panel.set(yscale='log', xlabel='Zero-plane gyrophase (radians)', ylabel='Individual second moment', title=f'Weak-kick neighborhood: {label} index {anchor_index}')
    for panel in panels.flat:
        panel.legend(fontsize=8); panel.grid(alpha=.2)
    fig.savefig(OUTPUT/'interpolation.png', dpi=170); plt.close(fig)
metadata.update(results=results, wall_s=perf_counter()-started, peak_rss_bytes=peak_rss_bytes(),
    timing_scope='One process, archived loading/table construction plus new direct solves and plotting; no JIT compilation.',
    memory_scope='Process lifetime peak; includes metadata/device initialization.', load_average_end=os.getloadavg(),
    experiment_dependency_sha256_end={str(path.relative_to(ROOT)): digest(path) for path in dependencies})
metadata['output_sha256'] = {path.name: digest(path) for path in OUTPUT.iterdir() if path.is_file()}
(OUTPUT/'summary.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
print(json.dumps(results, indent=2), flush=True)
if results['overall_status'] != 'passed':
    raise RuntimeError(f'Declared interpolation checks unresolved; evidence retained in {OUTPUT}')
