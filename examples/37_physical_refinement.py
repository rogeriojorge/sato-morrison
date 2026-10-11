"""Refine the two failed bounded-encounter checks and the large-field amplitude."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
import multiprocessing as mp
import hashlib
import json
import os
import resource
import sys
import numpy as np
from scipy.special import k0, k1
from sato_morrison.controls import (
    encounter_relative, encounter_thermal_moments, encounter_flux_quadrature,
    encounter_bounded_flux, encounter_moment_interpolator,
)
from sato_morrison.reference import run_metadata

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/physical_final_campaign'
OUTPUT = ROOT / os.environ.get('SM_PHYSICAL_REFINEMENT_OUTPUT', 'results/physical_refinement')
MASS, CHARGE, THETA, SCREENING, STRENGTH = 1., 1., .5, 3., .03
BOUNDS = ((1.2, 2.4), (.7, 1.5), (.2, .8))
DISTANCE, RTOL, MAX_STEP = 24., 1e-12, .2
B4_ORDERS = ((5, 5, 5, 16), (6, 6, 6, 16))
TABLE_ORDERS = (7, 7, 7, 32)
WEAK_STRENGTHS = (.003, .0015)
FRESH_COUNT, FRESH_SEED = 128, 3701
WORKERS, CHECKPOINT, RUNTIME_CAP = 4, 128, 1800.
MOMENT_TARGET, ENERGY_TARGET = .01, 1e-9
TABLE_RMS_TARGET, TABLE_P95_TARGET, ASYMPTOTIC_TARGET = .02, .05, .02

inputs = dict(mass=MASS, charge=CHARGE, theta=THETA, screening=SCREENING,
    strength=STRENGTH, bounds=BOUNDS, start_distance=DISTANCE, rtol=RTOL,
    max_step=MAX_STEP, B4_orders=B4_ORDERS, table_orders=TABLE_ORDERS,
    weak_B32_strengths=WEAK_STRENGTHS, weak_orders=(4, 4, 4, 16),
    fresh_count=FRESH_COUNT, fresh_seed=FRESH_SEED, workers=WORKERS,
    runtime_cap_s=RUNTIME_CAP, moment_target=MOMENT_TARGET,
    energy_target=ENERGY_TARGET, table_rms_target=TABLE_RMS_TARGET,
    table_p95_target=TABLE_P95_TARGET, weak_asymptotic_target=ASYMPTOTIC_TARGET,
    scope='Refinement of the same finite screened band; no plasma coefficient or mixing time.',
    units='m,q,velocity,length normalized; theta=kBT/m; U=kappa exp(-r/lambda)/r',
    stop_rule='Fixed cases and heldout seed; failures preserve all weights and invalidate their checks.')


def peak_rss_bytes():
    scale = 1 if sys.platform == 'darwin' else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*scale


def solve(spec):
    node, field, strength = spec
    started = perf_counter()
    row = dict(node=list(map(float, node)), field=field, strength=strength)
    try:
        b, u, p, phase = node
        result = encounter_relative(b, phase, u, p, field=field,
            strength=strength, screening=SCREENING, start_distance=DISTANCE,
            mass=MASS, charge=CHARGE, rtol=RTOL, max_step=MAX_STEP)
        omega = CHARGE*field/MASS
        def center(position, velocity):
            return position[:2]+np.array([velocity[1], -velocity[0]])/omega
        kick = center(result['final_relative_position'], result['final_relative_velocity'])-center(result['initial_relative_position'], result['initial_relative_velocity'])
        row.update(status='passed', moments=encounter_thermal_moments(result, THETA, MASS, field).tolist(),
            individual_gc_second=float(kick@kick/4), gc_relative_kick=kick.tolist(),
            **{key: result[key].tolist() for key in ('initial_relative_position', 'final_relative_position', 'initial_relative_velocity', 'final_relative_velocity')},
            **{key: result[key] for key in ('time', 'exit', 'energy_error', 'min_separation', 'max_actual_step')})
        if not np.all(np.isfinite(row['moments'])) or not np.isfinite(kick@kick):
            raise RuntimeError('Nonfinite observable')
    except Exception as error:
        row.update(status='unresolved', reason=str(error), error_type=type(error).__name__)
    row.update(wall_s=perf_counter()-started, worker_peak_rss_bytes=peak_rss_bytes())
    return row


if OUTPUT.exists():
    raise FileExistsError(f'Preserve existing evidence: {OUTPUT}; select a fresh output')
OUTPUT.mkdir(parents=True)
(OUTPUT/'declared_inputs.json').write_text(json.dumps(inputs, indent=2)+'\n')
from jax._src import xla_bridge
if xla_bridge.backends_are_initialized():
    raise RuntimeError('Run in a fresh Python process before a JAX backend initializes')
pool = ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context('fork'))
list(pool.map(float, range(WORKERS)))
metadata = run_metadata(inputs, model='focused bounded screened encounter refinement',
    boundary='first outgoing relative z=+/-24 planes', units=inputs['units'])
dependencies = [Path(__file__).resolve(), ROOT/'src/sato_morrison/controls.py', ROOT/'src/sato_morrison/reference.py']
hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in dependencies}
metadata['experiment_dependency_sha256'] = hashes
metadata['source_campaign_commit'] = json.loads((SOURCE/'summary.json').read_text())['commit']
metadata['source_artifact_sha256'] = {name: hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in ('summary.json', 'cases.json', 'heldout.npz')}
started = perf_counter()
attempts, cases, cache = [], [], {}


def evaluate(nodes, field, strength=STRENGTH):
    specs = [(tuple(map(float, node)), float(field), float(strength)) for node in nodes]
    pending = list(dict.fromkeys(spec for spec in specs if spec not in cache))
    for offset in range(0, len(pending), CHECKPOINT):
        if perf_counter()-started > RUNTIME_CAP:
            break
        batch = pending[offset:offset+CHECKPOINT]
        for spec, row in zip(batch, pool.map(solve, batch)):
            cache[spec] = row
            attempts.append(row)
            if row['status'] != 'passed':
                print('FAILED:', row, flush=True)
        (OUTPUT/'attempts_partial.json').write_text(json.dumps(attempts, allow_nan=False)+'\n')
        print(f'Saved {len(attempts)} encounters; elapsed {perf_counter()-started:.1f} s', flush=True)
    return [cache.get(spec, dict(status='not_run', node=list(spec[0]), reason='declared cap')) for spec in specs]


def integrate(label, orders, field, strength=STRENGTH):
    print(f'{label}: orders={orders}, B={field}, kappa={strength}', flush=True)
    nodes, weights = encounter_flux_quadrature(orders, *BOUNDS, THETA)
    rows = evaluate(nodes, field, strength)
    case = dict(case=label, orders=orders, field=field, strength=strength,
        complete=all(row['status']=='passed' for row in rows), nodes=len(nodes))
    if case['complete']:
        mu = MASS*THETA/field
        integral = weights@np.array([row['moments'] for row in rows])
        spatial = weights@np.array([row['individual_gc_second'] for row in rows])
        case.update(integrals=integral.tolist(), spatial_second_integral=float(spatial),
            normalized_mu_rate_per_density=float(integral[1]/mu**2),
            normalized_spatial_rate_per_density=float(spatial/SCREENING**2),
            flux_per_density=float(weights.sum()),
            flux_relative_error=float(abs(weights.sum()/encounter_bounded_flux(*BOUNDS, THETA)-1)),
            max_energy_error=max(row['energy_error'] for row in rows))
    cases.append(case)
    (OUTPUT/'cases_partial.json').write_text(json.dumps(cases, indent=2)+'\n')
    return case, nodes, rows, weights


def changes(a, b):
    if not a['complete'] or not b['complete']:
        return None
    ia = np.array(a['integrals']+[a['spatial_second_integral']])
    ib = np.array(b['integrals']+[b['spatial_second_integral']])
    scale = np.array([ia[1]/(MASS*THETA/a['field']), ia[1], ia[1], ia[3]])
    if np.any(scale<=0) or not np.all(np.isfinite(scale)):
        return None
    return (abs(ia-ib)/scale).tolist()


b4 = [integrate(f'B4_order{orders[0]}', orders, 4.)[0] for orders in B4_ORDERS]
table_case, table_nodes, table_rows, _ = integrate('B2_table7', TABLE_ORDERS, 2.)
axes = [np.unique(table_nodes[:, index]) for index in range(4)]
old = np.load(SOURCE/'heldout.npz')['nodes']
rng = np.random.default_rng(FRESH_SEED)
fresh = np.column_stack([rng.uniform(axis[0], axis[-1], FRESH_COUNT) for axis in axes[:3]]+[rng.uniform(0, 2*np.pi, FRESH_COUNT)])
table_checks = []
for label, nodes in (('original128', old), ('fresh128', fresh)):
    rows = evaluate(nodes, 2.)
    check = dict(sample=label, status='unresolved')
    if table_case['complete'] and all(row['status']=='passed' for row in rows):
        interpolator = encounter_moment_interpolator(axes, np.array([row['moments'] for row in table_rows]).reshape(*TABLE_ORDERS, 3))
        direct = np.array([row['moments'] for row in rows])
        predicted = interpolator(nodes)
        rms = np.sqrt(np.mean((predicted-direct)**2, axis=0))/np.sqrt(np.mean(direct**2, axis=0))
        p95 = float(np.quantile(abs(predicted[:, 1]/direct[:, 1]-1), .95))
        check.update(rms=rms.tolist(), second_p95=p95,
            status='passed' if rms[1]<=TABLE_RMS_TARGET and p95<=TABLE_P95_TARGET else 'unresolved')
        np.savez_compressed(OUTPUT/f'heldout_{label}.npz', nodes=nodes, direct=direct, predicted=predicted)
    table_checks.append(check)
weak = []
for strength in WEAK_STRENGTHS:
    case, nodes, rows, weights = integrate(f'B32_weak{strength:g}', (4, 4, 4, 16), 32., strength)
    if case['complete']:
        b, u, p, _ = nodes.T
        omega = CHARGE*32/MASS
        alpha = 2*strength*k0(b/SCREENING)/(MASS*omega*u*SCREENING**2)
        expected_mu = weights@(alpha**2*p**2/(8*THETA))
        expected_spatial = weights@(4*strength**2*k1(b/SCREENING)**2/(MASS**2*omega**2*u**2*SCREENING**4))
        case.update(asymptotic_mu_rate=float(expected_mu), asymptotic_spatial_rate=float(expected_spatial),
            asymptotic_relative_errors=[float(abs(case['normalized_mu_rate_per_density']/expected_mu-1)), float(abs(case['normalized_spatial_rate_per_density']/expected_spatial-1))])
    weak.append(case)
pool.shutdown()
refinement = changes(b4[-1], b4[-2])
end_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in dependencies}
gates = dict(B4_coordinate_refinement=refinement is not None and max(refinement)<=MOMENT_TARGET,
    table_both_heldouts=all(check['status']=='passed' for check in table_checks),
    energy=all(case.get('max_energy_error', np.inf)<=ENERGY_TARGET for case in cases),
    flux=all(case.get('flux_relative_error', np.inf)<=1e-5 for case in cases),
    weak_B32_asymptotic=all(case['complete'] and max(case['asymptotic_relative_errors'])<=ASYMPTOTIC_TARGET for case in weak),
    source_integrity=hashes==end_hashes)
results = dict(overall_status='passed' if all(gates.values()) else 'unresolved', required_gates=gates,
    B4_last_refinement_changes=refinement, table_checks=table_checks, weak_B32=weak,
    attempted_unique_encounters=len(attempts), cases=cases)
metadata.update(results=results, wall_s=perf_counter()-started,
    parent_process_peak_rss_bytes=peak_rss_bytes(),
    worker_process_peak_rss_bytes_max=max((row['worker_peak_rss_bytes'] for row in attempts), default=0),
    memory_scope='Separate process lifetime peaks; not a simultaneous aggregate job peak.',
    experiment_dependency_sha256_end=end_hashes)
(OUTPUT/'attempts.json').write_text(json.dumps(attempts, allow_nan=False)+'\n')
(OUTPUT/'cases.json').write_text(json.dumps(cases, indent=2, allow_nan=False)+'\n')
metadata['output_sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in OUTPUT.iterdir() if path.is_file()}
(OUTPUT/'summary.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
print(json.dumps(results, indent=2), flush=True)
if results['overall_status']!='passed':
    raise RuntimeError(f'Declared physical refinement unresolved; evidence retained in {OUTPUT}')
