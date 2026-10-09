"""Independent saved-stage Landau audit, executed directly with NumPy/SciPy.

Set SM_LANDAU_AUDIT_INPUT to audit another immutable campaign and
SM_LANDAU_AUDIT_OUTPUT to a fresh receipt path. Relative paths resolve from the
repository root. Existing receipts are protected. This source imports no
production collision kernel. It rebuilds Hermite measure and barycentric D,
replays every saved raw Coulomb root and entropy/KL identity, and compares full
nodal density increments on shared times in boxes 4/5 with GL32/48/64,
whole-quadrature moment increments, and empirical temporal difference ratios. The
Gaussian strong oracle checks only the sampled initial state; later densities
have independent nodal degrees of freedom. Outside-box sampled population is
not a continuum tail bound, and discrete root checks are not convergence proof.
"""
from pathlib import Path
from hashlib import sha256
import json
import time
import sys
import platform
import os
import importlib.metadata as metadata
import traceback
import subprocess
import numpy as np
from scipy.special import roots_hermitenorm, roots_legendre
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPOSITORY_ROOT / 'results' / 'landau_trajectory'
input_path = Path(os.environ.get('SM_LANDAU_AUDIT_INPUT', str(DEFAULT_INPUT))).expanduser()
INPUT_DIRECTORY = (input_path if input_path.is_absolute() else REPOSITORY_ROOT / input_path).resolve()
output_path = Path(os.environ.get('SM_LANDAU_AUDIT_OUTPUT', str(INPUT_DIRECTORY / 'independent_portable_audit_reproduction.json'))).expanduser()
OUTPUT = (output_path if output_path.is_absolute() else REPOSITORY_ROOT / output_path).resolve()
if OUTPUT.exists():
    raise RuntimeError(f'Audit output already exists: {OUTPUT}. Preserve it and set SM_LANDAU_AUDIT_OUTPUT to a fresh path.')
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
A = np.array([1.15, 1.15, 0.7])

def audit_failure(kind, error, frames):
    failure = {'status': 'failed_independent_audit', 'error_type': kind.__name__,
               'error': str(error), 'traceback': ''.join(traceback.format_exception(kind, error, frames)),
               'audit_source_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
               'partial_allstage_rows': globals().get('allstage_rows', globals().get('rows', [])),
               'input_npz_sha256': globals().get('initial_input_hashes', {})}
    OUTPUT.write_text(json.dumps(failure, indent=2) + '\n')
    sys.__excepthook__(kind, error, frames)

sys.excepthook = audit_failure

def gaussian(v):
    return np.exp(-np.sum(v * v / A, axis=-1) / 2) / np.sqrt(np.prod(2 * np.pi * A))

def strong(v, order=256):
    z, sw = np.polynomial.legendre.leggauss(order)
    t = (z + 1) / 2
    s = t / (1 - t)
    sw = sw / (2 * (1 - t) ** 2)
    d = 1 + 2 * A * s[:, None] ** 2
    V = A / d
    result = np.empty(len(v))
    for j in range(0, len(v), 256):
        x = v[j:j + 256]
        score = -x / A
        m = x[:, None, :] / d
        Z = np.prod(d, axis=1) ** (-0.5) * np.exp(-s * s * np.sum(x[:, None, :] ** 2 / d, axis=-1))
        base = np.sum(score * score - 1 / A, axis=-1)[:, None]
        contract = np.sum(V * (score[:, None, :] ** 2 - 1 / A), axis=-1) + np.sum(m * score[:, None, :], axis=-1) ** 2 - np.sum(m * m / A, axis=-1)
        result[j:j + len(x)] = 2 / np.sqrt(np.pi) * np.sum(sw * Z * (base - 2 * s * s * contract), axis=1) + 8 * np.pi * gaussian(x)
    return result

def gradient(h, D, n):
    h = h.reshape((n,) * 3)
    return np.stack([np.moveaxis(np.tensordot(D, h, axes=(1, k)), 0, k).ravel() for k in range(3)], axis=1)

def adjoint(flux, D, n):
    return sum((np.moveaxis(np.tensordot(D.T, flux[:, k].reshape((n,) * 3), axes=(1, k)), 0, k).ravel() for k in range(3)))

def pair_flux(v, pop, score, chunk=64):
    out = np.empty_like(v)
    for first in range(0, len(v), chunk):
        vv = v[first:first + chunk]
        w = vv[:, None, :] - v[None, :, :]
        r2 = np.sum(w * w, axis=-1)
        safe = np.where(r2 > 0, r2, 1.0)
        inv = np.where(r2 > 0, 1 / np.sqrt(safe), 0.0)
        delta = score[first:first + chunk, None, :] - score[None, :, :]
        projected = (delta - w * np.sum(w * delta, axis=-1)[..., None] / safe[..., None]) * inv[..., None]
        out[first:first + len(vv)] = pop[first:first + len(vv), None] * np.sum(pop[None, :, None] * projected, axis=1)
    return out

def hermite(n, theta):
    x, q = roots_hermitenorm(n)
    v1 = x * np.sqrt(theta)
    w1 = q * np.sqrt(theta) * np.exp(x * x / 2)
    diff = v1[:, None] - v1[None, :]
    np.fill_diagonal(diff, 1.0)
    b = 1 / np.prod(diff, axis=1)
    D = b[None, :] / (b[:, None] * diff)
    np.fill_diagonal(D, 0.0)
    np.fill_diagonal(D, -D.sum(axis=1))
    v = np.stack(np.meshgrid(v1, v1, v1, indexing='ij'), axis=-1).reshape(-1, 3)
    w = np.prod(np.meshgrid(w1, w1, w1, indexing='ij'), axis=0).ravel()
    return (v, w, D, v1)
ROOT = INPUT_DIRECTORY
digest = lambda p: sha256(p.read_bytes()).hexdigest()
summary_bytes = (ROOT / 'summary.json').read_bytes()
summary_start_sha256 = sha256(summary_bytes).hexdigest()
meta = json.loads(summary_bytes)
assert meta['inputs']['gamma'] == 1, 'Independent pairs and strong oracle require Gamma=1.'
assert meta['status'] != 'running', 'Audit an immutable campaign, not live files.'
source_hash = digest(Path(__file__))
initial_input_hashes = {p.name: digest(p) for p in ROOT.glob('*.npz')}
rows = []
start = time.perf_counter()
source_root = Path(__file__).resolve().parents[1]
source_commit = subprocess.check_output(['git', '-C', str(source_root), 'rev-parse', 'HEAD'], text=True).strip()
source_dirty = bool(subprocess.check_output(['git', '-C', str(source_root), 'status', '--porcelain'], text=True).strip())
hardware = {'processor': platform.processor(), 'machine': platform.machine()}
if platform.system() == 'Darwin':
    hardware['model'] = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
for case in meta['rows']:
    path = ROOT / (case['name'] + '.npz')
    if not path.exists():
        continue
    with np.load(path) as a:
        v = a['velocity'].copy()
        w = a['weights'].copy()
        savedD = a['derivative'].copy()
        states = a['density'].copy()
        times = a['time'].copy()
    n = case['order']
    theta = case['variance']
    vv, ww, D, x = hermite(n, theta)
    assert np.max(abs(v - vv)) < 1e-11 and np.max(abs((w - ww) / ww)) < 1e-10
    q = roots_hermitenorm(n)[1]
    Dz = np.sqrt(theta) * D
    weighted_sbp = np.diag(q) @ Dz + Dz.T @ np.diag(q) - np.diag(q) @ np.diag(x / np.sqrt(theta))
    baseline = np.column_stack([np.ones(len(v)), v, np.sum(v * v, axis=1) / 2])
    invariants = np.einsum('ti,i,ij->tj', states, w, baseline)
    invariant_error = np.max(abs(invariants - invariants[0]), axis=0)
    declared_rtol = float(case.get('rtol', meta['inputs'].get('rtol', 1e-10)))
    declared_linear_rtol = float(case.get('linear_rtol', meta['inputs'].get('linear_rtol', 1e-8)))
    row = {'declared_nonlinear_rtol': declared_rtol, 'declared_linear_rtol': declared_linear_rtol, 'nonlinear_acceptance_metric': 'fixed initial raw population', 'case': case['name'], 'declared_status': case['status'], 'source_arrays_sha256': digest(path), 'n': n, 'theta': theta, 'softening': case.get('softening', 0.0), 'saved_stages': len(states), 'last_time': float(times[-1]), 'nodal_min_density': float(states.min()), 'quadrature_max_relative_difference': float(np.max(abs((w - ww) / ww))), 'derivative_max_scaled_difference': float(np.max(abs(savedD - D) / np.maximum(abs(D), 1.0))), 'weighted_SBP_max_abs_defect': float(abs(weighted_sbp).max()), 'raw_invariant_max_abs_drifts': invariant_error.tolist(), 'stages': []}
    assert np.isfinite(states).all() and (states > 0).all() and (np.max(invariant_error) < 1e-09)
    assert len(states) == len(times) and np.all(np.diff(times) > 0)
    assert np.max(abs(states[0] / gaussian(v) - 1)) < 1e-10, 'Initial strong reference assumes the declared sampled Gaussian.'
    oldg = np.log(states[0])
    initialaction = adjoint(pair_flux(v, w * states[0], gradient(oldg, D, n)), D, n)
    if not case.get('softening', 0.0):
        rate256 = strong(v, 256)
        row['strong_scalar192_256_max_abs_difference'] = float(np.max(abs(rate256 - strong(v, 192))))
        assert row['strong_scalar192_256_max_abs_difference'] < 3e-12
        reference = states[0] * rate256
        row['independent_initial_L1'] = float(w @ abs(-initialaction / w - reference) / (w @ abs(reference)))
    for j in range(1, len(states)):
        old, new = states[j - 1:j + 1]
        dt = float(times[j] - times[j - 1])
        g0 = np.log(old)
        g = np.log(new)
        p0 = w * old
        delta = g - g0
        grad_g = gradient(g, D, n)
        if case.get('softening', 0.0):
            eps = case['softening']
            flux = np.empty_like(v)
            for k in range(0, len(v), 64):
                r = v[k:k + 64, None, :] - v[None, :, :]
                r2 = np.sum(r * r, axis=-1)
                safe = np.where(r2 > 0, r2, 1.0)
                inv = np.where(r2 > 0, 1 / np.sqrt(safe + eps * eps), 0.0)
                dg = grad_g[k:k + 64, None, :] - grad_g[None, :, :]
                project = (dg - r * np.sum(r * dg, axis=-1)[..., None] / safe[..., None]) * inv[..., None]
                flux[k:k + 64] = p0[k:k + 64, None] * np.sum(p0[None, :, None] * project, axis=1)
        else:
            flux = pair_flux(v, p0, grad_g)
        action = adjoint(flux, D, n)
        residual = p0 * np.expm1(delta) + dt * action
        norm = float(np.linalg.norm(residual / np.sqrt(p0)) / np.sqrt(p0.sum()))
        production = float(dt * g @ action)
        KL = float(np.sum(np.asarray(p0, dtype=np.longdouble) * (np.expm1(np.asarray(delta, dtype=np.longdouble)) - delta)))
        change = float(-np.sum(np.asarray(w, dtype=np.longdouble) * (np.asarray(new, dtype=np.longdouble) * g - np.asarray(old, dtype=np.longdouble) * g0)))
        defect = float((g + 1) @ residual)
        identity = change - production - KL + defect
        stage = {'j': j, 'time': float(times[j]), 'dt': dt, 'weighted_root_norm': norm, 'initial_reference_root_norm': float(np.linalg.norm(residual / np.sqrt(w * states[0])) / np.sqrt(np.sum(w * states[0]))), 'entropy_change': change, 'frozen_Gram_production': production, 'time_KL': KL, 'residual_entropy_defect': defect, 'entropy_identity_defect_with_residual': float(identity), 'min_f': float(new.min()), 'tail_mass_r_gt3': float((w * new)[np.linalg.norm(v, axis=1) > 3].sum()), 'tail_mass_maxcoord_gt4': float((w * new)[np.max(abs(v), axis=1) > 4].sum()), 'tail_mass_maxcoord_gt5': float((w * new)[np.max(abs(v), axis=1) > 5].sum())}
        row['stages'].append(stage)
        print(case['name'], j, 'root', norm, 'entropy', change, 'identity', identity, flush=True)
        assert stage['initial_reference_root_norm'] < declared_rtol, 'Original fixed-population root exceeds the row nonlinear tolerance.'
        assert change > -1e-10 and production > -1e-10 and KL > -1e-10
        assert abs(identity) < 1e-10
    rows.append(row)
allstage_rows = rows
assert sum((len(r['stages']) for r in rows)) > 0
# Reconstruct all saved cases, including partial cases, only at shared accepted times.
data = {}
for case in meta['rows']:
    case_path = ROOT / (case['name'] + '.npz')
    if not case_path.exists():
        continue
    with np.load(case_path) as arrays:
        data[case['name']] = {key: arrays[key].copy() for key in ['velocity', 'weights', 'density', 'time']}
    data[case['name']]['row'] = case
assert 'baseline' in data, 'A saved baseline is required for increment comparisons.'


def reconstruct(entry, density, queries):
    nodes = np.unique(entry['velocity'][:, 0])
    count = len(nodes)
    assert queries.min() > nodes.min() and queries.max() < nodes.max(), 'No extrapolation is permitted.'
    matrix = np.ones((len(queries), count))
    for i in range(count):
        for k in range(count):
            if i != k:
                matrix[:, i] *= (queries - nodes[k]) / (nodes[i] - nodes[k])
    correction = (np.log(density) - np.log(gaussian(entry['velocity']))).reshape((count,) * 3)
    correction = np.einsum('ai,bj,ck,ijk->abc', matrix, matrix, matrix, correction, optimize=True)
    points = np.stack(np.meshgrid(queries, queries, queries, indexing='ij'), axis=-1).reshape(-1, 3)
    result = gaussian(points) * np.exp(correction.ravel())
    assert np.isfinite(result).all() and (result > 0).all(), 'Reconstructed density must be positive finite.'
    return result


def whole_moments(entry, density):
    velocity = entry['velocity']
    population = entry['weights'] * density
    number = float(population.sum())
    mean = population @ velocity / number
    centered = velocity - mean
    covariance = population @ (centered * centered) / number
    speed2 = np.sum(velocity * velocity, axis=1)
    mu = (velocity[:, 0] ** 2 + velocity[:, 1] ** 2) / 2
    return {'number': number, 'momentum': population @ velocity,
            'energy': float(population @ speed2 / 2), 'covariance': covariance,
            'anisotropy': float((covariance[0] + covariance[1]) / 2 - covariance[2]),
            'mu_second': float(population @ (mu * mu) / number),
            'speed_fourth': float(population @ (speed2 * speed2) / number),
            'fourth_cumulant': population @ (centered ** 4) / number - 3 * covariance ** 2}


MOMENT_KEYS = ('anisotropy', 'mu_second', 'speed_fourth', 'fourth_cumulant')
TARGET = float(meta['inputs'].get('increment_relative_L1_target', .01))
NORMALIZATION_FLOOR = 1e-14
for entry in data.values():
    entry['moments'] = [whole_moments(entry, density) for density in entry['density']]
base = data['baseline']
shared_indices = {}
for name, entry in data.items():
    if name == 'baseline':
        continue
    shared = []
    for own_index, saved_time in enumerate(entry['time']):
        matches = np.flatnonzero(np.isclose(base['time'], saved_time, rtol=0, atol=1e-13))
        if saved_time > 0 and len(matches) == 1:
            shared.append((int(matches[0]), own_index, float(saved_time)))
    shared_indices[name] = shared

moment_rows = []
native_density_rows = []
for name, shared in shared_indices.items():
    entry = data[name]
    for base_index, own_index, saved_time in shared:
        observables = {}
        for key in MOMENT_KEYS:
            reference = np.asarray(base['moments'][base_index][key]) - np.asarray(base['moments'][0][key])
            own = np.asarray(entry['moments'][own_index][key]) - np.asarray(entry['moments'][0][key])
            scale = float(np.max(abs(reference)))
            assert scale > NORMALIZATION_FLOOR, f'{key} baseline relaxation increment is unresolved.'
            observables[key] = {'reference_increment': reference.tolist(), 'own_increment': own.tolist(),
                                'reference_increment_max_abs_scale': scale,
                                'relative_increment_error': float(np.max(abs(own - reference)) / scale)}
        moment_rows.append({'case': name, 'time': saved_time, 'observables': observables})
        if np.array_equal(entry['velocity'], base['velocity']) and np.array_equal(entry['weights'], base['weights']):
            scale = float(base['weights'] @ abs(base['density'][base_index] - base['density'][0]))
            assert scale > NORMALIZATION_FLOOR, 'Whole-quadrature density increment is unresolved.'
            native_density_rows.append({'case': name, 'time': saved_time,
                                        'relative_increment_L1': float(base['weights'] @ abs(entry['density'][own_index] - base['density'][base_index]) / scale)})

comparison_rows = []
box_status = []
for extent in (4., 5.):
    base_nodes = np.unique(base['velocity'][:, 0])
    for name, entry in data.items():
        if name == 'baseline':
            continue
        own_nodes = np.unique(entry['velocity'][:, 0])
        encloses = min(abs(base_nodes[0]), abs(base_nodes[-1]), abs(own_nodes[0]), abs(own_nodes[-1])) >= extent
        box_status.append({'case': name, 'box_extent': extent,
                           'status': 'compared_at_shared_saved_times' if encloses else 'not_applicable',
                           'reason': None if encloses else 'Saved node axes do not enclose this box; coarse12 skips box5 explicitly.'})
    for order in (32, 48, 64):
        nodes, weights = roots_legendre(order)
        queries = extent * nodes
        weights = extent * weights
        volume = np.prod(np.meshgrid(weights, weights, weights, indexing='ij'), axis=0).ravel()
        if min(abs(base_nodes[0]), abs(base_nodes[-1])) < extent:
            continue
        base_values = [reconstruct(base, density, queries) for density in base['density']]
        for name, shared in shared_indices.items():
            entry = data[name]
            own_nodes = np.unique(entry['velocity'][:, 0])
            if min(abs(own_nodes[0]), abs(own_nodes[-1])) < extent:
                continue
            for base_index, own_index, saved_time in shared:
                reference = base_values[base_index]
                actual = reconstruct(entry, entry['density'][own_index], queries)
                scale = float(volume @ abs(reference - base_values[0]))
                assert scale > NORMALIZATION_FLOOR, 'Common-box density increment is unresolved.'
                comparison_rows.append({'case': name, 'time': saved_time, 'box_extent': extent, 'GL_order': order,
                                        'baseline_increment_L1': scale,
                                        'relative_increment_L1_difference': float(volume @ abs(actual - reference) / scale)})

comparison_summaries = []
producer_agreement = []
for name in shared_indices:
    density_records = [row for row in comparison_rows if row['case'] == name]
    moments = [row for row in moment_rows if row['case'] == name]
    moment_maxima = {key: max((row['observables'][key]['relative_increment_error'] for row in moments), default=None) for key in MOMENT_KEYS}
    density_maxima = {str(extent): max((row['relative_increment_L1_difference'] for row in density_records if row['box_extent'] == extent), default=None) for extent in (4., 5.)}
    summary = {'case': name, 'shared_positive_times': [item[2] for item in shared_indices[name]],
               'earliest_shared_positive_time': min((item[2] for item in shared_indices[name]), default=None),
               'density_maxima_across_all_shared_times_and_GL_orders': density_maxima,
               'whole_quadrature_moment_increment_maxima': moment_maxima}
    refinements = []
    for extent in (4., 5.):
        for _, _, saved_time in shared_indices[name]:
            selected = {record['GL_order']: record for record in density_records
                        if record['box_extent'] == extent and record['time'] == saved_time}
            if 48 in selected and 64 in selected:
                low, high = selected[48], selected[64]
                refinements.append({'box_extent': extent, 'time': saved_time,
                                    'absolute_change_in_increment_error': abs(high['relative_increment_L1_difference'] - low['relative_increment_L1_difference']),
                                    'baseline_increment_relative_change': abs(high['baseline_increment_L1'] / low['baseline_increment_L1'] - 1)})
    summary['GL48_to_64_refinements'] = refinements
    quadrature_target = float(meta['inputs'].get('quadrature_increment_error_target', 1e-4))
    summary['quadrature_refinement_status'] = ('passed' if refinements
                                               and all(item['absolute_change_in_increment_error'] < quadrature_target
                                                       and item['baseline_increment_relative_change'] < TARGET for item in refinements) else 'unresolved')
    summary['target_status'] = ('passed' if density_records and moments
                                and all(value < TARGET for value in density_maxima.values() if value is not None)
                                and all(value is not None and value < TARGET for value in moment_maxima.values())
                                and summary['quadrature_refinement_status'] == 'passed' else 'unresolved')
    comparison_summaries.append(summary)
    published = next((entry for entry in meta.get('comparisons', []) if entry['case'] == name), None)
    if published is not None:
        primary_order = int(meta['inputs'].get('common_cube', [48, 4.])[0])
        primary = max(row['relative_increment_L1_difference'] for row in density_records if row['box_extent'] == 4. and row['GL_order'] == primary_order)
        difference = abs(primary - published['maximum_relative_increment_L1'])
        assert difference < 1e-7, 'Independent density comparison disagrees with producer.'
        moment_difference = None
        if 'whole_quadrature_moment_increment_errors' in published:
            moment_difference = max(abs(moment_maxima[key] - published['whole_quadrature_moment_increment_errors'][key]) for key in MOMENT_KEYS)
            assert moment_difference < 1e-7, 'Independent whole-moment comparison disagrees with producer.'
        producer_agreement.append({'case': name, 'primary_density_maximum_absolute_discrepancy': difference,
                                   'whole_moment_maximum_absolute_discrepancy': moment_difference})


def difference_order(first, second, third, weight=None):
    def norm(value):
        value = abs(np.asarray(value))
        return float(np.max(value)) if weight is None else float(weight @ value)
    coarse = norm(first - second)
    fine = norm(second - third)
    return {'coarse_medium_difference': coarse, 'medium_fine_difference': fine,
            'difference_ratio': coarse / fine if fine > NORMALIZATION_FLOOR else None,
            'observed_log2_order': float(np.log2(coarse / fine)) if min(coarse, fine) > NORMALIZATION_FLOOR else None}


def temporal_refinement(entries):
    names = ('dt_coarse', 'baseline', 'dt_fine')
    if not all(name in entries for name in names):
        return {'status': 'not_applicable', 'reason': 'Three completed factor-two time levels are absent.'}
    selected = [entries[name] for name in names]
    if any(entry['row']['status'] != 'passed_discrete_trajectory' for entry in selected):
        return {'status': 'not_applicable', 'reason': 'Temporal order requires three completed trajectories.'}
    steps = np.asarray([entry['row']['dt'] for entry in selected])
    final_times = np.asarray([entry['time'][-1] for entry in selected])
    if not np.allclose(steps[:-1] / steps[1:], 2., rtol=0, atol=1e-14) or not np.allclose(final_times, final_times[0], rtol=0, atol=1e-14):
        return {'status': 'not_applicable', 'reason': 'Temporal order requires factor-two steps and the same accepted endpoint.'}
    assert all(np.array_equal(entry['velocity'], selected[0]['velocity']) and np.array_equal(entry['weights'], selected[0]['weights']) for entry in selected)
    result = {'status': 'measured', 'case_names': list(names), 'time_steps': steps.tolist(), 'final_time': float(final_times[0]),
              'scope': 'Empirical finite-interval difference ratio and log2 ratio; no order theorem or continuum error estimate.',
              'native_density_L1': difference_order(*[entry['density'][-1] for entry in selected], weight=selected[0]['weights']),
              'whole_quadrature_moment_maxnorm': {key: difference_order(*[np.asarray(entry['moments'][-1][key]) for entry in selected]) for key in MOMENT_KEYS}, 'boxes': []}
    for extent in (4., 5.):
        if any(min(abs(np.unique(entry['velocity'][:, 0])[[0, -1]])) < extent for entry in selected):
            result['boxes'].append({'extent': extent, 'status': 'not_applicable', 'reason': 'Saved axes do not enclose this box.'})
            continue
        quadratures = []
        for order in (32, 48, 64):
            nodes, weights = roots_legendre(order)
            queries = extent * nodes
            weights = extent * weights
            volume = np.prod(np.meshgrid(weights, weights, weights, indexing='ij'), axis=0).ravel()
            values = [reconstruct(entry, entry['density'][-1], queries) for entry in selected]
            quadratures.append({'GL_order': order, **difference_order(*values, weight=volume)})
        result['boxes'].append({'extent': extent, 'status': 'measured', 'quadratures': quadratures})
    return result


temporal_order = temporal_refinement(data)
tails = []
for name, entry in data.items():
    axis_extent = float(np.max(abs(entry['velocity'][:, 0])))
    for index, saved_time in enumerate(entry['time']):
        population = entry['weights'] * entry['density'][index]
        tails.append({'case': name, 'time': float(saved_time), 'node_axis_extent': axis_extent,
                      'outside4_raw_nodal_population': float(population[np.max(abs(entry['velocity']), axis=1) > 4].sum()),
                      'outside5_raw_nodal_population': float(population[np.max(abs(entry['velocity']), axis=1) > 5].sum()),
                      'mass': float(population.sum())})

assert initial_input_hashes == {p.name: digest(p) for p in ROOT.glob('*.npz')}, 'Input changed during audit.'
assert digest(ROOT / 'summary.json') == summary_start_sha256, 'Summary changed during audit.'
result = {'status': 'passed_independent_saved_stage_and_comparison_replay', 'producer_status': meta['status'], 'accepted_stage_count': sum((len(r['stages']) for r in allstage_rows)), 'allstage_rows': allstage_rows, 'density_comparison_rows': comparison_rows, 'box_comparison_status': box_status, 'whole_moment_increment_rows': moment_rows, 'native_density_increment_rows': native_density_rows, 'comparison_summaries': comparison_summaries, 'producer_comparison_agreement': producer_agreement, 'empirical_temporal_order': temporal_order, 'raw_nodal_tail_populations': tails, 'input_summary_sha256': digest(ROOT / 'summary.json'), 'input_npz_sha256': initial_input_hashes, 'audit_source_sha256': source_hash, 'wall_s': time.perf_counter() - start, 'environment': {'python': sys.version, 'platform': platform.platform(), 'versions': {m: metadata.version(m) for m in ['numpy', 'scipy']}, 'thread_caps': {k: os.getenv(k) for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS']}}, 'scope': 'Independent NumPy ordered Coulomb pairs and adjoint, SciPy Hermite measure and barycentric derivatives, raw five invariants, positive density, original root, entropy/Gram/KL/residual identity for every saved accepted advance. Independent direct Lagrange reconstruction of log(F/F0), all saved-case/shared-time box4/5 comparisons and normalized whole-quadrature moment increments; no time interpolation or extrapolation. Separate per-case target status does not override producer global status.', 'limitations': ['Producer interruption/failure remains unresolved; failed Newton iterates and absent states are not accepted stages.', 'The initial Gaussian strong reference is an instantaneous check, not a Gaussian closure for later nodal densities.', 'Discrete root checks and shared-time density comparisons do not establish continuum finite-time convergence.', 'Raw outside-box nodal populations are not rigorous continuum tail bounds.', 'Coarse12 lacks nodes enclosing box5 and retains its box4 producer diagnostic.']}
result.update({'source_commit': source_commit, 'source_dirty': source_dirty,
               'hardware': hardware, 'units': meta.get('units'),
               'input_summary_sha256_at_start': summary_start_sha256,
               'input_summary_sha256_at_end': digest(ROOT / 'summary.json')})
OUTPUT.write_text(json.dumps(result, indent=2) + '\n')
print('Saved', OUTPUT, 'accepted advances', result['accepted_stage_count'], flush=True)
