"""Independent saved-stage Landau audit, executed directly with NumPy/SciPy.

Edit INPUT_DIRECTORY to audit another immutable campaign. This source imports no
production collision kernel. It rebuilds Hermite measure and barycentric D,
replays every saved raw Coulomb root and entropy/KL identity, and compares full
nodal density increments on shared times in boxes 4/5 with GL32/48/64. The
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
INPUT_DIRECTORY = Path(__file__).resolve().parents[1] / 'results' / 'landau_trajectory'
OUTPUT = INPUT_DIRECTORY / 'independent_portable_audit.json'
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
    row = {'case': case['name'], 'declared_status': case['status'], 'source_arrays_sha256': digest(path), 'n': n, 'theta': theta, 'softening': case.get('softening', 0.0), 'saved_stages': len(states), 'last_time': float(times[-1]), 'nodal_min_density': float(states.min()), 'quadrature_max_relative_difference': float(np.max(abs((w - ww) / ww))), 'derivative_max_scaled_difference': float(np.max(abs(savedD - D) / np.maximum(abs(D), 1.0))), 'weighted_SBP_max_abs_defect': float(abs(weighted_sbp).max()), 'raw_invariant_max_abs_drifts': invariant_error.tolist(), 'stages': []}
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
        audit_root_tolerance = max(2e-12, 2 * case.get('rtol', 1e-10))
        assert norm < audit_root_tolerance
        assert stage['initial_reference_root_norm'] < audit_root_tolerance
        assert change > -1e-10 and production > -1e-10 and KL > -1e-10
        assert abs(identity) < 1e-10
    rows.append(row)
allstage_rows = rows
assert sum((len(r['stages']) for r in rows)) > 0
data = {}
for name in ['baseline', 'grid20']:
    with np.load(ROOT / (name + '.npz')) as arrays:
        data[name] = {k: arrays[k].copy() for k in ['velocity', 'weights', 'density', 'time']}

def reconstruct(a, f, axis):
    x = np.unique(a['velocity'][:, 0])
    n = len(x)
    assert axis.min() > x.min() and axis.max() < x.max()
    L = np.ones((len(axis), n))
    for i in range(n):
        for k in range(n):
            if i != k:
                L[:, i] *= (axis - x[k]) / (x[i] - x[k])
    g = (np.log(f) - np.log(gaussian(a['velocity']))).reshape(n, n, n)
    q = np.einsum('ai,bj,ck,ijk->abc', L, L, L, g, optimize=True)
    v = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1).reshape(-1, 3)
    result = gaussian(v) * np.exp(q.ravel())
    assert np.all(np.isfinite(result)) and np.all(result > 0), 'Reconstructed density must be positive finite.'
    return result
rows = []
b = data['baseline']
fine = data['grid20']
for L in [4.0, 5.0]:
    for order in [32, 48, 64]:
        z, q = roots_legendre(order)
        axis = L * z
        q = L * q
        w = np.prod(np.meshgrid(q, q, q, indexing='ij'), axis=0).ravel()
        f0 = reconstruct(b, b['density'][0], axis)
        for k, t in enumerate(fine['time']):
            if t == 0:
                continue
            ids = np.flatnonzero(np.isclose(b['time'], t, rtol=0, atol=1e-13))
            assert len(ids) == 1
            j = int(ids[0])
            fb = reconstruct(b, b['density'][j], axis)
            ff = reconstruct(fine, fine['density'][k], axis)
            den = w @ abs(fb - f0)
            row = {'box_extent': L, 'GL_order': order, 'time': float(t), 'baseline_increment_L1': float(den), 'fine_increment_L1': float(w @ abs(ff - f0)), 'relative_increment_L1_difference': float(w @ abs(ff - fb) / den)}
            rows.append(row)
            print(row, flush=True)
tails = []
for name, a in data.items():
    for j, t in enumerate(a['time']):
        p = a['weights'] * a['density'][j]
        tails.append({'case': name, 'time': float(t), 'outside4_raw_nodal_population': float(p[np.max(abs(a['velocity']), axis=1) > 4].sum()), 'outside5_raw_nodal_population': float(p[np.max(abs(a['velocity']), axis=1) > 5].sum()), 'mass': float(p.sum())})
assert initial_input_hashes == {p.name: digest(p) for p in ROOT.glob('*.npz')}, 'Input changed during audit.'
assert digest(ROOT / 'summary.json') == summary_start_sha256, 'Summary changed during audit.'
result = {'status': 'passed_independent_saved_stages_and_expanded_boxes', 'producer_status': meta['status'], 'accepted_stage_count': sum((len(r['stages']) for r in allstage_rows)), 'allstage_rows': allstage_rows, 'expanded_box_rows': rows, 'raw_nodal_tail_populations': tails, 'input_summary_sha256': digest(ROOT / 'summary.json'), 'input_npz_sha256': initial_input_hashes, 'audit_source_sha256': source_hash, 'wall_s': time.perf_counter() - start, 'environment': {'python': sys.version, 'platform': platform.platform(), 'versions': {m: metadata.version(m) for m in ['numpy', 'scipy']}, 'thread_caps': {k: os.getenv(k) for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS']}}, 'scope': 'Independent NumPy ordered Coulomb pairs and adjoint, SciPy Hermite measure and barycentric derivatives, raw five invariants, positive density, original root, entropy/Gram/KL/residual identity for every saved accepted advance. Independent direct Lagrange reconstruction of log(F/F0) and Legendre cubature on shared saved times; no time interpolation or extrapolation.', 'limitations': ['Producer interruption/failure remains unresolved; failed Newton iterates and absent states are not accepted stages.', 'The initial Gaussian strong reference is an instantaneous check, not a Gaussian closure for later nodal densities.', 'Discrete root checks and shared-time density comparisons do not establish continuum finite-time convergence.', 'Raw outside-box nodal populations are not rigorous continuum tail bounds.', 'Coarse12 lacks nodes enclosing box5 and retains its box4 producer diagnostic.']}
result.update({'source_commit': source_commit, 'source_dirty': source_dirty,
               'hardware': hardware, 'units': meta.get('units'),
               'input_summary_sha256_at_start': summary_start_sha256,
               'input_summary_sha256_at_end': digest(ROOT / 'summary.json')})
OUTPUT.write_text(json.dumps(result, indent=2) + '\n')
print('Saved', OUTPUT, 'accepted advances', result['accepted_stage_count'], flush=True)
