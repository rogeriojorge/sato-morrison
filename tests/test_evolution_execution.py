"""Audit archived DG provenance and reject incomplete current campaign claims."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
from sato_morrison.geometry import Field


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT/'examples/12_nonuniform_evolution.py'
SEED = ROOT/'results/nonuniform_evolution_initial/summary.json'
# Preserve the validator that produced the archived DG campaign. The current
# lagged-mobility experiment intentionally cannot resume its different method.
VALIDATOR_COMMIT = 'b75a9dc'
VALIDATOR_SOURCE = subprocess.check_output(['git', '-C', str(ROOT), 'show',
    VALIDATOR_COMMIT+':examples/12_nonuniform_evolution.py']).decode()


@pytest.fixture(scope='module')
def execution():
    # Top-level examples intentionally execute their campaign when imported.
    # Load only the pure execution validators and their scientific constants.
    constants = {'FIELDS', 'BOUNDS', 'NX', 'NU', 'NMU', 'X_ORDERS', 'U_ORDERS', 'MU_ORDERS',
        'U_MAX', 'MU_MAX', 'D', 'FINAL_TIME', 'DT', 'DT_VALUES', 'CHUNK', 'NEWTON_RTOL',
        'RELATIVE_TARGET', 'CANONICAL_COMMIT', 'CANONICAL_INPUTS_SHA256', 'PARAMETER_NAMES', 'inputs'}
    functions = {'json_sha256', 'selected_field_names', 'case_parameters', 'scientific_ast',
        'campaign_source_identity', 'require_finite', 'validate_completed_row', 'load_completed_rows'}
    body = []
    for node in ast.parse(VALIDATOR_SOURCE).body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            body.append(node)
        elif isinstance(node, ast.Assign):
            targets = [name for target in node.targets for name in
                (target.elts if isinstance(target, ast.Tuple) else [target])]
            if all(isinstance(name, ast.Name) and name.id in constants for name in targets):
                body.append(node)
    namespace = {'ast': ast, 'hashlib': hashlib, 'json': json, 'Path': Path,
        'subprocess': subprocess, 'np': np, 'Field': Field}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(EXAMPLE), 'exec'), namespace)
    return namespace


def test_recovered_checkpoint_retains_original_run_provenance(execution):
    original = json.loads(SEED.read_text())
    rows, runs, receipt = execution['load_completed_rows'](SEED, ROOT)
    assert len(rows) == 4
    assert receipt['sha256'] == '775e7d165612e85fb603f85871e99ba45241329cd4902daeeabe1d9d5e69fd02'
    for row, before in zip(rows, original['rows']):
        assert {key: value for key, value in row.items() if key != 'provenance_id'} == before
        assert runs[row['provenance_id']] == original['metadata']
    assert len(execution['case_parameters']()) == 13
    assert execution['selected_field_names']('dipole') == ['dipole']
    assert execution['selected_field_names'](None) == ['mirror', 'dipole', 'nonaxisymmetric']
    with pytest.raises(ValueError):
        execution['selected_field_names']('mirror,mirror')
    with pytest.raises(ValueError):
        execution['selected_field_names']('other')


@pytest.mark.parametrize('mutation', ['inputs', 'source', 'duplicate', 'unfinished', 'nonpositive', 'residual', 'linear'])
def test_resume_rejects_incompatible_or_failed_evidence(execution, tmp_path, mutation):
    summary = json.loads(SEED.read_text())
    if mutation == 'inputs':
        summary['metadata']['inputs']['final_time'] = .01
    elif mutation == 'source':
        summary['metadata']['source_sha256'] = '0'*64
    elif mutation == 'duplicate':
        summary['rows'].append(copy.deepcopy(summary['rows'][0]))
    elif mutation == 'unfinished':
        summary['rows'][0]['history'].pop()
    elif mutation == 'nonpositive':
        summary['rows'][0]['history'][-1]['min_f'] = 0.
    elif mutation == 'residual':
        summary['rows'][0]['history'][-1]['nonlinear_relative_residual'] = 1e-8
    else:
        summary['rows'][0]['history'][-1]['maximum_linear_relative_residual'] = .251
    path = tmp_path/'summary.json'
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        execution['load_completed_rows'](path, ROOT)


def test_nested_resume_keeps_row_origins_and_rechecks_status(execution, tmp_path):
    rows, runs, _ = execution['load_completed_rows'](SEED, ROOT)
    summary = {'metadata': {'commit': 'not the row provenance'}, 'provenance_runs': runs,
        'rows': rows, 'checks': [{'status': 'passed'}], 'status': 'passed'}
    summary['rows'][0]['status'] = 'unresolved'
    path = tmp_path/'summary.json';path.write_text(json.dumps(summary))
    accepted, contexts, _ = execution['load_completed_rows'](path, ROOT)
    assert accepted[0]['status'] == 'passed'
    assert contexts == runs
    assert accepted[0]['provenance_id'] == rows[0]['provenance_id']
    summary['rows'][0]['provenance_id'] = 'unknown'
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='run context'):
        execution['load_completed_rows'](path, ROOT)


def test_scientific_anchor_includes_base_dt_and_functions(execution):
    source = VALIDATOR_SOURCE
    anchor = subprocess.check_output(['git', '-C', str(ROOT), 'show',
        execution['CANONICAL_COMMIT']+':examples/12_nonuniform_evolution.py']).decode()
    fingerprint = execution['scientific_ast']
    assert fingerprint(source) == fingerprint(anchor)
    assert fingerprint(source.replace('D, FINAL_TIME, DT = .1, .02, .005',
        'D, FINAL_TIME, DT = .1, .02, .01')) != fingerprint(anchor)
    assert fingerprint(source.replace('initial = jnp.exp(log_equilibrium+.1*observable)',
        'initial = jnp.exp(log_equilibrium+.2*observable)')) != fingerprint(anchor)


@pytest.fixture(scope='module')
def current_execution():
    constants = {'FIELDS', 'DT_VALUES', 'RELATIVE_TARGET', 'CASES', 'SEQUENCES'}
    functions = {'select_names', 'compare', 'refinement_checks'}
    body = []
    for node in ast.parse(EXAMPLE.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            body.append(node)
        elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id in constants for t in node.targets):
            body.append(node)
    namespace = {'Field': Field}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(EXAMPLE), 'exec'), namespace)
    return namespace


def test_case_jobs_are_distinct_and_partial_evidence_cannot_pass(current_execution):
    cases = current_execution['CASES']
    assert len(cases) == 15
    assert len({json.dumps(case, sort_keys=True) for case in cases.values()}) == 15
    with pytest.raises(ValueError):
        current_execution['select_names']('base,base', cases)
    with pytest.raises(ValueError):
        current_execution['select_names']('unplanned', cases)
    assert current_execution['select_names'](None, cases) == list(cases)
    rows = json.loads(SEED.read_text())['rows']
    for row, name in zip(rows, ['spatial3', 'base', 'spatial7', 'u17']):
        row['case'] = name
    # Only exercise aggregation using known measurements, not method equivalence.
    checks = current_execution['refinement_checks'](rows)
    assert checks[0]['spatial']['status'] == 'passed'
    assert checks[0]['parallel_velocity']['status'] == 'not_run'
    assert checks[0]['timestep']['status'] == 'not_run'
    assert all(check['status'] == 'unresolved' for check in checks)


def test_current_campaign_rejects_historical_resume(monkeypatch):
    import os
    node = next(node for node in ast.parse(EXAMPLE.read_text()).body
        if isinstance(node, ast.If) and 'SM_EVOLUTION_RESUME' in ast.unparse(node.test))
    monkeypatch.setenv('SM_EVOLUTION_RESUME', str(SEED))
    with pytest.raises(ValueError, match='cannot resume historical discrete-gradient'):
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(EXAMPLE), 'exec'), {'os': os})


def test_fine_pair_pass_keeps_unresolved_coarse_bias(current_execution):
    names=('entropy_gain_per_particle','final_production_per_particle',
           'relaxation_moment_change_per_particle')
    rows=[dict(nx=5,nu=25,nmu=21,umax=4.,mumax=20.,dt=dt,
        **{name:value for name in names}) for dt,value in ((.005,2.),(.0025,1.001),(.00125,1.))]
    result=current_execution['compare'](rows)
    assert result['status']=='passed'
    assert result['adjacent_comparisons'][0]['status']=='unresolved'
    assert result['coarsest_to_finest']['status']=='unresolved'
    assert result['coarsest_to_finest']['relative_changes'][names[0]]==1.
