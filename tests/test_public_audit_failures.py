"""Public audit commands must fail visibly when private evidence is corrupted."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('archive,script,input_variable,output_variable,corruption', [
    ('stratified_encounter_flux', 'audit_saved_states.py', 'SM_STRATIFIED_AUDIT_INPUT',
     'SM_STRATIFIED_AUDIT_OUTPUT', 'mass'),
    ('encounter_born_control', 'audit_saved_states.py', 'SM_BORN_AUDIT_INPUT',
     'SM_BORN_AUDIT_OUTPUT', 'mass'),
    ('antithetic_encounter_flux', 'audit_saved_states.py', 'SM_ANTITHETIC_AUDIT_INPUT',
     'SM_ANTITHETIC_AUDIT_OUTPUT', 'mass'),
    ('antithetic_encounter_flux', 'audit_contribution_maps.py', 'SM_ANTITHETIC_AUDIT_INPUT',
     'SM_ANTITHETIC_MAP_AUDIT_OUTPUT', 'map_bin'),
    ('antithetic_encounter_movie', 'audit_saved_paths.py', 'SM_PHASE_MOVIE_AUDIT_INPUT',
     'SM_PHASE_MOVIE_AUDIT_OUTPUT', 'mass'),
    ('antithetic_encounter_flux', 'audit_saved_refinement.py', 'SM_ANTITHETIC_AUDIT_INPUT',
     'SM_ANTITHETIC_REFINEMENT_AUDIT_OUTPUT', 'fine_control'),
    ('antithetic_encounter_flux', 'audit_cartesian.py', 'SM_ANTITHETIC_AUDIT_INPUT',
     'SM_ANTITHETIC_CARTESIAN_OUTPUT', 'producer_hash'),
])
def test_corrupted_public_evidence_writes_failed_receipt_and_exits_nonzero(
        tmp_path, archive, script, input_variable, output_variable, corruption):
    # Keep historical Git blobs available, but isolate every modified byte and receipt.
    (tmp_path / '.git').symlink_to(ROOT / '.git', target_is_directory=True)
    results = tmp_path / 'results'
    results.mkdir()
    for path in (ROOT / 'results').iterdir():
        if path.name != archive:
            (results / path.name).symlink_to(path, target_is_directory=path.is_dir())
    private = results / archive
    private.mkdir()
    changed_name = {'map_bin': 'contribution_maps.npz',
                    'fine_control': 'cartesian_audit.json'}.get(corruption, 'summary.json')
    for path in (ROOT / 'results' / archive).iterdir():
        target = private / path.name
        if path.name in (script, changed_name):
            shutil.copy2(path, target)
        else:
            target.symlink_to(path, target_is_directory=path.is_dir())
    changed = private / changed_name
    original = ROOT / 'results' / archive / changed_name
    original_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    assert not changed.is_symlink()
    if corruption == 'map_bin':
        with np.load(changed) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        arrays['b_ratio_mean_second'][0, 0] += 1.0
        np.savez(changed, **arrays)
    else:
        data = json.loads(changed.read_text())
        if corruption == 'mass':
            data['inputs']['mass'] *= 1.1
        elif corruption == 'fine_control':
            failed = next(row for row in data['replays'] if not row['passed'])
            failed['finest_available_control_comparison']['passed'] = False
        else:
            key = next(iter(data['experiment_dependency_sha256']))
            data['experiment_dependency_sha256'][key] = '0' * 64
        changed.write_text(json.dumps(data))
    receipt_path = tmp_path / 'fresh_audit.json'
    env = dict(os.environ, SM_AUDIT_ROOT=str(tmp_path))
    env[input_variable] = str(private.relative_to(tmp_path))
    env[output_variable] = str(receipt_path)
    run = subprocess.run([sys.executable, str(private / script)], cwd=tmp_path,
                         env=env, capture_output=True, text=True, timeout=120)
    assert hashlib.sha256(original.read_bytes()).hexdigest() == original_hash
    assert receipt_path.exists(), run.stdout + run.stderr
    receipt = json.loads(receipt_path.read_text())
    assert receipt['status'] == 'failed', run.stdout + run.stderr
    assert run.returncode != 0, run.stdout + run.stderr
    if corruption == 'producer_hash':
        # The real coarse Cartesian comparison already fails; prove the altered
        # producer identity was independently detected as well.
        assert not receipt['producer_source_integrity']
    elif corruption == 'fine_control':
        assert any(not row['refined_cartesian_passed'] for row in receipt['refinements'])
    elif corruption == 'map_bin':
        assert receipt['maximum_bin_absolute_errors']['b_ratio_mean_second'] > 0.9
