"""A corrupted receipt must fail independently without copying heavy arrays."""
from pathlib import Path
import json
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'results/reciprocal_encounter_core'


@pytest.mark.parametrize('corruption', ['core_moment', 'trajectory_gate'])
def test_reciprocal_audit_rejects_corruption_with_fresh_failed_receipt(tmp_path, corruption):
    if not (OUTPUT/'summary.json').exists():
        pytest.skip('The fixed reciprocal campaign has not produced its baseline yet.')
    baseline = json.loads((OUTPUT/'independent_audit.json').read_text())
    assert baseline['status'] == 'passed'
    copied = tmp_path/'results/reciprocal_encounter_core'
    copied.mkdir(parents=True)
    # Every heavy archive remains in its original location. Only the mutable
    # JSON target and the audit source are copied into this isolated tree.
    for folder in ('examples', 'src'):
        (tmp_path/folder).symlink_to(ROOT/folder, target_is_directory=True)
    for folder in ('antithetic_encounter_flux', 'stratified_encounter_flux'):
        (tmp_path/'results'/folder).symlink_to(ROOT/'results'/folder, target_is_directory=True)
    mutable = 'saved_balance.json' if corruption == 'core_moment' else 'summary.json'
    for path in OUTPUT.iterdir():
        if path.name in ('independent_audit.json', 'independent_audit.json.tmp') or path.is_dir():
            continue
        if path.name in (mutable, 'audit_saved_states.py'):
            shutil.copyfile(path, copied/path.name)
        else:
            (copied/path.name).symlink_to(path)
    changed = json.loads((copied/mutable).read_text())
    if corruption == 'core_moment':
        changed['cases'][0]['core'][0] += .01
    else:
        old = changed['results']['trajectory_gates']['endpoints']
        changed['results']['trajectory_gates']['endpoints'] = not old
    (copied/mutable).write_text(json.dumps(changed))
    # A stale passing receipt must never survive a failed verification.
    (copied/'independent_audit.json').write_text(json.dumps({'status': 'passed', 'stale': True}))
    run = subprocess.run([sys.executable, str(copied/'audit_saved_states.py')],
                         cwd=tmp_path, text=True, capture_output=True, timeout=120)
    assert run.returncode != 0, run.stdout+run.stderr
    receipt = json.loads((copied/'independent_audit.json').read_text())
    assert receipt['status'] == 'failed' and not receipt.get('stale', False)
    assert receipt['failed_checks'] > 0
    expected = ':core' if corruption == 'core_moment' else 'all_trajectory_gates_faithfully_reproduced'
    assert any(expected in name for name in receipt['failed_check_details'])
