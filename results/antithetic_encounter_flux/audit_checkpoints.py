"""Verify original checkpoint row bytes against the combined saved archive."""
from pathlib import Path
from time import perf_counter
import hashlib
import json
import os
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
D=Path(__file__).resolve().parent
SOURCE=Path(os.environ.get('SM_ANTITHETIC_CHECKPOINT_DIRECTORY',D))
OUTPUT=D/'checkpoint_manifest.json'
if OUTPUT.exists():raise FileExistsError(OUTPUT)
started=perf_counter()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
original=json.loads((D/'original_attempt.json').read_text())
with np.load(D/'states.npz') as saved: final={k:saved[k] for k in saved.files}
paths=sorted(SOURCE.glob('checkpoint_*.npz'))
if not paths:raise FileNotFoundError('Original checkpoint files required')
position=0;records=[]
for path in paths:
    endpoint=int(path.stem.split('_')[-1])
    with np.load(path) as saved:block={k:saved[k] for k in saved.files}
    count=len(block['nodes']);end=position+count
    columns={}
    for k in final:
        present=k in block
        shape=present and block[k].shape==(count,*final[k].shape[1:])
        # Concatenation promotes chunk string widths; compare the exact final dtype bytes.
        equal=shape and np.ascontiguousarray(block[k],dtype=final[k].dtype).tobytes()==np.ascontiguousarray(final[k][position:end]).tobytes()
        columns[k]=dict(passed=bool(equal),original_dtype=str(block[k].dtype) if present else None,final_dtype=str(final[k].dtype))
    digest=sha(path);expected=original['saved_artifact_sha256'].get(path.name)
    records.append(dict(logical_original_path=str((D/path.name).relative_to(ROOT)),sha256=digest,
        manifest_sha256=expected,start_row_inclusive=position,end_row_exclusive=end,row_count=count,
        passed=bool(endpoint==end and digest==expected and set(block)==set(final) and all(c['passed'] for c in columns.values())),columns=columns))
    position=end
passed=position==len(final['nodes'])==original['saved_new_encounters'] and all(r['passed'] for r in records)
script=Path(__file__)
receipt=dict(status='passed' if passed else 'failed',scope='Original compressed checkpoint hashes and ordered concatenation of every saved column exactly reconstruct states.npz; comparison is bitwise after the dtype promotion used by concatenation. No original artifact is rewritten or removed.',
    checkpoint_count=len(records),row_count=position,combined_state_count=len(final['nodes']),checkpoints=records,
    input_sha256={f.name:sha(f) for f in (D/'states.npz',D/'original_attempt.json',D/'failed_original_attempt.log')},
    source_sha256=sha(script),source_path=str(script.relative_to(ROOT)),source_uncommitted=bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--',str(script.relative_to(ROOT))],text=True).strip()),
    audit_git_identity=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
    numpy_version=np.__version__,python_version=sys.version.split()[0],wall_s=perf_counter()-started)
OUTPUT.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:receipt[k] for k in ('status','checkpoint_count','row_count','wall_s')},indent=2))
if not passed:sys.exit(1)
