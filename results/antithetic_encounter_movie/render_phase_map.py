"""Display the phase involution used by the fixed Cartesian paths."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
summary=json.loads((HERE/'summary.json').read_text())
assert abs(sum(summary['inputs']['phases'])-3.141592653589793)<1e-14
plt.rcParams.update({'svg.fonttype':'path','figure.facecolor':'white','text.color':'#253447'})
fig=plt.figure(figsize=(10.3,.75))
fig.text(.5,.40,r'$T(b,u,p,\varphi)=(b,u,p,(\pi-\varphi)\;\mathrm{mod}\;2\pi),\qquad T^2=I$',ha='center',fontsize=19)
for ext in ('svg','png'):
    fig.savefig(HERE/f'phase_map.{ext}',dpi=160,bbox_inches='tight',pad_inches=.15,metadata={'Creator':None,'Date':None} if ext=='svg' else None)
plt.close(fig)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(HERE/'phase_map_receipt.json').write_text(json.dumps(dict(scope='Phase-map equation only; no trajectory recomputation.',
    input_sha256={'summary.json':sha(HERE/'summary.json')},source_sha256=sha(Path(__file__)),
    outputs_sha256={f'phase_map.{ext}':sha(HERE/f'phase_map.{ext}') for ext in ('svg','png')}),indent=2)+'\n')
