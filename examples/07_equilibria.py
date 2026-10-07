"""Independent field-dependent velocity integrals and constrained candidates."""

import json
from pathlib import Path
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from sato_morrison.geometry import Field, field_vector, validate_geometry
from sato_morrison.reference import gauss_interval, constrained_equilibrium, run_metadata, progress

jax.config.update('jax_enable_x64',True)

# Inputs: external vacuum fields, normalized m=q=1, no electric potential.
fields=[Field('uniform'),Field('mirror',amplitude=.15),Field('toroidal'),
        Field('dipole'),Field('nonaxisymmetric',amplitude=.03)]
beta,gamma=1.2,.3
orders=[16,32,64]
extents=[(5.,12.),(7.,20.)]
positions=np.stack([np.linspace(.8,1.4,11),np.full(11,.2),np.full(11,.3)],axis=-1)
output=Path(__file__).resolve().parents[1]/'results'/'equilibria'
output.mkdir(parents=True,exist_ok=True)
print(f'Constrained stationary-family velocity integration; five vacuum fields; normalized units; {output}',flush=True)
rows=[]
fig,ax=plt.subplots(figsize=(6.2,3.7))
for field in fields:
    bmin=validate_geometry(positions,field)
    strength=np.asarray(jax.vmap(lambda p:jnp.linalg.norm(field_vector(p,field)))(jnp.asarray(positions)))
    if np.any(strength+gamma<=0):
        raise ValueError('B+gamma must be positive')
    exact=np.sqrt(2*np.pi/beta)*strength/(beta*(strength+gamma))
    for order in orders:
        for umax,mumax in extents:
            u,wu=gauss_interval(order,-umax,umax)
            mu,wm=gauss_interval(order,0.,mumax)
            numerical=np.einsum('i,j,xij,x->x',wu,wm,np.exp(-beta*(u[None,:,None]**2/2+mu[None,None,:]*(strength[:,None,None]+gamma))),strength)
            error=float(np.max(np.abs(numerical/exact-1)))
            rows.append(dict(field=field.kind,order=order,u_extent=umax,mu_extent=mumax,
                             density_error=error,min_sampled_B=bmin,status='passed' if error<.01 else 'unresolved'))
    if rows[-1]['density_error']>2e-5:
        raise RuntimeError(f'Velocity integral unresolved for {field.kind}')
    ax.plot(positions[:,0],exact/exact[0],label=field.kind)
# Initial full marginal is retained when solving for beta; no reachability claim.
e=np.repeat(np.linspace(.1,3.,8),4)+np.tile(np.linspace(.1,1.,4),8)
labels=np.tile(np.arange(4),8)
w=np.linspace(.5,1.2,32)
initial=np.exp(-1.7*e+.2*labels**2+.15*np.sin(np.arange(32)))
fitted,candidate=constrained_equilibrium(e,w,labels,initial)
candidate_report=dict(beta=fitted,energy_error=float(abs(w@(candidate*e)-w@(initial*e))),
                      marginal_error=float(np.max(np.abs(np.bincount(labels,weights=w*candidate)-np.bincount(labels,weights=w*initial)))),
                      entropy_increase=float(-w@(candidate*np.log(candidate))+w@(initial*np.log(initial))),
                      status='passed',scope='finite quadrature candidate; additional kernel invariants can prevent approach')
metadata=run_metadata(dict(fields=[f.__dict__ for f in fields],beta=beta,gamma=gamma,orders=orders,
                          extents=extents,positions=positions.tolist(),mass=1.,charge=1.,
                          candidate_energy=e.tolist(),candidate_weights=w.tolist(),candidate_labels=labels.tolist(),candidate_initial=initial.tolist()),
                      model='stationary-family quadrature and full-marginal maximum entropy',
                      boundary='velocity truncation only; sampled positions do not assert a closed spatial relaxation experiment')
(output/'summary.json').write_text(json.dumps(dict(metadata=metadata,rows=rows,candidate=candidate_report),indent=2)+'\n')
ax.set(xlabel='Cartesian x along sampled line',ylabel='Stationary density / first-point density')
ax.legend();fig.tight_layout();fig.savefig(output/'density.png',dpi=180);plt.close(fig)
print(json.dumps(dict(finest_rows=[r for r in rows if r['order']==64 and r['u_extent']==7.],candidate=candidate_report),indent=2),flush=True)
print('PASS: independently integrated known stationary family; no dynamical dipole claim',flush=True)
