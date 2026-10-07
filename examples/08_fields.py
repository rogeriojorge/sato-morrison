"""Nonuniform collision-only boxes: weak equilibria and nonlinear entropy steps."""

import json
from pathlib import Path
from time import perf_counter
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from sato_morrison.collisions import cartesian_grid,mobility_action,nonlinear_rhs
from sato_morrison.geometry import Field,validate_geometry
from sato_morrison.solver import discrete_gradient_compiler,discrete_gradient_step,invariant_diagnostics
from sato_morrison.reference import run_metadata,progress

jax.config.update('jax_enable_x64',True)

# Inputs: collision-only natural no-flux boxes, not a confined Hamiltonian domain.
fields=[Field('mirror',amplitude=.15),Field('dipole'),Field('nonaxisymmetric',amplitude=.03)]
spatial_orders=[3,5,7]
bounds=[(.8,1.2),(-.2,.2),(.1,.5)]
u=np.array([-.7,.1,.9])
mu=np.array([.15,.45,.9])
strength,dt,steps=.1,.002,3
output=Path(__file__).resolve().parents[1]/'results'/'fields'
output.mkdir(parents=True,exist_ok=True)
print(f'sm_local_nonlinear; fixed normalized eta projector, B dx du dmu; natural collision flux; {output}',flush=True)
rows=[]
for field in fields:
    for order in spatial_orders:
        axes=[np.linspace(a,b,order) for a,b in bounds]
        with progress(f'Setup and compile {field.kind} collision box {order}^3 x 3 x 3'):
            grid=cartesian_grid(*axes,u,mu,field)
            x,y,z,uu,mm=np.meshgrid(*axes,u,mu,indexing='ij')
            equilibrium=jnp.exp(-grid.energy-.2*jnp.asarray(mm.ravel()))
            h=jnp.asarray((.1*np.sin(x)*uu+.04*y*mm).ravel())
            initial=equilibrium*jnp.exp(h)
            apply=jax.jit(lambda state,test:mobility_action(grid,state,test,collision_strength=strength,chunk_size=128))
            start=perf_counter();kh=apply(initial,jnp.log(initial));kh.block_until_ready();compile_s=perf_counter()-start
            stationarity=float(jnp.max(jnp.abs(apply(equilibrium,jnp.log(equilibrium))/grid.weights)))
            production=float(jnp.vdot(jnp.log(initial),kh))
        positions=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
        row=dict(field=field.kind,spatial_order=order,nu=len(u),nmu=len(mu),
                 min_sampled_B=validate_geometry(positions,field),equilibrium_residual=stationarity,
                 initial_entropy_production=production,compile_and_first_s=compile_s,
                 state_bytes=grid.size*8,pair_count=len(grid.left),pair_chunk=128,
                 status='passed' if stationarity<1e-10 and production>=-1e-14 else 'failed')
        if order==spatial_orders[0]:
            with progress(f'Compiling nonlinear residual/Jacobian and stepping {field.kind}'):
                compiler=discrete_gradient_compiler(grid,collision_strength=strength)
                state=initial;increments=[];identity=[]
                for step_index in range(steps):
                    result=discrete_gradient_step(grid,state,dt,collision_strength=strength,compiled_residual=compiler,rtol=1e-12)
                    state=result.f;increments.append(result.entropy_change);identity.append(abs(result.entropy_identity_error))
                    print(f'  {field.kind} step {step_index+1}/{steps}: entropy +{increments[-1]:.3e}, residual {result.relative_residual:.2e}',flush=True)
                row.update(invariant_diagnostics(grid,state,initial))
                row.update(entropy_min_step=min(increments),entropy_identity_error=max(identity),dt=dt,steps=steps)
                if max(row[k] for k in ['number_error','energy_error','marginal_error'])>1e-9 or row['min_f']<=0:
                    raise RuntimeError(f'Nonlinear conservation/positivity failed: {row}')
        rows.append(row)
        print(row,flush=True)
    selected=[r for r in rows if r['field']==field.kind]
    finest_change=abs(selected[-1]['initial_entropy_production']/selected[-2]['initial_entropy_production']-1)
    selected[-1]['last_spatial_refinement_relative_change']=finest_change
    selected[-1]['spatial_convergence_status']='passed' if finest_change<.01 else 'unresolved'
metadata=run_metadata(dict(fields=[f.__dict__ for f in fields],spatial_orders=spatial_orders,bounds=bounds,
                          u=u.tolist(),mu=mu.tolist(),collision_strength=strength,dt=dt,steps=steps,
                          initial='exp(-E-0.2mu+0.1 sin(x)u+0.04 y mu)',pair_chunk=128),
                      model='sm_local_nonlinear discrete-energy projector',boundary='natural no-flux collision-only Cartesian box')
metadata['limitations']='Only spatial entropy-production refinement and coarse nonlinear steps here. Velocity/tail convergence for these boxes unresolved; no combined mirror/dipole/nonaxisymmetric confinement claim. Minimum B is sampled, not a proof over the full domain.'
(output/'summary.json').write_text(json.dumps(dict(metadata=metadata,rows=rows),indent=2)+'\n')
fig,ax=plt.subplots(figsize=(6.2,3.7))
for field in fields:
    selected=[r for r in rows if r['field']==field.kind]
    ax.plot(spatial_orders,[r['initial_entropy_production'] for r in selected],'o-',label=field.kind)
ax.set(xlabel='Nodes on each spatial axis',ylabel='Initial entropy production (normalized)')
ax.legend();fig.tight_layout();fig.savefig(output/'production.png',dpi=180);plt.close(fig)
if any(r['status']=='failed' for r in rows):
    raise RuntimeError('Collision field structural validation failed')
print('PASS: field-box equilibria and nonlinear structural steps; inspect convergence status per direction',flush=True)
