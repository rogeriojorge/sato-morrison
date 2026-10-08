"""Fixed tensor quadrature and the positive-amplitude mirror limit.

Q_h is a frozen-mobility Gram, not entropy production of the equilibrium f.
The finite-quadrature discrepancy vanishes under parallel-velocity refinement.
"""
import hashlib
import json
from pathlib import Path
from time import perf_counter
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from sato_morrison.geometry import Field,common_chart_action,energy_mu
from sato_morrison.reference import gauss_interval,progress,run_metadata

POSITION=np.array([1.,0.,.3])
MASS=CHARGE=STRENGTH=D=1.
UMAX,MUMAX,NMU=4.,5.,8
ORDERS=(8,16,32,64)
AMPLITUDES=np.logspace(-7,-1,10)
OUTPUT=Path(__file__).resolve().parents[1]/'results'/'near_uniform'


def analytic_forms(nu):
    """Exact finite-quadrature uniform form, positive-a limit and gap bound."""
    u,wu=gauss_interval(nu,-UMAX,UMAX)
    mu,wm=gauss_interval(NMU,0.,MUMAX)
    moments=[np.sum(wm*np.exp(-(STRENGTH+.2)*mu)*mu**k) for k in range(3)]
    moment_gram=moments[0]*moments[2]-moments[1]**2
    spatial=(POSITION[0]**2+POSITION[1]**2)/(CHARGE*STRENGTH)**2
    transverse=POSITION[0]**2/(CHARGE*STRENGTH)**2
    squared=spatial+4*POSITION[2]**2*(1/MASS**2+u*u)
    ratio=1-2*transverse/squared+transverse*spatial/squared**2
    moment_gram*=D/CHARGE**2
    uniform=moment_gram*np.sum(wu*np.exp(-MASS*u*u/2))**2
    gap=moment_gram*np.sum(wu*wu*np.exp(-MASS*u*u)*(1-ratio))
    bound=moment_gram*max(wu)*np.sum(wu*np.exp(-MASS*u*u))
    return dict(uniform=float(uniform),positive_amplitude_limit=float(uniform-gap),
        gap=float(gap),gap_bound=float(bound),relative_gap=float(gap/uniform),
        relative_bound=float(bound/uniform),max_parallel_weight=float(max(wu)),
        equal_u_uniform=float(moment_gram*np.sum(wu*wu*np.exp(-MASS*u*u))),
        equal_u_limit=float(moment_gram*np.sum(wu*wu*np.exp(-MASS*u*u)*ratio)))


def direct_form(nu,amplitude,*,check=True):
    """Every unordered pair, analytic vacuum flow/action and independent AD checks.

    Uses actual positive-amplitude directions; no singular projector is evaluated.
    The local pair measure is B² wu_i wmu_i wu_j wmu_j f_i f_j.
    """
    if not np.isfinite(amplitude) or amplitude<=0:
        raise ValueError('a strictly positive finite amplitude is required')
    x,y,z=POSITION;a=amplitude
    vector=np.array([-a*z*x,-a*z*y,STRENGTH+a*(z*z-(x*x+y*y)/2)])
    derivative=np.array([[-a*z,0.,-a*x],[0.,-a*z,-a*y],[-a*x,-a*y,2*a*z]])
    bmag=np.linalg.norm(vector);b=vector/bmag;grad=derivative.T@b
    t=np.cross(b,grad);parallel=b@grad
    u,wu=gauss_interval(nu,-UMAX,UMAX)
    mu,wm=gauss_interval(NMU,0.,MUMAX)
    uu,mm=map(np.ravel,np.meshgrid(u,mu,indexing='ij'))
    flow=np.column_stack((uu[:,None]*b+(MASS*uu*uu+mm*bmag)[:,None]*t/(CHARGE*bmag**2),
        -mm*parallel/MASS,mm*uu*parallel))
    # h=mu*x; the covector's mu entry is x and is a Poisson Casimir direction.
    ax=mm[:,None]*np.cross(b,np.array([1.,0.,0.]))/(CHARGE*bmag)
    au=-mm*(b[0]/MASS+uu*t[0]/(CHARGE*bmag**2))
    action=np.column_stack((ax,au,mm*(ax@grad)))
    error=0.
    if check:
        field=Field('mirror',amplitude=a,strength=STRENGTH)
        for i in (0,len(uu)//2,len(uu)-1):
            state=jnp.asarray(np.r_[POSITION,uu[i],mm[i]])
            expected_flow=np.asarray(common_chart_action(state,jax.grad(lambda point:energy_mu(point,field,MASS,CHARGE))(state),field,MASS,CHARGE))
            expected_action=np.asarray(common_chart_action(state,jnp.array([mm[i],0.,0.,0.,x]),field,MASS,CHARGE))
            for value,expected in ((flow[i],expected_flow),(action[i],expected_action)):
                error=max(error,float(np.linalg.norm(value-expected)/max(1.,np.linalg.norm(expected))))
    left,right=np.triu_indices(len(uu),1)
    xi=flow[left]-flow[right];norm=np.linalg.norm(xi,axis=1)
    if np.any(norm==0):raise ValueError('distinct-state energy-flow zero needs a separate limit')
    direction=xi/norm[:,None];delta=action[left]-action[right]
    projected=delta-direction*np.sum(direction*delta,axis=1)[:,None]
    f=np.exp(-MASS*uu*uu/2-(bmag+.2)*mm)
    weights=bmag*np.outer(wu,wm).ravel()
    values=D*weights[left]*weights[right]*f[left]*f[right]*np.sum(projected[:,:3]**2,axis=1)
    equal_u=left//NMU==right//NMU
    return dict(gram=float(values.sum()),equal_u_gram=float(values[equal_u].sum()),
        unequal_u_gram=float(values[~equal_u].sum()),ad_relative_error=error,
        minimum_distinct_flow_difference=float(min(norm)))


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
start=perf_counter();OUTPUT.mkdir(parents=True,exist_ok=True)
inputs=dict(position=POSITION.tolist(),mass=MASS,charge=CHARGE,strength=STRENGTH,D=D,
    h='mu*x',f='exp(-E-.2mu)',observable='frozen local Gram Q_h; equilibrium entropy production is zero',
    velocity_domain=[-UMAX,UMAX],moment_domain=[0.,MUMAX],mu_order=NMU,
    parallel_orders=list(ORDERS),positive_amplitudes=AMPLITUDES.tolist())
metadata=run_metadata(inputs,model='continuous local frozen-mobility Gram; mirror near-uniform quadrature audit',
    boundary='local volume density; finite positive velocity quadrature; no evolution boundaries',units='normalized')
import sato_morrison.geometry as geometry
import sato_morrison.reference as reference
dependencies=[Path(__file__).resolve(),Path(geometry.__file__),Path(reference.__file__)]
metadata['experiment_dependency_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
rows=[]
for nu in ORDERS:
    with progress(f'near-uniform mirror: {nu} parallel nodes'):
        row=dict(nu=nu,**analytic_forms(nu));direct=[]
        for amplitude in AMPLITUDES:
            direct.append(dict(amplitude=float(amplitude),**direct_form(nu,float(amplitude))))
        row['direct']=direct
        row['smallest_amplitude_limit_error']=abs(direct[0]['gram']/row['positive_amplitude_limit']-1)
        rows.append(row)
metadata['rows']=rows;metadata['quadrature_wall_seconds']=perf_counter()-start
metadata['checks']={
    'maximum_ad_relative_error':max(d['ad_relative_error'] for r in rows for d in r['direct']),
    'maximum_smallest_amplitude_limit_error':max(r['smallest_amplitude_limit_error'] for r in rows),
    'all_gap_bounds_hold':all(0<r['gap']<=r['gap_bound'] for r in rows),
    'deficit_decreases_under_refinement':all(b['relative_gap']<a['relative_gap'] for a,b in zip(rows,rows[1:]))}
passed=(metadata['checks']['maximum_ad_relative_error']<1e-12 and
    metadata['checks']['maximum_smallest_amplitude_limit_error']<2e-6 and
    metadata['checks']['all_gap_bounds_hold'] and metadata['checks']['deficit_decreases_under_refinement'])
metadata['status']='passed' if passed else 'unresolved'
(OUTPUT/'summary.json').write_text(json.dumps(metadata,indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
colors=['#087e8b','#c45b28','#614e9e','#315f91']
for row,color in zip(rows,colors):
    axes[0].semilogx(AMPLITUDES,[d['gram']/row['uniform'] for d in row['direct']],
        'o-',color=color,label=f"nᵤ={row['nu']}")
    axes[0].axhline(row['positive_amplitude_limit']/row['uniform'],color=color,linestyle=':',linewidth=1)
axes[0].axhline(1.,color='.25',linestyle='--',label='literal uniform prescription')
axes[0].set(xlabel='Positive mirror amplitude a',ylabel='Frozen Gram Qₕ / Quniform',
    title='A. Finite quadrature retains equal-u pairs')
axes[0].legend(fontsize=8,ncol=2,loc='center',bbox_to_anchor=(.48,.33))
axes[0].text(.03,.53,'Dotted lines: analytic a → 0⁺ limits',
    transform=axes[0].transAxes,fontsize=8,color='.35')
axes[1].loglog(ORDERS,[r['relative_gap'] for r in rows],'o-',color=colors[0],label='analytic a → 0⁺ deficit')
axes[1].loglog(ORDERS,[1-r['direct'][0]['gram']/r['uniform'] for r in rows],'x',color=colors[1],
    markersize=8,label=f'direct a={AMPLITUDES[0]:.0e}')
axes[1].loglog(ORDERS,[r['relative_bound'] for r in rows],'--',color='.4',label='proved max(wᵤ) bound')
axes[1].set(xlabel='Parallel-velocity quadrature order nᵤ',ylabel='(Quniform − Qₕ) / Quniform',
    title='B. Refinement removes the finite-node deficit')
axes[1].legend(fontsize=8)
for axis in axes:axis.grid(alpha=.2)
fig.savefig(OUTPUT/'near_uniform.png',dpi=180);plt.close(fig)
metadata['wall_seconds']=perf_counter()-start
(OUTPUT/'summary.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(f'Saved {OUTPUT}; finite-quadrature limit only, no physical discontinuity.',flush=True)
if not passed:raise RuntimeError('Near-uniform analytic/direct checks unresolved; evidence saved')
