"""A sufficient mixed local collision-null candidate on a dipole angle branch.

This checks local action cancellation, a spatially separated pair, and finite
spatial differentiation. It does not evolve a distribution or classify nulls.
"""
from pathlib import Path
import hashlib,json
from time import perf_counter
print('Check a dipole local collision moment: geometry, pairs, and spatial refinement.',flush=True)
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sato_morrison.collisions import cartesian_grid
from sato_morrison.geometry import Field,common_chart_action,energy_mu,poisson_mu
from sato_morrison.reference import gauss_interval,run_metadata

MASS,CHARGE,STRENGTH=1.7,.7,1.3
FIELD=Field('dipole',strength=STRENGTH)
BOUNDS=[(.8,1.2),(-.2,.2),(.1,.5)]
SPATIAL_ORDERS=[3,5,7,9]
U_DOMAIN,MU_DOMAIN=(-1.5,1.5),(.1,1.1)
POSITION=np.array([1.,.1,.3])
VELOCITIES=np.array([[-.8,.15],[.2,.6],[1.1,1.]])
SEPARATIONS=np.r_[0.,np.geomspace(1e-4,.2,31)]
OUTPUT=Path(__file__).resolve().parents[1]/'results'/'local_mixed_null'
OUTPUT.mkdir(parents=True,exist_ok=True)


def dipole_formula(position,strength):
    """Independent magnitude and gradient, without geometry AD helpers."""
    x,y,z=np.asarray(position);r2=x*x+y*y+z*z;d=x*x+y*y+4*z*z
    vector=strength*np.array([3*x*z,3*y*z,2*z*z-x*x-y*y])/r2**2.5
    magnitude=strength*np.sqrt(d)/r2**2
    grad=magnitude*(np.array([x,y,4*z])/d-4*np.array([x,y,z])/r2)
    return magnitude,vector/magnitude,grad


def phi_gradient(position,charge):
    x,y,z=np.asarray(position);r2=x*x+y*y;d=r2+2*z*z;theta=np.arctan2(y,x)
    coefficient=-z*(3*r2+8*z*z)/d
    derivative=np.array([4*x*z**3/d**2,4*y*z**3/d**2,-3-6*z*z/d+8*z**4/d**2])
    return charge*(coefficient*np.array([-y/r2,x/r2,0.])+theta*derivative)


def candidate(state,mass,charge,strength,*,mixed=True):
    x,y,z,u,_=state;r2=x*x+y*y
    magnitude=strength*jnp.sqrt(r2+4*z*z)/(r2+z*z)**2
    phi=-charge*jnp.arctan2(y,x)*z*(3*r2+8*z*z)/(r2+2*z*z)
    return mass*u/magnitude+(phi if mixed else 0.)


def action(state,field,mass,charge,*,mixed=True):
    gradient=jax.grad(lambda z:candidate(z,mass,charge,field.strength,mixed=mixed))(state)
    return common_chart_action(state,gradient,field,mass,charge)


def pair_evidence(left,right,field,mass,charge,*,power=1):
    left,right=map(jnp.asarray,(left,right))
    def observable_action(z):
        if power==1:return action(z,field,mass,charge)
        gradient=jax.grad(lambda w:candidate(w,mass,charge,field.strength)**power)(z)
        return common_chart_action(z,gradient,field,mass,charge)
    ah=observable_action(left)-observable_action(right)
    def energy_action(z):
        return common_chart_action(z,jax.grad(lambda w:energy_mu(w,field,mass,charge))(z),field,mass,charge)
    xi=energy_action(left)-energy_action(right);norm=float(jnp.dot(xi,xi))
    if not np.isfinite(norm) or norm<=0:raise RuntimeError('pair needs a nonzero energy direction')
    projected=ah-xi*jnp.dot(xi,ah)/jnp.dot(xi,xi)
    return {'action_difference_norm':float(jnp.linalg.norm(ah)),
        'projected_spatial_quadratic':float(jnp.dot(projected[:3],projected[:3])),
        'energy_direction_squared':norm}


inputs={'field':FIELD.__dict__,'mass':MASS,'charge':CHARGE,'bounds':BOUNDS,
    'candidate':'m*u/B-q*theta*z*(3R^2+8z^2)/(R^2+2z^2), theta=atan2(y,x), x>0',
    'position':POSITION.tolist(),'velocity_samples':VELOCITIES.tolist(),'charges_checked':[CHARGE,-1.3],
    'unit_control':{'mass':1.,'charge':1.,'strength':1.,'left':[1.,0.,.3,.4,.2],
        'separated_right':[1.1,.05,.35,-.6,.7],'same_position_right':[1.,0.,.3,-.6,.7],
        'squared_observable':'h^2'},
    'spatial_orders':SPATIAL_ORDERS,'u_order':3,'mu_order':2,
    'u_domain':U_DOMAIN,'mu_domain':MU_DOMAIN,'spatial_derivative':'global Lagrange polynomial',
    'separations_x':SEPARATIONS.tolist(),'pair_left_velocity':[.4,.3],
    'pair_right_velocity':[-.6,.8],'local_action_target':1e-12,
    'spatial_residual_reduction_target':.01,'seed':None}
metadata=run_metadata(inputs,model='sufficient local same-position weak collision null test',
    boundary='natural collision-only box on smooth x>0 angle branch',units='normalized')
metadata['example_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
metadata['limitations']=('A sufficient local collision moment; the squared-moment check is a concrete dipole counterexample, not a universal statement about all functions of h or fields; no ideal-flow conservation, full nullspace classification, '
    'stationary density, marginal, global periodic angle extension, or novelty claim. Separated-pair '
    'quadratics diagnose one interaction, not a calibrated finite-range rate. Spatial refinement only '
    'measures sampled tensor differentiation and its unmodified discrete-energy projector.')
start=perf_counter();identity=[]
for charge in (CHARGE,-1.3):
    for position in (POSITION,np.array([.85,-.15,.4]),np.array([1.15,.18,.15])):
        B,b,g=dipole_formula(position,STRENGTH);gradphi=phi_gradient(position,charge)
        expected=np.r_[b/B+np.cross(b,gradphi)/(charge*B),-np.dot(b,gradphi)/MASS,0.]
        states=jnp.asarray([np.r_[position,v] for v in VELOCITIES])
        actions=np.asarray(jax.vmap(lambda z:action(z,FIELD,MASS,charge))(states))
        gradient=np.asarray(jax.vmap(jax.grad(lambda z:candidate(z,MASS,charge,STRENGTH)))(states))
        gradients=np.stack([np.r_[-MASS*v[0]*g/B**2+gradphi,MASS/B,0.] for v in VELOCITIES])
        identity.append({'charge':charge,'position':position.tolist(),
            'gradient_error':float(np.max(np.abs(gradient-gradients))),
            'pde_error':abs(float(np.cross(b,g)@gradphi-charge*np.dot(b,g))),
            'action_error':float(np.max(np.abs(actions-expected))),
            'local_pair_action_norm':float(np.max(np.linalg.norm(actions-actions[0],axis=1)))})
left=np.r_[POSITION,.4,.3]
pairs=[{'separation':float(d),**pair_evidence(left,np.r_[POSITION+[d,0,0],-.6,.8],FIELD,MASS,CHARGE)} for d in SEPARATIONS]
control_field=Field('dipole')
control_left=jnp.array([1.,0.,.3,.4,.2]);control_right=jnp.array([1.1,.05,.35,-.6,.7])
control=pair_evidence(control_left,control_right,control_field,1.,1.)
local_right=control_right.at[:3].set(control_left[:3])
control['same_position_candidate']=pair_evidence(control_left,local_right,control_field,1.,1.)
control['same_position_squared_candidate']=pair_evidence(control_left,local_right,control_field,1.,1.,power=2)
control['ideal_derivative']=float(jax.grad(lambda z:candidate(z,1.,1.,1.))(control_right)@
    poisson_mu(control_right,control_field)@jax.grad(lambda z:energy_mu(z,control_field))(control_right))
if abs(control['projected_spatial_quadratic']-.3078910312969079)>1e-12:
    raise RuntimeError('independent reference pair disagrees')
control['analytic_ideal_derivative']=float(control_right[3])*float(dipole_formula(np.asarray(control_right[:3]),1.)[1]@phi_gradient(np.asarray(control_right[:3]),1.))
if (control['same_position_candidate']['projected_spatial_quadratic']>1e-24
    or control['same_position_squared_candidate']['projected_spatial_quadratic']<1e-4
    or abs(control['ideal_derivative']-control['analytic_ideal_derivative'])>1e-12):
    raise RuntimeError('squared moment or ideal derivative control failed')
refinement=[]
for order in SPATIAL_ORDERS:
    print(f'Starting spatial derivative grid: {order} nodes per spatial axis.',flush=True)
    spatial=[gauss_interval(order,a,b) for a,b in BOUNDS]
    u,wu=gauss_interval(3,*U_DOMAIN);mu,wm=gauss_interval(2,*MU_DOMAIN)
    grid=cartesian_grid(*[p[0] for p in spatial],u,mu,FIELD,mass=MASS,charge=CHARGE,
        spatial_weights=[p[1] for p in spatial],velocity_weights=(wu,wm),spatial_discretization='polynomial')
    xx,yy,zz,uu,mm=np.meshgrid(*[p[0] for p in spatial],u,mu,indexing='ij')
    states=jnp.asarray(np.stack((xx,yy,zz,uu,mm),axis=-1).reshape(-1,5))
    values=jax.vmap(lambda z:candidate(z,MASS,CHARGE,STRENGTH))(states)
    discrete=np.asarray(grid.action(values));leftids,rightids=map(np.asarray,(grid.left,grid.right))
    difference=discrete[leftids]-discrete[rightids];direction=np.asarray(grid.kernel_directions)
    projected=difference-direction*np.sum(direction*difference,axis=1)[:,None]
    f=np.exp(-np.asarray(grid.energy));weight=np.asarray(grid.pair_weights)*f[leftids]*f[rightids]
    total=weight.sum()
    refinement.append({'spatial_order':order,'nodes':grid.size,'unordered_pairs':grid.pair_count,
        'pair_weight_sum':float(total),'action_pair_rms':float(np.sqrt(weight@np.sum(difference**2,axis=1)/total)),
        'projected_spatial_pair_rms':float(np.sqrt(weight@np.sum(projected[:,:3]**2,axis=1)/total)),
        'maximum_pair_action_norm':float(np.linalg.norm(difference,axis=1).max())})
    print('Spatial differentiation:',refinement[-1],flush=True)
maximum_identity=max(row[key] for row in identity for key in ('gradient_error','pde_error','action_error','local_pair_action_norm'))
reduction=refinement[-1]['projected_spatial_pair_rms']/refinement[0]['projected_spatial_pair_rms']
passed=(maximum_identity<1e-12 and pairs[0]['projected_spatial_quadratic']<1e-24
    and pairs[-1]['projected_spatial_quadratic']>1e-4 and reduction<.01)
evidence={'metadata':metadata,'identity_checks':identity,'separated_pairs':pairs,
    'independent_reference_pair':control,'spatial_refinement':refinement,
    'maximum_identity_error':maximum_identity,'refinement_reduction':reduction,
    'angular_obstruction':{'R':float(np.linalg.norm(POSITION[:2])),'z':float(POSITION[2]),
        'phi_increment':float(-2*np.pi*CHARGE*POSITION[2]*(3*np.dot(POSITION[:2],POSITION[:2])+8*POSITION[2]**2)/(np.dot(POSITION[:2],POSITION[:2])+2*POSITION[2]**2)),
        'scope':'Nonzero integral of dphi/dtheta on this circular continuation rules out periodic phi there'},
    'checks_wall_s':perf_counter()-start,'cost_scope':'Exact identity and pair checks plus grid refinement including compilation; excludes imports, metadata collection, figure-only action sampling and rendering','status':'passed_sampled_checks' if passed else 'unresolved'}
text=json.dumps(evidence,indent=2,allow_nan=False)+'\n';(OUTPUT/'summary.json').write_text(text)

uplot=np.linspace(-1.5,1.5,61);states=jnp.asarray([np.r_[POSITION,v,.6] for v in uplot])
raw=np.asarray(jax.vmap(lambda z:action(z,FIELD,MASS,CHARGE,mixed=False))(states))
mixed=np.asarray(jax.vmap(lambda z:action(z,FIELD,MASS,CHARGE))(states))
theta=np.linspace(-np.pi,np.pi,201);R2=POSITION[0]**2+POSITION[1]**2;z=POSITION[2]
angular_coefficient=-CHARGE*z*(3*R2+8*z*z)/(R2+2*z*z)
fig,axes=plt.subplots(2,2,figsize=(10,7.5))
axes[0,0].plot(uplot,raw[:,3],color='#b75b28',label=r'$h_0=m\,u/B$')
axes[0,0].plot(uplot,mixed[:,3],color='#2166ac',label=r'$h=m\,u/B+\phi$')
axes[0,0].set(xlabel=r'Parallel velocity $u$',ylabel=r'Common action component $A_u$',title='Local cancellation of velocity dependence')
axes[0,0].legend(frameon=False)
axes[0,0].text(.04,.05,f'Full-action local pair error < {maximum_identity:.1e}\n'+r'$A_\eta h=0$; both charge signs checked',transform=axes[0,0].transAxes,fontsize=9)
axes[0,1].loglog([row['separation'] for row in pairs[1:]],
    [row['projected_spatial_quadratic'] for row in pairs[1:]],color='#b75b28',label='Spatially separated pair')
axes[0,1].set(xlabel=r'Spatial separation $\delta x$',ylabel=r'$\|I_xP_\xi\Delta A h\|^2$',title='A separated pair breaks the cancellation')
axes[0,1].text(.04,.07,f"Same-position value: {pairs[0]['projected_spatial_quadratic']:.1e}",transform=axes[0,1].transAxes,fontsize=9,bbox={'facecolor':'white','edgecolor':'none','alpha':.9})
orders=[row['spatial_order'] for row in refinement]
for name,label,color in [('action_pair_rms',r'Full $\Delta Ah$','#2166ac'),('projected_spatial_pair_rms',r'$I_xP_\xi\Delta Ah$','#27823b')]:
    axes[1,0].semilogy(orders,[row[name] for row in refinement],'o-',color=color,label=label)
axes[1,0].set(xlabel='Nodes per spatial axis',ylabel='Positive pair-weighted RMS',title='Unmodified tensor derivative: sampled refinement')
axes[1,0].legend(frameon=False)
axes[1,1].plot(theta,angular_coefficient*theta,color='#555555')
axes[1,1].axvspan(-np.pi/2,np.pi/2,color='#2166ac',alpha=.12,label=r'Smooth $x>0$ branch')
axes[1,1].set(xlabel=r'Formal unwrapped angle $\theta$',ylabel=r'$\phi=q\theta R s/t_\theta$',title='This dipole branch cannot be periodic')
axes[1,1].legend(frameon=False)
axes[1,1].text(.03,.05,f'Angular increment: {2*np.pi*angular_coefficient:.3f}\nNot an ideal-flow conservation claim',transform=axes[1,1].transAxes,fontsize=9)
for ax in axes.ravel():ax.grid(alpha=.2)
fig.suptitle('Dipole mixed local collision null: cancellation, separation, and scope',fontsize=13)
fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUTPUT/'local_mixed_null.png',dpi=180);plt.close(fig)
np.savez_compressed(OUTPUT/'plot_data.npz',u=uplot,raw_action=raw,mixed_action=mixed,
    theta=theta,phi=angular_coefficient*theta)
print(f'Saved {OUTPUT}; status={evidence["status"]}; local error={maximum_identity:.3e}',flush=True)
if not passed:raise RuntimeError('sampled local-null/refinement checks unresolved; evidence retained')
