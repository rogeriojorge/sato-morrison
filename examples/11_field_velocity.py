"""Independent velocity and tail quadrature checks for collision-only field boxes."""
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from scipy.special import erf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sato_morrison.collisions import cartesian_grid, mobility_action, trapezoid_weights
from sato_morrison.geometry import Field, field_vector, validate_geometry
from sato_morrison.reference import progress, run_metadata

# Same fields, box and initial functional form as 08_fields. Velocity domains
# include mu=0 and resolve thermal tails; no time-evolution claim is made here.
FIELDS = [Field('mirror', amplitude=.15), Field('dipole'),
          Field('nonaxisymmetric', amplitude=.03)]
BOUNDS = [(.8,1.2),(-.2,.2),(.1,.5)]
SPATIAL_ORDER = 3
U_MAX, MU_MAX = 4., 20.
NU, NMU = 25, 21
U_ORDERS, MU_ORDERS = [17,21,25], [13,17,21]
D, CHUNK = .1, 1024
SHOW_FIGURES = False
OUTPUT = Path(__file__).resolve().parents[1] / 'results' / 'field_velocity'
OUTPUT.mkdir(parents=True,exist_ok=True)
print(f'sm_local_nonlinear initial entropy production; fixed nonuniform collision-only boxes; '
      f'{SPATIAL_ORDER} nodes per spatial axis; u∈[-{U_MAX},{U_MAX}], mu∈[0,{MU_MAX}]; '
      f'positive Gauss-Legendre velocity quadrature and separate trapezoid controls; '
      f'natural no-flux weak boundary; normalized fixed D={D}; output={OUTPUT}',flush=True)


def nodes_weights(count, lower, upper, quadrature):
    if quadrature=='gauss':
        nodes,weights=np.polynomial.legendre.leggauss(count)
        return lower+(nodes+1)*(upper-lower)/2,weights*(upper-lower)/2
    if quadrature=='trapezoid':
        nodes=np.linspace(lower,upper,count)
        return nodes,trapezoid_weights(nodes)
    raise ValueError(f'unsupported quadrature {quadrature}')


def gaussian_integrals(alpha, umax):
    """Independent finite-domain ∫exp(-u²/2+alpha*u) and ∫u² exp(...)."""
    left,right=-umax,umax
    low,high=left-alpha,right-alpha
    zeroth=np.sqrt(np.pi/2)*(erf(high/np.sqrt(2))-erf(low/np.sqrt(2)))
    first=np.exp(-low**2/2)-np.exp(-high**2/2)
    second=zeroth+low*np.exp(-low**2/2)-high*np.exp(-high**2/2)
    factor=np.exp(alpha**2/2)
    return factor*zeroth,factor*(second+2*alpha*first+alpha**2*zeroth)


def evaluate(field, nu=NU, nmu=NMU, umax=U_MAX, mumax=MU_MAX, quadrature='gauss'):
    spatial_axes=[np.linspace(a,b,SPATIAL_ORDER) for a,b in BOUNDS]
    u,wu=nodes_weights(nu,-umax,umax,quadrature)
    mu,wmu=nodes_weights(nmu,0,mumax,quadrature)
    setup_start=perf_counter()
    grid=cartesian_grid(*spatial_axes,u,mu,field)
    positions=np.stack(np.meshgrid(*spatial_axes,indexing='ij'),axis=-1).reshape(-1,3)
    b=np.linalg.norm(np.asarray(jax.vmap(lambda point:field_vector(point,field))(
        jnp.asarray(positions))),axis=1)
    spatial_weights=np.einsum('i,j,k->ijk',*[trapezoid_weights(axis) for axis in spatial_axes]).ravel()
    nv=nu*nmu
    velocity_weights=b[:,None]*np.outer(wu,wmu).reshape(1,nv)
    weights=(spatial_weights[:,None]*velocity_weights).ravel()
    left,right=np.asarray(grid.left),np.asarray(grid.right)
    spatial_index=left//nv
    if not np.all(spatial_index==right//nv):
        raise RuntimeError('local pair list unexpectedly crosses spatial nodes')
    pair_weights=(spatial_weights[spatial_index]*velocity_weights[spatial_index,left%nv]*
                  velocity_weights[spatial_index,right%nv])
    # Specify positive quadrature consistently in BOTH the population measure
    # wX*B*wu*wmu and local pair measure wX*(B*wu*wmu)*(B*wu'*wmu').
    # Energy directions and derivative matrices are left unchanged. For the
    # quadratic u energy and linear-u log perturbation, the three-node derivative
    # is exact on both Gauss and equidistant nodes, including boundary stencils.
    grid=replace(grid,weights=jnp.asarray(weights),pair_weights=jnp.asarray(pair_weights))
    x,y,z,uu,mm=np.meshgrid(*spatial_axes,u,mu,indexing='ij')
    log_eq=-grid.energy-.2*jnp.asarray(mm.ravel())
    log_f=log_eq+.1*jnp.sin(jnp.asarray(x.ravel()))*jnp.asarray(uu.ravel())+.04*jnp.asarray(y.ravel()*mm.ravel())
    f=jnp.exp(log_f);equilibrium=jnp.exp(log_eq)
    setup_s=perf_counter()-setup_start
    apply=jax.jit(lambda distribution,test:mobility_action(grid,distribution,test,
                                  collision_strength=D,chunk_size=CHUNK))
    compile_start=perf_counter();flux=apply(f,log_f);flux.block_until_ready()
    compile_s=perf_counter()-compile_start
    repetitions=[]
    for _ in range(3):
        start=perf_counter();flux=apply(f,log_f);flux.block_until_ready()
        repetitions.append(perf_counter()-start)
    production=float(jnp.vdot(log_f,flux))
    stationarity=float(jnp.max(jnp.abs(apply(equilibrium,log_eq)/grid.weights)))
    number=float(jnp.sum(grid.weights*f));energy=float(jnp.vdot(grid.weights*f,grid.energy))
    number_rate=abs(float(jnp.sum(flux)))/(D*number)
    energy_rate=abs(float(jnp.vdot(grid.energy,flux)))/(D*energy)
    marginal=jnp.zeros(nmu).at[grid.mu_index].add(flux)
    marginal_rate=float(jnp.max(jnp.abs(marginal)))/(D*number)
    # Independent analytic velocity integration separates quadrature error from
    # the kernel's double-integral error and from omitted thermal populations.
    alpha=.1*np.sin(positions[:,0]);kappa=b+.2-.04*positions[:,1]
    i0,i2=gaussian_integrals(alpha,umax)
    j0=-np.expm1(-kappa*mumax)/kappa
    j1=(1-(1+kappa*mumax)*np.exp(-kappa*mumax))/kappa**2
    exact_number=float(np.sum(spatial_weights*b*i0*j0))
    exact_energy=float(np.sum(spatial_weights*b*(.5*i2*j0+b*i0*j1)))
    whole_i0=np.sqrt(2*np.pi)*np.exp(alpha**2/2)
    whole_number=float(np.sum(spatial_weights*b*whole_i0/kappa))
    whole_energy=float(np.sum(spatial_weights*b*(.5*whole_i0*(1+alpha**2)/kappa+b*whole_i0/kappa**2)))
    if min(number,energy,float(f.min()))<=0 or not np.isfinite(production):
        raise RuntimeError('nonfinite or nonpositive quadrature state')
    if stationarity>1e-10 or production < -1e-14 or max(number_rate,energy_rate,marginal_rate)>1e-9:
        raise RuntimeError('weak equilibrium, entropy or collision invariant check failed')
    row={'field':field.kind,'quadrature':quadrature,'nu':nu,'nmu':nmu,'umax':umax,'mumax':mumax,
         'production':production,'equilibrium_residual':stationarity,
         'number_rate_residual':number_rate,'energy_rate_residual':energy_rate,
         'marginal_rate_residual':marginal_rate,'min_f':float(f.min()),
         'number_quadrature_relative_error':abs(number/exact_number-1),
         'energy_quadrature_relative_error':abs(energy/exact_energy-1),
         'analytic_omitted_number_fraction':1-exact_number/whole_number,
         'analytic_omitted_energy_fraction':1-exact_energy/whole_energy,
         'min_sampled_B':validate_geometry(positions,field),'pair_count':len(left),
         'stored_pair_bytes':sum(array.nbytes for array in (grid.left,grid.right,grid.pair_weights,grid.kernel_directions)),
         'compile_and_first_s':compile_s,'setup_s':setup_s,
         'warm_median_s':float(np.median(repetitions)),
         'warm_min_s':min(repetitions),'warm_max_s':max(repetitions),'status':'passed'}
    print(f'  {field.kind}, {quadrature}, {nu}x{nmu}, U={umax:g}, M={mumax:g}: '
          f'production={production:.9e}; moment quadrature error={max(row["number_quadrature_relative_error"],row["energy_quadrature_relative_error"]):.2e}',flush=True)
    return row


def last_change(selected):
    previous,last=selected[-2:]
    error=abs(last['production']/previous['production']-1)
    return {'relative_production_change':error,'status':'passed' if error<.01 else 'unresolved',
            'previous_parameters':{k:previous[k] for k in ('nu','nmu','umax','mumax','quadrature')},
            'last_parameters':{k:last[k] for k in ('nu','nmu','umax','mumax','quadrature')}}

rows=[];checks=[]
with progress('Compile and evaluate field-box velocity and tail quadratures'):
    for field in FIELDS:
        cache={}
        def get(**parameters):
            values={'nu':NU,'nmu':NMU,'umax':U_MAX,'mumax':MU_MAX,'quadrature':'gauss',**parameters}
            key=tuple(values.values())
            if key not in cache:
                cache[key]=evaluate(field,**values)
                rows.append(cache[key])
            return cache[key]
        u_scan=[get(nu=count) for count in U_ORDERS]
        mu_scan=[get(nmu=count) for count in MU_ORDERS]
        u_tail=[get(),get(umax=5.)]
        mu_tail=[get(),get(mumax=24.)]
        # Verify the widened domains separately: changing a tail at fixed
        # Gauss order also changes quadrature node density. These checks bound
        # that source of ambiguity rather than calling it a fixed-spacing scan.
        u_tail_quadrature=[get(umax=5.),get(umax=5.,nu=29)]
        mu_tail_quadrature=[get(mumax=24.),get(mumax=24.,nmu=25)]
        trapezoid_scan=[get(nu=count,nmu=count,quadrature='trapezoid') for count in (9,17,25)]
        check={'field':field.kind,'u_resolution':last_change(u_scan),'mu_resolution':last_change(mu_scan),
               'u_tail':last_change(u_tail),'mu_tail':last_change(mu_tail),
               'wider_u_quadrature':last_change(u_tail_quadrature),
               'wider_mu_quadrature':last_change(mu_tail_quadrature),
               'original_trapezoid_joint_resolution':last_change(trapezoid_scan)}
        check['gauss_velocity_and_tail_status']='passed' if all(check[key]['status']=='passed' for key in
            ('u_resolution','mu_resolution','u_tail','mu_tail','wider_u_quadrature','wider_mu_quadrature')) else 'unresolved'
        checks.append(check)
        print(f'  {field.kind}: {check}',flush=True)
        jax.clear_caches()

metadata=run_metadata({'fields':[field.__dict__ for field in FIELDS],'spatial_order':SPATIAL_ORDER,
    'bounds':BOUNDS,'u_orders':U_ORDERS,'mu_orders':MU_ORDERS,'base_nu':NU,'base_nmu':NMU,
    'u_domain':[-U_MAX,U_MAX],'mu_domain':[0,MU_MAX],'u_tail':5.,'mu_tail':24.,
    'collision_strength':D,'pair_chunk':CHUNK,'seed':None,
    'initial':'exp(-E-.2mu+.1sin(x)u+.04y*mu)',
    'velocity_quadratures':'Gauss-Legendre on finite intervals; separate equidistant trapezoid controls',
    'spatial_quadrature':'fixed3-node-per-axis trapezoid; natural collision flux',
    'pair_measure':'wX*(B*wu*wmu)*(B*wu_prime*wmu_prime)'},
    model='sm_local_nonlinear discrete-energy projector',boundary='natural no-flux collision-only Cartesian boxes')
metadata['limitations']='Initial entropy production only; no box time-evolution convergence. Spatial mesh fixed at3^3, so no joint spatial/velocity continuum claim. Gauss tail checks change node density and separately verify quadrature on each wider domain. Original trapezoid controls retain their own status. No dimensional coefficient calibration or Hamiltonian confinement claim.'
metadata['derivative_consistency']='Quadratic u energy and linear-u log perturbation have exact three-node derivative on both node families; mu gradients are Casimir-null. Positive quadrature replaces state and pair weights together; kernels are unchanged.'
(OUTPUT/'summary.json').write_text(json.dumps({'metadata':metadata,'rows':rows,'checks':checks,
    'gauss_velocity_and_tail_status':'passed' if all(row['gauss_velocity_and_tail_status']=='passed' for row in checks) else 'unresolved'},indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(9,3.7))
for field in FIELDS:
    selected=[row for row in rows if row['field']==field.kind and row['quadrature']=='gauss' and row['nmu']==NMU and row['umax']==U_MAX and row['mumax']==MU_MAX]
    selected=sorted(selected,key=lambda row:row['nu'])
    axes[0].plot([row['nu'] for row in selected],[row['production'] for row in selected],'o-',label=field.kind)
    check=next(row for row in checks if row['field']==field.kind)
    keys=['u_resolution','mu_resolution','u_tail','mu_tail']
    axes[1].plot(range(4),[max(check[key]['relative_production_change'],1e-16) for key in keys],'o-',label=field.kind)
axes[0].set(xlabel='Parallel-velocity Gauss order',ylabel='Initial entropy production')
axes[1].set_xticks(range(4),['u nodes','mu nodes','u tail','mu tail']);axes[1].set_yscale('log')
axes[1].axhline(.01,color='black',linestyle='--');axes[1].set(ylabel='Last independent relative change')
axes[0].legend();fig.suptitle('Nonuniform collision-only boxes: velocity and tail checks');fig.tight_layout()
fig.savefig(OUTPUT/'production.png',dpi=180)
if SHOW_FIGURES:
    plt.show()
plt.close(fig)
print(f'Saved {OUTPUT/"summary.json"}; inspect Gauss and original trapezoid statuses separately.',flush=True)
