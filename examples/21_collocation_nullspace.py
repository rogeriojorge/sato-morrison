"""Raw finite-grid collision nulls and a tensor-polynomial angular alias."""
from pathlib import Path
import hashlib,json
from time import perf_counter
print('Audit raw collision factors and a finite tensor-grid angular alias.',flush=True)
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sato_morrison.collisions import cartesian_grid,mobility_action
from sato_morrison.geometry import Field
from sato_morrison.reference import gauss_interval,run_metadata

BOUNDS=((.8,1.2),(-.2,.2),(.1,.5))
FIELDS=(Field('mirror',amplitude=.15),Field('dipole'),Field('nonaxisymmetric',amplitude=.03))
THRESHOLDS=(1e-10,1e-12,1e-13,1e-14)
OUTPUT=Path(__file__).resolve().parents[1]/'results'/'collocation_nullspace'
OUTPUT.mkdir(parents=True,exist_ok=True)


def alias_polynomial(x,y,c=1.,k=.024):
    return (x*x-3*c*x)*y*y-3*c*c*x*x+c*(2*c*c+k)*x


def angular_residue(x,y,c=1.,k=.024):
    return 2*y*((x-c)**3-k*(x-c))-(2*x-3*c)*(y**3-k*y)


def angular_integral(a,c=1.):
    return 128*a**10/525+16*c*c*a**8/175


def polynomial_derivative(nodes):
    # Independently differentiate the shifted cardinal polynomials.
    derivative=np.zeros((len(nodes),len(nodes)))
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            basis=np.polynomial.Polynomial([1.])
            for k in range(len(nodes)):
                if k!=j:basis*=np.polynomial.Polynomial([nodes[i]-nodes[k],1.])/(nodes[j]-nodes[k])
            derivative[i,j]=basis.deriv()(0.)
    return derivative


def pair_factors(field,spatial,u_rule,mu_rule,collision_strength=.1):
    axes=[p[0] for p in spatial];u,wu=u_rule;mu,wm=mu_rule
    options=dict(spatial_weights=[p[1] for p in spatial],velocity_weights=(wu,wm),spatial_discretization='polynomial')
    grid=cartesian_grid(*axes,u,mu,field,**options)
    shape=grid.shape;N=grid.size;NX=int(np.prod(shape[:3]));NV=int(np.prod(shape[3:]))
    derivatives=[]
    for axis,nodes in enumerate(axes+[u]):
        matrix=np.ones((1,1))
        for d,size in enumerate(shape):matrix=np.kron(matrix,polynomial_derivative(nodes) if d==axis else np.eye(size))
        derivatives.append(matrix)
    C=np.asarray(grid.coefficients).reshape(N,5,4)
    A=np.einsum('iad,dij->iaj',C,np.stack(derivatives))
    energy=np.asarray(grid.energy);flow=A@energy
    xx,yy,zz,uu,mm=np.meshgrid(*axes,u,mu,indexing='ij')
    xyz=np.stack((xx[:,:,:,0,0],yy[:,:,:,0,0],zz[:,:,:,0,0]),axis=-1).reshape(NX,3)
    x,y,z=xyz.T
    if field.kind=='mirror':
        a=field.amplitude;Bvec=np.stack((-a*z*x,-a*z*y,field.strength+a*(z*z-(x*x+y*y)/2)),axis=-1)
    else:
        r2=x*x+y*y+z*z;Bvec=field.strength*np.stack((3*x*z,3*y*z,2*z*z-x*x-y*y),axis=-1)/r2[:,None]**2.5
        if field.kind=='nonaxisymmetric':Bvec+=field.amplitude*field.strength/field.length*np.stack((y,x,np.zeros_like(x)),axis=-1)
    B=np.linalg.norm(Bvec,axis=-1)
    wx=np.einsum('i,j,k->ijk',*[p[1] for p in spatial]).ravel();wv=np.outer(wu,wm).ravel()
    vi,vj=np.triu_indices(NV,k=1)
    left=(np.arange(NX)[:,None]*NV+vi).ravel();right=(np.arange(NX)[:,None]*NV+vj).ravel()
    pair_weight=(wx[:,None]*B[:,None]**2*wv[vi]*wv[vj]).ravel()
    xi=flow[left]-flow[right];xi2=np.sum(xi*xi,axis=1)
    if np.any(xi2<=0):raise RuntimeError('distinct-state energy direction is zero')
    P=np.eye(5)[None]-xi[:,:,None]*xi[:,None,:]/xi2[:,None,None]
    F=np.einsum('pab,pbj->paj',P,A[left]-A[right])[:,:3,:].reshape(-1,N)
    f=np.exp(-energy-.2*mm.ravel()+.1*np.sin(np.pi*yy.ravel()/.4)*uu.ravel())
    L=np.repeat(np.sqrt(collision_strength*pair_weight*f[left]*f[right]),3)[:,None]*F
    mass=np.asarray(grid.weights)*f
    r2=xx.ravel()**2+yy.ravel()**2;zall=zz.ravel();alias=alias_polynomial(xx,yy).ravel()
    known=np.column_stack([(np.asarray(grid.mu_index)==k).astype(float) for k in range(len(mu))]+[energy]+
        [r2**p*zall**q for p in (0,1) for q in range(3)])
    extra=np.column_stack([alias*zall**q for q in range(3)])
    return {'grid':grid,'F':F,'L':L,'entropy_scaled':L/np.sqrt(mass)[None,:],'f':f,'known':known,'extra':extra,
        'pair_weight_error':float(np.max(np.abs(pair_weight/np.asarray(grid.pair_weights)-1))),
        'minimum_pair_flow_squared':float(xi2.min())}


inputs={'fields':[f.__dict__ for f in FIELDS],'bounds':BOUNDS,'shape':[3]*5,
    'u_domain':[-4.,4.],'mu_domain':[0.,20.],'mass':1.,'charge':1.,'collision_strength':.1,
    'initial':'exp(-E-.2mu+.1sin(pi*y/.4)*u)','spatial_derivative':'global Lagrange polynomial',
    'u_derivative':'local quadratic; three nodes coincide with full polynomial differentiation',
    'projector':'original discrete-energy direction from the same tensor differentiation; no shifts or null removal',
    'singular_relative_thresholds':THRESHOLDS,'selected_numerical_rank_threshold':1e-12,
    'alias_parameters':{'c':1.,'a':.2,'k':.024},'overintegration_orders':[3,4,5,7,9],
    'production_probes':['sin(.13*node_index)','cos(.07*node_index)','sin(.17*node_index)+cos(.11*node_index)'],
    'independent_assembly':'tensor differentiation, unordered pair projector and measures; common chart coefficients from production geometry',
    'factor_residual_target':1e-13,'production_probe_relative_target':1e-11,'seed':None}
metadata=run_metadata(inputs,model='raw finite-dimensional local collision Gram diagnostic',
    boundary='natural collision-only Cartesian weak no-flux; no streaming',units='normalized')
metadata['example_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
metadata['limitations']=('One 243-node tensor grid. Numerical singular-value counts depend on scaling and threshold; no continuum spectral gap or full nullspace classification. '
    'The displayed tensor-polynomial interpolant is not continuum-null; these nodal values may admit a different nonpolynomial radial extension. '
    'Angular overintegration diagnoses the fixed polynomial and does not repair the collision operator. All factors and production actions are unchanged.')
start=perf_counter();rows=[];arrays={};spatial=[gauss_interval(3,a,b) for a,b in BOUNDS]
u_rule=gauss_interval(3,-4.,4.);mu_rule=gauss_interval(3,0.,20.)
for field in FIELDS:
    begin=perf_counter();print('Raw factors:',field.kind,flush=True)
    data=pair_factors(field,spatial,u_rule,mu_rule);F,L=data['F'],data['L'];grid=data['grid']
    spectra={name:la.svdvals(data[key]) for name,key in (('unweighted','F'),('physical','L'),('entropy_scaled','entropy_scaled'))}
    counts={name:{str(t):int(np.sum(s<s[0]*t)) for t in THRESHOLDS} for name,s in spectra.items()}
    expanded=np.column_stack((data['known'],data['extra']));basis_s=la.svdvals(expanded/la.norm(expanded,axis=0))
    baseline_s=la.svdvals(data['known']/la.norm(data['known'],axis=0))
    extra_residual=[float(la.norm(F@h)/(spectra['unweighted'][0]*la.norm(h))) for h in data['extra'].T]
    if field.kind=='nonaxisymmetric':candidates=data['known'][:,:4]
    else:candidates=expanded
    candidate_s=la.svdvals(candidates/la.norm(candidates,axis=0))
    _,sv,Vh=la.svd(F,full_matrices=False);null=Vh[sv<sv[0]*1e-12].T
    Q=la.orth(candidates,rcond=1e-12);overlap=la.svdvals(Q.T@null)
    indices=np.arange(grid.size);probes=np.column_stack((np.sin(.13*indices),np.cos(.07*indices),np.sin(.17*indices)+np.cos(.11*indices)))
    physical=L.T@(L@probes)
    action=jax.jit(jax.vmap(lambda h:mobility_action(grid,jnp.asarray(data['f']),h,collision_strength=.1),in_axes=1,out_axes=1))
    actual=np.asarray(action(jnp.asarray(probes)))
    error=float(la.norm(actual-physical)/la.norm(physical))
    expected=4 if field.kind=='nonaxisymmetric' else 12
    passed=(all(v['1e-12']==expected for v in counts.values()) and error<1e-11 and data['pair_weight_error']<1e-12
        and int(np.sum(candidate_s>candidate_s[0]*1e-12))==expected and
        np.min(overlap)>1-1e-10 and (field.kind=='nonaxisymmetric' or max(extra_residual)<1e-13))
    row={'field':field.kind,'nodes':grid.size,'unordered_pairs':grid.pair_count,'factor_shape':list(F.shape),
        'singular_counts':counts,'singular_maxima':{name:float(s[0]) for name,s in spectra.items()},
        'baseline_candidate_span_rank':int(np.sum(baseline_s>baseline_s[0]*1e-12)),
        'expanded_span_rank':int(np.sum(basis_s>basis_s[0]*1e-12)),
        'used_candidate_span_rank':int(np.sum(candidate_s>candidate_s[0]*1e-12)),
        'candidate_null_overlap_singular_values':overlap.tolist(),'extra_factor_relative_residual':extra_residual,
        'production_probe_relative_error':error,'pair_weight_relative_error':data['pair_weight_error'],
        'minimum_pair_energy_flow_squared':data['minimum_pair_flow_squared'],'checks_wall_s':perf_counter()-begin,
        'status':'passed_finite_grid_checks' if passed else 'unresolved'}
    rows.append(row);arrays.update({field.kind+'_'+name:s for name,s in spectra.items()})
    print(row,flush=True)
overintegration=[]
for n in inputs['overintegration_orders']:
    x,wx=gauss_interval(n,.8,1.2);y,wy=gauss_interval(n,-.2,.2);xx,yy=np.meshgrid(x,y,indexing='ij')
    integral=float(np.einsum('i,j,ij->',wx,wy,angular_residue(xx,yy)**2))
    overintegration.append({'order':n,'angular_residue_squared_integral':integral})
closed=angular_integral(.2)
passed=(overintegration[0]['angular_residue_squared_integral']<1e-28 and
    all(row['status']=='passed_finite_grid_checks' for row in rows) and all(abs(row['angular_residue_squared_integral']/closed-1)<1e-12 for row in overintegration[1:]))
evidence={'metadata':metadata,'rows':rows,'angular_overintegration':overintegration,'closed_angular_integral':closed,
    'identity':'Lphi=2y[(x-c)^3-k(x-c)]-(2x-3c)[y^3-k*y], L=-y*d_x+x*d_y',
    'planar_completeness':'Odd C(x)y vanishes only for C=0; even A(x)+B(x)y² has span{1,x²+y²,phi_alias} on these three nonzero x nodes.',
    'checks_wall_s':perf_counter()-start,'cost_scope':'Grid setup, independent factors/SVDs, selected production probes and angular quadrature; excludes imports, metadata and plots.',
    'status':'passed_finite_grid_checks' if passed else 'unresolved'}
np.savez_compressed(OUTPUT/'singular_values.npz',**arrays)
evidence['singular_arrays_sha256']=hashlib.sha256((OUTPUT/'singular_values.npz').read_bytes()).hexdigest()
(OUTPUT/'summary.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')

fig,ax=plt.subplots(1,2,figsize=(10.5,4),constrained_layout=True)
for row in rows:
    s=arrays[row['field']+'_unweighted'];ax[0].semilogy(np.arange(1,len(s)+1),s/s[0],'.',ms=3,label=row['field'])
ax[0].axhline(1e-12,color='k',ls=':',lw=1,label='selected numerical threshold')
ax[0].set(xlabel='singular-value index',ylabel=r'$s/s_{\max}$',title='Original rectangular pair factor');ax[0].legend(fontsize=8)
x,y=np.meshgrid(np.linspace(.8,1.2,121),np.linspace(-.2,.2,121),indexing='ij');residue=angular_residue(x,y);vmax=np.max(np.abs(residue))
picture=ax[1].pcolormesh(x,y,residue,cmap='RdBu_r',vmin=-vmax,vmax=vmax,shading='auto')
grid_x,grid_y=np.meshgrid(spatial[0][0],spatial[1][0],indexing='ij');ax[1].scatter(grid_x,grid_y,s=25,c='k',label='3x3 Gauss nodes')
ax[1].set(xlabel='x',ylabel='y',title=r'$L\phi$ of the fixed tensor polynomial');ax[1].legend(fontsize=8)
fig.colorbar(picture,ax=ax[1],label='angular derivative')
fig.suptitle('Nodal cancellation; positive squared-residue integral\n'
    rf'$\int (L\phi)^2 = {closed:.8g}$; Gauss 3 samples zero, Gauss 4+ is exact',fontsize=11)
fig.savefig(OUTPUT/'collocation_nullspace.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
for ax,name in zip(axes,('unweighted','physical','entropy_scaled')):
    for row in rows:
        s=arrays[row['field']+'_'+name];ax.semilogy(np.arange(1,len(s)+1),s/s[0],'.',ms=3,label=row['field'])
    for threshold in THRESHOLDS:ax.axhline(threshold,color='0.5',ls=':',lw=.7)
    ax.set(xlabel='singular-value index',ylabel=r'$s/s_{\max}$',title=name.replace('_',' '));ax.legend(fontsize=8)
fig.savefig(OUTPUT/'factor_scales.png',dpi=180);plt.close(fig)
print('Angular squared integral: Gauss3',overintegration[0]['angular_residue_squared_integral'],'closed form',closed,flush=True)
if not passed:raise RuntimeError('Finite-grid factor checks unresolved; retain diagnostics')
