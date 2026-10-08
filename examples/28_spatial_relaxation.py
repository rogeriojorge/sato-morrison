"""Dipole field geometry and three recorded constrained-relaxation snapshots."""
from pathlib import Path
from hashlib import sha256
from itertools import product, combinations
from time import perf_counter
import json
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from sato_morrison.geometry import Field, field_vector
from sato_morrison.reference import run_metadata

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'results'/'spatial_relaxation'
SOURCE=ROOT/'results'/'nonuniform_spatial_blocks'
INPUT_HASHES={
    'summary.json':'8826a377f7d6a8d363f6a49236e205de9c42f24c9e21a6b45ea78a2b92b773f4',
    'state_dipole_mu13.npz':'fcec71607fa066fc095b94de5ab92cf6fd4bfe7724620fd56d249cdf5f27c7a7',
    'dipole_mu13_pair_audit.json':'338882dc755486cd284baf0c13e98c47189f57b740ccea14a5521a46dc4202de'}
PRODUCER='93bf752da33a31ec158a229f87eaa285aeb5c58b'
BOUNDS=np.array([[.8,1.2],[-.2,.2],[.1,.5]])
NX,NU,NMU=5,25,13
MASS,FIELD_STRENGTH=1.,1.
TIMES=np.array([0.,.015,.02])
L_SHELLS=np.array([.7,1.05,1.4,1.8,2.2])
AZIMUTHS=np.linspace(0,2*np.pi,8,endpoint=False)
SOURCE_CUTOFF=.25
TEAL,ORANGE,INK='#147D92','#C66B28','#253447'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
    'text.color':INK,'axes.labelcolor':INK,'figure.facecolor':'white',
    'savefig.facecolor':'white','svg.fonttype':'path'})
print('Verify saved dipole states, measures and analytic field lines before rendering.',flush=True)
print('Geometry checks compile a field evaluation; collision states are read from accepted checkpoints.',flush=True)
started=perf_counter()
for name,digest in INPUT_HASHES.items():
    if sha256((SOURCE/name).read_bytes()).hexdigest()!=digest:
        raise RuntimeError(f'Saved source changed: {name}')
summary=json.loads((SOURCE/'summary.json').read_text())
audit=json.loads((SOURCE/'dipole_mu13_pair_audit.json').read_text())
row=next(r for r in summary['rows'] if r['field']=='dipole' and r['case']=='mu13')
origin=summary['provenance_runs'][row['provenance_id']]
if origin['commit']!=PRODUCER or row['status']!='passed' or audit['status']!='passed':
    raise RuntimeError('Required completed pilot and independent pair audit are missing')
for key,value in {'nx':NX,'nu':NU,'nmu':NMU,'dt':.005,'nodes':40625}.items():
    if row[key]!=value:raise RuntimeError(f'Pilot grid changed: {key}')


def gauss(n,a,b):
    x,w=np.polynomial.legendre.leggauss(n)
    return (a+b)/2+(b-a)*x/2,(b-a)*w/2


def dipole(points):
    points=np.asarray(points);r2=np.sum(points**2,axis=-1)
    if np.any(r2<=0):raise ValueError('Dipole source is excluded')
    x,y,z=np.moveaxis(points,-1,0)
    return FIELD_STRENGTH*np.stack([3*x*z,3*y*z,2*z*z-x*x-y*y],axis=-1)/r2[...,None]**2.5


def edges(nodes,bounds):
    return np.r_[bounds[0],(nodes[:-1]+nodes[1:])/2,bounds[1]]


def save(fig,name):
    fig.savefig(OUTPUT/f'{name}.png',dpi=180)
    fig.savefig(OUTPUT/f'{name}.svg',metadata={'Creator':None,'Date':None})
    plt.close(fig)


spatial=[gauss(NX,*b) for b in BOUNDS]
axes=[r[0] for r in spatial]
u,wu=gauss(NU,-4.,4.);mu,wm=gauss(NMU,0.,20.)
x,y,z=np.meshgrid(*axes,indexing='ij');points=np.stack([x,y,z],axis=-1)
B=np.linalg.norm(dipole(points),axis=-1)
wX=np.einsum('i,j,k->ijk',*[r[1] for r in spatial])
wV=wu[:,None]*wm[None,:]
w=wX[...,None,None]*B[...,None,None]*wV
energy=.5*MASS*u[None,None,None,:,None]**2+mu[None,None,None,None,:]*B[...,None,None]
with np.load(SOURCE/'state_dipole_mu13.npz') as saved:
    if saved['field'].item()!='dipole' or saved['case'].item()!='mu13' or saved['time'].item()!=TIMES[-1] or saved['provenance_id'].item()!=row['provenance_id']:
        raise RuntimeError('Checkpoint identity/time mismatch')
    states=np.stack([saved[k].reshape(NX,NX,NX,NU,NMU) for k in ['initial','previous','state']])
    reference_population=saved['reference_population'].reshape(w.shape)
if not np.all(np.isfinite(states)) or np.any(states<=0):raise RuntimeError('Invalid accepted state')
expected_initial=np.exp(-energy-.2*mu[None,None,None,None,:]+.1*np.sin(np.pi*y/.4)[...,None,None]*u[None,None,None,:,None])
initial_error=float(np.max(abs(states[0]/expected_initial-1)))
weight_error=float(np.max(abs(reference_population/(w*states[0])-1)))
if max(initial_error,weight_error)>2e-12:raise RuntimeError('Initial condition or invariant measure reconstruction failed')
numbers=np.einsum('tijkab,ijkab->t',states,w)
energies=np.einsum('tijkab,ijkab,ijkab->t',states,w,energy)
bins=np.einsum('tijkab,ijkab->tb',states,w)
number_error=float(np.max(abs(numbers/numbers[0]-1)))
energy_error=float(np.max(abs(energies/energies[0]-1)))
bin_error=float(np.max(abs(bins/bins[0]-1)))
if max(number_error,energy_error,bin_error)>1e-11 or abs(numbers[0]/row['number']-1)>1e-12:
    raise RuntimeError('Reconstructed conserved moments disagree')
for state_index,history_index in [(1,2),(2,3)]:
    if row['history'][history_index]['time']!=TIMES[state_index]:
        raise RuntimeError('Snapshot time differs from accepted history')
    if not np.allclose(bins[state_index],row['history'][history_index]['magnetic_moment_bin_populations'],rtol=1e-12,atol=0):
        raise RuntimeError('Snapshot bin populations disagree with accepted history')
local_density=B[None,...]*np.einsum('tijkab,ab->tijk',states,wV)
parallel_mean=B[None,...]*np.einsum('tijkab,a,ab->tijk',states,u,wV)/local_density
velocity_shape=np.einsum('tijkab,b->tijka',states,wm)
velocity_shape/=np.einsum('tijka,a->tijk',velocity_shape,wu)[...,None]
if np.max(abs(np.einsum('tijka,a->tijk',velocity_shape,wu)-1))>1e-13:
    raise RuntimeError('Conditional velocity distributions do not normalize')
thermal=np.exp(-u*u/2);thermal/=wu@thermal
relative_shape=velocity_shape/thermal-1
OUTPUT.mkdir(parents=True,exist_ok=True)
metadata=run_metadata({'input_sha256':INPUT_HASHES,'simulation_producer_commit':PRODUCER,
    'bounds':BOUNDS.tolist(),'nx':NX,'nu':NU,'nmu':NMU,'mass':MASS,'field_strength':FIELD_STRENGTH,
    'times':TIMES.tolist(),'u_bounds':[-4,4],'mu_bounds':[0,20],'z_slice_index':2,
    'local_velocity_points':[[2,1,2],[2,3,2]],'L_shells':L_SHELLS.tolist(),
    'azimuths':AZIMUTHS.tolist(),'field_line_source_cutoff':SOURCE_CUTOFF},
    model='Prescribed dipole geometry and accepted collision-only pilot snapshots',
    boundary='Natural no-flux collision box; field-line drawing extends outside the box without evolving a distribution',units='normalized')
metadata['example_sha256']=sha256(Path(__file__).read_bytes()).hexdigest()

# Each r=L sin²(theta) curve has constant dipole flux psi=1/L.
curves=[];tangents=[];flux_errors=[]
for L,phi in product(L_SHELLS,AZIMUTHS):
    theta=np.linspace(np.arcsin(np.sqrt(SOURCE_CUTOFF/L)),np.pi-np.arcsin(np.sqrt(SOURCE_CUTOFF/L)),181)
    radius=L*np.sin(theta)**2;R=radius*np.sin(theta);Z=radius*np.cos(theta)
    xyz=np.stack([R*np.cos(phi),R*np.sin(phi),Z],axis=-1)
    Rp=3*L*np.sin(theta)**2*np.cos(theta)
    Zp=L*(2*np.sin(theta)*np.cos(theta)**2-np.sin(theta)**3)
    tangent=np.stack([Rp*np.cos(phi),Rp*np.sin(phi),Zp],axis=-1)
    curves.append(xyz);tangents.append(tangent)
    psi=(xyz[:,0]**2+xyz[:,1]**2)/np.sum(xyz**2,axis=1)**1.5
    flux_errors.append(float(np.max(abs(psi*L-1))))
all_points=np.concatenate(curves);all_tangents=np.concatenate(tangents)
analytic=dipole(all_points)
field_values=np.asarray(jax.jit(jax.vmap(lambda p:field_vector(p,Field('dipole'))))(jnp.asarray(all_points)))
field_error=float(np.max(np.linalg.norm(field_values-analytic,axis=1)/np.linalg.norm(analytic,axis=1)))
tangent_error=float(np.max(np.linalg.norm(np.cross(all_tangents,field_values),axis=1)/(np.linalg.norm(all_tangents,axis=1)*np.linalg.norm(field_values,axis=1))))
if max(field_error,tangent_error,max(flux_errors))>2e-13:raise RuntimeError('Analytic dipole field lines fail field/flux checks')
print(f'  Field tangency error {tangent_error:.2e}; full μ-bin error {bin_error:.2e}. Render spatial views.',flush=True)

fig=plt.figure(figsize=(11.4,5.1));ax=fig.add_subplot(121,projection='3d');plane=fig.add_subplot(122)
for curve in curves:ax.plot(*curve.T,color=TEAL,alpha=.52,lw=.9)
corners=np.array(list(product(*BOUNDS)))
for a,b in combinations(corners,2):
    if np.count_nonzero(a!=b)==1:ax.plot(*np.stack([a,b]).T,color=ORANGE,lw=2.2)
ax.scatter([0],[0],[0],color=INK,marker='x',s=22)
ax.text(0,0,.20,'source',fontsize=8)
ax.text(1.24,-.2,.48,'collision box',color=ORANGE,fontsize=10)
ax.set(xlim=(-2.3,2.3),ylim=(-2.3,2.3),zlim=(-1.,1.),xlabel='x',ylabel='y',zlabel='z',title='Prescribed dipole field lines')
ax.set_box_aspect((4.6,4.6,2.));ax.view_init(elev=23,azim=-57);ax.grid(alpha=.2)
rr=np.linspace(.26,2.35,181);zz=np.linspace(-1.1,1.1,181);RR,ZZ=np.meshgrid(rr,zz)
section=np.stack([RR,np.zeros_like(RR),ZZ],axis=-1);BB=np.linalg.norm(dipole(section),axis=-1)
mesh=plane.pcolormesh(RR,ZZ,BB,norm=LogNorm(.06,120),cmap='viridis',shading='auto',rasterized=True)
for curve in curves[::len(AZIMUTHS)]:plane.plot(curve[:,0],curve[:,2],color='white',alpha=.8,lw=.85)
plane.add_patch(Rectangle((BOUNDS[0,0],BOUNDS[2,0]),np.ptp(BOUNDS[0]),np.ptp(BOUNDS[2]),fill=False,ec=ORANGE,lw=2.5))
plane.annotate('collision box',(.99,.49),(1.45,.83),color=ORANGE,fontsize=10,arrowprops={'arrowstyle':'->','color':ORANGE})
plane.set(xlim=(rr[0],rr[-1]),ylim=(zz[0],zz[-1]),xlabel='R at y = 0',ylabel='z',title='Field strength in a meridional section',aspect='equal')
fig.colorbar(mesh,ax=plane,label='|B|',fraction=.045,pad=.03)
fig.subplots_adjust(left=.02,right=.93,top=.85,bottom=.16,wspace=.17)
fig.suptitle('Dipole field and simulation domain',fontsize=15,weight='bold',y=.98)
fig.text(.5,.045,'Orange box: x ∈ [0.8,1.2], y ∈ [−0.2,0.2], z ∈ [0.1,0.5]. Curves follow the static magnetic field.',ha='center',fontsize=10)
save(fig,'dipole_geometry')

fig=plt.figure(figsize=(12.6,8.1));gs=fig.add_gridspec(2,3,left=.065,right=.89,bottom=.13,top=.86,hspace=.68,wspace=.65)
map_axes=[fig.add_subplot(gs[0,i]) for i in range(3)]
for i,ax in enumerate(map_axes):
    mesh=ax.pcolormesh(edges(axes[0],BOUNDS[0]),edges(axes[1],BOUNDS[1]),parallel_mean[i,:,:,2].T,cmap='RdBu_r',vmin=-.10,vmax=.10)
    xx,yy=np.meshgrid(axes[0],axes[1],indexing='ij');ax.scatter(xx,yy,s=4,c='k',alpha=.25)
    ax.set(xlabel='x',ylabel='y',title=f'Accepted state: t = {TIMES[i]:g}',aspect='equal',xticks=[.8,1.,1.2],yticks=[-.2,0,.2])
cax=fig.add_axes([.925,.59,.013,.26]);fig.colorbar(mesh,cax=cax,label='mean parallel velocity ⟨u⟩')
ax=fig.add_subplot(gs[1,0]);delta=parallel_mean[-1,:,:,2]-parallel_mean[0,:,:,2];limit=float(np.max(abs(delta)))
mesh=ax.pcolormesh(edges(axes[0],BOUNDS[0]),edges(axes[1],BOUNDS[1]),delta.T,cmap='RdBu_r',vmin=-limit,vmax=limit)
ax.set(xlabel='x',ylabel='y',title='Change from t = 0 to 0.02',aspect='equal',xticks=[.8,1.,1.2],yticks=[-.2,0,.2])
bar=fig.colorbar(mesh,ax=ax,fraction=.05,pad=.02);bar.ax.set_title('Δ⟨u⟩',fontsize=9,pad=7)
ax=fig.add_subplot(gs[1,1])
for iy,color in [(1,TEAL),(3,ORANGE)]:
    for it,style in [(0,'--'),(2,'-')]:
        ax.plot(u,relative_shape[it,2,iy,2],style,color=color,lw=1.7,label=f'y={axes[1][iy]:+.3f}' if it==2 else None)
ax.axhline(0,color='0.65',lw=.7)
ax.set(xlabel='parallel velocity u',ylabel=r'$p/p_M-1$',title='Local velocity asymmetry',xlim=(-4,4))
ax.legend(fontsize=9,frameon=False,loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2)
ax.set_title('Velocity asymmetry: dashed initial, solid final',fontsize=9.5,pad=32)
ax=fig.add_subplot(gs[1,2]);ax.semilogy(mu,bins[0]/numbers[0],'o-',color=INK,ms=5,label='initial')
ax.semilogy(mu,bins[-1]/numbers[-1],'x',color=ORANGE,ms=6,label='t = 0.02')
ax.set(xlabel='magnetic moment μ',ylabel='fraction of all particles in each bin',title='Magnetic-moment populations')
ax.legend(fontsize=9,frameon=False)
fig.suptitle('Opposing parallel flows relax while every μ-bin population is retained',fontsize=14,weight='bold',y=.98)
fig.text(.5,.91,'Dipole pilot • maps at z = 0.3 • dots mark the 5 × 5 spatial quadrature nodes',ha='center',fontsize=11)
fig.text(.5,.045,'40,625 phase-space nodes; Δt = 0.005; D = 0.1. Only the three archived spatial states are shown.',ha='center',fontsize=10)
save(fig,'dipole_relaxation')
np.savez_compressed(OUTPUT/'plotted_arrays.npz',x=axes[0],y=axes[1],z=axes[2],u=u,mu=mu,wu=wu,wm=wm,times=TIMES,
    density=local_density,mean_parallel_velocity=parallel_mean,conditional_velocity=velocity_shape,
    relative_velocity_shape=relative_shape,mu_bin_populations=bins,numbers=numbers,energies=energies,
    field_line_points=np.stack(curves),field_line_tangents=np.stack(tangents))
metadata.update({'status':'passed_saved_state_and_geometry_checks','checks':{
    'initial_formula_max_relative_error':initial_error,'weights_max_relative_error':weight_error,
    'number_max_relative_error':number_error,'energy_max_relative_error':energy_error,'mu_bin_max_relative_error':bin_error,
    'field_max_relative_error':field_error,'field_tangent_max_relative_cross':tangent_error,'flux_max_relative_error':max(flux_errors)},
    'observable_definitions':{'density':'B sum(wu wmu f)','mean_parallel_velocity':'B sum(wu wmu u f)/density',
    'conditional_velocity':'sum(wmu f)/sum(wu wmu f)','thermal_reference':'exp(-u²/2) normalized by the same wu quadrature',
    'mu_bin_population':'wmu sum(wX B wu f)'},
    'display':{'geometry':'Equal physical unit lengths on 3D axes; static field lines, prescribed source cutoff. The right panel has a logarithmic field-strength scale.',
    'maps':'Piecewise-constant display tiles associated with quadrature nodes, with midpoint display edges clipped at the box bounds. Tiles are not finite-volume cell averages.',
    'time':'Only t=0,.015,.02 accepted snapshots. No interpolation of simulation states.',
    'velocity':'Both local curves use x=1,z=.3, and the stated opposite-y quadrature nodes. Lines connect discrete velocity samples.'},
    'scope':'Saved finite-grid collision-only pilot; the full nonuniform refinement and physical collision-rate calibration remain unresolved.',
    'wall_s':perf_counter()-started,'wall_s_scope':'Input verification, analytic geometry evaluation and figure rendering; excludes imports and final metadata write.'})
metadata['outputs']={p.name:sha256(p.read_bytes()).hexdigest() for p in OUTPUT.iterdir() if p.suffix in ('.png','.svg','.npz')}
(OUTPUT/'summary.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
print(f'Spatial figures saved; render/check wall {metadata["wall_s"]:.2f}s.',flush=True)
