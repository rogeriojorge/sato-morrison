"""Matched-error linear correction costs at one frozen difficult dipole iterate.

This is a finite-dimensional correction benchmark, not a trajectory benchmark.
Both auxiliaries solve the same original old-population-scaled SPD Hessian.
The tightened iterative reference supplies empirical correction accuracy;
its residual/refinement checks do not certify pointwise tail accuracy.
"""
from pathlib import Path
import hashlib,json,os,resource,sys
from time import perf_counter
print('Benchmark one frozen dipole Newton correction: same Hessian and RHS, two auxiliary preconditioners.',flush=True)
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sato_morrison.collisions import cartesian_grid
from sato_morrison.geometry import Field
from sato_morrison.reference import gauss_interval,run_metadata,progress
from sato_morrison.solver import lagged_entropy_compiler

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'results'/'difficult_correction'
(OUTPUT/'failure.json').unlink(missing_ok=True)
D,DT,CHUNK=.1,.005,1048576
ROUTE_RTOL,ROUTE_BUDGET=1e-6,3000
REFERENCE_TOLS,REFERENCE_BUDGET=(1e-9,1e-11),6000
CORRECTION_TARGET,REFERENCE_CHANGE_TARGET,INDEPENDENT_RESIDUAL_TARGET=1e-7,1e-9,5e-11
REPEATS=3
SHAPE=(5,5,5,25,13)
BOUNDS=((.8,1.2),(-.2,.2),(.1,.5))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def failure_evidence(value):
    # Failed records retain nonfinite evidence as explicit strings; no number
    # is repaired and no invalid value can enter a passing result.
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return {'nonfinite':str(value)}
    if isinstance(value,dict):return {key:failure_evidence(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [failure_evidence(item) for item in value]
    return value


def require(condition,message):
    if not condition:
        if 'metadata' in globals():
            (OUTPUT/'failure.json').write_text(json.dumps(failure_evidence(
                {'status':'failed','metadata':metadata,'error':message}),indent=2,allow_nan=False)+'\n')
        if 'report' in globals():
            report.update(status='failed',error=message)
            (OUTPUT/'summary.json').write_text(json.dumps(failure_evidence(report),indent=2,allow_nan=False)+'\n')
        raise RuntimeError(message)


def finite(value):
    if isinstance(value,(float,np.floating)):require(np.isfinite(value),'Nonfinite benchmark evidence')
    elif isinstance(value,dict):
        for item in value.values():finite(item)
    elif isinstance(value,(list,tuple)):
        for item in value:finite(item)


def peak_rss_bytes():
    # macOS reports bytes; Linux reports KiB. This is the whole-process high
    # water mark, including all previous routes, references and JAX runtime.
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform=='darwin' else value*1024)


def memory(kernel):
    info=kernel.memory_analysis()
    return {key:int(getattr(info,key)) for key in ('argument_size_in_bytes','output_size_in_bytes',
        'temp_size_in_bytes','alias_size_in_bytes','generated_code_size_in_bytes')}


def cardinal_derivative(nodes,local=False):
    result=np.zeros((len(nodes),len(nodes)))
    for i in range(len(nodes)):
        start=min(max(i-1,0),len(nodes)-3)
        selected=range(start,start+3) if local else range(len(nodes))
        for j in selected:
            p=np.polynomial.Polynomial([1.])
            for k in selected:
                if k!=j:p*=np.polynomial.Polynomial([nodes[i]-nodes[k],1.])/(nodes[j]-nodes[k])
            result[i,j]=p.deriv()(0.)
    return result


class IndependentDipolePairs:
    """Analytic field/Jacobian and independent unordered NumPy pair sums."""
    def __init__(self,axes,u,mu,wu,wm,spatial_weights,energy):
        self.shape=SHAPE;self.nx=125;self.nv=325
        xx,yy,zz=np.meshgrid(*axes,indexing='ij');xyz=np.stack((xx,yy,zz),axis=-1)
        r2=np.sum(xyz*xyz,axis=-1)
        v=np.stack((3*xx*zz,3*yy*zz,2*zz**2-xx**2-yy**2),axis=-1)
        field=v/r2[...,None]**2.5;B=np.linalg.norm(field,axis=-1);b=field/B[...,None]
        dv=np.zeros(xx.shape+(3,3));dv[...,0,0]=3*zz;dv[...,0,2]=3*xx;dv[...,1,1]=3*zz;dv[...,1,2]=3*yy
        dv[...,2,0]=-2*xx;dv[...,2,1]=-2*yy;dv[...,2,2]=4*zz
        jac=dv/r2[...,None,None]**2.5-5*v[..., :,None]*xyz[...,None,:]/r2[...,None,None]**3.5
        grad=np.einsum('...ij,...i->...j',jac,b)
        db=(jac-b[..., :,None]*grad[...,None,:])/B[...,None,None]
        curl=np.stack((db[...,2,1]-db[...,1,2],db[...,0,2]-db[...,2,0],db[...,1,0]-db[...,0,1]),axis=-1)
        cross=np.zeros(xx.shape+(3,3));cross[...,0,1]=-b[...,2];cross[...,0,2]=b[...,1];cross[...,1,0]=b[...,2]
        cross[...,1,2]=-b[...,0];cross[...,2,0]=-b[...,1];cross[...,2,1]=b[...,0]
        _,_,_,uu,mm=np.meshgrid(*axes,u,mu,indexing='ij')
        C=np.zeros(SHAPE+(5,4));C[...,:3,:3]=cross[...,None,None,:,:]/B[...,None,None,None,None]
        C[...,:3,3]=(field[...,None,None,:]+uu[...,None]*curl[...,None,None,:])/B[...,None,None,None]
        C[...,3,:3]=-C[...,:3,3]
        C[...,4,:]=mm[...,None]*np.einsum('...i,...ij->...j',grad[...,None,None,:],C[...,:3,:])
        self.coefficients=C;self.derivatives=[cardinal_derivative(axis) for axis in axes]+[cardinal_derivative(u,True)]
        self.wx=np.einsum('i,j,k->ijk',*spatial_weights).ravel()
        self.wv=B.reshape(125,1)*np.outer(wu,wm).ravel()[None,:]
        analytic_energy=(uu**2/2+mm*B[...,None,None]).ravel()
        require(np.allclose(energy,analytic_energy,rtol=2e-14,atol=2e-14),'Independent analytic energy disagreement')
        # Retain the exact fixture energy defining the modified projector after
        # checking it analytically. No nullspace projection or repair is used.
        self.flow=self.action(energy)
    def action(self,h):
        values=np.asarray(h).reshape(SHAPE);terms=[]
        for axis,derivative in enumerate(self.derivatives):
            terms.append(np.moveaxis(np.tensordot(derivative,values,axes=(1,axis)),0,axis))
        return np.einsum('...ad,...d->...a',self.coefficients,np.stack(terms,axis=-1)).reshape(125,325,5)
    def transpose(self,a):
        local=np.einsum('...ad,...a->...d',self.coefficients,a.reshape(SHAPE+(5,)))
        result=np.zeros(SHAPE)
        for axis,derivative in enumerate(self.derivatives):
            result+=np.moveaxis(np.tensordot(derivative.T,local[...,axis],axes=(1,axis)),0,axis)
        return result.ravel()
    def apply(self,f,h):
        actions=self.action(h);f=np.asarray(f).reshape(125,325);fluxes=np.zeros_like(actions)
        for x in range(125):
            weighted=self.wv[x]*f[x]
            for start in range(0,325,64):
                left=np.arange(start,min(start+64,325));mask=np.arange(325)[None,:]>left[:,None]
                xi=self.flow[x,left,None]-self.flow[x,None];norm=np.sum(xi*xi,axis=-1)
                require(np.all(norm[mask]>0),'Independent pair energy direction is zero')
                safe=np.where(norm>0,norm,1.)
                difference=actions[x,left,None]-actions[x,None]
                projected=difference-xi*(np.sum(xi*difference,axis=-1)/safe)[...,None]
                spatial=np.zeros_like(projected);spatial[...,:3]=projected[...,:3]
                projected=spatial-xi*(np.sum(xi*spatial,axis=-1)/safe)[...,None]
                pair=D*self.wx[x]*weighted[left,None]*weighted[None,:]*mask
                flux=pair[...,None]*projected
                fluxes[x,left]+=flux.sum(axis=1);fluxes[x]-=flux.sum(axis=0)
        return self.transpose(fluxes)


manifest=json.loads((OUTPUT/'input_provenance.json').read_text())
require(sha(OUTPUT/'input.npz')==manifest['input_sha256'],'Frozen fixture hash mismatch')
saved=np.load(OUTPUT/'input.npz');N=int(np.prod(SHAPE))
for key in ('old_log','new_log','weights','energy','mu_index'):
    require(saved[key].shape==(N,) and np.all(np.isfinite(saved[key])),'Invalid frozen array '+key)
for key in ('weights',):require(np.all(saved[key]>0),'Nonpositive frozen weights')
require(float(saved['dt'])==DT,'Frozen timestep mismatch')
old=jnp.asarray(saved['old_log']);new=jnp.asarray(saved['new_log'])
require(np.all(np.exp(np.asarray(old))>0) and np.all(np.exp(np.asarray(new))>0),'Frozen density is unrepresentable')
inputs={'fixture_sha256':manifest['input_sha256'],'fixture_origin':manifest['scope'],'geometry':manifest['parameters'],
    'collision_strength':D,'timestep':DT,
    'route_linear_rtol':ROUTE_RTOL,'route_linear_budget':ROUTE_BUDGET,'reference_linear_tolerances':REFERENCE_TOLS,
    'reference_linear_budget':REFERENCE_BUDGET,'correction_relative_error_target':CORRECTION_TARGET,
    'reference_relative_change_target':REFERENCE_CHANGE_TARGET,'independent_reference_residual_target':INDEPENDENT_RESIDUAL_TARGET,
    'true_residual_guard':'fresh ||H v+rhs||/||rhs|| <=5*linear_rtol and native converged',
    'error_metric':'||sqrt(w*f_old)*(dlog-dlog_reference)||/||sqrt(w*f_old)*dlog_reference||',
    'warm_repeats':REPEATS,'excluded_runtime_warmups_per_route':1,'chunk':CHUNK,'route_order':'alternate xline/xyz and xyz/xline',
    'metric':'original old population; no rescaling or reference-mass change','seed':None}
metadata=run_metadata(inputs,model='one frozen lagged-entropy SPD linear correction',boundary='natural weak collision-only box',units='normalized')
metadata['example_sha256']=sha(__file__);metadata['fixture_manifest_sha256']=sha(OUTPUT/'input_provenance.json')
report={'metadata':metadata,'status':'running','load_start':os.getloadavg(),'routes':{},'reference':[],
    'limits':['One saved first-step Newton iterate, not an accepted trajectory endpoint or full nonuniform evolution benchmark.',
     'Reference refinement and independent residual give empirical finite-dimensional accuracy; no rigorous forward/pointwise tail bound.',
     'XLA memory is compiled buffer analysis, not measured RSS; RSS is a cumulative whole-process peak.',
     'Warm timings include the public correction kernel\'s auxiliary factorization; partner setup and compilation are separate.',
     'Concurrent machine load is recorded; no portable hardware or general solver speedup claim.']}
start=perf_counter()
def save():
    report['checks_and_benchmark_wall_s']=perf_counter()-start
    report['cumulative_process_peak_rss_bytes']=peak_rss_bytes();finite(report)
    (OUTPUT/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
save()
print('Preparing geometry and two bound correction compilers.',flush=True)
begin=perf_counter();spatial=[gauss_interval(5,*b) for b in BOUNDS];axes=[p[0] for p in spatial]
u,wu=gauss_interval(25,-4.,4.);mu,wm=gauss_interval(13,0.,20.)
grid=cartesian_grid(*axes,u,mu,Field('dipole'),compact=True,spatial_weights=[p[1] for p in spatial],velocity_weights=(wu,wm),spatial_discretization='polynomial')
for key in ('weights','energy','mu_index'):require(np.array_equal(np.asarray(getattr(grid,key)),saved[key]),'Fixture does not match reproduced grid '+key)
report['grid_setup_wall_s']=perf_counter()-begin
mass=grid.weights*jnp.exp(old);root=jnp.sqrt(mass);ratio=jnp.exp(new-old)
compilers={name:lagged_entropy_compiler(grid,collision_strength=D,chunk_size=CHUNK,
    linear_max_steps=ROUTE_BUDGET,preconditioner_axis=axis) for name,axis in [('xline',0),('xyz',(0,1,2))]}
prepared={};kernels={}
for name,compiler in compilers.items():
    print('Prepare/compile auxiliary:',name,flush=True)
    begin=perf_counter();pkernel=compiler.prepare_diagonal.lower(old).compile();prepare_compile=perf_counter()-begin
    begin=perf_counter();prepared[name]=pkernel(old);jax.tree.map(lambda v:v.block_until_ready(),prepared[name]);prepare_wall=perf_counter()-begin
    require(all(np.all(np.isfinite(v)) and np.all(np.asarray(v)>=0) for v in jax.tree.leaves(prepared[name])),'Auxiliary preparation invalid')
    begin=perf_counter();kernels[name]=compiler.correction.lower(new,old,DT,prepared[name],ROUTE_RTOL).compile();compile_wall=perf_counter()-begin
    report['routes'][name]={'prepare_compile_wall_s':prepare_compile,'prepare_execute_wall_s':prepare_wall,
        'correction_compile_wall_s':compile_wall,'xla_correction_memory':memory(kernels[name]),
        'xla_prepare_memory':memory(pkernel),'prepared_array_bytes':sum(v.nbytes for v in jax.tree.leaves(prepared[name])),
        'one_factor_array_bytes':(8125*5*5 if name=='xline' else 325*125*125)*8,
        'cumulative_process_peak_rss_after_compile_bytes':peak_rss_bytes(),'warm':[]}
    save()
base=compilers['xyz'];rhs=base.evaluate(new,old,DT)/root
operator=jax.jit(lambda v:ratio*v+DT*base.apply(old,v/root)/root)

def checked_answer(answer,tol):
    answer[0].block_until_ready();vector=answer[0]*root
    actual=float(jnp.linalg.norm(operator(vector)+rhs)/jnp.linalg.norm(rhs))
    require(np.all(np.isfinite(vector)) and bool(answer[3]) and np.isfinite(actual) and actual<=5*tol,
            f'True linear residual failed: {actual:g}, native={bool(answer[3])}, status={int(answer[6])}')
    return vector,{'iterations':int(answer[1]),'fresh_true_relative_residual':actual,'native_true_relative_residual':float(answer[2])}

reference_compiler=lagged_entropy_compiler(grid,collision_strength=D,chunk_size=CHUNK,
    linear_max_steps=REFERENCE_BUDGET,preconditioner_axis=(0,1,2))
reference_vectors=[]
for tol in REFERENCE_TOLS:
    print('Compile tightened reference:',tol,flush=True)
    begin=perf_counter();kernel=reference_compiler.correction.lower(new,old,DT,prepared['xyz'],tol).compile();compile_wall=perf_counter()-begin
    with progress(f'Tightened correction reference {tol:g}'):
        begin=perf_counter();answer=kernel(new,old,DT,prepared['xyz']);answer[0].block_until_ready();solve_wall=perf_counter()-begin
    vector,diagnostics=checked_answer(answer,tol);reference_vectors.append(vector)
    report['reference'].append({'rtol':tol,'compile_wall_s':compile_wall,'solve_wall_s':solve_wall,**diagnostics});save()
fine=reference_vectors[-1];reference_change=float(jnp.linalg.norm(reference_vectors[0]-fine)/jnp.linalg.norm(fine))
report['reference_relative_change']=reference_change;save()
require(reference_change<=REFERENCE_CHANGE_TARGET,'Tightened correction reference did not stabilize')
print('Verify fine reference with independent analytic NumPy unordered pairs.',flush=True)
begin=perf_counter();independent=IndependentDipolePairs(axes,u,mu,wu,wm,[p[1] for p in spatial],saved['energy'])
independent_Hv=np.asarray(ratio)*np.asarray(fine)+DT*independent.apply(np.exp(np.asarray(old)),np.asarray(fine/root))/np.asarray(root)
independent_true=float(np.linalg.norm(independent_Hv+np.asarray(rhs))/np.linalg.norm(np.asarray(rhs)))
report['independent_reference_true_residual']=independent_true;report['independent_pair_check_wall_s']=perf_counter()-begin;save()
require(np.isfinite(independent_true) and independent_true<=INDEPENDENT_RESIDUAL_TARGET,'Independent reference residual failed')
# Explicit compilation does not warm executable/runtime state. The 6000-step
# reference executable also differs from each 3000-step measured route.
for name in compilers:
    with progress(f'Excluded runtime warm-up: {name}'):
        load=os.getloadavg();begin=perf_counter();answer=kernels[name](new,old,DT,prepared[name]);answer[0].block_until_ready();wall=perf_counter()-begin
    vector,diagnostics=checked_answer(answer,ROUTE_RTOL)
    error=float(jnp.linalg.norm(vector-fine)/jnp.linalg.norm(fine))
    report['routes'][name]['excluded_runtime_warmup']={'wall_s':wall,'correction_relative_error':error,
        'load_start':load,'load_end':os.getloadavg(),'cumulative_process_peak_rss_bytes':peak_rss_bytes(),**diagnostics}
    save();require(error<=CORRECTION_TARGET,'Warm-up correction does not meet matched error target')
for repeat in range(REPEATS):
    order=('xline','xyz') if repeat%2==0 else ('xyz','xline')
    for name in order:
        with progress(f'Warm correction {repeat+1}/{REPEATS}: {name}'):
            load=os.getloadavg();begin=perf_counter();answer=kernels[name](new,old,DT,prepared[name]);answer[0].block_until_ready();wall=perf_counter()-begin
        vector,diagnostics=checked_answer(answer,ROUTE_RTOL)
        error=float(jnp.linalg.norm(vector-fine)/jnp.linalg.norm(fine))
        report['routes'][name]['warm'].append({'repeat':repeat+1,'wall_s':wall,'correction_relative_error':error,
            'load_start':load,'load_end':os.getloadavg(),'cumulative_process_peak_rss_bytes':peak_rss_bytes(),**diagnostics})
        save();require(error<=CORRECTION_TARGET,'Correction does not meet matched error target')
for name,row in report['routes'].items():
    row['median_warm_wall_s']=float(np.median([r['wall_s'] for r in row['warm']]))
    row['maximum_correction_relative_error']=max(r['correction_relative_error'] for r in row['warm'])
report.update(status='passed_one_frozen_matched_error_correction',load_end=os.getloadavg());save()
np.savez_compressed(OUTPUT/'reference_vectors.npz',coarse=np.asarray(reference_vectors[0]),fine=np.asarray(fine),rhs=np.asarray(rhs),root=np.asarray(root))
report['reference_vectors_sha256']=sha(OUTPUT/'reference_vectors.npz');save()
fig,ax=plt.subplots(1,3,figsize=(11.5,3.6),constrained_layout=True)
names=list(report['routes']);colors=['#3564a6','#d57936']
for i,(name,color) in enumerate(zip(names,colors)):
    row=report['routes'][name];times=[r['wall_s'] for r in row['warm']]
    ax[0].scatter(np.full(REPEATS,i),times,color=color,s=35);ax[0].plot([i-.18,i+.18],[row['median_warm_wall_s']]*2,color=color,lw=3)
    ax[1].bar(i,row['maximum_correction_relative_error'],color=color)
    ax[2].bar(i,row['xla_correction_memory']['temp_size_in_bytes']/2**20,color=color)
for panel in ax:panel.set_xticks(range(2),names)
ax[0].set(ylabel='seconds',title='Paired warm correction timings')
ax[1].set(yscale='log',ylabel='relative correction error',title='Same tightened iterative reference');ax[1].axhline(CORRECTION_TARGET,color='k',ls=':',label='matched target');ax[1].legend(fontsize=8)
ax[2].set(ylabel='MiB',title='XLA temporary buffer analysis')
fig.suptitle('One frozen dipole Newton correction; same exact Hessian/RHS\nNo full-trajectory or general speedup claim',fontsize=11)
fig.savefig(OUTPUT/'difficult_correction.png',dpi=180);plt.close(fig)
print('Matched correction target passed; observed warm times and memory are scoped to this input and run.',flush=True)
