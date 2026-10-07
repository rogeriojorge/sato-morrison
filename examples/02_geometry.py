"""Resolved linear collision plus Hamiltonian evolution in a vacuum toroidal annulus."""
from pathlib import Path
import json
from time import perf_counter
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sato_morrison.collisions import toroidal_grid, mobility_action, dense_mobility
from sato_morrison.solver import linear_step, toroidal_stream, invariant_diagnostics
from sato_morrison.reference import run_metadata, progress

# B=C/R, R in [1,1.5], theta,z periodic. R,u,mu are constant on ideal orbits.
BASE = {'nr':5,'ntheta':3,'nz':3,'nu':11,'nmu':13,'umax':4.,'mumax':5.,'dt':.04}
FINAL_TIME, D, C, AMPLITUDE = .24, .15, 1., .02
RTOL, PAIR_CHUNK = 2e-11, 1024
SHOW_FIGURES = False
OUTPUT = Path(__file__).resolve().parents[1] / 'results'
OUTPUT.mkdir(exist_ok=True)
(OUTPUT/'figures').mkdir(exist_ok=True)
print(f'Model=sm181_local linear weak discretization; vacuum toroidal B=C/R, C={C}; '
      f'base grid={BASE}; normalized units; theta,z periodic; tangent radial Hamiltonian walls; '
      f'natural radial/u collision no-flux; Q decay and convergence; output={OUTPUT}', flush=True)


def run_case(parameters, dynamics='combined'):
    radius=np.linspace(1.,1.5,parameters['nr'])
    theta=np.arange(parameters['ntheta'])*2*np.pi/parameters['ntheta']
    z=np.arange(parameters['nz'])*2*np.pi/parameters['nz']
    u=np.linspace(-parameters['umax'],parameters['umax'],parameters['nu'])
    mu=np.linspace(.1,parameters['mumax'],parameters['nmu'])
    setup_start=perf_counter()
    grid=toroidal_grid(radius,theta,z,u,mu,strength=C)
    equilibrium=jnp.exp(-grid.energy-.3*jnp.tile(jnp.asarray(mu),grid.size//len(mu)))
    mass=grid.weights*equilibrium
    rr,tt,zz,uu,mm=np.meshgrid(radius,theta,z,u,mu,indexing='ij')
    h=jnp.asarray(np.cos(tt+zz)*(uu**2/3-.4)*(.8+.2*rr)).ravel()
    initial=equilibrium*(1+AMPLITUDE*h)
    stiffness=jax.jit(lambda value:mobility_action(grid,equilibrium,value,
                             collision_strength=D,chunk_size=PAIR_CHUNK))
    solve=jax.jit(lambda value:linear_step(mass,stiffness,value,parameters['dt'],rtol=RTOL,max_steps=400))
    stream=jax.jit(lambda value,dt:toroidal_stream(value,grid.shape,radius,u,mu,dt,strength=C))
    compile_start=perf_counter()
    if dynamics!='ideal':
        result=solve(h);result.x.block_until_ready()
        if not bool(result.converged):
            raise RuntimeError(f'Compilation trial PCG failed: {int(result.status)}')
    if dynamics!='collision':
        stream(h,parameters['dt']/2).block_until_ready()
    compile_s=perf_counter()-compile_start
    q0=float(jnp.vdot(h,mass*h));qs=[q0];times=[0.]
    max_residual=0.;max_iterations=0
    steps=int(round(FINAL_TIME/parameters['dt']))
    start=perf_counter()
    for i in range(steps):
        if dynamics!='collision':
            h=stream(h,parameters['dt']/2)
        if dynamics!='ideal':
            previous=h
            result=solve(h);result.x.block_until_ready()
            true_res=float(jnp.linalg.norm(mass*result.x+parameters['dt']*stiffness(result.x)-mass*previous)/jnp.linalg.norm(mass*previous))
            if not bool(result.converged) or true_res>5*RTOL:
                raise RuntimeError(f'PCG rejected step {i}: residual={true_res:.3e}, status={int(result.status)}')
            max_residual=max(max_residual,true_res);max_iterations=max(max_iterations,int(result.iterations));h=result.x
        if dynamics!='collision':
            h=stream(h,parameters['dt']/2)
        h.block_until_ready()
        qs.append(float(jnp.vdot(h,mass*h)));times.append((i+1)*parameters['dt'])
    elapsed=perf_counter()-start
    final=equilibrium*(1+AMPLITUDE*h)
    diagnostic=invariant_diagnostics(grid,final,initial)
    angular_momentum=jnp.asarray((rr*uu).ravel())
    diagnostic['angular_momentum_error']=float(jnp.abs(jnp.vdot(grid.weights*(final-initial),angular_momentum))/jnp.sum(grid.weights*initial*jnp.abs(angular_momentum)))
    radial_delta=np.asarray(grid.weights*(final-initial)).reshape(grid.shape).sum(axis=(1,2,3,4))
    diagnostic['radial_population_error']=float(np.max(np.abs(radial_delta))/jnp.sum(grid.weights*initial))
    if diagnostic['min_f']<=0:
        raise RuntimeError('finite-amplitude linear reconstruction became nonpositive')
    if max(diagnostic[k] for k in ('number_error','energy_error','marginal_error'))>1e-9:
        raise RuntimeError(f'invariant budget failed: {diagnostic}')
    if np.max(np.diff(qs))>1e-9*q0:
        raise RuntimeError('Q increased beyond the solve residual bound')
    # Direct fine-grid theta,z reconstruction: geometry does not depend on them.
    coefficients=np.fft.fftn(np.asarray(final).reshape(grid.shape),axes=(1,2))/(len(theta)*len(z))
    fine_theta=np.arange(3*len(theta))*2*np.pi/(3*len(theta));fine_z=np.arange(3*len(z))*2*np.pi/(3*len(z))
    phase_theta=np.exp(1j*np.outer(fine_theta,np.fft.fftfreq(len(theta),d=1/len(theta))))
    phase_z=np.exp(1j*np.outer(fine_z,np.fft.fftfreq(len(z),d=1/len(z))))
    reconstruction=np.einsum('ai,bj,rijkm->rabkm',phase_theta,phase_z,coefficients).real
    zero=coefficients[:,0,0].real
    positivity_bound=float(np.min(zero-(np.sum(np.abs(coefficients),axis=(1,2))-np.abs(coefficients[:,0,0]))))
    if positivity_bound<=0:
        raise RuntimeError('Continuous theta,z Fourier positivity certificate failed')
    if reconstruction.min()<=0:
        raise RuntimeError('theta,z Fourier reconstruction became nonpositive')
    return {'parameters':parameters,'dynamics':dynamics,'status':'passed',
            'Q_ratio':qs[-1]/q0,'Q_dissipated_fraction':1-qs[-1]/q0,
            'invariants':diagnostic,'max_true_pcg_residual':max_residual,
            'max_pcg_iterations':max_iterations,'reconstruction_min':float(reconstruction.min()),
            'continuous_spatial_positivity_bound':positivity_bound,
            'compile_s':compile_s,'setup_s':compile_start-setup_start,'warm_s':elapsed,
            'times':times,'Q':np.asarray(qs).tolist(),'minimum_B':C/1.5,'maximum_B':C,
            'state_bytes':grid.size*8,'pair_count':len(grid.left),'pair_workspace_bytes':PAIR_CHUNK*5*8*3}

with progress('Compile and evolve toroidal collision-only, ideal-only, combined controls'):
    controls=[run_case(BASE,kind) for kind in ('collision','ideal','combined')]
    for row in controls:
        print(f'  {row["dynamics"]}: Q/Q0={row["Q_ratio"]:.8f}; invariant errors={row["invariants"]}',flush=True)

# Each coordinate, velocity tail, and timestep changes independently.
axes = {'nr':[3,5,7],'ntheta':[3,5,7],'nz':[3,5,7],
        'nu':[7,11,15],'nmu':[9,13,17],'dt':[.08,.04,.02],
        'umax':[4.,4.8],'mumax':[5.,5.+2*(5.-.1)/12]}
rows=[];convergence={}
with progress('Run independent toroidal resolution and tail scans'):
    for axis,values in axes.items():
        scan=[]
        for value in values:
            parameters={**BASE,axis:value}
            # Expand tails at exactly the base velocity spacing.
            if axis=='umax':
                parameters['nu']=int(round(2*value/(2*BASE['umax']/(BASE['nu']-1))))+1
            if axis=='mumax':
                parameters['nmu']=int(round((value-.1)/((BASE['mumax']-.1)/(BASE['nmu']-1))))+1
            matching=parameters==BASE
            row=controls[2] if matching else run_case(parameters)
            row={**row,'scan_axis':axis,'scan_value':value}
            rows.append(row);scan.append(row)
            print(f'  {axis}={value}: Q/Q0={row["Q_ratio"]:.8f}, '
                  f'dissipated={row["Q_dissipated_fraction"]:.6f}, solve={row["warm_s"]:.3f}s',flush=True)
        last,previous=scan[-1],scan[-2]
        error=abs(last['Q_ratio']-previous['Q_ratio'])/abs(last['Q_ratio'])
        decay_error=abs(last['Q_dissipated_fraction']-previous['Q_dissipated_fraction'])/max(abs(last['Q_dissipated_fraction']),1e-30)
        convergence[axis]={'relative_Q_change':error,'relative_dissipation_change':decay_error,
                           'status':'passed' if error<.01 else 'unresolved',
                           'dissipation_status':'passed' if decay_error<.01 else 'unresolved'}

with progress('Check unmodified tiny toroidal entropy-weighted spectrum'):
    tiny=toroidal_grid(np.linspace(1,1.5,3),np.arange(3)*2*np.pi/3,np.arange(3)*2*np.pi/3,
                      np.linspace(-2,2,3),np.linspace(.1,1.1,3))
    f0=jnp.exp(-tiny.energy-.3*jnp.tile(jnp.linspace(.1,1.1,3),tiny.size//3))
    gram=dense_mobility(tiny,f0,collision_strength=D)
    m=np.asarray(tiny.weights*f0)
    scaled=gram/np.sqrt(m[:,None]*m[None,:])
    eigenvalues,eigenvectors=np.linalg.eigh(scaled)
    threshold=1e-10*max(np.linalg.norm(scaled,2),1.)
    known_columns=[]
    for radial in range(tiny.shape[0]):
        for angular in range(tiny.shape[1]):
            for vertical in range(tiny.shape[2]):
                column=np.zeros(tiny.shape);column[radial,angular,vertical]=1
                known_columns.append(column.ravel())
    tr,tt,tz,tu,tm=np.meshgrid(np.linspace(1,1.5,3),np.arange(3),np.arange(3),
                               np.linspace(-2,2,3),np.linspace(.1,1.1,3),indexing='ij')
    for angular in range(tiny.shape[1]):
        angle=(tt.ravel()==angular).astype(float)
        known_columns.extend([angle*(np.asarray(tiny.mu_index)==i) for i in range(tiny.shape[-1])])
        known_columns.extend([angle*np.asarray(tiny.energy),angle*(tr*tu).ravel()])
    known=np.stack(known_columns,axis=1)
    known_rank=int(np.linalg.matrix_rank(known))
    known_residual=float(np.linalg.norm(gram@known)/(np.linalg.norm(gram)*np.linalg.norm(known)))
    spectral={'size':tiny.size,'minimum_eigenvalue':float(eigenvalues.min()),
        'observed_nullity':int(np.count_nonzero(abs(eigenvalues)<threshold)),
        'null_threshold':threshold,'algebraically_required_mu_plus_energy':len(np.unique(tiny.mu_index))+1,
        'verified_known_invariant_rank':known_rank,'known_invariant_relative_residual':known_residual,
        'additional_known_invariants':'all spatial populations, c(theta)g(mu), c(theta)E, c(theta)mRu; certified rank equals observed tiny-grid nullity; no continuum exhaustion claim',
        'eigenpair_max_residual':float(np.linalg.norm(scaled@eigenvectors-eigenvectors*eigenvalues)/max(np.linalg.norm(scaled),1e-30)),
        'eigenvalues':eigenvalues.tolist(),'scope':'unmodified finite-grid local collision matrix, no zero modes removed'}

report={**run_metadata({'base':BASE,'D':D,'C':C,'amplitude':AMPLITUDE,'time':FINAL_TIME,
    'radius_domain':[1,1.5],'theta_period':2*np.pi,'z_period':2*np.pi,'rtol':RTOL,
    'pair_chunk':PAIR_CHUNK,'seed':None,'initial':'f0*(1+.02cos(theta+z)(u²/3-.4)(.8+.2R))',
    'equilibrium':'exp(-E-.3mu)','spatial_quadrature':'periodic theta,z trapezoid; radial trapezoid',
    'velocity_quadrature':'u,mu trapezoid','state_reconstruction':'Fourier theta,z; positive piecewise-linear R,u,mu','time_method':'half exact streaming / backward Euler collision / half streaming'},
    model='sm181_local',boundary='periodic theta,z; Hamiltonian tangent radial walls; natural radial/u collision no-flux'),
    'equation':'Eq181 local weak Gram; common Cartesian eta chart; discrete energy-gradient projector',
    'normalization':'dimensionless fixed D; no Coulomb coefficient calibration',
    'status':'passed' if all(row['status']=='passed' and row['dissipation_status']=='passed' for row in convergence.values()) else 'unresolved',
    'controls':controls,'scans':rows,'convergence':convergence,'spectrum':spectral,
    'limits':'linear fixed-field distribution; radial/u/velocity-tail continuum convergence is separate from discrete invariants'}
(OUTPUT/'geometry.json').write_text(json.dumps(report,indent=2)+'\n')
fig,plots=plt.subplots(1,2,figsize=(10,3.6))
for control in controls:
    plots[0].plot(control['times'],np.asarray(control['Q'])/control['Q'][0],'o-',label=control['dynamics'])
plots[0].set(xlabel='time',ylabel='Q / Q initial');plots[0].legend()
labels=list(convergence);errors=[convergence[key]['relative_Q_change'] for key in labels]
plots[1].bar(labels,np.maximum(errors,1e-16));plots[1].set_yscale('log')
plots[1].axhline(.01,color='black',linestyle='--');plots[1].tick_params(axis='x',rotation=45)
plots[1].set(ylabel='last independent refinement: relative Q change')
fig.suptitle('Vacuum toroidal annulus: tangent flow and local collision weak form');fig.tight_layout()
fig.savefig(OUTPUT/'figures/geometry.png',dpi=170)
if SHOW_FIGURES:
    plt.show()
plt.close(fig)
print(f'Saved {OUTPUT/"geometry.json"}; convergence={convergence}; tiny spectrum nullity='
      f'{spectral["observed_nullity"]}; status={report["status"]}',flush=True)
