"""Shared homogeneous data: analytic mode evolution and quadrature checks."""
from pathlib import Path
from hashlib import sha256
from time import perf_counter
import json
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter, writers
from sato_morrison.controls import (GaussianMixture, dougherty_evolve,
    landau_gaussian_analytic, landau_gaussian_quadrature)
from sato_morrison.collisions import uniform_grid, nonlinear_rhs
from sato_morrison.reference import gauss_interval, progress, run_metadata

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'results'/'operator_comparison'
A = np.array([1.15, 1.15, .7])
MASS, FIELD, GAMMA = 1., 1., 1.
TAU = np.linspace(0., 3., 61)
ORDER, RADIUS, DEGREE, ANGULAR_ORDER = 64, 10., 56, 96
MU_PLOT = np.linspace(0., 6., 121)
U_PLOT = np.linspace(-4., 4., 81)
VPERP_PLOT = np.linspace(0., 4., 61)
FPS = 10
COLORS = ['#5F6670', '#147D92', '#C66B28']
LABELS = ['Constrained local model', 'Lorentz', 'Dougherty']
TOLERANCE = 2e-8
OUTPUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size':10, 'axes.spines.top':False,
    'axes.spines.right':False, 'svg.fonttype':'path', 'figure.facecolor':'white'})
print('Common initial distribution: n=m=B=1, U=0, covariance diag(1.15,1.15,0.7).', flush=True)
print('Match Lorentz and Dougherty initial anisotropy rates to the independently checked Landau rate.', flush=True)
print('Analytic mode evolution with angular reconstruction checks; homogeneous constrained null state; Landau evolution is separate.', flush=True)
started = perf_counter()
initial = GaussianMixture(np.ones(1), np.zeros((1,3)), np.diag(A)[None])
landau_rate = landau_gaussian_analytic(np.diag(A), GAMMA)
landau_check = landau_gaussian_quadrature(np.diag(A), GAMMA)
alpha = -(landau_rate[0,0]-landau_rate[2,2])/(A[0]-A[2])
TIMES = TAU/alpha
rates = {'landau_initial_anisotropy':float(alpha), 'lorentz_nu':float(alpha/3),
         'dougherty_nu':float(alpha/2), 'landau_gamma':GAMMA}


def gaussian(u, perpendicular, variance=A):
    return np.exp(-.5*(u*u/variance[2]+perpendicular**2/variance[0])) / (
        (2*np.pi)**1.5*variance[0]*np.sqrt(variance[2]))


def lorentz_coefficients(speed, degree, angular_order):
    xi, w = np.polynomial.legendre.leggauss(angular_order)
    basis = np.polynomial.legendre.legvander(xi, degree)
    f = gaussian(speed[...,None]*xi, speed[...,None]*np.sqrt(1-xi*xi))
    return (f*w) @ basis * (2*np.arange(degree+1)+1)/2


def populations(u, perpendicular, times, degree=DEGREE, angular_order=ANGULAR_ORDER):
    """Physical 3V density F, evaluated at gyrotropic velocity points."""
    speed = np.sqrt(u*u+perpendicular**2)
    xi = np.divide(u, speed, out=np.zeros_like(speed), where=speed>0)
    coefficients = lorentz_coefficients(speed, degree, angular_order)
    basis = np.polynomial.legendre.legvander(xi, degree)
    ell = np.arange(degree+1)
    output = np.empty((3,len(times),*u.shape))
    output[0] = gaussian(u, perpendicular)
    for k,t in enumerate(times):
        # At t=0 evaluate the common distribution directly, avoiding a truncated reconstruction.
        output[1,k] = gaussian(u, perpendicular) if t == 0 else np.sum(
            coefficients*basis*np.exp(-rates['lorentz_nu']*ell*(ell+1)*t/2), axis=-1)
        velocities = np.stack([perpendicular,np.zeros_like(u),u],axis=-1)
        output[2,k] = dougherty_evolve(initial,rates['dougherty_nu'],t).evaluate(velocities)
    if not np.all(np.isfinite(output)) or np.any(output<=0):
        raise RuntimeError(f'Spectral angular reconstruction lost positive finite density: minimum={np.min(output):.17e}')
    return output


def diagnostics(order, radius, times, degree=DEGREE, angular_order=ANGULAR_ORDER):
    r, wr = gauss_interval(order, 0., radius)
    xi, wx = np.polynomial.legendre.leggauss(order)
    rr, xx = np.meshgrid(r, xi, indexing='ij')
    u, vp = rr*xx, rr*np.sqrt(1-xx*xx)
    weight = 2*np.pi*rr**2*wr[:,None]*wx[None,:]
    f = populations(u, vp, times, degree, angular_order)
    fm = gaussian(u, vp, np.ones(3))
    moment = lambda observable: np.einsum('mtrp,rp->mt', f, weight*observable)
    n = moment(np.ones_like(rr))
    tp, tz = moment(vp**2/2), moment(u*u)
    return {'number':n, 'momentum':moment(u), 'energy':moment(rr*rr/2),
        'anisotropy':tp-tz, 'mean_mu':tp, 'mu_second':moment(vp**4/4),
        'speed_fourth':moment(rr**4),
        'relative_entropy':np.sum(weight*(f*np.log(f/fm)-f+fm),axis=(-2,-1)),
        'minimum':f.min(axis=(-2,-1))}


def mu_marginal(times, order=ORDER, extent=RADIUS, degree=DEGREE, angular_order=ANGULAR_ORDER):
    u, wu = gauss_interval(order,-extent,extent)
    uu, mm = np.meshgrid(u,MU_PLOT,indexing='ij')
    # d^3v=2*pi*B/m du dmu; B=m=1. G(mu) is a density with respect to dmu.
    f = populations(uu,np.sqrt(2*mm),times,degree,angular_order)
    return 2*np.pi*np.einsum('ktij,i->ktj',f,wu)


up,vp = np.meshgrid(U_PLOT,VPERP_PLOT,indexing='xy')
with progress('Evaluate exact common-problem curves and quadrature refinements'):
    curves = diagnostics(ORDER, RADIUS, TIMES)
    marginal = mu_marginal(TIMES)
    maps = populations(up,vp,TIMES)
    check_indices = [0,1,20,60]
    check_times = TIMES[check_indices]
    refinements = []
    for dimension, levels in [('quadrature',[32,48,64]), ('tail',[8.,10.,12.]),
                              ('angular_degree',[24,40,56]), ('angular_projection',[64,80,96])]:
        for level in levels:
            n = int(level) if dimension=='quadrature' else ORDER
            radius = level if dimension=='tail' else RADIUS
            degree = int(level) if dimension=='angular_degree' else DEGREE
            angular_order = int(level) if dimension=='angular_projection' else ANGULAR_ORDER
            try:
                value = diagnostics(n,radius,check_times,degree,angular_order)
                errors = {key:float(np.max(abs(value[key]-curves[key][:,check_indices])))
                          for key in ['number','energy','anisotropy','mu_second','speed_fourth','relative_entropy']}
                errors['mu_marginal'] = float(np.max(abs(mu_marginal(check_times,n,radius,degree,angular_order)-marginal[:,check_indices])))
                errors['plotted_density'] = float(np.max(abs(populations(up,vp,check_times,degree,angular_order)-maps[:,check_indices]))/gaussian(0.,0.))
                refinements.append({'direction':dimension,'level':level,'status':'passed' if max(errors.values())<=TOLERANCE else 'unresolved','absolute_errors':errors})
                print(f'  {dimension}={level}: largest observable change={max(errors.values()):.3e}',flush=True)
            except RuntimeError as error:
                refinements.append({'direction':dimension,'level':level,'status':'failed','reason':str(error)})
                print(f'  REJECTED {dimension}={level}: {error}',flush=True)
    try:
        diagnostics(ORDER,12.,check_times)
        extended_tail_probe = {'radius':12.,'status':'passed_positive_reconstruction'}
    except RuntimeError as error:
        extended_tail_probe = {'radius':12.,'status':'failed','reason':str(error)}
        print(f'  REJECTED extended tail probe: {error}',flush=True)
    r_eq,w_eq = gauss_interval(ORDER,0.,RADIUS)
    lorentz_equilibrium = lorentz_coefficients(r_eq,DEGREE,ANGULAR_ORDER)[:,0]
    maxwellian_equilibrium = gaussian(np.zeros_like(r_eq),r_eq,np.ones(3))
    lorentz_entropy_floor = float(np.sum(4*np.pi*r_eq*r_eq*w_eq*(lorentz_equilibrium*np.log(lorentz_equilibrium/maxwellian_equilibrium)-lorentz_equilibrium+maxwellian_equilibrium)))

# Check the actual nonlinear constrained kernel at a homogeneous finite-amplitude state.
print('Compile and evaluate the homogeneous constrained collision RHS.',flush=True)
sm_start = perf_counter()
sm_grid = uniform_grid(np.arange(3)*2*np.pi/3, np.linspace(-4,4,7),
                       np.linspace(.05,8,5), model='sm_local_nonlinear')
sm_u = np.broadcast_to(np.linspace(-4,4,7)[None,:,None],sm_grid.shape)
sm_vp = np.broadcast_to(np.sqrt(2*np.linspace(.05,8,5))[None,None,:],sm_grid.shape)
sm_f = jnp.asarray((2*np.pi*gaussian(sm_u,sm_vp)).ravel())
sm_rhs = np.asarray(jax.jit(lambda f: nonlinear_rhs(sm_grid,f))(sm_f))
sm_compile_check_s = perf_counter()-sm_start
checks = {
    'landau_rate_relative_error':float(np.linalg.norm(landau_check['covariance_rate']-landau_rate)/np.linalg.norm(landau_rate)),
    'sm_rhs_absolute':float(np.max(abs(sm_rhs))),
    'number_error':float(np.max(abs(curves['number']-1))),
    'momentum_absolute':float(np.max(abs(curves['momentum']))),
    'energy_error':float(np.max(abs(curves['energy']-1.5))),
    'matched_anisotropy_error':float(np.max(abs(curves['anisotropy'][1:]/.45-np.exp(-TAU)))),
    'lorentz_speed_fourth_error':float(np.max(abs(curves['speed_fourth'][1]-15.27))),
    'lorentz_mu_second_error':float(np.max(abs(curves['mu_second'][1]-(2.036+(.5742857142857143)*np.exp(-TAU)+(.034714285714285714)*np.exp(-10*TAU/3))))),
    'dougherty_mu_second_error':float(np.max(abs(curves['mu_second'][2]-2*(1+.15*np.exp(-TAU))**2))),
    'dougherty_entropy_error':float(np.max(abs(curves['relative_entropy'][2]+.5*np.log((1+.15*np.exp(-TAU))**2*(1-.3*np.exp(-TAU)))))),
    'initial_mu_marginal_error':float(np.max(abs(marginal[:,0]-np.exp(-MU_PLOT/1.15)/1.15))),
    'largest_entropy_increase':float(np.max(np.diff(curves['relative_entropy'],axis=1))) }
if max(checks.values()) > TOLERANCE:
    raise RuntimeError(f'Common comparison failed its declared tolerance: {checks}')
for direction in ['quadrature','tail','angular_degree','angular_projection']:
    rows = [r for r in refinements if r['direction']==direction]
    if any(row['status']!='passed' for row in rows[-2:]):
        raise RuntimeError(f'Unresolved {direction} refinement: {rows}')

fig, axes = plt.subplots(2,2,figsize=(10,7),layout='constrained')
for k,label in enumerate(LABELS):
    style = '--' if k==2 else '-'
    axes[0,0].plot(TAU,curves['anisotropy'][k]/.45,style,color=COLORS[k],label=label)
    axes[0,1].plot(TAU,curves['mu_second'][k],style,color=COLORS[k])
    axes[1,0].semilogy(TAU,curves['relative_entropy'][k],style,color=COLORS[k])
    axes[1,1].plot(MU_PLOT,marginal[k,-1]/np.exp(-MU_PLOT)-1,style,color=COLORS[k])
axes[0,0].set(xlabel=r'$\tau=\alpha t$',ylabel=r'$(T_\perp-T_\parallel)/0.45$',title='Matched pressure decay hides different distributions')
axes[0,0].legend(fontsize=8)
axes[0,1].axhline(2,color=COLORS[2],ls=':',label='Dougherty equilibrium: 2')
axes[0,1].axhline(2.036,color=COLORS[1],ls=':',label='Lorentz equilibrium: 2.036')
axes[0,1].set(xlabel=r'$\tau$',ylabel=r'$\langle\mu^2\rangle$',title='The magnetic-moment distributions separate')
axes[0,1].legend(fontsize=8)
axes[1,0].set(xlabel=r'$\tau$',ylabel=r'$H(F\,|\,F_M)$',title='Relative entropy to the energy-matched Maxwellian')
axes[1,0].axhline(lorentz_entropy_floor,color=COLORS[1],ls=':',label='Lorentz equilibrium')
axes[1,0].legend(fontsize=8)
axes[1,1].axhline(0,color='gray',lw=.8)
axes[1,1].set(xlabel=r'$\mu$',ylabel=r'$G(\mu)/\exp(-\mu)-1$',title=r'Magnetic-moment marginal at $\tau=3$')
fig.suptitle('One initial Gaussian, three collision constraints',fontweight='bold')
fig.savefig(OUTPUT/'comparison.png',dpi=180)
fig.savefig(OUTPUT/'comparison.svg',metadata={'Creator':None,'Date':None})
plt.close(fig)

fig=plt.figure(figsize=(11,5.8))
fig.text(.04,.94,'Collision operators used in the homogeneous comparison',fontsize=17,fontweight='bold')
fig.text(.04,.84,r'$F_0(\mathbf{v})=\dfrac{\exp[-(v_x^2+v_y^2)/(2a)-v_z^2/(2c)]}{(2\pi)^{3/2}a\sqrt{c}},\qquad a=1.15,\quad c=0.7$',fontsize=15)
equations=[
    ('Constrained local kernel',r'$C_{\rm SM}[F_0]=0$','A spatially homogeneous distribution is stationary in this uniform-field model.'),
    ('Constant-frequency Lorentz',r'$C_{\rm L}[F]=\dfrac{\nu_L}{2}\,\partial_\xi[(1-\xi^2)\partial_\xi F],\qquad \xi=v_z/|\mathbf{v}|$','Scattering changes direction and preserves each speed-shell population.'),
    ('Conserving Dougherty',r'$C_{\rm D}[F]=\nu_D\,\nabla_v\!\cdot[(\mathbf{v}-\mathbf{U})F+\theta\nabla_v F]$','The drift U and temperature θ are the distribution’s own conserved moments.'),
    ('Physical 3V Coulomb Landau',r'$C_{\rm La}[F]=\Gamma\nabla_v\!\cdot\!\int U(\mathbf{w})[F^{\prime}\nabla_vF-F\nabla_{v^{\prime}}F^{\prime}]\,d^3v^{\prime}$',
     r'$\mathbf{w}=\mathbf{v}-\mathbf{v}^{\prime},\quad U(\mathbf{w})=(I-\widehat{\mathbf{w}}\widehat{\mathbf{w}}^{T})/|\mathbf{w}|$')]
for y,(name,equation,description) in zip([.68,.51,.34,.17],equations):
    fig.text(.04,y,name,fontsize=11,fontweight='bold',color='#253447')
    fig.text(.31,y,equation,fontsize=13)
    fig.text(.31,y-.065,description,fontsize=10)
fig.savefig(OUTPUT/'equations.svg',bbox_inches='tight',metadata={'Creator':None,'Date':None})
fig.savefig(OUTPUT/'equations.png',bbox_inches='tight',dpi=170)
plt.close(fig)

fm = gaussian(up,vp,np.ones(3))
fig, axes = plt.subplots(1,3,figsize=(12,3.7))
fig.subplots_adjust(left=.065,right=.89,bottom=.23,top=.78,wspace=.26)
artists=[]
for k,ax in enumerate(axes):
    artist=ax.pcolormesh(U_PLOT,VPERP_PLOT,maps[k,0]/fm-1,cmap='RdBu_r',vmin=-.5,vmax=.5,shading='nearest',rasterized=True)
    ax.contour(up,vp,np.log10(fm),levels=[-5,-4,-3,-2],colors='#34404A',linewidths=.6,linestyles=':')
    ax.set(xlabel=r'parallel velocity $u$',ylabel=r'perpendicular speed $v_\perp$',title=LABELS[k],aspect='equal')
    artists.append(artist)
cax=fig.add_axes([.92,.25,.014,.45])
fig.colorbar(artists[0],cax=cax,extend='both',label=r'$F/F_M-1$')
clock=fig.suptitle('',fontsize=15,fontweight='bold')
fig.text(.5,.09,'Red: excess over the Maxwellian. Blue: deficit. Dotted contours: Maxwellian density.',ha='center',fontsize=10)
fig.text(.5,.035,'Normalized velocities; fixed color scale saturates beyond ±0.5. Constant Lorentz and Dougherty rates match initial Landau anisotropy decay.',ha='center',fontsize=9)


def frame(index):
    for k,artist in enumerate(artists):artist.set_array(maps[k,index]/fm-1)
    clock.set_text(f'Same initial distribution, different relaxation  |  τ = {TAU[index]:.2f}')
    return [*artists,clock]


with progress('Render distribution values at the analytic mode times'):
    frame(len(TAU)-1)
    fig.savefig(OUTPUT/'velocity_poster.png',dpi=180)
    animation=FuncAnimation(fig,frame,frames=len(TAU),interval=1000/FPS,blit=False)
    animation.save(OUTPUT/'comparison.gif',writer=PillowWriter(fps=FPS),dpi=100)
    if writers.is_available('ffmpeg'):
        animation.save(OUTPUT/'comparison.mp4',writer=FFMpegWriter(fps=FPS,codec='libx264',extra_args=['-pix_fmt','yuv420p']),dpi=100)
plt.close(fig)
np.savez_compressed(OUTPUT/'arrays.npz',tau=TAU,time=TIMES,mu=MU_PLOT,
    marginal=marginal,u_plot=U_PLOT,vperp_plot=VPERP_PLOT,density_maps=maps,**curves)
metadata=run_metadata({'initial_covariance':A.tolist(),'density':1.,'mass':MASS,'field':FIELD,
    'rates':rates,'tau':TAU.tolist(),'quadrature_order':ORDER,'radial_extent':RADIUS,
    'angular_degree':DEGREE,'projection_order':ANGULAR_ORDER,'mu_plot':MU_PLOT.tolist(),
    'plot_velocity_bounds':[-4.,4.,0.,4.],'tolerance':TOLERANCE},model='homogeneous constrained, Lorentz and Dougherty comparison',
    boundary='Infinite velocity domain analytically; independent radial and parallel tail checks')
metadata.update(status='passed',checks=checks,refinements=refinements,extended_tail_probe=extended_tail_probe,
    lorentz_equilibrium_entropy=lorentz_entropy_floor,
    scope='Three-model evolution of a shared gyrotropic Gaussian: exact modal time factors, converged finite angular reconstruction and velocity quadrature. Landau supplies only an independently checked instantaneous rate. Rate matching is a comparison convention, not physical calibration.',
    landau_initial_covariance_rate=landau_rate.tolist(),sm_compile_check_s=sm_compile_check_s,
    wall_s=perf_counter()-started,wall_scope='Arithmetic, checks, compilation and rendering; excludes imports and final metadata writing.',
    movie={'frames':len(TAU),'fps':FPS,'duration_s':len(TAU)/FPS,'time_rule':'Analytic mode-time factors and finite angular reconstruction at each declared time; no interpolation of kinetic solver states.'},
    outputs={p.name:sha256(p.read_bytes()).hexdigest() for p in sorted(OUTPUT.iterdir()) if p.name!='summary.json' and p.suffix in ['.png','.svg','.gif','.mp4','.npz']})
(OUTPUT/'summary.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(f'Common comparison passed; initial Landau anisotropy rate={alpha:.12g}; elapsed={metadata["wall_s"]:.2f}s',flush=True)
