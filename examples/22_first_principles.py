"""Reader's guide: exact orbit, conserved marginal, and checked uniform modes."""
from pathlib import Path
from hashlib import sha256
import json
from time import perf_counter
import jax
jax.config.update('jax_enable_x64', True)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sato_morrison.reference import run_metadata

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'results'/'first_principles'
OUTPUT.mkdir(parents=True, exist_ok=True)
MASS, CHARGE, B, U, V_PERP = 1., 1., 1., .4, .8
TIME = np.linspace(0., 4*np.pi, 401)
MU_BINS = np.array([.5, 1.5, 2.5])
POPULATION_A = np.array([.25, .50, .25])
POPULATION_B = np.array([.40, .20, .40])
TEAL, ORANGE, INK, FAINT = '#147D92', '#C66B28', '#253447', '#DCE8EB'
plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False,
    'axes.labelcolor':INK, 'text.color':INK, 'figure.facecolor':'white',
    'savefig.facecolor':'white', 'svg.fonttype':'path'})
inputs = {'orbit':{'mass':MASS,'charge':CHARGE,'B':B,'u':U,'v_perp':V_PERP,
    'times':TIME.tolist(),'potential':0.,'initial_gyro_angle':0.},
    'marginal_illustration':{'mu_bins':MU_BINS.tolist(),'population_A':POPULATION_A.tolist(),
    'population_B':POPULATION_B.tolist(),'same_parallel_velocity_distribution':True},
    'uniform_mode_source':'results/visual_summary/evolution.npz'}
metadata = run_metadata(inputs, model='Analytical educational illustrations and previously checked uniform-mode data',
    boundary='Uniform unbounded orbit; schematic magnetic-moment bins; periodic uniform Fourier mode',units='normalized')
metadata['example_sha256'] = sha256(Path(__file__).read_bytes()).hexdigest()
print('Build first-principles figures from an exact uniform orbit and saved operator evolution.',flush=True)
print('No new collision solve or compilation; verify every illustrated numerical identity.',flush=True)
start = perf_counter()


def save(fig, name):
    fig.savefig(OUTPUT/(name+'.svg'),bbox_inches='tight',pad_inches=.15,
        metadata={'Creator':None,'Date':None})
    fig.savefig(OUTPUT/(name+'.png'),dpi=180,bbox_inches='tight',pad_inches=.15)
    plt.close(fig)


# Exact Lorentz orbit, m dv/dt = q v x B; its center travels along B.
omega = CHARGE*B/MASS
radius = V_PERP/omega
orbit = np.column_stack((radius*np.sin(omega*TIME),radius*np.cos(omega*TIME),U*TIME))
velocity = np.column_stack((V_PERP*np.cos(omega*TIME),-V_PERP*np.sin(omega*TIME),np.full(TIME.size,U)))
acceleration = np.column_stack((-V_PERP*omega*np.sin(omega*TIME),-V_PERP*omega*np.cos(omega*TIME),np.zeros(TIME.size)))
lorentz_error = float(np.max(np.abs(MASS*acceleration-CHARGE*np.cross(velocity,[0.,0.,B]))))
mu = MASS*np.sum(velocity[:,:2]**2,axis=1)/(2*B)
energy = MASS*np.sum(velocity**2,axis=1)/2
if lorentz_error>1e-13 or np.ptp(mu)>1e-13 or np.ptp(energy)>1e-13:
    raise RuntimeError('Exact orbit check failed')
fig = plt.figure(figsize=(9,3.7))
ax = fig.add_axes([.00,.03,.43,.88],projection='3d')
ax.plot(*orbit.T,color=TEAL,lw=2)
ax.plot([0,0],[0,0],[0,U*TIME[-1]],'--',color=INK,lw=2)
ax.quiver(1.15,0,0,0,0,1.3,color=ORANGE,arrow_length_ratio=.2)
ax.text(1.15,0,1.6,'B',color=ORANGE,fontsize=13)
ax.set_box_aspect((1,1,2));ax.view_init(elev=15,azim=-55);ax.set_axis_off()
ax.set_title('1  Average the rapid orbit',pad=-5,fontsize=13,fontweight='bold')
fig.text(.015,.015,'Solid: particle orbit     Dashed: guiding center',fontsize=10)
fig.text(.49,.83,'2  Track a distribution of orbit centers',fontsize=13,fontweight='bold')
fig.text(.49,.66,r'$f(\mathbf{X},u,\mu,t)$',fontsize=24)
fig.text(.49,.52,'X: position     u: speed along the field',fontsize=12)
fig.text(.49,.35,r'$\mu=\dfrac{m v_\perp^2}{2B}$',fontsize=21)
fig.text(.49,.20,'μ labels the perpendicular orbit energy per field strength.',fontsize=10)
fig.text(.49,.055,r'$E=\frac{1}{2}mu^2+\mu B$'+'   (zero electric potential)',fontsize=14)
save(fig,'orbit_and_state')

fig = plt.figure(figsize=(9,3.2))
fig.text(.04,.87,'Motion transports the distribution; collisions change its shape.',fontsize=13,fontweight='bold')
fig.text(.5,.59,r'$\frac{\partial f}{\partial t}+\dot{\mathbf{X}}\cdot\nabla_{\mathbf{X}} f'
    r'+\dot u\,\frac{\partial f}{\partial u}=C[f]$',ha='center',fontsize=26)
fig.text(.04,.32,'Left: guiding-center motion in the prescribed field.',fontsize=12)
fig.text(.04,.18,'Right: the chosen collision model. Many runs isolate this term.',fontsize=12)
fig.text(.04,.04,r'Volume element: $d\Gamma=B\,d^3\!X\,du\,d\mu$  (constant normalization absorbed into f).',fontsize=12)
save(fig,'kinetic_equation')

number_error = float(abs(POPULATION_A.sum()-POPULATION_B.sum()))
mean_error = float(abs(MU_BINS@(POPULATION_A-POPULATION_B)))
if number_error>1e-14 or mean_error>1e-14 or np.allclose(POPULATION_A,POPULATION_B):
    raise RuntimeError('Same-mean/different-marginal illustration failed')
fig,axes = plt.subplots(1,2,figsize=(9,3.8),layout='constrained',gridspec_kw={'width_ratios':[1,1.18]})
ax=axes[0];ax.bar(MU_BINS-.09,POPULATION_A,width=.18,color=TEAL,label='A')
ax.bar(MU_BINS+.09,POPULATION_B,width=.18,color=ORANGE,label='B')
ax.set(xlabel=r'magnetic-moment bin $\mu$',ylabel='fraction of particles',xticks=MU_BINS,ylim=(0,.65),title='Same mean; different full distribution')
ax.legend(frameon=False,ncol=2,loc='upper right')
ax.text(.04,.89,r'$N_A=N_B=1$'+'\n'+r'$\langle\mu\rangle_A=\langle\mu\rangle_B=1.5$',transform=ax.transAxes,fontsize=10)
ax=axes[1];ax.axis('off')
ax.text(.02,.87,'The constrained model preserves every bin.',fontsize=12,fontweight='bold')
ax.text(.02,.62,r'$G(\mu,t)=\int B f\,d^3\!X\,du$',fontsize=23)
ax.text(.02,.42,r'$G(\mu,t)=G(\mu,0)$',fontsize=22,color=TEAL)
ax.text(.02,.22,'A cannot evolve into B under this constraint,\neven though their particle counts and means agree.',fontsize=11)
ax.text(.02,.025,'Schematic populations, not a simulated trajectory.\nThe full profile is a stronger constraint than one mean.',fontsize=10)
save(fig,'full_marginal')

source=ROOT/inputs['uniform_mode_source'];raw=source.read_bytes()
data=np.load(source);tau=data['tau'];weights=data['masses'];x=data['x']
# The first x node is cos(kx)=1. Weighted averaging extracts the density mode.
local=data['local'][:,:,0];finite=data['finite_range'][:,:,0]
density_local=local@weights/weights.sum();density_finite=finite@weights/weights.sum()
neutral_local=local-density_local[:,None]
neutral_amplitude=np.linalg.norm(neutral_local,axis=1)/np.linalg.norm(neutral_local[0])
source_meta_path=ROOT/'results/visual_summary/metadata.json'
source_meta=json.loads(source_meta_path.read_text());source_inputs=source_meta['inputs']
range_rate=1-np.exp(-source_inputs['width']**2*source_inputs['mode']**2/2)
mode_error=float(max(np.max(abs(density_local/density_local[0]-1)),
    np.max(abs(density_finite/density_finite[0]-np.exp(-range_rate*tau))),
    np.max(abs(neutral_amplitude-np.exp(-tau)))))
if mode_error>1e-11:raise RuntimeError('Saved uniform modes do not match the exact solution')
fig,axes=plt.subplots(1,2,figsize=(9,3.7),layout='constrained')
ax=axes[0]
ax.plot(x,local[0,4]*np.cos(x),color=INK,lw=2,label='initial, selected velocity node')
ax.plot(x,density_local[-1]*np.cos(x),color=TEAL,lw=2,label='local density part: survives')
ax.plot(x,(local[0,4]-density_local[0])*np.cos(x),color=ORANGE,lw=2,label='remaining part: decays')
ax.axhline(0,color='0.7',lw=.7);ax.set(xlabel='perpendicular position x',ylabel=r'relative perturbation $\delta f/f_0$',title='Split a perturbation into two parts',xticks=[0,np.pi,2*np.pi],xticklabels=['0','π','2π'])
ax.legend(frameon=False,fontsize=9,loc='lower left')
ax=axes[1]
ax.plot(tau,density_local/density_local[0],color=TEAL,lw=2,label='local density pattern')
ax.plot(tau,density_finite/density_finite[0],color=TEAL,lw=2,ls='--',label='density with finite range')
ax.plot(tau,neutral_amplitude,color=ORANGE,lw=2,label='zero-density part, both models')
ax.set(xlabel=r'model time $\tau=\lambda t$',ylabel='amplitude / initial',ylim=(0,1.15),title='Only permitted parts can relax')
ax.legend(frameon=False,fontsize=9,loc='upper right')
save(fig,'mode_decomposition')

fig=plt.figure(figsize=(9,3.25))
fig.text(.04,.86,'Uniform field + small perturbation: an exact test of Eq. (181)',fontsize=13,fontweight='bold')
fig.text(.5,.60,r'$\delta f=f_0\frac{\delta n}{n_0}+g,\qquad\int B g\,du\,d\mu=0$',ha='center',fontsize=23)
fig.text(.5,.34,r'$C_L[\delta f]=\frac{D n_0}{(qB)^2}\nabla_\perp^2g,\qquad'
    r'g_k(t)=g_k(0)e^{-\lambda t}$',ha='center',fontsize=22)
fig.text(.5,.075,r'$\lambda=\frac{D n_0 k_\perp^2}{(qB)^2},\qquad C_L[f_0\delta n/n_0]=0$',ha='center',fontsize=22)
save(fig,'oracle_decomposition')

metadata.update({'status':'passed','checks_wall_s':perf_counter()-start,
    'cost_scope':'Analytic checks, saved-data comparisons and figure rendering; excludes imports and initial metadata.',
    'lorentz_force_max_error':lorentz_error,'mu_orbit_range':float(np.ptp(mu)),
    'energy_orbit_range':float(np.ptp(energy)),'schematic_number_difference':number_error,
    'schematic_mean_mu_difference':mean_error,'uniform_saved_modes_max_error':mode_error,
    'sources':{str(source.relative_to(ROOT)):{'sha256':sha256(raw).hexdigest()},
    str(source_meta_path.relative_to(ROOT)):{'sha256':sha256(source_meta_path.read_bytes()).hexdigest(),
        'producer_commit':source_meta['commit']}},
    'limitations':'Orbit is exact in a uniform field; adiabatic magnetic-moment conservation in varying fields requires scale separation. Histogram is a constraints illustration, not dynamics. Mode curves reuse the checked uniform linear surrogate; D is prescribed, not a physical rate.'})
metadata['figures']={p.name:sha256(p.read_bytes()).hexdigest() for p in sorted(OUTPUT.glob('*')) if p.suffix in ('.svg','.png')}
(OUTPUT/'summary.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
print('Saved first-principles figures; maximum mode error',mode_error,flush=True)
