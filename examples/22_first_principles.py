"""Reader's guide: orbit, marginal, mobility, entropy and saved relaxation."""
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
NONUNIFORM_SOURCE = 'results/nonuniform_spatial_blocks/summary.json'
NONUNIFORM_SHA256 = '8826a377f7d6a8d363f6a49236e205de9c42f24c9e21a6b45ea78a2b92b773f4'
NONUNIFORM_PRODUCER = '93bf752da33a31ec158a229f87eaa285aeb5c58b'
FIELD_COLORS = {'mirror':TEAL, 'dipole':ORANGE, 'nonaxisymmetric':'#79669A'}
PAIR_EQUATIONS = [
    r'$\Delta_{ij}h=A_i h-A_j h,\qquad\xi_{ij}=\Delta_{ij}E\ne0$',
    r'$P_{ij}=I-\dfrac{\xi_{ij}\xi_{ij}^{T}}{|\xi_{ij}|^2}$',
    r'$Q_{ij}=P_{ij}\,\mathrm{diag}(1,1,1,0,0)\,P_{ij}$',
    r'$h^{T}K(f)k=D\sum_{i<j}\omega_{ij}f_i f_j'
    r'(\Delta_{ij}h)^{T}Q_{ij}\Delta_{ij}k$']
ENTROPY_EQUATIONS = [
    r'$n=w\odot f,\quad\ell=\log f,\quad\dfrac{dn}{dt}=-K(f)\ell$',
    r'$K(f)\mathbf{1}=K(f)E=K(f)\chi_a=0$',
    r'$S=-\sum_i n_i\log f_i,\qquad\dfrac{dS}{dt}=\ell^{T}K(f)\ell\geq0$',
    r'$R(\ell)=w\odot(\exp\ell-f_{\rm old})+\Delta t\,K_{\rm old}\ell=0$',
    r'$H(\ell)=\mathrm{diag}(w\odot\exp\ell)+\Delta t\,K_{\rm old}$']
plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False,
    'axes.labelcolor':INK, 'text.color':INK, 'figure.facecolor':'white',
    'savefig.facecolor':'white', 'svg.fonttype':'path'})
inputs = {'orbit':{'mass':MASS,'charge':CHARGE,'B':B,'u':U,'v_perp':V_PERP,
    'times':TIME.tolist(),'potential':0.,'initial_gyro_angle':0.},
    'marginal_illustration':{'mu_bins':MU_BINS.tolist(),'population_A':POPULATION_A.tolist(),
    'population_B':POPULATION_B.tolist(),'same_parallel_velocity_distribution':True},
    'uniform_mode_source':'results/visual_summary/evolution.npz',
    'nonuniform_source':NONUNIFORM_SOURCE,'nonuniform_source_sha256':NONUNIFORM_SHA256,
    'nonuniform_producer_commit':NONUNIFORM_PRODUCER,
    'nonuniform_selection':{'fields':list(FIELD_COLORS),'case':'mu13','nx':5,'nu':25,
        'nmu':13,'nodes':40625,'dt':.005,'final_time':.02,'collision_strength':.1},
    'equation_figures':{'pair_mobility':PAIR_EQUATIONS,'entropy_step':ENTROPY_EQUATIONS},
    'equation_normalization':'Common normalized eta chart; constant density scale f_ref=1.',
    'equation_pair_domain':'Nonzero xi for each displayed projected pair; positive pair weights and densities; unordered pairs counted once.',
    'equation_constraints':'E is the same discrete energy used in xi; chi_a is a resolved magnetic-moment-bin indicator.'}
metadata = run_metadata(inputs, model='Analytical illustrations and previously checked uniform/nonuniform collision data',
    boundary='Uniform unbounded orbit; schematic magnetic-moment bins; periodic uniform mode; natural no-flux collision-only Cartesian pilot boxes',units='normalized')
metadata['example_sha256'] = sha256(Path(__file__).read_bytes()).hexdigest()
metadata['equation_scope'] = 'Displayed definitions of the implemented discrete pair mobility and nodal entropy ODE/lagged root; no new solve, convergence, rate or conditioning claim.'
metadata['equation_reference'] = {'path':'notes/implementation.tex',
    'sections':['Linear weak discretization and SOLVAX',
                'Lagged mobility and an implicit entropy-variable step'],
    'pair_convention':'The ordered weak-form factor 1/2 cancels the two orientations; the displayed unordered sum has no extra 1/2.'}
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
fig.text(.49,.52,'X: position     u: signed velocity along the field',fontsize=12)
fig.text(.49,.35,r'$\mu=\dfrac{m v_\perp^2}{2B}$',fontsize=21)
fig.text(.49,.20,'μ labels the perpendicular orbit energy per field strength.',fontsize=10)
fig.text(.49,.055,r'$E=\frac{1}{2}mu^2+\mu B$'+'   (zero electric potential)',fontsize=14)
save(fig,'orbit_and_state')

fig = plt.figure(figsize=(9,3.2))
fig.text(.04,.87,'Fixed vacuum field: motion transports f; collisions change its shape.',fontsize=13,fontweight='bold')
fig.text(.5,.59,r'$\frac{\partial f}{\partial t}+\dot{\mathbf{X}}\cdot\nabla_{\mathbf{X}} f'
    r'+\dot u\,\frac{\partial f}{\partial u}=C[f]$',ha='center',fontsize=26)
fig.text(.04,.32,'Left: guiding-center motion in the prescribed field.',fontsize=12)
fig.text(.04,.18,'Right: the chosen collision model. Many runs isolate this term.',fontsize=12)
fig.text(.04,.04,r'Volume element: $d\Gamma=B\,d^3\!X\,du\,d\mu$  (constant normalization absorbed into f).',fontsize=12)
save(fig,'kinetic_equation')

fig = plt.figure(figsize=(10.4,5.5))
fig.text(.04,.92,'Discrete pair mobility in the common normalized η chart',fontsize=14,fontweight='bold')
for y, equation, size in zip((.77,.53,.31,.12),PAIR_EQUATIONS,(23,23,23,20)):
    fig.text(.5,y,equation,ha='center',fontsize=size)
fig.text(.04,.025,'Positive pair weights; each unordered pair counted once. A uses the same discrete derivative for h and E.',fontsize=12)
save(fig,'pair_mobility')

fig = plt.figure(figsize=(10.4,6.2))
fig.text(.04,.94,'Conservation and entropy in the nodal collision ODE',fontsize=14,fontweight='bold')
for y, equation in zip((.81,.66,.51),ENTROPY_EQUATIONS[:3]):
    fig.text(.5,y,equation,ha='center',fontsize=22)
fig.text(.04,.39,'Lagged entropy-variable step',fontsize=14,fontweight='bold',color=TEAL)
fig.text(.5,.27,ENTROPY_EQUATIONS[3],ha='center',fontsize=21)
fig.text(.5,.12,ENTROPY_EQUATIONS[4],ha='center',fontsize=22)
fig.text(.04,.018,r'$K_{\rm old}=K(f_{\rm old})$',fontsize=18)
fig.text(.43,.025,'E: discrete energy; χₐ: resolved μ-bin indicator.',fontsize=12)
save(fig,'entropy_step')

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
ax.text(.02,.87,'Closed constrained dynamics preserve every bin.',fontsize=12,fontweight='bold')
ax.text(.02,.62,r'$G(\mu,t)=\int B f\,d^3\!X\,du$',fontsize=23)
ax.text(.02,.42,r'$G(\mu,t)=G(\mu,0)$',fontsize=22,color=TEAL)
ax.text(.02,.22,'A cannot evolve into B under this constraint,\neven though their particle counts and means agree.',fontsize=11)
ax.text(.02,.025,'Two populations with equal number and mean μ.\nTheir full magnetic-moment distributions differ.',fontsize=10)
save(fig,'full_marginal')

source=ROOT/inputs['uniform_mode_source'];raw=source.read_bytes()
data=np.load(source);tau=data['tau'];weights=data['masses'];x=data['x']
# The first x node is cos(kx)=1. Weighted averaging extracts the density mode.
local=data['local'][:,:,0];finite=data['finite_range'][:,:,0]
density_local=local@weights/weights.sum();density_finite=finite@weights/weights.sum()
neutral_local=local-density_local[:,None]
neutral_amplitude=np.linalg.norm(neutral_local,axis=1)/np.linalg.norm(neutral_local[0])
neutral_finite=finite-density_finite[:,None]
neutral_finite_amplitude=np.linalg.norm(neutral_finite,axis=1)/np.linalg.norm(neutral_finite[0])
source_meta_path=ROOT/'results/visual_summary/metadata.json'
source_meta=json.loads(source_meta_path.read_text());source_inputs=source_meta['inputs']
range_rate=1-np.exp(-source_inputs['width']**2*source_inputs['mode']**2/2)
mode_error=float(max(np.max(abs(density_local/density_local[0]-1)),
    np.max(abs(density_finite/density_finite[0]-np.exp(-range_rate*tau))),
    np.max(abs(neutral_amplitude-np.exp(-tau))),
    np.max(abs(neutral_finite_amplitude-np.exp(-tau)))))
if mode_error>1e-11:raise RuntimeError('Saved uniform modes do not match the exact solution')
fig,axes=plt.subplots(1,2,figsize=(9,4.7),layout='constrained')
ax=axes[0]
ax.plot(x,local[0,4]*np.cos(x),color=INK,lw=2,label='initial, selected velocity node')
ax.plot(x,density_local[-1]*np.cos(x),color=TEAL,lw=2,label='local density part: survives')
ax.plot(x,(local[0,4]-density_local[0])*np.cos(x),color=ORANGE,lw=2,label='remaining part: decays')
ax.axhline(0,color='0.7',lw=.7);ax.set(xlabel='perpendicular position x',ylabel=r'relative perturbation $\delta f/f_0$',title='Split a perturbation into two parts',xticks=[0,np.pi,2*np.pi],xticklabels=['0','π','2π'])
ax.legend(frameon=False,fontsize=9,loc='lower center',bbox_to_anchor=(.5,1.02))
ax.set_title('Split a perturbation into two parts',pad=66)
ax=axes[1]
ax.plot(tau,density_local/density_local[0],color=TEAL,lw=2,label='local density pattern')
ax.plot(tau,density_finite/density_finite[0],color=TEAL,lw=2,ls='--',label='density with finite range')
ax.plot(tau,neutral_amplitude,color=ORANGE,lw=2,label='zero-density part, both models')
ax.set(xlabel=r'model time $\tau=\lambda t$',ylabel='amplitude / initial',ylim=(0,1.15),title='Exact mode amplitudes')
ax.legend(frameon=False,fontsize=9,loc='lower center',bbox_to_anchor=(.5,1.02))
ax.set_title('Exact mode amplitudes',pad=66)
save(fig,'mode_decomposition')

fig=plt.figure(figsize=(9,3.25))
fig.text(.04,.86,'Collision-only, uniform B and homogeneous f₀: Eq. (181)',fontsize=13,fontweight='bold')
fig.text(.5,.60,r'$\delta f=f_0\frac{\delta n}{n_0}+g,\qquad\int B g\,du\,d\mu=0$',ha='center',fontsize=23)
fig.text(.5,.34,r'$C_L[\delta f]=\frac{D n_0}{(qB)^2}\nabla_\perp^2g,\qquad'
    r'g_k(t)=g_k(0)e^{-\lambda t}$',ha='center',fontsize=22)
fig.text(.5,.075,r'$\lambda=\frac{D n_0 k_\perp^2}{(qB)^2},\qquad C_L[f_0\delta n/n_0]=0$',ha='center',fontsize=22)
save(fig,'oracle_decomposition')

pilot_path=ROOT/NONUNIFORM_SOURCE
pilot_raw=pilot_path.read_bytes()
if sha256(pilot_raw).hexdigest()!=NONUNIFORM_SHA256:
    raise RuntimeError('The immutable nonuniform pilot summary changed')
pilot=json.loads(pilot_raw)
rows=pilot['rows']
if len(rows)!=3 or {row['field'] for row in rows}!=set(FIELD_COLORS):
    raise RuntimeError('Expected exactly the three completed nonuniform pilot fields')
curves={}
for row in rows:
    origin=pilot['provenance_runs'][row['provenance_id']]
    if origin['commit']!=NONUNIFORM_PRODUCER or row['status']!='passed':
        raise RuntimeError('Nonuniform pilot provenance or completion changed')
    if any(row[key]!=value for key,value in {'case':'mu13','nx':5,'nu':25,
            'nmu':13,'nodes':40625,'dt':.005}.items()):
        raise RuntimeError('Nonuniform pilot grid or timestep differs')
    if origin['inputs']['collision_strength']!=.1 or origin['inputs']['final_time']!=.02:
        raise RuntimeError('Nonuniform pilot physical inputs differ')
    history=row['history']
    times=np.array([step['time'] for step in history])
    gains=np.array([step['entropy_change'] for step in history])
    bin_errors=np.array([step['relative_marginal_bin_errors'] for step in history])
    initial_entropy=row['number']*row['initial_relative_entropy_per_particle']
    if len(history)!=4 or bin_errors.shape!=(4,13) or not np.allclose(times,
            np.arange(1,5)*.005,rtol=0.,atol=1e-15):
        raise RuntimeError('Nonuniform pilot accepted-time or magnetic-moment-bin history differs')
    if not all(np.all(np.isfinite(a)) for a in (times,gains,bin_errors)) or not np.isfinite(initial_entropy) or initial_entropy<=0:
        raise RuntimeError('Nonfinite nonuniform pilot data or nonpositive initial relative entropy')
    entropy_ratio=np.r_[1.,1.-np.cumsum(gains)/initial_entropy]
    endpoint_error=float(abs(entropy_ratio[-1]-(1.-row['relative_entropy_decrease_fraction'])))
    if not np.all(np.isfinite(entropy_ratio)) or not np.isfinite(endpoint_error) or np.min(entropy_ratio)<-1e-12 or endpoint_error>1e-12:
        raise RuntimeError('Saved entropy history does not reproduce the reported relative-entropy decrease')
    maximum_bin_error=np.max(abs(bin_errors),axis=1)
    curves[row['field']]={'times':np.r_[0.,times].tolist(),
        'relative_entropy_ratio':entropy_ratio.tolist(),
        'accepted_times':times.tolist(),'maximum_relative_mu_bin_error':maximum_bin_error.tolist(),
        'reported_endpoint_identity_error':endpoint_error,'provenance_id':row['provenance_id']}
fig,axes=plt.subplots(1,2,figsize=(9.5,4.6),layout='constrained')
for field,color in FIELD_COLORS.items():
    curve=curves[field]
    axes[0].plot(curve['times'],curve['relative_entropy_ratio'],'o-',color=color,
        lw=2,ms=4,label=field)
    # Zero errors need no logarithmic placeholder; t=0 is omitted by construction.
    error=np.asarray(curve['maximum_relative_mu_bin_error'])
    positive=error>0
    axes[1].semilogy(np.asarray(curve['accepted_times'])[positive],error[positive],
        'o-',color=color,lw=2,ms=4,label=field)
axes[0].set(xlabel='time',ylabel=r'relative entropy $H(t)/H(0)$',ylim=(0,1.05),
    title='Relative entropy decreases')
axes[1].set(xlabel='accepted time',ylabel='max relative magnetic-moment-bin error',
    title='All 13 bin populations are retained')
for ax in axes:ax.set_xticks([0.,.01,.02]);ax.grid(alpha=.15)
axes[0].legend(frameon=False,fontsize=10)
fig.suptitle('Collision-only relaxation',fontsize=15,fontweight='bold')
fig.supxlabel('Δt = 0.005; T = 0.02; 40,625 nodes; D = 0.1. Full refinement remains unfinished.',fontsize=10)
save(fig,'nonuniform_relaxation')
metadata['nonuniform_saved_data']={'path':NONUNIFORM_SOURCE,'sha256':NONUNIFORM_SHA256,
    'producer_commit':NONUNIFORM_PRODUCER,'inputs':inputs['nonuniform_selection'],
    'curves':curves,'assertions':['exact three-field selection and original producer',
        'four accepted times, 13 finite bin errors per time',
        'reconstructed final entropy ratio agrees with reported fraction within 1e-12'],
    'relative_entropy_definition':'Generalized KL to the mass-normalized stationary exp(-E-.2mu) reference; its logarithm is a discrete invariant, so H(t)=H(0)-[S(t)-S(0)]. The reference is not asserted reachable.',
    'scope':'Saved collision-only pilot histories, nodal/quadrature populations; no new solve, full refinement or original timing claim.'}

metadata.update({'status':'passed','checks_wall_s':perf_counter()-start,
    'cost_scope':'Analytic checks, saved-data comparisons and figure rendering; excludes imports and initial metadata.',
    'lorentz_force_max_error':lorentz_error,'mu_orbit_range':float(np.ptp(mu)),
    'energy_orbit_range':float(np.ptp(energy)),'schematic_number_difference':number_error,
    'schematic_mean_mu_difference':mean_error,'uniform_saved_modes_max_error':mode_error,
    'sources':{str(source.relative_to(ROOT)):{'sha256':sha256(raw).hexdigest()},
    str(source_meta_path.relative_to(ROOT)):{'sha256':sha256(source_meta_path.read_bytes()).hexdigest(),
        'producer_commit':source_meta['commit']}},
    'limitations':'Orbit is exact in a uniform field; adiabatic magnetic-moment conservation in varying fields requires scale separation. Histogram is a constraints illustration, not dynamics. Mode curves reuse the checked uniform linear surrogate. Nonuniform curves reuse completed collision-only pilots; full refinement is unfinished and the stationary reference is not asserted reachable. D is prescribed, not a physical rate.'})
metadata['figures']={p.name:sha256(p.read_bytes()).hexdigest() for p in sorted(OUTPUT.glob('*')) if p.suffix in ('.svg','.png')}
(OUTPUT/'summary.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
print('Saved first-principles figures; maximum mode error',mode_error,flush=True)
