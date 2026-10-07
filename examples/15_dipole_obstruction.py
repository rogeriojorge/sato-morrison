"""Constructive dipole flux-marginal obstruction; no time evolution is inferred."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sato_morrison.reference import gauss_interval, run_metadata

# Dipole flux psi=C R^2/r^3 and magnetic latitude lambda. The azimuthal angle
# is integrated exactly over 2pi. All parameters and quadratures are normalized.
C, BETA, GAMMA_MU = 1., 1., .2
PSI_BOUNDS, LATITUDE_BOUNDS = (.4, 1.), (-np.pi/6, np.pi/6)
UMAX, MUMAX = 4., 12.
ORDER = (32, 24, 32, 32)  # psi, latitude, parallel velocity, magnetic moment
AMPLITUDE_BOUND = .4
OUTPUT = Path(__file__).resolve().parents[1] / 'results' / 'dipole_obstruction'
OUTPUT.mkdir(parents=True, exist_ok=True)
print('Dipole flux-marginal obstruction: positive quadrature on a finite flux-coordinate '
      'domain; match N, E, every mu population and mean psi. No time solve or physical '
      'lifetime is claimed.', flush=True)


def construct(order):
    psi, wp = gauss_interval(order[0], *PSI_BOUNDS)
    latitude, wl = gauss_interval(order[1], *LATITUDE_BOUNDS)
    u, wu = gauss_interval(order[2], -UMAX, UMAX)
    mu, wm = gauss_interval(order[3], 0., MUMAX)
    pp, ll, uu, mm = np.meshgrid(psi, latitude, u, mu, indexing='ij')
    radius = C*np.cos(ll)**2/pp
    bmag = C*np.sqrt(1+3*np.sin(ll)**2)/radius**3
    # dGamma = B r^2 cos(lambda)|dr/dpsi| dpsi dlambda dtheta du dmu.
    jacobian = C*np.cos(ll)*np.sqrt(1+3*np.sin(ll)**2)/pp
    weight = (2*np.pi*wp[:,None,None,None]*wl[None,:,None,None]*
              wu[None,None,:,None]*wm[None,None,None,:]*jacobian)
    energy = uu**2/2+mm*bmag
    reference = np.exp(-BETA*energy-GAMMA_MU*mm)
    population = weight*reference
    marginal = population.sum(axis=(0,1,2))

    def center(observable):
        mean = (population*observable).sum(axis=(0,1,2))/marginal
        return observable-mean[None,None,None,:]

    re, rp, target = center(energy), center(pp), center(pp**2)
    gram = np.array([[np.sum(population*a*b) for b in (re,rp)] for a in (re,rp)])
    rhs = np.array([np.sum(population*a*target) for a in (re,rp)])
    a, b = np.linalg.solve(gram, rhs)
    h = target-a*re-b*rp
    # A conservative bound valid over the entire truncated continuous domain:
    # each conditional mean lies in that observable's global min/max interval.
    field_max = PSI_BOUNDS[1]**3/C**2*np.sqrt(1+3*np.sin(LATITUDE_BOUNDS[1])**2)/np.cos(LATITUDE_BOUNDS[1])**6
    emax = UMAX**2/2+MUMAX*field_max
    bound = PSI_BOUNDS[1]**2-PSI_BOUNDS[0]**2+abs(a)*emax+abs(b)*(PSI_BOUNDS[1]-PSI_BOUNDS[0])
    epsilon = AMPLITUDE_BOUND/bound
    plus, minus = population*(1+epsilon*h), population*(1-epsilon*h)
    number = population.sum()
    budgets = {
        'number_relative_error':abs(plus.sum()-number)/number,
        'energy_relative_error':abs(np.sum((plus-population)*energy))/np.sum(population*energy),
        'mu_marginal_relative_error':float(np.max(np.abs(plus.sum(axis=(0,1,2))-marginal))/number),
        'mean_flux_relative_error':abs(np.sum((plus-population)*pp))/np.sum(population*pp),
    }
    mismatch = np.sum((plus-minus)*pp**2)/number
    independent_identity = 2*epsilon*np.sum(population*h*h)/number
    # Density H(psi) with respect to dpsi, normalized to total population one.
    flux_reference = population.sum(axis=(1,2,3))/(wp*number)
    flux_plus = plus.sum(axis=(1,2,3))/(wp*number)
    flux_minus = minus.sum(axis=(1,2,3))/(wp*number)
    lower_bound = np.sum(wp*flux_plus*np.log(flux_plus/flux_reference))
    full_divergence = np.sum(plus*np.log(plus/population))/number
    row = {'order':list(order), 'number':float(number), 'energy':float(np.sum(population*energy)),
           'projection_coefficients':[float(a),float(b)], 'gram_condition':float(np.linalg.cond(gram)),
           'epsilon':float(epsilon), 'analytic_relative_population_lower_bound':1-AMPLITUDE_BOUND,
           'minimum_relative_population':float(min((1+epsilon*h).min(),(1-epsilon*h).min())),
           'flux_second_moment_difference':float(mismatch),
           'independent_squared_residual_identity':float(independent_identity),
           'identity_relative_error':float(abs(mismatch-independent_identity)/independent_identity),
           'flux_KL_lower_bound':float(lower_bound), 'full_initial_KL':float(full_divergence),
           **{key:float(value) for key,value in budgets.items()}}
    if max(budgets.values())>1e-12 or row['identity_relative_error']>1e-9:
        raise RuntimeError('Constraint matching or independent squared-residual identity failed')
    if not 0<lower_bound<=full_divergence or row['minimum_relative_population']<.6:
        raise RuntimeError('Positive distribution or information lower bound failed')
    return row, (psi, flux_reference, flux_plus, flux_minus)


rows = []
for label, order in [('base',ORDER), ('psi16',(16,24,32,32)), ('psi24',(24,24,32,32)),
        ('latitude12',(32,12,32,32)), ('latitude18',(32,18,32,32)),
        ('u16',(32,24,16,32)), ('u24',(32,24,24,32)),
        ('mu16',(32,24,32,16)), ('mu24',(32,24,32,24)), ('joint',(40,30,40,40))]:
    print(f'  {label}: quadrature {order}', flush=True)
    row, curves = construct(order)
    row['case'] = label
    rows.append(row)
    if label=='base':
        base_curves = curves
    print(f'    constraints={max(row[k] for k in ["number_relative_error","energy_relative_error", "mu_marginal_relative_error","mean_flux_relative_error"]):.2e}; '
          f'Δ<psi²>={row["flux_second_moment_difference"]:.8g}; '
          f'KL floor={row["flux_KL_lower_bound"]:.8g}', flush=True)

reference = rows[-1]
for row in rows:
    row['mismatch_relative_change'] = abs(row['flux_second_moment_difference']/reference['flux_second_moment_difference']-1)
    row['KL_relative_change'] = abs(row['flux_KL_lower_bound']/reference['flux_KL_lower_bound']-1)
metadata = run_metadata({'C':C, 'beta':BETA, 'gamma_mu':GAMMA_MU,
    'psi_bounds':PSI_BOUNDS, 'latitude_bounds':LATITUDE_BOUNDS, 'umax':UMAX, 'mumax':MUMAX,
    'order':ORDER, 'amplitude_bound':AMPLITUDE_BOUND},
    model='analytic local collision nullspace and dipole accessibility counterexample',
    boundary='finite-domain quadrature; theorem requires closed flux-compatible ideal boundaries; no time evolution')
metadata['rows'] = rows
metadata['scope'] = ('Numerically constructed distributions with matched discrete constraints. The continuum theorem '
    'uses conditional integrals in place of sums. The geometric obstruction is conditional on exact local '
    'collision and closed ideal boundaries, and is not a physical dipole confinement prediction.')
(OUTPUT/'summary.json').write_text(json.dumps(metadata, indent=2)+'\n')
np.savez_compressed(OUTPUT/'flux_populations.npz', psi=base_curves[0],
    reference=base_curves[1], plus=base_curves[2], minus=base_curves[3])

fig, axes = plt.subplots(1,3,figsize=(13.6,4.3),layout='constrained')
latitude = np.linspace(*LATITUDE_BOUNDS, 240)
for psi in np.linspace(*PSI_BOUNDS, 7):
    radius = C*np.cos(latitude)**2/psi
    axes[0].plot(radius*np.cos(latitude),radius*np.sin(latitude),color=plt.cm.viridis((psi-.4)/.6))
axes[0].set(xlabel='Cylindrical radius R',ylabel='Height z',title='A. Distinct conserved flux surfaces',aspect='equal')
psi, eq, plus, minus = base_curves
axes[1].plot(psi,100*(plus/eq-1),color='#087e8b',label='Positive distribution +')
axes[1].plot(psi,100*(minus/eq-1),color='#c45b28',label='Positive distribution −')
axes[1].axhline(0,color='.4',linewidth=1,linestyle='--',label='Stationary candidate')
axes[1].set(xlabel='Poloidal flux ψ',ylabel='Flux-population difference (%)',title='B. Same N, E, G(μ) and mean ψ')
axes[1].legend(fontsize=8)
axes[2].bar(['Full initial\ndistance','Conserved-flux\nlower bound'],
             [rows[0]['full_initial_KL'],rows[0]['flux_KL_lower_bound']],color=['#614e9e','#087e8b'])
axes[2].set(ylabel='Relative entropy per particle',title='C. The candidate remains inaccessible')
axes[2].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
for axis in axes:
    axis.grid(alpha=.15)
fig.savefig(OUTPUT/'obstruction.png',dpi=180)
plt.close(fig)
if max(rows[0]['mismatch_relative_change'],rows[0]['KL_relative_change'])>1e-4:
    raise RuntimeError('Final joint-refinement target unresolved; evidence saved')
print(f'Saved {OUTPUT}; constructive obstruction passed independent refinement.',flush=True)
