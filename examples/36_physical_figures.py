"""Render saved screened-pair evidence; no new encounters or fitted distributions."""
from pathlib import Path
from hashlib import sha256
from time import perf_counter
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results'/'physical_final_campaign'
FIGURE_MODE = os.environ.get('PHYSICAL_FIGURE_MODE','original')
if FIGURE_MODE not in ('original','refinement'):
    raise ValueError('PHYSICAL_FIGURE_MODE must be original or refinement')
OUTPUT = Path(os.environ.get('PHYSICAL_FIGURE_OUTPUT',str(ROOT/'results'/'physical_final_figures'/('original_campaign' if FIGURE_MODE=='original' else 'refinement')))).resolve()
REFINEMENT_SOURCE = ROOT/'results'/'physical_refinement'
REFINEMENT_HASHES = {'summary.json': 'ed07f3503406aeaa4206dae785fdaca2b6440cf87b602ebf37328d640aeaff39', 'heldout_original128.npz': 'cf66503b21da01bb592c96b0be70dc524febc56fe602fc896b0dac6df58b7add', 'heldout_fresh128.npz': 'f94306332c1da62f81fbc74de261a2701a4f583a3d67158cacb98709e20aaa82'}
SOURCE_HASHES = {
    'summary.json':'ead69cb24da925a7f7a83356092e704e83d176e846af162cf475d9a08033d6cd',
    'cases.json':'5f103c9375fc4da5a8b7d45122b2ee9e2e8d36ea57bba59020421401fdc1e988',
    'ring.json':'63319556a1f8c4a6e05424d62e8adf2505a5da4097145d7145c9bdc5916da1c9',
    'heldout.npz':'9bc068beb91487246f3aa97d4e0881567e91891becdd559ce16a7ba6a9badfce',
    'declared_inputs.json':'e2bf1bc93d59629ff9249dd6f9a829f809a5bc8740a933b8adc7e4425e5106dc'}
FIELDS = (2., 4., 8., 16., 32.)
DELTAS = (np.pi/3, np.pi/2, 2*np.pi/3)
PHASES = (0., .8, 1.6, 2.4)
MASS, CHARGE, THETA, KAPPA, SCREENING = 1., 1., .5, .03, 3.
BASE_STEP, REFINED_STEP = .2, .0025
DPI = 180
INK, TEAL, ORANGE, PURPLE, RED, PALE = '#253447', '#147D92', '#C66B28', '#79669A', '#A13B34', '#F1F5F6'
plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False,
    'axes.labelcolor':INK, 'text.color':INK, 'axes.titleweight':'bold',
    'axes.edgecolor':'#82929C', 'grid.color':'#DCE4E7', 'grid.alpha':.6,
    'figure.facecolor':'white', 'savefig.facecolor':'white', 'svg.fonttype':'path'})
print('Verify immutable campaign inputs; render discrete observations and declared checks.', flush=True)
for name, expected in SOURCE_HASHES.items():
    if sha256((SOURCE/name).read_bytes()).hexdigest() != expected:
        raise RuntimeError(f'Saved campaign evidence changed: {name}')
source = json.loads((SOURCE/'summary.json').read_text())
results = source['results']
inputs = source['inputs']
cases = json.loads((SOURCE/'cases.json').read_text())
ring = json.loads((SOURCE/'ring.json').read_text())
heldout = np.load(SOURCE/'heldout.npz')
assert (inputs['mass'],inputs['charge'],inputs['theta'],inputs['strength'],inputs['screening']) == (MASS,CHARGE,THETA,KAPPA,SCREENING)
assert inputs['fields'] == list(FIELDS)
assert results['overall_status'] == 'unresolved'
assert results['required_gates']['ring_refinement_and_positive_broadening']
assert not results['required_gates']['phase_and_coordinate_moments']
assert not results['required_gates']['heldout_table']
started = perf_counter()
selected = {b:[r for r in ring if r['field']==b and r['max_step']==(REFINED_STEP if b>=16 else BASE_STEP)] for b in FIELDS}
for b, rows in selected.items():
    assert len(rows)==12 and all(r['status']=='passed' for r in rows)
    assert [(r['delta'],r['phase']) for r in rows] == [(d,p) for d in DELTAS for p in PHASES]
# The identity below uses each actual initial value, including floating-point roundoff.
identity_errors = []
for rows in selected.values():
    for r in rows:
        initial = np.array(r['initial_mu']); increments = np.array(r['increments'])
        reconstructed = 2*initial@increments + increments@increments
        identity_errors.append(abs(reconstructed-r['delta_sum_mu_squared'])/r['mu_reference']**2)
assert max(identity_errors)<1e-14
refined_source = None
if FIGURE_MODE == 'refinement':
    if not REFINEMENT_HASHES:
        raise RuntimeError('Focused refinement is not yet complete and pinned')
    for name,expected in REFINEMENT_HASHES.items():
        if sha256((REFINEMENT_SOURCE/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'Focused refinement evidence changed: {name}')
    refined_source = json.loads((REFINEMENT_SOURCE/'summary.json').read_text())
    for name,digest in refined_source['source_artifact_sha256'].items():
        assert digest==SOURCE_HASHES[name]
if OUTPUT.exists():
    raise FileExistsError(f'Preserve existing evidence: {OUTPUT}; choose a fresh declared destination')
OUTPUT.mkdir(parents=True)
FOOTER = 'Original campaign UNRESOLVED: B=4 coordinate refinement; heldout table p95.  Finite screened incoming model; no plasma mixing time.'
UNITS = r'$m=q=1,\ \theta=0.5,\ \kappa=0.03,\ \lambda=3;\quad \mu_0=m\theta/B$'


def save(fig, name):
    fig.text(.035,.027,FOOTER,fontsize=9,color=RED)
    fig.savefig(OUTPUT/(name+'.svg'),metadata={'Creator':None,'Date':None})
    fig.savefig(OUTPUT/(name+'.png'),dpi=DPI)
    plt.close(fig)
    print(f'  Saved {name}.svg and .png',flush=True)


if FIGURE_MODE == 'original':
    # Exact equations; all glyphs are SVG paths for portable GitHub rendering.
    fig = plt.figure(figsize=(12.4,10.3))
    fig.text(.035,.955,'What a screened pair encounter changes',fontsize=22,fontweight='bold')
    fig.text(.035,.917,UNITS+'   Uniform magnetic field; repulsive pair potential.',fontsize=12)
    for bottom,height in [(.645,.238),(.33,.293),(.076,.232)]:
        fig.add_artist(FancyBboxPatch((.026,bottom),.948,height,boxstyle='round,pad=0.008',
            transform=fig.transFigure,facecolor=PALE,edgecolor='#DCE4E7',zorder=0))
    fig.text(.045,.853,'1  Exact relative dynamics',fontsize=15,fontweight='bold',color=TEAL)
    fig.text(.045,.799,r'$\mathbf{r}=\mathbf{x}_1-\mathbf{x}_2,\quad \mathbf{w}=\mathbf{v}_1-\mathbf{v}_2,\quad \Omega=qB/m,\quad U(r)=\kappa e^{-r/\lambda}/r$',fontsize=18)
    fig.text(.045,.735,r'$\dot{\mathbf{r}}=\mathbf{w},\qquad \dot{\mathbf{w}}=\Omega\,\mathbf{w}\times\widehat{\mathbf{z}}+\dfrac{2\kappa}{m}\dfrac{e^{-r/\lambda}(1+r/\lambda)}{r^3}\,\mathbf{r}$',fontsize=21)
    fig.text(.045,.674,r'$E_{\rm rel}=m|\mathbf{w}|^2/4+U(r)$'+'     Solve from incoming z=−24 to the first outgoing z=±24 plane.',fontsize=13)
    fig.text(.045,.594,'2  Center-of-mass motion carries the individual exchange',fontsize=15,fontweight='bold',color=ORANGE)
    fig.text(.045,.544,r'$\dot{\mathbf{V}}=\Omega\,\mathbf{V}\times\widehat{\mathbf{z}},\quad \mathbf{v}_{1,2}=\mathbf{V}\pm\mathbf{w}/2,\quad \mu_{1,2}=\dfrac{m}{2B}|\mathbf{V}_\perp\pm\mathbf{w}_\perp/2|^2$',fontsize=18)
    fig.text(.045,.490,r'$\Delta\mu_{1,2}=c\pm a,\quad c=\dfrac{m}{8B}\Delta|\mathbf{w}_\perp|^2,\quad a=\dfrac{m}{2B}\mathbf{V}_{\perp,\rm in}\cdot\delta\mathbf{w}_\perp$',fontsize=19)
    fig.text(.045,.442,r'$\delta\mathbf{w}_\perp=R_{\rm free}(t)^{-1}\mathbf{w}_{\perp,\rm out}-\mathbf{w}_{\perp,\rm in},\qquad \Delta(\mu_1+\mu_2)=2c$',fontsize=17)
    fig.text(.045,.395,r'$\mathbf{V}_{\rm in}\sim N(0,\theta I/2):\quad \langle(\Delta\mu_1)^2\rangle_{\mathbf{V}}=c^2+\left(\dfrac{m}{2B}\right)^2\dfrac{\theta}{2}|\delta\mathbf{w}_\perp|^2$',fontsize=17)
    fig.text(.045,.352,r'Ring COM: $\mathbf{V}_{\perp,\rm in}=\frac{1}{2}\cot(\delta/2)\,\widehat{\mathbf{z}}\times\mathbf{w}_{\perp,\rm in}$; prescribed for each pair, distinct from thermal COM averaging.',fontsize=12)
    fig.text(.045,.280,'3  A narrow ring: full marginal change includes drift and squared kicks',fontsize=15,fontweight='bold',color=PURPLE)
    fig.text(.045,.226,r'$G_{\rm in}(\mu)=\delta(\mu-\mu_0),\qquad G_{\rm out}(\mu)=\dfrac{1}{24}\sum_{j=1}^{12}\sum_{i=1}^{2}\delta(\mu-\mu_{i,j}^{\rm out})$',fontsize=19)
    fig.text(.045,.165,r'$\Delta\!\sum_{i=1}^{2}\mu_i^2=2\mu_0\,\Delta(\mu_1+\mu_2)+\sum_{i=1}^{2}(\Delta\mu_i)^2$',fontsize=22)
    fig.text(.045,.112,'Twelve prescribed pair samples per B, each with equal initial individual μ. The displayed marginal is empirical.',fontsize=11)
    fig.text(.045,.084,'Conserving the mean would not conserve the full marginal; positive squared kicks alone omit the signed drift term.',fontsize=11)
    save(fig,'equations')

    # Every dot is a recorded particle value; no connecting lines or density reconstruction.
    fig, axes = plt.subplots(1,5,figsize=(15.8,5.5))
    fig.subplots_adjust(left=.057,right=.985,bottom=.31,top=.76,wspace=.34)
    fig.suptitle('Individual moments move away from the incoming ring',x=.035,ha='left',fontsize=21,y=.967,fontweight='bold')
    fig.text(.035,.878,UNITS+r'   $b=1.8,\ u=1.1,\ v_0=\sqrt{2\theta}=1$; 12 prescribed pair samples per field.',fontsize=12)
    markers = ('o','s','^')
    for axis,b in zip(axes,FIELDS):
        rows = selected[b]
        for d,marker in zip(DELTAS,markers):
            group = [r for r in rows if r['delta']==d]
            for i,color in [(0,TEAL),(1,ORANGE)]:
                axis.scatter([r['phase'] for r in group],[(r['final_mu'][i]-r['mu_reference'])/r['mu_reference']*1e3 for r in group],
                    color=color,marker=marker,s=39,alpha=.85,edgecolors='white',linewidths=.4)
        axis.axhline(0,color=INK,lw=.8)
        axis.set(xlabel='incoming phase φ [rad]',xticks=PHASES)
        axis.set_title(f'B = {b:g}',pad=32)
        axis.text(.5,1.055,'ring refinement passed' if b>=16 else 'ring step unrefined',transform=axis.transAxes,
            ha='center',fontsize=9,color=TEAL if b>=16 else RED)
        axis.grid(axis='y')
    axes[0].set_ylabel(r'$(\mu-\mu_0)/\mu_0\ \times 10^3$')
    handles = [Line2D([],[],color=c,marker='o',ls='',label=label) for c,label in [(TEAL,'particle 1'),(ORANGE,'particle 2')]]
    handles += [Line2D([],[],color=INK,marker=m,ls='',label=label) for m,label in zip(markers,['δ = π/3','δ = π/2','δ = 2π/3'])]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.51,.088),ncol=5,frameon=False)
    fig.text(.035,.164,'Each panel uses its own vertical scale. Shapes encode the initial relative ring angle δ; colors identify the two particles.',fontsize=10)
    save(fig,'ring_individual_samples')

    # These are pair sums, twice the single-particle empirical second-moment change.
    fig, axes = plt.subplots(1,3,figsize=(14.8,5.8))
    fig.subplots_adjust(left=.07,right=.98,bottom=.32,top=.76,wspace=.33)
    fig.suptitle('Small total-moment drift and positive squared kicks are distinct',x=.035,ha='left',fontsize=20,y=.963,fontweight='bold')
    fig.text(.035,.89,r'$S=\sum_i(\Delta\mu_i)^2/\mu_0^2,\quad T=2\Delta(\mu_1+\mu_2)/\mu_0,\quad \Delta\sum_i\mu_i^2/\mu_0^2=T+S$',fontsize=16)
    for axis,b in zip(axes[:2],(16.,32.)):
        rows = selected[b]; x = np.arange(1,13)
        scale = 1e7 if b==16 else 1e8
        kick = np.array([r['increment_squared_sum']/r['mu_reference']**2 for r in rows])
        total = np.array([2*r['delta_total_mu']/r['mu_reference'] for r in rows])
        full = np.array([r['normalized_marginal_broadening'] for r in rows])
        coarse = [r for r in ring if r['field']==b and r['max_step']==BASE_STEP]
        error = np.abs(full-np.array([r['normalized_marginal_broadening'] for r in coarse]))
        axis.scatter(x,kick*scale,color=TEAL,marker='s',s=36,label='S: positive squared kicks')
        axis.scatter(x,total*scale,color=ORANGE,marker='^',s=39,label='T: signed total-μ drift × 2')
        axis.errorbar(x,full*scale,yerr=error*scale,fmt='o',color=PURPLE,ms=4,capsize=2,label='T+S: full second moment')
        axis.axhline(0,color=INK,lw=.8)
        axis.set(xlabel='recorded pair sample j',ylabel=f'normalized contribution × 10^{int(np.log10(scale))}',xticks=[1,4,5,8,9,12])
        axis.set_title(f'B = {b:g} · ring refinement passed',fontsize=12)
        axis.grid(axis='y')
    axis = axes[2]
    for b,marker in [(16.,'o'),(32.,'s')]:
        rows = selected[b]
        axis.scatter([r['increment_squared_sum']/r['mu_reference']**2 for r in rows],
            [r['delta_total_mu']/r['mu_reference'] for r in rows],marker=marker,s=42,color=TEAL if b==16 else ORANGE,label=f'B = {b:g}')
    axis.axhline(0,color=INK,lw=.8)
    axis.set(xlabel=r'positive squared kick $S$',ylabel=r'total-moment drift $\Delta(\mu_1+\mu_2)/\mu_0$')
    axis.set_title('The total moment does not fix the kick',fontsize=12)
    axis.ticklabel_format(axis='both',style='sci',scilimits=(0,0));axis.legend(frameon=False,fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.073),ncol=3,frameon=False,fontsize=10)
    fig.text(.035,.203,'Dots are actual samples. Error bars: absolute coarse-to-fine change in T+S; max step 0.2 → 0.0025.',fontsize=10)
    fig.text(.035,.16,'Sample order: δ = π/3, π/2, 2π/3; within each angle φ = 0, 0.8, 1.6, 2.4 rad. No continuous marginal is inferred.',fontsize=10)
    save(fig,'ring_second_moment')

    # Predeclared checks in target units: values above one fail the numerical gate.
    fig, axes = plt.subplots(1,3,figsize=(15.3,6.2))
    fig.subplots_adjust(left=.055,right=.98,bottom=.31,top=.745,wspace=.34)
    fig.suptitle('The acceptance gates retain the failed refinements',x=.035,ha='left',fontsize=21,y=.961,fontweight='bold')
    fig.text(.035,.891,'21,524 unique encounters completed; energy conservation alone does not establish observable convergence.',fontsize=12)
    labels = ['mean increment','individual second moment','pair cross moment','spatial kick']
    colors = [ORANGE,TEAL,PURPLE,INK]
    for j,(label,color) in enumerate(zip(labels,colors)):
        axes[0].plot(FIELDS,[r['coordinate_changes'][j]/inputs['moment_target'] for r in results['scan_checks']],'-o',color=color,ms=4,label=label)
    axes[0].axhline(1,color=RED,ls='--',lw=1.3)
    axes[0].set(xscale='log',yscale='log',xticks=FIELDS,xlabel='B',ylabel='coordinate change / 1% target',title='B=4 coordinate refinement fails')
    axes[0].set_xticklabels([f'{b:g}' for b in FIELDS]);axes[0].xaxis.set_minor_locator(NullLocator());axes[0].grid(axis='y');axes[0].legend(fontsize=8.5,frameon=False)
    axes[0].annotate('1.93 × target',xy=(4,results['scan_checks'][1]['coordinate_changes'][1]/inputs['moment_target']),xytext=(6,2.8),color=RED,fontsize=10,arrowprops={'arrowstyle':'-','color':RED})
    relative = np.abs(heldout['predicted'][:,1]/heldout['direct'][:,1]-1)
    assert np.isclose(np.quantile(relative,.95),results['heldout_table']['second_p95'],rtol=1e-14)
    assert np.isclose(np.sqrt(np.mean((heldout['direct'][:,1]-heldout['predicted'][:,1])**2))/np.sqrt(np.mean(heldout['direct'][:,1]**2)),results['heldout_table']['rms'][1],rtol=1e-14)
    axes[1].scatter(np.arange(1,len(relative)+1),np.sort(relative)*100,s=13,color=TEAL)
    axes[1].axhline(inputs['table_p95_target']*100,color=RED,ls='--',label='declared p95 target: 5%')
    axes[1].axhline(results['heldout_table']['second_p95']*100,color=ORANGE,ls=':',label='observed p95: 7.89%')
    axes[1].set(xlabel='heldout sample rank (128 actual points)',ylabel='individual second-moment error [%]',title='Heldout tail fails; RMS passes')
    axes[1].legend(fontsize=9,frameon=False,loc='upper left');axes[1].grid(axis='y')
    metrics = [max(results['base_final_refinement_changes'])/inputs['moment_target'],
        max(results['boundary_changes_24_to_32'])/inputs['moment_target'],
        max(results['step_changes_01_to_005'])/inputs['step_target'],
        max(r['increment_refinement_error'] for r in results['ring_checks'])/inputs['ring_increment_normalized_target'],
        results['heldout_table']['rms'][1]/inputs['table_rms_target'],
        results['heldout_table']['second_p95']/inputs['table_p95_target']]
    names = ['B=2 final moments','receding planes','active timestep','ring increments','table RMS','table p95']
    y = np.arange(len(metrics));axes[2].barh(y,metrics,color=[TEAL if v<=1 else RED for v in metrics],height=.62)
    axes[2].axvline(1,color=RED,ls='--',lw=1.3)
    axes[2].set(xscale='log',xlim=(1e-6,4),yticks=y,yticklabels=names,xlabel='observed error / predeclared target',title='Passing checks cannot replace failures')
    axes[2].invert_yaxis();axes[2].grid(axis='x')
    for position,value in zip(y,metrics):axes[2].text(value*1.13,position,f'{value:.3g}',va='center',fontsize=9)
    fig.text(.035,.191,'Coordinate refinement: (3,3,3,16) → (4,4,4,16). Phase and shifted-phase checks pass at every B.',fontsize=10)
    fig.text(.035,.148,'Final B=2: (5,5,5,32). Planes: 24 → 32. Active step: 0.1 → 0.05. Every threshold was declared before the campaign.',fontsize=10)
    save(fig,'convergence')

    # Saved rates and independently derived leading large-B values; references are not fits.
    scans = [next(r for r in cases if r['case']==f'B{b:g}') for b in FIELDS]
    bvals = np.array(FIELDS)
    mu = np.array([r['normalized_mu_rate_per_density'] for r in scans])
    spatial = np.array([r['normalized_spatial_rate_per_density'] for r in scans])
    mu_ref = np.array([r['mu_rate_asymptotic'] for r in results['ordering']])
    spatial_ref = np.array([r['spatial_rate_asymptotic'] for r in results['ordering']])
    fig, axes = plt.subplots(1,3,figsize=(15.0,6.1))
    fig.subplots_adjust(left=.065,right=.985,bottom=.31,top=.745,wspace=.33)
    fig.suptitle('Bounded transport approaches screened large-B references',x=.035,ha='left',fontsize=21,y=.963,fontweight='bold')
    fig.text(.035,.890,UNITS+'   Rates per density n; μ uses μ₀² and the spatial kick uses the fixed length λ² = 9.',fontsize=12)
    axes[0].loglog(bvals,mu,'o-',color=TEAL,label='individual μ kicks')
    axes[0].loglog(bvals,spatial,'s-',color=ORANGE,label='individual guiding-center kicks')
    axes[0].loglog(bvals,mu_ref,':',color=TEAL,lw=2,label='large-B μ reference')
    axes[0].loglog(bvals,spatial_ref,'--',color=ORANGE,lw=1.8,label='large-B spatial reference')
    axes[0].scatter([4],[mu[1]],s=160,facecolors='none',edgecolors=RED,lw=1.8,zorder=6)
    axes[0].set(xlabel='B',ylabel='normalized kick rate / n',title='Finite bands; references ∝ B⁻²')
    axes[0].legend(fontsize=8.5,frameon=False);axes[0].grid(axis='y')
    axes[1].plot(bvals,mu/mu_ref,'o-',color=TEAL,label='μ rate / leading reference')
    axes[1].plot(bvals,spatial/spatial_ref,'s-',color=ORANGE,label='spatial rate / leading reference')
    axes[1].axhline(1,color=INK,ls=':',lw=1)
    axes[1].set(xscale='log',yscale='log',xlabel='B',ylabel='measured / analytic leading value',title='Finite-strength differences persist')
    axes[1].legend(fontsize=8.5,frameon=False);axes[1].grid(axis='y')
    axes[1].text(.04,.80,'B=32: μ +7.51%\nspatial +3.91%',transform=axes[1].transAxes,fontsize=10)
    controls = results['screening_asymptotic_controls'];x=np.arange(4)
    # Scale each family to its lambda=3 reference for a direct screening comparison.
    axes[2].plot(x,[r['normalized_mu_rate_per_density']/controls[0]['normalized_mu_rate_per_density'] for r in controls],'o-',color=TEAL,label='μ reference')
    axes[2].plot(x,[r['normalized_spatial_rate_per_density_fixed_lambda3']/controls[0]['normalized_spatial_rate_per_density_fixed_lambda3'] for r in controls],'s-',color=ORANGE,label='spatial reference')
    axes[2].set(xticks=x,xticklabels=['3','6','12','∞'],xlabel='screening length λ',ylabel='leading rate / its λ=3 value',title='B=32 screening: analytic only')
    axes[2].legend(fontsize=9,frameon=False);axes[2].grid(axis='y')
    for axis in axes[:2]:
        axis.set_xticks(FIELDS);axis.set_xticklabels([f'{b:g}' for b in FIELDS]);axis.xaxis.set_minor_locator(NullLocator())
    fig.text(.035,.191,'Incoming bands: 1.2 ≤ b ≤ 2.4; 0.7 ≤ u ≤ 1.5; 0.2 ≤ v⊥,relative ≤ 0.8. Phase quadrature: 16 nodes.',fontsize=10)
    fig.text(.035,.148,'Red circle marks unresolved B=4 coordinate refinement. Changed-screening references have no finite-B direct validation.',fontsize=10)
    save(fig,'magnetic_scaling')

else:
    refined = refined_source['results']
    refined_inputs = refined_source['inputs']
    FOOTER = f"Focused refinement {refined['overall_status'].upper()}; original campaign UNRESOLVED preserved.  Finite screened bands; no plasma mixing time."
    fig, axes = plt.subplots(1,3,figsize=(15.4,6.3))
    fig.subplots_adjust(left=.063,right=.982,bottom=.30,top=.72,wspace=.34)
    fig.suptitle('Focused refinement tests the original failed gates',x=.035,ha='left',fontsize=21,y=.958,fontweight='bold')
    fig.text(.035,.888,UNITS+'   Fresh heldout seed 3701; predeclared 1%, 5%, and 2% acceptance targets.',fontsize=12)
    x = np.arange(4)
    old_coordinate = np.array(results['scan_checks'][1]['coordinate_changes'])/inputs['moment_target']
    new_coordinate = np.array(refined['B4_last_refinement_changes'])/refined_inputs['moment_target']
    axes[0].bar(x-.19,old_coordinate,width=.37,color=RED,label='original: 3 → 4 per coordinate')
    axes[0].bar(x+.19,new_coordinate,width=.37,color=TEAL,label='focused: 5 → 6 per coordinate')
    axes[0].axhline(1,color=INK,ls='--',lw=1)
    axes[0].set(xticks=x,xticklabels=['mean','second','cross','spatial'],ylabel='B=4 coordinate change / 1% target',yscale='log',title='B=4 observable convergence')
    axes[0].legend(frameon=False,fontsize=8.5,loc='lower left',bbox_to_anchor=(0,1.09));axes[0].grid(axis='y')
    for coordinates,offset in [(old_coordinate,-.19),(new_coordinate,.19)]:
        for j,value in enumerate(coordinates):axes[0].text(j+offset,value*1.25,f'{value:.3g}',ha='center',fontsize=8)
    table_data = [('original table: original 128',heldout,RED,results['heldout_table'])]
    for name,color in [('original128',TEAL),('fresh128',PURPLE)]:
        dataset = np.load(REFINEMENT_SOURCE/f'heldout_{name}.npz')
        check = next(r for r in refined['table_checks'] if r['sample']==name)
        table_data.append((f'refined table: {"same" if name=="original128" else "fresh"} 128',dataset,color,check))
    for label,dataset,color,check in table_data:
        error = np.abs(dataset['predicted'][:,1]/dataset['direct'][:,1]-1)
        assert len(error)==128
        assert np.isclose(np.quantile(error,.95),check['second_p95'],rtol=1e-13)
        axes[1].scatter(np.arange(1,129),np.sort(error)*100,color=color,s=12,label=f'{label}; p95 {check["second_p95"]*100:.2f}%')
    axes[1].axhline(5,color=INK,ls='--',lw=1)
    axes[1].set(xlabel='heldout sample rank (actual points)',ylabel='individual second-moment error [%]',yscale='symlog',title='Heldout p95 passes; isolated tails persist')
    axes[1].set_ylim(bottom=0)
    axes[1].legend(frameon=False,fontsize=8,loc='upper left');axes[1].grid(axis='y')
    strengths = [KAPPA]+[row['strength'] for row in refined['weak_B32']]
    mu_errors = [results['ordering'][-1]['mu_asymptotic_relative_error']]+[row['asymptotic_relative_errors'][0] for row in refined['weak_B32']]
    spatial_errors = [results['ordering'][-1]['spatial_asymptotic_relative_error']]+[row['asymptotic_relative_errors'][1] for row in refined['weak_B32']]
    axes[2].plot(strengths,np.array(mu_errors)*100,'o-',color=TEAL,label='μ kicks / screened leading value')
    axes[2].plot(strengths,np.array(spatial_errors)*100,'s-',color=ORANGE,label='spatial kicks / screened leading value')
    axes[2].axhline(2,color=INK,ls='--',lw=1,label='declared weak-force target: 2%')
    axes[2].set(xscale='log',xticks=strengths,xticklabels=[f'{v:g}' for v in strengths],xlabel='force strength κ at B=32',ylabel='absolute leading-reference error [%]',title='Weak-force asymptotic control')
    axes[2].xaxis.set_minor_locator(NullLocator());axes[2].legend(frameon=False,fontsize=8.5);axes[2].grid(axis='y')
    fig.text(.035,.184,'Table refinement: (5,5,5,32) → (7,7,7,32). Same 128 states are replayed; a further 128 use the separately declared fresh seed.',fontsize=10)
    fig.text(.035,.142,'Weak-force tests: κ = 0.003 and 0.0015 at B=32. The κ = 0.03 point remains the original finite-strength comparison.',fontsize=10)
    save(fig,'focused_refinement')

render_processor = platform.processor()
if sys.platform == 'darwin':
    render_processor = subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()
render_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)

# Rendering provenance complements (and preserves) the producer's complete solver provenance.
metadata = {
    'model':'Saved-data scientific figure rendering; no new ODE, quadrature, or interpolation solve',
    'inputs':{'figure_mode':FIGURE_MODE,'source_directory':str(SOURCE.relative_to(ROOT)),'source_sha256':SOURCE_HASHES,
        'mass':MASS,'charge':CHARGE,'theta':THETA,'strength':KAPPA,'screening':SCREENING,
        'fields':list(FIELDS),'ring_deltas':list(DELTAS),'ring_phases':list(PHASES),
        'ring_selection':'max_step=0.0025 at B16/B32; original max_step=0.2 at B2/B4/B8',
        'sample_scope':'12 predeclared pair states per B, 24 individual point masses, no inferred continuous distribution',
        'thresholds':{k:inputs[k] for k in ['moment_target','table_rms_target','table_p95_target','step_target','ring_increment_normalized_target']},
        'svg_fonttype':'path','raster_dpi':DPI},
    'source_campaign_provenance':{k:source[k] for k in ['commit','working_tree','source_sha256','versions','logical_cpus','thread_environment','python','platform','processor','devices','x64','experiment_dependency_sha256','experiment_dependency_sha256_end','experiment_dependencies_unchanged','wall_s','parent_process_peak_rss_bytes','worker_process_peak_rss_bytes_max','memory_scope']},
    'source_status':results['overall_status'],'source_gates':results['required_gates'],
    'focused_refinement_provenance':refined_source,
    'focused_refinement_input_sha256':REFINEMENT_HASHES if FIGURE_MODE=='refinement' else None,
    'render_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    'render_working_tree':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).strip(),
    'render_script_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
    'render_versions':{name:importlib.metadata.version(name) for name in ['numpy','matplotlib']},
    'render_python':sys.version,'render_platform':platform.platform(),'render_processor':render_processor,
    'render_logical_cpus':os.cpu_count(),'render_wall_s':perf_counter()-started,
    'render_parent_process_peak_rss_bytes':render_peak_rss,
    'render_memory_scope':'Single renderer process lifetime peak; includes imports, no worker processes.',
    'render_runtime_scope':'Input parsing, validation and rendering; excludes imports, initial hash verification and final metadata writing.',
    'ring_second_moment_identity_max_normalized_roundoff':max(identity_errors),
    'outputs_sha256':{p.name:sha256(p.read_bytes()).hexdigest() for p in sorted(OUTPUT.iterdir())}}
(OUTPUT/'summary.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
render_status = refined_source['results']['overall_status'] if refined_source else results['overall_status']
print(f'Rendered {FIGURE_MODE} saved-data figures; displayed run status={render_status}; original campaign={results["overall_status"]}; wall={metadata["render_wall_s"]:.2f}s',flush=True)
