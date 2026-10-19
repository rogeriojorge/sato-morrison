"""Saved-array phase-pair sampling and physical second-contribution maps."""
from pathlib import Path
from time import perf_counter
import hashlib
import json
import os
import subprocess
import platform
import resource
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=ROOT/'results/stratified_encounter_flux'
OUTPUT=ROOT/os.environ.get('SM_ANTITHETIC_FIGURE_OUTPUT',str(HERE))
expected=['contribution_maps.npz','publication_render_receipt.json']+[f'{name}.{ext}' for name in ('sampling_readable','contribution_maps','pair_weights') for ext in ('svg','png')]
existing=[str(OUTPUT/name) for name in expected if (OUTPUT/name).exists()]
if existing:
    raise FileExistsError(f'Preserve existing figure evidence: {existing}; select a fresh SM_ANTITHETIC_FIGURE_OUTPUT')
OUTPUT.mkdir(parents=True,exist_ok=True)
started=perf_counter()
summary=json.loads((HERE/'summary.json').read_text())
inputs=summary['inputs'];results=summary['results']
if not all(c['complete'] for c in results['cases']):
    raise ValueError('Final cases contain missing physical states; no contribution map can be normalized. Preserve saved failures and original weights.')
seeds=inputs['seeds'];level=max(inputs['base_levels'])
files=[HERE/'summary.json',HERE/'states.npz',HERE/'populations.npz',BASE/'summary.json',BASE/'states.npz']
with np.load(BASE/'states.npz') as archive:
    base={k:archive[k] for k in ('nodes','moments','status')}
with np.load(HERE/'states.npz') as archive:
    new={k:archive[k] for k in ('nodes','moments','status')}
with np.load(HERE/'populations.npz') as archive:
    populations={k:archive[k] for k in archive.files}
base_count=inputs['base_state_count']
assert base_count==len(base['nodes'])
def extract(name,ids):
    old=ids<base_count
    shape=(len(ids),)+base[name].shape[1:]
    out=np.empty(shape,dtype=base[name].dtype)
    out[old]=base[name][ids[old]]
    out[~old]=new[name][ids[~old]-base_count]
    return out
# Fixed full-domain bins. These are contribution sums, not point counts or densities.
b_edges=np.linspace(1.2,2.4,33)
ratio_edges=np.linspace(0,2.5/(abs(inputs['charge']*inputs['field']/inputs['mass'])*1.2),33)
phase_edges=np.linspace(0,2*np.pi,49)
hist_b=np.zeros((32,32));hist_phase=np.zeros((48,32));reconstruction=[]
omega=abs(inputs['charge']*inputs['field']/inputs['mass'])
for seed in seeds:
    label=f'seed{seed}_N{level}'
    case=next(c for c in results['cases'] if c['case']==label)
    original=populations[label+'_original'];partner=populations[label+'_partner']
    ids=np.concatenate((original,partner))
    if np.any(ids<0) or not np.all(extract('status',ids)=='passed'):
        raise ValueError('No failed or missing physical members may be dropped')
    nodes=extract('nodes',ids);second=extract('moments',ids)[:,1]
    weights=np.tile(populations[label+'_pair_weights']/2,2)
    contribution=weights*second
    if not np.all(contribution>=0):raise ValueError('Negative squared-kick contribution')
    ratio=nodes[:,2]/(omega*nodes[:,0])
    hb=np.histogram2d(nodes[:,0],ratio,bins=(b_edges,ratio_edges),weights=contribution)[0]
    hp=np.histogram2d(nodes[:,3]%(2*np.pi),ratio,bins=(phase_edges,ratio_edges),weights=contribution)[0]
    for value in (contribution.sum(),hb.sum(),hp.sum()):
        if not np.isclose(value,case['integrals'][1],rtol=2e-12,atol=0):
            raise ValueError('Map omitted physical contributions')
    hist_b+=hb/len(seeds);hist_phase+=hp/len(seeds)
    reconstruction.append(dict(seed=seed,members=len(ids),total_flux_weight=float(weights.sum()),second_integral=float(contribution.sum())))
mean_second=float(np.mean([c['integrals'][1] for c in results['cases'] if c['case'] in [f'seed{s}_N{level}' for s in seeds]]))
np.savez_compressed(OUTPUT/'contribution_maps.npz',b_edges=b_edges,ratio_edges=ratio_edges,phase_edges=phase_edges,
    b_ratio_mean_second=hist_b,phase_ratio_mean_second=hist_phase,arithmetic_mean_second=mean_second)
plt.rcParams.update({'font.size':13,'axes.titlesize':14,'legend.fontsize':9,
    'figure.facecolor':'white','svg.fonttype':'path'})
outputs=[OUTPUT/'contribution_maps.npz']
def save(fig,name):
    for ext in ('svg','png'):
        path=OUTPUT/f'{name}.{ext}';fig.savefig(path,dpi=160,bbox_inches='tight',pad_inches=.12,
            metadata={'Creator':None,'Date':None} if ext=='svg' else None);outputs.append(path)
    plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10.3,4.2),layout='constrained')
for seed in seeds:
    cc=[c for c in results['cases'] if c['seed']==seed and c['distance']==24.]
    axes[0].plot([c['physical_member_count'] for c in cc],[c['integrals'][0] if c.get('integrals') is not None else np.nan for c in cc],'o-',label=str(seed))
axes[0].set(xlabel='physical members per replica',ylabel='signed mean integral per density (normalized)',title='Actual phase partners')
axes[0].set_xticks([15360,30720,61440],['15,360','30,720','61,440'])
axes[0].ticklabel_format(axis='y',style='sci',scilimits=(0,0));axes[0].legend(ncol=2,title='independent sample seed')
checks=results['comparisons']
groups=[('sample',[k for k in checks if k.startswith('sample_')],.05),
    ('replicas',[k for k in checks if k.startswith('replica_')],.05),
    ('partition',[k for k in checks if k.startswith('partition_')],.05),
    ('L24→40',['boundary_L24_to40'],.005),('L40→48',['boundary_ensemble'],.005),
    ('selected\nplanes',['boundary_controls','boundary_controls_members'],.005),
    ('step',['active_step','active_step_members'],.001)]
missing_groups=[]
for j,name in enumerate(('signed mean','second','cross','spatial second')):
    y=[]
    for index,(_,keys,target) in enumerate(groups):
        values=[checks.get(k) for k in keys]
        valid=all(v is not None and len(v)>j and np.isfinite(v[j]) for v in values)
        y.append(max(v[j] for v in values)/target if valid else np.nan)
        if not valid:missing_groups.append(index)
    axes[1].plot(range(len(groups)),y,'o-',ms=4,label=name)
for index in sorted(set(missing_groups)):
    axes[1].text(index,.06,'missing',rotation=90,color='crimson',ha='center',transform=axes[1].get_xaxis_transform())
axes[1].axhline(1,color='.4',ls=':',label='fixed target')
axes[1].set_xticks(range(len(groups)),[g[0] for g in groups],rotation=35,ha='right')
axes[1].set(ylabel='worst change / own fixed target',title=f'Fixed checks: {results["status"]}')
axes[1].legend(ncol=2,fontsize=9)
save(fig,'sampling_readable')

fig,axes=plt.subplots(1,2,figsize=(10.3,4.0),layout='constrained')
percent_b=100*hist_b/mean_second;percent_phase=100*hist_phase/mean_second
positive=np.concatenate((percent_b[percent_b>0],percent_phase[percent_phase>0]))
norm=LogNorm(vmin=float(positive.min()),vmax=float(positive.max()))
for ax,h,xedges,xlabel in zip(axes,(percent_b,percent_phase),(b_edges,phase_edges),('guiding-center impact b (normalized)','incoming phase φ (radians)')):
    artist=ax.pcolormesh(xedges,ratio_edges,np.ma.masked_equal(h.T,0),norm=norm,cmap='viridis',rasterized=True)
    ax.set(xlabel=xlabel,ylabel='relative gyro-radius / impact, ρ/b',ylim=(ratio_edges[0],ratio_edges[-1]))
    ax.axhline(1,color='white',lw=1,ls=':')
axes[0].set_title('Physical second-moment contributions')
axes[1].set_title('Four-sample arithmetic mean')
axes[1].set_xticks([0,np.pi/2,np.pi,3*np.pi/2,2*np.pi],['0','π/2','π','3π/2','2π'])
fig.colorbar(artist,ax=axes,label='fraction of mean second integral per bin (%)',shrink=.9)
save(fig,'contribution_maps')

fig=plt.figure(figsize=(10.3,2.7))
fig.text(.5,.81,r'$T(b,u,p,\varphi)=(b,u,p,(\pi-\varphi)\;\mathrm{mod}\;2\pi),\qquad T^2=I$',ha='center',fontsize=17)
fig.text(.5,.50,r'$A_{\mathrm{even}}(z)=\frac{A(z)+A(Tz)}{2}$',ha='center',fontsize=22)
fig.text(.5,.10,r'$\widehat I_A^{\mathrm{pair}}=\sum_s\frac{W_s}{N_s}\sum_{j=1}^{N_s}A_{\mathrm{even}}(z_{sj})=\sum_s\sum_{j=1}^{N_s}\frac{W_s}{2N_s}\left[A(z_{sj})+A(Tz_{sj})\right]$',ha='center',fontsize=18)
save(fig,'pair_weights')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def location(path):
    try:return str(path.relative_to(ROOT))
    except ValueError:return str(path)
renderer_commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
source_status=subprocess.run(['git','status','--porcelain','--',str(Path(__file__).relative_to(ROOT))],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
receipt=dict(scope='Complete final saved-array rendering only; no trajectory solve, control fit or gate change.',
    input_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},source_sha256=sha(Path(__file__)),
    outputs_sha256={location(p):sha(p) for p in outputs},
    renderer_commit=renderer_commit,source_uncommitted=bool(source_status),
    platform=platform.platform(),architecture=platform.machine(),processor=summary.get('processor',platform.processor()),
    scientific_units=summary['units'],producer_commit=summary.get('commit'),
    missing_comparison_groups=[groups[i][0] for i in sorted(set(missing_groups))],
    maps=dict(b_edges=b_edges.tolist(),ratio_edges=ratio_edges.tolist(),phase_edges=phase_edges.tolist(),
        gyro_radius='rho=p/abs(qB/m)',members='all finest original and partner physical members from each of four independent samples at L24; reflected members retained',
        per_member_weight='original pair weight/2',per_bin_quantity='sum of incoming physical flux weight times thermal-COM-averaged individual squared magnetic-moment increment',
        replica_combination='arithmetic mean of four unnormalized contribution histograms; display divides by arithmetic mean full second integral',
        no_bin_convergence_claim=True,reconstruction=reconstruction,color_percent_limits=[norm.vmin,norm.vmax]),
    versions=dict(numpy=np.__version__,matplotlib=matplotlib.__version__,python=platform.python_version()),
    wall_s=perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
(OUTPUT/'publication_render_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
