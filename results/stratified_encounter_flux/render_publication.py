"""Readable equations and figures from saved encounter summaries."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
CONTROL=HERE.parent/'encounter_born_control'
raw=json.loads((HERE/'summary.json').read_text())
cv=json.loads((CONTROL/'summary.json').read_text())
inputs=raw['inputs']; results=raw['results']; seeds=inputs['seeds']
plt.rcParams.update({'font.size':13,'axes.titlesize':14,'legend.fontsize':10,
    'figure.facecolor':'white','savefig.facecolor':'white','svg.fonttype':'path'})
outputs=[]
def save(fig,folder,name):
    for ext in ('svg','png'):
        p=folder/f'{name}.{ext}'
        fig.savefig(p,dpi=160,bbox_inches='tight',pad_inches=.12)
        outputs.append(p)
    plt.close(fig)

def groups(d):
    checks=d['comparisons']
    return [('sample',[k for k in checks if k.startswith('sample_')],.05),
        ('replicas',[k for k in checks if k.startswith('replica_')],.05),
        ('partition',[k for k in checks if k.startswith('partition_')],.05),
        ('L24→40',['boundary_L24_to40'],.005),
        ('L40→48',['boundary_ensemble'],.005),
        ('selected\nplanes',['boundary_controls'],.005),
        ('step',['active_step'],.001)]
def gates(ax,d):
    gg=groups(d)
    for j,name in enumerate(('signed mean','second','cross','spatial second')):
        y=[max(d['comparisons'][k][j] for k in keys)/target for _,keys,target in gg]
        ax.plot(range(len(gg)),y,'o-',ms=4,label=name)
    ax.axhline(1,color='.4',ls=':',label='fixed target')
    ax.set_xticks(range(len(gg)),[g[0] for g in gg],rotation=35,ha='right')
    ax.set(ylabel='worst change / own fixed target',title='Unresolved fixed checks')
    ax.legend(ncol=2,loc='upper right',fontsize=9)

fig,axes=plt.subplots(1,2,figsize=(10.3,4.1),layout='constrained')
for seed in seeds:
    cc=[c for c in results['cases'] if c['seed']==seed and c['distance']==24.]
    for j,ax in zip((1,0),axes):
        ax.plot([c['node_count'] for c in cc],[c['integrals'][j] for c in cc],'o-',label=str(seed))
for ax,title,ylabel in zip(axes,('Squared moment kicks','Signed mean kicks'),('second integral per density','mean integral per density')):
    ax.set(xlabel='incoming nodes per replica',ylabel=ylabel+' (normalized)',title=title)
    ax.set_xticks([7680,15360,30720],['7,680','15,360','30,720'])
    ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
axes[0].legend(title='independent seed',ncol=2,fontsize=9)
save(fig,HERE,'sampling_readable')

fig,axes=plt.subplots(1,2,figsize=(10.3,4.3),layout='constrained')
flux=inputs['bounded_flux_per_density']
x=np.arange(8)
axes[0].bar(x,[s['flux_per_density']/flux for s in inputs['main_strata']],color='.83',label='exact incoming flux')
for j,seed in enumerate(seeds):
    c=next(c for c in results['cases'] if c['seed']==seed and c['base_count']==2048 and c['distance']==24.)
    axes[0].plot(x+(j-1.5)*.055,[s['integrals'][1]/c['integrals'][1] for s in c['strata']],'o',ms=4,label=f'second: {seed}')
axes[0].set_xticks(x,['LLC','LLW','LHC','LHW','HLC','HLW','HHC','HHW'])
axes[0].set(yscale='log',ylabel='fraction of bounded flux or second integral',xlabel='impact / perpendicular speed / phase cell',title='Rare flux carries large kicks')
axes[0].legend(fontsize=9,loc='lower center',bbox_to_anchor=(.5,1.10),ncol=3)
axes[0].set_title('Rare flux carries large kicks',pad=54)
gates(axes[1],results)
save(fig,HERE,'evidence_readable')

fig,axes=plt.subplots(1,2,figsize=(10.3,4.1),layout='constrained')
for seed in seeds:
    cc=[c for c in cv['results']['cases'] if c['seed']==seed and c['distance']==24.]
    color=axes[0].plot([c['node_count'] for c in cc],[c['integrals'][0] for c in cc],'o-',label=str(seed))[0].get_color()
    axes[0].plot([c['node_count'] for c in cc],[c['raw_integrals'][0] for c in cc],':',color=color,alpha=.6)
axes[0].set(xlabel='same saved incoming nodes',ylabel='mean integral per density (normalized)',title='Known-zero control: solid; raw: dotted')
axes[0].set_xticks([7680,15360,30720],['7,680','15,360','30,720'])
axes[0].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
axes[0].legend(ncol=2,fontsize=9)
gates(axes[1],cv['results'])
save(fig,CONTROL,'control_readable')

fig=plt.figure(figsize=(10.3,1.8))
fig.text(.5,.73,r'$z_{sn}=(b,u,p,\varphi)_{sn},\qquad W_s=\int_{\mathcal{D}_s}\mathrm{d}\mathcal{F},\qquad \sum_s W_s=\mathcal{F}(\mathcal{D})$',ha='center',fontsize=17)
fig.text(.5,.23,r'$\widehat I_A=\sum_s\frac{W_s}{N_s}\sum_{n=1}^{N_s}A(z_{sn})$',ha='center',fontsize=22)
save(fig,HERE,'weights')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
receipt=dict(scope='Saved-summary scientific display only; no encounter or control recomputation.',
    input_sha256={str(p.relative_to(HERE.parent)):digest(p) for p in (HERE/'summary.json',CONTROL/'summary.json')},
    source_sha256=digest(Path(__file__)),outputs_sha256={str(p.relative_to(HERE.parent)):digest(p) for p in outputs})
(HERE/'publication_render_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
