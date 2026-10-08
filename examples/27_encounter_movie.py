"""Animate two verified Cartesian encounter paths, stopping each at its exit."""
from pathlib import Path
from time import perf_counter
import hashlib,json,shutil,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype']='path'
from matplotlib.animation import FuncAnimation,PillowWriter,FFMpegWriter
import PIL
from sato_morrison.reference import run_metadata,progress

ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'results'/'encounter_phase'
OUTPUT=ROOT/'results'/'encounter_movie'
PATHS=DIRECTORY/'independent_pair_paths.npz'
PATHS_SHA='31960cced83d49aaa1cd3e35792a75ab8b62a12b29da65cbd2f5eac28f411960'
SUMMARY_SHA='33f5e206d6a748518abe6808945b96d003b4e2fef283d318bd63f04973d2812e'
INDICES=(0,128)
STEP,MASS,FIELD,DISTANCE=.025,1.,2.,24.
FRAMES,FPS,DPI=120,10,110
POSITION_TARGET,MU_TARGET,ENERGY_TARGET=1e-8,1e-10,1e-8
COLORS=('#147D92','#C66B28')


def load_paths():
    summary=json.loads((DIRECTORY/'summary.json').read_text())
    audit=json.loads((DIRECTORY/'independent_audit.json').read_text())
    if hashlib.sha256(PATHS.read_bytes()).hexdigest()!=PATHS_SHA or hashlib.sha256((DIRECTORY/'summary.json').read_bytes()).hexdigest()!=SUMMARY_SHA:
        raise ValueError('Verified encounter source files changed')
    if summary['independent_pair_paths_sha256']!=PATHS_SHA or audit['input_sha256']['independent_pair_paths.npz']!=PATHS_SHA or audit['input_sha256']['summary.json']!=SUMMARY_SHA:
        raise ValueError('Trajectory/summary/audit associations differ')
    if audit['status']!='passed_saved_array_checks' or audit['producer_commit']!=summary['commit']:
        raise ValueError('Saved-path independent audit is not associated with the producer')
    nodes_path=DIRECTORY/'nodes.json'
    nodes_sha=hashlib.sha256(nodes_path.read_bytes()).hexdigest()
    if nodes_sha!=summary['nodes_sha256'] or nodes_sha!=audit['input_sha256']['nodes.json']:
        raise ValueError('Verified endpoint nodes changed')
    if summary['inputs']['mass']!=MASS or summary['inputs']['field']!=FIELD:
        raise ValueError('Movie units differ from trajectory units')
    nodes=json.loads(nodes_path.read_text());arrays=np.load(PATHS);records=[]
    for index,exit_kind in zip(INDICES,['transmitted','reflected']):
        control=next(r for r in summary['results']['controls'] if r['master_phase_index']==index)
        fine=next(r for r in control['pair_refinements'] if r['max_step']==STEP)
        matches=[r for r in nodes if r['master_phase_index']==index and r['max_step']==STEP]
        if len(matches)!=1 or control['status']!='passed' or fine['status']!='passed':
            raise ValueError('Required fine Cartesian path was not verified')
        node=matches[0];time=arrays[f'time_{index}_{STEP:g}'];position=arrays[f'positions_{index}_{STEP:g}'];velocity=arrays[f'velocities_{index}_{STEP:g}']
        if node['status']!='outgoing_event' or node['exit']!=exit_kind or time[-1]!=node['time']:
            raise ValueError('Path does not end at its declared outgoing event')
        if position.shape!=(len(time),2,3) or velocity.shape!=position.shape or time[0]!=0 or not np.all(np.diff(time)>0) or not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
            raise ValueError('Invalid saved Cartesian trajectory arrays')
        relative=position[:,0]-position[:,1];radius=np.linalg.norm(relative,axis=1)
        mu=MASS*np.sum(velocity[:,0,:2]**2,axis=1)/(2*FIELD)
        energy=MASS/2*np.sum(velocity**2,axis=(1,2))+summary['inputs']['strength']*np.exp(-radius/summary['inputs']['screening'])/radius
        check={'position_error':float(np.max(abs(relative[-1]-node['final_relative_position']))),
            'mu_increment_error':float(abs(mu[-1]-mu[0]-node['moments'][0])),
            'relative_energy_error':float(np.max(abs(energy-energy[0]))/abs(energy[0])),
            'COM_position_max':float(np.max(abs(position[:,0]+position[:,1]))),
            'COM_velocity_max':float(np.max(abs(velocity[:,0]+velocity[:,1])))}
        if check['position_error']>POSITION_TARGET or check['mu_increment_error']>MU_TARGET or check['relative_energy_error']>ENERGY_TARGET or check['COM_position_max']>POSITION_TARGET or check['COM_velocity_max']>POSITION_TARGET or not mu[0]>0:
            raise ValueError('Saved path failed independent movie-input arithmetic checks')
        records.append({'index':index,'phase':node['phase'],'exit':exit_kind,'time':time,
            'z':relative[:,2],'mu_ratio':mu/mu[0],'initial_mu':float(mu[0]),'check':check})
    return summary,audit,records


inputs={'saved_paths_sha256':PATHS_SHA,'saved_summary_sha256':SUMMARY_SHA,
    'master_phase_indices':INDICES,'saved_individual_path_step':STEP,'mass':MASS,'field':FIELD,
    'relative_outgoing_planes':[-DISTANCE,DISTANCE],'frames':FRAMES,'fps':FPS,'dpi':DPI,
    'moment_definition':'mu_1=m*(v_1x^2+v_1y^2)/(2B); normalize by that path initial mu_1',
    'time_definition':'shared normalized time, revealing original samples up to each frame time',
    'exit_rule':'Each curve and its endpoint marker stop at its own actual exit time. No hold or extrapolated trajectory beyond exit.',
    'position_target':POSITION_TARGET,'mu_increment_target':MU_TARGET,'energy_target':ENERGY_TARGET,
    'scope':'Two individual12D zero-COM paths, not a COM-averaged ensemble, phase-converged integral, collision rate or lifetime.'}
print('Render verified phase0 and π/2 paths; no new trajectory solve or rate claim.',flush=True)
started=perf_counter();summary,audit,records=load_paths()
OUTPUT.mkdir(parents=True,exist_ok=True)
metadata=run_metadata(inputs,model='animation of two verified individual Cartesian encounter paths',
    boundary='each saved trajectory ends at its first outgoing relative z=+/-24 plane',units='normalized')
metadata['trajectory_producer_commit']=summary['commit']
metadata['trajectory_phase_convergence_status']=summary['results']['phase_status']
metadata['pillow_version']=PIL.__version__
metadata['ffmpeg_version']=subprocess.run(['ffmpeg','-version'],capture_output=True,text=True,check=True).stdout.splitlines()[0] if shutil.which('ffmpeg') else None
dependencies=[Path(__file__).resolve(),ROOT/'src/sato_morrison/reference.py']
metadata['experiment_dependency_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
metadata['input_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [PATHS,DIRECTORY/'summary.json',DIRECTORY/'nodes.json',DIRECTORY/'independent_audit.json']}
(OUTPUT/'declared_inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
end=max(r['time'][-1] for r in records);frame_times=np.linspace(0,end,FRAMES)
fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharex=True)
fig.subplots_adjust(left=.08,right=.98,bottom=.26,top=.78,wspace=.25)
fig.suptitle('One incoming triple, two gyrophases',fontsize=17,y=.96)
axes[0].set(xlim=(0,end),ylim=(-27,27),xlabel='normalized time',ylabel='relative separation z',title='Transmitted and reflected paths')
for plane in [-DISTANCE,DISTANCE]:axes[0].axhline(plane,color='0.5',ls=':',lw=1)
ratios=np.concatenate([r['mu_ratio'] for r in records]);pad=.08*(ratios.max()-ratios.min())
axes[1].set(xlim=(0,end),ylim=(ratios.min()-pad,ratios.max()+pad),xlabel='normalized time',ylabel=r'$\mu_1(t)/\mu_1(0)$',title=r'$\mu_1=m(v_{1x}^2+v_{1y}^2)/(2B)$')
lines=[];markers=[];guides=[]
for ax in axes:
    guides.append(ax.axvline(0,color='0.7',ls='--',lw=.8))
    for color,r in zip(COLORS,records):
        label=(r'$\varphi=0$: transmitted' if r['index']==0 else r'$\varphi=\pi/2$: reflected')
        lines.append(ax.plot([],[],color=color,lw=2,label=label)[0])
        markers.append(ax.plot([],[],color=color,marker='o',ms=6,ls='')[0])
    ax.legend(loc='upper left',fontsize=9)
status=fig.text(.5,.16,'',ha='center',fontsize=11)
fig.text(.5,.07,'Individual zero-COM paths; normalized time. No COM average, phase integral or plasma rate.',ha='center',fontsize=10)
printed=set()


def update(frame):
    current=frame_times[frame];messages=[]
    for guide in guides:guide.set_xdata([current,current])
    for index,r in enumerate(records):
        n=np.searchsorted(r['time'],current,side='right');exited=current>=r['time'][-1]
        for axis,quantity in enumerate([r['z'],r['mu_ratio']]):
            artist=2*axis+index
            lines[artist].set_data(r['time'][:n],quantity[:n])
            markers[artist].set_data([r['time'][-1]] if exited else [],[quantity[-1]] if exited else [])
        name='phase 0' if r['index']==0 else 'phase π/2'
        messages.append(f'{name}: exited at t={r["time"][-1]:.2f}' if exited else f'{name}: path shown through t={current:.2f}')
    status.set_text(' | '.join(messages))
    if frame%20==0 and frame not in printed:
        print(f'  render frame {frame+1}/{FRAMES}, normalized time {current:.2f}',flush=True);printed.add(frame)
    return lines+markers+guides+[status]


movie=FuncAnimation(fig,update,frames=FRAMES,interval=1000/FPS,blit=False,repeat=True)
with progress('Render verified paths; each line stops at its own outgoing event'):
    movie.save(OUTPUT/'encounter.gif',writer=PillowWriter(fps=FPS),dpi=DPI)
    mp4_status='ffmpeg_unavailable'
    if shutil.which('ffmpeg') and FFMpegWriter.isAvailable():
        try:
            movie.save(OUTPUT/'encounter.mp4',writer=FFMpegWriter(fps=FPS,codec='libx264',extra_args=['-pix_fmt','yuv420p']),dpi=DPI)
            mp4_status='rendered'
        except (RuntimeError,OSError,subprocess.CalledProcessError) as error:mp4_status='unresolved: '+str(error)
update(FRAMES-1)
fig.savefig(OUTPUT/'poster.png',dpi=180);fig.savefig(OUTPUT/'poster.svg',metadata={'Creator':None,'Date':None});plt.close(fig)
metadata['experiment_dependency_sha256_end']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
unchanged=metadata['experiment_dependency_sha256']==metadata['experiment_dependency_sha256_end']
metadata['results']={'status':'verified_saved_path_render' if unchanged else 'unresolved_source_change',
    'paths':[{'phase':r['phase'],'exit':r['exit'],'exit_time':float(r['time'][-1]),'initial_mu':r['initial_mu'],
        'final_mu_ratio':float(r['mu_ratio'][-1]),'arithmetic_checks':r['check']} for r in records],
    'frames':FRAMES,'fps':FPS,'display_duration_s':FRAMES/FPS,'mp4_status':mp4_status,
    'dependencies_unchanged':unchanged,'wall_s':perf_counter()-started,
    'wall_s_scope':'Input arithmetic validation and movie/poster rendering; no new ODE solve.',
    'scope':inputs['scope']}
metadata['output_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUTPUT.iterdir() if p.suffix in ('.gif','.mp4','.png','.svg')}
(OUTPUT/'summary.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(f'Finished rendering: {metadata["results"]["status"]}; {metadata["results"]["wall_s"]:.1f}s',flush=True)
if not unchanged:raise RuntimeError('Movie source dependencies changed')
