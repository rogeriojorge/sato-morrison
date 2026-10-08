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
SPATIAL_BOX_ASPECT=(1.,1.,1.4)
SPATIAL_CAMERA={'elevation':20.,'azimuth':-55.}
CLOSE_HALF_WIDTH=1.5


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
            'z':relative[:,2],'mu_ratio':mu/mu[0],'initial_mu':float(mu[0]),'check':check,
            'positions':position,'velocities':velocity,'closest_sample_index':int(np.argmin(radius)),
            'sampled_minimum_separation':float(radius.min())})
    arrays.close()
    return summary,audit,records


inputs={'saved_paths_sha256':PATHS_SHA,'saved_summary_sha256':SUMMARY_SHA,
    'master_phase_indices':INDICES,'saved_individual_path_step':STEP,'mass':MASS,'field':FIELD,
    'relative_outgoing_planes':[-DISTANCE,DISTANCE],'frames':FRAMES,'fps':FPS,'dpi':DPI,
    'moment_definition':'mu_1=m*(v_1x^2+v_1y^2)/(2B); normalize by that path initial mu_1',
    'time_definition':'shared normalized time, revealing original samples up to each frame time',
    'exit_rule':'Each curve and its endpoint marker stop at its own actual exit time. No hold or extrapolated trajectory beyond exit.',
    'position_target':POSITION_TARGET,'mu_increment_target':MU_TARGET,'energy_target':ENERGY_TARGET,
    'spatial_view':{'particle_colors':{'particle_1':COLORS[0],'particle_2':COLORS[1]},
        'camera':SPATIAL_CAMERA,'box_aspect':SPATIAL_BOX_ASPECT,
        'axis_scaling':'3D box is anisotropic: z visually compressed; x-z close projection has equal physical length aspect.',
        'coordinate_units':'normalized length, same physical coordinates as saved trajectories',
        'close_projection_half_width':CLOSE_HALF_WIDTH,
        'head_rule':'Last original saved sample at or before the common clock; no interpolation. Head disappears at exit; square marks the actual outgoing endpoint.',
        'closest_rule':'Star marks the closest saved trajectory sample only after its physical time; not a certified continuous minimum.'},
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
fig.subplots_adjust(left=.08,right=.98,bottom=.30,top=.76,wspace=.25)
fig.suptitle('One incoming triple, two gyrophases',fontsize=17,y=.97)
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
fig.legend(handles=lines[:2],loc='upper center',bbox_to_anchor=(.5,.90),ncol=2,fontsize=10)
status=fig.text(.5,.12,'',ha='center',fontsize=11)
fig.text(.5,.035,'Individual zero-COM paths; normalized time. No COM average, phase integral or plasma rate.',ha='center',fontsize=10)
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

# Actual spatial paths: the two columns share physical limits, camera and clock.
# Compress the long z extent visibly and disclose it; the lower projection uses
# equal physical scales, making the close approach readable without distorting it.
xy_limit=1.1*max(float(np.max(abs(r['positions'][:,:,:2]))) for r in records)
z_limit=1.06*DISTANCE/2
spatial_fig=plt.figure(figsize=(10,7.5))
grid=spatial_fig.add_gridspec(2,2,left=.06,right=.96,bottom=.20,top=.88,
    height_ratios=[1.5,1.],hspace=.35,wspace=.20)
spatial_artists=[];spatial_messages=[];closest_labels=[]
spatial_fig.suptitle('Two particles in a uniform magnetic field',fontsize=16,y=.99)
clock=spatial_fig.text(.5,.935,'',ha='center',fontsize=12)
for column,r in enumerate(records):
    ax=spatial_fig.add_subplot(grid[0,column],projection='3d')
    ax.set(xlim=(-xy_limit,xy_limit),ylim=(-xy_limit,xy_limit),zlim=(-z_limit,z_limit),
        xlabel='x',ylabel='y',zlabel='z',title='Phase 0: transmitted' if r['index']==0 else 'Phase π/2: reflected')
    ax.set_box_aspect(SPATIAL_BOX_ASPECT)
    ax.view_init(elev=SPATIAL_CAMERA['elevation'],azim=SPATIAL_CAMERA['azimuth'])
    ax.tick_params(labelsize=8,pad=0)
    ax.xaxis.labelpad=ax.yaxis.labelpad=ax.zaxis.labelpad=1
    ax.set_xticks([-1,0,1]);ax.set_yticks([-1,0,1]);ax.set_zticks([-12,0,12])
    ax.quiver(-.85*xy_limit,.85*xy_limit,5.,0.,0.,4.,color='#253447',
        arrow_length_ratio=.18,lw=1.5)
    ax.text(-.85*xy_limit,.85*xy_limit,9.5,'B = 2; +z',fontsize=9)
    close=spatial_fig.add_subplot(grid[1,column])
    close.set(xlim=(-CLOSE_HALF_WIDTH,CLOSE_HALF_WIDTH),ylim=(-CLOSE_HALF_WIDTH,CLOSE_HALF_WIDTH),
        xlabel='x',ylabel='z',title='Close approach: x–z projection')
    close.set_aspect('equal',adjustable='box');close.grid(alpha=.15)
    close.spines[['top','right']].set_visible(False)
    closest_labels.append(close.text(.03,.97,'',transform=close.transAxes,va='top',
        fontsize=9,bbox={'facecolor':'white','edgecolor':'none','alpha':.85}))
    pair_artists=[]
    for particle,color in enumerate(COLORS):
        line=ax.plot([],[],[],color=color,lw=1.25,label=f'particle {particle+1}')[0]
        head=ax.plot([],[],[],color=color,marker='o',ms=5,ls='')[0]
        endpoint=ax.plot([],[],[],color=color,marker='s',ms=5,ls='')[0]
        initial=r['positions'][0,particle]
        ax.plot([initial[0]],[initial[1]],[initial[2]],color=color,marker='o',
            mfc='white',ms=4,ls='')
        trace=close.plot([],[],color=color,lw=1.25)[0]
        close_head=close.plot([],[],color=color,marker='o',ms=4,ls='')[0]
        closest=close.plot([],[],color=color,marker='*',ms=8,ls='')[0]
        pair_artists.append((line,head,endpoint,trace,close_head,closest))
    spatial_artists.append(pair_artists)
    spatial_messages.append(spatial_fig.text(.27 if column==0 else .73,.135,'',ha='center',fontsize=10))
spatial_fig.legend(handles=[a[0] for a in spatial_artists[0]],loc='upper center',
    bbox_to_anchor=(.5,.035),ncol=2,frameon=False,fontsize=10)
spatial_fig.text(.5,.085,'Coordinates: normalized length. 3D z scale is compressed; close projections have equal scales.',ha='center',fontsize=10)
spatial_fig.text(.5,.058,'Open circle: start. Square: exit. Star: closest saved sample. Trails stop at each actual exit.',ha='center',fontsize=9)
spatial_printed=set()


def set_spatial(artist,points):
    artist.set_data(points[:,0],points[:,1]);artist.set_3d_properties(points[:,2])


def update_spatial(frame):
    current=frame_times[frame];clock.set_text(f'Shared normalized time: {current:.2f}')
    for column,r in enumerate(records):
        n=np.searchsorted(r['time'],current,side='right');exited=current>=r['time'][-1]
        for particle,artists in enumerate(spatial_artists[column]):
            line,head,endpoint,trace,close_head,closest=artists
            points=r['positions'][:n,particle]
            empty=np.empty((0,3))
            set_spatial(line,points)
            set_spatial(head,points[-1:] if n and not exited else empty)
            set_spatial(endpoint,r['positions'][-1:,particle] if exited else empty)
            inside=abs(points[:,2])<=CLOSE_HALF_WIDTH
            trace.set_data(np.where(inside,points[:,0],np.nan),np.where(inside,points[:,2],np.nan))
            show_head=n and not exited and inside[-1]
            close_head.set_data([points[-1,0]] if show_head else [],[points[-1,2]] if show_head else [])
            seen_closest=n>r['closest_sample_index']
            p=r['positions'][r['closest_sample_index'],particle]
            closest.set_data([p[0]] if seen_closest else [],[p[2]] if seen_closest else [])
        if exited:
            message=f'Exited at t = {r["time"][-1]:.2f}; trajectory ends'
        else:
            message=f'Last saved sample: t = {r["time"][n-1]:.2f}' if n else 'No saved sample yet'
        spatial_messages[column].set_text(message)
        closest_labels[column].set_text(f'Closest saved separation: {r["sampled_minimum_separation"]:.4g}'
            if n>r['closest_sample_index'] else '')
    if frame%20==0 and frame not in spatial_printed:
        print(f'  spatial frame {frame+1}/{FRAMES}, normalized time {current:.2f}',flush=True)
        spatial_printed.add(frame)
    return [clock,*spatial_messages,*closest_labels,*[artist for pair in spatial_artists for group in pair for artist in group]]


spatial_movie=FuncAnimation(spatial_fig,update_spatial,frames=FRAMES,
    interval=1000/FPS,blit=False,repeat=True)
with progress('Render individual spatial paths with a shared clock and fixed view'):
    spatial_movie.save(OUTPUT/'encounter_spatial.gif',writer=PillowWriter(fps=FPS),dpi=DPI)
    spatial_mp4_status='ffmpeg_unavailable'
    if shutil.which('ffmpeg') and FFMpegWriter.isAvailable():
        try:
            spatial_movie.save(OUTPUT/'encounter_spatial.mp4',writer=FFMpegWriter(fps=FPS,
                codec='libx264',extra_args=['-pix_fmt','yuv420p']),dpi=DPI)
            spatial_mp4_status='rendered'
        except (RuntimeError,OSError,subprocess.CalledProcessError) as error:
            spatial_mp4_status='unresolved: '+str(error)
update_spatial(FRAMES-1)
spatial_fig.savefig(OUTPUT/'spatial_poster.png',dpi=180)
spatial_fig.savefig(OUTPUT/'spatial_poster.svg',metadata={'Creator':None,'Date':None})
plt.close(spatial_fig)
metadata['experiment_dependency_sha256_end']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
unchanged=metadata['experiment_dependency_sha256']==metadata['experiment_dependency_sha256_end']
metadata['results']={'status':'verified_saved_path_render' if unchanged else 'unresolved_source_change',
    'paths':[{'phase':r['phase'],'exit':r['exit'],'exit_time':float(r['time'][-1]),'initial_mu':r['initial_mu'],
        'final_mu_ratio':float(r['mu_ratio'][-1]),'arithmetic_checks':r['check']} for r in records],
    'frames':FRAMES,'fps':FPS,'display_duration_s':FRAMES/FPS,'mp4_status':mp4_status,
    'spatial_mp4_status':spatial_mp4_status,
    'spatial_view':{'xy_limits':[-xy_limit,xy_limit],'z_limits':[-z_limit,z_limit],
        'box_aspect':SPATIAL_BOX_ASPECT,'camera':SPATIAL_CAMERA,
        'z_unit_visual_size_relative_to_x':SPATIAL_BOX_ASPECT[2]/SPATIAL_BOX_ASPECT[0]*xy_limit/z_limit,
        'close_projection_limits':[-CLOSE_HALF_WIDTH,CLOSE_HALF_WIDTH],
        'close_projection_equal_aspect':True,'sample_reveal_rule':'searchsorted(time,current,side=right); original saved points only',
        'frame_times':frame_times.tolist(),
        'paths':[{'phase':r['phase'],'sample_count':len(r['time']),
            'position_min':r['positions'].min(axis=(0,1)).tolist(),
            'position_max':r['positions'].max(axis=(0,1)).tolist(),
            'closest_saved_sample_index':r['closest_sample_index'],
            'closest_saved_sample_time':float(r['time'][r['closest_sample_index']]),
            'sampled_minimum_separation':r['sampled_minimum_separation']} for r in records]},
    'dependencies_unchanged':unchanged,'wall_s':perf_counter()-started,
    'wall_s_scope':'Input arithmetic validation and movie/poster rendering; no new ODE solve.',
    'scope':inputs['scope']}
metadata['output_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUTPUT.iterdir() if p.suffix in ('.gif','.mp4','.png','.svg')}
(OUTPUT/'summary.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(f'Finished rendering: {metadata["results"]["status"]}; {metadata["results"]["wall_s"]:.1f}s',flush=True)
if not unchanged:raise RuntimeError('Movie source dependencies changed')
