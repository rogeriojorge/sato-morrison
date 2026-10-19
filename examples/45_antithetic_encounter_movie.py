"""Actual Cartesian paths for one fixed nonlinear phase pair."""
from pathlib import Path
from time import perf_counter
import hashlib
import json
import os
import resource
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import PIL
from PIL import Image
from sato_morrison.controls import encounter_relative, binary_encounter
from sato_morrison.reference import run_metadata

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/os.environ.get('SM_ANTITHETIC_MOVIE_OUTPUT','results/antithetic_encounter_movie')
B,U,P,PHASE=1.2161710500723555,1.546221967601278,2.406217687668966,1.5892040962495242
PHASES=(PHASE,float((np.pi-PHASE)%(2*np.pi)))
MASS,CHARGE,FIELD,STRENGTH,SCREENING,DISTANCE=1.,1.,2.,.03,3.,24.
STEP,RTOL,ENERGY_TARGET,ENDPOINT_TARGET=.0125,2e-13,1e-9,1e-8
FRAMES,FPS,DPI=120,10,120
PARTICLE_COLORS=('#147D92','#C66B28')
CASE_COLORS=('#305EA8','#7B3B8C')
CLOSE_HALF_WIDTH=.75
INPUTS=dict(impact=B,parallel_speed=U,perpendicular_speed=P,phases=PHASES,
    mass=MASS,charge=CHARGE,field=FIELD,strength=STRENGTH,screening=SCREENING,
    start_distance=DISTANCE,max_step=STEP,rtol=RTOL,energy_target=ENERGY_TARGET,
    endpoint_target=ENDPOINT_TARGET,frames=FRAMES,fps=FPS,dpi=DPI,
    camera=dict(elevation=22,azimuth=-55,box_aspect=(1,1,1.4)),
    close_projection_half_width=CLOSE_HALF_WIDTH,
    units='normalized length, velocity, time and moment; mu=m*v_perp^2/(2B)',
    selection='Fixed phase pair from archived example41 peak; no replacement based on outcomes.',
    COM='Position and velocity center of mass zero in each independent physical experiment.',
    clock='One shared physical clock. 84 of 120 frames span the two closest saved times plus/minus0.5; playback slows there. Only original Cartesian samples at or before each clock time are shown.',
    exit_rule='Each Cartesian integration stops at its own relative-solver outgoing-event time; endpoint position and velocity must independently agree. No extrapolation or synthetic motion beyond exit.',
    scope='Two individual zero-COM physical experiments; no thermal COM average, ensemble quadrature, flux reweighting, physical rate or lifetime.')
if OUTPUT.exists():
    raise FileExistsError(f'Preserve existing evidence: {OUTPUT}; choose a fresh output directory')
OUTPUT.mkdir(parents=True)
(OUTPUT/'declared_inputs.json').write_text(json.dumps(INPUTS,indent=2)+'\n')
started=perf_counter()
metadata=run_metadata(INPUTS,model='fixed phase-pair Cartesian encounter movie',
    boundary='first outgoing relative z=+/-24 plane',units=INPUTS['units'])
dependencies=[Path(__file__).resolve(),ROOT/'src/sato_morrison/controls.py',ROOT/'src/sato_morrison/reference.py']
def hashes(paths):return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
metadata['experiment_dependency_sha256']=hashes(dependencies)
records=[];arrays={};failures=[]
for index,phase in enumerate(PHASES):
    print(f'Integrate fixed phase {phase:.16g}',flush=True)
    try:
        relative=encounter_relative(B,phase,U,P,field=FIELD,strength=STRENGTH,
            screening=SCREENING,start_distance=DISTANCE,mass=MASS,charge=CHARGE,
            max_step=STEP,rtol=RTOL,flight_time_factor=48.)
        path=binary_encounter(relative['initial_relative_position'],relative['initial_relative_velocity'],
            field=FIELD,strength=STRENGTH,screening=SCREENING,mass=MASS,charge=CHARGE,
            duration=relative['time'],max_step=STEP,rtol=RTOL)
        t=path['time'];positions=path['positions'];velocities=path['velocities']
        final_r=positions[-1,0]-positions[-1,1]
        final_w=velocities[-1,0]-velocities[-1,1]
        separation=positions[:,0]-positions[:,1]
        radius=np.linalg.norm(separation,axis=1)
        mu=MASS*np.sum(velocities[:,0,:2]**2,axis=1)/(2*FIELD)
        checks=dict(energy=path['energy_error']<=ENERGY_TARGET,
            relative_energy=relative['energy_error']<=ENERGY_TARGET,
            endpoint_position=float(np.max(abs(final_r-relative['final_relative_position'])))<=ENDPOINT_TARGET,
            endpoint_velocity=float(np.max(abs(final_w-relative['final_relative_velocity'])))<=ENDPOINT_TARGET,
            outgoing_plane=abs(abs(final_r[2])-DISTANCE)<=ENDPOINT_TARGET,
            COM=bool(np.max(abs(positions.sum(axis=1)))<=ENDPOINT_TARGET and np.max(abs(velocities.sum(axis=1)))<=ENDPOINT_TARGET),
            finite_ordered=bool(t[0]==0 and np.all(np.diff(t)>0) and np.all(np.isfinite(positions)) and np.all(np.isfinite(velocities))))
        record=dict(index=index,phase=phase,exit=relative['exit'],exit_time=float(t[-1]),
            relative_energy_error=relative['energy_error'],cartesian_energy_error=path['energy_error'],
            endpoint_position_error=float(np.max(abs(final_r-relative['final_relative_position']))),
            endpoint_velocity_error=float(np.max(abs(final_w-relative['final_relative_velocity']))),
            sampled_minimum_separation=float(radius.min()),closest_sample_index=int(np.argmin(radius)),
            closest_sample_time=float(t[np.argmin(radius)]),initial_mu=float(mu[0]),final_mu=float(mu[-1]),
            delta_mu=float(mu[-1]-mu[0]),checks=checks)
        records.append(record)
        arrays.update({f'time_{index}':t,f'positions_{index}':positions,f'velocities_{index}':velocities,f'mu_{index}':mu})
        if not all(checks.values()):failures.append(record)
    except Exception as error:
        failures.append(dict(index=index,phase=phase,error_type=type(error).__name__,reason=str(error)))
np.savez_compressed(OUTPUT/'paths.npz',**arrays)
trajectory_wall=perf_counter()-started
(OUTPUT/'trajectory_checks.json').write_text(json.dumps(dict(records=records,failures=failures),indent=2)+'\n')
metadata['results']=dict(status='unresolved' if failures else 'passed',records=records,failures=failures,
    trajectory_wall_s=trajectory_wall,new_relative_solves=2,new_cartesian_solves=len(records))
metadata['experiment_dependency_sha256_end']=hashes(dependencies)
if metadata['experiment_dependency_sha256']!=metadata['experiment_dependency_sha256_end']:
    failures.append(dict(reason='source changed during execution'))
if failures:
    metadata['results']['status']='unresolved'
    (OUTPUT/'summary.json').write_text(json.dumps(metadata,indent=2)+'\n')
    raise RuntimeError('Fixed movie trajectories failed declared checks; saved evidence retained')

plt.rcParams.update({'font.size':11,'axes.titlesize':12,'svg.fonttype':'path'})
fig=plt.figure(figsize=(10.3,8.1),layout='constrained')
grid=fig.add_gridspec(3,2,height_ratios=(1.35,1.,.95))
spatial=[];close=[];artists=[]
xy=1.08*max(float(np.max(abs(arrays[f'positions_{i}'][:,:,:2]))) for i in range(2))
for i,record in enumerate(records):
    ax=fig.add_subplot(grid[0,i],projection='3d')
    ax.set(xlim=(-xy,xy),ylim=(-xy,xy),zlim=(-12.7,12.7),xlabel='x',ylabel='y',zlabel='z',
        title=f'Phase {record["phase"]:.5f} rad: {record["exit"]}')
    ax.set_box_aspect((1,1,1.4));ax.view_init(elev=22,azim=-55)
    ax.tick_params(labelsize=9,pad=0)
    cx=fig.add_subplot(grid[1,i]);cx.set(xlim=(-CLOSE_HALF_WIDTH,CLOSE_HALF_WIDTH),ylim=(-CLOSE_HALF_WIDTH,CLOSE_HALF_WIDTH),
        xlabel='x (normalized)',ylabel='z (normalized)',title='Close approach: equal spatial scales')
    cx.set_aspect('equal');cx.grid(alpha=.15)
    pairs=[]
    for particle,color in enumerate(PARTICLE_COLORS):
        line=ax.plot([],[],[],color=color,lw=1.1,label=f'particle {particle+1}')[0]
        head=ax.plot([],[],[],color=color,marker='o',ms=4,ls='')[0]
        trace=cx.plot([],[],color=color,lw=1.2)[0]
        marker=cx.plot([],[],color=color,marker='o',ms=4,ls='')[0]
        pairs.append((line,head,trace,marker))
    artists.append(pairs)
muax=fig.add_subplot(grid[2,:])
muax.set(xlim=(0,max(r['exit_time'] for r in records)),xlabel='shared normalized time',ylabel=r'$\mu_1(t)/\mu_1(0)$')
mu_lines=[]
all_mu=[]
for i,r in enumerate(records):
    mu_lines.append(muax.plot([],[],color=CASE_COLORS[i],lw=1.7,label=f'phase {r["phase"]:.5f} rad')[0])
    all_mu.extend((arrays[f'mu_{i}']/r['initial_mu']).tolist())
span=max(all_mu)-min(all_mu);muax.set_ylim(min(all_mu)-.05*span,max(all_mu)+.05*span)
fig.legend(handles=[item[0] for item in artists[0]],loc='upper center',bbox_to_anchor=(.5,.955),ncol=2,fontsize=10,frameon=False)
muax.legend(ncol=2,fontsize=10);guide=muax.axvline(0,color='.6',ls=':',lw=1)
clock=fig.suptitle('',fontsize=13)
fig.supxlabel('Zero COM; two physical experiments. 3D z compressed; close views equal scale.\nSquares: exits. Playback slows near encounter; traces use original samples only.',fontsize=10)
end=max(r['exit_time'] for r in records)
a=max(0,min(r['closest_sample_time'] for r in records)-.5)
b=min(end,max(r['closest_sample_time'] for r in records)+.5)
frame_times=np.concatenate((np.linspace(0,a,18,endpoint=False),np.linspace(a,b,84,endpoint=False),np.linspace(b,end,18)))
frame_hashes={}
def update(frame):
    current=frame_times[frame];clock.set_text(f'Fixed nonlinear phase pair | normalized time {current:.4f}')
    guide.set_xdata([current,current])
    for i,r in enumerate(records):
        t=arrays[f'time_{i}'];pos=arrays[f'positions_{i}'];n=np.searchsorted(t,current,side='right')
        exited=current>=t[-1]
        for particle,(line,head,trace,marker) in enumerate(artists[i]):
            points=pos[:n,particle];line.set_data(points[:,0],points[:,1]);line.set_3d_properties(points[:,2])
            last=points[-1:] if n else np.empty((0,3))
            head.set_data(last[:,0],last[:,1]);head.set_3d_properties(last[:,2]);head.set_marker('s' if exited else 'o')
            trace.set_data(points[:,0],points[:,2]);marker.set_data(last[:,0],last[:,2]);marker.set_marker('s' if exited else 'o')
        mu_lines[i].set_data(t[:n],arrays[f'mu_{i}'][:n]/r['initial_mu'])
    return current
writer=PillowWriter(fps=FPS)
with writer.saving(fig,str(OUTPUT/'phase_pair.gif'),dpi=DPI):
    for frame in range(FRAMES):
        update(frame);fig.canvas.draw()
        frame_hashes[str(frame)]=hashlib.sha256(fig.canvas.buffer_rgba()).hexdigest()
        writer.grab_frame()
        if frame%20==0:print(f'Render frame {frame+1}/{FRAMES}',flush=True)
update(FRAMES-1)
fig.savefig(OUTPUT/'poster.png',dpi=160)
fig.savefig(OUTPUT/'poster.svg',metadata={'Creator':None,'Date':None})
plt.close(fig)
with Image.open(OUTPUT/'phase_pair.gif') as gif:
    decoded=[]
    for frame in range(gif.n_frames):
        gif.seek(frame);decoded.append(hashlib.sha256(gif.convert('RGBA').tobytes()).hexdigest())
metadata['results'].update(render_wall_s=perf_counter()-started-trajectory_wall,
    total_wall_s=perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
    memory_scope='single process lifetime peak; trajectory arrays and frame encoding included',
    frame_times=frame_times.tolist(),canvas_rgba_sha256=frame_hashes,gif_decoded_frame_sha256=decoded,
    encoded_frames=len(decoded),pillow_version=PIL.__version__)
metadata['output_sha256']=hashes([OUTPUT/p for p in ('paths.npz','trajectory_checks.json','phase_pair.gif','poster.png','poster.svg')])
(OUTPUT/'summary.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(f'Finished fixed-pair movie in {perf_counter()-started:.2f}s; all trajectory checks pass',flush=True)
