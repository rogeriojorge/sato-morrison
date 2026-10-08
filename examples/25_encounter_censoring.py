"""Locate and verify finite-budget censoring between opposite plane exits."""
from pathlib import Path
from time import perf_counter
import hashlib,json
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sato_morrison.controls import encounter_relative,encounter_incoming,binary_encounter
from sato_morrison.reference import run_metadata,progress

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'results'/'encounter_censoring'
PRIOR=ROOT/'results'/'encounter_phase'/'summary.json'
PRIOR_SHA='33f5e206d6a748518abe6808945b96d003b4e2fef283d318bd63f04973d2812e'
MASS,CHARGE,FIELD,STRENGTH,SCREENING=1.,1.,2.,.03,3.
IMPACT,PARALLEL,PERPENDICULAR=.9938225266093732,1.9056642462041216,1.9668376625283577
DISTANCE,FLIGHT_FACTOR,MAX_STEP,RTOL=24.,12.,.05,1e-10
LEFT,RIGHT=127*2*np.pi/512,np.pi/2
MAX_BISECTIONS,PHASE_WIDTH_TARGET=20,1e-6
REFINED_STEP=.025
LARGER_FACTORS=(24.,48.)
POSITION_TARGET,VELOCITY_TARGET,ENERGY_TARGET=1e-8,1e-8,1e-8


def recorded_path(phase,duration,step):
    """Record a 6D path to a prescribed time; same equations/method, no event claim."""
    omega=CHARGE*FIELD/MASS
    position,velocity=encounter_incoming(IMPACT,phase,PARALLEL,PERPENDICULAR,DISTANCE,omega)
    def rhs(t,y):
        radius=np.linalg.norm(y[:3])
        force=2*STRENGTH/MASS*np.exp(-radius/SCREENING)*(1+radius/SCREENING)/radius**3
        acceleration=force*y[:3]+omega*np.array([y[4],-y[3],0.])
        return np.concatenate((y[3:],acceleration))
    path=solve_ivp(rhs,(0.,duration),np.concatenate((position,velocity)),method='DOP853',
        rtol=RTOL,atol=RTOL*.01,max_step=step)
    if not path.success or not np.all(np.isfinite(path.y)):
        raise RuntimeError('Recorded path integration failed: '+path.message)
    radius=np.linalg.norm(path.y[:3],axis=0)
    energy=MASS/4*np.sum(path.y[3:]**2,axis=0)+STRENGTH*np.exp(-radius/SCREENING)/radius
    error=float(np.max(abs(energy-energy[0]))/abs(energy[0]))
    return path.t,path.y.T,error


def attempt(phase,step,factor):
    begin=perf_counter();budget=factor*DISTANCE/PARALLEL
    row={'phase':float(phase),'max_step':step,'flight_time_factor':factor,'flight_budget':budget}
    try:
        result=encounter_relative(IMPACT,phase,PARALLEL,PERPENDICULAR,field=FIELD,
            strength=STRENGTH,screening=SCREENING,start_distance=DISTANCE,mass=MASS,
            charge=CHARGE,max_step=step,rtol=RTOL,flight_time_factor=factor)
        row.update(status='outgoing_event',exit=result['exit'],time=result['time'],
            energy_error=result['energy_error'],evaluations=result['evaluations'],
            final_position=np.asarray(result['final_relative_position']).tolist(),
            final_velocity=np.asarray(result['final_relative_velocity']).tolist())
    except (RuntimeError,ValueError,FloatingPointError) as error:
        diagnostics=getattr(error,'integration_diagnostics',{})
        valid=(diagnostics.get('solver_success') is True and diagnostics.get('solver_status')==0
            and diagnostics.get('final_time')==budget
            and np.all(np.isfinite(diagnostics.get('final_position',[np.nan])))
            and np.all(np.isfinite(diagnostics.get('final_velocity',[np.nan])))
            and abs(diagnostics.get('final_position',[0.,0.,np.inf])[2])<DISTANCE)
        row.update(status='finite_budget_nonexit' if valid else 'integration_unresolved',
            reason=str(error),integration_diagnostics=diagnostics)
        if valid:
            row.update(time=budget,final_position=diagnostics['final_position'],
                final_velocity=diagnostics['final_velocity'],evaluations=diagnostics['evaluations'])
    if row['status']!='integration_unresolved':
        try:
            time,state,error=recorded_path(phase,row['time'],step)
            row['recorded_path_energy_error']=error
            row['recorded_path_position_error']=float(np.max(abs(state[-1,:3]-row['final_position'])))
            row['recorded_path_velocity_error']=float(np.max(abs(state[-1,3:]-row['final_velocity'])))
            row['recorded_path_check']='passed' if error<=ENERGY_TARGET and row['recorded_path_position_error']<=POSITION_TARGET and row['recorded_path_velocity_error']<=VELOCITY_TARGET else 'unresolved'
            row['energy_error']=max(error,row.get('energy_error',0.))
        except (RuntimeError,ValueError,FloatingPointError) as error:
            row['recorded_path_check']='unresolved';row['recorded_path_reason']=str(error)
            time,state=np.array([]),np.empty((0,6))
    else:time,state=np.array([]),np.empty((0,6))
    row['wall_s']=perf_counter()-begin
    return row,time,state


inputs={'mass':MASS,'charge':CHARGE,'field':FIELD,'strength':STRENGTH,'screening':SCREENING,
    'fixed_incoming_triple':[IMPACT,PARALLEL,PERPENDICULAR],'phase_definition':'free incoming helix at z=0',
    'phase_bracket':[LEFT,RIGHT],'prior_summary_sha256':PRIOR_SHA,
    'start_distance':DISTANCE,'flight_time_factor':FLIGHT_FACTOR,'max_step':MAX_STEP,'rtol':RTOL,
    'max_bisections':MAX_BISECTIONS,'phase_width_target':PHASE_WIDTH_TARGET,
    'search_rule':'Bisect opposite outgoing-exit endpoints; stop at first numerical nonexit, failed solve, width target or attempt cap.',
    'refined_step':REFINED_STEP,'larger_flight_factors':LARGER_FACTORS,
    'position_absolute_target':POSITION_TARGET,'velocity_absolute_target':VELOCITY_TARGET,'energy_target':ENERGY_TARGET,
    'diagnostic_pass_rule':'Opposite initial exits; found nonexit retained at .025; recorded-path checks, half-step and independent12D endpoint errors below targets; unchanged source dependencies.',
    'limits':'No estimate of censor interval width, phase integral, asymptotic trapping, plasma rate or lifetime; larger-budget outcomes remain diagnostic. An outgoing-scattering integral is not defined on infinite-time nonexit phases without an extension convention or an almost-everywhere finite-exit result.'}
if hashlib.sha256(PRIOR.read_bytes()).hexdigest()!=PRIOR_SHA:raise ValueError('Prior phase receipt changed')
prior=json.loads(PRIOR.read_text());prior_nodes_path=PRIOR.with_name('nodes.json')
prior_nodes_sha=hashlib.sha256(prior_nodes_path.read_bytes()).hexdigest()
if prior_nodes_sha!=prior['nodes_sha256']:raise ValueError('Prior phase nodes changed')
prior_nodes=json.loads(prior_nodes_path.read_text());prior_endpoints=[]
for index,phase,exit_kind in [(127,LEFT,'transmitted'),(128,RIGHT,'reflected')]:
    matches=[r for r in prior_nodes if r['master_phase_index']==index and r['max_step']==MAX_STEP]
    if len(matches)!=1 or matches[0]['phase']!=phase or matches[0]['status']!='outgoing_event' or matches[0]['exit']!=exit_kind:
        raise ValueError('Prior declared bracket endpoint is inconsistent')
    prior_endpoints.append(matches[0])
OUTPUT.mkdir(parents=True,exist_ok=True)
metadata=run_metadata(inputs,model='fixed-triple finite-flight-budget exit-basin diagnostic',
    boundary='relative outgoing planes z=+/-24; fixed and extended finite flight budgets',units='normalized')
metadata['prior_evidence']={'summary_path':str(PRIOR.relative_to(ROOT)),'summary_sha256':PRIOR_SHA,
    'nodes_path':str(prior_nodes_path.relative_to(ROOT)),'nodes_sha256':prior_nodes_sha,
    'producer_commit':prior['commit'],'declared_endpoint_rows':prior_endpoints}
dependencies=[Path(__file__).resolve(),ROOT/'src/sato_morrison/controls.py',ROOT/'src/sato_morrison/reference.py']
metadata['experiment_dependency_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
(OUTPUT/'declared_inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
started=perf_counter();rows=[];paths={}


def evaluate(phase,step,factor,label):
    row,time,state=attempt(phase,step,factor);row['label']=label;rows.append(row)
    if len(time):paths.update({f'{label}_time':time,f'{label}_state':state})
    (OUTPUT/'attempts_partial.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(OUTPUT/'paths_partial.npz',**paths)
    print(f'  {label}: phase={phase:.15g}, {row["status"]}, exit={row.get("exit")}, time={row.get("time")}',flush=True)
    return row


found=None;search_reason='initial_bracket_unresolved';left,right=LEFT,RIGHT
with progress('Bisect the declared opposite-exit bracket; keep every nonexit and failure'):
    a=evaluate(left,MAX_STEP,FLIGHT_FACTOR,'left_endpoint')
    b=evaluate(right,MAX_STEP,FLIGHT_FACTOR,'right_endpoint')
    bracket_ok=a.get('exit')=='transmitted' and b.get('exit')=='reflected'
    if bracket_ok:
        search_reason='attempt_cap'
        for index in range(MAX_BISECTIONS):
            if right-left<=PHASE_WIDTH_TARGET:search_reason='width_cap';break
            phase=(left+right)/2
            row=evaluate(phase,MAX_STEP,FLIGHT_FACTOR,f'bisection_{index:02d}')
            if row['status']=='finite_budget_nonexit':found=row;search_reason='nonexit_found';break
            if row['status']!='outgoing_event' or row.get('recorded_path_check')!='passed':search_reason='integration_unresolved';break
            if row['exit']=='transmitted':left=phase
            elif row['exit']=='reflected':right=phase
            else:search_reason='invalid_exit';break

fine=None;pair_check=None;extensions=[]
if found is not None:
    with progress('Replay the same nonexit at half timestep and in individual Cartesian coordinates'):
        fine=evaluate(found['phase'],REFINED_STEP,FLIGHT_FACTOR,'nonexit_half_step')
        if fine['status']=='finite_budget_nonexit':
            position,velocity=encounter_incoming(IMPACT,found['phase'],PARALLEL,PERPENDICULAR,DISTANCE,CHARGE*FIELD/MASS)
            try:
                pair=binary_encounter(position,velocity,field=FIELD,strength=STRENGTH,
                    mass=MASS,charge=CHARGE,screening=SCREENING,duration=fine['flight_budget'],max_step=REFINED_STEP,rtol=RTOL)
                r=pair['positions'][-1,0]-pair['positions'][-1,1]
                w=pair['velocities'][-1,0]-pair['velocities'][-1,1]
                pair_check={'position_error':float(np.max(abs(r-fine['final_position']))),
                    'velocity_error':float(np.max(abs(w-fine['final_velocity']))),
                    'energy_error':pair['energy_error'],'final_relative_position':r.tolist(),
                    'final_relative_velocity':w.tolist(),'evaluations':pair['evaluations']}
                pair_check['status']='passed' if pair_check['position_error']<=POSITION_TARGET and pair_check['velocity_error']<=VELOCITY_TARGET and pair_check['energy_error']<=ENERGY_TARGET and abs(r[2])<DISTANCE else 'unresolved'
                paths.update(pair_time=pair['time'],pair_positions=pair['positions'],pair_velocities=pair['velocities'])
            except (RuntimeError,ValueError,FloatingPointError) as error:
                pair_check={'status':'unresolved','reason':str(error)}
    with progress('Extend only the selected phase flight budget; preserve each classification'):
        for factor in LARGER_FACTORS:
            extensions.append(evaluate(found['phase'],REFINED_STEP,factor,f'extended_{int(factor)}'))

half_check=None
if found is not None and fine is not None and fine['status']=='finite_budget_nonexit':
    half_check={'position_error':float(np.max(abs(np.asarray(found['final_position'])-fine['final_position']))),
        'velocity_error':float(np.max(abs(np.asarray(found['final_velocity'])-fine['final_velocity'])))}
    half_check['status']='passed' if half_check['position_error']<=POSITION_TARGET and half_check['velocity_error']<=VELOCITY_TARGET else 'unresolved'
metadata['experiment_dependency_sha256_end']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
unchanged=metadata['experiment_dependency_sha256']==metadata['experiment_dependency_sha256_end']
passed=(bracket_ok and found is not None and fine is not None and fine['status']=='finite_budget_nonexit'
    and half_check['status']=='passed' and pair_check is not None and pair_check['status']=='passed'
    and all(r.get('recorded_path_check')=='passed' for r in rows) and unchanged)
results={'status':'passed_finite_budget_diagnostic' if passed else 'unresolved',
    'search_stop':search_reason,'last_search_bracket':[left,right],
    'last_search_width':right-left,'found_phase':None if found is None else found['phase'],
    'half_step_check':half_check,'independent_12D_check':pair_check,'extended_budget_results':extensions,
    'attempts':rows,'dependencies_unchanged':unchanged,'wall_s':perf_counter()-started,
    'wall_s_scope':'Numerical solves/checks and intermediate checkpoint writes; excludes imports, metadata initialization, plots and final output.',
    'scope':'Finite-budget nonexit is censoring, not a trapping claim. The last search bracket is not a censored-interval width or coverage bound. The original phase-integral result remains unresolved. Classifications are floating-point ODE evidence, not interval-certified trajectories. Bounds on eventual outgoing moments do not define a scattering integral on phases with no finite exit.'}
metadata['results']=results
np.savez_compressed(OUTPUT/'paths.npz',**paths)
metadata['paths_sha256']=hashlib.sha256((OUTPUT/'paths.npz').read_bytes()).hexdigest()
(OUTPUT/'summary.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
path_styles=[('left_endpoint','transmitted neighbor','-'),
    ('right_endpoint','reflected neighbor','-'),('nonexit_half_step','same phase: budget T','-'),
    ('extended_24','same phase: budget 2T','--'),('extended_48','same phase: budget 4T',':')]
for label,title,style in path_styles:
    if f'{label}_time' in paths:
        axes[0].plot(paths[f'{label}_time'],paths[f'{label}_state'][:,2],ls=style,label=title)
for plane in [-DISTANCE,DISTANCE]:axes[0].axhline(plane,color='0.5',ls=':',lw=1)
axes[0].set(xlabel='time',ylabel='relative z',title='Fixed planes; finite flight budgets');axes[0].legend(fontsize=8)
colors={'transmitted':'#147D92','reflected':'#C66B28','finite_budget_nonexit':'#873DA4','integration_unresolved':'0.5'}
for row in rows:
    if row['flight_time_factor']==FLIGHT_FACTOR:
        label=row.get('exit',row['status'])
        axes[1].scatter(row['phase'],row.get('time',row['flight_budget']),color=colors[label],marker='x' if row['status']!='outgoing_event' else 'o')
axes[1].axhline(FLIGHT_FACTOR*DISTANCE/PARALLEL,color='0.5',ls=':')
legend=[Line2D([],[],color=colors['transmitted'],marker='o',ls='',label='transmitted'),
    Line2D([],[],color=colors['reflected'],marker='o',ls='',label='reflected'),
    Line2D([],[],color=colors['finite_budget_nonexit'],marker='x',ls='',label='not exited by T'),
    Line2D([],[],color='0.5',ls=':',label='original flight budget T')]
axes[1].set(xlabel='incoming phase',ylabel='exit or censored time',title='Declared opposite-exit bracket',xlim=(LEFT,RIGHT));axes[1].legend(handles=legend,fontsize=8)
subtitle='finite-budget diagnostic passed' if passed else 'finite-budget diagnostic unresolved'
fig.suptitle('A phase grid can miss slow encounters\n'+subtitle+'; no trapping or phase-integral conclusion',fontsize=11)
fig.savefig(OUTPUT/'finite_budget_censoring.png',dpi=180);plt.close(fig)
print(f'Finished: {results["status"]}; {results["wall_s"]:.1f}s',flush=True)
if not passed:raise RuntimeError('Finite-budget diagnostic unresolved; all attempts retained')
