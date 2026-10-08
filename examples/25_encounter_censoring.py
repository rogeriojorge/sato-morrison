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
OUTPUT=ROOT/'results'/'encounter_censoring'/'precision_checked'
PRIOR=ROOT/'results'/'encounter_censoring'/'summary.json'
PRIOR_SHA='13a304f77197aca56fc0e8d96d480d9a22d47d3a7a1996c20b5cc84524b5b158'
MASS,CHARGE,FIELD,STRENGTH,SCREENING=1.,1.,2.,.03,3.
IMPACT,PARALLEL,PERPENDICULAR=.9938225266093732,1.9056642462041216,1.9668376625283577
DISTANCE,FLIGHT_FACTOR,MAX_STEP,RTOL=24.,12.,.025,1e-12
LEFT,RIGHT=1.5610876065934445,1.561088355607501
MAX_BISECTIONS,INTERIOR_TARGET=40,DISTANCE/2
REFINED_STEP=.0125
LARGER_FACTORS=(24.,48.)
PRECISION_TOLERANCES=(1e-12,1e-13)
POSITION_TARGET,VELOCITY_TARGET,ENERGY_TARGET=1e-8,1e-8,1e-8


def recorded_path(phase,duration,step,*,rtol=RTOL):
    """Record a 6D path to a prescribed time; same equations/method, no event claim."""
    omega=CHARGE*FIELD/MASS
    position,velocity=encounter_incoming(IMPACT,phase,PARALLEL,PERPENDICULAR,DISTANCE,omega)
    def rhs(t,y):
        radius=np.linalg.norm(y[:3])
        force=2*STRENGTH/MASS*np.exp(-radius/SCREENING)*(1+radius/SCREENING)/radius**3
        acceleration=force*y[:3]+omega*np.array([y[4],-y[3],0.])
        return np.concatenate((y[3:],acceleration))
    path=solve_ivp(rhs,(0.,duration),np.concatenate((position,velocity)),method='DOP853',
        rtol=rtol,atol=rtol*.01,max_step=step)
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


def full_budget_attempt(phase,step,factor):
    """Unabsorbed IVP supplies the continuous search observable F(phi)=z(T)."""
    begin=perf_counter();budget=factor*DISTANCE/PARALLEL
    row={'phase':float(phase),'max_step':step,'flight_time_factor':factor,'flight_budget':budget}
    try:
        time,state,error=recorded_path(phase,budget,step)
        row.update(status='full_budget_path',time=float(time[-1]),search_F=float(state[-1,2]),
            final_position=state[-1,:3].tolist(),final_velocity=state[-1,3:].tolist(),
            energy_error=error,recorded_path_energy_error=error,
            recorded_path_check='passed' if error<=ENERGY_TARGET else 'unresolved')
    except (RuntimeError,ValueError,FloatingPointError) as error:
        row.update(status='integration_unresolved',reason=str(error),recorded_path_check='unresolved')
        time,state=np.array([]),np.empty((0,6))
    row['wall_s']=perf_counter()-begin
    return row,time,state


def endpoint_difference(state,reference):
    return {'position_error':float(np.max(abs(state[:3]-reference[:3]))),
        'velocity_error':float(np.max(abs(state[3:]-reference[3:])))}


def outside_label(z):
    return 'transmitted' if z>DISTANCE else 'reflected' if z<-DISTANCE else 'not_outside_planes_at_T'


inputs={'mass':MASS,'charge':CHARGE,'field':FIELD,'strength':STRENGTH,'screening':SCREENING,
    'fixed_incoming_triple':[IMPACT,PARALLEL,PERPENDICULAR],'phase_definition':'free incoming helix at z=0',
    'phase_bracket':[LEFT,RIGHT],'prior_summary_sha256':PRIOR_SHA,
    'start_distance':DISTANCE,'flight_time_factor':FLIGHT_FACTOR,'max_step':MAX_STEP,'rtol':RTOL,
    'max_bisections':MAX_BISECTIONS,'interior_z_target':INTERIOR_TARGET,
    'search_rule':'Recheck prior opposite exits and full-IVP F=z(T) signs; bisect F signs without monotonicity assumption; stop at |F|<L/2, failed solve,40 attempts or machine bracket width.',
    'refined_step':REFINED_STEP,'larger_flight_factors':LARGER_FACTORS,
    'fallback_precision_controls':'If no interior candidate is found, check both final endpoints at the sameT and saved binary64 initial arrays.',
    'precision_control_rtol':PRECISION_TOLERANCES,'precision_control_max_step':REFINED_STEP,
    'precision_cartesian_rtol':PRECISION_TOLERANCES[-1],
    'position_absolute_target':POSITION_TARGET,'velocity_absolute_target':VELOCITY_TARGET,'energy_target':ENERGY_TARGET,
    'diagnostic_pass_rule':'Rechecked opposite exits; full-IVP interior candidate confirmed nonexit by public event helper at both steps; half-step and independent12D endpoint errors below unchanged targets; all recorded paths pass and dependencies unchanged.',
    'limits':'No censor-interval width, phase integral, asymptotic trapping, plasma rate or lifetime; larger-budget outcomes remain diagnostic. An outgoing-scattering integral needs almost-everywhere finite exits or an explicit extension convention.'}
if hashlib.sha256(PRIOR.read_bytes()).hexdigest()!=PRIOR_SHA:raise ValueError('Original censoring receipt changed')
prior=json.loads(PRIOR.read_text());prior_paths=PRIOR.with_name('paths.npz')
if hashlib.sha256(prior_paths.read_bytes()).hexdigest()!=prior['paths_sha256']:raise ValueError('Original paths changed')
if prior['results']['last_search_bracket']!=[LEFT,RIGHT] or prior['inputs']['fixed_incoming_triple']!=inputs['fixed_incoming_triple']:
    raise ValueError('Declared continuation inputs differ from original receipt')
prior_endpoints=[]
for phase,kind in [(LEFT,'transmitted'),(RIGHT,'reflected')]:
    matches=[r for r in prior['results']['attempts'] if r['phase']==phase]
    if len(matches)!=1 or matches[0]['status']!='outgoing_event' or matches[0]['exit']!=kind:
        raise ValueError('Original final bracket endpoints are inconsistent')
    prior_endpoints.append(matches[0])
OUTPUT.mkdir(parents=True,exist_ok=True)
metadata=run_metadata(inputs,model='full-IVP phase search for finite-budget plane censoring',
    boundary='unabsorbed IVP at fixed T; separate first-outgoing-plane event checks; extended finite budgets',units='normalized')
metadata['prior_evidence']={'summary_path':str(PRIOR.relative_to(ROOT)),'summary_sha256':PRIOR_SHA,
    'paths_path':str(prior_paths.relative_to(ROOT)),'paths_sha256':prior['paths_sha256'],
    'producer_commit':prior['commit'],'declared_endpoint_rows':prior_endpoints}
dependencies=[Path(__file__).resolve(),ROOT/'src/sato_morrison/controls.py',ROOT/'src/sato_morrison/reference.py']
metadata['experiment_dependency_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
(OUTPUT/'declared_inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
started=perf_counter();rows=[];paths={}


def evaluate(phase,step,factor,label,full=False):
    row,time,state=(full_budget_attempt if full else attempt)(phase,step,factor)
    row['label']=label;rows.append(row)
    if len(time):paths.update({f'{label}_time':time,f'{label}_state':state})
    (OUTPUT/'attempts_partial.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(OUTPUT/'paths_partial.npz',**paths)
    print(f'  {label}: phase={phase:.17g}, {row["status"]}, exit={row.get("exit")}, F={row.get("search_F")}, time={row.get("time")}',flush=True)
    return row


found=None;search_reason='initial_bracket_unresolved';left,right=LEFT,RIGHT
with progress('Recheck and bisect the continuous full-IVP value z(T)'):
    left_event=evaluate(left,MAX_STEP,FLIGHT_FACTOR,'left_event')
    right_event=evaluate(right,MAX_STEP,FLIGHT_FACTOR,'right_event')
    a=evaluate(left,MAX_STEP,FLIGHT_FACTOR,'left_endpoint',full=True)
    b=evaluate(right,MAX_STEP,FLIGHT_FACTOR,'right_endpoint',full=True)
    bracket_ok=(left_event.get('exit')=='transmitted' and right_event.get('exit')=='reflected'
        and a.get('recorded_path_check')=='passed' and b.get('recorded_path_check')=='passed'
        and a.get('search_F',0)>DISTANCE and b.get('search_F',0)<-DISTANCE)
    if bracket_ok:
        fa=a['search_F'];search_reason='attempt_cap'
        for index in range(MAX_BISECTIONS):
            phase=(left+right)/2
            if phase==left or phase==right:search_reason='machine_width';break
            row=evaluate(phase,MAX_STEP,FLIGHT_FACTOR,f'bisection_{index:02d}',full=True)
            if row.get('recorded_path_check')!='passed':search_reason='integration_unresolved';break
            f=row['search_F']
            if abs(f)<INTERIOR_TARGET:found=row;search_reason='interior_candidate_found';break
            if np.sign(f)==np.sign(fa):left=phase;fa=f
            else:right=phase

confirmation=None;fine=None;pair_check=None;extensions=[];half_check=None
if found is not None:
    with progress('Check the fixed candidate with public events, half timestep and individual Cartesian equations'):
        confirmation=evaluate(found['phase'],MAX_STEP,FLIGHT_FACTOR,'candidate_event_check')
        fine=evaluate(found['phase'],REFINED_STEP,FLIGHT_FACTOR,'nonexit_half_step')
        if fine['status']=='finite_budget_nonexit':
            half_check={'position_error':float(np.max(abs(np.asarray(found['final_position'])-fine['final_position']))),
                'velocity_error':float(np.max(abs(np.asarray(found['final_velocity'])-fine['final_velocity'])))}
            half_check['status']='passed' if half_check['position_error']<=POSITION_TARGET and half_check['velocity_error']<=VELOCITY_TARGET else 'unresolved'
        position,velocity=encounter_incoming(IMPACT,found['phase'],PARALLEL,PERPENDICULAR,DISTANCE,CHARGE*FIELD/MASS)
        try:
            pair=binary_encounter(position,velocity,field=FIELD,strength=STRENGTH,
                mass=MASS,charge=CHARGE,screening=SCREENING,duration=found['flight_budget'],max_step=REFINED_STEP,rtol=RTOL)
            r=pair['positions'][-1,0]-pair['positions'][-1,1]
            w=pair['velocities'][-1,0]-pair['velocities'][-1,1]
            reference=fine if fine['status']=='finite_budget_nonexit' else found
            pair_check={'position_error':float(np.max(abs(r-reference['final_position']))),
                'velocity_error':float(np.max(abs(w-reference['final_velocity']))),
                'energy_error':pair['energy_error'],'final_relative_position':r.tolist(),
                'final_relative_velocity':w.tolist(),'evaluations':pair['evaluations'],
                'comparison':'half-step public event helper' if reference is fine else 'primary full-IVP candidate'}
            pair_check['status']='passed' if pair_check['position_error']<=POSITION_TARGET and pair_check['velocity_error']<=VELOCITY_TARGET and pair_check['energy_error']<=ENERGY_TARGET and abs(r[2])<DISTANCE else 'unresolved'
            paths.update(pair_time=pair['time'],pair_positions=pair['positions'],pair_velocities=pair['velocities'])
        except (RuntimeError,ValueError,FloatingPointError) as error:
            pair_check={'status':'unresolved','reason':str(error)}
    with progress('Extend only the same candidate flight budget; preserve all classifications'):
        for factor in LARGER_FACTORS:
            extensions.append(evaluate(found['phase'],REFINED_STEP,factor,f'extended_{int(factor)}'))


precision_controls=[];input_preparation=[]
if found is None:
    with progress('Fixed-endpoint timestep, tolerance and Cartesian controls; no new phase search'):
        for index,phase in enumerate([left,right]):
            label=f'precision_endpoint_{index}'
            reference_rows=[r for r in rows if r['phase']==phase and 'search_F' in r]
            if not reference_rows:
                precision_controls.append({'endpoint_index':index,'phase':phase,'status':'unresolved','reason':'No baseline fullT path'})
                continue
            reference=reference_rows[0];baseline=paths[reference['label']+'_state'][-1]
            initial=paths[reference['label']+'_state'][0]
            omega=CHARGE*FIELD/MASS;shift=omega*DISTANCE/PARALLEL
            input_preparation.append({'endpoint_index':index,'phase':phase,'phase_hex':float(phase).hex(),
                'free_flight_angle':shift,'free_flight_angle_hex':float(shift).hex(),
                'incoming_angle':phase+shift,'incoming_angle_hex':float(phase+shift).hex(),
                'initial_double_state':initial.tolist(),'initial_double_state_hex':[float(x).hex() for x in initial]})
            previous=None
            for tolerance in PRECISION_TOLERANCES:
                begin=perf_counter();key=label+f'_relative_{tolerance:g}'
                check={'endpoint_index':index,'phase':phase,'method':'relative6D_DOP853',
                    'rtol':tolerance,'max_step':REFINED_STEP,'baseline_outside_label':outside_label(baseline[2])}
                try:
                    time,state,error=recorded_path(phase,FLIGHT_FACTOR*DISTANCE/PARALLEL,REFINED_STEP,rtol=tolerance)
                    if not np.array_equal(state[0],initial):raise ValueError('Precision control changed initial arrays')
                    delta=endpoint_difference(state[-1],baseline)
                    check.update(final_state=state[-1].tolist(),search_F=float(state[-1,2]),energy_error=error,
                        outside_plane_label=outside_label(state[-1,2]),difference_from_primary=delta,
                        same_initial_double_arrays=True)
                    if previous is not None:check['difference_from_half_step']=endpoint_difference(state[-1],previous)
                    agreements=[delta]+([check['difference_from_half_step']] if previous is not None else [])
                    check['status']='passed' if error<=ENERGY_TARGET and all(d['position_error']<=POSITION_TARGET and d['velocity_error']<=VELOCITY_TARGET for d in agreements) else 'unresolved'
                    previous=state[-1];paths.update({key+'_time':time,key+'_state':state})
                except (RuntimeError,ValueError,FloatingPointError) as error:
                    check.update(status='unresolved',reason=str(error))
                check['wall_s']=perf_counter()-begin;precision_controls.append(check)
                (OUTPUT/'precision_controls_partial.json').write_text(json.dumps(precision_controls,indent=2,allow_nan=False)+'\n')
                np.savez_compressed(OUTPUT/'paths_partial.npz',**paths)
                print(f'  precision endpoint{index}: relative rtol={tolerance}, F={check.get("search_F")}, {check["status"]}',flush=True)
            begin=perf_counter();key=label+'_individual12D'
            check={'endpoint_index':index,'phase':phase,'method':'individual12D_DOP853',
                'rtol':PRECISION_TOLERANCES[-1],'max_step':REFINED_STEP,'baseline_outside_label':outside_label(baseline[2])}
            try:
                pair=binary_encounter(initial[:3],initial[3:],field=FIELD,strength=STRENGTH,mass=MASS,
                    charge=CHARGE,screening=SCREENING,duration=FLIGHT_FACTOR*DISTANCE/PARALLEL,
                    max_step=REFINED_STEP,rtol=PRECISION_TOLERANCES[-1])
                pair_initial=np.concatenate((pair['positions'][0,0]-pair['positions'][0,1],pair['velocities'][0,0]-pair['velocities'][0,1]))
                if not np.array_equal(pair_initial,initial):raise ValueError('Cartesian control changed initial arrays')
                endpoint=np.concatenate((pair['positions'][-1,0]-pair['positions'][-1,1],pair['velocities'][-1,0]-pair['velocities'][-1,1]))
                delta=endpoint_difference(endpoint,previous if previous is not None else baseline)
                check.update(final_state=endpoint.tolist(),search_F=float(endpoint[2]),energy_error=pair['energy_error'],
                    outside_plane_label=outside_label(endpoint[2]),difference_from_primary=endpoint_difference(endpoint,baseline),
                    difference_from_tight_relative=delta,same_initial_double_arrays=True,evaluations=pair['evaluations'])
                check['status']='passed' if pair['energy_error']<=ENERGY_TARGET and all(d['position_error']<=POSITION_TARGET and d['velocity_error']<=VELOCITY_TARGET for d in [delta,check['difference_from_primary']]) else 'unresolved'
                paths.update({key+'_time':pair['time'],key+'_positions':pair['positions'],key+'_velocities':pair['velocities']})
            except (RuntimeError,ValueError,FloatingPointError) as error:
                check.update(status='unresolved',reason=str(error))
            check['wall_s']=perf_counter()-begin;precision_controls.append(check)
            (OUTPUT/'precision_controls_partial.json').write_text(json.dumps(precision_controls,indent=2,allow_nan=False)+'\n')
            np.savez_compressed(OUTPUT/'paths_partial.npz',**paths)
            print(f'  precision endpoint{index}: Cartesian F={check.get("search_F")}, {check["status"]}',flush=True)

metadata['experiment_dependency_sha256_end']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
unchanged=metadata['experiment_dependency_sha256']==metadata['experiment_dependency_sha256_end']
passed=(bracket_ok and found is not None and confirmation is not None and confirmation['status']=='finite_budget_nonexit'
    and fine is not None and fine['status']=='finite_budget_nonexit' and half_check is not None and half_check['status']=='passed'
    and pair_check is not None and pair_check['status']=='passed'
    and all(r.get('recorded_path_check')=='passed' for r in rows)
    and all(r['status']=='passed' for r in precision_controls) and unchanged)
results={'status':'passed_finite_budget_diagnostic' if passed else 'unresolved',
    'search_stop':search_reason,'last_search_bracket':[left,right],
    'last_search_width':right-left,'found_phase':None if found is None else found['phase'],
    'selected_F':None if found is None else found['search_F'],
    'half_step_check':half_check,'independent_12D_check':pair_check,'extended_budget_results':extensions,
    'attempts':rows,'precision_controls':precision_controls,'input_preparation':input_preparation,
    'precision_controls_scope':'Same physical inputs and exact initial double arrays; all paths use DOP853. Endpoint disagreement is not error relative to a known exact trajectory. Smaller energy error alone cannot certify trajectory accuracy.',
    'dependencies_unchanged':unchanged,'wall_s':perf_counter()-started,
    'wall_s_scope':'Numerical solves/checks and intermediate checkpoint writes; excludes imports, metadata initialization, plots and final output.',
    'scope':'Finite-budget nonexit is censoring, not trapping. Last search bracket is not a censor-interval width or coverage bound. Original diagnostics and phase integral remain unresolved. Classifications are floating-point ODE evidence, not interval certification. Eventual outgoing-moment bounds do not define a scattering integral on non-exiting phases.'}
metadata['results']=results
np.savez_compressed(OUTPUT/'paths.npz',**paths)
metadata['paths_sha256']=hashlib.sha256((OUTPUT/'paths.npz').read_bytes()).hexdigest()
(OUTPUT/'summary.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
styles=[('left_event','transmitted neighbor','-'),('right_event','reflected neighbor','-'),
    ('nonexit_half_step','same phase: budget T','-'),('extended_24','same phase: budget 2T','--'),('extended_48','same phase: budget 4T',':')]
for label,title,style in styles:
    if f'{label}_time' in paths:axes[0].plot(paths[f'{label}_time'],paths[f'{label}_state'][:,2],ls=style,label=title)
for plane in [-DISTANCE,DISTANCE]:axes[0].axhline(plane,color='0.5',ls=':',lw=1)
axes[0].set(xlabel='time',ylabel='relative z',title='Fixed planes; finite flight budgets');axes[0].legend(fontsize=8)
for row in rows:
    if 'search_F' in row:
        color='#873DA4' if abs(row['search_F'])<INTERIOR_TARGET else '#147D92' if row['search_F']>0 else '#C66B28'
        axes[1].scatter((row['phase']-LEFT)*1e6,row['search_F'],color=color)
axes[1].axhspan(-INTERIOR_TARGET,INTERIOR_TARGET,color='#873DA4',alpha=.12,label='declared interior target')
for plane in [-DISTANCE,DISTANCE]:axes[1].axhline(plane,color='0.5',ls=':',lw=1)
axes[1].set(xlabel='phase − left endpoint (μrad)',ylabel='unabsorbed full-IVP z(T)',title='Continuous observable; no monotonicity assumed');axes[1].legend(fontsize=8)

for check in precision_controls:
    if 'energy_error' in check:
        delta=check.get('difference_from_tight_relative',check['difference_from_primary'])
        axes[2].scatter(check['energy_error'],max(delta['position_error'],np.finfo(float).tiny),
            color=['#147D92','#C66B28'][check['endpoint_index']],
            marker='x' if check['method']=='individual12D_DOP853' else 'o')
axes[2].set_xscale('log');axes[2].set_yscale('log')
axes[2].axhline(POSITION_TARGET,color='0.5',ls=':',label='endpoint agreement target')
axes[2].set(xlabel='relative energy error',ylabel='endpoint position disagreement',title='Small energy error can hide trajectory failure')
axes[2].legend(handles=[Line2D([],[],color='0.5',marker='o',ls='',label='relative: primary comparison'),
    Line2D([],[],color='0.5',marker='x',ls='',label='Cartesian: tight relative comparison'),
    Line2D([],[],color='0.5',ls=':',label='agreement target 1e−8')],fontsize=8)
subtitle='finite-budget diagnostic passed' if passed else 'finite-budget diagnostic unresolved'
fig.suptitle('A phase grid can miss slow encounters\n'+subtitle+'; no trapping or phase-integral conclusion',fontsize=11)
fig.savefig(OUTPUT/'finite_budget_censoring.png',dpi=180);plt.close(fig)
print(f'Finished: {results["status"]}; {results["wall_s"]:.1f}s',flush=True)
if not passed:raise RuntimeError('Refined finite-budget diagnostic unresolved; all attempts retained')
