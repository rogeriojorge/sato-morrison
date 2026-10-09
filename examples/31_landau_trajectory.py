"""Bounded physical-3V Landau campaign; every accepted density is retained."""
from pathlib import Path
from hashlib import sha256
from time import perf_counter
import json
import os
import resource
import sys
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter, writers
from scipy.interpolate import BarycentricInterpolator
from sato_morrison.controls import (GaussianMixture,
    landau_gaussian_analytic, landau_hermite_grid, landau_entropy_compiler,
    landau_entropy_step, landau_gaussian_relative_rate)
from sato_morrison.reference import run_metadata, progress, gauss_interval

ROOT = Path(__file__).resolve().parents[1]
output_override=os.environ.get('SM_LANDAU_OUTPUT')
OUTPUT=Path(output_override) if output_override else ROOT/'results'/'landau_trajectory_dt_refined'
if not OUTPUT.is_absolute():OUTPUT=ROOT/OUTPUT
A = np.array([1.15, 1.15, .7])
GAMMA = 1.
FINAL_TIME = .2
# The campaign is fixed before execution. A failed baseline prevents downstream
# finite-time scans, whose initial collision terms are still checked independently.
CASES = [
    {'name':'baseline', 'order':16, 'variance':.6, 'dt':.025},
    {'name':'grid12', 'order':12, 'variance':.6, 'dt':.025},
    {'name':'dt_coarse', 'order':16, 'variance':.6, 'dt':.05},
    {'name':'dt_fine', 'order':16, 'variance':.6, 'dt':.0125},
    {'name':'tail_narrow', 'order':16, 'variance':.585, 'dt':.025},
    {'name':'tail_wide', 'order':16, 'variance':.625, 'dt':.025},
    {'name':'kernel02', 'order':16, 'variance':.6, 'dt':.025, 'softening':.2},
    {'name':'kernel01', 'order':16, 'variance':.6, 'dt':.025, 'softening':.1},
    {'name':'tolerance', 'order':16, 'variance':.6, 'dt':.025, 'rtol':1e-12, 'linear_rtol':1e-10},
    {'name':'nonlinear_tolerance', 'order':16, 'variance':.6, 'dt':.025, 'rtol':1e-12},
    {'name':'linear_tolerance', 'order':16, 'variance':.6, 'dt':.025, 'linear_rtol':1e-10},
    {'name':'grid20', 'order':20, 'variance':.6, 'dt':.025, 'case_budget_s':2400., 'linear_rtol':1e-12, 'max_log_step':None},
]
RTOL, LINEAR_RTOL = 1e-10, 1e-8
LINEAR_MAX_STEPS, NEWTON_MAX_STEPS, CHUNK = 200, 30, 128
MAX_LOG_STEP = 2.
PRECONDITIONER = 'tensor'
INITIAL_TARGET, INCREMENT_TARGET, INVARIANT_TARGET = .01, .01, 1e-9
MOMENT_INCREMENT_FLOOR, DENSITY_INCREMENT_FLOOR = 1e-14, 1e-14
SCALAR_ORDERS = [96, 192]
COMPARE_ORDER, COMPARE_EXTENT = 48, 4.
COMMON_QUADRATURE_ORDERS = [32, 48, 64]
EXPANDED_EXTENT, EXPANDED_ORDER = 5., 48
EXPANDED_QUADRATURE_ORDERS = [32, 48, 64]
QUADRATURE_INCREMENT_TARGET = 1e-4
CASE_BUDGET_S, CAMPAIGN_BUDGET_S = 900., 5400.
FPS = 3
OUTPUT.mkdir(parents=True, exist_ok=True)
if any(OUTPUT.iterdir()):
    raise RuntimeError('Output already exists: preserve it and record a new output destination before rerunning.')
plt.rcParams.update({'font.size':10, 'axes.spines.top':False,
    'axes.spines.right':False, 'svg.fonttype':'path', 'figure.facecolor':'white'})
started = perf_counter()
initial = GaussianMixture(np.ones(1), np.zeros((1,3)), np.diag(A)[None])
rate = landau_gaussian_analytic(np.diag(A), GAMMA)
alpha = float(-(rate[0,0]-rate[2,2])/(A[0]-A[2]))
metadata = run_metadata({'covariance':A.tolist(), 'density':1., 'mass':1.,
    'gamma':GAMMA, 'field_for_mu':1., 'final_time':FINAL_TIME, 'cases':CASES,
    'rtol':RTOL, 'linear_rtol':LINEAR_RTOL, 'linear_max_steps':LINEAR_MAX_STEPS,
    'newton_max_steps':NEWTON_MAX_STEPS, 'chunk':CHUNK, 'max_log_step':MAX_LOG_STEP,
    'preconditioner':PRECONDITIONER, 'reference_population':'fixed raw sampled initial physical population w*F0; no normalization', 'scalar_orders':SCALAR_ORDERS,
    'initial_relative_L1_target':INITIAL_TARGET, 'increment_relative_L1_target':INCREMENT_TARGET,
    'invariant_target':INVARIANT_TARGET, 'common_cube':[COMPARE_ORDER,COMPARE_EXTENT],
    'normalization_floors':{'moment_increment':MOMENT_INCREMENT_FLOOR,'density_increment':DENSITY_INCREMENT_FLOOR,'behavior':'Reject unresolved increments; no denominator clipping'},
    'common_quadrature_orders':COMMON_QUADRATURE_ORDERS, 'quadrature_increment_error_target':QUADRATURE_INCREMENT_TARGET,
    'expanded_common_cube':[EXPANDED_ORDER,EXPANDED_EXTENT], 'expanded_quadrature_orders':EXPANDED_QUADRATURE_ORDERS,
    'case_budget_s':CASE_BUDGET_S, 'campaign_budget_s':CAMPAIGN_BUDGET_S},
    model='Full physical 3V Coulomb Landau distribution; no Gaussian closure',
    boundary='Infinite-domain Hermite quadrature with natural finite weak adjoint; measured tail and scale scans')
metadata.update(status='running', rows=[], comparisons=[], matched_alpha=alpha,
    plan='Initial oracle checks for every declared row; baseline first, cheaper grid/time/scale/kernel/tolerance scans next, fine20 last. Scale span .585/.6/.625 is explicitly asymmetric and chosen from recorded initial-only preflights, including failed .575. Budgets checked at accepted stages and Newton callbacks. No post-outcome campaign extension.',
    scope='Nodal positivity and discrete invariants are distinct from physical trajectory convergence. Scalar quadrature, initial full strong collision term, full density increments on the common cube, tail populations and every finite step are reported separately.')
metadata['followup_plan']='Separate twelve-row campaign after ff536d6 time accuracy failure: baseline .025, coarse .05, fine .0125, common timestep for grid/order/scale/kernel scans, separate nonlinear and linear tolerance variations plus retained joint row. The predeclared sixteen-step .0125 preflight passed independent raw replay and all eight shared-time accuracy comparisons against .025. Physical inputs and 1% targets are unchanged; all former outcomes remain archived.'
metadata['moment_definitions']='Raw quadrature number, momentum and energy are conserved budgets. Covariance is centered and divided by actual quadrature number; anisotropy, mu_second, speed_fourth and fourth_cumulant are number-normalized expectations. No state or population is rescaled.'
source_paths = [Path(__file__), ROOT/'src/sato_morrison/controls.py', ROOT/'src/sato_morrison/solver.py']
metadata['frozen_sources'] = {str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in source_paths}

def save():
    metadata['wall_s'] = perf_counter()-started
    metadata['process_peak_rss_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)
    (OUTPUT/'summary.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')


def diagnostics(v, w, f):
    p = w*f
    number = float(p.sum())
    mean = p@v/number
    centered = v-mean
    covariance = p@(centered*centered)/number
    mu = (v[:,0]**2+v[:,1]**2)/2
    speed2 = np.sum(v*v,axis=1)
    return {'number':number, 'momentum':(p@v).tolist(), 'mean_velocity':mean.tolist(),
        'energy':float(p@speed2/2), 'covariance':covariance.tolist(),
        'anisotropy':float((covariance[0]+covariance[1])/2-covariance[2]),
        'mu_second':float(p@(mu*mu)/number), 'speed_fourth':float(p@(speed2*speed2)/number),
        'fourth_cumulant':(p@(centered**4)/number-3*covariance**2).tolist(),
        'entropy':float(-p@np.log(f)), 'min_f':float(f.min()),
        'outside_common_cube_mass':float(p[np.max(abs(v),axis=1)>COMPARE_EXTENT].sum()),
        'outside_expanded_cube_mass':float(p[np.max(abs(v),axis=1)>EXPANDED_EXTENT].sum())}


def checkpoint(row, v, w, derivative, states, times):
    np.savez_compressed(OUTPUT/(row['name']+'.npz'), velocity=v, weights=w,
        derivative=derivative, density=np.asarray(states), time=np.asarray(times))
    row['saved_states'] = len(states)
    row['last_accepted_time'] = float(times[-1])
    save()


print('Physical 3V Landau: bounded full-distribution trajectory campaign; all stages are saved.', flush=True)
print(f'Common Gaussian A={A}; Gamma={GAMMA}; final t={FINAL_TIME}; matched alpha={alpha:.12g}.', flush=True)
save()
completed = {}
baseline_ok = False
for specification in CASES:
    row = dict(specification)
    row.update(softening=specification.get('softening',0.),
        rtol=specification.get('rtol',RTOL), linear_rtol=specification.get('linear_rtol',LINEAR_RTOL),
        max_log_step=specification.get('max_log_step',MAX_LOG_STEP),
        case_budget_s=specification.get('case_budget_s',CASE_BUDGET_S),
        status='running', steps=[])
    metadata['rows'].append(row)
    case_started = perf_counter()
    grid = landau_hermite_grid(row['order'], thermal_variance=row['variance'])
    v,w,D = map(np.asarray,(grid.velocity,grid.weights,grid.derivative))
    f = initial.evaluate(v)
    states,times = [f.copy()],[0.]
    row['initial'] = diagnostics(v,w,f)
    row['axis_extent'] = float(np.max(abs(v)))
    compiler = landau_entropy_compiler(grid,gamma=GAMMA,softening=row['softening'],
        chunk_size=CHUNK,linear_max_steps=LINEAR_MAX_STEPS,
        preconditioner=PRECONDITIONER,reference_population=w*f)
    checkpoint(row,v,w,D,states,times)
    print(f'Initial strong-oracle gate: {row["name"]}, n={row["order"]}, variance={row["variance"]}, epsilon={row["softening"]}.',flush=True)
    try:
        with progress('Compile physical pair action and independently refine continuum scalar integral'):
            logf=jnp.log(jnp.asarray(f))
            computed=-np.asarray(compiler.apply(logf,logf))/w
            references=[landau_gaussian_relative_rate(v,A,scalar_order=q,gamma=GAMMA) for q in SCALAR_ORDERS]
        exact=f*references[-1]
        scalar_defect=float(np.max(abs(references[0]-references[1])))
        row['initial_gate']={'relative_L1_error':float(w@abs(computed-exact)/(w@abs(exact))),
            'scalar_refinement_max_abs':scalar_defect,
            'population_weighted_relative_rate_RMS_error':float(np.sqrt((w*f)@((computed/f-references[-1])**2))),
            'core_L1_over_full_reference':float(w[np.linalg.norm(v,axis=1)<=3]@abs(computed-exact)[np.linalg.norm(v,axis=1)<=3]/(w@abs(exact))),
            'pressure_rate':((w*computed)@(v*v)).tolist(),
            'status':'passed' if scalar_defect<1e-10 and float(w@abs(computed-exact)/(w@abs(exact)))<INITIAL_TARGET else 'unresolved'}
        np.savez_compressed(OUTPUT/(row['name']+'_initial_rhs.npz'), computed=computed, exact=exact)
        print(f'  initial L1={row["initial_gate"]["relative_L1_error"]:.3%}; gate={row["initial_gate"]["status"]}.',flush=True)
        save()
        if row['name']!='baseline' and not baseline_ok:
            row.update(status='not_run', reason='Baseline did not reach final time; downstream trajectory scans remain unexecuted.')
            continue
        if perf_counter()-started>CAMPAIGN_BUDGET_S:
            row.update(status='not_run',reason='Declared total campaign budget reached before finite-time scan.')
            continue
        count=int(round(FINAL_TIME/row['dt']))
        if abs(count*row['dt']-FINAL_TIME)>1e-14:
            raise RuntimeError('Declared timestep does not divide the common final time.')
        for step in range(1,count+1):
            def iteration(record):
                record.update(case=row['name'], step=step)
                with (OUTPUT/'iterations.jsonl').open('a') as handle:
                    clean={key:(value.item() if isinstance(value,np.generic) else value) for key,value in record.items()}
                    nonfinite=[key for key,value in clean.items() if isinstance(value,float) and not np.isfinite(value)]
                    for key in nonfinite:clean[key]=None
                    if nonfinite:clean['nonfinite_fields']=nonfinite
                    handle.write(json.dumps(clean,allow_nan=False)+'\n')
                print(f'  {row["name"]} step {step}/{count} Newton {record["iteration"]}: residual={record["relative_residual"]:.3e}, PCG={record["linear_iterations"]}, true={record["true_linear_relative_residual"]:.3e}',flush=True)
                if perf_counter()-case_started>row['case_budget_s'] or perf_counter()-started>CAMPAIGN_BUDGET_S:
                    raise RuntimeError('Declared campaign or case budget reached; latest accepted stages retained.')
            tick=perf_counter()
            with progress(f'{row["name"]}: full-density step {step}/{count}'):
                answer=landau_entropy_step(grid,f,row['dt'],gamma=GAMMA,softening=row['softening'],
                    compiled=compiler,chunk_size=CHUNK,linear_max_steps=LINEAR_MAX_STEPS,
                    rtol=row['rtol'],linear_rtol=row['linear_rtol'],max_steps=NEWTON_MAX_STEPS,
                    iteration_callback=iteration,max_log_step=row['max_log_step'])
            f=np.asarray(answer.f)
            states.append(f.copy());times.append(step*row['dt'])
            diagnostic=diagnostics(v,w,f)
            step_record={'step':step,'time':times[-1],'wall_s':perf_counter()-tick,
                'newton_iterations':answer.iterations,'linear_iterations':answer.linear_iterations,
                'linear_relative_residual':answer.linear_relative_residual, 'residual_metric':answer.residual_metric,
                'relative_residual':answer.relative_residual,'entropy_change':answer.entropy_change,
                'entropy_production':answer.entropy_production,'generalized_KL':answer.generalized_KL,
                'residual_entropy_defect':answer.residual_entropy_defect,
                'entropy_defect_bound':answer.entropy_defect_bound,'diagnostics':diagnostic}
            initial_budget=np.array([row['initial']['number'],*row['initial']['momentum'],row['initial']['energy']])
            current_budget=np.array([diagnostic['number'],*diagnostic['momentum'],diagnostic['energy']])
            invariant_drift=float(np.max(abs(current_budget-initial_budget)))
            previous_entropy=row['steps'][-1]['diagnostics']['entropy'] if row['steps'] else row['initial']['entropy']
            direct_entropy_gain=diagnostic['entropy']-previous_entropy
            entropy_budget_error=direct_entropy_gain-answer.entropy_production-answer.generalized_KL+answer.residual_entropy_defect
            structural=(invariant_drift<INVARIANT_TARGET and diagnostic['min_f']>0 and direct_entropy_gain>=-answer.entropy_defect_bound
                and answer.entropy_production>=-answer.entropy_defect_bound and answer.generalized_KL>=-answer.entropy_defect_bound
                and abs(entropy_budget_error)<=2*answer.entropy_defect_bound)
            step_record.update(maximum_absolute_invariant_drift=invariant_drift, direct_entropy_gain=direct_entropy_gain,
                direct_entropy_budget_error=entropy_budget_error, structural_status='passed' if structural else 'failed')
            row['steps'].append(step_record)
            checkpoint(row,v,w,D,states,times)
            if not structural:raise RuntimeError('Host-accepted state failed the producer structural budget; checkpoint retained and trajectory stopped.')
            print(f'  Accepted t={times[-1]:.4g}; minF={f.min():.3e}; entropy gain={answer.entropy_change:.3e}.',flush=True)
        row['status']='passed_discrete_trajectory'
        row['final']=diagnostics(v,w,f)
        completed[row['name']]=(row,v,w,D,np.asarray(states),np.asarray(times))
        if row['name']=='baseline':baseline_ok=True
    except Exception as error:
        row.update(status='failed',reason=f'{type(error).__name__}: {error}')
        if hasattr(error,'line_search_diagnostics'):row['line_search_diagnostics']=error.line_search_diagnostics
        print(f'FAILED {row["name"]}: {error}; partial states retained.',flush=True)
    finally:
        row['wall_s']=perf_counter()-case_started
        checkpoint(row,v,w,D,states,times)
        del compiler,grid
        jax.clear_caches()

# Compare actual density increments rather than divide by the much larger F0.
# Polynomial interpolation of log F is a diagnostic reconstruction only; no
# positivity claim is made for interpolation of F between solver nodes.
xq,wq=gauss_interval(COMPARE_ORDER,-COMPARE_EXTENT,COMPARE_EXTENT)
vq=np.stack(np.meshgrid(xq,xq,xq,indexing='ij'),axis=-1).reshape(-1,3)
wcommon=np.prod(np.meshgrid(wq,wq,wq,indexing='ij'),axis=0).ravel()


def common_density(v, f, axis=xq):
    n=round(len(v)**(1/3));x=v.reshape(n,n,n,3)[:,0,0,0]
    if axis.min()<x.min() or axis.max()>x.max():
        raise RuntimeError('Common density comparison would extrapolate beyond the saved velocity nodes.')
    if not np.all(np.isfinite(f)) or np.any(f<=0):
        raise RuntimeError('Saved nodal density is nonfinite or nonpositive; reconstruction is unresolved.')
    value=np.log(f).reshape(n,n,n)
    # Interpolate the non-Gaussian correction; the initial quadratic log F0 is
    # reproduced exactly and no fitted evolving covariance enters the state.
    value=value-np.log(initial.evaluate(v)).reshape(n,n,n)
    for d in range(3):
        value=BarycentricInterpolator(x,value,axis=d)(axis)
    vertices=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        reconstructed=initial.evaluate(vertices)*np.exp(value.ravel())
    if not np.all(np.isfinite(reconstructed)) or np.any(reconstructed<=0):
        raise RuntimeError('Diagnostic density reconstruction is nonfinite or nonpositive; raw accepted states remain retained.')
    return reconstructed


def density_comparison(v,states,times,bv,bs,bt,extent,order,quadrature_orders):
    matched=[]
    for k,t in enumerate(times):
        match=np.flatnonzero(np.isclose(bt,t,rtol=0,atol=1e-13))
        if t>0 and len(match)==1:matched.append((k,int(match[0]),float(t)))
    if not matched:raise RuntimeError('No shared accepted times for density comparison.')
    records={}
    for q in sorted(set([order,*quadrature_orders])):
        axis,weights=gauss_interval(q,-extent,extent)
        weight=np.prod(np.meshgrid(weights,weights,weights,indexing='ij'),axis=0).ravel()
        f0=common_density(bv,bs[0],axis=axis)
        values=[]
        for k,j,t in matched:
            base=common_density(bv,bs[j],axis=axis)
            own=common_density(v,states[k],axis=axis)
            increment=float(weight@abs(base-f0))
            error=float(weight@abs(own-base)/increment) if increment>DENSITY_INCREMENT_FLOOR else np.inf
            if not np.isfinite(error):raise RuntimeError('Density increment comparison has no finite nonzero normalization.')
            values.append({'order':q,'time':t,'baseline_increment_L1':increment,'relative_increment_L1_error':error})
        records[q]=values
    refinements=[]
    low,high=quadrature_orders[-2:]
    for a,b in zip(records[low],records[high]):
        refinements.append({'time':a['time'],
            'absolute_change_in_increment_error':abs(b['relative_increment_L1_error']-a['relative_increment_L1_error']),
            'baseline_increment_relative_change':abs(b['baseline_increment_L1']/a['baseline_increment_L1']-1)})
    error=max(r['relative_increment_L1_error'] for r in records[order])
    highest_error=max(r['relative_increment_L1_error'] for r in records[quadrature_orders[-1]])
    qe=max(r['absolute_change_in_increment_error'] for r in refinements)
    qi=max(r['baseline_increment_relative_change'] for r in refinements)
    return {'status':'passed' if max(error,highest_error)<INCREMENT_TARGET and qe<QUADRATURE_INCREMENT_TARGET and qi<INCREMENT_TARGET else 'unresolved',
        'extent':extent,'order':order,'maximum_relative_increment_L1':error,
        'highest_order':quadrature_orders[-1],'highest_order_maximum_relative_increment_L1':highest_error,
        'shared_times':[r['time'] for r in records[order]],
        'shared_time_common_quadrature':[{ 'time':t, 'values':[records[q][i] for q in quadrature_orders]} for i,(_,_,t) in enumerate(matched)],
        'final_common_quadrature':[records[q][-1] for q in quadrature_orders],
        'quadrature_refinements':refinements,'quadrature_absolute_change_in_increment_error':qe,
        'quadrature_baseline_increment_relative_change':qi,
        'common_quadrature_status':'passed' if qe<QUADRATURE_INCREMENT_TARGET and qi<INCREMENT_TARGET else 'unresolved'}


if baseline_ok:
    base=completed['baseline'];br,bv,bw,bD,bs,bt=base
    reconstruction_ok=True
    try:
        baseline_increment=float(wcommon@abs(common_density(bv,bs[-1])-common_density(bv,bs[0])))
        metadata['baseline_distribution_increment_L1_common_cube']=baseline_increment
    except (RuntimeError,ValueError,FloatingPointError) as error:
        reconstruction_ok=False
        metadata['baseline_reconstruction_failure']=str(error)
    metadata['baseline_distribution_increment_L1_native_nodes']=float(bw@abs(bs[-1]-bs[0]))
    metadata['moment_refinement_norm']='Each whole-quadrature observable compares changes from its own initial sampled value; error is divided by the nonzero baseline relaxation increment at each shared saved time. The fourth-cumulant vector uses its maximum absolute component increment as scale.'
    metadata['tail_diagnostic_scope']='Outside4/5 values sum the actual nodal quadrature populations whose sampled coordinates lie outside the boxes. These are sampled tail diagnostics, not rigorous continuum tail bounds.'
    metadata['comparison_domain_scope']='Primary [-4,4]^3 and expanded[-5,5]^3 norms are restricted to node overlap. Neither is a full-domain density norm. Raw full quadrature moments and outside-cube populations are reported independently. Hermite order and thermal variance both change outer-node extent and core resolution.'
    metadata['comparison_norm']='Maximum over shared saved times >0 of integral_common_cube(abs(Fcase-Fbaseline))/integral_common_cube(abs(Fbaseline-F0)); same-node scans also report native-node norm. No temporal interpolation.'
    for name,entry in completed.items():
        row,v,w,D,states,times=entry
        if name=='baseline':continue
        try:
            primary=density_comparison(v,states,times,bv,bs,bt,COMPARE_EXTENT,COMPARE_ORDER,COMMON_QUADRATURE_ORDERS)
            expanded={'status':'not_applicable','reason':'Coarse node axes do not contain the expanded box.'}
            if np.max(abs(v[:,0]))>=EXPANDED_EXTENT and np.max(abs(bv[:,0]))>=EXPANDED_EXTENT:
                expanded=density_comparison(v,states,times,bv,bs,bt,EXPANDED_EXTENT,EXPANDED_ORDER,EXPANDED_QUADRATURE_ORDERS)
            native=[];moment_errors={key:[] for key in ['anisotropy','mu_second','speed_fourth','fourth_cumulant']}
            moment_scales={key:[] for key in moment_errors}
            base_diagnostics=[diagnostics(bv,bw,f) for f in bs]
            own_initial=diagnostics(v,w,states[0])
            for k,t in enumerate(times):
                match=np.flatnonzero(np.isclose(bt,t,rtol=0,atol=1e-13))
                if t==0 or len(match)!=1:continue
                j=int(match[0]);own=diagnostics(v,w,states[k])
                for key in moment_errors:
                    reference_increment=np.asarray(base_diagnostics[j][key])-np.asarray(base_diagnostics[0][key])
                    own_increment=np.asarray(own[key])-np.asarray(own_initial[key])
                    scale=np.max(abs(reference_increment))
                    moment_scales[key].append({'time':float(t),'absolute_baseline_increment_scale':float(scale)})
                    if scale<=MOMENT_INCREMENT_FLOOR:raise RuntimeError(f'{key} relaxation increment has no nonzero normalization scale')
                    value=float(np.max(abs(own_increment-reference_increment))/scale)
                    if not np.isfinite(value):raise RuntimeError(f'{key} increment comparison is nonfinite')
                    moment_errors[key].append(value)
                if np.array_equal(v,bv) and np.array_equal(w,bw):
                    native.append(float(bw@abs(states[k]-bs[j])/(bw@abs(bs[j]-bs[0]))))
            comparison=dict(primary,case=name,common_saved_times=len(primary['shared_times']),
                native_maximum_relative_increment_L1=max(native) if native else None,
                whole_quadrature_moment_increment_errors={key:max(value) for key,value in moment_errors.items()},
                whole_quadrature_moment_increment_normalization=moment_scales,
                expanded_cube=expanded,reconstruction_status='passed')
            comparison['status']='passed' if primary['status']=='passed' and expanded['status'] in ['passed','not_applicable'] and max(max(value) for value in moment_errors.values())<INCREMENT_TARGET else 'unresolved'
            print(f'Full-density increment comparison {name}: {primary["maximum_relative_increment_L1"]:.3%}, {comparison["status"]}.',flush=True)
        except (RuntimeError,ValueError,FloatingPointError) as error:
            reconstruction_ok=False
            comparison={'case':name,'status':'unresolved','reconstruction_status':'failed','reason':str(error),
                'expanded_cube':{'status':'unresolved','reason':'Invalid diagnostic reconstruction; raw stages remain retained.'}}
            print(f'Full-density increment comparison {name}: UNRESOLVED: {error}',flush=True)
        metadata['comparisons'].append(comparison)
        save()
    # Three resolution levels and two independent parameter directions are kept
    # separately; no averaging conceals a failed refinement.
    required=['grid20','dt_fine','tail_narrow','tail_wide','kernel01','tolerance','nonlinear_tolerance','linear_tolerance']
    byname={r['case']:r for r in metadata['comparisons']}
    continuum_rows=[r for r in metadata['rows'] if r['name'] in ['baseline','grid20','tail_narrow','tail_wide']]
    initial_ok=all(r.get('initial_gate',{}).get('status')=='passed' for r in continuum_rows)
    checks=[]
    for name,entry in completed.items():
        row,v,w,D,states,times=entry
        invariant=np.column_stack([np.ones(len(v)),v,np.sum(v*v,axis=1)/2])
        moments=(states*w)@invariant
        defect=float(np.max(abs(moments-moments[0])))
        row['maximum_absolute_invariant_drift']=defect
        checks.append(defect<INVARIANT_TARGET)
    coarse_complete=all(name in completed for name in ['grid12','dt_coarse','kernel02'])
    metadata['three_level_trajectory_completion']='passed' if coarse_complete else 'unresolved'
    expanded_ok=all(name in byname and byname[name]['expanded_cube']['status']=='passed' for name in required)
    converged=reconstruction_ok and coarse_complete and expanded_ok and all(name in byname and byname[name]['status']=='passed' for name in required)
    metadata['status']='passed' if initial_ok and converged and all(checks) else 'unresolved'
else:
    metadata['status']='unresolved'
    metadata['baseline_failure']='No completed baseline density trajectory; four-operator comparison is withheld.'

metadata['accepted_stage_structural_status']=('passed' if all(step['structural_status']=='passed' for row in metadata['rows'] for step in row['steps']) else 'failed') if any(row['steps'] for row in metadata['rows']) else 'not_run'
if metadata['accepted_stage_structural_status']!='passed':metadata['status']='unresolved'

# The state movie always uses saved accepted states, including a failed partial
# baseline. Repeated playback frames do not create new physical timestamps.
baseline_row=metadata['rows'][0]
with np.load(OUTPUT/'baseline.npz') as data:
    v,w,states,times=[data[key].copy() for key in ['velocity','weights','density','time']]
n=baseline_row['order'];x=v.reshape(n,n,n,3)[:,0,0,0]
# Tensor quadrature weights reconstructed directly from stored volume weights.
# The ij tensor cube has w[i,j,k]=wx[i]wx[j]wx[k].
axis_w=np.cbrt(np.array([w.reshape(n,n,n)[i,i,i] for i in range(n)]))
marginal=np.einsum('tijk,j->tik',states.reshape(-1,n,n,n),axis_w)
initial_marginal=marginal[0]
change=marginal-initial_marginal
limit=max(float(np.max(abs(change))),1e-15)
fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
mesh=axes[0].pcolormesh(x,x,change[-1].T,cmap='RdBu_r',vmin=-limit,vmax=limit,shading='nearest')
axes[0].contour(x,x,initial_marginal.T,levels=[.005,.02,.05,.1],colors='#34404a',linewidths=.6)
axes[0].set(xlabel=r'$v_x$',ylabel=r'$v_z$',aspect='equal',xlim=(-4,4),ylim=(-4,4),title='Distribution change, integrated over vᵧ')
fig.colorbar(mesh,ax=axes[0],label='∫ (F(t)−F(0)) dvᵧ')
curve=[]
for f in states:
    curve.append(diagnostics(v,w,f))
anisotropy_line,=axes[1].plot(times,[d['anisotropy']/.45 for d in curve],'o-',color='#7256a8',label='Saved Landau states')
axes[1].set(xlabel='Time t',ylabel='Normalized anisotropy',title='Pressure anisotropy')
cumulant_line,=axes[2].plot(times,[d['fourth_cumulant'][2] for d in curve],'s-',color='#197f8e')
axes[2].set(xlabel='Time t',ylabel=r'$\langle v_z^4\rangle-3\langle v_z^2\rangle^2$',title='Departure from a Gaussian')
for ax in axes[1:]:ax.grid(alpha=.15)
clock=fig.suptitle(f'Baseline {n}³ Landau • {metadata["status"]} • saved t={times[-1]:.3f}',fontweight='bold')
fig.savefig(OUTPUT/'trajectory.png',dpi=180)
fig.savefig(OUTPUT/'trajectory.svg',metadata={'Creator':None,'Date':None})


def frame(k):
    mesh.set_array(change[k].T)
    anisotropy_line.set_data(times[:k+1],[d['anisotropy']/.45 for d in curve[:k+1]])
    cumulant_line.set_data(times[:k+1],[d['fourth_cumulant'][2] for d in curve[:k+1]])
    clock.set_text(f'Baseline {n}³ Landau • {metadata["status"]} • saved t={times[k]:.3f}')
    return [mesh,anisotropy_line,cumulant_line,clock]


frame_indices=[0]*3+list(range(1,len(times)))+[len(times)-1]*4
if len(times)>1:
    with progress('Render accepted states; no interpolation in time'):
        animation=FuncAnimation(fig,frame,frames=frame_indices,interval=1000/FPS,blit=False)
        animation.save(OUTPUT/'trajectory.gif',writer=PillowWriter(fps=FPS),dpi=100)
        if writers.is_available('ffmpeg'):
            animation.save(OUTPUT/'trajectory.mp4',writer=FFMpegWriter(fps=FPS,codec='libx264',extra_args=['-pix_fmt','yuv420p']),dpi=100)
plt.close(fig)
metadata['movie']={'status':'passed' if len(times)>1 else 'not_run',
    'saved_physical_times':times.tolist(),'frame_physical_times':[float(times[k]) for k in frame_indices] if len(times)>1 else [],
    'frames':len(frame_indices) if len(times)>1 else 0,'fps':FPS,'duration_s':len(frame_indices)/FPS if len(times)>1 else 0.,
    'scope':'Each frame is a quadrature marginal of one actual saved accepted physical 3V state; first and final states are held for readability. No Gaussian closure and no temporal interpolation.'}

# Keep the exact values behind every figure, including explicit missing scans.
row_names=np.array([row['name'] for row in metadata['rows']])
row_labels=np.array([{'baseline':'baseline 16³','grid12':'grid 12³','grid20':'grid 20³','dt_coarse':f'Δt={row["dt"]:g}','dt_fine':f'Δt={row["dt"]:g}','tail_narrow':f'θ={row["variance"]:g}','tail_wide':f'θ={row["variance"]:g}','kernel02':f'ε={row["softening"]:g}','kernel01':f'ε={row["softening"]:g}','tolerance':'both tolerances','nonlinear_tolerance':'nonlinear tolerance','linear_tolerance':'linear tolerance'}[row['name']] for row in metadata['rows']])
initial_errors=np.array([row.get('initial_gate',{}).get('relative_L1_error',np.nan) for row in metadata['rows']])
comparison_by_name={row['case']:row for row in metadata['comparisons']}
density_errors=np.array([comparison_by_name.get(name,{}).get('maximum_relative_increment_L1',np.nan) for name in row_names])
expanded_density_errors=np.array([comparison_by_name.get(name,{}).get('expanded_cube',{}).get('maximum_relative_increment_L1',np.nan) for name in row_names])
moment_names=['anisotropy','mu_second','speed_fourth','fourth_cumulant']
moment_errors=np.array([[comparison_by_name.get(name,{}).get('whole_quadrature_moment_increment_errors',{}).get(key,np.nan) for key in moment_names] for name in row_names])
np.savez_compressed(OUTPUT/'plotted_arrays.npz', velocity_axis=x, time=times,
    integrated_density=marginal, integrated_density_change=change,
    anisotropy=np.array([d['anisotropy'] for d in curve]),
    fourth_cumulant=np.array([d['fourth_cumulant'] for d in curve]),
    row_names=row_names, row_labels=row_labels, row_status=np.array([row['status'] for row in metadata['rows']]),
    initial_RHS_L1_error=initial_errors, density_increment_L1_error=density_errors,
    expanded_density_increment_L1_error=expanded_density_errors,
    moment_names=np.array(moment_names), moment_increment_errors=moment_errors,
    common_quadrature_orders=np.array(COMMON_QUADRATURE_ORDERS),
    final_common_quadrature_errors=np.array([[r['relative_increment_L1_error'] for r in comparison_by_name[name]['final_common_quadrature']] if name in comparison_by_name and 'final_common_quadrature' in comparison_by_name[name] else [np.nan]*len(COMMON_QUADRATURE_ORDERS) for name in row_names]),
    expanded_quadrature_orders=np.array(EXPANDED_QUADRATURE_ORDERS),
    final_expanded_quadrature_errors=np.array([[r['relative_increment_L1_error'] for r in comparison_by_name[name]['expanded_cube']['final_common_quadrature']] if name in comparison_by_name and 'final_common_quadrature' in comparison_by_name[name]['expanded_cube'] else [np.nan]*len(EXPANDED_QUADRATURE_ORDERS) for name in row_names]))
fig,axes=plt.subplots(1,3,figsize=(12,6.7),sharey=True,layout='constrained')
DISPLAY_LOWER=1e-5
moment_labels=['anisotropy',r'$\langle\mu^2\rangle$',r'$\langle|v|^4\rangle$','fourth cumulant']
y=np.arange(len(row_names))
colors=['#147d92','#c66b28','#7256a8','#5f6670']
for j,row in enumerate(metadata['rows']):
    if np.isfinite(initial_errors[j]):
        axes[0].plot(initial_errors[j],j,'o',color='#147d92' if row['initial_gate']['status']=='passed' else '#c66b28')
    else:axes[0].text(.03,j,'not measured',transform=axes[0].get_yaxis_transform(),fontsize=8)
    if np.isfinite(density_errors[j]):
        axes[1].plot(max(density_errors[j],DISPLAY_LOWER),j-.07,marker='<' if density_errors[j]<DISPLAY_LOWER else 'o',linestyle='none',color='#7256a8',clip_on=False)
        if np.isfinite(expanded_density_errors[j]):
            axes[1].plot(max(expanded_density_errors[j],DISPLAY_LOWER),j+.07,marker='<' if expanded_density_errors[j]<DISPLAY_LOWER else 's',linestyle='none',color='#147d92',clip_on=False)
        for k,key in enumerate(moment_names):
            axes[2].plot(max(moment_errors[j,k],DISPLAY_LOWER),j+.07*(k-1.5),marker='<' if moment_errors[j,k]<DISPLAY_LOWER else 'o',linestyle='none',clip_on=False,color=colors[k],label=moment_labels[k] if j==next(i for i in range(len(row_names)) if np.isfinite(density_errors[i])) else None)
    else:
        label='reference' if row['name']=='baseline' and baseline_ok else f'{row["status"]}, t={row["last_accepted_time"]:g}'
        for ax in axes[1:]:ax.text(.03,j,label,transform=ax.get_yaxis_transform(),fontsize=8,color='#8e3c39' if row['status']=='failed' else '.4')
for ax,title in zip(axes,['Initial continuum collision term','Density increment (GL48)\n[−4,4]³ circles / [−5,5]³ squares','Whole-quadrature moment increments']):
    ax.set_xscale('log');ax.set(xlabel='Relative error',title=title)
    ax.axvline(INCREMENT_TARGET,color='.45',ls=':',lw=1)
    ax.grid(axis='x',alpha=.18)
comparison_values=np.concatenate([density_errors,expanded_density_errors,moment_errors.ravel()])
display_upper=max(.1,1.25*float(np.max(comparison_values[np.isfinite(comparison_values)]))) if np.any(np.isfinite(comparison_values)) else .1
for ax in axes[1:]:ax.set_xlim(DISPLAY_LOWER,display_upper)
axes[0].set_yticks(y,row_labels,fontsize=9);axes[0].set_ylim(len(row_names)-.5,-.5)
axes[2].legend(fontsize=8,loc='upper left',bbox_to_anchor=(1.02,1.)) if np.any(np.isfinite(density_errors)) else None
fig.suptitle(f'Landau trajectory validation: {metadata["status"]}',fontweight='bold')
fig.supxlabel('Initial RHS / continuum RHS; density and moments / actual baseline relaxation increment. Circles: cube 4; squares: cube 5. Dotted line: 1% target.\nLeft triangles: below 1e−5; exact arrays and acceptance gates retain their true values.',fontsize=8)
fig.savefig(OUTPUT/'convergence.png',dpi=180)
fig.savefig(OUTPUT/'convergence.svg',metadata={'Creator':None,'Date':None})
plt.close(fig)
metadata['convergence_figure']='Separate initial strong RHS, common-cube full-density increments and raw whole-quadrature moment increment refinements. Missing/failed scans have text only; no invented values. Exact error arrays retained; errors below1e-5 are explicitly shown by left triangles at the lower display bound. Acceptance gates use exact errors.'

# Explain the measure without identifying the quadrature reference with F(t).
z=np.linspace(-4,4,241)
reference=np.exp(-z*z/2)
variance=baseline_row['variance']
fig=plt.figure(figsize=(12,4.6),layout='constrained')
axes=[fig.add_subplot(1,3,k+1) for k in range(3)]
axes[0].plot(z,reference,color='#5f6670')
axes[0].plot(x/np.sqrt(variance),np.exp(-x*x/(2*variance)),'o',color='#147d92',markersize=3)
axes[0].set(xlabel='Hermite coordinate z',ylabel='Fixed reference weight',title='Choose quadrature nodes and weights')
axes[1].scatter(*np.meshgrid(x,x,indexing='ij'),color='#147d92',s=7)
axes[1].set(xlabel='Physical vₓ',ylabel=r'Physical $v_z$',title=f'One slice of a {n}³ tensor grid',aspect='equal')
axes[2].axis('off')
axes[2].text(0,.79,r'$q_a$: Gauss weights for $e^{-z^2/2}\,dz$',fontsize=9,transform=axes[2].transAxes)
for vertical,text in [( .92,r'$v_a=\sqrt{\theta}\,z_a$'),
    (.70,r'$w_{abc}=\theta^{3/2}q_aq_bq_c\,e^{(z_a^2+z_b^2+z_c^2)/2}$'),
    (.50,r'$n_i(t)=w_iF_i(t),\quad F_i(t)>0$'),
    (.30,r'$w_i\,\dot F_i=-[K(F)\log F]_i$')]:
    axes[2].text(0,vertical,text,fontsize=12,transform=axes[2].transAxes)
axes[2].text(0,.05,'Physical velocity-volume weights stay fixed.\nEvery density value evolves through all 3V pairs.',fontsize=10,transform=axes[2].transAxes)
fig.suptitle('Hermite quadrature, full three-velocity Landau dynamics',fontweight='bold')
fig.savefig(OUTPUT/'quadrature.png',dpi=180)
fig.savefig(OUTPUT/'quadrature.svg',metadata={'Creator':None,'Date':None})
plt.close(fig)
np.savez_compressed(OUTPUT/'quadrature_plot_arrays.npz',reference_coordinate=z,reference_weight=reference,
    physical_axis=x,physical_axis_weights=axis_w,thermal_variance=variance)
metadata['quadrature_figure']='Probabilists Hermite rule for reference exp(−z²/2), converted explicitly to physical Lebesgue d³v weights. Middle panel projects one tensor plane; all n³ nodal F values are free unknowns. The Gaussian reference is a quadrature measure, with no assumed Gaussian closure.'

# Only after all declared validation gates pass is Landau added to the matched
# four-operator comparison. Lorentz/Dougherty moments are exact for the same F0.
if metadata['status']=='passed':
    tau=alpha*times
    moment=np.array([d['mu_second'] for d in curve])
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    series=[('Constrained local',np.ones_like(tau),np.full_like(tau,2*1.15**2),'#5f6670'),
        ('Lorentz',np.exp(-tau),2.036+.5742857142857143*np.exp(-tau)+.034714285714285714*np.exp(-10*tau/3),'#147d92'),
        ('Dougherty',np.exp(-tau),2*(1+.15*np.exp(-tau))**2,'#c66b28'),
        ('Landau',np.array([d['anisotropy']/.45 for d in curve]),moment,'#7256a8')]
    for label,anisotropy,mu2,color in series:
        axes[0].plot(tau,anisotropy,'o-',color=color,label=label,markersize=3)
        axes[1].plot(tau,mu2-2*1.15**2,'o-',color=color,markersize=3)
    axes[0].set(xlabel='Matched time τ = αt',ylabel='Anisotropy / 0.45',title='One Gaussian, four collision operators')
    axes[1].set(xlabel='Matched time τ = αt',ylabel='Change in ⟨μ²⟩',title='Distribution moments separate')
    axes[0].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.15)
    fig.savefig(OUTPUT/'four_operator_comparison.png',dpi=180)
    plt.close(fig)
    metadata['four_operator_comparison']='passed; actual saved Landau states, homogeneous constrained null state and exact common-Gaussian Lorentz/Dougherty moment formulas from example29'
else:
    metadata['four_operator_comparison']='not_run; declared Landau trajectory validation is unresolved'

metadata['unchanged_frozen_sources']=metadata['frozen_sources']=={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in source_paths}
if not metadata['unchanged_frozen_sources']:metadata['status']='unresolved'
metadata['outputs']={p.name:sha256(p.read_bytes()).hexdigest() for p in sorted(OUTPUT.iterdir()) if p.name!='summary.json'}
save()
print(f'Campaign status={metadata["status"]}; elapsed={metadata["wall_s"]:.1f}s; partial data and actual saved-state figures retained.',flush=True)
if metadata['status']!='passed':
    raise RuntimeError('Declared physical Landau trajectory validation remains unresolved; see the preserved gates and scan rows.')
