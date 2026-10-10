"""Resolve saved sensitive encounters using independent arbitrary-precision Taylor integration."""
from pathlib import Path
from time import perf_counter
import gzip,hashlib,json,platform,sys
import mpmath as mp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sato_morrison.reference import run_metadata
import time

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'results'/'encounter_precision'/'saved_binary64'
PHASE_SOURCE=ROOT/'results'/'encounter_phase'/'all_moments8192'/'nodes.json'
CENSOR_SOURCE=ROOT/'results'/'encounter_censoring'/'precision_checked'/'summary.json'
REFINEMENTS=((50,28,'1e-30'),(70,36,'1e-44'))
POSITION_TARGET,VELOCITY_TARGET,TIME_TARGET=1e-8,1e-8,1e-8

def coefficient(state,order,strength,screening,omega):
    z=mp.mpf(0)
    a=[[v]+[z]*order for v in state]
    u=[z]*order; radius=[z]*order; inverse_cube=[z]*order; exponential=[z]*order; force=[z]*order
    for n in range(order):
        u[n]=sum(sum(a[j][k]*a[j][n-k] for k in range(n+1)) for j in range(3))
        if n==0:
            radius[0]=mp.sqrt(u[0]); inverse_cube[0]=u[0]**mp.mpf('-1.5'); exponential[0]=mp.exp(-radius[0]/screening)
        else:
            radius[n]=sum((mp.mpf('1.5')*k-n)*u[k]*radius[n-k] for k in range(1,n+1))/(n*u[0])
            inverse_cube[n]=sum((-mp.mpf('.5')*k-n)*u[k]*inverse_cube[n-k] for k in range(1,n+1))/(n*u[0])
            exponential[n]=-sum(k*radius[k]*exponential[n-k] for k in range(1,n+1))/(n*screening)
        product=sum(exponential[k]*radius[n-k] for k in range(n+1))/screening+exponential[n]
        # Store exp(-r/lambda)*(1+r/lambda); convolve with inverse-cube.
        force[n]=product
        factor=2*strength*sum(force[k]*inverse_cube[n-k] for k in range(n+1))
        # factor coefficients retained separately from the first product.
        if n==0: factors=[factor]
        else:factors.append(factor)
        for j in range(3):a[j][n+1]=a[j+3][n]/(n+1)
        acc=[sum(factors[k]*a[j][n-k] for k in range(n+1)) for j in range(3)]
        acc[0]+=omega*a[4][n]; acc[1]-=omega*a[3][n]
        for j in range(3):a[j+3][n+1]=acc[j]/(n+1)
    return a

def evaluate(a,h):
    result=[]
    for polynomial in a:
        v=polynomial[-1]
        for c in polynomial[-2::-1]:v=v*h+c
        result.append(v)
    return result

def energy(y,strength,screening):
    radius=mp.sqrt(sum(v*v for v in y[:3]))
    return sum(v*v for v in y[3:])/4+strength*mp.exp(-radius/screening)/radius

def solve(initial,duration,*,dps=50,order=28,tolerance='1e-30',max_step='.5',events=False, progress_label=''):
    mp.mp.dps=dps
    # mp.mpf(float) preserves the exact binary64 value, not decimal spelling.
    y=[mp.mpf(float(v)) for v in initial]; t=mp.mpf(0); T=mp.mpf(float(duration))
    strength=mp.mpf(float(.03)); screening=mp.mpf(float(3.)); omega=mp.mpf(float(2.)); plane=mp.mpf(24)
    tol=mp.mpf(tolerance); cap=mp.mpf(max_step); start=time.perf_counter(); report=start
    ts=[str(t)]; states=[[str(v) for v in y]]; E0=energy(y,strength,screening); max_energy=mp.mpf(0); count=0; event=None
    while t<T:
        a=coefficient(y,order,strength,screening,omega)
        h=min(cap,T-t)
        for n in range(order-3,order+1):
            scale=max(abs(p[n]) for p in a)
            if scale:h=min(h,mp.mpf('.75')*(tol/scale)**(mp.mpf(1)/n))
        yn=evaluate(a,h)
        which=None
        if y[2]<plane and yn[2]>=plane and yn[5]>0:which=plane
        if y[2]>-plane and yn[2]<=-plane and yn[5]<0:which=-plane
        if which is not None and event is None:
            left=mp.mpf(0); right=h
            for _ in range(dps*4):
                midpoint=(left+right)/2; ym=evaluate(a,midpoint)
                if (ym[2]-which)*(yn[2]-which)>0:right=midpoint
                else:left=midpoint
                if right-left<mp.mpf(10)**(-dps+8):break
            he=(left+right)/2; ye=evaluate(a,he)
            event={'time':str(t+he),'state':[str(v) for v in ye],'exit':'transmitted' if which>0 else 'reflected'}
            if events:h=he; yn=ye
        t+=h; y=yn; count+=1; max_energy=max(max_energy,abs(energy(y,strength,screening)-E0)/abs(E0))
        ts.append(str(t)); states.append([str(v) for v in y])
        if time.perf_counter()-report>25:
            print(progress_label,'step',count,'t',float(t),'r',float(mp.sqrt(sum(v*v for v in y[:3]))),'wall',time.perf_counter()-start,flush=True); report=time.perf_counter()
        if events and event is not None:break
    return {'time':str(t),'state':[str(v) for v in y],'event':event,'steps':count,'energy_error':str(max_energy),'wall_s':time.perf_counter()-start,'dps':dps,'order':order,'tolerance':tolerance,'max_step':max_step},ts,states

phase_rows=json.loads(PHASE_SOURCE.read_text())
censor=json.loads(CENSOR_SOURCE.read_text())
phase=next(r for r in phase_rows if r['master_phase_index']==2054 and r['max_step']==.025)
cases=[{'label':'phase2054','initial':phase['initial_relative_position']+phase['initial_relative_velocity'],
        'duration':phase['flight_budget'],'original_record':phase,'absorb_at_first_outgoing_plane':True}]
for row in censor['results']['input_preparation']:
    cases.append({'label':f"endpoint{row['endpoint_index']}",'initial':row['initial_double_state'],
        'duration':censor['inputs']['flight_time_factor']*24/1.9056642462041216,
        'original_record':row,'absorb_at_first_outgoing_plane':False})
for case in cases:
    case['initial_binary64_hex']=[float(v).hex() for v in case['initial']]
inputs={'cases':cases,'mass':1.,'charge':1.,'field':2.,'strength':.03,'screening':3.,'start_distance':24.,
    'refinements':REFINEMENTS,'max_step':'.5','position_absolute_target':POSITION_TARGET,
    'velocity_absolute_target':VELOCITY_TARGET,'time_absolute_target':TIME_TARGET,
    'initial_convention':'Convert each saved binary64 state component and physical parameter exactly using mp.mpf(float). No trigonometric re-preparation; phase parameter and initial-state uncertainty are distinct.',
    'method':'Cartesian6D analytic Taylor coefficients; sqrt, inverse-cube power and exponential series; adaptive steps from the last4 coefficients, with safety0.75. First outgoing event located within the step polynomial by high-precision bisection.',
    'event_convention':'First upward z=+24 or downward z=-24 crossing; initial incoming -24 is excluded. Phase2054 absorbs there. Saved endpoint cases continue the unabsorbed IVP to identical binary64 T.',
    'acceptance':'Independent50/70-digit endpoint and first-exit agreement at unchanged absolute1e-8 targets; energy recorded separately. This is numerical convergence evidence, without interval certification.',
    'scope':'Fixed saved states only; preserves original failures. No new phase search, continuous gyrophase integral, rate, sourceD calibration, trapping or lifetime conclusion.'}
OUTPUT.mkdir(parents=True,exist_ok=True)
metadata=run_metadata(inputs,model='arbitrary-precision screened equal-particle encounter validation',
    boundary='first outgoing z=+/-24 and separate full-budget unabsorbed endpoints',units='normalized')
metadata['mpmath_version']=mp.__version__
metadata['exact_binary64_physical_parameters']={k:float(inputs[k]).hex() for k in ['mass','charge','field','strength','screening','start_distance']}
sources=[Path(__file__).resolve(),ROOT/'src/sato_morrison/controls.py',ROOT/'src/sato_morrison/reference.py',PHASE_SOURCE,CENSOR_SOURCE]
metadata['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
metadata['retained_original_censoring_results']=censor['results']
(OUTPUT/'declared_inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
runs=[];arrays={};comparisons=[];started=perf_counter()
for case in cases:
    pair=[]
    for dps,order,tolerance in REFINEMENTS:
        label=f"{case['label']}_dps{dps}"
        print('Integrating',label,'from exact saved binary64 initial state',flush=True)
        result,times,states=solve(case['initial'],case['duration'],dps=dps,order=order,
            tolerance=tolerance,max_step='.5',events=case['absorb_at_first_outgoing_plane'],progress_label=label)
        result['label']=label;pair.append(result);runs.append(result)
        with gzip.open(OUTPUT/f'{label}_full_precision_path.json.gz','wt') as stream:
            json.dump({'time':times,'state':states},stream)
        arrays[label+'_time']=np.asarray(times,float);arrays[label+'_state']=np.asarray(states,float)
        print(label,'first_exit=',result['event']['exit'] if result['event'] else None,
              'event_time=',result['event']['time'] if result['event'] else None,
              'final_z=',result['state'][2],'energy_error=',result['energy_error'],'wall_s=',result['wall_s'],flush=True)
        metadata['results']={'status':'incomplete','runs':runs}
        (OUTPUT/'summary_partial.json').write_text(json.dumps(metadata,indent=2)+'\n')
    mp.mp.dps=70
    low,high=pair;difference=[abs(mp.mpf(a)-mp.mpf(b)) for a,b in zip(low['state'],high['state'])]
    event_time_difference=abs(mp.mpf(low['event']['time'])-mp.mpf(high['event']['time'])) if low['event'] and high['event'] else None
    check={'case':case['label'],'position_difference':str(max(difference[:3])),
        'velocity_difference':str(max(difference[3:])),
        'first_exit_time_difference':str(event_time_difference) if event_time_difference is not None else None,
        'matching_first_exit':bool(low['event'] and high['event'] and low['event']['exit']==high['event']['exit'])}
    check['passed']=(max(difference[:3])<=POSITION_TARGET and max(difference[3:])<=VELOCITY_TARGET
        and event_time_difference is not None and event_time_difference<=TIME_TARGET and check['matching_first_exit'])
    comparisons.append(check)
initial_difference=max(abs(a-b) for a,b in zip(cases[1]['initial'],cases[2]['initial']))
ends=[next(r for r in runs if r['label']==f'endpoint{i}_dps70') for i in range(2)]
final_difference=max(abs(mp.mpf(a)-mp.mpf(b)) for a,b in zip(ends[0]['state'],ends[1]['state']))
metadata['results']={'status':'passed_fixed_state_precision_checks' if all(c['passed'] for c in comparisons) else 'unresolved',
    'runs':runs,'precision_comparisons':comparisons,'wall_s':perf_counter()-started,
    'adjacent_saved_state_sensitivity':{'max_initial_binary64_difference':initial_difference,
        'max_final_state_difference':str(final_difference),'finite_difference_amplification':str(final_difference/mp.mpf(initial_difference)),
        'interpretation':'Large finite perturbation amplification; not a Lyapunov exponent or a certified error bound.'},
    'original_precision_checked_status':censor['results']['status'],
    'original_opposite_exit_bracket_reproduced':False if all(e['event'] and e['event']['exit']=='transmitted' for e in ends) else None,
    'limits':inputs['scope']}
metadata['source_sha256_end']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
metadata['results']['dependencies_unchanged']=metadata['source_sha256']==metadata['source_sha256_end']
np.savez_compressed(OUTPUT/'paths_binary64.npz',**arrays)
metadata['output_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUTPUT.glob('*path.json.gz')}
metadata['output_sha256']['paths_binary64.npz']=hashlib.sha256((OUTPUT/'paths_binary64.npz').read_bytes()).hexdigest()
fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
for axis,case in zip(axes,cases):
    for dps,_,_ in REFINEMENTS:
        label=f"{case['label']}_dps{dps}";axis.plot(arrays[label+'_time'],arrays[label+'_state'][:,2],label=f'{dps} decimal digits',ls='-' if dps==70 else '--')
    for plane in [-24,24]:axis.axhline(plane,color='.6',ls=':',lw=1)
    axis.set(xlabel='normalized time',ylabel='relative z',title=case['label']);axis.legend(fontsize=8)
fig.suptitle('Saved binary64 states: independent precision refinement; original failures retained')
fig.savefig(OUTPUT/'precision_paths.png',dpi=180);plt.close(fig)
(OUTPUT/'summary.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Finished:',metadata['results']['status'],'wall_s=',metadata['results']['wall_s'],flush=True)
if not all(c['passed'] for c in comparisons) or not metadata['results']['dependencies_unchanged']:
    raise RuntimeError('Fixed-state precision validation unresolved; all paths retained')
