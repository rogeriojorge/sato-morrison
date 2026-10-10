"""Bounded screened scattering: Born control and magnetic ordering campaign."""
from pathlib import Path
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import hashlib, json, resource, os, csv
import numpy as np
from scipy.special import k0, k1
from sato_morrison.controls import (encounter_relative, encounter_thermal_moments,
    encounter_flux_quadrature, encounter_bounded_flux, encounter_moment_interpolator,
    binary_encounter)
from sato_morrison.reference import run_metadata, progress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'results'/'physical_final_campaign'
MASS,CHARGE,THETA,SCREENING=1.,1.,.5,3.
STRENGTH=.03
FIELDS=(2.,4.,8.,16.,32.)
BOUNDS=((1.2,2.4),(.7,1.5),(.2,.8))
SCAN_ORDERS=(4,4,4,16)
COARSE_ORDERS=(3,3,3,16)
FINAL_ORDERS=(5,5,5,32)
DISTANCE,RTOL,MAX_STEP=24.,1e-12,.2
BOUNDARY_DISTANCES=(18.,24.,32.)
STEP_LEVELS=(.2,.1,.05)
WEAK_STRENGTHS=(.003,.0015)
HELDOUT_COUNT,HELDOUT_SEED=128,3401
BORN_ORDERS=(256,512)
WORKERS,RUNTIME_CAP,CHECKPOINT=4,2400.,128
MOMENT_TARGET,ENERGY_TARGET=.01,1e-9
RING_DELTAS=(np.pi/3,np.pi/2,2*np.pi/3)
RING_PHASES=(0.,.8,1.6,2.4)
SCREENING_CONTROLS=(3.,6.,12.,np.inf)
BORN_TARGET,TABLE_RMS_TARGET,TABLE_P95_TARGET=.03,.02,.05
RING_INCREMENT_TARGET=1e-9
RING_REFINED_STEP=.0025
STEP_TARGET,FLUX_TARGET=.001,1e-5
CARTESIAN_POSITION_TARGET,CARTESIAN_VELOCITY_TARGET=1e-8,1e-8
CUTOFF_DOMAINS={
    'inner_impact':((.6,1.2),BOUNDS[1],BOUNDS[2]),
    'outer_impact':((2.4,4.8),BOUNDS[1],BOUNDS[2]),
    'far_impact':((4.8,8.),BOUNDS[1],BOUNDS[2]),
    'slow_parallel':(BOUNDS[0],(.35,.7),BOUNDS[2]),
    'fast_parallel':(BOUNDS[0],(1.5,3.),BOUNDS[2]),
    'high_perpendicular':(BOUNDS[0],BOUNDS[1],(.8,1.6))}
# Cutoff bands are one-coordinate extensions; they do not cover their corners.
inputs={'mass':MASS,'charge':CHARGE,'theta':THETA,'screening':SCREENING,'strength':STRENGTH,
    'fields':FIELDS,'bounds':BOUNDS,'scan_orders':SCAN_ORDERS,'coarse_orders':COARSE_ORDERS,
    'final_B2_orders':FINAL_ORDERS,'start_distance':DISTANCE,'boundary_distances':BOUNDARY_DISTANCES,
    'rtol':RTOL,'max_step':MAX_STEP,'step_levels':STEP_LEVELS,'weak_strengths':WEAK_STRENGTHS,
    'heldout_count':HELDOUT_COUNT,'heldout_seed':HELDOUT_SEED,'born_orders':BORN_ORDERS,
    'cutoff_domains':CUTOFF_DOMAINS,'workers':WORKERS,'runtime_cap_s':RUNTIME_CAP,
    'moment_target':MOMENT_TARGET,'energy_target':ENERGY_TARGET,'born_target':BORN_TARGET,
    'table_rms_target':TABLE_RMS_TARGET,'table_p95_target':TABLE_P95_TARGET,
    'ring_deltas':RING_DELTAS,'ring_phases':RING_PHASES,'ring_speed':np.sqrt(2*THETA),
    'ring_impact':1.8,'ring_parallel_speed':1.1,'ring_refined_fields':(16.,32.),
    'ring_increment_normalized_target':RING_INCREMENT_TARGET,'ring_refined_max_step':RING_REFINED_STEP,'ring_broadening_signal_to_refinement_minimum':10.,
    'step_target':STEP_TARGET,'flux_target':FLUX_TARGET,'fine_to_coarse_actual_step_ratio_target':.6,
    'cartesian_position_absolute_target':CARTESIAN_POSITION_TARGET,'cartesian_velocity_absolute_target':CARTESIAN_VELOCITY_TARGET,
    'required_gates':['complete_base_scan','energy','phase_and_coordinate_moments','final_B2_refinement','incoming_flux','receding_planes','active_timestep','heldout_table','weak_Born','ring_refinement_and_positive_broadening','cartesian_trajectory','full_moment_consistency','source_integrity'],
    'screening_asymptotic_controls':[3.,6.,12.,'infinity'],
    'units':'m,q,velocity,length normalized; theta=kBT/m; potential=kappa exp(-r/lambda)/r',
    'flux':'2*(2*pi*b db)*u*F_relative*(vperp dvperp du dphi), F covariance=2theta I',
    'mu_reference':'m theta/B; spatial_reference=lambda fixed, independent of B',
    'mean_error_scale':'second moment divided by mu_reference; retain signed drift separately',
    'ordering':'rho_thermal=sqrt(2theta)/Omega; rho_rel=vperp/Omega; b90=2kappa/(m(u^2+vperp^2)); tau_enc=2lambda/u; near_duration=2(b-rho_rel)/u; uniform L_B=infinity',
    'scope':'Finite screened independent-pair incoming model; no plasma coefficient, kinetic mixing time, or lifetime.',
    'stop_rule':'All planned cases through B32; cap checked between batches; failures keep original weights and invalidate integral; no outcome-driven extension.'}


def solve_task(spec):
    node,field,strength,distance,step=spec
    started=perf_counter()
    row={'node':list(map(float,node)),'field':field,'strength':strength,'distance':distance,'max_step':step}
    try:
        b,u,p,phase=node
        r=encounter_relative(b,phase,u,p,field=field,strength=strength,screening=SCREENING,
            start_distance=distance,mass=MASS,charge=CHARGE,max_step=step,rtol=RTOL)
        moment=encounter_thermal_moments(r,THETA,MASS,field)
        omega=CHARGE*field/MASS
        def gc(position,velocity):return position[:2]+np.array([velocity[1],-velocity[0]])/omega
        kick=gc(r['final_relative_position'],r['final_relative_velocity'])-gc(r['initial_relative_position'],r['initial_relative_velocity'])
        row.update(status='passed',moments=moment.tolist(),individual_gc_second=float(kick@kick/4),
            gc_relative_kick=kick.tolist(),initial_relative_velocity=r['initial_relative_velocity'].tolist(),
            final_relative_velocity=r['final_relative_velocity'].tolist(),**{key:r[key] for key in ('time','exit','energy_error','min_separation','end_force','evaluations','max_actual_step')})
        if not np.all(np.isfinite(moment)) or not np.isfinite(kick@kick):raise RuntimeError('Nonfinite observable')
    except Exception as error:
        row.update(status='unresolved',reason=str(error),error_type=type(error).__name__,diagnostics=getattr(error,'integration_diagnostics',{}))
    row.update(wall_s=perf_counter()-started,worker_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return row


born_axes={order:np.polynomial.legendre.leggauss(order) for order in BORN_ORDERS}

def born(node,field,strength,distance,order):
    """First force variation along the exact free helix; independent fixed quadrature."""
    b,u,p,phase=node;omega=CHARGE*field/MASS
    x,weights=born_axes[order]
    s=x*distance/u;weights=weights*distance/u
    angle=phase-omega*s
    wx,wy=p*np.cos(angle),p*np.sin(angle)
    rx,ry,rz=b-wy/omega,wx/omega,u*s
    radius=np.sqrt(rx*rx+ry*ry+rz*rz)
    factor=2*strength/MASS*np.exp(-radius/SCREENING)*(1+radius/SCREENING)/radius**3
    cosine,sine=np.cos(omega*s),np.sin(omega*s)
    delta=np.array([weights@(factor*(cosine*rx-sine*ry)),weights@(factor*(sine*rx+cosine*ry))])
    c=MASS/(4*field)*np.array([p*np.cos(phase),p*np.sin(phase)])@delta
    variance=(MASS/(2*field))**2*THETA/2*(delta@delta)
    return np.array([c,c*c+variance,c*c-variance])


if OUTPUT.exists():raise FileExistsError(f'Preserve existing evidence: {OUTPUT}; choose a fresh declared destination')
OUTPUT.mkdir(parents=True)
(OUTPUT/'declared_inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
# Start fork workers before run_metadata initializes a device runtime.
from jax._src import xla_bridge
if xla_bridge.backends_are_initialized():raise RuntimeError('Fork requires no initialized JAX backend; run this example in a fresh Python process')
pool=ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('fork'))
list(pool.map(float,range(WORKERS)))
import jax
jax.config.update('jax_enable_x64',True)
metadata=run_metadata(inputs,model='screened equal-particle bounded encounter and independent Born controls',boundary='first outgoing relative z=+/-L planes; reflected retained',units=inputs['units'])
dependencies=[Path(__file__).resolve(),ROOT/'src/sato_morrison/controls.py',ROOT/'src/sato_morrison/reference.py']
metadata['experiment_dependency_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
started=perf_counter();cache={};cases=[];attempts=[];budget_exhausted=False
print(f'Physical campaign: {WORKERS} CPU workers; fixed kappa={STRENGTH}; fields={FIELDS}; output={OUTPUT}',flush=True)


def key(spec):return tuple(map(float,(*spec[0],*spec[1:])))


def evaluate(nodes,field=2.,strength=STRENGTH,distance=DISTANCE,step=MAX_STEP):
    global budget_exhausted
    specs=[(tuple(node),float(field),float(strength),float(distance),float(step)) for node in nodes]
    missing={key(spec):spec for spec in specs if key(spec) not in cache}
    pending=list(missing.values())
    for offset in range(0,len(pending),CHECKPOINT):
        if perf_counter()-started>RUNTIME_CAP:
            budget_exhausted=True;break
        batch=pending[offset:offset+CHECKPOINT]
        for spec,row in zip(batch,pool.map(solve_task,batch)):
            cache[key(spec)]=row;attempts.append(row)
            if row['status']!='passed':print(f'  FAILED: {row}',flush=True)
        (OUTPUT/'attempts_partial.json').write_text(json.dumps(attempts,allow_nan=False)+'\n')
        print(f'  Saved {len(attempts)} unique encounters; wall={perf_counter()-started:.1f}s',flush=True)
    return [cache.get(key(spec),{'status':'not_run','node':list(spec[0]),'reason':'predeclared cap'}) for spec in specs]


def integrate(label,orders=SCAN_ORDERS,field=2.,strength=STRENGTH,bounds=BOUNDS,distance=DISTANCE,offset=0.):
    nodes,weights=encounter_flux_quadrature(orders,*bounds,THETA,offset)
    records=evaluate(nodes,field,strength,distance)
    complete=all(r['status']=='passed' for r in records)
    flux=encounter_bounded_flux(*bounds,THETA)
    row={'case':label,'field':field,'strength':strength,'bounds':bounds,'orders':orders,'distance':distance,
        'nodes':len(nodes),'flux_per_density':flux,'flux_quadrature_relative_error':float(abs(weights.sum()/flux-1)),
        'complete':complete,'missing_count':sum(r['status']!='passed' for r in records)}
    if complete:
        moments=np.array([r['moments'] for r in records]);gc=np.array([r['individual_gc_second'] for r in records])
        integral=weights@moments;gcintegral=weights@gc;mu=MASS*THETA/field
        row.update(moments_per_crossing=(integral/flux).tolist(),drift_per_density=float(integral[0]),
            diffusion_mu_per_density=float(integral[1]/2),cross_per_density=float(integral[2]),
            gc_trace_diffusion_per_density=float(gcintegral/2),mu_reference=mu,
            normalized_mu_rate_per_density=float(integral[1]/mu**2),
            normalized_spatial_rate_per_density=float(gcintegral/SCREENING**2),
            max_energy_error=max(r['energy_error'] for r in records),min_separation=min(r['min_separation'] for r in records),
            reflected_count=sum(r['exit']=='reflected' for r in records))
        row['moment_to_spatial_rate_ratio']=row['normalized_mu_rate_per_density']/row['normalized_spatial_rate_per_density']
    cases.append(row);(OUTPUT/'cases_partial.json').write_text(json.dumps(cases,indent=2)+'\n')
    print(f'  {label}: complete={complete}, mu_rate/n={row.get("normalized_mu_rate_per_density")}, gc_rate/n={row.get("normalized_spatial_rate_per_density")}',flush=True)
    return row,nodes,weights,records


def changes(a,b):
    if not a['complete'] or not b['complete']:return [None]*4
    ia=np.array([a['drift_per_density'],2*a['diffusion_mu_per_density'],a['cross_per_density'],2*a['gc_trace_diffusion_per_density']])
    ib=np.array([b['drift_per_density'],2*b['diffusion_mu_per_density'],b['cross_per_density'],2*b['gc_trace_diffusion_per_density']])
    scales=np.array([abs(ia[1])/a['mu_reference'],abs(ia[1]),abs(ia[1]),abs(ia[3])])
    if not np.all(np.isfinite(scales)) or np.any(scales<=0):return [None]*4
    return (abs(ia-ib)/scales).tolist()


scans=[];scan_checks=[];ordering=[];final=None
with progress('Fixed physical-strength magnetic scan, phase and coordinate refinement'):
    for field in FIELDS:
        coarse,*_=integrate(f'B{field:g}_coarse',COARSE_ORDERS,field)
        accepted,nodes,weights,records=integrate(f'B{field:g}',SCAN_ORDERS,field)
        phase8,*_=integrate(f'B{field:g}_phase8',SCAN_ORDERS[:3]+(8,),field)
        shifted,*_=integrate(f'B{field:g}_shifted',SCAN_ORDERS,field,offset=np.pi/SCAN_ORDERS[-1])
        checked={'field':field,'coordinate_changes':changes(accepted,coarse),'phase_changes':changes(accepted,phase8),'shifted_phase_changes':changes(accepted,shifted)}
        # Energy must be below the target; no favorable invariant can replace observable convergence.
        checked['status']='passed' if all(value is not None and value<=MOMENT_TARGET for group in ('coordinate_changes','phase_changes','shifted_phase_changes') for value in checked[group]) and accepted.get('max_energy_error',1)<=ENERGY_TARGET else 'unresolved'
        scan_checks.append(checked);scans.append(accepted)
        if accepted['complete']:
            b,u,p,_=nodes.T;omega=CHARGE*field/MASS;rho=p/omega;gap=b-rho;b90=2*STRENGTH/(MASS*(u*u+p*p))
            observable_weights=weights*np.array([r['moments'][1] for r in records]);observable_weights/=observable_weights.sum()
            measures={'rho_over_lambda':rho/SCREENING,'rho_over_b90':rho/b90,'rho_thermal_over_b90':np.sqrt(2*THETA)/omega/b90,'rho_over_gap':rho/gap,'omega_tau_screen':2*omega*SCREENING/u,'omega_tau_gap':2*omega*gap/u}
            ordrow={'field':field,'rho_thermal_over_lambda':np.sqrt(2*THETA)/omega/SCREENING,'rho_over_LB':0.,'LB':'infinity; uniform field',
                'nu_crossing_per_density_over_omega':accepted['flux_per_density']/omega,'nu_transport_mu_per_density_over_omega':accepted['normalized_mu_rate_per_density']/omega}
            for name,values in measures.items():ordrow[name]={'minimum':float(values.min()),'maximum':float(values.max()),'mu_transport_weighted_mean':float(observable_weights@values)}
            alpha=2*STRENGTH*k0(b/SCREENING)/(MASS*omega*u*SCREENING**2)
            mu_asymptotic=weights@(alpha*alpha*p*p/(8*THETA))
            spatial_asymptotic=weights@(4*STRENGTH**2*k1(b/SCREENING)**2/(MASS**2*omega**2*u*u*SCREENING**4))
            ordrow.update(mu_rate_asymptotic=float(mu_asymptotic),spatial_rate_asymptotic=float(spatial_asymptotic),
                mu_asymptotic_relative_error=float(abs(accepted['normalized_mu_rate_per_density']/mu_asymptotic-1)),
                spatial_asymptotic_relative_error=float(abs(accepted['normalized_spatial_rate_per_density']/spatial_asymptotic-1)))
            ordering.append(ordrow)
    final,finalnodes,finalweights,finalrecords=integrate('B2_final',FINAL_ORDERS)

with progress('Receding planes, cutoffs, and active step refinements'):
    boundaries=[]
    for distance in BOUNDARY_DISTANCES:
        row,*_=integrate(f'boundary_{distance:g}',distance=distance);boundaries.append(row)
    cutoff_rows=[]
    for label,bounds in CUTOFF_DOMAINS.items():
        row,*_=integrate(label,COARSE_ORDERS,bounds=bounds);cutoff_rows.append(row)
    rng=np.random.default_rng(HELDOUT_SEED)
    axes=[np.unique(finalnodes[:,i]) for i in range(4)]
    heldout=np.column_stack([rng.uniform(axis[0],axis[-1],HELDOUT_COUNT) for axis in axes[:3]]+[rng.uniform(0,2*np.pi,HELDOUT_COUNT)])
    step_rows={str(step):evaluate(heldout[:16],step=step) for step in STEP_LEVELS}

with progress('Fresh heldout conditional table and independent small-force Born control'):
    held_records=evaluate(heldout)
    table_check={'status':'unresolved'};born_checks=[]
    if final['complete'] and all(r['status']=='passed' for r in held_records):
        direct=np.array([r['moments'] for r in held_records])
        table=encounter_moment_interpolator(axes,np.array([r['moments'] for r in finalrecords]).reshape(*FINAL_ORDERS,3))
        predicted=table(heldout);rms=np.sqrt(np.mean((direct-predicted)**2,axis=0))/np.sqrt(np.mean(direct**2,axis=0))
        p95=float(np.quantile(abs(predicted[:,1]/direct[:,1]-1),.95))
        table_check={'rms':rms.tolist(),'second_p95':p95,'status':'passed' if rms[1]<=TABLE_RMS_TARGET and p95<=TABLE_P95_TARGET else 'unresolved'}
        np.savez_compressed(OUTPUT/'heldout.npz',nodes=heldout,direct=direct,predicted=predicted)
    for strength in WEAK_STRENGTHS:
        records=evaluate(heldout,strength=strength)
        predicted=np.array([born(node,2.,strength,DISTANCE,BORN_ORDERS[-1]) for node in heldout])
        previous=np.array([born(node,2.,strength,DISTANCE,BORN_ORDERS[0]) for node in heldout])
        if all(r['status']=='passed' for r in records):
            direct=np.array([r['moments'] for r in records]);norm=np.sqrt(np.mean(direct*direct,axis=0))
            rms=np.sqrt(np.mean((direct-predicted)**2,axis=0))/norm
            quad=np.sqrt(np.mean((predicted-previous)**2,axis=0))/norm
            row={'strength':strength,'rms':rms.tolist(),'born_quadrature_change':quad.tolist(),
                'phase_mean_scope':'First-order phase mean vanishes; per-state first-order c tested, exact ensemble drift not predicted.',
                'status':'passed' if max(rms[1:])<=BORN_TARGET and max(quad)<=.001 else 'unresolved'}
            born_checks.append(row)
            np.savez_compressed(OUTPUT/f'born_{strength:g}.npz',nodes=heldout,direct=direct,predicted=predicted,coarse=previous)

ring_rows=[]
with progress('Nonthermal equal-individual-moment ring: exact COM substitutions'):
    ring_nodes=np.array([(1.8,1.1,2*np.sqrt(2*THETA)*np.sin(delta/2),phase) for delta in RING_DELTAS for phase in RING_PHASES])
    for field in FIELDS:
        for step in ((MAX_STEP,RING_REFINED_STEP) if field in (16.,32.) else (MAX_STEP,)):
            records=evaluate(ring_nodes,field=field,step=step)
            for index,row in enumerate(records):
                delta=RING_DELTAS[index//len(RING_PHASES)]
                audit={'field':field,'max_step':step,'delta':float(delta),'phase':float(ring_nodes[index,3]),'status':row['status'],'max_actual_step':row.get('max_actual_step')}
                if row['status']=='passed':
                    win=np.array(row['initial_relative_velocity']);wout=np.array(row['final_relative_velocity']);angle=field*row['time']
                    vcenter=.5/np.tan(delta/2)*np.array([-win[1],win[0]])
                    finalcenter=np.array([np.cos(angle)*vcenter[0]+np.sin(angle)*vcenter[1],-np.sin(angle)*vcenter[0]+np.cos(angle)*vcenter[1]])
                    initialmu=MASS/(2*field)*np.array([np.sum((vcenter+win[:2]/2)**2),np.sum((vcenter-win[:2]/2)**2)])
                    finalmu=MASS/(2*field)*np.array([np.sum((finalcenter+wout[:2]/2)**2),np.sum((finalcenter-wout[:2]/2)**2)])
                    increments=finalmu-initialmu
                    audit.update(initial_mu=initialmu.tolist(),final_mu=finalmu.tolist(),increments=increments.tolist(),
                        delta_total_mu=float(sum(increments)),delta_sum_mu_squared=float(sum(finalmu**2-initialmu**2)),
                        increment_squared_sum=float(sum(increments**2)),mu_reference=MASS*THETA/field,
                        normalized_marginal_broadening=float(sum(finalmu**2-initialmu**2)/(MASS*THETA/field)**2),
                        energy_error=row['energy_error'])
                ring_rows.append(audit)
    (OUTPUT/'ring.json').write_text(json.dumps(ring_rows,indent=2,allow_nan=False)+'\n')

screening_asymptotic=[]
b,u,p,_=encounter_flux_quadrature(SCAN_ORDERS,*BOUNDS,THETA)[0].T
_,weights=encounter_flux_quadrature(SCAN_ORDERS,*BOUNDS,THETA)
for length in SCREENING_CONTROLS:
    omega=FIELDS[-1]
    if np.isfinite(length):
        alpha=2*STRENGTH*k0(b/length)/(MASS*omega*u*length**2)
        relative_gc=4*STRENGTH*k1(b/length)/(MASS*omega*u*length)
    else:
        alpha=np.zeros_like(b);relative_gc=4*STRENGTH/(MASS*omega*u*b)
    screening_asymptotic.append({'screening':float(length) if np.isfinite(length) else 'infinity',
        'field':FIELDS[-1],'normalized_mu_rate_per_density':float(weights@(alpha*alpha*p*p/(8*THETA))),
        'normalized_spatial_rate_per_density_fixed_lambda3':float(weights@(relative_gc*relative_gc/(4*SCREENING**2))),
        'scope':'Analytic infinite-plane leading large-B reference; no finite-B direct validation at changed screening.'})
pool.shutdown()
with progress('Saved Cartesian trajectories, scientific figures and animation'):
    paths={};trajectory_checks=[]
    for index,phase in enumerate((.0,.8,1.6)):
        node=(1.8,1.1,.5,phase)
        row=solve_task((node,2.,STRENGTH,DISTANCE,.05))
        r=encounter_relative(node[0],node[3],node[1],node[2],field=2.,strength=STRENGTH,screening=SCREENING,start_distance=DISTANCE,rtol=RTOL,max_step=.05)
        pair=binary_encounter(r['initial_relative_position'],r['initial_relative_velocity'],field=2.,strength=STRENGTH,screening=SCREENING,duration=r['time'],rtol=RTOL,max_step=.025)
        position=pair['positions'][-1,0]-pair['positions'][-1,1];velocity=pair['velocities'][-1,0]-pair['velocities'][-1,1]
        trajectory_checks.append({'phase':phase,'position_absolute_error':float(max(abs(position-r['final_relative_position']))),'velocity_absolute_error':float(max(abs(velocity-r['final_relative_velocity']))),'energy_error':pair['energy_error']})
        paths.update({f'time{index}':pair['time'],f'positions{index}':pair['positions'],f'velocities{index}':pair['velocities']})
    np.savez_compressed(OUTPUT/'trajectories.npz',**paths)
    fig,ax=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    bvals=np.array([r['field'] for r in scans if r['complete']]);mus=np.array([r['normalized_mu_rate_per_density'] for r in scans if r['complete']]);gcs=np.array([r['normalized_spatial_rate_per_density'] for r in scans if r['complete']])
    ax[0,0].loglog(bvals,mus,'o-',label='moment / mu_ref squared');ax[0,0].loglog(bvals,gcs,'s-',label='spatial / lambda squared')
    ax[0,0].loglog(bvals,[r['mu_rate_asymptotic'] for r in ordering],':',label='screened large-B moment reference');ax[0,0].loglog(bvals,[r['spatial_rate_asymptotic'] for r in ordering],'--',label='screened large-B spatial reference')
    ax[0,0].set(xlabel='magnetic field B',ylabel='conditional rate / partner density',title='Fixed strength and identical incoming measure');ax[0,0].legend(fontsize=7)
    ax[0,1].plot(bvals,mus/gcs,'o-');ax[0,1].set(xlabel='B',ylabel='normalized moment / spatial rate',title='Finite ratio: kick-based proxy, no kinetic mixing time')
    for name in ('rho_over_b90','omega_tau_gap'):
        ax[0,2].loglog(bvals,[r[name]['mu_transport_weighted_mean'] for r in ordering],'o-',label=name)
    ax[0,2].set(xlabel='B',ylabel='moment-transport weighted ordering',title='Ordering measured on actual transport weights');ax[0,2].legend(fontsize=8)
    ax[1,0].bar(range(len(cutoff_rows)+1),[scans[0].get('normalized_mu_rate_per_density',np.nan)]+[r.get('normalized_mu_rate_per_density',np.nan) for r in cutoff_rows])
    for index,row in enumerate(cutoff_rows,1):
        if not row['complete']:ax[1,0].text(index,0,'unresolved',rotation=90,ha='center',va='bottom',fontsize=7)
    ax[1,0].set_xticks(range(len(cutoff_rows)+1),['base']+[r['case'] for r in cutoff_rows],rotation=45,ha='right',fontsize=7);ax[1,0].set(ylabel='conditional moment rate / density',title='Omitted bands can materially alter the rate')
    if table_check['status'] in ('passed','unresolved') and 'rms' in table_check:
        data=np.load(OUTPUT/'heldout.npz');ax[1,1].loglog(data['direct'][:,1],data['predicted'][:,1],'.');limits=[data['direct'][:,1].min(),data['direct'][:,1].max()];ax[1,1].plot(limits,limits,':',color='gray')
    ax[1,1].set(xlabel='fresh direct second moment',ylabel='conditional table prediction',title=f'Heldout table: {table_check["status"]}')
    for index in range(3):
        t=paths[f'time{index}'];pos=paths[f'positions{index}'];ax[1,2].plot(t,np.linalg.norm(pos[:,0]-pos[:,1],axis=1),label=f'phase {(0,.8,1.6)[index]:g}')
    ax[1,2].set(xlabel='actual integration time',ylabel='pair separation',title='Saved Cartesian physical trajectories');ax[1,2].legend()
    fig.savefig(OUTPUT/'physical_validation.png',dpi=170);plt.close(fig)
    fig=plt.figure(figsize=(8,5));axis=fig.add_subplot(111,projection='3d');lines=[axis.plot([],[],[],color=color)[0] for color in ('tab:blue','tab:orange')]
    pos=paths['positions1'];times=paths['time1'];axis.set(xlim=(-2,2),ylim=(-2,2),zlim=(-13,13),xlabel='x',ylabel='y',zlabel='z',title='Screened encounter: actual individual Cartesian paths')
    frames=np.linspace(0,len(times)-1,100).astype(int)
    def animate(frame):
        stop=frames[frame]+1
        for i,line in enumerate(lines):line.set_data_3d(pos[:stop,i,0],pos[:stop,i,1],pos[:stop,i,2])
        axis.set_title(f'Screened encounter: t={times[stop-1]:.2f}, B=2, kappa=.03')
        return lines
    FuncAnimation(fig,animate,frames=len(frames),interval=70).save(OUTPUT/'encounter.gif',writer=PillowWriter(fps=14));plt.close(fig)

step_changes={}
for label,records in step_rows.items():
    if all(r['status']=='passed' for r in records):step_changes[label]=np.array([r['moments'] for r in records])
step_error=None
if '0.05' in step_changes and '0.1' in step_changes:
    a,b=step_changes['0.05'],step_changes['0.1'];scale=np.array([np.sqrt(np.mean(a[:,1]**2))/.25,np.sqrt(np.mean(a[:,1]**2)),np.sqrt(np.mean(a[:,1]**2))])
    if np.all(np.isfinite(scale)) and np.all(scale>0):step_error=(np.sqrt(np.mean((a-b)**2,axis=0))/scale).tolist()
step_activity={'status':'unresolved'}
if all(row['status']=='passed' for row in step_rows['0.1']+step_rows['0.05']):
    ratios=[fine['max_actual_step']/coarse['max_actual_step'] for coarse,fine in zip(step_rows['0.1'],step_rows['0.05'])]
    step_activity={'fine_over_coarse_actual_step_ratios':ratios,'status':'passed' if all(ratio<=.6 for ratio in ratios) else 'unresolved'}
ring_checks=[]
for field in (16.,32.):
    selected=[row for row in ring_rows if row['field']==field]
    for delta in RING_DELTAS:
        for phase in RING_PHASES:
            pair=[row for row in selected if row['delta']==delta and row['phase']==phase]
            check={'field':field,'delta':float(delta),'phase':phase,'status':'unresolved'}
            if len(pair)==2 and all(row['status']=='passed' for row in pair):
                coarse,fine=pair;mu=fine['mu_reference']
                error=float(max(abs(np.array(coarse['increments'])-fine['increments']))/mu)
                broadening_change=abs(coarse['normalized_marginal_broadening']-fine['normalized_marginal_broadening'])
                signal=fine['normalized_marginal_broadening']
                check.update(increment_refinement_error=error,normalized_broadening=signal,broadening_refinement_error=broadening_change,
                    status='passed' if error<=RING_INCREMENT_TARGET and signal>10*broadening_change and signal>0 else 'unresolved')
            ring_checks.append(check)

def within(values,target):return values is not None and all(value is not None and np.isfinite(value) and value<=target for value in values)
dependency_end={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
gates={
    'complete_base_scan':all(row['complete'] for row in scans) and final['complete'],
    'energy':all(row.get('max_energy_error',np.inf)<=ENERGY_TARGET for row in scans+[final]),
    'phase_and_coordinate_moments':all(row['status']=='passed' for row in scan_checks),
    'final_B2_refinement':within(changes(final,scans[0]),MOMENT_TARGET),
    'incoming_flux':all(row['flux_quadrature_relative_error']<=FLUX_TARGET for row in scans+[final]),
    'receding_planes':within(changes(boundaries[-1],boundaries[-2]),MOMENT_TARGET),
    'active_timestep':within(step_error,STEP_TARGET) and step_activity['status']=='passed',
    'heldout_table':table_check['status']=='passed',
    'weak_Born':len(born_checks)==len(WEAK_STRENGTHS) and all(row['status']=='passed' for row in born_checks),
    'ring_refinement_and_positive_broadening':len(ring_checks)==24 and all(row['status']=='passed' for row in ring_checks),
    'cartesian_trajectory':len(trajectory_checks)==3 and all(row['position_absolute_error']<=CARTESIAN_POSITION_TARGET and row['velocity_absolute_error']<=CARTESIAN_VELOCITY_TARGET and row['energy_error']<=ENERGY_TARGET for row in trajectory_checks),
    'full_moment_consistency':all(row['moments'][1]>=abs(row['moments'][2]) and np.all(np.isfinite(row['moments'])) for row in attempts if row['status']=='passed'),
    'source_integrity':metadata['experiment_dependency_sha256']==dependency_end}
overall_status='passed' if all(gates.values()) and not budget_exhausted else 'unresolved'
summary={'overall_status':overall_status,'required_gates':gates,'ring_checks':ring_checks,'scan_checks':scan_checks,'ordering':ordering,'born_checks':born_checks,'heldout_table':table_check,
    'boundary_changes_24_to_32':changes(boundaries[-1],boundaries[-2]),'base_final_refinement_changes':changes(final,scans[0]),
    'step_changes_01_to_005':step_error,'step_activity':step_activity,'trajectory_checks':trajectory_checks,'budget_exhausted':budget_exhausted,
    'ring_rows':ring_rows,'screening_asymptotic_controls':screening_asymptotic,
    'attempted_unique_encounters':len(attempts),'failed_attempts':[r for r in attempts if r['status']!='passed'],
    'limits':['All coefficients are for explicit finite screened incoming bands. Omitted corners and tails remain unbounded.',
        'No density is prescribed; nu/(n Omega) is reported and a physical nu/Omega requires an externally specified density.',
        'Uniform B has rho/L_B=0; this does not prove nonuniform magnetic-moment conservation.',
        'Kick-derived spatial rate is a local random-increment proxy; no successive-encounter correlation or kinetic mixing time was measured.',
        'No extreme-magnetization SM closure, physical SM coefficient, metastability or dipole lifetime follows.']}
with (OUTPUT/'summary.csv').open('w',newline='') as stream:
    columns=['case','field','strength','distance','nodes','flux_per_density','drift_per_density','diffusion_mu_per_density','cross_per_density','gc_trace_diffusion_per_density','normalized_mu_rate_per_density','normalized_spatial_rate_per_density','max_energy_error','complete','missing_count']
    writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(cases)
metadata['results']=summary;metadata['wall_s']=perf_counter()-started
metadata['parent_process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
metadata['worker_process_peak_rss_bytes_max']=max((r['worker_peak_rss_bytes'] for r in attempts),default=0)
metadata['memory_scope']='Separate process lifetime peaks; not a simultaneous aggregate job peak.'
metadata['experiment_dependency_sha256_end']=dependency_end
metadata['experiment_dependencies_unchanged']=metadata['experiment_dependency_sha256']==metadata['experiment_dependency_sha256_end']
metadata['output_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUTPUT.iterdir() if p.is_file()}
(OUTPUT/'summary.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
(OUTPUT/'cases.json').write_text(json.dumps(cases,indent=2,allow_nan=False)+'\n')
(OUTPUT/'attempts.json').write_text(json.dumps(attempts,allow_nan=False)+'\n')
print(json.dumps(summary,indent=2),flush=True)
if overall_status!='passed':raise RuntimeError(f'Declared physical campaign acceptance unresolved; saved every result at {OUTPUT}')
