from pathlib import Path
from hashlib import sha256
from dataclasses import replace
import json,time,sys,platform,os,numpy as np,jax
jax.config.update('jax_enable_x64',True)
from sato_morrison.controls import LandauVelocityGrid,landau_entropy_compiler,landau_entropy_step,landau_gaussian_relative_rate
ROOT=Path(__file__).parents[1];OUT=ROOT/'outputs';OUT.mkdir(exist_ok=False)
INPUT=ROOT/'immutable_input.npz'
with np.load(INPUT) as z:v,w,D,states,times=[z[k].copy() for k in ['velocity','weights','derivative','density','time']]
assert np.array_equal(times,np.array([0.]))
f0=states[0].copy();state=states[-1].copy();states=list(states);times=list(times)
(OUT/'immutable_input.npz').write_bytes(INPUT.read_bytes())
grid=LandauVelocityGrid((20,)*3,v,w,D)
receipt={'status':'running','initial_time':0.,'thermal_variance':.6,'case_budget_s':3000.,'target_time':.2,'dt':.05,'rtol':1e-10,'linear_rtol':1e-12,'linear_max_steps':200,'max_newton':8,'max_log_step':None,'reference':'fixed raw sampled initial wF0','optimizer':'Original unregularized Newton, Hessian, Armijo objective, population congruence and root; no initial log-direction cap','input_sha256':sha256(INPUT.read_bytes()).hexdigest(),'source_hashes':{str(p):sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('/Users/rogerio/local/sato-morrison/src/sato_morrison/controls.py'),Path('/Users/rogerio/local/sato-morrison/src/sato_morrison/solver.py')]},'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'jax':jax.__version__,'thread_variables':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']}},'steps':[]}
p=OUT/'receipt.json'
def save():p.write_text(json.dumps(receipt,indent=2,allow_nan=False))
save();print('Compile UNCAPPED original Newton n20 theta=.6: same physical initialGaussian/raw Coulomb; fixed sampledinitialmetric',flush=True)
c0=landau_entropy_compiler(grid,preconditioner='tensor',reference_population=w*f0,linear_max_steps=200)
active={}
def correct(*args):
 ans=c0.correction(*args);ans[0].block_until_ready()
 active.update(log=np.asarray(args[0]).copy(),direction=np.asarray(ans[0]).copy())
 np.savez_compressed(OUT/f"correction_{len(receipt['steps'])}_{len(active['record']['iterations'])//2+1}.npz",log=active['log'],old_log=np.asarray(args[1]),direction=active['direction'])
 return ans
def objective(*args):
 value=c0.objective_difference(*args);scalar=float(value)
 active['record']['objective_trials'].append({'fraction':float(args[4]),'objective_change':scalar if np.isfinite(scalar) else None,'armijo_rhs':1e-4*float(args[4])*float(args[5])})
 save();return value
c=replace(c0,correction=correct,objective_difference=objective)
print('Compile initial full Coulomb action and evaluate independent Gaussian strong oracle.',flush=True)
A=np.array([1.15,1.15,.7]);g0=np.log(f0)
computed=-np.asarray(c0.apply(g0,g0))/w
rates=[landau_gaussian_relative_rate(v,A,scalar_order=q) for q in [96,192]]
exact=f0*rates[-1];L1=float(w@abs(computed-exact)/(w@abs(exact)));defect=float(np.max(abs(rates[0]-rates[1])))
receipt['initial_gate']={'status':'passed' if L1<.01 and defect<1e-10 else 'failed','relative_L1_error':L1,'scalar_refinement_max_abs':defect,'max_abs_relative_rate':float(np.max(abs(computed/f0))),'sampled_mass':float(w@f0),'axis_extent':float(np.max(abs(v)))}
np.savez_compressed(OUT/'initial_rhs.npz',computed=computed,exact=exact)
save();print('INITIAL_GATE',receipt['initial_gate'],flush=True)
if receipt['initial_gate']['status']!='passed':raise RuntimeError('Initial full-strong consistency gate failed; no trajectory attempted.')
start=time.perf_counter()
for index,target in enumerate([.05,.1,.15,.2]):
 record={'time':target,'status':'running','iterations':[],'objective_trials':[],'trial_history':[]};receipt['steps'].append(record);active['record']=record
 old=state.copy()
 def iteration(row):
  record['iterations'].append(row)
  if 'fraction' in row:
   for back in range(row['backtracks']+1):
    fraction=2.**(-back);candidate=active['log']+fraction*active['direction']
    log_valid=bool(np.all(np.isfinite(candidate)) and np.all(candidate<=np.log(np.finfo(float).max)) and np.all(candidate>=np.log(np.nextafter(0.,1.))))
    population=w*np.asarray(jax.numpy.exp(jax.numpy.asarray(candidate))) if log_valid else None
    pop_valid=bool(population is not None and np.all(np.isfinite(population)) and np.all(population>0))
    record['trial_history'].append({'newton_iteration':row['iteration'],'backtrack':back,'fraction':fraction,'minimum_log':float(np.min(candidate)),'maximum_log':float(np.max(candidate)),'log_representable':log_valid,'population_positive_finite':pop_valid,'selected':back==row['backtracks']})
  save();print('target',target,row,flush=True)
  if time.perf_counter()-start>3000.:raise RuntimeError('Declared3000s private trajectory budget reached; accepted states retained.')
 tick=time.perf_counter()
 try:
  answer=landau_entropy_step(grid,state,.05,compiled=c,linear_max_steps=200,rtol=1e-10,linear_rtol=1e-12,max_steps=8,max_log_step=None,iteration_callback=iteration)
  state=np.asarray(answer.f);record.update(status='passed',wall_s=time.perf_counter()-tick,newton=answer.iterations,linear=answer.linear_iterations,root_relative_residual=answer.relative_residual,entropy_change=answer.entropy_change)
  np.savez_compressed(OUT/f'step_{index+1}.npz',initial=f0,old=old,new=state,velocity=v,weights=w,derivative=D,time=target,dt=.05)
  states.append(state.copy());times.append(target)
 except Exception as error:record.update(status='failed',error=str(error),wall_s=time.perf_counter()-tick);receipt['status']='failed';save();print('FAILED',record,flush=True);break
 np.savez_compressed(OUT/'accepted_states.npz',density=np.asarray(states),time=np.asarray(times),velocity=v,weights=w,derivative=D)
 save();print('ACCEPTED',target,record['wall_s'],flush=True)
else:receipt['status']='passed'
receipt['wall_s']=time.perf_counter()-start
np.savez_compressed(OUT/'accepted_states.npz',density=np.asarray(states),time=np.asarray(times),velocity=v,weights=w,derivative=D)
receipt['output_sha256']={x.name:sha256(x.read_bytes()).hexdigest() for x in OUT.glob('*.npz')}
save();print('FINAL',receipt['status'],receipt['wall_s'],flush=True)
