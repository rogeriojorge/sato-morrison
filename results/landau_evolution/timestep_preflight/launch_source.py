"""Predeclared independent time preflight; original public Landau equation."""
from pathlib import Path
from hashlib import sha256
import json,time,sys,platform,subprocess,os,resource,importlib.metadata as metadata
import numpy as np,jax
jax.config.update('jax_enable_x64',True)
from sato_morrison.controls import LandauVelocityGrid,landau_entropy_compiler,landau_entropy_step
P=Path(__file__).parent;R=Path('/Users/rogerio/local/sato-morrison');OUTPUT=P/'outputs';OUTPUT.mkdir(exist_ok=True)
INPUT=P/'immutable_input.npz';manifest=json.loads((P/'prepared_manifest.json').read_text());digest=lambda p:sha256(p.read_bytes()).hexdigest()
assert digest(INPUT)==manifest['input_sha256']
for source,expected in manifest['source_sha256'].items():assert digest(Path(source))==expected
with np.load(INPUT) as a:f0,v,w,D=[a[k].copy() for k in ['initial','velocity','weights','derivative']]
assert np.isfinite(f0).all() and (f0>0).all()
state=f0.copy();grid=LandauVelocityGrid((16,)*3,v,w,D);states=[state.copy()];times=[0.]
receipt={'status':'running','plan':{'n':16,'variance':.6,'covariance':[1.15,1.15,.7],'gamma':1,'softening':0,'dt':.0125,'final_time':.2,'steps':16,'rtol':1e-10,'linear_rtol':1e-8,'linear_max_steps':200,'newton_max_steps':30,'max_log_step':2,'preconditioner':'tensor','reference':'fixed original w*F0','density_boxes':[4,5],'GL_orders':[32,48,64],'comparison_reference_dt':.025,'density_moment_target':.01,'moment_normalization_floor':1e-14,'case_budget_s':900,'purpose':'Determine sufficient next time refinement before another full campaign; no change to physical equation/gates'},'launch_manifest':manifest,'source_commit':subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip(),'source_dirty':bool(subprocess.check_output(['git','-C',str(R),'status','--porcelain'],text=True).strip()),'hardware':{'platform':platform.platform(),'processor':platform.processor(),'machine':platform.machine(),'model':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()},'units':'same nondimensional physical velocity/time/population as parent campaign; Gamma=mass=density=1','versions':{k:metadata.version(k) for k in ['numpy','jax','scipy','solvax']},'python':sys.version,'thread_environment':{k:os.getenv(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']},'steps':[]}
path=OUTPUT/'receipt.json'
def save():path.write_text(json.dumps(receipt,indent=2,allow_nan=False))
save();start=time.perf_counter();compiler=landau_entropy_compiler(grid,preconditioner='tensor',reference_population=w*f0,linear_max_steps=200,chunk_size=128)
for j in range(1,17):
 target=j*.0125;record={'time':target,'status':'running','iterations':[]};receipt['steps'].append(record)
 def callback(row):
  record['iterations'].append(row);save();print('target',target,row,flush=True)
  if time.perf_counter()-start>900:raise RuntimeError('Predeclared900s preflight budget exhausted; no current step accepted')
 tick=time.perf_counter()
 try:
  result=landau_entropy_step(grid,state,.0125,compiled=compiler,linear_max_steps=200,rtol=1e-10,linear_rtol=1e-8,max_steps=30,max_log_step=2,iteration_callback=callback)
  state=np.asarray(result.f);states.append(state.copy());times.append(target);record.update(status='passed',wall_s=time.perf_counter()-tick,newton=result.iterations,linear=result.linear_iterations,relative_residual=result.relative_residual,entropy_change=result.entropy_change)
  np.savez_compressed(OUTPUT/'accepted_states.npz',density=np.asarray(states),time=np.asarray(times),velocity=v,weights=w,derivative=D)
 except Exception as error:record.update(status='failed_no_accepted_step',error=str(error),wall_s=time.perf_counter()-tick);receipt['status']='failed';break
 save();print('ACCEPTED',target,flush=True)
else:receipt['status']='passed_finite_trajectory_pending_independent_root_and_accuracy_comparison'
receipt['wall_s']=time.perf_counter()-start;raw_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;receipt['process_peak_rss_bytes']=int(raw_peak if platform.system()=='Darwin' else raw_peak*1024);receipt['peak_rss_scope']='Measured process resource peak including imports/compiler/trajectory; concurrent fine20 process is outside this counter.';receipt['accepted_times']=times;receipt['outputs_sha256']={f.name:digest(f) for f in OUTPUT.glob('*.npz')};save();print('FINAL',receipt['status'],receipt['wall_s'],receipt['process_peak_rss_bytes'],flush=True)
