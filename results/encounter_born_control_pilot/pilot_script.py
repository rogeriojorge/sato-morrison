"""Frozen historical first-Born zero-expectation control diagnostic."""
from pathlib import Path
from time import perf_counter
import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import version
import numpy as np
from scipy.special import roots_legendre
from scipy.integrate import quad

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get('SM_BORN_PILOT_OUTPUT',Path(__file__).with_name('summary.json')))
if OUT.exists():raise FileExistsError(OUT)
started=perf_counter()
root=ROOT
source=root/'results/broad_speed_extended'
declared=dict(source='results/broad_speed_extended',population_keys=['primary2048_phase64','replica2048_phase64'],
    incident_prefix_each=512,phases_each=64,coefficient=1.,gap_mask_min=.3,start_distance=24.,
    quadrature_orders=[256,512,1024],batch_size=256,phase_half_width=.25,impact_split=1.5,perpendicular_split=2.,
    scope='Historical descriptive variance only; no fresh campaign inspection or tuning, no convergence or physical closure claim.')
Path(__file__).with_name('declared_inputs.json').write_text(json.dumps(declared,indent=2)+'\n')
with np.load(source/'populations.npz') as p:ids=np.concatenate([p[k][:512*64] for k in declared['population_keys']])
with np.load(source/'states.npz') as a:
    nodes=a['nodes'][ids];c=a['moments'][ids,0]
B,omega,kappa,screen,L=2.,2.,.03,3.,24.
mask=nodes[:,0]-nodes[:,2]/omega>=.3
def born(order):
    result=np.zeros(len(nodes));z,w=roots_legendre(order);z=z*L;w=w*L
    for start in range(0,len(nodes),256):
        select=np.arange(start,min(start+256,len(nodes)));select=select[mask[select]]
        b,u,p,phi=nodes[select].T
        psi=phi[:,None]-omega*z[None,:]/u[:,None];rho=p/omega
        radius=np.sqrt(b[:,None]**2+rho[:,None]**2-2*b[:,None]*rho[:,None]*np.sin(psi)+z[None,:]**2)
        integrand=np.cos(psi)*np.exp(-radius/screen)*(1+radius/screen)/radius**3
        result[select]=kappa*b*p/(2*B*u)*(integrand@w)
    return result
controls={n:born(n) for n in declared['quadrature_orders']}
fine=controls[1024];residual=c-fine
variance=[]
for ib in range(2):
 for ip in range(2):
  for ia in range(2):
    angular=abs((nodes[:,3]-np.pi/2+np.pi)%(2*np.pi)-np.pi)<.25
    select=((nodes[:,0]<1.5)==(ib==0))&((nodes[:,2]<2.)==(ip==0))&(angular==(ia==0))
    before=float(np.var(c[select],ddof=1));after=float(np.var(residual[select],ddof=1))
    variance.append(dict(impact_bin=ib,perpendicular_bin=ip,phase_bin=ia,count=int(np.sum(select)),
        controlled_count=int(np.sum(select&mask)),original_variance=before,residual_variance=after,
        variance_ratio=after/before,original_mean=float(np.mean(c[select])),residual_mean=float(np.mean(residual[select])),
        sample_control_mean=float(np.mean(fine[select]))))

# The historical regular grid is closed under phi -> pi-phi, so its discrete
# control mean should vanish for each incident point as well as continuously.
odd_error=0.;zero_error=0.
for block in (fine[:512*64].reshape(512,64),fine[512*64:].reshape(512,64)):
    odd_error=max(odd_error,float(np.max(abs(block+block[:,(32-np.arange(64))%64]))))
    zero_error=max(zero_error,float(np.max(abs(np.mean(block,axis=1)))))

def action(node,phi):
    b,u,p,_=node;rho=p/omega
    def f(z):
        radius=np.sqrt(b*b+rho*rho-2*b*rho*np.sin(phi-omega*z/u)+z*z)
        return kappa*np.exp(-radius/screen)/radius/u
    return quad(f,-L,L,epsabs=2e-12,epsrel=2e-11,limit=400)[0]
derivatives=[]
for i in (0,17,65,127,2048,8191):
    if not mask[i]:continue
    node=nodes[i];h=2e-5
    finite=omega/(2*B)*(action(node,node[3]+h)-action(node,node[3]-h))/(2*h)
    derivatives.append(dict(index=i,node=node.tolist(),analytic=float(fine[i]),finite_phase_derivative=float(finite),
        absolute_error=float(abs(finite-fine[i]))))
max_derivative_error=max(r['absolute_error'] for r in derivatives)
quadrature={f'{n}_to1024':dict(maximum_absolute_error=float(np.max(abs(controls[n]-fine))),
        rms_error=float(np.sqrt(np.mean((controls[n]-fine)**2))),
        residual_variance=float(np.var(c-controls[n],ddof=1))) for n in (256,512)}
passed=odd_error<1e-12 and zero_error<1e-13 and max_derivative_error<2e-9
report=dict(status='passed' if passed else 'failed',declared=declared,rows=len(nodes),controlled_rows=int(np.sum(mask)),
    formula='c_B=kappa*b*p/(2*B*u) integral[-L,L] cos(phi-Omega*z/u)*exp(-r/lambda)*(1+r/lambda)/r^3 dz; r^2=b^2+(p/Omega)^2-2b(p/Omega)sin(phi-Omega*z/u)+z^2',
    units='magnetic moment, m*v^2/B; coefficient fixed one',
    symmetry='c_B(pi-phi)=-c_B(phi) for symmetric z bounds/quadrature; both fixed phase strata invariant under this involution, hence each has exact zero conditional phase expectation.',
    checks=dict(maximum_antithetic_odd_residual=odd_error,maximum_historical_incident_control_mean=zero_error,maximum_finite_phase_derivative_error=max_derivative_error),
    derivative_probes=derivatives,quadrature=quadrature,stratum_variance=variance,
    overall_original_variance=float(np.var(c,ddof=1)),overall_residual_variance=float(np.var(residual,ddof=1)),
    overall_variance_ratio=float(np.var(residual,ddof=1)/np.var(c,ddof=1)),wall_s=perf_counter()-started,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    input_sha256={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ('summary.json','states.npz','populations.npz')})
before_proxy=after_proxy=0.
for row in variance:
    ib,ip,ia=row['impact_bin'],row['perpendicular_bin'],row['phase_bin']
    bl,bh=((1.2,1.5),(1.5,2.4))[ib];pl,ph=((0.,2.),(2.,2.5))[ip]
    mass=16*np.pi**2*.5**2*(bh*bh-bl*bl)/(4*np.pi*.5)**1.5
    mass*=(np.exp(-.35**2/2)-np.exp(-3.**2/2))*(np.exp(-pl*pl/2)-np.exp(-ph*ph/2))
    mass*=(.5 if ia==0 else 2*np.pi-.5)/(2*np.pi)
    allocation=2048*(8 if (ib,ip,ia)==(0,1,0) else 1)
    before_proxy+=mass*mass*row['original_variance']/allocation
    after_proxy+=mass*mass*row['residual_variance']/allocation
script=Path(__file__).resolve();relative=str(script.relative_to(root))
report.update(historical_variance_proxy=dict(formula='sum_s W_s^2 Var_s(c)/N_s versus sum_s W_s^2 Var_s(c-c_B)/N_s',
    original=before_proxy,residual=after_proxy,variance_ratio=after_proxy/before_proxy,
    standard_deviation_ratio=float(np.sqrt(after_proxy/before_proxy)),
    scope='Independent-sampling diagnostic from historical finite phase grids and fixed cell masses/allocations; not a Sobol variance estimate, confidence interval, or convergence claim.'),
    proof=dict(force='F=-grad U; partial_phi r_perp=-w_perp/Omega; F dot w_perp=Omega partial_phi U',
      mean='c_B=m Delta(p^2)_B/(8B)=(1/(2B)) integral F dot w_perp dt=(Omega/(2B)) partial_phi integral U dt',
      expectation='For symmetric z bounds and Gauss nodes, c_B(pi-phi)=-c_B(phi). Each central interval and its wrapped complement is invariant under this map; each conditional phase expectation is exactly zero. The phi-independent gap mask preserves zero.',
      quadrature='Any fixed quadrature defines a known-zero control. Finite quadrature differences measure its efficiency approximation, not a bias in the corrected moment.'),
    git_head_at_execution=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
    source_path=relative,source_uncommitted=bool(subprocess.check_output(['git','-C',str(root),'status','--porcelain','--',relative],text=True).strip()),
    versions=dict(python=sys.version.split()[0],numpy=version('numpy'),scipy=version('scipy')),
    platform=platform.platform(),processor=platform.processor(),
    cost_scope='Read-only historical Cartesian samples and fixed scalar/quadrature computations; no new physical trajectory integrations.',
    wall_s=perf_counter()-started,
    limitations=['The historical conditional phase grid is finite; the central interval contains five regular phase nodes.',
      'The 1024-node global Gauss approximation is not declared quadrature-converged.',
      'No fresh incident data enter this pilot; variance reduction requires a frozen independent validation.',
      'No plasma rate, screening closure, reverse-complete population, or lifetime claim.'])
OUT.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('status','rows','controlled_rows','checks','quadrature','overall_variance_ratio','wall_s')},indent=2))
