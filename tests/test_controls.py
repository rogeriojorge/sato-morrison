"""Independent exact equations and physical 3V weak-moment checks."""
import numpy as np
import pytest
from sato_morrison.controls import (GaussianMixture, lorentz_evolve,
    dougherty_evolve, dougherty_rhs, mixture_diagnostics, landau_tensor,
    landau_gaussian_analytic, landau_gaussian_quadrature,
    landau_gaussian_cartesian, landau_weak_moments, binary_encounter,
    coulomb_prefactor)


def mixture():
    return GaussianMixture(np.array([.4, .6]),
        np.array([[-1.2, .4, .2], [1.1, -.1, .2]]),
        np.array([np.diag([.65, 1., .8]), np.diag([.9, .7, 1.2])]))


@pytest.mark.parametrize('ell,rate', [(0,0.),(1,1.),(2,3.),(3,6.)])
def test_lorentz_exact_modes_count_energy_and_momentum(ell, rate):
    xi, weights = np.polynomial.legendre.leggauss(24)
    mode = np.polynomial.legendre.Legendre.basis(ell)(xi)
    f = 1 + .15*mode
    evolved = lorentz_evolve(xi, f, .7, .9, 3)
    expected = 1+.15*np.exp(-rate*.7*.9)*mode
    np.testing.assert_allclose(evolved, expected, atol=2e-15)
    np.testing.assert_allclose(weights@evolved, weights@f, atol=1e-14)
    # Speed-shell energy is a constant times the conserved shell population.
    if ell == 1:
        np.testing.assert_allclose(weights@(xi*evolved), np.exp(-.7*.9)*(weights@(xi*f)), atol=1e-14)


def test_shell_coordinate_measure():
    # u=v xi, mu=m v^2(1-xi^2)/(2B); |Jac|=m v^2/B.
    v, xi, mass, field = 1.7, .36, 2.3, 1.4
    jacobian = np.array([[xi, v], [mass*v*(1-xi*xi)/field, -mass*v*v*xi/field]])
    np.testing.assert_allclose(abs(np.linalg.det(jacobian))*2*np.pi*field/mass, 2*np.pi*v*v)


@pytest.mark.parametrize('ell,rate', [(0,0.),(1,1.),(2,3.),(3,6.)])
def test_lorentz_evolved_modes_in_u_mu_measure(ell, rate):
    # A mapped speed/pitch quadrature in (u,mu), rather than a rectangular box.
    # Its full 3V measure must preserve both the radial and angular moments.
    mass, field, nu, time, amplitude = 1.7, 2.3, .6, .8, .15
    lower, upper = .4, 2.
    speed_nodes, speed_weights = np.polynomial.legendre.leggauss(12)
    speed_nodes = lower+(upper-lower)*(speed_nodes+1)/2
    speed_weights *= (upper-lower)/2
    xi, pitch_weights = np.polynomial.legendre.leggauss(12)
    speed, pitch = np.meshgrid(speed_nodes, xi, indexing='ij')
    u = speed*pitch
    mu = mass*speed**2*(1-pitch**2)/(2*field)
    recovered_speed = np.sqrt(u*u+2*field*mu/mass)
    recovered_pitch = u/recovered_speed
    # Independent determinant of the actual coordinate map at every node.
    jacobian = np.empty(speed.shape+(2,2))
    jacobian[...,0,0], jacobian[...,0,1] = pitch, speed
    jacobian[...,1,0] = mass*speed*(1-pitch*pitch)/field
    jacobian[...,1,1] = -mass*speed*speed*pitch/field
    measure = (2*np.pi*field/mass * abs(np.linalg.det(jacobian))
               * speed_weights[:,None]*pitch_weights[None,:])
    radial = 1+.3*recovered_speed**2
    mode = np.polynomial.legendre.Legendre.basis(ell)(recovered_pitch)
    initial = radial*(1+amplitude*mode)
    evolved = np.array([lorentz_evolve(recovered_pitch[i], initial[i], nu, time, 3)
                        for i in range(len(speed_nodes))])
    np.testing.assert_allclose(evolved,radial*(1+amplitude*np.exp(-rate*nu*time)*mode),
                               rtol=2e-14,atol=2e-14)
    integral = lambda power: ((upper**(power+1)-lower**(power+1))/(power+1)
                            +.3*(upper**(power+3)-lower**(power+3))/(power+3))
    factor = 1+amplitude if ell==0 else 1.
    energy = mass*u*u/2+field*mu
    for population in (initial,evolved):
        assert np.min(population)>0
        np.testing.assert_allclose(np.sum(measure*population),4*np.pi*integral(2)*factor,rtol=2e-14)
        np.testing.assert_allclose(np.sum(measure*energy*population),2*np.pi*mass*integral(4)*factor,rtol=2e-14)
    initial_momentum = 4*np.pi*mass*amplitude*integral(3)/3 if ell==1 else 0.
    np.testing.assert_allclose(np.sum(measure*mass*u*initial),initial_momentum,atol=2e-13)
    np.testing.assert_allclose(np.sum(measure*mass*u*evolved),initial_momentum*np.exp(-nu*time),atol=2e-13)
    if ell==1:
        assert np.sum(measure*mass*u*evolved)<initial_momentum


def test_nonlinear_dougherty_mixture_exact_strong_equation():
    initial = mixture()
    v = np.random.default_rng(301).normal(size=(23,3))
    time, step, nu = .4, 1e-5, .7
    dt = (dougherty_evolve(initial,nu,time+step).evaluate(v)-
          dougherty_evolve(initial,nu,time-step).evaluate(v))/(2*step)
    rhs = dougherty_rhs(dougherty_evolve(initial,nu,time),v,nu)
    np.testing.assert_allclose(dt, rhs, rtol=2e-7, atol=3e-11)
    drift, covariance = initial.moments()
    for time in [.1,.8,3.]:
        m, c = dougherty_evolve(initial,nu,time).moments()
        np.testing.assert_allclose(m,drift,atol=1e-15)
        theta = np.trace(covariance)/3
        np.testing.assert_allclose(c,theta*np.eye(3)+(covariance-theta*np.eye(3))*np.exp(-2*nu*time),atol=8e-16)
        np.testing.assert_allclose(np.trace(c),np.trace(covariance),atol=1e-15)


def test_dougherty_entropy_positive_density_and_quadrature_budgets():
    initial = mixture()
    previous_entropy = -np.inf
    drift,cov = initial.moments()
    for time in [0.,.5,2.]:
        distribution = dougherty_evolve(initial,.7,time)
        diagnostic = mixture_diagnostics(distribution,order=40,extent=9.)
        assert diagnostic['min_f'] > 0
        assert diagnostic['entropy'] > previous_entropy
        assert abs(diagnostic['number']-1) < 2e-9
        np.testing.assert_allclose(diagnostic['mean'],drift,atol=2e-9)
        np.testing.assert_allclose(np.trace(diagnostic['covariance']),np.trace(cov),atol=2e-8)
        previous_entropy = diagnostic['entropy']
    maxwellian=GaussianMixture(np.ones(1),np.array([[.2,-.5,.8]]),np.array([1.3*np.eye(3)]))
    v=np.random.default_rng(7).normal(size=(20,3))
    np.testing.assert_allclose(dougherty_rhs(maxwellian,v,.7),0.,atol=5e-17)


def test_landau_covariance_independent_integrations_and_budgets():
    a=np.diag([1.15,1.15,.7])
    exact=landau_gaussian_analytic(a)
    quadrature=landau_gaussian_quadrature(a,radial_order=28,polar_order=24,phase_order=32)
    np.testing.assert_allclose(quadrature['covariance_rate'],exact,atol=5e-12)
    assert quadrature['entropy_production'] > 0
    assert abs(quadrature['energy_rate']) < 2e-16
    assert abs(np.trace(exact)) < 2e-14
    # Gaussian entropy S=constant+log det(A)/2 has this moment derivative.
    np.testing.assert_allclose(quadrature['entropy_production'],np.trace(np.linalg.solve(a,exact))/2,atol=3e-12)
    errors=[np.linalg.norm(landau_gaussian_cartesian(a,order=n)-exact) for n in [8,16,32]]
    assert errors[2] < errors[1] < errors[0]
    assert errors[2]/np.linalg.norm(exact) < .01


def test_landau_gyro_tail_and_singularity_convergence():
    a=np.diag([1.15,1.15,.7])
    reference=landau_gaussian_analytic(a)
    # All three physical dimensions remain in this gyrotropic reference.
    q=[landau_gaussian_quadrature(a,phase_order=n)['covariance_rate'] for n in [8,16,32]]
    np.testing.assert_allclose(q[-1],q[-2],atol=1e-14)
    tails=[np.linalg.norm(landau_gaussian_quadrature(a,radius=r)['covariance_rate']-reference) for r in [5.,8.,12.]]
    assert tails[2] < tails[1] < tails[0]
    soft=[np.linalg.norm(landau_gaussian_quadrature(a,softening=e)['covariance_rate']-reference) for e in [.2,.1,.05]]
    assert soft[2] < soft[1] < soft[0]


def test_landau_general_positive_nongaussian_and_drift_temperature_maxwellians():
    rng=np.random.default_rng(109)
    velocity=rng.normal(size=(87,3))
    weights=np.full(87,1/87)
    gradients=np.zeros((87,5,3))
    gradients[:,1:4,:]=np.eye(3)
    gradients[:,4,:]=velocity
    for theta,drift in [(.4,np.zeros(3)),(1.2,np.array([.7,-.4,.2])),(2.3,np.array([-.2,.6,.9]))]:
        distribution=GaussianMixture(np.ones(1),drift[None],(theta*np.eye(3))[None])
        f,gradient,_=distribution.evaluate(velocity,True)
        score=gradient/f[:,None]
        relative=velocity[:,None]-velocity[None]
        impulse=np.einsum('abij,abj->abi',landau_tensor(relative),score[:,None]-score[None])
        assert np.max(np.linalg.norm(impulse,axis=-1)) < 5e-14
        moments,entropy=landau_weak_moments(velocity,weights,f,score,gradients)
        np.testing.assert_allclose(moments,0.,atol=1e-18)
        assert abs(entropy)<1e-18
    f,gradient,_=mixture().evaluate(velocity,True)
    moments,entropy=landau_weak_moments(velocity,weights,f,gradient/f[:,None],gradients,chunk=19)
    np.testing.assert_allclose(moments,0.,atol=1e-18)
    assert entropy > 0
    # Independent unordered pair sum checks the factor 1/2.
    entropy_unordered=0.
    score=gradient/f[:,None]
    for i in range(len(velocity)):
        for j in range(i):
            d=score[i]-score[j]
            entropy_unordered+=weights[i]*f[i]*weights[j]*f[j]*(d@landau_tensor(velocity[i]-velocity[j])@d)
    np.testing.assert_allclose(entropy,entropy_unordered,rtol=4e-14)


def test_landau_rotational_covariance_and_invalid_2v():
    rotation=np.array([[.8,-.6,0.],[.6,.8,0.],[0.,0.,1.]])
    a=np.diag([.8,1.3,.9]);rotated=rotation@a@rotation.T
    expected=rotation@landau_gaussian_analytic(a)@rotation.T
    actual=landau_gaussian_quadrature(rotated,radial_order=30,polar_order=30,phase_order=48)['covariance_rate']
    np.testing.assert_allclose(actual,expected,atol=2e-12)
    with pytest.raises(ValueError):
        landau_tensor(np.array([1.,2.]))
    assert coulomb_prefactor(1.,1.,1.,epsilon0=1.) == 1/(8*np.pi)


def test_encounter_zero_interaction_and_energy():
    result=binary_encounter([1.,0.,-8.],[.2,.1,1.],field=.5,strength=0.,duration=16.,max_step=.15)
    np.testing.assert_allclose(result['delta_mu'],0.,atol=1e-14)
    np.testing.assert_allclose(result['delta_gc_perpendicular'],0.,atol=2e-14)
    assert result['energy_error']<1e-13
    result=binary_encounter([1.,0.,-12.],[0.,0.,1.],strength=.08,duration=26.,max_step=.15)
    assert result['energy_error']<1e-9
    final_relative=result['velocities'][-1,0]-result['velocities'][-1,1]
    angle=np.arctan2(np.linalg.norm(final_relative[:2]),final_relative[2])
    # Finite endpoints converge to 2 atan(b90/b), b90=2 strength/v_rel^2.
    assert abs(angle-2*np.arctan(.16)) < .006


def test_incoming_helix_has_fixed_reference_phase_and_impact():
    from sato_morrison.controls import encounter_incoming
    impact,phase,speed,perpendicular,omega=1.2,.7,1.,.35,2.
    for distance in [7.,13.]:
        position,velocity=encounter_incoming(impact,phase,speed,perpendicular,distance,omega)
        result=binary_encounter(position,velocity,field=omega,strength=0.,duration=distance/speed,max_step=.2)
        relative=result['velocities'][-1,0]-result['velocities'][-1,1]
        np.testing.assert_allclose(relative,[perpendicular*np.cos(phase),perpendicular*np.sin(phase),speed],atol=3e-10)
        actual=result['positions'][-1,0]-result['positions'][-1,1]
        np.testing.assert_allclose(actual,[impact-perpendicular*np.sin(phase)/omega,perpendicular*np.cos(phase)/omega,0.],atol=3e-10)


def test_landau_small_anisotropy_two_references_and_linear_limit():
    linear_rate=4/(5*np.sqrt(np.pi))
    errors=[]
    for epsilon in [.03,.01,.003]:
        covariance=np.diag([1+epsilon,1+epsilon,1-2*epsilon])
        analytic=landau_gaussian_analytic(covariance)
        quadrature=landau_gaussian_quadrature(covariance,radial_order=28,polar_order=24,phase_order=32)['covariance_rate']
        np.testing.assert_allclose(quadrature,analytic,atol=2e-13)
        rate=-(analytic[0,0]-analytic[2,2])/(3*epsilon)
        errors.append(abs(rate-linear_rate))
    assert errors[2] < errors[1] < errors[0]
    assert errors[-1]/linear_rate < .002


def test_encounter_incoming_flux_normalization_and_jacobians():
    from sato_morrison.controls import encounter_flux_quadrature, encounter_bounded_flux
    bounds=((1.2,2.4),(.7,1.5),(.2,.8));theta=.5
    nodes,weights=encounter_flux_quadrature((8,8,8,12),*bounds,theta)
    exact=encounter_bounded_flux(*bounds,theta)
    np.testing.assert_allclose(weights.sum(),exact,rtol=2e-14)
    assert np.all(weights>0)
    # Component transformation (V,w)->(v1,v2) has absolute determinant one.
    transform=np.block([[np.eye(3),.5*np.eye(3)],[np.eye(3),-.5*np.eye(3)]])
    np.testing.assert_allclose(abs(np.linalg.det(transform)),1.)
    # Flux is area weighted, not uniform in impact parameter.
    measured=weights@nodes[:,0]/weights.sum()
    lower,upper=bounds[0]
    expected=2*(upper**3-lower**3)/(3*(upper**2-lower**2))
    np.testing.assert_allclose(measured,expected,rtol=2e-14)


@pytest.mark.parametrize('mass,charge',[(1.,1.),(2.3,-.7)])
def test_relative_encounter_matches_independent_pair_path_and_exit_policy(mass,charge):
    from sato_morrison.controls import encounter_relative, encounter_incoming
    b,phase,par,perp,field,distance=1.4,.6,1.1,.4,2.,10.
    relative=encounter_relative(b,phase,par,perp,field=field,strength=.03,screening=3.,start_distance=distance,mass=mass,charge=charge,max_step=.15,rtol=1e-11)
    incoming,velocity=encounter_incoming(b,phase,par,perp,distance,charge*field/mass)
    pair=binary_encounter(incoming,velocity,field=field,strength=.03,screening=3.,duration=relative['time'],mass=mass,charge=charge,max_step=.15,rtol=1e-11)
    np.testing.assert_allclose(relative['final_relative_velocity'],pair['velocities'][-1,0]-pair['velocities'][-1,1],atol=2e-10)
    np.testing.assert_allclose(relative['conditional_mu_mean'],pair['delta_mu'][0],atol=2e-12)
    assert relative['exit']=='transmitted' and relative['parallel_turns']==0
    assert relative['energy_error']<2e-11
    reflected=encounter_relative(.4,.6,.25,.1,field=2.,strength=1.,screening=3.,start_distance=10.,max_step=.15,rtol=1e-11)
    assert reflected['exit']=='reflected' and reflected['parallel_turns']==1


@pytest.mark.parametrize('mass,charge',[(1.,1.),(2.3,-.7)])
def test_thermal_center_moment_formula_independent_gaussian_integration(mass,charge):
    from sato_morrison.controls import encounter_relative, encounter_thermal_moments
    field,theta=2.,.7
    result=encounter_relative(1.4,.6,1.1,.4,field=field,strength=.03,screening=3.,start_distance=10.,mass=mass,charge=charge,max_step=.2,rtol=1e-11)
    expected=encounter_thermal_moments(result,theta,mass=mass,field=field)
    # Gauss-Hermite exactly integrates the quadratic polynomial in V_center.
    x,w=np.polynomial.hermite.hermgauss(3)
    centers=np.sqrt(theta)*np.stack(np.meshgrid(x,x,x,indexing='ij'),axis=-1).reshape(-1,3)
    probabilities=np.prod(np.meshgrid(w,w,w,indexing='ij'),axis=0).ravel()/np.pi**1.5
    angle=charge*field/mass*result['time']
    rotation=np.array([[np.cos(angle),np.sin(angle),0.],[-np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
    final_centers=centers@rotation.T
    initial_relative=result['initial_relative_velocity'];final_relative=result['final_relative_velocity']
    increments=[]
    for sign in [1,-1]:
        initial=centers+sign*initial_relative/2
        final=final_centers+sign*final_relative/2
        increments.append(mass*(np.sum(final[:,:2]**2,axis=1)-np.sum(initial[:,:2]**2,axis=1))/(2*field))
    computed=np.array([probabilities@increments[0],probabilities@(increments[0]**2),probabilities@(increments[0]*increments[1])])
    np.testing.assert_allclose(computed,expected,rtol=2e-10,atol=3e-16)


def test_flux_inverse_cdf_and_invalid_inputs():
    from sato_morrison.controls import encounter_flux_samples, encounter_bounded_flux, encounter_thermal_moments
    theta=.5;bounds=((.2,4.8),(.35,3.),(0.,2.5))
    unit=np.array([[0.,0.,0.],[1.,1.,1.],[.3,.6,.7]])
    nodes=encounter_flux_samples(unit,*bounds,theta)
    np.testing.assert_allclose(nodes[0],[.2,.35,0.],atol=1e-14)
    np.testing.assert_allclose(nodes[1],[4.8,3.,2.5],atol=1e-14)
    recovered=(np.exp(-bounds[1][0]**2/(4*theta))-np.exp(-nodes[:,1]**2/(4*theta)))/(np.exp(-bounds[1][0]**2/(4*theta))-np.exp(-bounds[1][1]**2/(4*theta)))
    np.testing.assert_allclose(recovered,unit[:,1],atol=2e-15)
    with pytest.raises(ValueError):
        encounter_bounded_flux((1.,np.nan),(.7,1.5),(.2,.8),theta)
    with pytest.raises(ValueError):
        encounter_thermal_moments({},np.nan)


def test_periodic_positive_collision_interpolator():
    from sato_morrison.controls import encounter_moment_interpolator
    axes=[np.linspace(1.,2.,4),np.linspace(.6,1.6,4),np.linspace(.2,.8,4),2*np.pi*np.arange(16)/16]
    b,p,t,phase=np.meshgrid(*axes,indexing='ij')
    mean=1e-3*(b-p+t)*(np.sin(phase)+.2*np.cos(2*phase))
    variance=np.exp(-b+2*p-t+.1*np.cos(phase)+.08*np.sin(3*phase))*1e-5
    moments=np.stack((mean,mean**2+variance,mean**2-variance),axis=-1)
    interpolation=encounter_moment_interpolator(axes,moments)
    points=np.array([[1.4,1.1,.45,0.],[1.4,1.1,.45,2*np.pi],[1.4,1.1,.45,-1e-8],[1.4,1.1,.45,1e-8]])
    predictions=interpolation(points)
    np.testing.assert_allclose(predictions[0],predictions[1],atol=2e-17)
    np.testing.assert_allclose(predictions[2],predictions[3],atol=2e-11)
    h=1e-6
    seam=interpolation(np.array([[1.4,1.1,.45,0.],[1.4,1.1,.45,h],[1.4,1.1,.45,2*np.pi-h]]))
    left=(seam[0]-seam[2])/h;right=(seam[1]-seam[0])/h
    np.testing.assert_allclose(left,right,atol=3e-9,rtol=1e-5)
    assert np.all(predictions[:,1]>=predictions[:,0]**2)
    np.testing.assert_allclose(predictions[:,1]+predictions[:,2],2*predictions[:,0]**2,atol=1e-20)


@pytest.mark.parametrize('mass,charge',[(1.,1.),(1.7,-.8)])
def test_reverse_encounter_negates_mean_and_preserves_variance(mass,charge):
    from sato_morrison.controls import encounter_relative, encounter_reverse_incoming, encounter_thermal_moments
    kwargs=dict(field=2.,strength=.03,screening=3.,start_distance=12.,mass=mass,max_step=.15,rtol=1e-11)
    forward=encounter_relative(1.4,.6,1.1,.4,charge=charge,**kwargs)
    inverse,angle=encounter_reverse_incoming(forward,field=2.,mass=mass,charge=charge,start_distance=12.)
    b,parallel,perpendicular,phase=inverse
    reverse=encounter_relative(b,phase,parallel,perpendicular,charge=-charge,**kwargs)
    initial=forward['initial_relative_velocity']
    cosine,sine=np.cos(angle),np.sin(angle)
    expected=np.array([cosine*initial[0]+sine*initial[1],-sine*initial[0]+cosine*initial[1],initial[2]])
    np.testing.assert_allclose(reverse['final_relative_velocity'],expected,atol=3e-9)
    moments_forward=encounter_thermal_moments(forward,.5,mass,2.)
    moments_reverse=encounter_thermal_moments(reverse,.5,mass,2.)
    np.testing.assert_allclose(moments_reverse,moments_forward*np.array([-1,1,1]),atol=2e-11)



def test_encounter_explicit_flight_budget_and_failure_evidence():
    from sato_morrison.controls import encounter_relative
    inputs=dict(field=2.,strength=.03,screening=3.,start_distance=8.,max_step=.2)
    default=encounter_relative(1.4,.6,1.1,.4,**inputs)
    explicit=encounter_relative(1.4,.6,1.1,.4,flight_time_factor=6.,**inputs)
    np.testing.assert_array_equal(default['final_relative_velocity'],explicit['final_relative_velocity'])
    assert default['time']==explicit['time'] and default['evaluations']==explicit['evaluations']
    with pytest.raises(RuntimeError) as caught:
        encounter_relative(1.4,.6,1.1,.4,flight_time_factor=.5,**inputs)
    diagnostic=caught.value.integration_diagnostics
    assert diagnostic['solver_success'] and diagnostic['solver_status']==0
    assert diagnostic['final_time']==diagnostic['budget']
    assert diagnostic['final_position'][2]<8.
    for factor in [0.,-1.,float('nan')]:
        with pytest.raises(ValueError):
            encounter_relative(1.4,.6,1.1,.4,flight_time_factor=factor,**inputs)
