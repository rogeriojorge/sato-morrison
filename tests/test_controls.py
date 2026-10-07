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
