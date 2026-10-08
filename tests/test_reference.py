"""Small independent checks of the uniform-field oracle."""

import numpy as np
import pytest

from sato_morrison.reference import gauss_interval, pair_rhs


@pytest.mark.parametrize("charge", [1.3, -1.3])
@pytest.mark.parametrize("field", [[0., 0., 2.], [1., -2., 3.]])
def test_fourier_reduction(charge, field):
    u1, wu = gauss_interval(7, -4., 4.)
    eta1, we = gauss_interval(4, 0., 8.)
    u, eta = (a.ravel() for a in np.meshgrid(u1, eta1, indexing="ij"))
    w = (wu[:, None] * we[None, :]).ravel()
    f0 = np.exp(-u*u/2-eta)
    n0 = np.dot(w, f0)
    h = u*u + eta
    k = np.array([0.4, 0.7, -0.2])
    field = np.array(field)
    b = field / np.linalg.norm(field)
    coefficient = 0.8 * np.dot(k-np.dot(k, b)*b, k-np.dot(k, b)*b) / (charge*np.linalg.norm(field))**2
    args = dict(charge=charge, field=field, mass=1., diffusion=0.8)
    actual = pair_rhs(u, w, f0, h, 2*u, k=k, **args)
    expected = -coefficient * f0 * (n0*h - np.dot(w*f0, h))
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
    zero = pair_rhs(u, w, f0, h, 2*u, k=np.zeros(3), **args)
    np.testing.assert_allclose(zero, 0, atol=1e-14)
    density = pair_rhs(u, w, f0, np.ones_like(h), np.zeros_like(h), k=k, **args)
    np.testing.assert_allclose(density, 0, atol=1e-14)
    parallel = pair_rhs(u, w, f0, h, 2*u, k=b, **args)
    np.testing.assert_allclose(parallel, 0, atol=1e-13)


def test_quadrature_input():
    with pytest.raises(ValueError):
        gauss_interval(1, -1, 1)
    x, w = gauss_interval(5, -2, 3)
    np.testing.assert_allclose(np.sum(w), 5.)
    np.testing.assert_allclose(np.dot(w, x*x), 35./3)


@pytest.mark.parametrize('unit_scales',[(2.3,.7,1.9,.8,1.4),(.6,1.8,.9,2.1,.75)])
@pytest.mark.parametrize('charge',[1.3,-1.3])
def test_uniform_collision_observable_under_unit_map(unit_scales,charge):
    # Base units (length,time,mass,charge,number-density); no model parameter changes.
    length,time,mass_unit,charge_unit,density_unit=unit_scales
    speed_unit=length/time;energy_unit=mass_unit*speed_unit**2
    field_unit=mass_unit/(charge_unit*time)
    diffusion_unit=mass_unit**2*length**2/(time**3*density_unit)
    u1,wu=gauss_interval(5,-3,3);eta1,we=gauss_interval(4,0,5)
    u,eta=(a.ravel() for a in np.meshgrid(u1,eta1,indexing='ij'))
    weights=(wu[:,None]*we[None,:]).ravel()
    f=np.exp(-.6*u*u-.9*eta)
    h=u*u+.4*eta;h-=np.dot(weights*f,h)/np.dot(weights,f)
    derivative=2*u;field=np.array([.4,-.8,1.6]);k=np.array([.3,.7,-.1])
    mass,diffusion=1.7,.8
    rhs=pair_rhs(u,weights,f,h,derivative,charge=charge,field=field,
        mass=mass,diffusion=diffusion,k=k)
    changed_weights=weights/(speed_unit*energy_unit)
    changed_f=f*speed_unit*energy_unit/density_unit
    changed=pair_rhs(u/speed_unit,changed_weights,changed_f,h,
        derivative*speed_unit,charge=charge/charge_unit,field=field/field_unit,
        mass=mass/mass_unit,diffusion=diffusion/diffusion_unit,k=k*length)
    # d f_hat/d t_hat = T*U*E/N times the original collision RHS.
    np.testing.assert_allclose(changed,rhs*time*speed_unit*energy_unit/density_unit,
        rtol=3e-12,atol=3e-13)
    rate=-np.dot(weights*h,rhs).real/np.dot(weights*f,h*h)
    changed_rate=-np.dot(changed_weights*h,changed).real/np.dot(changed_weights*changed_f,h*h)
    assert rate>0
    np.testing.assert_allclose(changed_rate/time,rate,rtol=3e-12)
