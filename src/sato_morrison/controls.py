"""Independent homogeneous controls and full three-velocity Coulomb moments.

The Gaussian Landau routines evaluate weak instantaneous moments. The nodal
Landau evolution advances a full 3V density, with no Gaussian closure.
The Dougherty solution is exact for any positive finite Gaussian mixture.
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.special import logsumexp
from .solver import LaggedEntropyCompiler


def lorentz_evolve(xi, population, nu, time, degree):
    """Legendre Galerkin solution on a fixed speed shell; count measure dxi."""
    xi, population = np.asarray(xi), np.asarray(population)
    if nu < 0 or time < 0 or degree < 0 or population.shape != xi.shape:
        raise ValueError("Invalid Lorentz inputs.")
    nodes, weights = np.polynomial.legendre.leggauss(xi.size)
    if not np.allclose(xi, nodes, atol=1e-14, rtol=0):
        raise ValueError("Use Gauss-Legendre pitch-angle nodes.")
    basis = np.polynomial.legendre.legvander(xi, degree)
    ell = np.arange(degree + 1)
    coefficients = (2 * ell + 1) / 2 * (basis.T @ (weights * population))
    rates = nu * ell * (ell + 1) / 2
    return basis @ (coefficients * np.exp(-rates * time))


@dataclass(frozen=True)
class GaussianMixture:
    """A normalized 3V density, component means, and positive covariance matrices."""
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray

    def __post_init__(self):
        p, means, cov = map(lambda a: np.asarray(a, dtype=float),
                            (self.weights, self.means, self.covariances))
        if p.ndim != 1 or means.shape != (len(p), 3) or cov.shape != (len(p), 3, 3):
            raise ValueError("Need K weights, Kx3 means, and Kx3x3 covariances.")
        if not np.all(p > 0) or not np.isclose(p.sum(), 1) or not np.all(np.isfinite(means)):
            raise ValueError("Finite means and normalized positive weights required.")
        if not np.allclose(cov, cov.swapaxes(-1, -2)) or np.any(np.linalg.eigvalsh(cov) <= 0):
            raise ValueError("Covariances must be positive definite and symmetric.")
        object.__setattr__(self, 'weights', p)
        object.__setattr__(self, 'means', means)
        object.__setattr__(self, 'covariances', cov)

    def moments(self):
        mean = self.weights @ self.means
        centered = self.means - mean
        covariance = np.einsum('k,kij->ij', self.weights, self.covariances)
        covariance += np.einsum('k,ki,kj->ij', self.weights, centered, centered)
        return mean, covariance

    def evaluate(self, velocity, derivatives=False):
        """Density and optionally its exact gradient and Laplacian."""
        v = np.asarray(velocity)
        delta = v[..., None, :] - self.means
        inverse = np.linalg.inv(self.covariances)
        score = -np.einsum('kij,...kj->...ki', inverse, delta)
        log_terms = np.log(self.weights) - 1.5*np.log(2*np.pi)
        log_terms = log_terms - 0.5*np.linalg.slogdet(self.covariances)[1]
        log_terms = log_terms + 0.5*np.einsum('...ki,...ki->...k', delta, score)
        log_density = logsumexp(log_terms, axis=-1)
        density = np.exp(log_density)
        if not derivatives:
            return density
        terms = np.exp(log_terms)
        gradient = np.einsum('...k,...ki->...i', terms, score)
        laplacian = np.sum(terms*(np.sum(score**2, axis=-1)-np.trace(inverse, axis1=1, axis2=2)), axis=-1)
        return density, gradient, laplacian


def dougherty_evolve(initial, nu, time):
    """Exact nonlinear conserving Dougherty evolution of a nonGaussian mixture.

    U and theta are the moments of the evolved species, fixed by its own
    conservation laws. There is no independently prescribed reservoir.
    """
    if nu < 0 or time < 0:
        raise ValueError("Rate and time must be nonnegative.")
    drift, covariance = initial.moments()
    theta = np.trace(covariance)/3
    decay = np.exp(-nu*time)
    return GaussianMixture(initial.weights, drift + decay*(initial.means-drift),
        decay**2*initial.covariances + theta*(1-decay**2)*np.eye(3))


def dougherty_rhs(distribution, velocity, nu):
    """Strong 3V Fokker-Planck action evaluated using mixture derivatives."""
    drift, covariance = distribution.moments()
    f, gradient, laplacian = distribution.evaluate(velocity, derivatives=True)
    return nu*(3*f + np.sum((np.asarray(velocity)-drift)*gradient, axis=-1)
               + np.trace(covariance)/3*laplacian)


def cartesian_quadrature(order, extent):
    """Physical velocity volume quadrature over [-extent,extent]^3."""
    if order < 2 or extent <= 0:
        raise ValueError("Invalid velocity quadrature.")
    x, w = np.polynomial.legendre.leggauss(order)
    axes = np.meshgrid(extent*x, extent*x, extent*x, indexing='ij')
    wx = np.meshgrid(extent*w, extent*w, extent*w, indexing='ij')
    return np.stack(axes, axis=-1).reshape(-1, 3), np.prod(wx, axis=0).ravel()


def mixture_diagnostics(distribution, order=36, extent=9.):
    v, w = cartesian_quadrature(order, extent)
    f = distribution.evaluate(v)
    mass = w@f
    mean = np.einsum('n,ni->i', w*f, v)/mass
    c = v-mean
    covariance = np.einsum('n,ni,nj->ij', w*f, c, c)/mass
    entropy = -np.sum(w*f*np.log(np.maximum(f, np.finfo(float).tiny)))
    return {'number': float(mass), 'mean': mean, 'covariance': covariance,
            'entropy': float(entropy), 'min_f': float(f.min())}


def coulomb_prefactor(charge, mass, coulomb_log, epsilon0=8.8541878128e-12):
    """SI Gamma=q^4 ln(Lambda)/(8 pi epsilon0^2 m^2), F integrates to n.

    For F=(n/v0^3) f(v/v0), the normalized rate is Gamma*n/v0^3.
    The Coulomb logarithm/cutoffs must be supplied; no magnetic rate calibration.
    """
    if mass <= 0 or coulomb_log <= 0 or epsilon0 <= 0 or charge == 0:
        raise ValueError("Invalid Coulomb parameters.")
    return charge**4*coulomb_log/(8*np.pi*epsilon0**2*mass**2)


def landau_tensor(relative_velocity, softening=0.):
    """Physical 3V transverse Coulomb kernel. Coincident sample is zero.

    Zero at coincidence is a quadrature prescription, not the directional limit.
    Gaussian spherical quadrature instead never samples coincidence. Optional
    softening changes only 1/r, preserving the exact energy null direction.
    """
    w = np.asarray(relative_velocity)
    if w.shape[-1] != 3 or softening < 0:
        raise ValueError("Landau kernel needs physical three-velocities.")
    r2 = np.sum(w*w, axis=-1)
    safe = np.where(r2 > 0, r2, 1.)
    projector = np.eye(3)-w[..., :, None]*w[..., None, :]/safe[..., None, None]
    denominator = np.sqrt(safe+softening**2)
    return np.where((r2 > 0)[..., None, None], projector/denominator[..., None, None], 0.)


def _covariance(covariance):
    a = np.asarray(covariance, dtype=float)
    if a.shape != (3, 3) or not np.allclose(a, a.T) or np.any(np.linalg.eigvalsh(a) <= 0):
        raise ValueError("Need a symmetric positive definite 3V covariance.")
    return a


def landau_gaussian_analytic(covariance, gamma=1.):
    """Independent 1D Laplace-transform covariance derivative, n=1.

    Integrates Gaussian quadratic moments analytically before a scalar adaptive
    integral. Does not call the spherical or Cartesian quadrature or kernel.
    """
    a = _covariance(covariance)
    eigenvalues, rotation = np.linalg.eigh(a)
    rates = []
    for i in range(3):
        def integrand(s):
            # t=s^2 removes the t^-1/2 endpoint singularity.
            t = s*s
            z = np.prod(1+4*eigenvalues*t)**-0.5
            variance = 2*eigenvalues/(1+4*eigenvalues*t)
            factor = np.ones(3)
            factor[i] = 3
            return 2*z*variance[i]*(1/eigenvalues[i] - 2*t*np.sum(factor*variance/eigenvalues))
        rates.append(gamma/np.sqrt(np.pi)*quad(integrand, 0, np.inf, epsabs=2e-12, epsrel=2e-12)[0])
    return rotation @ np.diag(rates) @ rotation.T


def landau_gaussian_quadrature(covariance, gamma=1., radial_order=48,
                               polar_order=32, phase_order=48, radius=12., softening=0.):
    """3V Gaussian weak moments with explicit radial, polar, gyrophase integrals.

    Relative w has covariance 2A; center velocity integrates analytically. Count
    and momentum vanish by their zero pair gradient; energy uses U(w)w=0.
    The returned covariance is an instantaneous derivative, not a Gaussian
    closure time evolution. Entropy production is evaluated independently from
    the pair log-density score. F integrates to one.
    """
    a = _covariance(covariance)
    if gamma < 0 or min(radial_order, polar_order, phase_order) < 2 or radius <= 0:
        raise ValueError("Invalid Landau quadrature.")
    r, wr = np.polynomial.legendre.leggauss(radial_order)
    r, wr = radius*(r+1)/2, radius*wr/2
    z, wz = np.polynomial.legendre.leggauss(polar_order)
    phi = 2*np.pi*np.arange(phase_order)/phase_order
    rr, zz, pp = np.meshgrid(r, z, phi, indexing='ij')
    directions = np.stack((np.sqrt(1-zz*zz)*np.cos(pp), np.sqrt(1-zz*zz)*np.sin(pp), zz), axis=-1)
    w = (rr[..., None]*directions).reshape(-1, 3)
    weights = (wr[:, None, None]*wz[None, :, None]*np.ones((1, 1, phase_order))
               *2*np.pi/phase_order*rr**2).ravel()
    inverse = np.linalg.inv(a)
    g = -(w @ inverse.T)
    relative_density = np.exp(-0.25*np.einsum('ni,ij,nj->n', w, inverse, w)) / np.sqrt((4*np.pi)**3*np.linalg.det(a))
    uw = np.einsum('nij,nj->ni', landau_tensor(w, softening), g)
    measure = weights*relative_density
    covariance_rate = -gamma/2*np.einsum('n,ni,nj->ij', measure, uw, w)
    covariance_rate += covariance_rate.T
    entropy = gamma/2*np.sum(measure*np.einsum('ni,ni->n', g, uw))
    energy_residual = -gamma/2*np.sum(measure*np.einsum('ni,ni->n', w, uw))
    return {'covariance_rate': covariance_rate, 'entropy_production': float(entropy),
            'number_rate': 0., 'momentum_rate': np.zeros(3),
            'energy_rate': float(energy_residual), 'relative_probability': float(measure.sum())}


def landau_gaussian_cartesian(covariance, gamma=1., order=32):
    """Independent Cartesian Gauss-Hermite expectation; no pair diagonal nodes.

    Even orders avoid w=0. Slow convergence near w=0 is measured, not hidden.
    """
    a = _covariance(covariance)
    if order < 2 or order % 2:
        raise ValueError("Use an even Cartesian Hermite order.")
    x, p = np.polynomial.hermite.hermgauss(order)
    grid = np.stack(np.meshgrid(x, x, x, indexing='ij'), axis=-1).reshape(-1, 3)
    prob = np.prod(np.meshgrid(p, p, p, indexing='ij'), axis=0).ravel()/np.pi**1.5
    w = 2*grid @ np.linalg.cholesky(a).T
    score = -w @ np.linalg.inv(a).T
    uw = np.einsum('nij,nj->ni', landau_tensor(w), score)
    rate = -gamma/2*np.einsum('n,ni,nj->ij', prob, uw, w)
    return rate+rate.T


def binary_encounter(relative_position, relative_velocity, *, field=0.,
                     strength=0.1, mass=1., charge=1., screening=np.inf,
                     duration=40., max_step=0.05, rtol=1e-10):
    """Direct equal-mass/equal-charge repulsive screened encounter in uniform Bz.

    Units are specified by the caller: potential=strength*exp(-r/lambda)/r.
    Symmetric center-of-mass initial data. This is one finite incoming condition,
    without plasma incident-flux sampling or a calibrated encounter frequency.
    """
    r, v = np.asarray(relative_position, float), np.asarray(relative_velocity, float)
    if r.shape != (3,) or v.shape != (3,) or np.linalg.norm(r) == 0 or mass <= 0:
        raise ValueError("Invalid incoming data.")
    if strength < 0 or screening <= 0 or duration <= 0 or max_step <= 0:
        raise ValueError("Use nonnegative repulsive strength and positive scales.")
    initial = np.concatenate((r/2, -r/2, v/2, -v/2))
    bfield = np.array([0., 0., field])
    def potential(radius):
        return strength*np.exp(-radius/screening)/radius
    def rhs(t, y):
        positions, velocities = y[:6].reshape(2, 3), y[6:].reshape(2, 3)
        separation = positions[0]-positions[1]
        radius = np.linalg.norm(separation)
        force = strength*np.exp(-radius/screening)*(1+radius/screening)*separation/radius**3
        acceleration = charge/mass*np.cross(velocities, bfield)
        acceleration += np.array([force, -force])/mass
        return np.concatenate((velocities.ravel(), acceleration.ravel()))
    result = solve_ivp(rhs, (0, duration), initial, method='DOP853', rtol=rtol,
                       atol=rtol*0.01, max_step=max_step)
    if not result.success or not np.all(np.isfinite(result.y)):
        raise RuntimeError("Binary encounter integration failed: "+result.message)
    positions = result.y[:6].T.reshape(-1, 2, 3)
    velocities = result.y[6:].T.reshape(-1, 2, 3)
    separation = positions[:, 0]-positions[:, 1]
    energy = mass/2*np.sum(velocities**2, axis=(1, 2))+potential(np.linalg.norm(separation, axis=1))
    scale = max(abs(energy[0]), np.finfo(float).tiny)
    output = {'time': result.t, 'positions': positions, 'velocities': velocities,
        'energy_error': float(np.max(np.abs(energy-energy[0]))/scale),
        'end_separation': float(np.linalg.norm(separation[-1])),
        'min_separation': float(np.min(np.linalg.norm(separation, axis=1))),
        'evaluations': result.nfev}
    if field != 0:
        moment = mass/(2*abs(field))*np.sum(velocities[..., :2]**2, axis=-1)
        output['delta_mu'] = moment[-1]-moment[0]
        output['mu_initial'] = moment[0]
        omega = charge*field/mass
        gc = positions+np.cross(velocities, np.array([0., 0., 1.]))/omega
        output['delta_gc_perpendicular'] = gc[-1, :, :2]-gc[0, :, :2]
        free_velocity = velocities[0].copy()
        cosine, sine = np.cos(omega*duration), np.sin(omega*duration)
        free_velocity[:, 0] = velocities[0, :, 0]*cosine + velocities[0, :, 1]*sine
        free_velocity[:, 1] = velocities[0, :, 1]*cosine - velocities[0, :, 0]*sine
        output['delta_velocity_from_free'] = velocities[-1]-free_velocity
    return output


def landau_weak_moments(velocity, weights, density, score, gradients, gamma=1., chunk=64):
    """General 3V weak moment quadrature on positive sampled distributions.

    gradients[n,k,:] is grad(phi_k) at node n; score=grad(log F). Ordered
    pair sum carries the 1/2 factor. Diagonal terms use zero contribution;
    convergence is required before treating this nodal rule as a physical rate.
    """
    v, weights, density, score, gradients = map(np.asarray,
        (velocity, weights, density, score, gradients))
    n = len(v)
    if v.shape != (n, 3) or score.shape != (n, 3) or gradients.shape[0] != n or gradients.shape[-1] != 3:
        raise ValueError("Three-dimensional node and gradient arrays required.")
    if weights.shape != (n,) or density.shape != (n,) or np.any(weights <= 0) or np.any(density <= 0) or gamma < 0 or chunk < 1:
        raise ValueError("Positive quadrature/density and nonnegative rate required.")
    moments = np.zeros(gradients.shape[1])
    entropy = 0.
    mass = weights*density
    for start in range(0, n, chunk):
        stop = min(start+chunk, n)
        relative = v[start:stop, None]-v[None]
        delta_score = score[start:stop, None]-score[None]
        impulse = np.einsum('abij,abj->abi', landau_tensor(relative), delta_score)
        measure = mass[start:stop, None]*mass[None]
        delta_gradient = gradients[start:stop, None]-gradients[None]
        moments -= gamma/2*np.einsum('ab,abki,abi->k', measure, delta_gradient, impulse)
        entropy += gamma/2*np.einsum('ab,abi,abi->', measure, delta_score, impulse)
    return moments, float(entropy)


def encounter_incoming(impact, gyrophase, parallel_speed, perpendicular_speed,
                       start_distance, omega):
    """Same incoming free helix at different start planes, GC impact held fixed.

    Relative velocity gyrophase is defined where the unperturbed relative z=0;
    the relative guiding-center position is (impact,0). Finite interaction tails
    are excluded from preparation and checked by receding start/end planes.
    """
    if impact <= 0 or parallel_speed <= 0 or perpendicular_speed < 0 or start_distance <= 0 or omega == 0:
        raise ValueError("Positive scales and nonzero gyrofrequency required.")
    phase = gyrophase + omega*start_distance/parallel_speed
    velocity = np.array([perpendicular_speed*np.cos(phase),
                         perpendicular_speed*np.sin(phase), parallel_speed])
    position = np.array([impact-velocity[1]/omega, velocity[0]/omega, -start_distance])
    return position, velocity


def encounter_flux_quadrature(orders, impact_bounds, parallel_bounds,
                              perpendicular_bounds, temperature, phase_offset=0.):
    """Positive bounded two-sided incoming-plane flux, per unit partner density.

    Here temperature denotes theta=k_B*T/m, the single-particle velocity
    variance, rather than thermodynamic temperature for a general mass.
    Equal Maxwellians have relative velocity covariance 2*temperature*I.
    Coordinates are guiding-center impact b, positive parallel speed, relative
    perpendicular speed, and gyrophase where the incoming free orbit crosses z=0.
    The measure is 2*(2*pi*b db)*v_parallel*F_rel*(v_perp dv_perp dv_parallel dphi).
    Isotropy allows impact orientation to integrate analytically. This counts
    bounded incoming crossings, not a validated independent-encounter plasma rate.
    """
    if len(orders) != 4 or any(int(n) != n or n < 2 for n in orders) or not np.isfinite(temperature) or temperature <= 0 or not np.isfinite(phase_offset):
        raise ValueError("Four quadrature orders >=2 and positive temperature required.")
    bounds = [impact_bounds, parallel_bounds, perpendicular_bounds]
    if any(len(x) != 2 or not np.all(np.isfinite(x)) or not 0 <= x[0] < x[1] for x in bounds) or impact_bounds[0] <= 0 or parallel_bounds[0] <= 0:
        raise ValueError("Positive impact/parallel lower bounds and ordered speed bounds required.")
    axes=[]; weights=[]
    for n, (lower,upper) in zip(orders[:3],bounds):
        x,w=np.polynomial.legendre.leggauss(n)
        axes.append(lower+(upper-lower)*(x+1)/2)
        weights.append(w*(upper-lower)/2)
    phases=(2*np.pi*np.arange(orders[3])/orders[3]+phase_offset) % (2*np.pi)
    axes.append(phases);weights.append(np.full(orders[3],2*np.pi/orders[3]))
    mesh=np.meshgrid(*axes,indexing='ij')
    nodes=np.stack(mesh,axis=-1).reshape(-1,4)
    volume=np.prod(np.meshgrid(*weights,indexing='ij'),axis=0).ravel()
    b,parallel,perpendicular,_=nodes.T
    relative_density=np.exp(-(parallel**2+perpendicular**2)/(4*temperature))/(4*np.pi*temperature)**1.5
    flux_weights=2*(2*np.pi*b)*parallel*relative_density*perpendicular*volume
    return nodes,flux_weights


def encounter_bounded_flux(impact_bounds, parallel_bounds, perpendicular_bounds, temperature):
    """Analytic bounded incoming flux per density; temperature is k_B*T/m."""
    bounds=(impact_bounds,parallel_bounds,perpendicular_bounds)
    if not np.isfinite(temperature) or temperature<=0 or any(len(x)!=2 or not np.all(np.isfinite(x)) or not 0<=x[0]<x[1] for x in bounds):
        raise ValueError("Finite ordered nonnegative bounds and positive velocity variance required.")
    area=np.pi*(impact_bounds[1]**2-impact_bounds[0]**2)
    def radial_flux(bounds):
        return 2*temperature*(np.exp(-bounds[0]**2/(4*temperature))-np.exp(-bounds[1]**2/(4*temperature)))
    return 2*area*2*np.pi*radial_flux(parallel_bounds)*radial_flux(perpendicular_bounds)/(4*np.pi*temperature)**1.5


def encounter_relative(impact, phase, parallel_speed, perpendicular_speed, *,
                       field, strength, screening, start_distance, mass=1.,
                       charge=1., max_step=.3, rtol=1e-10, flight_time_factor=6.):
    """Equal-particle 6D relative Lorentz-force solve to the first outgoing plane.

    Initial data share a free incoming helix. Terminal z=+L crossing is transmitted;
    z=-L with negative velocity is reflected. Initial incoming -L is not an exit.
    No exit by flight_time_factor*L/v_parallel rejects the encounter (default6).
    A finite budget failure does not imply trapping. Repulsive parallel acceleration
    has sign(z), hence at most one parallel turn. Event planes have finite tails.
    """
    if not np.all(np.isfinite([impact,phase,parallel_speed,perpendicular_speed,field,strength,start_distance,mass,charge,max_step,rtol,flight_time_factor])) or field <= 0 or strength < 0 or not screening > 0 or mass <= 0 or charge == 0 or max_step <= 0 or rtol <= 0 or flight_time_factor <= 0:
        raise ValueError("Invalid encounter physical/integration parameters.")
    omega=charge*field/mass
    position,velocity=encounter_incoming(impact,phase,parallel_speed,perpendicular_speed,start_distance,omega)
    initial=np.concatenate((position,velocity))
    def rhs(t,y):
        r=y[:3];w=y[3:]
        radius=np.sqrt(np.dot(r,r))
        factor=2*strength/mass*np.exp(-radius/screening)*(1+radius/screening)/radius**3
        return np.array([w[0],w[1],w[2],omega*w[1]+factor*r[0],-omega*w[0]+factor*r[1],factor*r[2]])
    def transmitted(t,y):
        return y[2]-start_distance
    transmitted.terminal=True;transmitted.direction=1
    def reflected(t,y):
        return y[2]+start_distance
    reflected.terminal=True;reflected.direction=-1
    result=solve_ivp(rhs,(0.,flight_time_factor*start_distance/parallel_speed),initial,method='DOP853',rtol=rtol,atol=rtol*.01,max_step=max_step,events=[transmitted,reflected])
    if not result.success or result.status != 1 or not np.all(np.isfinite(result.y)):
        error=RuntimeError('Encounter failed to reach an outgoing plane: '+result.message)
        error.integration_diagnostics={'solver_success':bool(result.success),'solver_status':int(result.status),
            'flight_time_factor':float(flight_time_factor),'budget':float(flight_time_factor*start_distance/parallel_speed),
            'final_time':float(result.t[-1]),'final_position':result.y[:3,-1].tolist(),
            'final_velocity':result.y[3:,-1].tolist(),'step_count':len(result.t)-1,'evaluations':result.nfev}
        raise error
    radius=np.linalg.norm(result.y[:3],axis=0)
    energy=mass/4*np.sum(result.y[3:]**2,axis=0)+strength*np.exp(-radius/screening)/radius
    final=result.y[3:,-1]
    angle=omega*result.t[-1]
    # Undo the common free gyromotion: R(-Omega*t) maps outgoing velocity to incoming time.
    unrotated=np.array([final[0]*np.cos(angle)-final[1]*np.sin(angle),final[0]*np.sin(angle)+final[1]*np.cos(angle),final[2]])
    delta_perpendicular=unrotated[:2]-velocity[:2]
    conditional_mu_mean=mass/(8*field)*(np.dot(final[:2],final[:2])-perpendicular_speed**2)
    return {'conditional_mu_mean':float(conditional_mu_mean),'delta_relative_perpendicular':delta_perpendicular,
        'initial_relative_velocity':velocity,'final_relative_velocity':final,
        'initial_relative_position':position,'final_relative_position':result.y[:3,-1],
        'step_count':len(result.t)-1,'min_step':float(np.diff(result.t).min()),
        'max_actual_step':float(np.diff(result.t).max()),'time':float(result.t[-1]),
        'flight_time_factor':float(flight_time_factor),'flight_budget':float(flight_time_factor*start_distance/parallel_speed),
        'energy_error':float(np.max(np.abs(energy-energy[0]))/abs(energy[0])),
        'end_force':float(strength*np.exp(-radius[-1]/screening)*(1+radius[-1]/screening)/radius[-1]**2),
        'min_separation':float(radius.min()),'exit':'transmitted' if len(result.t_events[0]) else 'reflected',
        'parallel_turns':int(np.count_nonzero(np.diff(np.sign(result.y[5])))),
        'evaluations':result.nfev}


def encounter_thermal_moments(result, temperature, mass=1., field=1.):
    """Analytically average individual moment increments over independent COM.

    temperature means theta=k_B*T/m, the single-particle velocity variance.
    V_center~N(0,temperature/2 I); it follows its unperturbed common gyromotion.
    Delta_mu_1=c+m/(2B)*V_center_perp.dot(delta_w_perp), Delta_mu_2=c-minus.
    Center averaging is essential: symmetric zero-center trajectories alone omit
    this positive contribution to the individual second moment.
    """
    if not np.all(np.isfinite([temperature,mass,field])) or temperature<=0 or mass<=0 or field<=0:
        raise ValueError("Finite positive thermal/mass/field inputs required.")
    mean=result['conditional_mu_mean']
    variance=(mass/(2*field))**2*temperature/2*np.dot(result['delta_relative_perpendicular'],result['delta_relative_perpendicular'])
    return np.array([mean,mean**2+variance,mean**2-variance])


def encounter_flux_samples(unit_cube, impact_bounds, parallel_bounds,
                           perpendicular_bounds, temperature):
    """Inverse CDF of the normalized bounded incoming flux, excluding gyrophase.

    Three unit-cube coordinates sample b^2 uniformly and v_parallel^2,v_perp^2
    with truncated exponentials. Angular quadrature remains explicit. The exact
    analytic bounded flux supplies the normalization, rather than a fitted rate.
    """
    unit_cube=np.asarray(unit_cube,float)
    encounter_bounded_flux(impact_bounds,parallel_bounds,perpendicular_bounds,temperature)
    if unit_cube.ndim!=2 or unit_cube.shape[1]!=3 or not np.all(np.isfinite(unit_cube)) or np.any((unit_cube<0)|(unit_cube>1)):
        raise ValueError("Need finite Nx3 points in the unit cube.")
    impact=np.sqrt(impact_bounds[0]**2+(impact_bounds[1]**2-impact_bounds[0]**2)*unit_cube[:,0])
    speeds=[]
    for axis,bounds in enumerate([parallel_bounds,perpendicular_bounds],start=1):
        lower=np.exp(-bounds[0]**2/(4*temperature))
        upper=np.exp(-bounds[1]**2/(4*temperature))
        speeds.append(np.sqrt(-4*temperature*np.log(lower-unit_cube[:,axis]*(lower-upper))))
    return np.column_stack((impact,*speeds))


def encounter_moment_interpolator(axes, moments):
    """Cubic space and truly periodic cubic phase, preserving COM constraints.

    Interpolate signed c and log(var), where second=c^2+var and cross=c^2-var.
    Spatial axes need >=4 nodes. A periodic CubicSpline closes the phase values
    and first/second derivatives exactly. Predictions remain conditional table
    moments; independent held-out scattering states must verify their accuracy.
    """
    from scipy.interpolate import RegularGridInterpolator, CubicSpline
    axes=[np.asarray(a,float) for a in axes]
    moments=np.asarray(moments,float)
    if len(axes)!=4 or any(a.ndim!=1 or len(a)<4 or not np.all(np.diff(a)>0) for a in axes) or moments.shape!=tuple(map(len,axes))+(3,):
        raise ValueError("Need four increasing axes with >=4 nodes and matching moment table.")
    phase=axes[3]
    spacing=2*np.pi/len(phase)
    if not np.allclose(phase,spacing*np.arange(len(phase)),atol=1e-13,rtol=0):
        raise ValueError("Gyrophase axis must uniformly cover [0,2pi).")
    variance=(moments[...,1]-moments[...,2])/2
    if not np.all(np.isfinite(moments)) or np.any(variance<=0):
        raise ValueError("Strictly positive resolved COM variance required for logarithmic interpolation.")
    fields=np.stack((moments[...,0],np.log(variance)),axis=-1)
    spatial=RegularGridInterpolator(axes[:3],fields,method='cubic',bounds_error=True,
        solver_args={'rtol':1e-12,'atol':1e-14})
    closed_phase=np.concatenate((phase,[2*np.pi]))
    def evaluate(nodes):
        nodes=np.asarray(nodes,float)
        if nodes.shape[-1]!=4 or not np.all(np.isfinite(nodes)):
            raise ValueError("Finite four-coordinate query nodes required.")
        original_shape=nodes.shape[:-1]
        flat=nodes.reshape(-1,4)
        angles=flat[:,3]%(2*np.pi)
        raw=spatial(flat[:,:3])
        closed_values=np.concatenate((raw,raw[:,:1]),axis=1)
        spline=CubicSpline(closed_phase,closed_values,axis=1,bc_type='periodic')
        intervals=np.minimum((angles/spacing).astype(int),len(phase)-1)
        delta=angles-phase[intervals]
        coefficients=spline.c[:,intervals,np.arange(len(flat))]
        interpolated=((coefficients[0]*delta[:,None]+coefficients[1])*delta[:,None]+coefficients[2])*delta[:,None]+coefficients[3]
        mean=interpolated[:,0];variance=np.exp(interpolated[:,1])
        answer=np.stack((mean,mean**2+variance,mean**2-variance),axis=-1)
        return answer.reshape(original_shape+(3,))
    return evaluate


def encounter_reverse_incoming(result, *, field, mass=1., charge=1., start_distance):
    """Parity plus time reversal prepares the inverse encounter at reversed qB.

    For transmitted encounters, reverse (r,w)->(-r_out,w_out), charge->-charge.
    Rotate its guiding-center impact to +x and recover the zero-plane gyrophase.
    The reversed solve should return the original incoming velocities up to that
    rotation, and negate c. This tests microscopic reversibility, not cutoff
    detailed balance: the outgoing inverse data can leave the incoming bands.
    """
    if result['exit']!='transmitted':
        raise ValueError("This preparation requires a transmitted encounter.")
    omega=-charge*field/mass
    position=-np.asarray(result['final_relative_position'])
    velocity=np.asarray(result['final_relative_velocity'])
    gc=position+np.cross(velocity,[0.,0.,1.])/omega
    angle=np.arctan2(gc[1],gc[0])
    cosine,sine=np.cos(angle),np.sin(angle)
    rotated=np.array([cosine*velocity[0]+sine*velocity[1],-sine*velocity[0]+cosine*velocity[1],velocity[2]])
    impact=np.linalg.norm(gc[:2])
    parallel=rotated[2];perpendicular=np.linalg.norm(rotated[:2])
    phase=(np.arctan2(rotated[1],rotated[0])-omega*start_distance/parallel)%(2*np.pi)
    return np.array([impact,parallel,perpendicular,phase]),float(angle)


@dataclass(frozen=True,eq=False)
class LandauVelocityGrid:
    shape:tuple
    velocity:object
    weights:object
    derivative:object
    def __post_init__(self):
        import jax
        import jax.numpy as jnp
        if not jax.config.jax_enable_x64:
            raise ValueError('Landau evolution requires JAX x64; enable jax_enable_x64 before constructing its grid')
        if (type(self.shape) is not tuple or len(self.shape)!=3
            or any(type(n) is not int or n<3 for n in self.shape)
            or len(set(self.shape))!=1):
            raise ValueError('Landau grid needs three equal integer axes of length>=3')
        velocity,weights,derivative=map(np.asarray,(self.velocity,self.weights,self.derivative))
        n=self.shape[0];size=int(np.prod(self.shape))
        if (velocity.shape!=(size,3) or weights.shape!=(size,) or derivative.shape!=(n,n)
            or not all(np.all(np.isfinite(a)) for a in (velocity,weights,derivative))
            or np.any(weights<=0)):
            raise ValueError('Finite physical 3V nodes/derivatives and positive quadrature weights required')
        axes=[np.unique(velocity[:,d]) for d in range(3)]
        if any(len(a)!=n for a in axes) or not all(np.array_equal(a,axes[0]) for a in axes):
            raise ValueError('Three equal Cartesian velocity axes required')
        expected=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
        if not np.array_equal(velocity,expected):raise ValueError('Velocity tensor ordering must be ij')
        x=axes[0]
        for h,dh in [(np.ones(n),np.zeros(n)),(x,np.ones(n)),(x*x,2*x)]:
            if not np.allclose(derivative@h,dh,rtol=1e-11,atol=1e-11):
                raise ValueError('Velocity derivative must reproduce quadratics without repairs')
        for name,a in [('velocity',velocity),('weights',weights),('derivative',derivative)]:
            object.__setattr__(self,name,jnp.array(a.copy()))
    @property
    def size(self):return int(np.prod(self.shape))

def landau_velocity_grid(order=12,extent=5.):
    """Positive 3V Gauss cube with quadratic-exact global polynomial gradients.

    Density is represented at every node, with natural no-flux weak collision
    boundary. Positivity concerns nodal/quadrature densities, not polynomial
    interpolation between nodes. Grid and tail convergence must be demonstrated.
    """
    from .collisions import derivative_matrix
    if type(order) is not int or order<3 or not np.isfinite(extent) or extent<=0:
        raise ValueError('Need integer order>=3 and finite positive velocity extent')
    x,w=np.polynomial.legendre.leggauss(order);x=extent*x;w=extent*w
    v=np.stack(np.meshgrid(x,x,x,indexing='ij'),axis=-1).reshape(-1,3)
    weights=np.prod(np.meshgrid(w,w,w,indexing='ij'),axis=0).ravel()
    return LandauVelocityGrid((order,)*3,v,weights,derivative_matrix(x,method='polynomial'))

@dataclass(frozen=True,eq=False)
class LandauEntropyCompiler(LaggedEntropyCompiler):
    """Physical Landau kernels bound to one grid, Gamma, softening and budgets."""
    softening:float=0.


def landau_entropy_compiler(grid,*,gamma=1.,softening=0.,chunk_size=128,
                            linear_max_steps=1000,reference_population=None):
    """Matrix-free Coulomb weak Gram, convex residual and SPD PCG correction.

    h^T K(f)k=Gamma sum(i<j) w_i w_j f_i f_j Delta grad h^T U Delta grad k.
    U=(I-rr^T/|r|^2)/|r| in physical 3V. Coincident samples contribute zero;
    the smooth weak integrand tends to zero, but finite quadrature error remains.
    Optional softening changes the denominator to sqrt(r^2+epsilon^2), not the
    energy projector. It is a different finite-epsilon kernel; convergence to
    zero must be checked. No Maxwellian/Gaussian closure is imposed.

    Polynomial reproduction gives exact discrete constant/momentum/energy null
    directions. All pair cross terms remain in K. The diagonal preconditioner
    drops correlations only as an auxiliary SPD approximation. This compiler
    reuses the distribution-independent convex entropy-variable host solver.
    """
    import jax
    import jax.numpy as jnp
    from solvax import pcg
    from .solver import _population_difference,_expm1_minus_x
    if not isinstance(grid,LandauVelocityGrid) or not np.isfinite(gamma) or gamma<0 or not np.isfinite(softening) or softening<0:
        raise ValueError('Physical 3V grid, finite nonnegative gamma/softening required')
    if type(chunk_size) is not int or chunk_size<1 or type(linear_max_steps) is not int or linear_max_steps<0:
        raise ValueError('Positive integer chunk and nonnegative linear budget required')
    N=grid.size;chunks=(N+chunk_size-1)//chunk_size;pad=chunks*chunk_size-N
    v=jnp.pad(grid.velocity,((0,pad),(0,0)));D=grid.derivative
    if reference_population is not None:
        a=np.asarray(reference_population)
        if a.shape!=(N,) or not np.all(np.isfinite(a)) or np.any(a<=0) or not np.isfinite(a.sum()):raise ValueError('Invalid fixed reference populations')
        reference_population=jnp.array(a.copy())
    def gradient(h):
        h=h.reshape(grid.shape)
        return jnp.stack([jnp.moveaxis(jnp.tensordot(D,h,axes=(1,d)),0,d).ravel() for d in range(3)],axis=-1)
    def adjoint(g):
        return sum(jnp.moveaxis(jnp.tensordot(D.T,g[:,d].reshape(grid.shape),axes=(1,d)),0,d).ravel() for d in range(3))
    def geometry(start):
        target=jax.lax.dynamic_slice(v,(start,0),(chunk_size,3))
        w=target[:,None,:]-v[None,:,:];r2=jnp.sum(w*w,axis=-1)
        safe=jnp.where(r2>0,r2,1.)
        inv=jnp.where(r2>0,1/jnp.sqrt(safe+softening**2),0.)
        return w,safe,inv
    def apply(old_log,h):
        n=jnp.pad(grid.weights*jnp.exp(old_log),(0,pad));g=jnp.pad(gradient(h),((0,pad),(0,0)))
        def body(i,out):
            start=i*chunk_size;w,r2,inv=geometry(start)
            target_g=jax.lax.dynamic_slice(g,(start,0),(chunk_size,3))
            delta=target_g[:,None,:]-g[None,:,:]
            projected=(delta-w*jnp.sum(w*delta,axis=-1)[...,None]/r2[...,None])*inv[...,None]
            ni=jax.lax.dynamic_slice(n,(start,),(chunk_size,))
            flux=gamma*ni[:,None]*jnp.sum(n[None,:,None]*projected,axis=1)
            return jax.lax.dynamic_update_slice(out,flux,(start,0))
        return adjoint(jax.lax.fori_loop(0,chunks,body,jnp.zeros((N+pad,3)))[:N])
    def prepare(old_log):
        n=jnp.pad(grid.weights*jnp.exp(old_log),(0,pad))
        def body(i,out):
            start=i*chunk_size;w,r2,inv=geometry(start)
            diag=(1-w*w/r2[...,None])*inv[...,None]
            ni=jax.lax.dynamic_slice(n,(start,),(chunk_size,))
            q=gamma*ni[:,None]*jnp.sum(n[None,:,None]*diag,axis=1)
            return jax.lax.dynamic_update_slice(out,q,(start,0))
        q=jax.lax.fori_loop(0,chunks,body,jnp.zeros((N+pad,3)))[:N]
        return sum(jnp.moveaxis(jnp.tensordot((D*D).T,q[:,d].reshape(grid.shape),axes=(1,d)),0,d).ravel() for d in range(3))
    def evaluate(new_log,old_log,dt):return _population_difference(new_log,old_log,grid.weights)+dt*apply(old_log,new_log)
    def correction(new_log,old_log,dt,diagonal,linear_rtol):
        oldmass=grid.weights*jnp.exp(old_log)
        metric=oldmass if reference_population is None else reference_population
        root=jnp.sqrt(metric);ratio=grid.weights*jnp.exp(new_log)/metric
        value=evaluate(new_log,old_log,dt)/root
        def operator(x):return ratio*x+dt*apply(old_log,x/root)/root
        pre=ratio+dt*diagonal/metric
        answer=pcg(operator,-value,precond=lambda x:x/pre,rtol=linear_rtol,atol=0.,max_steps=linear_max_steps)
        d=answer.x/root
        true=jnp.linalg.norm(operator(answer.x)+value)/jnp.maximum(jnp.linalg.norm(value),1e-300)
        return d,answer.iterations,true,answer.converged,jnp.vdot(value,answer.x).real,jnp.vdot(d,apply(old_log,d)).real,answer.status
    def objective(new_log,old_log,dt,direction,fraction,descent,curvature):
        return fraction*descent+jnp.sum(grid.weights*jnp.exp(new_log)*_expm1_minus_x(fraction*direction))+.5*dt*fraction**2*curvature
    compiled=LandauEntropyCompiler(jax.jit(evaluate),jax.jit(correction,static_argnums=(4,)),jax.jit(objective),jax.jit(prepare),jax.jit(apply),grid,float(gamma),chunk_size,linear_max_steps,None,reference_population,float(softening))
    return compiled

def landau_entropy_step(grid,f,dt,*,gamma=1.,softening=0.,compiled=None,chunk_size=128,linear_max_steps=1000,**kwargs):
    """Advance the full 3V density using first-order lagged-mobility backward Euler.

    Number, three momenta and energy are conserved by the unrepaired weak Gram.
    Entropy increase splits into frozen Landau-Gram dissipation and positive
    time-discretization KL. The reused host checks strict nodal positivity, true
    PCG residuals, objective descent and entropy/finite budgets; failures raise
    StepFailure and return no accepted step. Kernel/grid/tail/time convergence
    and physical normalization are separate from these discrete guarantees.
    """
    from .solver import lagged_entropy_step
    if compiled is not None and (not isinstance(compiled,LandauEntropyCompiler)
        or compiled.softening!=float(softening)):
        raise ValueError('Landau compiler kernel configuration mismatch')
    if compiled is None:compiled=landau_entropy_compiler(grid,gamma=gamma,softening=softening,chunk_size=chunk_size,linear_max_steps=linear_max_steps)
    return lagged_entropy_step(grid,f,dt,collision_strength=gamma,compiled=compiled,chunk_size=chunk_size,linear_max_steps=linear_max_steps,**kwargs)
