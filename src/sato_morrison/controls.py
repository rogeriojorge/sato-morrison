"""Independent homogeneous controls and full three-velocity Coulomb moments.

The Landau routines evaluate weak instantaneous moments, not a kinetic time
integrator. Gaussian relative velocities retain all three physical dimensions.
The Dougherty solution is exact for any positive finite Gaussian mixture.
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.special import logsumexp


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
