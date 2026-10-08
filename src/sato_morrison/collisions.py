"""Positive-quadrature weak local and finite-range constrained collisions.

All vectors used by the kernel are in the fixed, normalized Cartesian eta
chart. The energy direction is computed with exactly the discrete derivative
used for test functions, making the discrete energy degeneracy explicit.
"""
from dataclasses import dataclass
from numbers import Real
import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class WeakGrid:
    shape: tuple
    derivatives: tuple  # (tensor axis, derivative matrix)
    coefficients: object  # shape + (5, number of derivatives)
    weights: object
    energy: object
    mu_index: object
    left: object
    right: object
    pair_weights: object
    kernel_directions: object
    spatial_shape: tuple
    velocity_shape: tuple
    description: str
    local_quadrature: object = None
    energy_flow: object = None
    uniform_direction: object = None

    @property
    def kernels(self):
        if self.kernel_directions is None:
            raise ValueError('compact local grids do not store a full pair tensor')
        direction = self.kernel_directions
        projector = jnp.eye(5) - direction[..., :, None]*direction[..., None, :]
        return projector @ jnp.diag(jnp.array([1.,1.,1.,0.,0.])) @ projector

    @property
    def pair_count(self):
        if self.local_quadrature is None:
            return len(self.left)
        nx, nv = self.local_quadrature[1].shape
        return nx*nv*(nv-1)//2

    @property
    def size(self):
        return int(np.prod(self.shape))

    def action(self, h):
        """Common eta-chart J grad(h), with tensor differentiation."""
        h = jnp.asarray(h).reshape(self.shape)
        terms = []
        for axis, derivative in self.derivatives:
            value = jnp.tensordot(derivative, h, axes=(1, axis))
            terms.append(jnp.moveaxis(value, 0, axis))
        gradient = jnp.stack(terms, axis=-1)
        return jnp.einsum('...ad,...d->...a', self.coefficients, gradient).reshape(-1, 5)

    def transpose_action(self, covector):
        local = jnp.einsum('...ad,...a->...d', self.coefficients,
                           covector.reshape(self.shape + (5,)))
        answer = jnp.zeros(self.shape, dtype=local.dtype)
        for component, (axis, derivative) in enumerate(self.derivatives):
            value = jnp.tensordot(derivative.T, local[..., component], axes=(1, axis))
            answer = answer + jnp.moveaxis(value, 0, axis)
        return answer.reshape(-1)


def derivative_matrix(nodes, *, period=None,method='local_quadratic'):
    """Spectral periodic derivative, or local quadratic nonperiodic derivative.

    Nonperiodic matrices enter a weak form with natural zero collision flux;
    no value is wrapped across a nonperiodic boundary.
    """
    nodes = np.asarray(nodes, dtype=float)
    n = len(nodes)
    if n < 3 or not np.all(np.isfinite(nodes)) or np.any(np.diff(nodes) <= 0):
        raise ValueError('derivative nodes must increase and contain at least three points')
    if method not in ('local_quadratic','polynomial'):
        raise ValueError(f'unsupported derivative method: {method}')
    if method=='polynomial':
        if period is not None:
            raise ValueError('global polynomial derivative requires nonperiodic nodes')
        # Lagrange collocation derivative. The weak Gram assembly needs no
        # summation-by-parts identity; positive quadrature and a matching energy
        # derivative establish the discrete conservation/entropy identities.
        difference=nodes[:,None]-nodes[None,:]
        np.fill_diagonal(difference,1.)
        barycentric=1/np.prod(difference,axis=1)
        result=barycentric[None,:]/barycentric[:,None]/difference
        np.fill_diagonal(result,0.)
        np.fill_diagonal(result,-result.sum(axis=1))
        return result
    if period is not None:
        if not np.isfinite(period) or period <= 0:
            raise ValueError('finite positive period required')
        if not np.allclose(np.diff(nodes), period / n):
            raise ValueError('periodic derivative requires uniformly spaced nodes')
        modes = 2 * np.pi * np.fft.fftfreq(n, d=period / n)
        if n % 2 == 0:
            modes[n // 2] = 0  # real skew derivative of unresolved Nyquist mode
        return np.fft.ifft(1j * modes[:, None] * np.fft.fft(np.eye(n), axis=0), axis=0).real
    answer = np.zeros((n, n))
    for i in range(n):
        indices = np.arange(min(max(i - 1, 0), n - 3), min(max(i - 1, 0), n - 3) + 3)
        offsets = nodes[indices] - nodes[i]
        answer[i, indices] = np.linalg.solve(np.stack([np.ones(3), offsets, offsets**2]),
                                            np.array([0., 1., 0.]))
    return answer


def trapezoid_weights(nodes):
    nodes = np.asarray(nodes)
    if len(nodes)<2 or not np.all(np.isfinite(nodes)) or np.any(np.diff(nodes)<=0):
        raise ValueError('quadrature nodes must be finite and strictly increasing')
    return np.r_[np.diff(nodes)[0] / 2, (nodes[2:] - nodes[:-2]) / 2,
                 np.diff(nodes)[-1] / 2]


def _kernel(xi, *, uniform_direction=None):
    norm = np.sum(xi**2, axis=-1)
    safe_norm = np.where(norm > 0, norm, 1.)
    projector = np.eye(5) - xi[..., :, None] * xi[..., None, :] / safe_norm[..., None, None]
    degenerate = norm == 0
    if np.any(degenerate):
        if uniform_direction is None:
            raise ValueError('zero pair energy direction: specify and converge a physical limit')
        direction = np.r_[uniform_direction, 0., 0.]
        limit = np.eye(5) - np.outer(direction, direction)
        projector[degenerate] = limit
    ix = np.diag([1., 1., 1., 0., 0.])
    return projector @ ix @ projector


def _complete_grid(shape, derivatives, coefficients, spatial_weights, velocity_weights,
                   energy, description, *, uniform_direction=None, model='sm181_local',
                   spatial_kernel=None, compact=False):
    if model not in ('sm181_local', 'sm_local_nonlinear', 'sm_finite_range'):
        raise ValueError(f'unsupported constrained collision model: {model}')
    if compact and model == 'sm_finite_range':
        raise ValueError('compact contraction is defined only for spatially local kernels')
    nx, nv = len(spatial_weights), int(np.prod(shape) // len(spatial_weights))
    vw = np.broadcast_to(velocity_weights, (nx, nv))
    weights = (np.asarray(spatial_weights)[:, None] * vw).reshape(-1)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError('quadrature weights must be positive')
    dummy = WeakGrid(shape, tuple((a, jnp.asarray(d)) for a, d in derivatives),
                     jnp.asarray(coefficients), jnp.asarray(weights), jnp.asarray(energy).reshape(-1),
                     jnp.tile(jnp.arange(shape[-1]), int(np.prod(shape[:-1]))),
                     None, None, None, None, shape[:-2], shape[-2:], description)
    flow = np.asarray(dummy.action(dummy.energy))
    if not np.all(np.isfinite(flow)):
        raise ValueError('discrete energy flow must be finite')
    if compact:
        local_flow = flow.reshape(nx,nv,5)
        if uniform_direction is None:
            # Check distinct pairs in bounded host blocks. Diagonal self-pairs
            # have zero test-function difference and need no kernel limit.
            for x in range(nx):
                for start in range(0,nv,32):
                    stop=min(start+32,nv)
                    difference=local_flow[x,start:stop,None]-local_flow[x,None]
                    squared=np.sum(difference**2,axis=-1)
                    diagonal=np.arange(start,stop)[:,None]==np.arange(nv)[None]
                    if np.any((squared==0)&~diagonal):
                        raise ValueError('zero pair energy direction: specify and converge a physical limit')
        return WeakGrid(**{**dummy.__dict__,
            'local_quadrature':(jnp.asarray(spatial_weights),jnp.asarray(vw)),
            'energy_flow':jnp.asarray(local_flow),
            'uniform_direction':None if uniform_direction is None else jnp.asarray(np.r_[uniform_direction,0.,0.])})
    left, right, pair_weight = [], [], []
    if model == 'sm_finite_range':
        spatial_kernel = np.asarray(spatial_kernel)
        if spatial_kernel.shape != (nx, nx) or not np.allclose(spatial_kernel, spatial_kernel.T):
            raise ValueError('finite-range kernel must be an explicit symmetric spatial matrix')
        if np.any(spatial_kernel < 0):
            raise ValueError('spatial kernel must be nonnegative')
        for i in range(nx * nv):
            for j in range(i + 1, nx * nv):
                a, b = i // nv, j // nv
                value = weights[i] * weights[j] * spatial_kernel[a, b]
                if value > 0:
                    left.append(i); right.append(j); pair_weight.append(value)
    else:
        for x in range(nx):
            for i in range(nv):
                for j in range(i + 1, nv):
                    left.append(x * nv + i); right.append(x * nv + j)
                    pair_weight.append(spatial_weights[x] * vw[x, i] * vw[x, j])
    left, right = np.asarray(left, dtype=int), np.asarray(right, dtype=int)
    xi = flow[left] - flow[right]
    if uniform_direction is not None:
        # Analytic uniform limit includes equal parallel speeds; tiny spatial
        # derivative roundoff must not choose a false direction at xi=0.
        direction = np.broadcast_to(np.r_[uniform_direction, 0., 0.], xi.shape).copy()
    else:
        norms = np.linalg.norm(xi, axis=-1)
        if np.any(norms == 0):
            raise ValueError('zero pair energy direction: specify and converge a physical limit')
        direction = xi / norms[:,None]
    return WeakGrid(**{**dummy.__dict__, 'left':jnp.asarray(left),'right':jnp.asarray(right),
                       'pair_weights':jnp.asarray(pair_weight),'kernel_directions':jnp.asarray(direction)})


def _velocity_quadrature(u,mu,weights):
    if weights is None:
        return np.outer(trapezoid_weights(u),trapezoid_weights(mu)).ravel()
    wu,wm=map(np.asarray,weights)
    if wu.shape!=(len(u),) or wm.shape!=(len(mu),) or not np.all(np.isfinite(wu)) or not np.all(np.isfinite(wm)) or np.any(wu<=0) or np.any(wm<=0):
        raise ValueError('positive finite velocity quadrature weights must match nodes')
    return np.outer(wu,wm).ravel()


def uniform_grid(x, u, mu, *, magnetic_field=1., mass=1., charge=1.,
                 period=2*np.pi, model='sm181_local', spatial_kernel=None,
                 velocity_weights=None, compact=False):
    """One perpendicular periodic coordinate with the full five-vector kernel.

    Domain measure is B dx du dmu per unit omitted spatial area. The common
    eta chart retains u and perpendicular kinetic energy even though no mu
    derivative enters the Poisson action.
    """
    if not np.all(np.isfinite([magnetic_field,mass,charge])) or magnetic_field <= 0 or mass <= 0 or charge == 0:
        raise ValueError('positive field and mass and nonzero charge required')
    x, u, mu = map(np.asarray, (x, u, mu))
    if np.any(mu < 0):
        raise ValueError('magnetic moment must be nonnegative')
    shape = len(x), len(u), len(mu)
    coefficient = np.zeros(shape + (5, 2))
    coefficient[..., 1, 0] = 1 / (charge * magnetic_field)
    coefficient[..., 2, 1] = 1 / mass
    energy = np.broadcast_to(mass*u[None, :, None]**2/2 + magnetic_field*mu[None, None, :], shape)
    velocity_weight = magnetic_field * _velocity_quadrature(u,mu,velocity_weights)
    return _complete_grid(shape, ((0, derivative_matrix(x, period=period)),
                                  (1, derivative_matrix(u))), coefficient,
                          np.full(len(x), period/len(x)), velocity_weight, energy,
                          'uniform perpendicular periodic slab; natural velocity collision flux',
                          uniform_direction=np.array([0., 0., 1.]), model=model,
                          spatial_kernel=spatial_kernel,compact=compact)


def toroidal_grid(radius, theta, z, u, mu, *, strength=1., mass=1., charge=1.,
                  theta_period=2*np.pi, z_period=2*np.pi):
    """Vacuum toroidal annulus: periodic theta,z, radial collision no-flux.

    Hamiltonian flow is tangent at both radius boundaries and has no u or mu
    flux. Local collisions are evaluated in Cartesian eta vectors, never by
    recreating a Euclidean projector in cylindrical coordinates.
    """
    from .geometry import Field, common_chart_action
    if not np.all(np.isfinite([strength,mass,charge])) or strength<=0 or mass<=0 or charge==0:
        raise ValueError('positive finite field and mass and nonzero charge required')
    radius, theta, z, u, mu = map(np.asarray, (radius, theta, z, u, mu))
    if np.any(radius <= 0) or np.any(mu < 0):
        raise ValueError('toroidal radius must be positive and mu nonnegative')
    shape = tuple(map(len, (radius, theta, z, u, mu)))
    field = Field('toroidal', strength=strength)
    rr, tt, zz, uu, mm = np.meshgrid(radius, theta, z, u, mu, indexing='ij')
    nodes = np.stack((rr*np.cos(tt), rr*np.sin(tt), zz, uu, mm), axis=-1).reshape(-1, 5)
    covectors = np.zeros((len(nodes), 4, 5))
    covectors[:, 0, 0] = np.cos(tt).ravel(); covectors[:, 0, 1] = np.sin(tt).ravel()
    covectors[:, 1, 0] = (-np.sin(tt)/rr).ravel(); covectors[:, 1, 1] = (np.cos(tt)/rr).ravel()
    covectors[:, 2, 2] = 1.; covectors[:, 3, 3] = 1.
    action = jax.vmap(lambda point, gradients: jax.vmap(
        lambda gradient: common_chart_action(point, gradient, field, mass=mass, charge=charge))(gradients))
    coefficient = np.asarray(action(jnp.asarray(nodes), jnp.asarray(covectors))).swapaxes(1,2).reshape(shape+(5,4))
    dr = trapezoid_weights(radius)
    spatial_weights = np.broadcast_to((dr*radius)[:,None,None] *
                         theta_period/len(theta) * z_period/len(z), shape[:3]).ravel()
    velocity_weights = np.outer(trapezoid_weights(u), trapezoid_weights(mu)).reshape(-1)
    velocity_weights = np.broadcast_to((strength/rr[...,0,0]).ravel()[:,None] * velocity_weights,
                                       (np.prod(shape[:3]), np.prod(shape[3:])))
    energy = mass*uu**2/2 + mm*strength/rr
    return _complete_grid(shape, ((0, derivative_matrix(radius)),
        (1, derivative_matrix(theta, period=theta_period)),
        (2, derivative_matrix(z, period=z_period)), (3, derivative_matrix(u))),
        coefficient, spatial_weights, velocity_weights, energy,
        'vacuum toroidal annulus; periodic theta,z; natural radial/u collision flux')


def _project_pair(difference,directions):
    first=difference-directions*jnp.sum(directions*difference,axis=-1)[...,None]
    spatial=first.at[...,3:].set(0.)
    return spatial-directions*jnp.sum(directions*spatial,axis=-1)[...,None]


def _compact_local_action(grid,f,action,collision_strength,chunk_size,*,
                          density_ratio=None,base_action=None):
    wx,vw=grid.local_quadrature
    nx,nv=vw.shape
    # Each target row contracts all velocity neighbours. At least one row is
    # needed, so the effective pair workspace is max(chunk_size,Nv).
    budget=65536 if chunk_size is None else chunk_size
    if budget<1:
        raise ValueError('positive pair chunk_size required')
    rows=max(1,budget//nv)
    rows=min(rows,grid.size)
    count=(grid.size+rows-1)//rows
    aa=action.reshape(nx,nv,5);ff=f.reshape(nx,nv)
    if density_ratio is not None:
        ratio=density_ratio.reshape(nx,nv)
        base=base_action.reshape(nx,nv,5)
    def body(index,accumulator):
        targets=index*rows+jnp.arange(rows)
        valid=targets<grid.size
        targets=jnp.minimum(targets,grid.size-1)
        x,i=targets//nv,targets%nv
        delta=aa[x,i,None]-aa[x]
        if density_ratio is not None:
            delta=delta+(ratio[x,i,None]+ratio[x])[...,None]*(base[x,i,None]-base[x])
        if grid.uniform_direction is None:
            xi=grid.energy_flow[x,i,None]-grid.energy_flow[x]
            squared=jnp.sum(xi**2,axis=-1)
            directions=xi/jnp.sqrt(jnp.where(squared>0,squared,1.))[...,None]
        else:
            directions=grid.uniform_direction
        projected=_project_pair(delta,directions)
        neighbour=jnp.sum((vw[x]*ff[x])[...,None]*projected,axis=1)
        flux=(collision_strength*wx[x]*vw[x,i]*ff[x,i]*valid)[:,None]*neighbour
        # Targets are distinct except padding. Padding has zero flux.
        return accumulator.at[targets].add(flux)
    return jax.lax.fori_loop(0,count,body,jnp.zeros_like(action))


def compact_mobility_tangent(grid,f,action_h,delta_h,density_ratio,*,
                             collision_strength=1.,chunk_size=None):
    """Exact variation of K(f)h, using cached A h and delta_f/f.

    This is a bounded-workspace tangent, not a frozen-mobility approximation.
    A pair variation combines ΔA(delta_h)+(delta_f_i/f_i+delta_f_j/f_j)ΔA h
    inside the same fixed energy projector. Only nodal quantities are cached.
    """
    if grid.local_quadrature is None:
        raise ValueError('compact local grid required for prepared mobility tangent')
    tangent=grid.action(delta_h)
    flux=_compact_local_action(grid,jnp.asarray(f),tangent,collision_strength,chunk_size,
                              density_ratio=jnp.asarray(density_ratio),base_action=action_h)
    return grid.transpose_action(flux)


def mobility_action(grid, f, h, *, collision_strength=1., chunk_size=None):
    """K(f)h from unordered pairs, with optional bounded pair workspace.

    The stored kernel is its five-component unit energy direction. P I_x P
    is applied by two vector projections without a (pair,5,5) allocation.
    chunk_size bounds transient pair work; it leaves quadrature unchanged.
    """
    if isinstance(collision_strength, Real) and (not np.isfinite(collision_strength) or collision_strength<0):
        raise ValueError('finite nonnegative collision strength required')
    action = grid.action(h)
    f = jnp.asarray(f).reshape(-1)
    if grid.local_quadrature is not None:
        return grid.transpose_action(_compact_local_action(grid,f,action,collision_strength,chunk_size))
    def pair_flux(left, right, weights, directions):
        difference = action[left] - action[right]
        projected = _project_pair(difference,directions)
        factors = collision_strength*weights*f[left]*f[right]
        return factors[:,None]*projected
    if chunk_size is None:
        flux = pair_flux(grid.left, grid.right, grid.pair_weights, grid.kernel_directions)
        accumulated = jnp.zeros_like(action).at[grid.left].add(flux).at[grid.right].add(-flux)
    else:
        if chunk_size < 1:
            raise ValueError('positive pair chunk_size required')
        count = len(grid.left)
        padded = ((count+chunk_size-1)//chunk_size)*chunk_size
        left=jnp.pad(grid.left,(0,padded-count)); right=jnp.pad(grid.right,(0,padded-count))
        weights=jnp.pad(grid.pair_weights,(0,padded-count))
        directions=jnp.pad(grid.kernel_directions,((0,padded-count),(0,0)))
        def body(index, accumulator):
            start=index*chunk_size
            ll=jax.lax.dynamic_slice_in_dim(left,start,chunk_size)
            rr=jax.lax.dynamic_slice_in_dim(right,start,chunk_size)
            ww=jax.lax.dynamic_slice_in_dim(weights,start,chunk_size)
            dd=jax.lax.dynamic_slice_in_dim(directions,start,chunk_size)
            flux=pair_flux(ll,rr,ww,dd)
            return accumulator.at[ll].add(flux).at[rr].add(-flux)
        accumulated=jax.lax.fori_loop(0,padded//chunk_size,body,jnp.zeros_like(action))
    return grid.transpose_action(accumulated)


def linear_rhs(grid, equilibrium, h, *, collision_strength=1.):
    mass = grid.weights * jnp.asarray(equilibrium).reshape(-1)
    return -mobility_action(grid, equilibrium, h, collision_strength=collision_strength) / mass


def nonlinear_rhs(grid, f, *, collision_strength=1.):
    """Specified fixed-field nonlinear SM local surrogate, not full Eq.119."""
    return -mobility_action(grid, f, jnp.log(jnp.asarray(f).reshape(-1)),
                            collision_strength=collision_strength) / grid.weights


def dense_mobility(grid, f, *, collision_strength=1.):
    """Independent host Gram assembly for tiny-grid verification."""
    if grid.local_quadrature is not None:
        raise ValueError('assemble an explicit-pair tiny grid for the independent dense reference')
    basis = np.eye(grid.size)
    actions = np.stack([np.asarray(grid.action(column)) for column in basis], axis=-1)
    delta = actions[np.asarray(grid.left)] - actions[np.asarray(grid.right)]
    factors = collision_strength * np.asarray(grid.pair_weights) * np.asarray(f).ravel()[np.asarray(grid.left)] * np.asarray(f).ravel()[np.asarray(grid.right)]
    return np.einsum('pai,pab,pbj,p->ij', delta, np.asarray(grid.kernels), delta, factors)


def cartesian_grid(x, y, z, u, mu, field, *, mass=1., charge=1.,
                   velocity_weights=None,compact=False,spatial_weights=None,
                   spatial_discretization='local_quadratic'):
    """Collision-only Cartesian box with natural zero flux on every face.

    Nonperiodic field values are never wrapped. Combined Hamiltonian evolution
    needs an independent admissible boundary construction for the selected box.
    """
    from .geometry import common_chart_action, field_vector, validate_geometry
    x,y,z,u,mu=map(np.asarray,(x,y,z,u,mu))
    if np.any(mu<0):
        raise ValueError('magnetic moment must be nonnegative')
    shape=tuple(map(len,(x,y,z,u,mu)))
    xx,yy,zz,uu,mm=np.meshgrid(x,y,z,u,mu,indexing='ij')
    nodes=np.stack((xx,yy,zz,uu,mm),axis=-1).reshape(-1,5)
    validate_geometry(nodes[:,:3],field,mass=mass,charge=charge)
    gradients=jnp.eye(5)[:4]
    action=jax.vmap(lambda point:jax.vmap(lambda gradient:common_chart_action(
        point,gradient,field,mass=mass,charge=charge))(gradients))
    coefficients=np.asarray(action(jnp.asarray(nodes))).swapaxes(1,2).reshape(shape+(5,4))
    positions=np.stack(np.meshgrid(x,y,z,indexing='ij'),axis=-1).reshape(-1,3)
    strength=np.linalg.norm(np.asarray(jax.vmap(lambda point:field_vector(point,field))(
        jnp.asarray(positions))),axis=1)
    if spatial_weights is None:
        spatial_weights=tuple(trapezoid_weights(axis) for axis in (x,y,z))
    if len(spatial_weights)!=3 or any(np.shape(weight)!=np.shape(axis) or
        not np.all(np.isfinite(weight)) or np.any(np.asarray(weight)<=0)
        for weight,axis in zip(spatial_weights,(x,y,z))):
        raise ValueError('positive finite spatial quadrature weights must match three axes')
    spatial_weights=np.einsum('i,j,k->ijk',*spatial_weights).ravel()
    velocity_weights=strength[:,None]*_velocity_quadrature(u,mu,velocity_weights).reshape(1,-1)
    energy=mass*uu**2/2+mm*strength.reshape(shape[:3]+(1,1))
    derivatives=tuple((axis,derivative_matrix(nodes,method=spatial_discretization if axis<3
        else 'local_quadratic')) for axis,nodes in enumerate((x,y,z,u)))
    return _complete_grid(shape,derivatives,coefficients,spatial_weights,velocity_weights,energy,
        f'{field.kind} collision-only box; natural no-flux collision boundaries',compact=compact)


def mobility_channel_covariance(grid,f,*,collision_strength=1.,chunk_size=65536):
    """Positive partner covariance q_d for auxiliary preconditioners.

    q_i,d = D wX v_i f_i sum(j!=i) v_j f_j ||I_x P_ij C_i,d||².
    All derivative and pair cross terms remain in the exact mobility/Hessian.
    """
    if isinstance(collision_strength, Real) and (not np.isfinite(collision_strength) or collision_strength<0):
        raise ValueError('finite nonnegative collision strength required')
    f=jnp.asarray(f)
    if grid.local_quadrature is None:
        raise ValueError('lumped diagonal requires a compact local grid')
    if chunk_size<1:
        raise ValueError('positive pair chunk_size required')
    wx,vw=grid.local_quadrature;nx,nv=vw.shape
    count_d=len(grid.derivatives)
    coefficients=grid.coefficients.reshape(nx,nv,5,count_d).swapaxes(-1,-2)
    population=f.reshape(nx,nv)
    rows=min(grid.size,max(1,chunk_size//nv));blocks=(grid.size+rows-1)//rows
    def body(index,result):
        target=index*rows+jnp.arange(rows);valid=target<grid.size
        target=jnp.minimum(target,grid.size-1);x,i=target//nv,target%nv
        if grid.uniform_direction is None:
            difference=grid.energy_flow[x,i,None]-grid.energy_flow[x]
            squared=jnp.sum(difference*difference,axis=-1)
            direction=difference/jnp.sqrt(jnp.where(squared>0,squared,1.))[...,None]
        else:
            direction=jnp.broadcast_to(grid.uniform_direction,(rows,nv,5))
        ci=coefficients[x,i,None]
        projected=ci-direction[:,:,None,:]*jnp.sum(ci*direction[:,:,None,:],axis=-1)[...,None]
        channel=jnp.sum(projected[...,:3]**2,axis=-1)
        channel=jnp.where((i[:,None]!=jnp.arange(nv)[None,:])[...,None],channel,0.)
        q=collision_strength*(wx[x]*vw[x,i]*population[x,i])[:,None]*jnp.sum((vw[x]*population[x])[...,None]*channel,axis=1)
        return result.at[target].add(jnp.where(valid[:,None],q,0.))
    q=jax.lax.fori_loop(0,blocks,body,jnp.zeros((grid.size,count_d),dtype=f.dtype)).reshape(grid.shape+(count_d,))
    return q


def mobility_channel_diagonals(grid,q):
    """Positive channel diagonals (D_d squared).T q_d, before summing."""
    values=[]
    for component,(axis,derivative) in enumerate(grid.derivatives):
        local=jnp.tensordot((derivative*derivative).T,q[...,component],axes=(1,axis))
        values.append(jnp.moveaxis(local,0,axis))
    return jnp.stack(values,axis=-1)


def lumped_mobility_diagonal(grid,f,*,collision_strength=1.,chunk_size=65536):
    """Positive approximate diagonal for preconditioning only, not exact K."""
    q=mobility_channel_covariance(grid,f,collision_strength=collision_strength,chunk_size=chunk_size)
    return jnp.sum(mobility_channel_diagonals(grid,q),axis=-1).reshape(-1)
