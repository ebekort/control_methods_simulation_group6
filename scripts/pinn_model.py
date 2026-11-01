'''
Lagrangian neural network (LNN) model for the UR7e

Four small networks give the dynamics terms as functions of the joint angles
q (Liu et al. 2024):

    M_theta(q)  mass matrix           (Cholesky factor + eps, positive def.)
    V_theta(q)  potential energy      (scalar, gravity is its gradient)
    D_theta(q)  dissipation matrix    (same construction, positive semi-def.)
    A_theta     actuation matrix, three options (see init_params):
                'network'   A(q), a network, as in the paper
                'diagonal'  a constant per-joint gain (learned)
                'identity'  A = I, only if the torque interface is known

Everything else follows from the Euler-Lagrange equation by autodiff:

    L(q, qd) = 1/2 qd^T M(q) qd - V(q)
    d/dt dL/dqd - dL/dq = A(q) u - D(q) qd

Training: integrate the learned dynamics one sample period with RK4 (u held
constant) and compare with the measured next state (Eq. 8).

    python scripts/pinn_model.py        runs sanity checks of the derivation

This file only defines the model. Data loading and the training loop are
separate.
'''
import jax
import jax.numpy as jnp

# float64: the state changes by ~1e-3 per sample, float32 would be too coarse
jax.config.update('jax_enable_x64', True)

N = 6  # joints
N_TRI = N * (N + 1) // 2  # entries of a lower-triangular N x N matrix
EPS = 1e-5  # added to M, keep well below the smallest true inertia
# diagonal of M at initialisation [kg m^2]: heavy at the base, light at the
# wrist, as in any serial arm (a generic guess, not the UR7e values)
MASS_INIT = jnp.geomspace(1.0, 3e-3, N)


def init_mlp(key, sizes, last_scale=1.0, last_bias=None):
    '''List of (W, b) layers. The last layer is scaled by `last_scale`.'''
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    for i, (k, n_in, n_out) in enumerate(zip(keys, sizes[:-1], sizes[1:])):
        w = jax.random.normal(k, (n_in, n_out)) / jnp.sqrt(n_in)
        if i == len(sizes) - 2:
            w = w * last_scale
        b = jnp.zeros(n_out)
        if i == len(sizes) - 2 and last_bias is not None:
            b = jnp.asarray(last_bias, dtype=b.dtype)
        params.append((w, b))
    return params


def mlp(params, x):
    '''
    Fully connected network. tanh is smooth, so second derivatives exist.

    (relu would not work: the Lagrangian is differentiated twice.)
    '''
    for w, b in params[:-1]:
        x = jnp.tanh(x @ w + b)
    w, b = params[-1]
    return x @ w + b


def _init_mass_net(key, hidden, mass_init):
    '''Mass network. 'ramp' starts it at the diagonal MASS_INIT.'''
    if mass_init == 'flat':
        return init_mlp(key, [N, *hidden, N_TRI])
    if mass_init != 'ramp':
        raise ValueError(f'unknown mass_init: {mass_init}')
    # softplus(b)^2 = MASS_INIT on the diagonal of the Cholesky factor
    rows, cols = jnp.tril_indices(N)
    inv_softplus = jnp.log(jnp.expm1(jnp.sqrt(MASS_INIT)))
    bias = jnp.where(rows == cols, inv_softplus[rows], 0.0)
    return init_mlp(key, [N, *hidden, N_TRI], last_scale=0.01, last_bias=bias)


def init_params(key, hidden=(64, 64), actuation='diagonal', mass_init='flat'):
    '''
    Parameters of the model.

    actuation chooses how A is represented, without assuming the robot:
        'network'   A(q) = I + network(q), a full matrix that depends on q
        'diagonal'  A = diag(exp(a)), one learned gain per joint
        'identity'  A = I, no parameters (the torque command is the joint
                    torque, only valid when that is known to be true)
    All options start at A = I.

    mass_init chooses the starting mass matrix:
        'flat'  the default network initialisation (M about 0.5 per joint)
        'ramp'  diagonal MASS_INIT, heavy at the base and light at the wrist

    A first test with 'ramp' on the UR7e data gave a worse model overall.
    '''
    km, kv, kd, ka = jax.random.split(key, 4)
    if actuation == 'network':
        a = init_mlp(ka, [N, *hidden, N * N], last_scale=0.0)
    elif actuation == 'diagonal':
        a = jnp.zeros(N)  # log of the gains
    elif actuation == 'identity':
        a = None
    else:
        raise ValueError(f'unknown actuation: {actuation}')
    return {
        'M': _init_mass_net(km, hidden, mass_init),
        'V': init_mlp(kv, [N, *hidden, 1]),
        'D': init_mlp(kd, [N, *hidden, N_TRI], last_scale=0.1),
        'A': a,
    }


def _psd_matrix(net, q):
    '''Positive semi-definite matrix L L^T from a lower-triangular L.'''
    vec = mlp(net, q)
    low = jnp.zeros((N, N)).at[jnp.tril_indices(N)].set(vec)
    diag = jnp.diag(jax.nn.softplus(jnp.diag(low)))  # positive diagonal
    low = low - jnp.diag(jnp.diag(low)) + diag
    return low @ low.T


def mass_matrix(params, q):
    return _psd_matrix(params['M'], q) + EPS * jnp.eye(N)


def potential(params, q):
    return mlp(params['V'], q)[0]


def damping_matrix(params, q):
    return _psd_matrix(params['D'], q)


def actuation_matrix(params, q):
    '''A(q), the form depends on how the parameters were initialised.'''
    a = params['A']
    if a is None:
        return jnp.eye(N)
    if isinstance(a, list):
        return jnp.eye(N) + mlp(a, q).reshape(N, N)
    return jnp.diag(jnp.exp(a))


def actuation_penalty(params, q):
    '''
    Mean squared distance of A(q) from the identity over a batch of q.

    A soft prior that fixes the arbitrary scale of the learned model (the
    matrix P): the loss is flat along that direction, so without a penalty
    A drifts away from the identity during training.
    '''
    def distance(qq):
        return jnp.sum((actuation_matrix(params, qq) - jnp.eye(N)) ** 2)

    return jnp.mean(jax.vmap(distance)(q))


def gravity(params, q):
    '''G_L(q), the gradient of the learned potential.'''
    return jax.grad(potential, argnums=1)(params, q)


def lagrangian(params, q, qd):
    return 0.5 * qd @ mass_matrix(params, q) @ qd - potential(params, q)


def acceleration(params, q, qd, u):
    '''
    qdd from the Euler-Lagrange equation, using autodiff only.

    d/dt (dL/dqd) = H qdd + (d/dq dL/dqd) qd, with H = d2L/dqd2, so
    H qdd = F + dL/dq - (d/dq dL/dqd) qd, with F = A u - D qd.
    '''
    dl_dq = jax.grad(lagrangian, argnums=1)(params, q, qd)
    hess = jax.hessian(lagrangian, argnums=2)(params, q, qd)
    dqd_dq = jax.jacfwd(jax.grad(lagrangian, argnums=2), argnums=1)(
        params, q, qd
    )
    force = actuation_matrix(params, q) @ u - damping_matrix(params, q) @ qd
    return jnp.linalg.solve(hess, force + dl_dq - dqd_dq @ qd)


def required_force(params, q, qd, qdd):
    '''
    Generalised force A u that produces qdd (the feedforward of Eq. 25).

    M_L(q) qdd + C_L(q, qd) qd + G_L(q) + D_L(q) qd, written with the same
    autodiff derivatives as `acceleration`: M qdd + C qd + G equals
    H qdd + (d/dq dL/dqd) qd - dL/dq.
    '''
    dl_dq = jax.grad(lagrangian, argnums=1)(params, q, qd)
    hess = jax.hessian(lagrangian, argnums=2)(params, q, qd)
    dqd_dq = jax.jacfwd(jax.grad(lagrangian, argnums=2), argnums=1)(
        params, q, qd
    )
    return (hess @ qdd + dqd_dq @ qd - dl_dq + damping_matrix(params, q) @ qd)


def dynamics(params, x, u):
    '''Time derivative of the state x = (q, qd)'''
    q, qd = x[:N], x[N:]
    return jnp.concatenate([qd, acceleration(params, q, qd, u)])


def rk4_step(params, x, u, dt):
    '''One RK4 step with u held constant (zero-order hold)'''
    k1 = dynamics(params, x, u)
    k2 = dynamics(params, x + 0.5 * dt * k1, u)
    k3 = dynamics(params, x + 0.5 * dt * k2, u)
    k4 = dynamics(params, x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def loss_fn(params, x, u, x_next, dt, weights=None):
    '''
    Eq. 8: mean squared error between predicted and measured next state.

    x, x_next: (batch, 2N) states (q, qd); u: (batch, N).
    weights: optional (2N,) weights, e.g. to balance joints of different scale.
    '''
    pred = jax.vmap(rk4_step, in_axes=(None, 0, 0, None))(params, x, u, dt)
    err = (pred - x_next) ** 2
    if weights is not None:
        err = err * weights
    return jnp.mean(jnp.sum(err, axis=1))


def run_checks():
    for mode in ('network', 'diagonal', 'identity'):
        print(f'--- actuation: {mode}')
        check(init_params(jax.random.PRNGKey(0), actuation=mode))


def check(params):
    kq, kv, ku = jax.random.split(jax.random.PRNGKey(1), 3)
    q = jax.random.uniform(kq, (N,), minval=-1.5, maxval=1.5)
    qd = jax.random.normal(kv, (N,))
    u = jax.random.normal(ku, (N,))

    # M is symmetric positive definite
    mass = mass_matrix(params, q)
    eig = jnp.linalg.eigvalsh(mass)
    print(
        'M symmetric:', bool(jnp.allclose(mass, mass.T)),
        '| smallest eigenvalue:', float(eig[0])
    )

    # static equilibrium: u = A^-1 G gives qdd = 0 when qd = 0
    u_eq = jnp.linalg.solve(actuation_matrix(params, q), gravity(params, q))
    qdd = acceleration(params, q, jnp.zeros(N), u_eq)
    print('max |qdd| at equilibrium (expect 0):', float(jnp.abs(qdd).max()))

    # energy balance with u = 0: dE/dt = -qd^T D qd
    def energy(x):
        qq, vv = x[:N], x[N:]
        return 0.5 * vv @ mass_matrix(params, qq) @ vv + potential(params, qq)

    x = jnp.concatenate([q, qd])
    dt = 1e-5
    x_next = rk4_step(params, x, jnp.zeros(N), dt)
    de_dt = (energy(x_next) - energy(x)) / dt
    expected = -qd @ damping_matrix(params, q) @ qd
    print('dE/dt (RK4):', float(de_dt), '| -qd^T D qd:', float(expected))

    # required_force is the inverse of acceleration
    qdd_t = jax.random.normal(jax.random.PRNGKey(5), (N,))
    u_inv = jnp.linalg.solve(
        actuation_matrix(params, q), required_force(params, q, qd, qdd_t)
    )
    err = jnp.abs(acceleration(params, q, qd, u_inv) - qdd_t).max()
    print('max |qdd(u_inv) - qdd| (expect 0):', float(err))

    # the loss is differentiable and finite
    xs = jnp.stack([x, x])
    loss, grads = jax.value_and_grad(loss_fn)(
        params, xs, jnp.stack([u, u]), xs, 0.01
    )
    leaves = jax.tree_util.tree_leaves(grads)
    print(
        'loss:', float(loss), '| all gradients finite:',
        all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
    )


if __name__ == '__main__':
    run_checks()
