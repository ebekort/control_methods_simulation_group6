'''
Evaluate a trained LNN (see pinn_train.py) on the test trajectories

1. One-step prediction: the test loss split into its 12 state components,
   to see which joints are hard to predict.
2. Gravity: the regulation controller (Eq. 13) uses A_L(q)^-1 G_L(q) as its
   gravity term. It is compared with PyBullet's true gravity vector at the
   test states (data generated with --log-truth). A gravity error dG gives a
   steady-state offset of about dG / K_P (Eq. 16), with K_P in real torque
   units per rad. This assumes the controller gains are scaled by A_L, i.e.
   the controller is A_L^-1 G_L + K_P e - K_D qd, because the scale of the
   learned model (the matrix P) is arbitrary.
3. Actuation matrix: the learned model can only match the truth up to a
   constant matrix P (Eqs. 10 to 12), A_L = P A. P_hat = A_L A^-1 is shown,
   with how close it is to a constant times the identity and whether it is
   positive definite.
4. Damping: A_L^-1 D_L against A^-1 D, in which P cancels.

The true A and D come from meta.json (data made with --inject-a/--inject-d);
without them A = I and D = 0, as in PyBullet. The gravity is compared in
input units, A_L^-1 G_L against A^-1 G, which is in Nm when A = I.

    python scripts/pinn_eval.py model.pkl data_100_wide
    python scripts/pinn_eval.py model.pkl data_100_wide --kp 10
'''
import argparse
import json
import pickle
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from cli_utils import HelpFormatter
from pinn_data import load_gravity, load_pairs
from pinn_model import (
    N, actuation_matrix, damping_matrix, gravity, rk4_step
)


def chunks(n, size):
    for i in range(0, n, size):
        yield slice(i, min(i + size, n))


def one_step_errors(params, weights, x, u, x_next, dt, size=1000):
    '''Mean weighted squared error per state component (2N,)'''
    step = jax.jit(jax.vmap(rk4_step, in_axes=(None, 0, 0, None)))
    total = np.zeros(2 * N)
    for part in chunks(len(x), size):
        pred = np.asarray(step(params, x[part], u[part], dt))
        total += (weights * (pred - x_next[part]) ** 2).sum(axis=0)
    return total / len(x)


def true_terms(folder):
    '''Actuation A and damping D the data were made with (meta.json)'''
    inject = json.loads((Path(folder) / 'meta.json').read_text()).get(
        'inject', {}
    )
    a = np.array(inject.get('A', np.eye(N)), dtype=float)
    d = np.array(inject.get('D', np.zeros((N, N))), dtype=float)
    return a, d


def learned_terms(params, q, size=5000):
    '''
    A_L^-1 G_L, A_L and A_L^-1 D_L for every row of q

    The damping is averaged over the states to keep the memory small.
    '''

    def terms(p, qq):
        a = actuation_matrix(p, qq)
        return (
            jnp.linalg.solve(a, gravity(p, qq)), a,
            jnp.linalg.solve(a, damping_matrix(p, qq))
        )

    fn = jax.jit(jax.vmap(terms, in_axes=(None, 0)))
    g_hat, a_mat, d_sum = [], [], np.zeros((N, N))
    for part in chunks(len(q), size):
        g, a, d = fn(params, q[part])
        g_hat.append(np.asarray(g))
        a_mat.append(np.asarray(a))
        d_sum += np.asarray(d).sum(axis=0)
    return np.vstack(g_hat), np.concatenate(a_mat), d_sum / len(q)


def rel(err, scale, floor=0.01):
    '''Relative error as text, or - when the true value is about zero'''
    return '-' if scale < floor else f'{err / scale:.2f}'


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=HelpFormatter
    )
    ap.add_argument('model', help='pickle file written by pinn_train.py')
    ap.add_argument('folder', help='data folder, e.g. data_100_wide')
    ap.add_argument(
        '--split', choices=('train', 'test', 'all'), default='test',
        help='which trajectories to evaluate on'
    )
    ap.add_argument(
        '--kp', type=float, default=5.0,
        help='proportional gain for the offset estimate [Nm/rad]'
    )
    args = ap.parse_args()

    with open(args.model, 'rb') as fh:
        model = pickle.load(fh)
    params, weights, dt = model['params'], model['weights'], model['dt']

    x, u, x_next, dt_data = load_pairs(args.folder, args.split)
    if abs(dt - dt_data) > 1e-9:
        raise ValueError('model and data have a different dt')

    err = one_step_errors(params, weights, x, u, x_next, dt)
    print(f'one-step loss: {err.sum():.4f} (weighted, per component below)')
    print('joint'.ljust(6) + 'q'.rjust(10) + 'qd'.rjust(10))
    for i in range(N):
        print(f'q{i + 1:<5}{err[i]:>10.4f}{err[N + i]:>10.4f}')

    a_true, d_true = true_terms(args.folder)
    injected = not (np.allclose(a_true, np.eye(N)) and not d_true.any())
    if injected:
        print()
        print('true A and D from meta.json (injected):')
        print('A =', np.array2string(a_true, precision=2, prefix='A = '))
        print('D =', np.array2string(d_true, precision=2, prefix='D = '))

    q, g_true = load_gravity(args.folder, args.split)
    g_true = np.linalg.solve(a_true, g_true.T).T  # A^-1 G, input units
    g_hat, a_mat, d_hat = learned_terms(params, q)
    resid = g_hat - g_true
    rmse = np.sqrt(np.mean(resid ** 2, axis=0))
    offset = rmse / args.kp  # steady-state offset estimate [rad]
    peak = np.abs(g_hat - g_true).max(axis=0)
    scale = np.sqrt(np.mean(g_true ** 2, axis=0))
    print()
    print('gravity A_L^-1 G_L versus A^-1 G (PyBullet) [input units]')
    print(
        'joint'.ljust(6) + 'rms true'.rjust(10) + 'rms err'.rjust(10)
        + 'max err'.rjust(10) + 'rel'.rjust(8)
        + f'  offset @ Kp={args.kp:g} [rad]'
    )
    for i in range(N):
        print(
            f'q{i + 1:<5}{scale[i]:>10.3f}{rmse[i]:>10.3f}{peak[i]:>10.3f}'
            f'{rel(rmse[i], scale[i]):>8}'
            f'  {offset[i]:>10.4f}'
        )

    p_hat = a_mat @ np.linalg.inv(a_true)  # = A_L when A = I
    diag = np.diagonal(p_hat, axis1=1, axis2=2)
    off = np.abs(p_hat - p_hat * np.eye(N)).sum()
    off /= len(p_hat) * N * (N - 1)
    sym = 0.5 * (p_hat + p_hat.transpose(0, 2, 1))
    pos = np.mean(np.linalg.eigvalsh(sym)[:, 0] > 0)
    print()
    print('P_hat = A_L A^-1 (should be about a constant times I)')
    print(
        f'diagonal: mean {diag.mean():.3f}, std {diag.std():.3f}, '
        f'min {diag.min():.3f}, max {diag.max():.3f}'
    )
    print(f'mean |off-diagonal|: {off:.3f}')
    print(f'positive definite (symmetric part) at {100 * pos:.1f}% of states')
    if injected:
        # A_L / c with c the mean diagonal of P_hat, against the true A
        a_scaled = a_mat.mean(axis=0) / diag.mean()
        print('mean A_L / (mean diagonal of P_hat), compare with A:')
        print(np.array2string(a_scaled, precision=2))

    d_ref = np.linalg.solve(a_true, d_true)
    print()
    print('damping A_L^-1 D_L (mean over the states) versus A^-1 D')
    print(
        'joint'.ljust(6) + 'true'.rjust(10) + 'learned'.rjust(10)
        + '   (diagonal)'
    )
    for i in range(N):
        print(f'q{i + 1:<5}{d_ref[i, i]:>10.3f}{d_hat[i, i]:>10.3f}')
    off_true = np.abs(d_ref - np.diag(np.diag(d_ref))).max()
    off_hat = np.abs(d_hat - np.diag(np.diag(d_hat))).max()
    print(f'max |off-diagonal|: true {off_true:.3f}, learned {off_hat:.3f}')


if __name__ == '__main__':
    main()
