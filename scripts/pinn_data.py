'''
Data loader for training the LNN

Reads .csv files made by pd+gravity_control_data.py and builds
one-step transition pairs. Row `k` of a file holds the state at `t_k` and the
torque `u_k` held constant over `[t_k, t_k + dt]`, so the pairs are

    (x_k, u_k) -> x_k+1        with x = (q, qd)

Pairs never span two files, because trajectories are independent. Arrays are
plain float64 numpy arrays, convert them to JAX arrays in the training script.

    python scripts/pinn_data.py data_100_wide
    python scripts/pinn_data.py data_100_wide --split test
'''
import argparse
from pathlib import Path
import numpy as np
from cli_utils import HelpFormatter

N = 6  # joints
DT_TOL = 1e-9  # allowed spread in the sampling time [s]


def read_trajectory(path):
    '''Time, state (q, qd) and torque of one .csv file'''
    with open(path) as f:
        header = f.readline().strip().split(',')
    names = [f'{p}{i + 1}' for p in ('q', 'qd', 'u') for i in range(N)]
    cols = [header.index('t')] + [header.index(n) for n in names]
    data = np.loadtxt(path, delimiter=',', skiprows=1, usecols=cols)
    return data[:, 0], data[:, 1:2 * N + 1], data[:, 2 * N + 1:]


def load_pairs(folder, split='train'):
    '''
    One-step transitions of all trajectories in [folder]/[split]

    Returns x (M, 2N), u (M, N), x_next (M, 2N) and the sampling time dt
    split is 'train', 'test' or 'all'
    '''
    names = ('train', 'test') if split == 'all' else (split,)
    files = sorted(
        f for n in names for f in (Path(folder) / n).glob('*.csv')
    )
    if not files:
        raise SystemExit(f'no .csv files found in {folder} ({split})')

    x, u, x_next, dt = [], [], [], None
    for f in files:
        t, state, torque = read_trajectory(f)
        steps = np.diff(t)
        if len(t) < 2 or np.ptp(steps) > DT_TOL:
            raise ValueError(f'{f}: time steps are not uniform')
        if dt is not None and abs(steps[0] - dt) > DT_TOL:
            raise ValueError(f'{f}: dt differs from the other files')
        dt = steps[0]
        x.append(state[:-1])
        u.append(torque[:-1])
        x_next.append(state[1:])
    return np.vstack(x), np.vstack(u), np.vstack(x_next), float(dt)


def load_gravity(folder, split='test'):
    '''
    Joint angles and PyBullet gravity vectors of every row (M, N) each

    Needs data generated with `--log-truth`, which adds the columns g1..g6
    '''
    names = ('train', 'test') if split == 'all' else (split,)
    files = sorted(
        f for n in names for f in (Path(folder) / n).glob('*.csv')
    )
    if not files:
        raise SystemExit(f'no .csv files found in {folder} ({split})')
    q, g = [], []
    for f in files:
        with open(f) as fh:
            header = fh.readline().strip().split(',')
        if 'g1' not in header:
            raise SystemExit(f'{f} has no g columns, use --log-truth')
        cols = [
            header.index(f'{p}{i + 1}') for p in ('q', 'g') for i in range(N)
        ]
        data = np.loadtxt(f, delimiter=',', skiprows=1, usecols=cols)
        q.append(data[:, :N])
        g.append(data[:, N:])
    return np.vstack(q), np.vstack(g)


def batches(x, u, x_next, batch_size, rng):
    '''
    Shuffled minibatches for one epoch

    The last incomplete batch is dropped, so every batch has the same shape
    for jit compilation
    '''
    order = rng.permutation(len(x))
    for i in range(0, len(x) - batch_size + 1, batch_size):
        idx = order[i:i + batch_size]
        yield x[idx], u[idx], x_next[idx]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=HelpFormatter
    )
    ap.add_argument('folder', help='data folder, e.g. data_100_wide')
    ap.add_argument(
        '--split', choices=('train', 'test', 'all'), default='train',
        help='which trajectories to load'
    )
    args = ap.parse_args()

    x, u, x_next, dt = load_pairs(args.folder, args.split)
    print(f'{len(x)} transition pairs ({args.split}), dt = {dt:g} s')
    print('joint'.ljust(6) + 'std qd'.rjust(10) + 'std u'.rjust(10))
    for i in range(N):
        print(
            f'q{i + 1:<5}{x[:, N + i].std():>10.4f}{u[:, i].std():>10.4f}'
        )


if __name__ == '__main__':
    main()
