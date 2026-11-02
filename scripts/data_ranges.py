'''
Prints the min and max of each joint over a data folder, and checks whether
a `--target` pose lies inside it

    python scripts/data_ranges.py data_100
    python scripts/data_ranges.py data_100 --split all --target Q1 ... Q6
'''
import argparse
from pathlib import Path
import numpy as np

N = 6  # arm joints
EDGE = 0.1  # a target this close to the range edge is flagged [rad]


class HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    '''Show argument defaults and keep the docstring layout.'''


def load_q(folder, split):
    '''Joint angles of all trajectories in the folder, stacked row-wise.'''
    names = ('train', 'test') if split == 'all' else (split,)
    files = sorted(f for n in names for f in (folder / n).glob('*.csv'))
    if not files:
        raise SystemExit(f'no .csv files found in {folder} ({split})')
    q = []
    for f in files:
        with open(f) as fh:
            header = fh.readline().strip().split(',')
        cols = [header.index(f'q{i + 1}') for i in range(N)]
        q.append(np.loadtxt(f, delimiter=',', skiprows=1, usecols=cols))
    return np.vstack(q), len(files)


def status(value, low, high):
    '''Describe where a target value lies relative to the visited range.'''
    if value < low:
        return f'OUTSIDE ({low - value:.2f} below min)'
    if value > high:
        return f'OUTSIDE ({value - high:.2f} above max)'
    if min(value - low, high - value) < EDGE:
        return 'at the edge'
    return 'ok'


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=HelpFormatter
    )
    ap.add_argument('folder', help='data folder, e.g. data_100')
    ap.add_argument(
        '--split', choices=('train', 'test', 'all'), default='train',
        help='which trajectories to include'
    )
    ap.add_argument(
        '--target', type=float, nargs=N, metavar='Q',
        help='joint angles of a target configuration to check [rad]'
    )
    args = ap.parse_args()

    q, n_files = load_q(Path(args.folder), args.split)
    low, high = q.min(axis=0), q.max(axis=0)
    print(f'{n_files} trajectories, {len(q)} samples ({args.split})')

    header = 'joint'.ljust(6) + 'min'.rjust(8) + 'max'.rjust(8)
    if args.target:
        header += 'target'.rjust(8) + '  status'
    print(header)
    for i in range(N):
        line = f'q{i + 1:<5}{low[i]:>8.2f}{high[i]:>8.2f}'
        if args.target:
            t = args.target[i]
            line += f'{t:>8.2f}  {status(t, low[i], high[i])}'
        print(line)


if __name__ == '__main__':
    main()
