'''
Plot and summarise the logs written by pinn_contact.py --log

Top panel: end-effector position and reference along the push axis.
Bottom panel: measured contact force (solid) and spring force
K_x (x_d - x) (dashed), one colour per file. The summary table gives,
for the last --tail seconds, the mean contact and spring force, their
difference and the drift of the contact force (N/s). A drift close to
zero means the run is in steady state.

    python scripts/pinn_contact_plot.py logs/contact_learned.csv \\
        logs/contact_true.csv --labels learned true --out contact.png
    python scripts/pinn_contact_plot.py logs/contact_learned.csv --axis z
'''
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from cli_utils import HelpFormatter


def load(path):
    '''Structured array with the columns of the pinn_contact.py log'''
    return np.genfromtxt(path, delimiter=',', names=True)


def tail_stats(data, seconds):
    '''Mean contact force, mean spring force and contact drift [N/s]'''
    mask = data['t'] >= data['t'][-1] - seconds
    t = data['t'][mask]
    f_contact = data['f_contact'][mask]
    f_spring = np.abs(data['f_spring'][mask])
    drift = np.polyfit(t, f_contact, 1)[0]
    return f_contact.mean(), f_spring.mean(), drift


def summary(files, labels, datas, tail):
    print(f'last {tail:g} s of every run')
    print(
        'run'.ljust(16) + 'peak [N]'.rjust(10) + 'contact [N]'.rjust(13)
        + 'spring [N]'.rjust(12) + 'diff [N]'.rjust(10)
        + 'drift [N/s]'.rjust(13)
    )
    for label, data in zip(labels, datas):
        f_c, f_s, drift = tail_stats(data, tail)
        print(
            f'{label[:15]:<16}{data["f_contact"].max():>10.1f}'
            f'{f_c:>13.2f}{f_s:>12.2f}{f_c - f_s:>10.2f}{drift:>13.3f}'
        )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=HelpFormatter
    )
    ap.add_argument('files', nargs='+', help='.csv logs of pinn_contact.py')
    ap.add_argument(
        '--labels', nargs='+', default=None,
        help='legend label per file (default: file name)'
    )
    ap.add_argument(
        '--axis', choices=('x', 'y', 'z'), default='z',
        help='push axis of the runs'
    )
    ap.add_argument(
        '--surface', type=float, default=None,
        help='table surface height along the axis [m]'
    )
    ap.add_argument(
        '--tail', type=float, default=1.0,
        help='seconds at the end used for the summary'
    )
    ap.add_argument(
        '--ymax', type=float, default=None,
        help='upper limit of the force axis [N], to cut the impact peak'
    )
    ap.add_argument(
        '--out', default=None, help='save the figure instead of showing it'
    )
    args = ap.parse_args()

    labels = args.labels or [Path(f).stem for f in args.files]
    if len(labels) != len(args.files):
        ap.error('--labels needs one label per file')
    datas = [load(f) for f in args.files]
    summary(args.files, labels, datas, args.tail)

    fig, (ax_x, ax_f) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    for i, (label, data) in enumerate(zip(labels, datas)):
        color = f'C{i}'
        ax_x.plot(
            data['t'], data[f'x{args.axis}'], color=color, label=label
        )
        ax_x.plot(
            data['t'], data[f'xd{args.axis}'], color=color, linestyle=':',
            linewidth=1
        )
        ax_f.plot(
            data['t'], data['f_contact'], color=color,
            label=f'{label} contact'
        )
        ax_f.plot(
            data['t'], np.abs(data['f_spring']), color=color, linestyle='--',
            label=f'{label} spring'
        )
    if args.surface is not None:
        ax_x.axhline(
            args.surface, color='gray', linewidth=1, label='surface'
        )
    ax_x.set_ylabel(f'{args.axis} [m] (dotted: reference)')
    ax_x.legend(loc='best')
    ax_f.set_ylabel('force [N]')
    ax_f.set_xlabel('time [s]')
    if args.ymax is not None:
        ax_f.set_ylim(-0.05 * args.ymax, args.ymax)
    ax_f.legend(loc='best', ncol=2, fontsize='small')
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=200)
        print(f'saved {args.out}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
