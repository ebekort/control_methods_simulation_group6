'''
Figures and the summary table for the report, from the logs in runs/

Reads only files written by the other scripts (no simulation), so it runs
without PyBullet. Every figure shows the five seeds of a recipe as a mean
line with a +-1 standard deviation band, next to the exact-model run.

    report_tracking_torques   joint torques of Eq. 25 tracking, per joint
    report_contact            contact force on the table and on the box
    report_progression        q2 gravity error and q6 inertia per stage,
                              measured on the same states (tracking path)
    report_progression_joint1..6   the same per joint (appendix)

    python scripts/report_plots.py
    python scripts/report_plots.py --runs ../runs --out ../figs

Inputs (names as written by the commands in the report notes):
    {prefix}_s{0..4}_trk_learned.csv   pinn_track.py --log
    {prefix}_s{0..4}_contact.csv       pinn_contact.py --log
    {prefix}_s{0..4}_box.csv           pinn_contact.py --scene box --log
    ref_trk_true.csv, ref_contact.csv, ref_box.csv   exact-model runs
    *_check.txt                        pinn_check_dynamics.py output
    *_reg.txt, *_trk_*.txt, *_box.txt, *_contact.txt   for the table
'''
import argparse
import re
from pathlib import Path
import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from cli_utils import HelpFormatter  # noqa: E402

N = 6
SEEDS = range(5)
TRUE_M66 = 0.00044  # PyBullet q6 inertia at the home pose [kg m^2]

# colours: categorical slots 1 and 2 of the reference palette, neutral ink
LEARNED = '#2a78d6'
SPRING = '#eb6834'
EXACT = '#52514e'
INK = '#0b0b0b'
MUTED = '#52514e'
GRID = '#e4e3df'
BAND_ALPHA = 0.2

# stages of the progression figure: label and check-file names
STAGES = [
    ('100 Hz\nCT data\n--a-reg', [
        'hz100_dither_net_reg_check.txt', 'hz100_dither_net_reg_s1_check.txt'
    ]),
    ('1 kHz\nCT data\n--a-reg', [f's{s}_check.txt' for s in SEEDS]),
    ('1 kHz\nPD data\n--a-reg', [f'pd_s{s}_check.txt' for s in SEEDS]),
    ('1 kHz\nPD data\nno reg', [f'pd0_s{s}_check.txt' for s in SEEDS]),
]


def style():
    '''Small print sizes for a two-column paper, recessive axes.'''
    plt.rcParams.update({
        'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.edgecolor': GRID, 'axes.linewidth': 0.6,
        'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5,
        'grid.linestyle': '-', 'axes.spines.top': False,
        'axes.spines.right': False, 'lines.linewidth': 1.2,
        'lines.solid_capstyle': 'round', 'legend.frameon': False,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
    })


def load(path):
    '''A .csv log as a dict of columns.'''
    data = np.genfromtxt(path, delimiter=',', names=True)
    return {name: data[name] for name in data.dtype.names}


def load_seeds(runs, prefix, kind):
    '''The logs of the five seeds, cut to a common length.'''
    logs = [load(runs / f'{prefix}_s{s}_{kind}.csv') for s in SEEDS]
    n = min(len(log['t']) for log in logs)
    return {key: np.array([log[key][:n] for log in logs]) for key in logs[0]}


def band(ax, t, runs, color, label):
    '''Mean line and +-1 std band over the first axis of `runs`.'''
    mean, std = runs.mean(axis=0), runs.std(axis=0)
    ax.fill_between(
        t, mean - std, mean + std, color=color, alpha=BAND_ALPHA, linewidth=0
    )
    ax.plot(t, mean, color=color, label=label)


def save(fig, out, name, tight=True):
    '''tight=False keeps the figure size, so a series of figures scales
    the same in the report.'''
    with plt.rc_context({'savefig.bbox': 'tight' if tight else 'standard'}):
        for ext in ('pdf', 'png'):
            fig.savefig(out / f'{name}.{ext}')
    plt.close(fig)
    print(f'wrote {out / name}.pdf/.png')


def tracking_torques(runs, out, prefix):
    seeds = load_seeds(runs, prefix, 'trk_learned')
    ref = load(runs / 'ref_trk_true.csv')
    t = seeds['t'][0]
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.3), sharex=True)
    for i, ax in enumerate(axes.flat):
        key = f'u{i + 1}'
        # joint 6 needs about 1e-3 Nm, shown in mNm to keep the ticks short
        scale, unit = (1e3, 'mNm') if i == 5 else (1.0, 'Nm')
        band(ax, t, seeds[key] * scale, LEARNED, 'learned model (5 seeds)')
        # the exact run on top, thin and dashed: it lies on the learned
        # mean, so a solid line would hide one of the two
        ax.plot(
            ref['t'], ref[key] * scale, color=INK, linewidth=0.8,
            linestyle=(0, (3, 2)), label='exact model'
        )
        ax.set_title(f'joint {i + 1} [{unit}]', loc='left', color=INK)
        if i >= 3:
            ax.set_xlabel('time [s]')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.04)
    )
    fig.tight_layout()
    fig.suptitle(
        'Joint torques during tracking with the learned '
        'feedforward: mean and \u00b11 std band over five seeds',
        y=1.11, color=INK
    )
    save(fig, out, 'report_tracking_torques')


def peak(path):
    '''
    Peak contact force from the text output (the .csv keeps every 10th
    step, so its maximum misses the short impact).'''
    return number(path, r'peak contact force:')


def contact(runs, out, prefix, clip):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5), sharey=True)
    for ax, kind, ref_name, title in (
        (axes[0], 'contact', 'ref_contact.csv', 'table under the start pose'),
        (axes[1], 'box', 'ref_box.csv', 'box of the impedance experiment (held fixed)'),
    ):
        seeds = load_seeds(runs, prefix, kind)
        ref = load(runs / ref_name)
        t = seeds['t'][0]
        # logged as K_x (x_d - x) along +z, negative while pushing down
        spring = np.clip(-seeds['f_spring'].mean(axis=0), 0, None)
        ax.plot(t, spring, color=SPRING, label='commanded spring force')
        ax.plot(
            ref['t'], ref['f_contact'], color=EXACT, linewidth=0.9,
            label='contact, exact gravity'
        )
        band(
            ax, t, seeds['f_contact'], LEARNED,
            'contact, learned gravity (5 seeds)'
        )
        peaks = np.array([
            peak(runs / f'{prefix}_s{s}_{kind}.txt') for s in SEEDS
        ])
        ax.text(
            0.98, 0.95,
            f'impact peaks {peaks.min():.0f} to {peaks.max():.0f} N '
            f'(clipped)', transform=ax.transAxes, ha='right', va='top',
            color=MUTED, fontsize=7
        )
        ax.set_ylim(0, clip)
        ax.set_title(title, loc='left', color=INK)
        ax.set_xlabel('time [s]')
    axes[0].set_ylabel('force [N]')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.06)
    )
    fig.tight_layout()
    fig.suptitle(
        'Contact force with learned gravity: mean and \u00b11 std '
        'band over five seeds', y=1.14, color=INK
    )
    save(fig, out, 'report_contact')


def check_values(path):
    '''q2 gravity RMS error along the reference and the learned q6 M_66.'''
    text = path.read_text()
    m66 = float(re.search(r'^q6\s+\S+\s+(\S+)\s*$', text, re.M).group(1))
    row = re.search(r'^q2\s+\S+\s+\S+\s+(\S+)/(\S+)\s+\S+\s*$', text, re.M)
    return float(row.group(2)), m66


def progression(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.4))
    for k, (label, files) in enumerate(STAGES):
        values = np.array([check_values(runs / f) for f in files])
        for ax, v in zip(axes, (values[:, 0], values[:, 1] / TRUE_M66)):
            x = np.full(len(v), k) + np.linspace(-0.12, 0.12, len(v))
            ax.plot(
                x, v, 'o', color=LEARNED, markersize=4,
                markeredgecolor='white', markeredgewidth=0.8
            )
            ax.plot(
                [k - 0.25, k + 0.25], [np.mean(v)] * 2, color=INK,
                linewidth=1.2
            )
    labels = [label for label, _ in STAGES]
    axes[0].set_ylabel('q2 gravity RMS error [Nm]')
    axes[0].set_ylim(bottom=0)
    axes[1].set_yscale('log')
    axes[1].axhline(1, color=EXACT, linewidth=0.9)
    axes[1].text(
        len(STAGES) - 0.5, 1.3, 'true value', ha='right',
        color=MUTED, fontsize=7
    )
    axes[1].set_ylabel('learned / true q6 inertia')
    for ax in axes:
        ax.set_xticks(range(len(STAGES)), labels)
        ax.set_xlim(-0.5, len(STAGES) - 0.5)
        ax.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.suptitle(
        'Learned model per development stage '
        '(dots: seeds, bar: mean over seeds)', y=1.05, color=INK
    )
    save(fig, out, 'report_progression')


def check_all(path):
    '''
    Learned / true inertia M_ii at the home pose and the gravity RMS
    error along the reference [Nm], per joint.'''
    text = path.read_text()
    inertia = re.findall(r'^q\d\s+(\S+)\s+(\S+)\s*$', text, re.M)[:N]
    ratio = [float(b) / float(a) for a, b in inertia]
    grav = re.findall(
        r'^q\d\s+\S+\s+\S+\s+\S+/(\S+)\s+\S+\s*$', text, re.M
    )[:N]
    return np.array(ratio), np.array([float(g) for g in grav])


def progression_joints(runs, out):
    '''The progression figure per joint, one file each (appendix).'''
    values = [[check_all(runs / f) for f in files] for _, files in STAGES]
    short = ['100 Hz', 'CT', 'PD', 'final']
    names = [
        'vertical axis, no gravity', 'shoulder', 'elbow', 'wrist 1',
        'wrist 2', 'wrist 3, tool axis, no gravity'
    ]
    for j in range(N):
        fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.3))
        for k, stage in enumerate(values):
            grav = np.array([v[1][j] for v in stage])
            ratio = np.array([v[0][j] for v in stage])
            x = np.full(len(stage), k) + np.linspace(-0.15, 0.15, len(stage))
            for ax, v in zip(axes, (grav, ratio)):
                ax.plot(
                    x, v, 'o', color=LEARNED, markersize=3.5,
                    markeredgecolor='white', markeredgewidth=0.6
                )
                ax.plot(
                    [k - 0.3, k + 0.3], [v.mean()] * 2, color=INK,
                    linewidth=1.0
                )
        axes[0].set_ylim(bottom=0)
        axes[0].set_title('gravity RMS error [Nm]', loc='left', color=INK)
        axes[1].set_title('learned / true inertia', loc='left', color=INK)
        # log scale only where the stages span more than a factor 10
        low, high = axes[1].dataLim.y0, axes[1].dataLim.y1
        if high / low > 10:
            axes[1].set_yscale('log')
            axes[1].yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}')
            )
            axes[1].yaxis.set_minor_formatter(
                matplotlib.ticker.NullFormatter()
            )
        axes[1].axhline(1, color=EXACT, linewidth=0.8)
        for ax in axes:
            ax.set_xticks(range(len(STAGES)), short)
            ax.set_xlim(-0.5, len(STAGES) - 0.5)
            ax.grid(axis='x', visible=False)
        # fixed page size (no tight crop) so all six scale the same
        fig.suptitle(
            f'Joint {j + 1} ({names[j]})\ndots: seeds, bar: mean over seeds',
            y=0.98, color=INK
        )
        fig.tight_layout()
        save(fig, out, f'report_progression_joint{j + 1}', tight=False)


def number(path, pattern):
    '''First float after `pattern` in a text output file.'''
    match = re.search(pattern + r'\s*([-+]?\d+\.\d+)', path.read_text())
    return float(match.group(1))


def summary(runs, prefix):
    '''Mean and std over the seeds of the closed-loop numbers.'''
    rows = [
        ('regulation error norm [rad]', 'reg', r'final error norm:'),
        (
            'tracking RMS, Eq. 25 [rad]', 'trk_learned',
            r'RMS error over all joints:'
        ),
        (
            'tracking RMS, PD + gravity [rad]', 'trk_gravity',
            r'RMS error over all joints:'
        ),
        ('table: contact - spring [N]', 'contact', r'measured excess:'),
        ('table: gravity part [N]', 'contact', r'predicted excess:.*?='),
        ('box: contact - spring [N]', 'box', r'measured excess:'),
        ('box: gravity part [N]', 'box', r'predicted excess:.*?='),
    ]
    print(f'\nclosed loop, {prefix}_s0..4 (mean +- std over seeds)')
    for name, kind, pattern in rows:
        v = np.array([
            number(runs / f'{prefix}_s{s}_{kind}.txt', pattern) for s in SEEDS
        ])
        print(f'  {name:36s} {v.mean():+9.4f} +- {v.std():.4f}')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=HelpFormatter
    )
    ap.add_argument('--runs', default='runs', help='folder with the logs')
    ap.add_argument('--out', default='figs', help='folder for the figures')
    ap.add_argument(
        '--prefix', default='pd0',
        help='file prefix of the five seeds of the recipe'
    )
    ap.add_argument(
        '--clip', type=float, default=40.0,
        help='upper limit of the contact force axis [N]'
    )
    args = ap.parse_args()
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    tracking_torques(runs, out, args.prefix)
    contact(runs, out, args.prefix, args.clip)
    progression(runs, out)
    progression_joints(runs, out)
    summary(runs, args.prefix)


if __name__ == '__main__':
    main()
