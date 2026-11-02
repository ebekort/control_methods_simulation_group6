'''
Drives the simulated robot along random sinusoids with a tracking
controller (see --controller) and writes .csv data

    python scripts/pd+gravity_control_data.py --n-traj 100 --out data
    python scripts/pd+gravity_control_data.py --n-traj 3 --gui
    python scripts/pd+gravity_control_data.py --dt 0.001 --out data_1khz
    python scripts/pd+gravity_control_data.py --inject-a --inject-d
    python scripts/pd+gravity_control_data.py --controller pd --dt 0.001

Controller (--controller):
    model  u = G(q) + M(q) [qdd_ref + w^2 e + 2 w ed]   computed torque with
           PyBullet's exact M and G; one gain setting suits every joint
    pd     u = K_P e + K_D ed                            plain PD, no model:
           fixed per-joint gains (KP_PD, KD_PD), so the arm sags under
           gravity and lags the reference, which only shifts the visited
           states. The whole pipeline is then model-free. Use --dt 0.001:
           the light wrist is unstable with these gains at 100 Hz.

Injected actuation and damping (--inject-a, --inject-d): in PyBullet the true
A is I and the true D is 0, so a learned A_L near c I does not show that the
method can identify anything else. With these options the logged input u is
not the joint torque: the motor torque tau = A u (A = A_INJECT), and the
joints feel an extra viscous torque -D qd (D = D_INJECT), so the data obeys

    M qdd + C qd + G = A u - D qd           (Eq. 6 with known A and D)

The controller compensates the damping (it knows D, as it knows M and G), so
the trajectories stay close to the ones without injection. Both matrices are
constant and stored in meta.json for pinn_eval.py

Output file paths:
    data/train/traj_0000.csv ...   one file per trajectory
    data/test/traj_0080.csv ...
    data/meta.json                 settings, split and sinusoid parameters

.csv columns (one row per sample, every `--dt` seconds):
    t, q1..q6, qd1..qd6, u1..u6 [, g1..g6 with --log-truth]
Row k has state at t_k and input u_k held constant over
[t_k, t_k + dt]. A training pair is (q_k, qd_k, u_k) -> (q_k+1, qd_k+1)
'''
import argparse
import functools
import json
from pathlib import Path
import numpy as np
import ur_simulation.classic_control.robots.ur7e as ur7e_mod
from ur_simulation.pybullet import PyBullet
from cli_utils import HelpFormatter

N = 6  # arm joints

# sampling / control (the period dt is a script argument)
DT_DEFAULT = 0.01  # default sampling and control period [s]
WARMUP = 0.5  # seconds discarded at the start of each trajectory
OMEGA = 8.0  # tracking bandwidth [rad/s]
TRAIN_FRAC = 0.8
MAX_RETRIES = 20

# sinusoid ranges
CENTER_SPREAD = 0.8  # +- offset of c around HOME [rad]
A_RANGE = (0.1, 0.6)  # amplitude [rad]
F_RANGE = (0.05, 0.5)  # frequency [Hz]
V_REF_MAX = 1.5  # peak reference speed [rad/s]

# safety/limits
Q_LIM = np.array([
    2 * np.pi, 2 * np.pi, np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi
])  # from ur7e.urdf
LIMIT_MARGIN = 0.3  # distance kept from the joint limits [rad]
V_MAX = 0.9 * np.pi  # abort if |qdot| exceeds this [rad/s]
TAU_LIM = np.array([150, 150, 150, 28, 28, 28.0])  # UR7e max torques [Nm]
FLOOR_MARGIN = 0.02  # minimum link height [m]
BASE_RADIUS = 0.15  # keep-out cylinder around the base [m]
BASE_HEIGHT = 0.30  # [m]

# known actuation and damping for --inject-a / --inject-d: unequal joint
# scales plus wrist coupling (as in a belt or differential drive), and a
# damping of a few percent of the torque range per rad/s
A_INJECT = np.array([
    [0.5, 0, 0, 0, 0, 0],
    [0, 2.0, 0, 0, 0, 0],
    [0, 0, 1.0, 0, 0, 0],
    [0, 0, 0, 1.5, 0.3, 0],
    [0, 0, 0, 0, 0.8, 0.3],
    [0, 0, 0, 0, 0, 1.2],
])
D_INJECT = np.diag([3.0, 3.0, 1.5, 0.3, 0.3, 0.05])  # [Nm s/rad]

# gains of --controller pd, picked by hand: K_P against the gravity load
# (sag of about 0.1 rad on q2 and q3) and K_D near critical damping for the
# inertia at the home pose (1.5, 3.1, 1.6, 0.085, 0.036, 0.00044 kg m^2)
KP_PD = np.array([300, 500, 400, 100, 50, 10.0])  # [Nm/rad]
KD_PD = np.array([40, 80, 50, 5, 2.5, 0.12])  # [Nm s/rad]
# --dither of --controller pd is a torque: --dither times this per joint,
# about the home-pose inertia, so 1.0 is comparable to the model controller
DITHER_PD = np.array([1.5, 3.0, 1.5, 0.08, 0.04, 0.0005])  # [Nm]

# pose from pd+gravity_control.py (classic_control.py has 2.4 for joint 3)
HOME = np.array([1.57, -1.7, 1.7, -1.57, -1.57, -1.57])


def sample_reference(rng):
    '''
    Random per-joint sinusoid c + a sin(2 pi f t + phi)

    The amplitude is shrunk where needed so the sinusoid stays inside the
    joint limit margin and the peak reference speed stays below V_REF_MAX
    '''
    c = HOME + rng.uniform(-CENTER_SPREAD, CENTER_SPREAD, N)
    f = rng.uniform(*F_RANGE, N)
    a = rng.uniform(*A_RANGE, N)
    phi = rng.uniform(0, 2 * np.pi, N)

    room = Q_LIM - LIMIT_MARGIN - np.abs(c)
    a = np.minimum(a, np.maximum(room, 0.0))
    a = np.minimum(a, V_REF_MAX / (2 * np.pi * f))
    return c, a, f, phi


def reference(t, c, a, f, phi):
    '''Reference position, velocity and acceleration at time t'''
    w = 2 * np.pi * f
    s = w * t + phi
    return c + a * np.sin(s), a * w * np.cos(s), -a * w**2 * np.sin(s)


def make_robot(gui):
    if not gui:
        # force headless mode
        ur7e_mod.PyBullet = functools.partial(
            PyBullet, render_mode='rgb_array', renderer='Tiny'
        )
    robot = ur7e_mod.UR7e(
        block_gripper=True, neutral_joints=HOME, n_substeps=1,
        step_frequency=1000,
    )
    robot.sim.create_plane(0)
    return robot


def safety_violation(robot, q, qd):
    '''Return reason string if state is unsafe, else None'''
    pc = robot.sim.physics_client
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(qd))):
        return 'non-finite state'
    if np.any(np.abs(q) > Q_LIM - LIMIT_MARGIN / 2):
        return 'joint limit'
    if np.any(np.abs(qd) > V_MAX):
        return 'joint speed'
    first = robot.arm_indices[0]
    contacts = [
        c for c in pc.getContactPoints(bodyA=robot.robot_id)
        if c[3] >= first
    ]
    if contacts:
        return f'contact (link {contacts[0][3]}, body {contacts[0][2]})'
    for link in range(first, robot.ee_link + 1):
        if pc.getLinkState(robot.robot_id, link)[0][2] < FLOOR_MARGIN:
            return 'floor'
    ee = np.asarray(robot.robot_state.get_end_effector_position())
    if np.hypot(ee[0], ee[1]) < BASE_RADIUS and ee[2] < BASE_HEIGHT:
        return 'base'
    return None


def run_trajectory(
    robot, rng, duration, dt, log_truth, realtime, dither,
    a_mat, d_mat, controller
):
    '''
    Simulate one trajectory, with actuation a_mat and damping d_mat and the
    data controller 'model' or 'pd'

    Returns (rows, params), or (None, reason) if a safety check failed
    '''
    c, a, f, phi = sample_reference(rng)
    robot.set_joint_angles(reference(0.0, c, a, f, phi)[0])
    robot.sim.physics_client.performCollisionDetection()

    kp, kd = OMEGA ** 2, 2 * OMEGA
    substeps = int(round(dt / robot.sim.timestep))
    n_samples = int(round(duration / dt))
    n_warm = int(round(WARMUP / dt))
    rows = []

    for k in range(n_warm + n_samples + 1):
        t = k * dt
        q = np.asarray(robot.robot_state.get_joint_angles(), dtype=float)
        qd = np.asarray(robot.robot_state.get_joint_velocities(), dtype=float)
        reason = safety_violation(robot, q, qd)
        if reason:
            return None, reason

        q_r, qd_r, qdd_r = reference(t, c, a, f, phi)
        xi = dither * rng.standard_normal(N) if dither else 0.0
        if controller == 'model' or log_truth:
            dyn = robot.robot_model.get_dynamics()
            g = np.asarray(dyn.gravity_vector, dtype=float)
        if controller == 'model':
            # u = G(q) + M(q) [qdd_ref + Kp e + Kd ed], gravity at the
            # current q (Eq. 13); the mass matrix scales the gains so one
            # (kp, kd) suits heavy and light joints alike. It knows D too.
            mass = np.asarray(dyn.mass_matrix, dtype=float)
            tau = g + mass @ (qdd_r + kp * (q_r - q) + kd * (qd_r - qd) + xi)
            tau = tau + d_mat @ qd
        else:
            # plain PD, no model; g is only logged (--log-truth)
            tau = (KP_PD * (q_r - q) + KD_PD * (qd_r - qd) + DITHER_PD * xi)
        tau = np.clip(tau, -TAU_LIM, TAU_LIM)  # motor torque
        u = np.linalg.solve(a_mat, tau)  # logged input, tau = A u
        # the damping acts in the joint, outside the motor limit
        # it is held over the sample like u (exact for dt = 1 ms, one
        # physics step)
        applied = tau - d_mat @ qd

        if k >= n_warm:
            row = [t - n_warm * dt, *q, *qd, *u]
            if log_truth:
                row += list(g)
            rows.append(row)
        if k == n_warm + n_samples:
            break

        # torques are cleared after every physics step, so redo the same u
        # at each step
        for _ in range(substeps):
            robot.control_torques(applied)
            robot.sim.step(1, realtime=realtime)

    return np.array(rows), {'c': c, 'a': a, 'f': f, 'phi': phi}


def columns(log_truth):
    '''Columns to log dependent on `--log-truth`'''
    cols = ['t']
    for name in ('q', 'qd', 'u') + (('g',) if log_truth else ()):
        cols += [f'{name}{i + 1}' for i in range(N)]
    return cols


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=HelpFormatter
    )
    ap.add_argument('--out', default='data', help='output directory')
    ap.add_argument('--n-traj', type=int, default=100)
    ap.add_argument(
        '--duration', type=float, default=20.0, help='seconds per trajectory'
    )
    ap.add_argument(
        '--dt', type=float, default=DT_DEFAULT,
        help='sampling and control period [s], multiple of 0.001'
    )
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument(
        '--dither', type=float, default=0.0,
        help='std of a random acceleration added to the command [rad/s^2]'
    )
    ap.add_argument(
        '--log-truth', action='store_true',
        help='also log the PyBullet gravity vector g1..g6'
    )
    ap.add_argument(
        '--controller', choices=('model', 'pd'), default='model',
        help='data controller: computed torque with exact M and G, or PD'
    )
    ap.add_argument(
        '--inject-a', action='store_true',
        help='log u = A^-1 tau with the known A_INJECT instead of tau'
    )
    ap.add_argument(
        '--inject-d', action='store_true',
        help='add the known viscous damping -D_INJECT qd to the joints'
    )
    ap.add_argument(
        '--gui', action='store_true',
        help='show the simulation (runs in real time)'
    )
    args = ap.parse_args()
    # the control period must be a whole number of 1 ms physics steps
    steps = args.dt * 1000
    if steps < 1 or abs(steps - round(steps)) > 1e-6:
        ap.error('--dt must be a multiple of 0.001 s')

    out = Path(args.out)
    for split_name in ('train', 'test'):
        (out / split_name).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    robot = make_robot(args.gui)
    cols = columns(args.log_truth)
    a_mat = A_INJECT if args.inject_a else np.eye(N)
    d_mat = D_INJECT if args.inject_d else np.zeros((N, N))

    # split by trajectory (random but seeded)
    n_train = int(round(TRAIN_FRAC * args.n_traj))
    split = np.array(
        ['train'] * n_train + ['test'] * (args.n_traj - n_train)
    )
    rng.shuffle(split)

    meta = {
        'args': vars(args), 'columns': cols, 'dt': args.dt, 'warmup': WARMUP,
        'omega': OMEGA, 'controller': args.controller, 'trajectories': [],
        'inject': {'A': a_mat.tolist(), 'D': d_mat.tolist()},
        'constants': {
            'center_spread': CENTER_SPREAD, 'a_range': A_RANGE,
            'f_range': F_RANGE, 'v_ref_max': V_REF_MAX,
            'limit_margin': LIMIT_MARGIN, 'v_max': V_MAX,
            'floor_margin': FLOOR_MARGIN, 'home': HOME.tolist(),
            'kp_pd': KP_PD.tolist(), 'kd_pd': KD_PD.tolist(),
            'dither_pd': DITHER_PD.tolist(),
        },
    }
    rejected = {}
    for i in range(args.n_traj):
        for _ in range(MAX_RETRIES):
            rows, info = run_trajectory(
                robot, rng, args.duration, args.dt, args.log_truth, args.gui,
                args.dither, a_mat, d_mat, args.controller
            )
            if rows is not None:
                break
            rejected[info] = rejected.get(info, 0) + 1
        else:
            raise RuntimeError(
                f'trajectory {i}: no safe trajectory found: {rejected}'
            )

        path = out / split[i] / f'traj_{i:04d}.csv'
        np.savetxt(
            path, rows, delimiter=',', header=','.join(cols), comments='',
            fmt='%.8e'
        )
        meta['trajectories'].append({
            'file': str(path.relative_to(out)), 'n_rows': len(rows),
            **{key: val.tolist() for key, val in info.items()},
        })
        print(f'[{i + 1}/{args.n_traj}] {path} ({len(rows)} rows)')

    meta['rejected'] = rejected
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print('rejected trajectories by reason:', rejected or 'none')

    robot.sim.close()


if __name__ == '__main__':
    main()
