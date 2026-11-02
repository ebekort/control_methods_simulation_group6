'''
Joint-space trajectory tracking with the learned model (Eq. 25)

The reference is a smooth joint-space motion that starts at rest at the home
pose of pd+gravity_control.py:

    q_ref(t) = HOME + amp (1 - cos(2 pi f t))      (per joint f)

The controller is Eq. 25, with the PD gains in real torque units (as in
pinn_control.py, K_P(Eq. 25) = A_L K_P):

    u = A_L(q)^-1 [M_L qdd_ref + C_L qd_ref + D_L qd_ref + G_L]_(q_ref)
        + K_P (q_ref - q) + K_D (qd_ref - qd)

The feedforward bracket depends only on the reference, so it is computed for
the whole trajectory before the run. --ff chooses what is compared:

    learned       Eq. 25 with the learned model (the paper's controller)
    true          the same feedforward from PyBullet inverse dynamics
    gravity       no feedforward: PD + learned gravity at the current q
    gravity-true  no feedforward: PD + PyBullet gravity at the current q

    python scripts/pinn_track.py model.pkl
    python scripts/pinn_track.py model.pkl --ff true
    python scripts/pinn_track.py model.pkl --ff gravity --log run.csv
'''
import argparse
import pickle
import jax
import jax.numpy as jnp
import numpy as np
from cli_utils import HelpFormatter
from pinn_control import HOME, learned_gravity_fn, make_robot
from pinn_model import N, actuation_matrix, required_force

FF_CHOICES = ('learned', 'true', 'gravity', 'gravity-true')


def reference(t, amp, freq):
    '''Position, velocity and acceleration of the reference at times t'''
    w = 2 * np.pi * freq
    s = np.outer(t, w)
    q = HOME + amp * (1 - np.cos(s))
    qd = amp * w * np.sin(s)
    qdd = amp * w**2 * np.cos(s)
    return q, qd, qdd


def learned_feedforward(params, q, qd, qdd):
    '''Rows of the bracket in Eq. 25 for every time step (batched)'''
    fn = jax.jit(jax.vmap(required_force, in_axes=(None, 0, 0, 0)))
    return np.asarray(fn(
        params, jnp.asarray(q), jnp.asarray(qd), jnp.asarray(qdd)
    ))


def true_feedforward(robot, q, qd, qdd):
    '''PyBullet inverse dynamics M qdd + C qd + G at the reference'''
    from ur_simulation.classic_control.robot_state.pybullet_robot_state \
        import JointType

    pc = robot.sim.physics_client
    full = robot.robot_state.get_joint_angles(
        JointType.REVOLUTE | JointType.PRISMATIC
    )
    extra = len(full) - N  # e.g. blocked gripper joints
    out = np.zeros((len(q), N))
    for k in range(len(q)):
        out[k] = pc.calculateInverseDynamics(
            robot.robot_id,
            list(q[k]) + list(full[N:]),
            list(qd[k]) + [0.0] * extra,
            list(qdd[k]) + [0.0] * extra,
        )[:N]
    return out


def actuation_solver(params):
    '''(q, force) -> A_L(q)^-1 force.'''

    @jax.jit
    def fn(q, force):
        return jnp.linalg.solve(actuation_matrix(params, q), force)

    return lambda q, force: np.asarray(fn(q, force))


def run_tracking(
    robot, ff_mode, params, kp, kd, duration, amp, freq, realtime, log
):
    '''Run the controller. Returns the tracking error of every step'''
    dt = robot.sim.dt
    n_steps = int(round(duration / dt))
    t = np.arange(n_steps) * dt
    q_ref, qd_ref, qdd_ref = reference(t, amp, freq)

    solve = None
    if ff_mode == 'learned':
        ff = learned_feedforward(params, q_ref, qd_ref, qdd_ref)
        solve = actuation_solver(params)
    elif ff_mode == 'true':
        ff = true_feedforward(robot, q_ref, qd_ref, qdd_ref)
    elif ff_mode == 'gravity':
        gravity_fn = learned_gravity_fn(params)
    errors = np.zeros((n_steps, N))

    for k in range(n_steps):
        q = np.asarray(robot.robot_state.get_joint_angles(), dtype=float)
        qd = np.asarray(robot.robot_state.get_joint_velocities(), dtype=float)
        e = (q_ref[k] - q + np.pi) % (2 * np.pi) - np.pi
        feedback = kp * e + kd * (qd_ref[k] - qd)
        if ff_mode == 'learned':
            u = solve(q, ff[k]) + feedback
        elif ff_mode == 'true':
            u = ff[k] + feedback
        elif ff_mode == 'gravity':
            u = gravity_fn(q) + feedback
        else:
            g = robot.robot_model.get_dynamics().gravity_vector
            u = np.asarray(g) + feedback
        errors[k] = e
        if log is not None and k % 10 == 0:
            log.append([t[k], *q, *q_ref[k], *e, *u])
        robot.control_torques(u)
        robot.sim.step(1, realtime=realtime)
    return errors


def report(errors):
    '''RMS and peak tracking error per joint, ignoring the first second'''
    skip = min(1000, len(errors) // 5)
    err = errors[skip:]
    rms = np.sqrt((err**2).mean(axis=0))
    peak = np.abs(err).max(axis=0)
    print(f'RMS error over all joints: {np.sqrt((err**2).mean()):.4f} rad')
    print('joint'.ljust(6) + 'RMS [rad]'.rjust(12) + 'max [rad]'.rjust(12))
    for i in range(N):
        print(f'q{i + 1:<5}{rms[i]:>12.4f}{peak[i]:>12.4f}')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=HelpFormatter
    )
    ap.add_argument('model', help='pickle file written by pinn_train.py')
    ap.add_argument(
        '--ff', choices=FF_CHOICES, default='learned',
        help='feedforward / gravity term, see above'
    )
    ap.add_argument(
        '--kp', type=float, nargs=N, default=[100, 100, 100, 5, 5, 20],
        help='proportional gains [Nm/rad]'
    )
    ap.add_argument(
        '--kd', type=float, nargs=N, default=[40, 40, 40, 5, 2, 0.3],
        help='derivative gains [Nm s/rad]'
    )
    ap.add_argument(
        '--amp', type=float, default=0.25,
        help='reference half-amplitude, joints move 0..2 amp'
        ' [rad]'
    )
    ap.add_argument(
        '--freq', type=float, nargs=N,
        default=[0.15, 0.18, 0.21, 0.24, 0.27, 0.30],
        help='reference frequency per joint [Hz]'
    )
    ap.add_argument(
        '--duration', type=float, default=10.0, help='simulated seconds'
    )
    ap.add_argument(
        '--log', default=None, help='write a .csv with the time series'
    )
    ap.add_argument(
        '--gui', action='store_true',
        help='show the simulation (runs in real time)'
    )
    args = ap.parse_args()

    with open(args.model, 'rb') as fh:
        params = pickle.load(fh)['params']
    robot = make_robot(args.gui)
    log = [] if args.log else None
    errors = run_tracking(
        robot, args.ff, params, np.array(args.kp), np.array(args.kd),
        args.duration, args.amp, np.array(args.freq), args.gui, log
    )
    report(errors)

    if args.log:
        cols = ['t'] + [
            f'{p}{i + 1}' for p in ('q', 'qref', 'e', 'u') for i in range(N)
        ]
        np.savetxt(
            args.log, np.array(log), delimiter=',',
            header=','.join(cols), comments='', fmt='%.8e'
        )
        print(f'saved {args.log}')


if __name__ == '__main__':
    main()
