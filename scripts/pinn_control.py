'''
Regulation to a target pose with the learned gravity term (Eq. 13)

The controller is PD plus gravity compensation, as in pd+gravity_control.py,
with the gravity term taken from the learned model:

    u = A_L(q)^-1 G_L(q) + K_P (q_ref - q) - K_D qd

The gains K_P and K_D are in real torque units. In the paper's form of Eq. 13
this means K_P(Eq. 13) = A_L K_P, which keeps the gains independent of the
arbitrary scale of the learned model. With --gravity true the PyBullet
gravity vector is used instead, as a reference.

At rest the arm settles where K_P e = G - G_hat, so the measured offset e is
compared with the prediction (G - G_hat) / K_P (Eq. 16).

    python scripts/pinn_control.py model.pkl
    python scripts/pinn_control.py model.pkl --gravity true --gui
    python scripts/pinn_control.py model.pkl --log run.csv
'''
import argparse
import functools
import pickle
import jax
import jax.numpy as jnp
import numpy as np
from cli_utils import HelpFormatter
from pinn_model import N, actuation_matrix, gravity
import ur_simulation.classic_control.robots.ur7e as ur7e_mod
from ur_simulation.pybullet import PyBullet

# pose and target of pd+gravity_control.py
HOME = np.array([1.57, -1.7, 1.7, -1.57, -1.57, -1.57])
TARGET_POS = [0.45, 0.45, 0.45]


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


def learned_gravity_fn(params):
    '''q -> A_L(q)^-1 G_L(q), the gravity torque the model asks for'''
    @jax.jit
    def fn(q):
        return jnp.linalg.solve(
            actuation_matrix(params, q), gravity(params, q)
        )

    return lambda q: np.asarray(fn(q))


def run_control(robot, gravity_fn, q_ref, kp, kd, duration, realtime, log):
    '''
    Run the controller. Returns final q and the gravity torques at the end.

    log is a list that receives one row (t, q, e, u, g_hat, g_true) every
    10 ms, or None.
    '''
    dt = robot.sim.dt
    for k in range(int(round(duration / dt))):
        q = np.asarray(robot.robot_state.get_joint_angles(), dtype=float)
        qd = np.asarray(robot.robot_state.get_joint_velocities(), dtype=float)
        e = (q_ref - q + np.pi) % (2 * np.pi) - np.pi
        g_hat = gravity_fn(q)
        u = g_hat + kp * e - kd * qd
        if log is not None and k % 10 == 0:
            dyn = robot.robot_model.get_dynamics()
            g_true = np.asarray(dyn.gravity_vector)
            log.append([k * dt, *q, *e, *u, *g_hat, *g_true])
        robot.control_torques(u)
        robot.sim.step(1, realtime=realtime)

    q = np.asarray(robot.robot_state.get_joint_angles(), dtype=float)
    g_true = np.asarray(robot.robot_model.get_dynamics().gravity_vector)
    return q, gravity_fn(q), g_true


def report(q, q_ref, g_hat, g_true, kp):
    e = (q_ref - q + np.pi) % (2 * np.pi) - np.pi
    predicted = (g_true - g_hat) / kp
    print(f'final error norm: {np.linalg.norm(e):.4f} rad')
    print(
        'joint'.ljust(6) + 'e [rad]'.rjust(10) + 'predicted'.rjust(11)
        + 'G_hat'.rjust(10) + 'G true'.rjust(10)
    )
    for i in range(N):
        print(
            f'q{i + 1:<5}{e[i]:>10.4f}{predicted[i]:>11.4f}'
            f'{g_hat[i]:>10.2f}{g_true[i]:>10.2f}'
        )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=HelpFormatter
    )
    ap.add_argument('model', help='pickle file written by pinn_train.py')
    ap.add_argument(
        '--gravity', choices=('learned', 'true'), default='learned',
        help='gravity term: the learned model or PyBullet (reference)'
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
    kp, kd = np.array(args.kp), np.array(args.kd)

    robot = make_robot(args.gui)
    target_rot = robot.robot_state.get_end_effector_orientation()
    q_ref = np.asarray(
        robot.robot_model.get_inverse_kinematics(TARGET_POS, target_rot)
    )
    # nearest equivalent angle to the start pose (IK may return +2 pi)
    q0 = np.asarray(robot.robot_state.get_joint_angles(), dtype=float)
    q_ref = q0 + (q_ref - q0 + np.pi) % (2 * np.pi) - np.pi
    print('start q :', np.round(q0, 3))
    print('target q:', np.round(q_ref, 3))
    if args.gravity == 'learned':
        gravity_fn = learned_gravity_fn(params)
    else:
        def gravity_fn(q):
            return np.asarray(robot.robot_model.get_dynamics().gravity_vector)

    log = [] if args.log else None
    q, g_hat, g_true = run_control(
        robot, gravity_fn, q_ref, kp, kd, args.duration, args.gui, log
    )
    report(q, q_ref, g_hat, g_true, kp)

    if args.log:
        cols = ['t'] + [
            f'{p}{i + 1}' for p in ('q', 'e', 'u', 'ghat', 'g')
            for i in range(N)
        ]
        np.savetxt(
            args.log, np.array(log), delimiter=',', header=','.join(cols),
            comments='', fmt='%.8e'
        )
        print(f'saved {args.log}')


if __name__ == '__main__':
    main()
