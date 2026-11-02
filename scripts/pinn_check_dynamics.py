'''
Compare the learned inertia, Coriolis and gravity terms with PyBullet

Evaluates the three parts of the Eq. 25 feedforward along the tracking
reference of pinn_track.py, for the learned model and for PyBullet:

    inertia   M qdd_ref         (mass matrix times reference acceleration)
    velocity  C qd_ref       (+ D qd_ref, which is zero in the simulation)
    gravity   G(q_ref)

Every learned term is multiplied by A_L(q)^-1, so the unknown common scale
P of the learned model cancels (A_L^-1 M_L = A^-1 M). The table gives, per
joint, the RMS of the true term and of the error, and how large the error is
relative to the total feedforward.

    python scripts/pinn_check_dynamics.py model.pkl
    python scripts/pinn_check_dynamics.py model.pkl --amp 0.5
'''
import argparse
import pickle
import jax
import jax.numpy as jnp
import numpy as np
from cli_utils import HelpFormatter
from pinn_control import make_robot
from pinn_model import (EPS, N, actuation_matrix, mass_matrix, required_force)
from pinn_track import reference
from ur_simulation.classic_control.robot_state.pybullet_robot_state import (
    JointType
)

# use every 10th reference sample
STRIDE = 10


def learned_terms(params, q, qd, qdd):
    '''A_L^-1 times the inertia, velocity and gravity parts, one sample'''
    zero = jnp.zeros(N)
    f_g = required_force(params, q, zero, zero)
    f_m = required_force(params, q, zero, qdd) - f_g
    f_v = required_force(params, q, qd, zero) - f_g
    a_inv = jnp.linalg.inv(actuation_matrix(params, q))
    return a_inv @ f_m, a_inv @ f_v, a_inv @ f_g


def joint_padding(robot):
    '''Angles of the joints after the arm (e.g. a blocked gripper)'''
    full = robot.robot_state.get_joint_angles(
        JointType.REVOLUTE | JointType.PRISMATIC
    )
    return list(full[N:])


def true_terms(robot, q, qd, qdd):
    '''PyBullet inertia, velocity and gravity parts, every sample'''
    pc = robot.sim.physics_client
    pad = joint_padding(robot)
    zeros = [0.0] * len(pad)
    zero_arm = [0.0] * N
    m_out, v_out, g_out = (np.zeros((len(q), N)) for _ in range(3))
    for k in range(len(q)):
        qf = list(q[k]) + pad
        grav = np.asarray(pc.calculateInverseDynamics(
            robot.robot_id, qf, zero_arm + zeros, zero_arm + zeros
        ))[:N]
        vel = np.asarray(pc.calculateInverseDynamics(
            robot.robot_id, qf, list(qd[k]) + zeros, zero_arm + zeros
        ))[:N]
        mass = np.asarray(pc.calculateMassMatrix(robot.robot_id, qf))
        m_out[k] = mass[:N, :N] @ qdd[k]
        v_out[k] = vel - grav
        g_out[k] = grav
    return m_out, v_out, g_out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=HelpFormatter
    )
    ap.add_argument('model', help='pickle file written by pinn_train.py')
    ap.add_argument(
        '--amp', type=float, default=0.25,
        help='reference half-amplitude [rad]'
    )
    ap.add_argument(
        '--freq', type=float, nargs=N,
        default=[0.15, 0.18, 0.21, 0.24, 0.27, 0.30],
        help='reference frequency per joint [Hz]'
    )
    ap.add_argument(
        '--duration', type=float, default=10.0, help='reference length [s]'
    )
    args = ap.parse_args()

    with open(args.model, 'rb') as fh:
        params = pickle.load(fh)['params']
    robot = make_robot(False)
    t = np.arange(0, args.duration, 0.001 * STRIDE)
    q, qd, qdd = reference(t, args.amp, np.array(args.freq))

    fn = jax.jit(jax.vmap(learned_terms, in_axes=(None, 0, 0, 0)))
    learned = [np.asarray(x) for x in fn(
        params, jnp.asarray(q), jnp.asarray(qd), jnp.asarray(qdd)
    )]
    true = true_terms(robot, q, qd, qdd)

    # inertia on the diagonal at the home pose: true against learned
    home = jnp.asarray(q[0])
    a_inv = jnp.linalg.inv(actuation_matrix(params, home))
    m_true = np.asarray(robot.sim.physics_client.calculateMassMatrix(
        robot.robot_id, list(q[0]) + joint_padding(robot)
    ))[:N, :N]
    m_learned = np.asarray(a_inv @ mass_matrix(params, home))
    print(
        'inertia M_ii at the home pose [kg m^2] (epsilon in the model: '
        f'{EPS:g})'
    )
    print('joint'.ljust(6) + 'true'.rjust(12) + 'learned'.rjust(12))
    for i in range(N):
        print(f'q{i + 1:<5}{m_true[i, i]:>12.5f}{m_learned[i, i]:>12.5f}')

    total = sum(true)
    names = ('inertia', 'velocity', 'gravity')
    print(
        '\nfeedforward terms along the reference [Nm]: RMS true / RMS error'
    )
    print(
        'joint'.ljust(6) + ''.join(n.rjust(20) for n in names)
        + 'total error'.rjust(14)
    )
    for i in range(N):
        line = f'q{i + 1:<5}'
        for tr, le in zip(true, learned):
            rms_true = np.sqrt((tr[:, i] ** 2).mean())
            rms_err = np.sqrt(((le[:, i] - tr[:, i]) ** 2).mean())
            line += f'{rms_true:>10.3f}/{rms_err:<9.3f}'
        err_total = sum(le - tr for tr, le in zip(true, learned))[:, i]
        rms_total = np.sqrt((total[:, i] ** 2).mean())
        rel = np.sqrt((err_total ** 2).mean()) / max(rms_total, 1e-9)
        line += f'{rel:>13.0%}'
        print(line)


if __name__ == '__main__':
    main()
