'''
Impedance control (impedance_control.py) in the scene of pinn_contact.py

Runs the task-space impedance controller of impedance_control.py on the
same experiment as `pinn_contact.py --scene box`, so the model-based and
the learned (model-free) interaction controllers can be compared:

- the same fixed box (8 cm cube at (0.4, 0.1, 0.1), mass 0),
- the start pose of impedance_control.py (the PINN start pose HOME has
  the tool pointing down, an Euler pitch of pi/2, where the analytic
  Jacobian is singular, condition number about 1e20),
- the same reference: from the start position to 0.10 m above the box top
  in --ramp seconds, then down to --push metres below the top in another
  --ramp seconds; the orientation is held at the start orientation,
- the same measurements: contact force on the box, commanded spring force
  K_p,z (x_d,z - x_z), and a .csv log with the columns of pinn_contact.py

The control law and its helpers are unchanged from impedance_control.py
Three things are different here:
- the contact wrench estimate uses its given analytic Jacobian,
- the orientation error is wrapped to [-pi, pi],
- the first J_dot is zero instead of J / dt

    python scripts/impedance_contact.py
    python scripts/impedance_contact.py --kp-pos 300 --kd-pos 40 --gui
    python scripts/impedance_contact.py --log contact.csv
'''
import argparse
import functools
import numpy as np
import pybullet as p
import ur_simulation.classic_control.robots.ur7e as ur7e_mod
from ur_simulation.pybullet import PyBullet
from cli_utils import HelpFormatter
from impedance_control import (
    compute_force_in_workspace, compute_j_dot, compute_workspace_matrices,
    compute_y, geometric_to_analytic_jacobian, torques,
)

N = 6
# start pose of impedance_control.py
START = np.array([1.57, -1.7, 2.4, -1.57, -1.57, -1.57])
# as in impedance_control.py
BOX_CENTRE = np.array([0.4, 0.1, 0.1])
BOX_HALF = 0.04


def make_robot(gui):
    '''UR7e, 1 kHz physics, with the fixed box'''
    if not gui:
        # headless, as make_robot in pinn_control.py
        ur7e_mod.PyBullet = functools.partial(
            PyBullet, render_mode='rgb_array', renderer='Tiny'
        )
    robot = ur7e_mod.UR7e(
        block_gripper=True, neutral_joints=START, n_substeps=1,
        step_frequency=1000,
    )
    robot.sim.create_plane(0)
    robot.sim.create_box(
        body_name='contact_object', half_extents=np.full(3, BOX_HALF),
        mass=0.0, position=BOX_CENTRE,
        rgba_color=np.array([0.95, 0.55, 0.15, 1.0]), lateral_friction=0.3,
    )
    return robot


def smoothstep(tau):
    '''
    0 to 1 with zero slope at both ends, and its first two derivatives with
    respect to tau
    '''
    tau = np.clip(tau, 0.0, 1.0)
    return (tau * tau * (3 - 2 * tau), 6 * tau * (1 - tau), 6 - 12 * tau)


def reference(t, waypoints, ramp):
    '''Position, velocity and acceleration of the piecewise reference'''
    k = min(int(t // ramp), len(waypoints) - 2)
    tau = t / ramp - k
    s, ds, dds = smoothstep(tau)
    if tau >= 1.0:
        ds = dds = 0.0
    step = waypoints[k + 1] - waypoints[k]
    return (waypoints[k] + step * s, step * ds / ramp, step * dds / ramp ** 2)


def contact_wrench(robot, tau_applied, j_a):
    '''End-effector wrench equivalent to the external joint torques'''
    tau_ext = np.asarray(
        robot.robot_state.get_external_joint_forces(tau_applied), dtype=float
    )
    return np.linalg.lstsq(j_a.T, tau_ext, rcond=None)[0]


def contact_force(robot, box_id):
    '''Normal force between the robot and the box, summed over links'''
    pc = robot.sim.physics_client
    return sum(c[9] for c in pc.getContactPoints(
        bodyA=robot.robot_id, bodyB=box_id
    ))


def pose(robot):
    '''End-effector position and Euler angles'''
    return np.concatenate((
        np.asarray(robot.robot_state.get_end_effector_position(), dtype=float),
        np.asarray(p.getEulerFromQuaternion(
            robot.robot_state.get_end_effector_orientation()
        ), dtype=float),
    ))


def row(label, text):
    '''Print a label padded to a fixed width, then the text'''
    print(f'{label:<27}{text}')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=HelpFormatter
    )
    ap.add_argument(
        '--kp-pos', type=float, default=100.0,
        help='translational stiffness K_p [N/m]'
    )
    ap.add_argument(
        '--kd-pos', type=float, default=10.0,
        help='translational damping K_d [N s/m]'
    )
    ap.add_argument(
        '--kp-rot', type=float, default=1.0,
        help='rotational stiffness [Nm/rad]'
    )
    ap.add_argument(
        '--kd-rot', type=float, default=1.0,
        help='rotational damping [Nm s/rad]'
    )
    ap.add_argument(
        '--gap', type=float, default=0.10,
        help='height above the box top after the first ramp [m]'
    )
    ap.add_argument(
        '--push', type=float, default=0.05,
        help='how far below the box top the reference ends [m]'
    )
    ap.add_argument(
        '--ramp', type=float, default=4.0, help='seconds per reference segment'
    )
    ap.add_argument(
        '--duration', type=float, default=14.0, help='simulated seconds'
    )
    ap.add_argument(
        '--no-fc', action='store_true',
        help='leave out the estimated contact wrench F_c'
    )
    ap.add_argument(
        '--log', default=None,
        help='write a .csv with the columns of pinn_contact.py'
    )
    ap.add_argument(
        '--gui', action='store_true',
        help='show the simulation (runs in real time)'
    )
    args = ap.parse_args()

    robot = make_robot(args.gui)
    box_id = robot.sim.bodies_idx['contact_object']
    dt = robot.sim.dt
    m_d = np.eye(N)
    k_p = np.diag([args.kp_pos] * 3 + [args.kp_rot] * 3)
    k_d = np.diag([args.kd_pos] * 3 + [args.kd_rot] * 3)

    x0 = pose(robot)
    surface = BOX_CENTRE[2] + BOX_HALF
    above = np.concatenate((BOX_CENTRE, x0[3:]))
    above[2] = surface + args.gap
    target = above.copy()
    target[2] = surface - args.push
    waypoints = [x0, above, target]
    print('start x :', np.round(x0[:3], 3))
    print(f'box surface along z: {surface:.3f} m')

    tau_applied = np.zeros(N)
    previous_j = None
    n_steps = int(round(args.duration / dt))
    # z, contact force, spring force
    out = np.zeros((n_steps, 3))
    log, peak = [], (0.0, 0.0)
    for k in range(n_steps):
        t = k * dt
        x = pose(robot)
        j_a = geometric_to_analytic_jacobian(
            robot.robot_model.get_jacobian(), x[3:]
        )
        if previous_j is None:
            # first J_dot is zero
            previous_j = j_a.copy()
        j_dot, previous_j = compute_j_dot(j_a, previous_j, robot)
        qd = np.asarray(robot.robot_state.get_joint_velocities(), dtype=float)
        x_dot = j_a @ qd

        x_d, xd_d, xdd_d = reference(t, waypoints, args.ramp)
        e = x_d - x
        # angles wrapped
        e[3:] = (e[3:] + np.pi) % (2 * np.pi) - np.pi
        e_dot = xd_d - x_dot
        f_c = (
            contact_wrench(robot, tau_applied, j_a)
            if k and not args.no_fc else np.zeros(N)
        )

        m_hat, c_hat_xdot, g_hat = compute_workspace_matrices(
            robot.robot_model.get_dynamics(), j_a, j_dot, qd
        )
        y = compute_y(xdd_d, m_d, k_d, k_p, e_dot, e, f_c)
        force = compute_force_in_workspace(m_hat, y, c_hat_xdot, g_hat, f_c)
        tau_applied = np.clip(
            torques(j_a, force), -robot.arm_joint_forces,
            robot.arm_joint_forces
        )

        f_contact = contact_force(robot, box_id)
        f_spring = args.kp_pos * e[2]
        peak = max(peak, (f_contact, t))
        out[k] = (x[2], f_contact, f_spring)
        if args.log and k % 10 == 0:
            log.append([
                t, *x[:3], *x_d[:3], f_contact, f_spring, *tau_applied
            ])
        robot.control_torques(tau_applied)
        robot.sim.step()

    tail = out[-int(round(1.0 / dt)):]
    z, f_c, f_s = tail.mean(axis=0)
    f_s = abs(f_s)
    row('peak contact force:', f'{peak[0]:8.2f} N at t = {peak[1]:.3f} s')
    row('final contact force:', f'{f_c:8.2f} N (last second)')
    row('final spring force:', (
        f'{f_s:8.2f} N  (K_p,z times the reference below the end effector)'
    ))
    row('final height vs surface:', f'{z - surface:+8.4f} m')
    row('measured excess:', f'{f_c - f_s:+8.2f} N (contact - spring)')
    row('final orientation error:', (
        f'{np.round(np.abs(e[3:]).max(), 4)} rad (largest Euler angle)'
    ))

    if args.log:
        cols = (
            ['t'] + [f'x{a}' for a in 'xyz'] + [f'xd{a}' for a in 'xyz']
            + ['f_contact', 'f_spring'] + [f'u{i + 1}' for i in range(N)]
        )
        np.savetxt(
            args.log, np.array(log), delimiter=',',
            header=','.join(cols), comments='', fmt='%.8e'
        )
        print(f'saved {args.log}')


if __name__ == '__main__':
    main()
