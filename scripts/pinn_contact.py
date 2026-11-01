'''
Contact with a table: Cartesian impedance control with a learned gravity term

The end effector is pushed down onto a table. The controller is a
gravity-compensated Cartesian impedance law for the end-effector position x:

    u = J(q)^T [K_x (x_d - x) - D_x xd] + g(q)
        + K_w (q_home - q) - D_w qd                  (wrist joints only)

K_x is a virtual spring and D_x a virtual damper, so the end effector acts
like a mass on a spring instead of a stiff position servo. The gravity
term g(q) is the learned A_L^-1 G_L(q) (--gravity learned) or PyBullet's
exact vector (--gravity true). The reference x_d moves down along --axis,
passes through the table surface and stops --push metres below it, so in
steady state the table carries the spring force K_x (x_d - x).

The script reports the measured contact force next to that spring force.
Without gravity error they agree. The difference is the effect of the
model error on the contact.

Scenes (--scene):
    table  a thin "table" --gap below the start pose (the default)
    box    the 8 cm box of impedance_control.py at (0.4, 0.1, 0.1), fixed
           (--box-mass 0): the reference first moves --gap above the box
           top in --ramp seconds, then pushes down through it in another
           --ramp seconds, so the two interaction controllers share a scene

Diagnostics: the contact force is summed over every robot link that touches
the table, and the script prints which links those are at the first contact,
at the peak and at the end (the default --table-half 0.1 keeps the table
small, so that only the end effector can reach it; 0.4 also hits the upper
arm). It also predicts the excess contact force
(contact minus spring) from statics, split into the part due to the gravity
error and the part due to the wrist term, and prints it next to the measured
excess.

    python scripts/pinn_contact.py model.pkl
    python scripts/pinn_contact.py model.pkl --gravity true
    python scripts/pinn_contact.py model.pkl --kx 1000 --log contact.csv
    python scripts/pinn_contact.py model.pkl --table-half 0.4
'''
import argparse
import pickle
import numpy as np
from cli_utils import HelpFormatter
from pinn_control import HOME, learned_gravity_fn, make_robot
from pinn_model import N

AXES = {'x': 0, 'y': 1, 'z': 2}
# the object of impedance_control.py (centre and half size [m])
BOX_CENTRE = np.array([0.4, 0.1, 0.1])
BOX_HALF = 0.04
WRIST = np.array([0, 0, 0, 1, 1, 1], dtype=float)  # joints held by K_w


def smoothstep(s):
    '''0 to 1 with zero slope at both ends'''
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3 - 2 * s)


def contact_by_link(robot, wall_id):
    '''Normal force per robot link touching the table, {link: N}'''
    pc = robot.sim.physics_client
    forces = {}
    for point in pc.getContactPoints(bodyA=robot.robot_id, bodyB=wall_id):
        forces[point[3]] = forces.get(point[3], 0.0) + point[9]
    return forces


def link_name(robot, idx):
    '''Readable name of a robot link'''
    if idx < 0:
        return 'base'
    pc = robot.sim.physics_client
    info = pc.getJointInfo(robot.robot_id, idx)
    return f'{idx} ({info[12].decode()})'


def fmt_links(robot, links):
    '''Text of {link: force} pairs'''
    if not links:
        return 'none'
    return ', '.join(
        f'{link_name(robot, i)} {f:.1f} N' for i, f in sorted(links.items())
    )


def ramp_reference(waypoints, ramp):
    '''x_d(t) through the waypoints, one smooth ramp of `ramp` s each'''
    def ref(t):
        k = min(int(t // ramp), len(waypoints) - 2)
        s = smoothstep(t / ramp - k)
        return waypoints[k] + (waypoints[k + 1] - waypoints[k]) * s
    return ref


def run_contact(
    robot, wall_id, gravity_fn, reference, axis, kx, dx, kw,
    dw, duration, realtime, log
):
    '''
    Run the controller. Returns the log array and a contact summary.

    The array has one row per step: position along the axis, contact force
    and spring force. The summary holds the first contact, the peak and the
    final contact, each with the force per link.
    '''
    dt = robot.sim.dt
    n_steps = int(round(duration / dt))
    # axis position, contact, spring force
    out = np.zeros((n_steps, 3))
    first, peak, by_link = None, (0.0, 0.0, {}), {}
    for k in range(n_steps):
        t = k * dt
        x_d = reference(t)
        q = np.asarray(robot.robot_state.get_joint_angles(), dtype=float)
        qd = np.asarray(
            robot.robot_state.get_joint_velocities(), dtype=float
        )
        x = np.asarray(
            robot.robot_state.get_end_effector_position(), dtype=float
        )
        jac = np.asarray(robot.robot_model.get_jacobian())[:3]
        force = kx * (x_d - x) - dx * (jac @ qd)
        u = (
            jac.T @ force + gravity_fn(q)
            + WRIST * (kw * (HOME - q) - dw * qd)
        )
        by_link = contact_by_link(robot, wall_id)
        f_contact = sum(by_link.values())
        if by_link and first is None:
            first = (t, x[axis], dict(by_link))
        if f_contact > peak[0]:
            peak = (f_contact, t, dict(by_link))
        out[k] = (x[axis], f_contact, kx * (x_d[axis] - x[axis]))
        if log is not None and k % 10 == 0:
            log.append([
                t, *x, *x_d, f_contact, kx * (x_d[axis] - x[axis]), *u
            ])
        robot.control_torques(u)
        robot.sim.step(1, realtime=realtime)
    return out, {'first': first, 'peak': peak, 'final': by_link}


def predicted_excess(robot, gravity_fn, kw, dw, axis):
    '''
    Excess contact force (contact minus spring) predicted from statics

    At rest J^T (F_s + f_t) = G - g_hat - w, with F_s the commanded
    Cartesian force, f_t the force of the table on the end effector and w
    the wrist term. With y = J^-T (dG + w) and dG = g_hat - G this gives
    contact - spring = -y[axis]. The contact is assumed to act at the
    end-effector frame (any moment is absorbed by the angular rows of J).

    Returns the part due to the gravity error, the part due to the wrist
    term and their sum [N].
    '''
    q = np.asarray(robot.robot_state.get_joint_angles(), dtype=float)
    qd = np.asarray(robot.robot_state.get_joint_velocities(), dtype=float)
    jac = np.asarray(robot.robot_model.get_jacobian(), dtype=float)
    g_true = np.asarray(
        robot.robot_model.get_dynamics().gravity_vector, dtype=float
    )
    d_g = gravity_fn(q) - g_true
    wrist = WRIST * (kw * (HOME - q) - dw * qd)
    return tuple(
        -np.linalg.solve(jac.T, v)[axis] for v in (d_g, wrist, d_g + wrist)
    )


def row(label, text):
    '''Print a label padded to a fixed width, then the text'''
    print(f'{label:<27}{text}')


def report(robot, out, surface, dt, info, pred):
    '''Peak and final contact force, penetration and the spring force'''
    # last second
    tail = out[-int(round(1.0 / dt)):]
    pos, f_c, f_s = tail.mean(axis=0)
    f_s = abs(f_s)
    f_peak, t_peak, peak_links = info['peak']
    first, final = info['first'], info['final']
    row('peak contact force:', f'{f_peak:8.2f} N at t = {t_peak:.3f} s')
    row('  links at the peak:', fmt_links(robot, peak_links))
    row('final contact force:', f'{f_c:8.2f} N (last second)')
    row('final spring force:', (
        f'{f_s:8.2f} N  (K_x times the reference below the end effector)'
    ))
    row('final height vs surface:', f'{pos - surface:+8.4f} m')
    row('measured excess:', f'{f_c - f_s:+8.2f} N (contact - spring)')
    g_part, w_part, total = pred
    row('predicted excess:', (
        f'{total:+8.2f} N = {g_part:+.2f} (gravity error) '
        f'{w_part:+.2f} (wrist term)'
    ))
    row('end-effector link:', robot.ee_link)
    if first is not None:
        row('first contact:', (
            f't = {first[0]:.3f} s, end effector at {first[1]:.3f} m'
        ))
        row('  links at first contact:', fmt_links(robot, first[2]))
    row('links in contact at end:', fmt_links(robot, final))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=HelpFormatter
    )
    ap.add_argument('model', help='pickle file written by pinn_train.py')
    ap.add_argument(
        '--gravity', choices=('learned', 'true'), default='learned',
        help='gravity term: the learned model or PyBullet'
    )
    ap.add_argument(
        '--scene', choices=('table', 'box'), default='table',
        help='what is pushed: a table below the start pose or the box'
    )
    ap.add_argument(
        '--box-mass', type=float, default=0.0,
        help='mass of the box [kg], 0 keeps it fixed'
    )
    ap.add_argument(
        '--axis', choices=tuple(AXES), default='z',
        help='world axis of the push (down for z)'
    )
    ap.add_argument(
        '--gap', type=float, default=0.10,
        help='distance from the start pose to the table [m]'
    )
    ap.add_argument(
        '--push', type=float, default=0.05,
        help='how far below the surface the reference ends [m]'
    )
    ap.add_argument(
        '--table-half', type=float, default=0.1,
        help=(
            'half width of the table in the two other axes '
            '[m], small values keep other links off the table'
        )
    )
    ap.add_argument(
        '--kx', type=float, default=300.0, help='virtual stiffness [N/m]'
    )
    ap.add_argument(
        '--dx', type=float, default=40.0, help='virtual damping [N s/m]'
    )
    ap.add_argument(
        '--kw', type=float, nargs=N, default=[0, 0, 0, 5, 5, 20],
        help='wrist joint stiffness [Nm/rad] (zero elsewhere)'
    )
    ap.add_argument(
        '--dw', type=float, nargs=N, default=[0, 0, 0, 5, 2, 0.3],
        help='wrist joint damping [Nm s/rad] (zero elsewhere)'
    )
    ap.add_argument(
        '--ramp', type=float, default=4.0,
        help='seconds for the reference to reach its end'
    )
    ap.add_argument(
        '--duration', type=float, default=None,
        help='simulated seconds (default 10 for table, 14 for box)'
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
    axis = AXES[args.axis]
    x0 = np.asarray(robot.robot_state.get_end_effector_position(), dtype=float)
    if args.scene == 'box':
        if args.axis != 'z':
            ap.error('--scene box pushes down, use --axis z')
        surface = BOX_CENTRE[2] + BOX_HALF
        above = BOX_CENTRE.copy()
        above[2] = surface + args.gap
        x_target = above.copy()
        x_target[2] = surface - args.push
        waypoints = [x0, above, x_target]
        centre, half, mass = BOX_CENTRE, np.full(3, BOX_HALF), args.box_mass
        color = np.array([0.95, 0.55, 0.15, 1.0])
    else:
        surface = x0[axis] - args.gap
        if args.axis == 'z' and surface < 0.02:
            raise SystemExit(
                f'start height {x0[2]:.3f} m leaves no room for '
                f'a table {args.gap} m below; lower --gap'
            )
        x_target = x0.copy()
        x_target[axis] = surface - args.push
        waypoints = [x0, x_target]
        centre = x0.copy()
        # table of 4 cm thickness
        centre[axis] = surface - 0.02
        half = np.full(3, args.table_half)
        half[axis] = 0.02
        mass, color = 0.0, np.array([0.6, 0.6, 0.6, 1.0])
    duration = args.duration
    if duration is None:
        duration = 10.0 if args.scene == 'table' else 14.0
    print('start x :', np.round(x0, 3))
    print(f'{args.scene} surface along {args.axis}: {surface:.3f} m')

    robot.sim.create_box(
        'table', half_extents=half, mass=mass, position=centre,
        rgba_color=color, lateral_friction=0.3,
    )
    wall_id = robot.sim.bodies_idx['table']

    if args.gravity == 'learned':
        gravity_fn = learned_gravity_fn(params)
    else:
        def gravity_fn(q):
            return np.asarray(robot.robot_model.get_dynamics().gravity_vector)

    kw, dw = np.array(args.kw), np.array(args.dw)
    log = [] if args.log else None
    out, info = run_contact(
        robot, wall_id, gravity_fn, ramp_reference(waypoints, args.ramp),
        axis, args.kx, args.dx, kw, dw, duration, args.gui, log
    )
    pred = predicted_excess(robot, gravity_fn, kw, dw, axis)
    report(robot, out, surface, robot.sim.dt, info, pred)

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
