import argparse
import time
import numpy as np
import pybullet as p
from ur_simulation.classic_control.robots.ur7e import UR7e
from ur_simulation.pybullet import PyBullet
from ur_simulation.classic_control.robot_state.pybullet_robot_state import JointType


def create_scene():
    # initializing robot scene with start position
    init_joint_angles = np.array([1.57, -1.7, 2.4, -1.57, -1.57, -1.57])
    robot = UR7e(
        block_gripper=True,
        neutral_joints=init_joint_angles,
    )
    robot.sim.create_plane(0)

    robot.sim.create_box(
        body_name="contact_object",
        half_extents=np.array([0.04, 0.04, 0.04]),
        mass=0.15,
        position= np.array([0.4, 0.1, 0.1]),
        rgba_color=np.array([0.95, 0.55, 0.15, 1.0]),
    )
    return robot

def torques(jacobian, F):
    """Compute joint torques from the Jacobian and a force vector."""
    J = np.asarray(jacobian, dtype=float)
    F = np.asarray(F, dtype=float)
    tau = J.T @ F
    return tau

def compute_force_in_workspace(M_hat, y, c_hat_xdot, g_hat, F_c):
    """Compute the force in the workspace using the impedance control law."""
    M_hat = np.asarray(M_hat, dtype=float)
    y = np.asarray(y, dtype=float)
    c_hat_xdot = np.asarray(c_hat_xdot, dtype=float)
    g_hat = np.asarray(g_hat, dtype=float)
    F_c = np.asarray(F_c, dtype=float)

    # Impedance control law: F = M_hat * y + C_hat * x_dot + g_hat - F_c
    F = M_hat @ y + c_hat_xdot + g_hat - F_c
    return F

def compute_y(x_doubledot_d, M_d, K_d, K_p, e_dot, e, F_c):
    """Compute the desired acceleration in the workspace using the impedance control law."""
    parantheses = K_d @ e_dot + K_p @ e - F_c
    return x_doubledot_d + np.linalg.solve(M_d, parantheses)



def compute_workspace_matrices(dynamics, J, J_dot, q_dot):
    M_q = dynamics.mass_matrix
    c_q = dynamics.coriolis_vector
    g_q = dynamics.gravity_vector

    # M_hat = (J M^-1 J^T)^-1
    M_inv_JT = np.linalg.inv(M_q) @ J.T
    M_hat = np.linalg.inv(J @ M_inv_JT)

    c_hat_xdot = (
        np.linalg.solve(J.T, c_q)
        - M_hat @ J_dot @ q_dot
)

    # g_hat = M_hat J M^-1 g_q = J_-T g_q
    g_hat = np.linalg.inv(J.T) @ g_q

    return M_hat, c_hat_xdot, g_hat


def compute_j_dot(J, previous_J, robot):
    """Compute the time derivative of the Jacobian matrix."""
    if previous_J is None:
        J_dot = np.zeros_like(J)
    else:
        J_dot = (J - previous_J) / robot.sim.dt
    return J_dot, J.copy()

def compute_es(x, x_dot, x_doubledot, x_d, x_dot_d, x_doubledot_d):
    """Compute the error in the workspace."""
    x = np.asarray(x, dtype=float)
    x_dot = np.asarray(x_dot, dtype=float)
    x_doubledot = np.asarray(x_doubledot, dtype=float)
    x_d = np.asarray(x_d, dtype=float)
    x_dot_d = np.asarray(x_dot_d, dtype=float)
    x_doubledot_d = np.asarray(x_doubledot_d, dtype=float)

    e = x_d - x
    e_dot = x_dot_d - x_dot
    e_doubledot = x_doubledot_d - x_doubledot

    return e, e_dot, e_doubledot

def compute_F_c(robot, tau_applied, J_A):
    """Estimate an equivalent end-effector wrench from external joint torques."""
    tau_applied = np.asarray(tau_applied, dtype=float)
    J_A = np.asarray(J_A, dtype=float)
    tau_external = np.asarray(
        robot.robot_state.get_external_joint_forces(tau_applied),
        dtype=float,
    )

    if jacobian.shape[1] != tau_external.size:
        raise ValueError(
            "Jacobian joint dimension must match the external joint torque vector"
        )

    # tau_external ~= J.T @ F_c; least squares also handles non-square Jacobians.
    F_c = np.linalg.lstsq(J_A.T, tau_external, rcond=None)[0]
    return F_c

def geometric_to_analytic_jacobian(J_G, euler):
    """
    Convert a geometric Jacobian to an analytic Jacobian.

    Assumptions:
    - J_G maps q_dot to [v_world, omega_world]
    - euler = [roll, pitch, yaw]
    - orientation convention:
          R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """

    J_G = np.asarray(J_G, dtype=float)
    roll, pitch, yaw = np.asarray(euler, dtype=float)

    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    # Maps Euler angle rates:
    # [roll_dot, pitch_dot, yaw_dot]
    # to world-frame angular velocity:
    # [wx, wy, wz]
    E = np.array([
        [cy * cp, -sy, 0.0],
        [sy * cp,  cy, 0.0],
        [-sp,      0.0, 1.0],
    ])

    # [v]
    # [w] = T_A [p_dot]
    #             [euler_dot]
    T_A = np.zeros((6, 6))
    T_A[:3, :3] = np.eye(3)
    T_A[3:, 3:] = E

    # J_G = T_A @ J_A
    # therefore J_A = T_A^-1 @ J_G
    J_A = np.linalg.solve(T_A, J_G)

    return J_A

def run():
    robot = create_scene()

    M_d = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # Desired mass matrix
    K_d = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])  # Desired damping matrix
    K_p = np.diag([100.0, 100.0, 100.0, 1.0, 1.0, 1.0])  # Desired stiffness matrix

    x_d = np.array([0.4, 0.1, 0.2, 1.0, 0.0, 0.0])  # Desired position in workspace
    tau_applied = np.zeros(6)
    has_previous_torque_sample = False

    # Eenmalig vóór de simulatie-loop
    previous_x_dot = None
    previous_J_A = None

    while True:
        # # get predefined gravity compensation from the robot model
        # gravity = robot.robot_model.get_dynamics().gravity_vector

        x = np.concatenate((
            np.asarray(robot.robot_state.get_end_effector_position(), dtype=float),
            np.asarray(p.getEulerFromQuaternion(
                    robot.robot_state.get_end_effector_orientation()), dtype=float)
        ))
        # print(x)
        # print(x.shape)
        if previous_J_A is None:
            previous_J_A = np.zeros((6, 6))

        J_G = robot.robot_model.get_jacobian()
        J_A = geometric_to_analytic_jacobian(J_G, x[3:])
        J_A_dot, previous_J_A = compute_j_dot(J_A, previous_J_A, robot)

        q_dot = np.asarray(robot.robot_state.get_joint_velocities(), dtype=float)
        x_dot = J_A @ q_dot

        # Compute the acceleration in the workspace using finite differences
        if previous_x_dot is None:
            x_doubledot = np.zeros_like(x_dot)
        else:
            x_doubledot = (x_dot - previous_x_dot) / robot.sim.dt

        previous_x_dot = x_dot.copy()


        x_dot_d = np.zeros(6)  # Desired velocity in workspace
        x_doubledot_d = np.zeros(6)  # Desired acceleration in workspace

        e, e_dot, e_doubledot = compute_es(x, x_dot, x_doubledot, x_d, x_dot_d, x_doubledot_d)

        if has_previous_torque_sample:
            F_c = compute_F_c(robot, tau_applied, J_A)
        else:
            F_c = np.zeros(6)




        dynamics = robot.robot_model.get_dynamics()

        M_hat, c_hat_xdot, g_hat = compute_workspace_matrices(dynamics, J_A, J_A_dot, q_dot)

        # getting robot state
        joint_angles = robot.robot_state.get_joint_angles()
        joint_velocities = robot.robot_state.get_joint_velocities()

        # # calculating output for each joint and compensating for gravity
        # new_joint_torques=0
        # torques = new_joint_torques + gravity

        # Joint torques from the Jacobian and a force vector
        x_doubledot_d = np.zeros(6)  # Example desired acceleration
        y = compute_y(x_doubledot_d, M_d, K_d, K_p, e_dot, e, F_c)
        F = compute_force_in_workspace(M_hat, y, c_hat_xdot, g_hat, F_c)  # Example force vector
        tau = torques(J_A, F)

        # controlling the robot
        tau_applied = np.clip(tau, -robot.arm_joint_forces, robot.arm_joint_forces)
        robot.control_torques(tau_applied)

        # running the pybullet simulation
        robot.sim.step()
        has_previous_torque_sample = True


if __name__ == "__main__":
    run()