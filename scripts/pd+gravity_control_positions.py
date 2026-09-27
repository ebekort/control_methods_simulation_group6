import argparse
import time
import numpy as np
import pybullet as p
from ur_simulation.classic_control.robots.ur7e import UR7e
from ur_simulation.pybullet import PyBullet
from ur_simulation.classic_control.robot_state.pybullet_robot_state import JointType

import matplotlib.pyplot as plt


def create_scene():
    init_joint_angles = np.array([1.57, -1.7, 1.7, -1.57, -1.57, -1.57])
    robot = UR7e(
        block_gripper=True,
        neutral_joints=init_joint_angles,
    )
    robot.sim.create_plane(0)
    robot.sim.physics_client.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, True)
    return robot


def run():
    '''
    PD+gravity controller implementation from the torques formula:
        u = gravity + Kp * eq - Kd * qdot
    Runs through 4 points from `target_poss`
    '''
    robot = create_scene()

    target_poss = [
        [0.45, 0.45, 0.45],
        [0.45, -0.45, 0.45],
        [-0.45, 0.45, 0.45],
        [-0.45, -0.45, 0.45],
    ]

    # target_poss = [
    #     [0.45, 0.45, 0.45],
    #     [0.45, -0.45, 0.45],
    #     [-0.45, -0.45, 0.45],
    #     [-0.45, 0.45, 0.45],
    # ]

    for i, target_pos in enumerate(target_poss):
        robot.sim.create_box(
            f'box{i}', np.array([0.01, 0.01, 0.01]), 0, target_pos,
            rgba_color=[0, 1, 0, 1], ghost=True
        )

    kp = np.diag([5, 5, 5, 5, 5, 5])
    kd = np.diag([5, 5, 5, 5, 5, 5])

    for target_pos in target_poss:
        target_rot = robot.robot_state.get_end_effector_orientation()
        qd = robot.robot_model.get_inverse_kinematics(target_pos, target_rot)

        qdot = robot.robot_state.get_joint_velocities()
        # until end effector position is close enough to the target pos
        # and its velocities are small enough
        while (
            not np.allclose(target_pos, robot.robot_state.get_end_effector_position(), atol=1e-01)
            or not np.allclose(qdot, np.zeros_like(qdot), atol=1e-03)
        ):
            gravity = robot.robot_model.get_dynamics().gravity_vector
            q = robot.robot_state.get_joint_angles()
            qdot = robot.robot_state.get_joint_velocities()

            eq = qd - q
            eq = (eq + np.pi) % (2 * np.pi) - np.pi
            torques = gravity + kp @ eq - kd @ qdot

            robot.control_torques(torques)

            robot.sim.step()


def plots():
    '''
    PD+gravity controller implementation from the torques formula:
        u = gravity + Kp * eq - Kd * qdot
    Plots last joint's stats for debugging it spinning continually over
     `max_steps` simulation steps
    '''
    robot = create_scene()

    target_pos = [0.45, 0.45, 0.45]
    target_rot = robot.robot_state.get_end_effector_orientation()
    qd = robot.robot_model.get_inverse_kinematics(target_pos, target_rot)

    kp = np.diag([2, 2, 2, 2, 2, 5])
    kd = np.diag([2, 2, 2, 2, 2, 5])

    # kp = np.diag([5, 5, 5, 5, 5, 5])
    # kd = np.diag([5, 5, 5, 5, 5, 5])

    q_last_log = []
    eq_last_log = []
    qd_last_log = []
    qdot_last_log = []
    ee_error_log = []

    max_steps = 4000
    for step in range(max_steps):
        gravity = robot.robot_model.get_dynamics().gravity_vector
        q = robot.robot_state.get_joint_angles()
        qdot = robot.robot_state.get_joint_velocities()

        eq = qd - q
        eq = (eq + np.pi) % (2 * np.pi) - np.pi
        torques = gravity + kp @ eq - kd @ qdot

        robot.control_torques(torques)

        ee_error = np.linalg.norm(
            np.array(target_pos) - robot.robot_state.get_end_effector_position()
        )

        q_last_log.append(q[-1])
        eq_last_log.append(eq[-1])
        qd_last_log.append(qd[-1])
        qdot_last_log.append(qdot[-1])
        ee_error_log.append(ee_error)

        robot.sim.step()

    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(8, 10))
    fig.suptitle('kp/kd (diag([2, 2, 2, 2, 2, 5])) capped')
    axes[0].plot(eq_last_log)
    axes[0].set_ylabel('eq[-1]')

    axes[1].plot(qd_last_log)
    axes[1].set_ylabel('qd[-1]')

    axes[2].plot(q_last_log)
    axes[2].set_ylabel('q[-1]')
    axes[2].axhline(np.pi, color='gray', linestyle='--', linewidth=0.5)
    axes[2].axhline(-np.pi, color='gray', linestyle='--', linewidth=0.5)

    axes[3].plot(qdot_last_log)
    axes[3].set_ylabel('qdot[-1]')

    axes[4].plot(ee_error_log)
    axes[4].set_ylabel('linalg pos distance')
    axes[4].set_xlabel('sim step')

    plt.tight_layout()
    plt.show()

    plt.plot(q_last_log[350:450])
    plt.title('q[1] zoomed (350-450)')
    plt.show()


if __name__ == '__main__':
    run()
