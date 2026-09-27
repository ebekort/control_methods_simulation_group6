import argparse
import time
import numpy as np
import pybullet as p
from ur_simulation.classic_control.robots.ur7e import UR7e
from ur_simulation.pybullet import PyBullet
from ur_simulation.classic_control.robot_state.pybullet_robot_state import JointType


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
    '''
    robot = create_scene()

    target_pos = [0.45, 0.45, 0.45]

    robot.sim.create_box(
        'box', np.array([0.01, 0.01, 0.01]), 0, target_pos,
        rgba_color=[0, 1, 0, 1], ghost=True
    )

    kp = np.diag([5, 5, 5, 5, 5, 5])
    kd = np.diag([5, 5, 5, 5, 5, 5])

    target_rot = robot.robot_state.get_end_effector_orientation()
    qd = robot.robot_model.get_inverse_kinematics(target_pos, target_rot)
    while True:
        gravity = robot.robot_model.get_dynamics().gravity_vector
        q = robot.robot_state.get_joint_angles()
        qdot = robot.robot_state.get_joint_velocities()

        eq = qd - q
        eq = (eq + np.pi) % (2 * np.pi) - np.pi
        torques = gravity + kp @ eq - kd @ qdot

        robot.control_torques(torques)

        robot.sim.step()


if __name__ == "__main__":
    run()
