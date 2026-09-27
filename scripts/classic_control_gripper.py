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
        block_gripper=False,
        neutral_joints=init_joint_angles,
    )
    robot.sim.create_plane(0)
    return robot


def run():
    robot = create_scene()
    while True:
        gravity = robot.robot_model.get_dynamics().gravity_vector
        joint_angles = robot.robot_state.get_joint_angles()
        joint_velocities = robot.robot_state.get_joint_velocities()

        # calculating output for each joint and compensating for gravity
        
        new_joint_torques = 0
        torques = new_joint_torques + gravity

        # controlling the robot
        robot.control_torques(torques)

        # controlling the gripper
        robot.control_finger_width(0.0)

        # running the pybullet simulation
        robot.sim.step()


if __name__ == "__main__":
    run()
