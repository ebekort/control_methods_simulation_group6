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