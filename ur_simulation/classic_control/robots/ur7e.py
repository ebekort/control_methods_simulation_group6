import time
from typing import Optional
from enum import Enum, auto

import numpy as np
from importlib.resources import files
from pathlib import Path

from ur_simulation.classic_control.robot_models import PyBulletRobotModel
from ur_simulation.classic_control.robot_state import PyBulletRobotState
from ur_simulation.classic_control.robot_state.pybullet_robot_state import JointType
from ur_simulation.pybullet import PyBullet


class UR7e:
    """UR7e robot in PyBullet.

    Args:
        block_gripper (bool, optional): Whether the gripper is blocked. Defaults to False.
        base_position (np.ndarray, optional): Position of the base base of the robot, as (x, y, z). Defaults to (0, 0, 0).
        neutral_joints (np.ndarray, optional): Default joint positions of the robot arm
    """

    CONTROLLABLE_JOINTS = np.array([2, 3, 4, 5, 6, 7, 13, 14])
    ARM_JOINTS = np.array([2, 3, 4, 5, 6, 7])
    FINGER_JOINTS = np.array([13, 14])

    # From https://www.universal-robots.com/articles/ur/robot-care-maintenance/max-joint-torques-cb3-and-e-series/
    MAX_JOINT_TORQUES = np.array([150, 150, 150, 28, 28, 28, 130, 130])

    # Default neutral angles
    NEUTRAL_JOINTS = np.array([1.57,-1.7,2.4,-1.57,-1.57,-1.57])
    EE_LINK = 9

    def __init__(
        self,
        block_gripper: bool = False,
        base_position: Optional[np.ndarray] = None,
        neutral_joints: Optional[np.ndarray] = None,
        n_substeps: int = 1,
        step_frequency: int = 1000,
        interactive_gui: bool = False,
    ) -> None:
        base_position = base_position if base_position is not None else np.zeros(3)
        self.block_gripper = block_gripper

        if block_gripper:
            urdf = Path(files("ur_simulation.assets")) / "urdf" / "ur7e.urdf"
        else:
            urdf = Path(files("ur_simulation.assets")) / "urdf" / "ur7e_hande.urdf"

        self.arm_indices = self.ARM_JOINTS
        self.fingers_indices = self.FINGER_JOINTS
        self.arm_joint_forces = self.MAX_JOINT_TORQUES[:-2]
        self.finger_joint_forces = self.MAX_JOINT_TORQUES[-2:]
        self.neutral_joint_values = self.NEUTRAL_JOINTS if neutral_joints is None else neutral_joints
        self.ee_link = self.EE_LINK

        self.sim = PyBullet(n_substeps=n_substeps, step_frequency=step_frequency, interactive_gui=interactive_gui)
        self.body_name = "ur7e"
        with self.sim.no_rendering():
            self._load_robot(urdf.as_posix(), base_position)
        self.robot_id = self.sim.bodies_idx[self.body_name]
        self.robot_state = PyBulletRobotState(self.robot_id, self.sim.dt, self.ee_link, self.fingers_indices)
        self.robot_model = PyBulletRobotModel(self.robot_id, self.robot_state)

    def step(self, realtime: bool = True) -> None:
        # store velocity for estimating acceleration in the robot state
        self.robot_state.previous_joint_velocities = self.robot_state.get_joint_velocities(JointType.REVOLUTE | JointType.PRISMATIC)
        self.sim.step(realtime)
        self.sim.render()

    def _load_robot(self, file_name: str, base_position: np.ndarray) -> None:
        """Load the robot.

        Args:
            file_name (str): The URDF file name of the robot.
            base_position (np.ndarray): The position of the robot, as (x, y, z).
        """
        self.sim.loadURDF(
            body_name=self.body_name,
            fileName=file_name,
            basePosition=base_position,
            useFixedBase=True,
        )
        self.set_joint_neutral()

        # Set high finger friction for interaction
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[0], lateral_friction=1.0)
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[1], lateral_friction=1.0)

        self.sim.enable_torque_control(self.body_name, self.arm_indices, disable_damping=True)

    def control_torques(self, torques: np.ndarray) -> None:
        """
        Control the joints using explicit torques.
        Args:
            torques: The torques to apply. Clipped to stay within allowed ranges
        """
        torques = torques.clip(-self.arm_joint_forces, self.arm_joint_forces)
        self.sim.control_torques(
            body=self.body_name,
            joints=self.arm_indices,
            torques=torques,
        )

    def set_joint_angles(self, angles: np.ndarray) -> None:
        """Set the joint position of a body. Can induce collisions.

        Args:
            angles (list): Joint angles.
        """
        self.sim.set_joint_angles(self.body_name, joints=self.arm_indices, angles=angles)

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        self.set_joint_angles(self.neutral_joint_values)

    def inverse_kinematics(self, link: int, position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """Compute the inverse kinematics and return the new joint values.

        Args:
            link (int): The link.
            position (x, y, z): Desired position of the link.
            orientation (x, y, z, w): Desired orientation of the link.

        Returns:
            List of joint values.
        """
        inverse_kinematics = self.sim.inverse_kinematics(self.body_name, link=link, position=position, orientation=orientation)
        return inverse_kinematics

    def control_finger_width(self, finger_width: float) -> None:
        half_width = finger_width / 2

        self.sim.control_position(
            body=self.body_name,
            joints=self.fingers_indices,
            target_angles=np.array([half_width, half_width]),
            forces=self.finger_joint_forces
        )