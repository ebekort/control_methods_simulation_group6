from dataclasses import dataclass

import math
import numpy as np
import pybullet as p
from ur_simulation.classic_control.robot_state.pybullet_robot_state import PyBulletRobotState, JointType


@dataclass
class Dynamics:
    mass_matrix: np.ndarray
    gravity_vector: np.ndarray
    coriolis_vector: np.ndarray


class PyBulletRobotModel:
    def __init__(self, robot_id, robot_state: PyBulletRobotState):
        self.robot_id = robot_id
        self.robot_state = robot_state


    def get_inverse_kinematics(self, position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        """
        Given the position and quaternion of the end effector, compute corresponding joint angles.
        """
        joint_angles = p.calculateInverseKinematics(
            bodyIndex=self.robot_id,
            endEffectorLinkIndex=self.robot_state.ee_index,
            targetPosition=position,
            targetOrientation=quaternion,
        )
        return joint_angles


    def get_jacobian(self) -> np.ndarray:
        """
        Get the Jacobian matrix for the current position.
        """
        n_joints = len(self.robot_state.get_joint_angles(JointType.REVOLUTE))
        joint_angles = self.robot_state.get_joint_angles(JointType.REVOLUTE | JointType.PRISMATIC).tolist()
        zero_control = np.zeros_like(joint_angles).tolist()

        ee = p.getLinkState(self.robot_id, self.robot_state.ee_index)
        linear_jacobian, angle_jacobian = p.calculateJacobian(
            self.robot_id,
            self.robot_state.ee_index,
            ee[2],
            joint_angles,
            zero_control,
            zero_control,
        )

        jacobian = np.vstack((linear_jacobian, angle_jacobian))[:, :n_joints]
        return jacobian

    def get_dynamics(self) -> Dynamics:
        """
        Compute the dynamics for the current state of the robot. This includes:
            - Mass matrix: Joint-space inertia matrix.
            - Gravity vector: Generalized forces required to compensate for gravity.
            - Coriolis/centrifugal vector: Velocity-dependent generalized forces.

        Returns: Dynamics object containing the above dynamic quantities.
        """
        n_joints = len(self.robot_state.get_joint_angles(JointType.REVOLUTE))
        joint_angles = self.robot_state.get_joint_angles(JointType.REVOLUTE | JointType.PRISMATIC).tolist()
        joint_velocities = self.robot_state.get_joint_velocities(JointType.REVOLUTE | JointType.PRISMATIC).tolist()
        zero_control = np.zeros_like(joint_angles).tolist()

        mass_matrix = np.array(
            p.calculateMassMatrix(self.robot_id, joint_angles)
        )[:n_joints, :n_joints]

        gravity_vector = np.array(
            p.calculateInverseDynamics(self.robot_id, joint_angles, zero_control, zero_control)
        )[:n_joints]

        coriolis_vector = np.array(
            p.calculateInverseDynamics(self.robot_id, joint_angles, joint_velocities, zero_control)
        )[:n_joints] - gravity_vector

        return Dynamics(
            mass_matrix=mass_matrix,
            gravity_vector=gravity_vector,
            coriolis_vector=coriolis_vector,
        )

