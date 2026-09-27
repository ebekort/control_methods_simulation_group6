"""
For the following state class, a JointType bitmask is used to filter the output of certain functions. Otherwise,
pybullet would return states for uncontrollable joints specified in URDF files as well. By default, only REVOLUTE joints
will be included.

 Example to include revolute and prismatic joints:
 ```
 robot_state.get_joint_angles(JointType.REVOLUTE | JointType.PRISMATIC)
 ````
"""
from enum import Flag, auto

import pybullet as p

import numpy as np




class JointType(Flag):
    NONE = 0
    REVOLUTE = auto()
    PRISMATIC = auto()
    FIXED = auto()
    ALL = REVOLUTE | PRISMATIC | FIXED


class PyBulletRobotState:
    def __init__(
        self,
        robot_id: int,
        dt: float,
        ee_link_index: int,
        finger_indices: list[int],
    ):
        """
        Initialize a robot state interface backed by a PyBullet simulation.

        Args:
            robot_id: Unique ID of the robot body in the PyBullet simulation.
            dt: Time difference between two subsequent simulation steps.
            ee_link_index: Index of the end-effector link in the pybullet simulation.
            finger_indices: List of indices of the finger joints in the pybullet simulation.
        """
        self.robot_id = robot_id
        self.ee_index = ee_link_index
        self.finger_indices = finger_indices
        self.total_joints = p.getNumJoints(robot_id)
        self.previous_joint_velocities = np.zeros_like(self.get_joint_velocities(JointType.REVOLUTE | JointType.PRISMATIC))
        self.dt = dt


    def get_joint_angles(self, joint_type: JointType = JointType.REVOLUTE) -> np.ndarray:
        """
        Get the current joint positions.

        Args:
            joint_type: Bitmask specifying which joint types to include.
                By default, all joint types are returned.

        Returns:
            A one-dimensional array containing the joint positions in the
            order they appear in the URDF, filtered by ``joint_type``.
        """
        joint_angles = []
        joint_info = [p.getJointInfo(self.robot_id, idx) for idx in range(self.total_joints)]
        joint_states = p.getJointStates(self.robot_id, range(self.total_joints))

        for info, state in zip(joint_info, joint_states):
            if any([
                info[2] == p.JOINT_REVOLUTE and JointType.REVOLUTE & joint_type,
                info[2] == p.JOINT_PRISMATIC and JointType.PRISMATIC & joint_type,
                info[2] == p.JOINT_FIXED and JointType.FIXED & joint_type,
            ]):
                joint_angles.append(state[0])
        return np.array(joint_angles)

    def get_joint_velocities(self, joint_type: JointType = JointType.REVOLUTE) -> np.ndarray:
        """
        Get the current joint velocities.

        Args:
            joint_type: Bitmask specifying which joint types to include.
                By default, all joint types are returned.

        Returns:
            A one-dimensional array containing the joint velocities in the
            order they appear in the URDF, filtered by ``joint_type``.
        """
        joint_angles = []
        joint_info = [p.getJointInfo(self.robot_id, idx) for idx in range(self.total_joints)]
        joint_states = p.getJointStates(self.robot_id, range(self.total_joints))

        for info, state in zip(joint_info, joint_states):
            if any([
                info[2] == p.JOINT_REVOLUTE and JointType.REVOLUTE & joint_type,
                info[2] == p.JOINT_PRISMATIC and JointType.PRISMATIC & joint_type,
                info[2] == p.JOINT_FIXED and JointType.FIXED & joint_type,
            ]):
                joint_angles.append(state[1])
        return np.array(joint_angles)


    def get_external_joint_forces(
            self,
            applied_torques: np.ndarray
    ) -> np.ndarray:
        """
        Estimate external joint torques using the dynamics residual.

        Args:
            applied_torques: A one-dimensional array containing torques applied during the last simulation step.

        Returns:
            A one-dimensional array containing estimated external joint torques.
        """
        joint_angles = self.get_joint_angles(JointType.REVOLUTE | JointType.PRISMATIC)
        joint_velocities = self.get_joint_velocities(JointType.REVOLUTE | JointType.PRISMATIC)

        # Estimate joint accelerations
        if not hasattr(self, "previous_joint_velocities"):
            self.previous_joint_velocities = joint_velocities.copy()
            return np.zeros_like(joint_velocities)

        joint_accelerations = (joint_velocities - self.previous_joint_velocities) / self.dt

        self.previous_joint_velocities = joint_velocities.copy()

        inverse_dynamics_forces = p.calculateInverseDynamics(
            self.robot_id,
            joint_angles.tolist(),
            joint_velocities.tolist(),
            joint_accelerations.tolist(),
        )

        return np.asarray(inverse_dynamics_forces) - applied_torques

    def get_end_effector_position(self) -> np.ndarray:
        """
        Get the current end-effector position.

        Returns:
            A length-3 array containing the end-effector position in the world
            frame as ``[x, y, z]``.
        """
        link_state = p.getLinkState(
            self.robot_id,
            self.ee_index,
        )
        return np.array(link_state[4])

    def get_end_effector_orientation(self) -> np.ndarray:
        """
        Get the current end-effector orientation.

        Returns:
            A length-4 array containing the end-effector orientation in the
            world frame as a quaternion ``[x, y, z, w]``.
        """
        link_state = p.getLinkState(
            self.robot_id,
            self.ee_index,
        )
        return np.array(link_state[5])

    def get_end_effector_linear_velocity(self) -> np.ndarray:
        """
        Get the current end-effector linear velocity.

        Returns:
            An array containing the end-effector linear velocity in the
            world frame as ``[vx, vy, vz]``.
        """
        link_state = p.getLinkState(
            self.robot_id,
            self.ee_index,
            computeLinkVelocity=1,
        )
        return np.array(link_state[6])

    def get_end_effector_angular_velocity(self) -> np.ndarray:
        """
        Get the current end-effector angular velocity.

        Returns:
            An array containing the end-effector angular velocity in the
            world frame as ``[wx, wy, wz]``.
        """
        link_state = p.getLinkState(
            self.robot_id,
            self.ee_index,
            computeLinkVelocity=1,
        )
        return np.array(link_state[7])

    def get_fingers_width(self) -> float:
        finger1 = p.getJointState(self.robot_id, self.finger_indices[0])[0]
        finger2 = p.getJointState(self.robot_id, self.finger_indices[1])[0]
        return finger1 + finger2
