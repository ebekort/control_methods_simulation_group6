import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

import numpy as np
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc

import ur_simulation.assets


class PyBullet:
    """Convenient class to use PyBullet physics engine.

    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        n_substeps (int, optional): Number of sim substep when step() is called. Defaults to 20.
        background_color (np.ndarray, optional): The background color as (red, green, blue).
            Defaults to np.array([223, 54, 45]).
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.
    """

    def __init__(
        self,
        render_mode: str = "human",
        n_substeps: int = 20,
        background_color: Optional[np.ndarray] = None,
        renderer: str = "Tiny",
        step_frequency: int = 100,
        interactive_gui: bool = False,
    ) -> None:
        self.render_mode = render_mode
        background_color = background_color if background_color is not None else np.array([223.0, 54.0, 45.0])
        self.background_color = background_color.astype(np.float32) / 255
        options = "--background_color_red={} --background_color_green={} --background_color_blue={}".format(
            *self.background_color
        )
        if self.render_mode == "human":
            self.connection_mode = p.GUI
        elif self.render_mode == "rgb_array":
            if renderer == "OpenGL":
                self.connection_mode = p.GUI
            elif renderer == "Tiny":
                self.connection_mode = p.DIRECT
            else:
                raise ValueError("The 'renderer' argument is must be in {'Tiny', 'OpenGL'}")
        else:
            raise ValueError("The 'render' argument is must be in {'rgb_array', 'human'}")
        self.physics_client = bc.BulletClient(connection_mode=self.connection_mode, options=options)
        self.physics_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.physics_client.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, interactive_gui)

        self.n_substeps = n_substeps
        self.step_frequency = step_frequency
        self.timestep = 1.0 / step_frequency
        self.physics_client.setTimeStep(self.timestep)
        self.physics_client.resetSimulation()
        self.physics_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.physics_client.setGravity(0, 0, -9.81)
        self.bodies_idx = {}
        self.elapsed_time = 0

    @property
    def dt(self):
        """Timestep."""
        return self.timestep * self.n_substeps

    def step(self, n_steps: int = None, realtime: bool = True) -> None:
        """Step the simulation."""
        n_steps = n_steps if n_steps is not None else self.n_substeps
        for _ in range(n_steps):
            self.physics_client.stepSimulation()
        self.elapsed_time += self.dt
        if realtime:
            time.sleep(self.dt)

    def close(self) -> None:
        """Close the simulation."""
        if self.physics_client.isConnected():
            self.physics_client.disconnect()

    def save_state(self) -> int:
        """Save the current simulation state.

        Returns:
            int: A state id assigned by PyBullet, which is the first non-negative
            integer available for indexing.
        """
        return self.physics_client.saveState()

    def restore_state(self, state_id: int) -> None:
        """Restore a simulation state.

        Args:
            state_id: The simulation state id returned by save_state().
        """
        self.physics_client.restoreState(state_id)

    def remove_state(self, state_id: int) -> None:
        """Remove a simulation state. This will make this state_id available again for returning in save_state().

        Args:
            state_id: The simulation state id returned by save_state().
        """
        self.physics_client.removeState(state_id)

    def render(
        self,
        width: int = 720,
        height: int = 480,
        target_position: Optional[np.ndarray] = None,
        distance: float = 1.4,
        yaw: float = 45,
        pitch: float = -30,
        roll: float = 0,
    ) -> Optional[np.ndarray]:
        """Render.

        If render mode is "rgb_array", return an RGB array of the scene. Else, do nothing and return None.

        Args:
            width (int, optional): Image width. Defaults to 720.
            height (int, optional): Image height. Defaults to 480.
            target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).
                Defaults to [0., 0., 0.].
            distance (float, optional): Distance of the camera. Defaults to 1.4.
            yaw (float, optional): Yaw of the camera. Defaults to 45.
            pitch (float, optional): Pitch of the camera. Defaults to -30.
            roll (int, optional): Roll of the camera. Defaults to 0.
            mode (str, optional): Deprecated: This argument is deprecated and will be removed in a future
                version. Use the render_mode argument of the constructor instead.

        Returns:
            RGB np.ndarray or None: An RGB array if mode is 'rgb_array', else None.
        """
        if self.render_mode == "rgb_array":
            target_position = target_position if target_position is not None else np.zeros(3)
            view_matrix = self.physics_client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=target_position,
                distance=distance,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                upAxisIndex=2,
            )
            proj_matrix = self.physics_client.computeProjectionMatrixFOV(
                fov=60, aspect=float(width) / height, nearVal=0.1, farVal=100.0
            )
            (_, _, rgba, _, _) = self.physics_client.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                shadow=True,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
            )
            # With Python3.10, pybullet return flat tuple instead of array. So we need to build create the array.
            rgba = np.array(rgba, dtype=np.uint8).reshape((height, width, 4))
            return rgba[..., :3]


    def set_joint_angles(self, body: str, joints: np.ndarray, angles: np.ndarray) -> None:
        """Set the angles of the joints of the body.

        Args:
            body (str): Body unique name.
            joints (np.ndarray): List of joint indices, as a list of ints.
            angles (np.ndarray): List of target angles, as a list of floats.
        """
        for joint, angle in zip(joints, angles):
            self.set_joint_angle(body=body, joint=joint, angle=angle)

    def set_joint_angle(self, body: str, joint: int, angle: float) -> None:
        """Set the angle of the joint of the body.

        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body.
            angle (float): Target angle.
        """
        self.physics_client.resetJointState(bodyUniqueId=self.bodies_idx[body], jointIndex=joint, targetValue=angle)

    def enable_torque_control(
            self,
            body: str,
            joint_indices: np.ndarray,
            disable_damping: bool = True,
    ) -> None:
        """
        Enables pybullet torque control.
        Args:
            body: Body unique name
            joint_indices: Joint indices to enable control for
            disable_damping: Whether to disable pybullet's internal damping.

        Returns:

        """
        p.setJointMotorControlArray(
            bodyUniqueId=self.bodies_idx[body],
            jointIndices=joint_indices,
            controlMode=p.VELOCITY_CONTROL,
            forces=np.zeros_like(joint_indices)
        )
        if disable_damping:
            for idx in joint_indices:
                p.changeDynamics(self.bodies_idx[body], idx, linearDamping=0, angularDamping=0, jointDamping=0)

    def control_joints(self, body: str, joints: np.ndarray, target_angles: np.ndarray, forces: np.ndarray) -> None:
        """Control the joint motors using positional control.

        Args:
            body (str): Body unique name.
            joints (np.ndarray): List of joint indices, as a list of ints.
            target_angles (np.ndarray): List of target angles, as a list of floats.
            forces (np.ndarray): Forces to apply, as a list of floats.
        """
        self.physics_client.setJointMotorControlArray(
            self.bodies_idx[body],
            jointIndices=joints,
            controlMode=self.physics_client.POSITION_CONTROL,
            targetPositions=target_angles,
            forces=forces,
        )

    def control_position(self, body: str, joints: np.ndarray, target_angles: np.ndarray, forces: np.ndarray) -> None:
        """Control the joint motors using positional control.

        Args:
            body (str): Body unique name.
            joints (np.ndarray): List of joint indices, as a list of ints.
            target_angles (np.ndarray): List of target angles, as a list of floats.
            forces (np.ndarray): Forces to apply, as a list of floats.
        """
        self.physics_client.setJointMotorControlArray(
            self.bodies_idx[body],
            jointIndices=joints,
            controlMode=self.physics_client.POSITION_CONTROL,
            targetPositions=target_angles,
            forces=forces,
        )

    def control_torques(self, body: str, joints: np.ndarray, torques: np.ndarray) -> None:
        """Control the joint motors using torque control.

        Args:
            body (str): Body unique name.
            joints (np.ndarray): List of joint indices, as a list of ints.
            torques (np.ndarray): Forces to apply, as a list of floats.
        """
        self.physics_client.setJointMotorControlArray(
            bodyUniqueId=self.bodies_idx[body],
            jointIndices=joints,
            controlMode=p.TORQUE_CONTROL,
            forces=torques
        )

    def inverse_kinematics(self, body: str, link: int, position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """Compute the inverse kinematics and return the new joint state.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            position (np.ndarray): Desired position of the end-effector, as (x, y, z).
            orientation (np.ndarray): Desired orientation of the end-effector as quaternion (x, y, z, w).

        Returns:
            np.ndarray: The new joint state.
        """
        joint_state = self.physics_client.calculateInverseKinematics(
            bodyIndex=self.bodies_idx[body],
            endEffectorLinkIndex=link,
            targetPosition=position,
            targetOrientation=orientation,
        )
        return np.array(joint_state)

    def place_visualizer(self, target_position: np.ndarray, distance: float, yaw: float, pitch: float) -> None:
        """Orient the camera used for rendering.

        Args:
            target (np.ndarray): Target position, as (x, y, z).
            distance (float): Distance from the target position.
            yaw (float): Yaw.
            pitch (float): Pitch.
        """
        self.physics_client.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=target_position,
        )

    @contextmanager
    def no_rendering(self) -> Iterator[None]:
        """Disable rendering within this context."""
        self.physics_client.configureDebugVisualizer(self.physics_client.COV_ENABLE_RENDERING, 0)
        yield
        self.physics_client.configureDebugVisualizer(self.physics_client.COV_ENABLE_RENDERING, 1)

    def loadURDF(self, body_name: str, **kwargs: Any) -> None:
        """Load URDF file.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
        """
        self.bodies_idx[body_name] = self.physics_client.loadURDF(**kwargs)

    def create_box(
        self,
        body_name: str,
        half_extents: np.ndarray,
        mass: float,
        position: np.ndarray,
        rgba_color: Optional[np.ndarray] = None,
        specular_color: Optional[np.ndarray] = None,
        ghost: bool = False,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
        texture: Optional[str] = None,
    ) -> None:
        """Create a box.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            half_extents (np.ndarray): Half size of the box in meters, as (x, y, z).
            mass (float): The mass in kg.
            position (np.ndarray): The position, as (x, y, z).
            rgba_color (np.ndarray, optional): Body color, as (r, g, b, a). Defaults as [0, 0, 0, 0]
            specular_color (np.ndarray, optional): Specular color, as (r, g, b). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
            texture (str or None, optional): Texture file name. Defaults to None.
        """
        rgba_color = rgba_color if rgba_color is not None else np.zeros(4)
        specular_color = specular_color if specular_color is not None else np.zeros(3)
        visual_kwargs = {
            "halfExtents": half_extents,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"halfExtents": half_extents}
        self._create_geometry(
            body_name,
            geom_type=self.physics_client.GEOM_BOX,
            mass=mass,
            position=position,
            ghost=ghost,
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )
        if texture is not None:
            texture_path = os.path.join(ur_simulation.assets.get_data_path(), texture)
            texture_uid = self.physics_client.loadTexture(texture_path)
            self.physics_client.changeVisualShape(self.bodies_idx[body_name], -1, textureUniqueId=texture_uid)

    def create_cylinder(
        self,
        body_name: str,
        radius: float,
        height: float,
        mass: float,
        position: np.ndarray,
        rgba_color: Optional[np.ndarray] = None,
        specular_color: Optional[np.ndarray] = None,
        ghost: bool = False,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
    ) -> None:
        """Create a cylinder.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            radius (float): The radius in meter.
            height (float): The height in meter.
            mass (float): The mass in kg.
            position (np.ndarray): The position, as (x, y, z).
            rgba_color (np.ndarray, optional): Body color, as (r, g, b, a). Defaults as [0, 0, 0, 0]
            specular_color (np.ndarray, optional): Specular color, as (r, g, b). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
        """
        rgba_color = rgba_color if rgba_color is not None else np.zeros(4)
        specular_color = specular_color if specular_color is not None else np.zeros(3)
        visual_kwargs = {
            "radius": radius,
            "length": height,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"radius": radius, "height": height}
        self._create_geometry(
            body_name,
            geom_type=self.physics_client.GEOM_CYLINDER,
            mass=mass,
            position=position,
            ghost=ghost,
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )

    def create_sphere(
        self,
        body_name: str,
        radius: float,
        mass: float,
        position: np.ndarray,
        rgba_color: Optional[np.ndarray] = None,
        specular_color: Optional[np.ndarray] = None,
        ghost: bool = False,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
    ) -> None:
        """Create a sphere.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            radius (float): The radius in meter.
            mass (float): The mass in kg.
            position (np.ndarray): The position, as (x, y, z).
            rgba_color (np.ndarray, optional): Body color, as (r, g, b, a). Defaults as [0, 0, 0, 0]
            specular_color (np.ndarray, optional): Specular color, as (r, g, b). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
        """
        rgba_color = rgba_color if rgba_color is not None else np.zeros(4)
        specular_color = specular_color if specular_color is not None else np.zeros(3)
        visual_kwargs = {
            "radius": radius,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"radius": radius}
        self._create_geometry(
            body_name,
            geom_type=self.physics_client.GEOM_SPHERE,
            mass=mass,
            position=position,
            ghost=ghost,
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )

    def _create_geometry(
        self,
        body_name: str,
        geom_type: int,
        mass: float = 0.0,
        position: Optional[np.ndarray] = None,
        ghost: bool = False,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
        visual_kwargs: Dict[str, Any] = {},
        collision_kwargs: Dict[str, Any] = {},
    ) -> None:
        """Create a geometry.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            geom_type (int): The geometry type. See self.physics_client.GEOM_<shape>.
            mass (float, optional): The mass in kg. Defaults to 0.
            position (np.ndarray, optional): The position, as (x, y, z). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
            visual_kwargs (dict, optional): Visual kwargs. Defaults to {}.
            collision_kwargs (dict, optional): Collision kwargs. Defaults to {}.
        """
        position = position if position is not None else np.zeros(3)
        baseVisualShapeIndex = self.physics_client.createVisualShape(geom_type, **visual_kwargs)
        if not ghost:
            baseCollisionShapeIndex = self.physics_client.createCollisionShape(geom_type, **collision_kwargs)
        else:
            baseCollisionShapeIndex = -1
        self.bodies_idx[body_name] = self.physics_client.createMultiBody(
            baseVisualShapeIndex=baseVisualShapeIndex,
            baseCollisionShapeIndex=baseCollisionShapeIndex,
            baseMass=mass,
            basePosition=position,
        )

        if lateral_friction is not None:
            self.set_lateral_friction(body=body_name, link=-1, lateral_friction=lateral_friction)
        if spinning_friction is not None:
            self.set_spinning_friction(body=body_name, link=-1, spinning_friction=spinning_friction)

    def create_plane(self, z_offset: float) -> None:
        """Create a plane. (Actually, it is a thin box.)

        Args:
            z_offset (float): Offset of the plane.
        """
        self.create_box(
            body_name="plane",
            half_extents=np.array([3.0, 3.0, 0.01]),
            mass=0.0,
            position=np.array([0.0, 0.0, z_offset - 0.01]),
            specular_color=np.zeros(3),
            rgba_color=np.array([0.15, 0.15, 0.15, 1.0]),
        )

    def create_table(
        self,
        length: float,
        width: float,
        height: float,
        x_offset: float = 0.0,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
    ) -> None:
        """Create a fixed table. Top is z=0, centered in y.

        Args:
            length (float): The length of the table (x direction).
            width (float): The width of the table (y direction)
            height (float): The height of the table.
            x_offset (float, optional): The offset in the x direction.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
        """
        self.create_box(
            body_name="table",
            half_extents=np.array([length, width, height]) / 2,
            mass=0.0,
            position=np.array([x_offset, 0.0, -height / 2]),
            specular_color=np.zeros(3),
            rgba_color=np.array([0.95, 0.95, 0.95, 1]),
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
        )

    def set_lateral_friction(self, body: str, link: int, lateral_friction: float) -> None:
        """Set the lateral friction of a link.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            lateral_friction (float): Lateral friction.
        """
        self.physics_client.changeDynamics(
            bodyUniqueId=self.bodies_idx[body],
            linkIndex=link,
            lateralFriction=lateral_friction,
        )

    def set_spinning_friction(self, body: str, link: int, spinning_friction: float) -> None:
        """Set the spinning friction of a link.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            spinning_friction (float): Spinning friction.
        """
        self.physics_client.changeDynamics(
            bodyUniqueId=self.bodies_idx[body],
            linkIndex=link,
            spinningFriction=spinning_friction,
        )
