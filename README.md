# ur-simulation

Repository for control methods of a simulated UR7e

## Installation

This repository is compatible with python > 3.9

```bash
git clone https://github.com/theyseemeRobin/control_methods_simulation.git
pip install -e control_methods_simulation/
```

## Usage

### Classic control
This repository implements a PyBullet Simulation for the UR7e. You can run a demonstrative example script using the 
following command:
```commandline
python scripts/classic_control.py
```
This will launch the simulation. At every step, it computes the gravity vector and uses those forces as the commanded 
torques to compensate for gravity. As a result, the robot should remain stationary.

To implement your own controllers, you can use the [PyBulletRobotState](ur_simulation/classic_control/robot_state/pybullet_robot_state.py) class. 
This class gives you access to relevant state variables, such as the positions and velocities of the joints and end-effector. 
Jacobain and Dynamic quantities such as the gravity vector, or mass-inertia matrix are exposed through the 
[PybulletRobotModel](ur_simulation/classic_control/robot_models/pybullet_robot_model.py).

A basic example showing how to use the model and state to obtain torques to apply on the joints, as well as how to apply them,
is shown in the [control_demo.py](scripts/control_demo.py)
