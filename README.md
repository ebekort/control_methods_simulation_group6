<!-- # ur-simulation -->

Four controllers for a simulated UR7e (PyBullet):

| | Free motion | Contact |
|---|---|---|
| **Model-based** (exact model from PyBullet) | PD+gravity (`pd+gravity_control.py`) | Impedance control (`impedance_control.py`) |
| **Model-free** (learned model) | Regulation and tracking with a physics-informed neural network (`pinn_control.py`, `pinn_track.py`) | Cartesian impedance with learned gravity (`pinn_contact.py`) |

The model-free controllers follow Liu et al. (2024), *Physics-Informed Neural Networks to Model and Control Robots*: four networks learn the mass matrix, potential energy, damping and actuation matrix of the Euler-Lagrange equation from logged joint data, with no part of the robot model used to collect the data or in the controllers.

## Installation

Tested with Python 3.13. From the folder that will hold the repository and a virtual environment:

```bash
git clone https://github.com/ebekort/control_methods_simulation_group6.git
python -m venv .venv
source .venv/bin/activate                              # Windows: .venv\Scripts\activate
pip install -e control_methods_simulation_group6/      # simulator wrapper (PyBullet, NumPy)
pip install -r control_methods_simulation_group6/requirements.txt   # JAX, Optax, Matplotlib
cd control_methods_simulation_group6
```

All commands below are run from the repository folder; the paths in them (`models/`, `data/`, `runs/`) are relative to it. The simulator wrapper `ur_simulation` comes from the course repository [theyseemeRobin/control_methods_simulation](https://github.com/theyseemeRobin/control_methods_simulation).

The learned-model scripts run on a CPU. Training is faster on a GPU (see the JAX installation guide for a CUDA build, or `hpc/` below).

## Trained models

`models/` holds one trained model per development stage (see [models/README.md](models/README.md)); `models/best.pkl` is the final recipe. The model-free control scripts need a trained model. The repository includes the trained models from the report, so these scripts can be run straight after installation, without generating data or training first.

```bash
python scripts/pinn_control.py models/best.pkl --gui              # regulation
python scripts/pinn_track.py models/best.pkl --gui                # tracking
python scripts/pinn_contact.py models/best.pkl --scene box --gui  # contact
```

## Usage

Every script with command-line options prints its purpose, all options with their defaults, and examples with

```bash
python scripts/<script>.py -h
```

Scripts without options (marked "no" below) open the GUI and run a fixed demo (`impedance_control.py` takes the name of its scene as argument). Their purpose is in the docstring at the top of the file. Scripts with options run headless unless `--gui` is given (the GUI runs in real time).

| Script | What it does | `-h` |
|---|---|---|
| **Model-based** | | |
| `pd+gravity_control.py` | PD+gravity regulation to one target with the exact gravity | no |
| `pd+gravity_control_positions.py` | The same as `pd+gravity_control.py` but through four targets, plus a debug plot of the last joint | no |
| `PINN.py` | Regulation as in Eq. 13 of Liu et al. with the exact gravity at the reference pose | no |
| `impedance_control.py` | Task-space impedance control with the exact model; run with `free_space` or `external_force` (the scene with the box), scenes in `scripts/experiments/` | no |
| `impedance_contact.py` | `impedance_control.py`'s law on the shared box experiment of `pinn_contact.py --scene box` | yes |
| **Data** | | |
| `pd+gravity_control_data.py` | Drives the arm along random joint sinusoids and logs q, q̇ and the applied torque as `.csv`; `--controller pd` (no model) or `model`, `--dt`, `--dither`, `--inject-a`/`--inject-d` (known actuation and damping), `--log-truth` (true gravity, for evaluation only) | yes |
| `data_ranges.py` | Joint ranges covered by a data folder; checks whether a pose lies inside | yes |
| **Learned model** | | |
| `pinn_model.py` | The model: networks, Euler-Lagrange dynamics, RK4 step, loss. Run directly for sanity checks of the derivation | no options |
| `pinn_data.py` | Loads the one-step transitions of a data folder | yes |
| `pinn_train.py` | Trains the model (Adam, cosine decay) and saves a `.pkl` | yes |
| `pinn_eval.py` | One-step error, learned gravity against the truth, actuation matrix, learned damping | yes |
| `pinn_check_dynamics.py` | Learned inertia, Coriolis and gravity terms against PyBullet along the tracking path | yes |
| **Model-free control** | | |
| `pinn_control.py` | Regulation with the learned gravity (Eq. 13). Prints the offset next to its prediction (Eq. 16) | yes |
| `pinn_track.py` | Joint-space tracking with the learned feedforward (Eq. 25) | yes |
| `pinn_contact.py` | Push onto a table or the box with learned gravity. Contact force against the commanded force and its prediction from statics | yes |
| **Figures** | | |
| `report_plots.py` | The report figures and the summary table, from the logs of the scripts above | yes |
| `pinn_contact_plot.py` | Plots `pinn_contact.py --log` files | yes |
| **Helpers and examples** | | |
| `cli_utils.py` | Shared `-h` formatter | – |
| `classic_control.py`, `classic_control_gripper.py` | Examples of the original repository: gravity compensation, with and without the gripper | no |
| **Cluster** (`hpc/`) | | |
| `setup_hpc.sh` | One-time virtual environment with JAX for CUDA on a cluster login node | – |
| `hpc_train.sh` | Slurm job: train and evaluate one seed per GPU (`--array`); settings as environment variables, documented at the top | – |

## How the scripts were used

The report's results come from these steps, run from `control_methods_simulation_group6`. Data, models and logs go to `data/`, `models/` and `runs/`, figures to `figs/`.

1. **Collect data without a model:** 100 trajectories of 20 s at 1 kHz with a plain PD controller.
   ```bash
   python scripts/pd+gravity_control_data.py --controller pd --n-traj 100 --dt 0.001 --dither 1.0 --log-truth --out data/pd_1khz
   ```
2. **Train** five seeds and **evaluate** them. On a single machine:
   ```bash
   python scripts/pinn_train.py data/pd_1khz --actuation network --epochs 30 --seed 0 --out models/pd0_s0.pkl
   python scripts/pinn_eval.py models/pd0_s0.pkl data/pd_1khz
   ```
   The report's models were trained on the RUG's HPC, 15 to 20 minutes per seed: `bash hpc/setup_hpc.sh` once on the login node (a virtual environment with JAX for CUDA), then `hpc/hpc_train.sh --array=0-4` (its defaults are the final recipe), which trains and evaluates one seed per GPU. The partition and GPU options at the top of `hpc/hpc_train.sh` depend on the cluster.
3. **Closed loop** for each model, with logs. Add `--gui` to watch.
   ```bash
   python scripts/pinn_control.py models/pd0_s0.pkl --log runs/pd0_s0_reg.csv
   python scripts/pinn_track.py models/pd0_s0.pkl --log runs/pd0_s0_trk_learned.csv
   python scripts/pinn_contact.py models/pd0_s0.pkl --scene box --log runs/pd0_s0_box.csv
   ```
   The same commands with `--gravity true` (or `--ff true`) give the exact-model reference.
4. **Figures:** `python scripts/report_plots.py`.

Other experiments in the report used the same scripts with other options:
- **Sampling rate and data controller:** data at `--dt 0.01` and with `--controller model` (computed torque with the exact model), trained with `--a-reg 0.01`.
- **Identifiability of actuation and damping:** data with `--inject-a --inject-d`, trained with and without `--a-reg`, evaluated with `pinn_eval.py`, which reads the injected matrices from the data folder.
- **Coverage of the training data:** `data_ranges.py` for the poses used in the experiments.
- **Learned terms:** `pinn_check_dynamics.py` for every model (used for the progression figure).
- **Shared contact scene:** `pinn_contact.py --scene box` and `impedance_contact.py`.

## Simulator interface

The robot state (joint and end-effector positions and velocities) is available through [PyBulletRobotState](ur_simulation/classic_control/robot_state/pybullet_robot_state.py), and the Jacobian and dynamics (gravity vector, mass matrix, Coriolis vector) through [PybulletRobotModel](ur_simulation/classic_control/robot_models/pybullet_robot_model.py). `scripts/classic_control.py` shows how to read them and apply joint torques.
