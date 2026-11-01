#!/bin/bash
#SBATCH --job-name=pinn_train
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --no-requeue
#
# Train the LNN on 1 kHz data (pinn_train.py), then evaluate it
# (pinn_eval.py). Check the partition name and the GPU option in the
# cluster docs, they are the lines most likely to need changing.
#
# Run it from anywhere: started outside Slurm, the script moves to the
# repository root (the folder above hpc/, which holds scripts/ and data/),
# creates logs/ and models/ and submits itself. Options are passed on to
# sbatch:
#
#   hpc/hpc_train.sh                            one run, seed 0
#   hpc/hpc_train.sh --array=0-4                five seeds, one GPU each
#   EPOCHS=1 hpc/hpc_train.sh --time=00:30:00   timing test
#   A_REG=0 ACTUATION=identity hpc/hpc_train.sh other settings
#   DATA=data/inject_1khz TAG=inject hpc/hpc_train.sh   other data set
#
# The defaults are the final recipe of the report (PD data at 1 kHz, no
# actuation penalty, 30 epochs), so five seeds are
#   hpc/hpc_train.sh --array=0-4
#
# CPU-only run: delete the --gpus-per-node line above (some clusters
# reject --gpus-per-node=0), then, with the partition from the cluster docs:
#   REQUIRE_GPU=0 ./hpc_train.sh --partition=regular
#
# Settings are environment variables with the defaults below. In an array
# job the array index is the seed. The venv comes from hpc/setup_hpc.sh.

if [ -z "${SLURM_JOB_ID:-}" ]; then
    script=$(readlink -f "$0")
    cd "$(dirname "$script")/.." || exit 1
    mkdir -p logs models
    # --export=ALL: the settings below reach the job even on clusters that
    # default to a clean environment
    exec sbatch --export=ALL "$@" "$script"
fi

DATA=${DATA:-data/pd_1khz}
SCRIPTS=${SCRIPTS:-scripts}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-256}
ACTUATION=${ACTUATION:-network}
A_REG=${A_REG:-0}
TAG=${TAG:-pd}  # first part of the model name, one per data set
VENV=${VENV:-$HOME/venvs/pinn}
REQUIRE_GPU=${REQUIRE_GPU:-1}
SEED=${SLURM_ARRAY_TASK_ID:-${SEED:-0}}

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

# the venv brings its own Python and CUDA libraries, so no modules are
# loaded; purge keeps loaded CUDA modules from clashing with them
module purge
source "$VENV/bin/activate"
set -u
export PYTHONUNBUFFERED=1

echo "host: $(hostname)  job: $SLURM_JOB_ID  seed: $SEED"
nvidia-smi --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader || echo 'no GPU visible'

python - <<'EOF'
from importlib.metadata import version
import jax
print(*(f'{p} {version(p)}' for p in ('jax', 'jaxlib', 'optax', 'numpy')))
print('devices:', jax.devices())
EOF

if [ "$REQUIRE_GPU" = 1 ]; then
    python - <<'EOF'
import sys
import jax
if jax.devices()[0].platform != 'gpu':
    sys.exit('JAX runs on the CPU, not on the GPU: stopping')
EOF
fi

NAME="hpc_${TAG}_${ACTUATION}_areg${A_REG}_e${EPOCHS}_s${SEED}"

python -u "$SCRIPTS/pinn_train.py" "$DATA" \
    --epochs "$EPOCHS" --batch-size "$BATCH" \
    --actuation "$ACTUATION" --a-reg "$A_REG" --seed "$SEED" \
    --out "models/$NAME.pkl"

python -u "$SCRIPTS/pinn_eval.py" "models/$NAME.pkl" "$DATA" \
    | tee "models/$NAME.eval.txt"

echo "done: models/$NAME.pkl"
