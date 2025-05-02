#!/bin/bash
#SBATCH --account=def-martin4      # ton compte Compute Canada
#SBATCH --job-name=vrp-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2 # At least two cpus is recommended - one for the main process and one for the wandB process
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%A_%a.log      # log par tâche Array
#SBATCH --array=1-6               
#SBATCH --export=ALL,WANDB_API_KEY

# -----------------------------------------------------------------------------
# 1. Modules
# -----------------------------------------------------------------------------
module load python/3.11
module load scipy-stack/2023b
module load opencv/4.10.0

# -----------------------------------------------------------------------------
# 2. Virtualenv sur le disque du nœud
# -----------------------------------------------------------------------------
echo "Creating venv in $SLURM_TMPDIR/env ..."
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"

# pip à jour
pip install --no-index --upgrade pip

# -----------------------------------------------------------------------------
# 3. Installation des dépendances
# -----------------------------------------------------------------------------
pip install --no-index -r "$SLURM_SUBMIT_DIR/requirements-clean.txt"

# -----------------------------------------------------------------------------
# 3.1. WandB non-interactif
# -----------------------------------------------------------------------------
# (assure-toi d'avoir lancé `export WANDB_API_KEY=<ta_cle>` avant sbatch)
python -m wandb login --relogin "$WANDB_API_KEY"

python -m wandb online


# -----------------------------------------------------------------------------
# 4. Multirun Hydra selon la tâche Array
# -----------------------------------------------------------------------------
# Définit les combinaisons à tester
declare -a OVERRIDES
OVERRIDES[1]="model=vgg model.params.epochs=20,50,100 model.params.batch_size=8,16,32 model.params.learning_rate=0.001,0.0005,0.00001,0.000005"
OVERRIDES[2]="model=resnet model.params.epochs=20,50,100 model.params.batch_size=8,16,32,64 model.kernel_size=3,5,7,15 model.params.learning_rate=0.001,0.0005,0.00001,0.000005"
OVERRIDES[3]="model=deit_tiny model.params.epochs=20,50,100 model.params.batch_size=8,16,32 model.params.learning_rate=0.0001,0.0005,0.00001"
OVERRIDES[4]="model=multi model.params.epochs=20,50,100 model.params.batch_size=8,16,32 model.params.learning_rate=0.001,0.0005,0.00001,0.000005"
OVERRIDES[5]="model=cnn model.params.epochs=20,50,100 model.params.batch_size=8,16,32 model.params.learning_rate=0.001,0.0005,0.00001,0.000005"
OVERRIDES[6]="model=MFCN model.params.epochs=20,50,100 model.params.batch_size=8,16,32 model.params.learning_rate=0.001,0.0005,0.00001,0.000005"


# Sélectionne la ligne correspondante à SLURM_ARRAY_TASK_ID
CMD_OVR="${OVERRIDES[$SLURM_ARRAY_TASK_ID]}"
echo "[Task $SLURM_ARRAY_TASK_ID] Overrides: $CMD_OVR"

# Exécute la multirun Hydra (note le -m)
python "$SLURM_SUBMIT_DIR/train.py" -m $CMD_OVR
