#!/bin/bash
#SBATCH --account=def-martin4
#SBATCH --job-name=vrp-sweep-model
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sweep-%A_%a.log
#SBATCH --array=4
#SBATCH --export=ALL,WANDB_API_KEY

module load python/3.11 scipy-stack/2023b opencv/4.10.0

# Setup virtualenv
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r "$SLURM_SUBMIT_DIR/requirements-clean.txt"

wandb login --relogin "$WANDB_API_KEY"

# Define models array
MODELS=("vgg" "resnet" "deit_tiny" "multi" "cnn" "MFCN")
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

SWEEP_CONFIG="$SLURM_SUBMIT_DIR/sweep_${MODEL}.yaml"


SWEEP_ID=$(grep "^$MODEL=" "$SLURM_SUBMIT_DIR/sweep/sweep_ids.txt" | cut -d= -f2)

echo "Launching sweep for model=$MODEL (SWEEP_ID=$SWEEP_ID)"

# Lancer agent
wandb agent "polytechnique-rabasse/VRP/$SWEEP_ID"

echo "Launching sweep for model=$MODEL (SWEEP_ID=$SWEEP_ID)"
