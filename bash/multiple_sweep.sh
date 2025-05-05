#!/bin/bash
#SBATCH --account=def-martin4
#SBATCH --job-name=vrp-sweep-model
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sweep-%A_%a.log
#SBATCH --array=0-5
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

# Launch sweep for this model and extract its ID
SWEEP_CONFIG="$SLURM_SUBMIT_DIR/sweep/sweep_${MODEL}.yaml"
SWEEP_ID=$(wandb sweep "$SWEEP_CONFIG" | grep "Created sweep with ID:" | awk '{print $NF}')

echo "Launching sweep for model=$MODEL (SWEEP_ID=$SWEEP_ID)"
wandb agent "$SWEEP_ID"
