#!/bin/bash
#SBATCH --account=def-martin4
#SBATCH --job-name=vrp-sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/wandb-agent-%A_%a.log
#SBATCH --export=ALL,WANDB_API_KEY

module load python/3.11 scipy-stack/2023b opencv/4.10.0

# venv sur nœud
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r "$SLURM_SUBMIT_DIR/requirements-clean.txt"

# login W&B
wandb login --relogin "$WANDB_API_KEY"

# votre sweep ID
# SWEEP_ID=polytechnique-rabasse/VRP/d6ovvlqc
SWEEP_ID=polytechnique-rabasse/VRP/e7ssemue

# chaque tâche Array lance un agent
wandb agent "$SWEEP_ID"
