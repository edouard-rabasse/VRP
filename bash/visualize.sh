#!/bin/bash
#SBATCH --account=def-martin4
#SBATCH --job-name=visualise
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/visualize-%A_%a.log
#SBATCH --export=ALL,WANDB_API_KEY

module load python/3.11 scipy-stack/2023b opencv/4.10.0

# venv sur nœud
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r "$SLURM_SUBMIT_DIR/requirements-clean.txt"

python visualize.py model.weight_path=checkpoints/stored/vgg_32_30.pth model=vgg 
