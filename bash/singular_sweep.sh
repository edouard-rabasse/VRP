#!/bin/bash
#SBATCH --account=def-martin4
#SBATCH --job-name=vrp-sweep-model
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sweep-%A_%a.log
#SBATCH --export=ALL,WANDB_API_KEY

module load python/3.11 scipy-stack/2023b opencv/4.10.0

# Setup virtualenv
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r "$SLURM_SUBMIT_DIR/requirements-clean.txt"


### comment the second line and uncomment the first to use the local wandb ###
# wandb offline
wandb login --relogin "$WANDB_API_KEY"

# 

# Define models array
MODEL="vgg_seg"

SWEEP_CONFIG="$SLURM_SUBMIT_DIR/sweep/sweep_${MODEL}.yaml"


create_sweep_id() {
    NEW_ID=$(wandb sweep --project "VRP" "$SWEEP_CONFIG" 2>&1 | grep "Creating sweep with ID" | awk '{print $NF}')
    if [ -n "$NEW_ID" ]; then
        echo "$MODEL=$NEW_ID" >> "$SWEEP_ID_FILE"
        echo "$NEW_ID"
    else
        echo "ERROR: Failed to create sweep for $MODEL" >&2
        exit 1
    fi
}

SWEEP_ID=$(create_sweep_id)

# SWEEP_ID=$(grep "^$MODEL=" "$SLURM_SUBMIT_DIR/sweep/sweep_ids.txt" | cut -d= -f2)

echo "Launching sweep for model=$MODEL (SWEEP_ID=$SWEEP_ID)"

# Lancer agent
wandb agent "polytechnique-rabasse/VRP/$SWEEP_ID"

echo "Launching sweep for model=$MODEL (SWEEP_ID=$SWEEP_ID)"
