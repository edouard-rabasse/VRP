#!/bin/bash
#SBATCH --account=def-martin4
#SBATCH --job-name=vrp-sweep
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/sweep_multi-%A_%a.log
#SBATCH --export=ALL,WANDB_API_KEY

# ── Environment setup ────────────────────────────────────────────────────────
module load python/3.11 scipy-stack/2023b opencv/4.10.0

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r "$SLURM_SUBMIT_DIR/requirements-clean.txt"

# ── WandB login ──────────────────────────────────────────────────────────────
wandb login --relogin "$WANDB_API_KEY"

sleep 720

# ── Configuration ────────────────────────────────────────────────────────────
# Ajoute ce bloc pour récupérer le modèle passé en argument
if [ -z "$1" ]; then
    echo "❌ ERROR: No model specified. Usage: sbatch launch_sweep.sh <model_name>"
    exit 1
fi

MODEL="$1"  # 👈 récupère le modèle depuis la ligne de commande
SWEEP_CONFIG="$SLURM_SUBMIT_DIR/sweep/sweep_${MODEL}.yaml"
SWEEP_ID_FILE="$SLURM_SUBMIT_DIR/sweep/sweep_id_${MODEL}.txt"

# ── Create sweep ID (task 0 only) ────────────────────────────────────────────
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    echo "[Task 0] Creating sweep from $SWEEP_CONFIG..."
    SWEEP_ID=$(wandb sweep --project "VRP" "$SWEEP_CONFIG" 2>&1 | grep "Creating sweep with ID" | awk '{print $NF}')

    if [ -n "$SWEEP_ID" ]; then
        echo "$SWEEP_ID" > "$SWEEP_ID_FILE"
        sync  # Ensure file is flushed to disk
        echo "[Task 0] Sweep created: $SWEEP_ID"
    else
        echo "[Task 0] ERROR: Failed to create sweep." >&2
        exit 1
    fi
else
    # ── Wait for sweep ID to be written ───────────────────────────────────────
    echo "[Task $SLURM_ARRAY_TASK_ID] Waiting for sweep ID... in $SWEEP_ID_FILE"
    TIMEOUT=60
    WAITED=0
    sleep 60 # Initial wait to allow task 0 to create the sweep ID file
    while [ ! -s "$SWEEP_ID_FILE" ]; do
        sleep 2
        WAITED=$((WAITED + 2))
        if [ "$WAITED" -ge "$TIMEOUT" ]; then
            echo "[Task $SLURM_ARRAY_TASK_ID] ERROR: Timeout waiting for sweep ID file." >&2
            exit 1
        fi
    done
    SWEEP_ID=$(cat "$SWEEP_ID_FILE" | tr -d '\n')
    echo "[Task $SLURM_ARRAY_TASK_ID] Retrieved sweep ID: $SWEEP_ID"
fi

# ── Launch agent ─────────────────────────────────────────────────────────────
echo "[Task $SLURM_ARRAY_TASK_ID] Starting WandB agent for sweep $SWEEP_ID..."
wandb agent "polytechnique-rabasse/VRP/$SWEEP_ID"
