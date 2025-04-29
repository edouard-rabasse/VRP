#!/bin/bash
#SBATCH --account=def-martin4      # ← ton compte CC (def-xxx ou rrg-xxx)
#SBATCH --job-name=vrp-train
#SBATCH --gres=gpu:1                # ou --cpus-per-task, selon ton besoin
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.log          # fichier de log : vrp-train-<jobid>.log

# -----------------------------------------------------------------------------
# 1. Modules
# -----------------------------------------------------------------------------
module load python/3.11            # même version que pour requirements.txt
# Si tu as besoin de NumPy / SciPy compilés par Compute Canada :
module load scipy-stack/2023b

# -----------------------------------------------------------------------------
# 2. Virtualenv temporaire sur disque local du nœud
# -----------------------------------------------------------------------------
echo "Creating venv in $SLURM_TMPDIR/env ..."
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"

# pip récent
pip install --no-index --upgrade pip

# -----------------------------------------------------------------------------
# 3. Installation des dépendances
# -----------------------------------------------------------------------------
pip install --no-index -r $SLURM_SUBMIT_DIR/requirements.txt

# -----------------------------------------------------------------------------
# 4. Lancement du programme
# -----------------------------------------------------------------------------
echo "Starting training..."
python $SLURM_SUBMIT_DIR/test.py #--config config.yaml

# (optionnel) Sauvegarde d’artefacts dans $PROJECT ou $SCRATCH
# cp checkpoints/*.pth $PROJECT/VRP/checkpoints/$SLURM_JOB_ID/
