#!/bin/bash
#SBATCH --account=def-martin4
#SBATCH --job-name=solver
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/solver-%A_%a.log
#SBATCH --export=ALL,WANDB_API_KEY
#SBATCH --array=0-8


# a_idx = task_id / (nb * nc)
# b_idx = (task_id / nc) % nb
# c_idx = task_id % nc

list_thresholds = (0.0000002)
list_walking = (0.1 0.5 1 5)
list_multiplier = (0.1 0.5 1)

threshold=${list_thresholds[$SLURM_ARRAY_TASK_ID / 9]}
walking=${list_walking[($SLURM_ARRAY_TASK_ID / 3) % 4]}
multiplier=${list_multiplier[$SLURM_ARRAY_TASK_ID % 3]}

GUROBI_VERSION="11.0.0"
GUROBI_BASE="/cvmfs/restricted.computecanada.ca/easybuild/software/2020/Core/gurobi/${GUROBI_VERSION}/lib"
GUROBI_TMP_LIB="$SLURM_TMPDIR/gurobi_lib"

echo $SLURM_TMPDIR

# === Créer le dossier temporaire ===
mkdir -p "$GUROBI_TMP_LIB"

module load python/3.11 scipy-stack/2023b opencv/4.10.0
module load StdEnv/2023
module load gurobi/$GUROBI_VERSION
module load java/21.0.1

# === Copier uniquement les fichiers nécessaires ===
cp "$GUROBI_BASE"/*.so "$GUROBI_TMP_LIB/"
cp "$GUROBI_BASE"/gurobi.jar "$GUROBI_TMP_LIB/"


jar tf $GUROBI_TMP_LIB | grep GRBException

# === Affichage ===
echo "✅ Fichiers copiés dans $GUROBI_TMP_LIB"
echo "➡️ Lancement Java avec classe $MAIN_CLASS"

# venv sur nœud
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r "$SLURM_SUBMIT_DIR/requirements-clean.txt"

python optimized_vrp_pipeline.py solver=host +threshold=$threshold +walking=$walking +multiplier=$multiplier
