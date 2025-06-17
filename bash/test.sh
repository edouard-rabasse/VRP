#!/bin/bash
#SBATCH --account=def-martin4
#SBATCH --job-name=gurobi
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=logs/test-%A_%a.log

# === Configuration ===
GUROBI_VERSION="11.0.0"
GUROBI_BASE="/cvmfs/restricted.computecanada.ca/easybuild/software/2023/Core/gurobi/${GUROBI_VERSION}"
GUROBI_JAR="$GUROBI_BASE/lib/gurobi.jar"
GUROBI_TMP_LIB="$SLURM_TMPDIR/gurobi_lib"
GUROBI_LIB="$GUROBI_BASE/lib"

# Dossier temporaire local SLURM
if [ -z "$SLURM_TMPDIR" ]; then
  echo "❌ SLURM_TMPDIR is not defined"
  exit 1
fi

# === Chargement des modules ===
module load StdEnv/2023
module load gurobi/${GUROBI_VERSION}
module load java/21.0.1


# === Déplacements dans le bon répertoire ===
cd MSH/MSH || { echo "❌ Cannot cd into MSH/MSH"; exit 1; }

# === Compilation Java ===
mkdir -p bin
javac -cp "bin:$EBROOTGUROBI/lib/gurobi.jar" -d bin src/main/Main_customCosts.java

echo "✅ Compilation terminée"

# === Paramètres d'exécution ===
MAIN_CLASS="main.Main_customCosts"
EXPERIMENT_FILE="Coordinates_3.txt"
COSTS_FILE="Costs_3_1.txt"
CONFIG_FILE="configurationCustomCosts2.xml"
ARCS_FILE="Arcs_3_1.txt"
SUFFIX="1"

# === Lancement ===
echo "➡️ Lancement de Java"
java \
  -Xmx6000m \
  -Djava.library.path="$EBROOTGUROBI/lib" \
  -cp "bin:$EBROOTGUROBI/lib/gurobi.jar" \
  $MAIN_CLASS $EXPERIMENT_FILE $COSTS_FILE $CONFIG_FILE $ARCS_FILE $SUFFIX
