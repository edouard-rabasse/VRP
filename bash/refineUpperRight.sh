#!/bin/bash
#SBATCH --account=def-martin4
#SBATCH --job-name=refineUpperRight
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/refineUpper-%A_%a.log


# === Configuration ===
GUROBI_VERSION="11.0.0"
GUROBI_BASE="/cvmfs/restricted.computecanada.ca/easybuild/software/2020/Core/gurobi/${GUROBI_VERSION}/lib"
GUROBI_TMP_LIB="$SLURM_TMPDIR/gurobi_lib"
# MAIN_CLASS="main.Main_refineSolution_v2"
# MAIN_CLASS="main.Main_gurobi"
MAIN_CLASS="main.Main_refineUpperRightConstraint"
BIN_DIR="bin"
SRC_DIR="src"
JAR="$GUROBI_TMP_LIB/gurobi.jar"
CP_COMPILE="$BIN_DIR:$GUROBI_JAR"                     # javac doesn't need BIN_DIR
CP_RUN="$BIN_DIR:$GUROBI_JAR"  

# === Fichiers d'entrée ===
EXPERIMENT_FILE="ExperimentsAllSets.txt"
CONFIG_FILE="configurationCustomCosts.xml"
# OTHER_ARGS="1 8"

# === Créer le dossier temporaire ===
mkdir -p "$GUROBI_TMP_LIB"


module load StdEnv/2023
module load gurobi/$GUROBI_VERSION
module load java/21.0.1

# === Copier uniquement les fichiers nécessaires ===
cp "$GUROBI_BASE"/*.so "$GUROBI_TMP_LIB/"
cp "$GUROBI_BASE"/gurobi.jar "$GUROBI_TMP_LIB/"

# === Vérifier que tout a bien été copié ===
if [[ ! -f "$JAR" ]]; then
  echo "❌ ERREUR : gurobi.jar non copié dans $GUROBI_TMP_LIB"
  exit 1
fi




cd MSH/MSH
# === Affichage ===
echo "✅ Fichiers copiés dans $GUROBI_TMP_LIB"
echo "➡️ Lancement Java avec classe $MAIN_CLASS"


echo "🔍 Vérif classpath :"
ls -lh $GUROBI_TMP_LIB/gurobi.jar


## ===== Recompilation du code Java =====
echo "⏳ (re)compiling …"         # list up‑to‑date files

javac  -cp "$CP_RUN"  -d "$BIN_DIR"  src/main/CreateInstances2.java
javac  -cp "$CP_COMPILE"  -d "$BIN_DIR"  src/main/Main_refineSolution_v2.java
javac  -cp "$CP_COMPILE"  -d "$BIN_DIR"  src/main/Main_refineUpperRightConstraint.java
javac  -cp "$CP_COMPILE"  -d "$BIN_DIR"  src/main/Main_gurobi.java

echo "✅ compilation done"

# # === Lancement de la commande Java ===


# java \
#   -Djava.library.path="$GUROBI_TMP_LIB" \
#   -cp "$BIN_DIR:$JAR" \
#   main.CreateInstances2 10 10 20 1001 3000


for i in {1001..3000}; do
java \
  -Djava.library.path="$GUROBI_TMP_LIB" \
  -cp "$BIN_DIR:$JAR" \
  "$MAIN_CLASS" "$EXPERIMENT_FILE" $i  "$CONFIG_FILE" 
done