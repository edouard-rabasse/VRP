#!/bin/bash
#SBATCH --account=def-martin4
#SBATCH --job-name=gurobi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=logs/gurobi-%A_%a.log




# === Fichiers d'entrée ===
EXPERIMENT_FILE="ExperimentsAllSets.txt"
CONFIG_FILE="configuration2.xml"
OTHER_ARGS="1 8"



module load StdEnv/2023
module load gurobi/$GUROBI_VERSION
module load java/21.0.1



cd MSH/MSH
java \
  -Djava.library.path="lib" \
  -jar jars/MSH_handmade.jar \
  ExperimentsAllSets.txt 302 configuration1.xml 1 8