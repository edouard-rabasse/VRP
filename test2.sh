#!/usr/bin/env bash
set -e

# 1) Assurez-vous que conda est disponible
if command -v conda &>/dev/null; then
  # injecte conda dans le shell
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate VRP
else
  echo "[WARN] conda non trouvé, assurez-vous d'activer manuellement votre env VRP"
fi

# 2) Déplacez-vous dans le dossier du script pour avoir le bon chemin
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# # 3) Installez (ou mettez à jour) vos deps
# pip install --no-index --upgrade pip
# pip install --no-index -r requirements.txt

# 4) Lancez votre code
python train.py -m model=cnn,deit_tiny \
       model.params.epochs=1,2