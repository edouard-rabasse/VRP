# Wandb
# Pour run wandb : 
Aller chercher une clef API sur wandb puis faire un

    export WANDB_API_KEY=xxx 
dans le terminal avant de lancer le script.

Puis 

    wandb login --relogin "$WANDB_API_KEY"


## Pour créer un sweep
A partir d'un fichier sweep.yaml, faire un 

    wandb sweep sweep.yaml
Ensuite, il faut récupérer l'ID du sweep et faire un 

    wandb agent <sweep_id>

## Pour clone un github : 
Obtenir le PAT (Personal Access Token) sur github puis faire un 

    git clone https://<PAT>@github.com/username/repo.git 
dans le terminal.

## pour obtenir la liste des requirements clean
    pip list --format=freeze --exclude-editable > requirements-clean.txt
Pas de paquet d'une build locale

## Pour modifier un élément de config hydra
    python main.py +config=path.to.config
Sachant que 
- "+" est un opérateur d'ajout
- "++" est pour ajouter ou modifier un élément de config
- "-" est pour supprimer un élément de config
- "" est pour remplacer un élément de config


## Si vsc sur ssh bug
s'y connecter puis 
rm -rf ~/.vscode-server

## Pour faire un commit sur git
    git add .
    git commit -m "message"
    git push origin <branch_name>