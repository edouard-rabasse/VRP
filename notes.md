# Notes

- séparer sets
plus de données
- config 2
- pas de back propagation pour mask nul
- masque avec juste les arêtes à enlever. -> les prédire.
tester si c'est plus simple.
code de couleur sur le masque.

- autres métriques

proba sur les arêtes
arêtes voisines
pixels plus gros.

## 24/04

Modèle Distances
stride plus petits ou plus grands.
calcul canada : bash script.

Avec config 6 ou 7.
GNN
GAN, matrices d'adjacence.
plus de rotations.

## TODO

Clean pour les hyperparamètres.

## 06/05

Clients différents
Nombre de clients différents.

Points d'attention.

- Licence Gurobi pour alliance.

Comment automatiser les points d'attention pour modifier la boucle.

Simplifier en un modèle plus simple pour mettre dans un solver.
Changer 4 arcs. Minimiser le nombre de variables modifiées tout en passant la proba sous un seuil.

Faire un modèle pour prédire le nombre d'arcs à changer. (3e loss)

tester avec une autre configuration (2)

- \+ et - de clients.

Benchmark ? Si on connaissait la valeur de la contrainte. Distance en coût + dépassement de la vraie contrainte.
Vecteur de features avec les instances. features : nb de tournées à pied, distances, distance à pied la plus longue, moyenne, en voiture.

Sinon heuristique qui fait des mouvements dans la région locale.

Faire des mouvements pour diminuer le treshold.
 identifier la tournée avec les coordonnées.

nouvelle config avec le nombre de boucle à pied <=1.

## 20/05

- simuler des zones de parking
- avoir un modèle qui prend direct les coordonnées.
- implémenter métriques pour la heatmap

## 27/05

Zones de parking

- grosse zone parking sans data augmentation. 1/4 de l'image
- zones piétonnes
- zones difficiles à stationner
- changer la couleur des noeuds
- réseau routier, en background
  - pipeline complète (évaluation)
- heuristique. solutions qui respectent la contrainte.
- énumérer puis garder les x% meilleurs en terme de distance. Précalculer la matrice des distances (pour les arcs modifiés).
  - contrainte voiture/piéton
  - métriques
  - benchmark avec padding.
- calculer différentes features, plus de features.
- benchmark avec le rectangle.
- faire le heatmap sans plot. Une classe par arc (avec padding)
- ajouter des métriques pour évaluer les performances.
