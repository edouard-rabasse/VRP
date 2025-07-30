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
  - pipeline complète (évaluation) --> dans graph flagging
- heuristique. solutions qui respectent la contrainte.
- énumérer puis garder les x% meilleurs en terme de distance. Précalculer la matrice des distances (pour les arcs modifiés).
  - contrainte voiture/piéton
  - métriques
  - benchmark avec padding.
- calculer différentes features, plus de features.
- benchmark avec le rectangle.
- faire le heatmap sans plot. Une classe par arc (avec padding)
- ajouter des métriques pour évaluer les performances.

## 03/06

- Dans le TSP, garder les circuits comme ville de départ, ville d'arrivée.
- plus proches voisins randomisé.
- interdire arcs pointillé.
- augmenter de façon artificielle le coût des arcs pointillés (1+lambda) par mode. Puis rerun.
- mettre un coût initial pour les arcs à pied
- faire la fonction de loss heatmap

## 10/06

Solutions qui n'augeentent pas le prix.

Mesurer le nombre de fois qu'on a SP = SI = SPI
idem pour SP != SI = SPI -> mesurer le delta. Combien on perd en qualité de la solution.
SP != SI != PI

- cas dans lequel f(SI) < f(SPI)
- autre cas.
filtre les instances qui changent pas.

Planificateur bête :

- Si l'arc est trop long, prend la voiture pour cet arc. Veiller à ce qu'on ait toujours assez de distances pour evenir à la voiture à pied.

## 17/06

carré sans stationnement
inspection manuelle des cas différents :  vérifier si on est loin des arcs à modifier.
Pour les arcs qui ont été flag, distance minimale avec les vrais arcs
un seul arc par itération.
identifier les valeurs lorsque la solution est valide. Puis regarder retrospectivement.
faire des graphes.
à quel point on est proches d'être valide

Métrique heatmap : combien d'arcs correctement identifiés et inversement. (courbes ROC/AUC)

## 03/07

De combien on respecte pas la contrainte. Ou combien de fois on respecte pas la contrainte.
Faire tourner plus longtemps.
Critère d'arrêt : si on ajuste deux fois le même arc.
Ne pas l'arrêter. Seuil, s'il est valide.
Essayer avec juste une arête. à quel point l'arc flaggé est problématique.

## 10/07

- Si pas d'arc problématique, tous les arcs devraient avoir les mêmes moyennes. Regarder le top 5/10 des arcs. Regarder la variance de la heatmap, entropie. Cross-entropy, seuil.
- 10%, 30% puis rise.
- Nombre d'itération sans amélioration.
- est-ce que ça fonctionne pour le rectangle ?

Plusieurs critères d'arrêt. Générer toutes les données. Arbre de décision. Pour une même instance, qu'est-ce qui permet de la classifier comme valide.

- durée en temps
- nombre d'iterations,
- valeur de classification,
- entropie de la heatmap,
- ajouter la score pour la valeur suivante. (pour itérer)

Limiter la profondeur de l'arbre., y=valide ou non.

Faire des courbes ROC/AUC pour les arcs flaggés.

## 22/07

- le comportement n'est pas celui du planner : on crée des nouveaux chemins à pied.
- tracer l'évolution
- critère d'arrêt, si valide ok, sinon relancer.
- faire un random search sur les différents paramètres à utiliser (score, variation...) et leurs valeurs.
- train,test,val
- essayer d'indentifier quelles valeurs (à l'aide des courbes).
- max \sum y+ - \sum y- , x>=0, y \in {9,1
- score y <=x

challenge amazon. Transportation science
Papier de Jorge Mortes Alcaraz.

Exploitation des éléments visuels.

Contraintes managériales vs réalité du terrain. flow chart -> logiciel, planificateur.
Zoom dans la boîte planifcateur

Experimentations,
Résultats.
Expliquer sommairement l'algo de Nicolas
Les modèles de ML
La génération des données.

Nettoyer le code.

# 29/07
augmenter dataset
combiner segmentation et class pour le rectangle ?

Proposer 3, 6 ou 9
Prendre la meilleure et la pire parmi ces 3, 6 ou 9.
