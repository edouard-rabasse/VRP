## The doubly open park-and-loop routing problem
https://www.sciencedirect.com/science/article/abs/pii/S0305054822000570

3 constraints:
- daily walking distance
- route duration
- subtour duration limit

Heuristic : route-first, assemble-second
generate a set of route with TSP heuristics
then recombining some of the rout using a set partitioning model.

DOPLRP : generalisation of the VRP (vehicle routing problem).
To continue (p13)


## Grad-CAM : Visual explanations from deep networks via gradient-based localization
https://arxiv.org/abs/1610.02391


## PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization
https://arxiv.org/pdf/2011.08785v1
On prend un CNN préentrainé --> sort les outputs relevant.
On prend un patch de l'image et on fait le moyenne et variance des embeddings pour toutes les images, on obtient une matrice d'estimation.

On a $X_{ij} = \{x_{ij}^k, k\in [1,N]\} $ $x_{ij}^k$ est le patch de l'image $k$ dans la position $(i,j)$.
En gros $x_{ij}^k$ C'est le vecteur qui contient les embeddings du patch $ij$ de l'image $k$, aux différents niveaux de la CNN, concaténés.

$\Sigma_{i,j} = \frac{1}{N-1} \sum_{k=1}^{N} (x_{ij}^k - \mu_{ij})(x_{ij}^k - \mu_{ij})^T +\epsilon I$

Score anomalie : 
Mahalanobis distance entre le patch et la distribution de patchs normaux.
$M(x_{ij}) = \sqrt{(x_{ij} - \mu_{ij})^T \Sigma_{ij}^{-1} (x_{ij} - \mu_{ij})}$
Les scores élevés sur cette map indiquent une zone anormale.

--> voir anomalib


# Image Splicing Localization Using A Multi-Task Fully Convolutional Network (MFCN)
https://ar5iv.labs.arxiv.org/html/1709.02016#:~:text=localization%20output%20in%20certain%20cases,finer%20localization%20than%20the%20SFCN
Using Fully COnvolutional Network

Plus pour photoshop. FCN VGG-16
With skip connection. Plus axé sur les photos -> Un peu HS

## Quality analysis for the VRP solutions using computer vision techniques
https://www.researchgate.net/publication/320010264_Quality_analysis_for_the_VRP_solutions_using_computer_vision_techniques
C'est pas les graphes qu'on a. C'est plutôt en utilisant les coordonnées parallèles pour un nombre de caractéristiques. On a ensuite une distinction visuelle entre les solutions optimales et les mauvaises.
--> A explorer ?

## Image Splicing Localization Using a Multi Task Fully COnvolutional Network
https://ar5iv.labs.arxiv.org/html/1709.02016#:~:text=localization%20output%20in%20certain%20cases,finer%20localization%20than%20the%20SFCN

Uses a modified version of VGG-16, but they add batch normalization and class weighting.

## Classification of urban morphology with deep learning: Application on urban vitality
https://www.sciencedirect.com/science/article/pii/S0198971521001137
For classification of road networks.