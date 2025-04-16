To do : 
From a dataset of optimal routes and modified routes, train a vision model that can :
- give probability of modification
- visualize where it will be modified.
- inclure une fonction de loss qui prend en compte la localisation de la modification.
- créer un masque de la route modifiée. Ok, à améliorer
- automatiser, fichier de log, 
- hyperparamètres, augmentation de données,
- automatiser la heatmap pour des batches.
- faire un mutli avec VGG
- 

Ideas : 
Using ViT : Use a ViT to predict if a route will be modified or not. Use attention to visualize where it will be modified.

Using CNN : Use a CNN to predict if a route will be modified or not. Use Grad-CAM to visualize where it will be modified.

Using masks as labels and have a model that predicts the mask of the modified route. The mask can be 



Bosser sur la modification des images : plus de contraste, grossir les traits...

