This is a project to train a model to predict if a route will be modified by a planner or not. Then the objecive is to visualize where it will be modified.

The scripts are 'train.py' ans 'visualize.py'. The first one is used to train the model and the second one is used to visualize the results.

## training
    python train.py <config_change1> <config_change2>
<config_change1> can be model=cnn


# visualize


## To do : 
From a dataset of optimal routes and modified routes, train a vision model that can :
- give probability of modification
- visualize where it will be modified.
- inclure une fonction de loss qui prend en compte la localisation de la modification.
- créer un masque de la route modifiée. Ok, à améliorer
- automatiser, fichier de log, 
- hyperparamètres, augmentation de données,
- automatiser la heatmap pour des batches.
- faire un multi avec VGG
- 

## Ideas : 
Using ViT : Use a ViT to predict if a route will be modified or not. Use attention to visualize where it will be modified.

Using CNN : Use a CNN to predict if a route will be modified or not. Use Grad-CAM to visualize where it will be modified.

Using masks as labels and have a model that predicts the mask of the modified route. The mask can be 



Bosser sur la modification des images : plus de contraste, grossir les traits...

TODO : uniformiser avec des hydra.utils.instatiate