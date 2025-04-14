print("[DEBUG] Loading grad_cam.py")

import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Initialisation de GradCAM pour un modèle CNN.

        Args:
            model (torch.nn.Module): Le modèle CNN sur lequel effectuer la visualisation.
            target_layer (str): Le nom de la couche sur laquelle on souhaite récupérer les feature maps.
        """
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        # Liste pour stocker les handles des hooks
        self.hook_handles = []
        # Inscription des hooks sur la couche cible
        self._register_hooks()

    def _register_hooks(self):
        """
        Inscrit des hooks pour récupérer à la fois les activations (forward) et les gradients (backward)
        de la couche cible.
        """
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                # Enregistrement du hook pour les activations (forward)
                self.hook_handles.append(module.register_forward_hook(self.save_feature_maps))
                # Enregistrement du hook pour récupérer les gradients lors de la rétropropagation
                self.hook_handles.append(module.register_full_backward_hook(self.save_gradients))

    def save_feature_maps(self, module, input, output):
        """
        Hook appelé lors de la propagation avant pour sauvegarder les feature maps.
        """
        self.feature_maps = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        """
        Hook appelé lors de la rétropropagation pour sauvegarder les gradients.
        """
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, class_index):
        """
        Calcule et retourne la carte de chaleur GradCAM pour une image d'entrée et une classe cible donnée.
        
        Args:
            input_tensor (torch.Tensor): Image d'entrée format tensor (de taille [N, C, H, W]).
            class_index (int): L'indice de la classe pour laquelle on souhaite visualiser l'activation.
        
        Returns:
            heatmap (numpy.ndarray): Carte de chaleur redimensionnée aux dimensions de l'image d'entrée.
        """
        # Réinitialiser les gradients du modèle
        self.model.zero_grad()
        # Propagation avant
        output = self.model(input_tensor)
        
        # Création du vecteur one-hot pour la classe cible
        one_hot = torch.zeros_like(output)
        one_hot[0, class_index] = 1
        
        # Rétropropagation pour calculer les gradients
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Calcul des poids par une moyenne globale (sur hauteur et largeur) des gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Pondération des feature maps par les poids et agrégation linéaire
        grad_cam_map = torch.sum(weights * self.feature_maps, dim=1)
        
        # Application de ReLU pour ne garder que les valeurs positives (régions importantes)
        grad_cam_map = F.relu(grad_cam_map)
        
        # Normalisation de la carte de chaleur pour qu'elle soit comprise entre 0 et 1
        grad_cam_map = grad_cam_map - grad_cam_map.min()
        if grad_cam_map.max() != 0:
            grad_cam_map = grad_cam_map / grad_cam_map.max()
        
        # Passage sur CPU et conversion en numpy (on suppose un batch de taille 1)
        grad_cam_map = grad_cam_map.cpu().numpy()[0]
        
        # Redimensionnement de la carte de chaleur à la taille de l'image d'entrée
        height, width = input_tensor.size(2), input_tensor.size(3)
        heatmap = cv2.resize(grad_cam_map, (width, height))
        return heatmap

    def remove_hooks(self):
        """
        Supprime les hooks pour éviter les fuites mémorielles.
        """
        for handle in self.hook_handles:
            handle.remove()
