import torch
import torch.nn as nn
import os


def load_deit(weights_path,device='cpu'):
    """
    Charge le modèle tiny DEIT de Facebook, modifie sa dernière couche (head)
    pour qu'elle ait deux sorties, et y affecte les poids contenus dans le fichier
    spécifié par weights_path.

    Parameters:
        weights_path (str): Chemin vers le fichier contenant les poids pour la tête.

    Returns:
        model: Le modèle DEIT modifié avec la tête à 2 sorties et les poids chargés.
    """
    # Charger le modèle pré-entraîné depuis torch.hub
    model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
    
    # Modifier la couche de classification (head) pour avoir 2 sorties.
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, 2)
    
    # Charger les poids personnalisés pour la nouvelle tête depuis le fichier
    head_state_dict = torch.load(weights_path, map_location=torch.device(device))
    model.head.load_state_dict(head_state_dict)
    
    
    return model


def load_model(model_name, device, cfg):
    
    if model_name == 'cnn':
        from src.models.VisualScoringModel import VisualScoringModel
        image_size = cfg.image_size
        input_shape = (3, image_size[0], image_size[1])

        model = VisualScoringModel(input_shape=input_shape).to(device)

    elif model_name == 'deit_tiny':
        # from src.models.deit_tiny import load_deit
        # model = load_deit(model_name, device, out_features=2)
        # model.to(device)
        try:
            # model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
            import timm

            model = timm.create_model('deit_tiny_patch16_224', pretrained=True, num_classes=2)
        except RuntimeError as e:
            print(f"Error loading model from torch hub: {e}")
            print("Attempting to load model manually...")
            # Manual loading code here (if available)
            raise

    # Changing the last layer to have 2 classes
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, 2)
        model.to(device)
        
    elif model_name == 'multi_task':
        from src.models.MultiTaskVisualModel import MultiTaskVisualScoringModel
        image_size = cfg.image_size
        input_shape = (3, image_size[0], image_size[1])
        mask_shape = (10,10)

        model = MultiTaskVisualScoringModel(input_shape=input_shape,mask_shape = mask_shape).to(device)

    elif model_name == 'vgg':
        from src.models.vgg import load_vgg
        model = load_vgg()
        model.to(device)
    elif model_name == 'MFCN':
        from src.models.MFCN import MultiTaskVGG
        model = MultiTaskVGG(mask_shape=cfg.mask_shape)
        model.to(device)
    else:
        raise ValueError("Unknown model name: {}".format(model_name))

    # Load the model weights
    if cfg.load_model :
        if os.path.exists(cfg.weight_path):
            print(f"Loading weights from {cfg.weight_path}")
            model.load_state_dict(torch.load(cfg.weight_path, map_location=device))
        else:
            raise FileNotFoundError(f"Weight file not found: {cfg.weight_path}")

    return model

def save_model(model, path):
    """
    Save the model state dictionary to the specified path.
    
    Parameters:
        model (torch.nn.Module): The model to save.
        path (str): The path where the model will be saved.
    """
    torch.save(model.state_dict(), path)
