from torch.utils.data import DataLoader
import torch

def train(model_name,model, train_loader, test_loader, num_epochs, device, learning_rate,criterion=None,cfg=None):
    """
    Trains the model using the specified parameters.
    Args:
        model_name (str): The name of the model to train.
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): The DataLoader for the training set.
        test_loader (torch.utils.data.DataLoader): The DataLoader for the test set.
        num_epochs (int): The number of epochs to train for.
        device (str): The device to train on ('cpu' or 'cuda')."""
    if model_name == 'VisualScoringModel':
        from models.VisualScoringModel import train_model
        results = train_model(
            model,
            train_loader,
            test_loader,
            num_epochs=num_epochs,
            device=device,
            learning_rate=learning_rate,
            criterion=criterion,
            )
    elif model_name == 'deit_tiny':
        # Define your training loop for DEIT here
        from models.deit_tiny import precompute_deit_tiny_features, train_deit, train_deit_no_precompute
        # print("starting precompute")
        # train_features = precompute_deit_tiny_features(model, train_loader, device=device)
        # test_features = precompute_deit_tiny_features(model, test_loader, device=device)
        
        # train_loader = DataLoader(train_features, batch_size=cfg.batch_size, shuffle=True)
        # test_loader = DataLoader(test_features, batch_size=cfg.batch_size, shuffle=False)

        # print("done precompute")
        # print("starting train")
        # train_deit(model, train_loader, test_loader, device=device, num_epochs=num_epochs, learning_rate=learning_rate, criterion=criterion)


        results = train_deit_no_precompute(model, train_loader, test_loader, device=device, num_epochs=num_epochs, learning_rate=learning_rate, criterion=criterion)
        # return train_loader # TODO: remove this line, just for debugging
    elif model_name =="multi_task":
        from models.MultiTaskVisualModel import train_model_multi_task
        if cfg.MODEL_PARAMS["lambda_seg"] is not None:
            lambda_seg = cfg.MODEL_PARAMS["lambda_seg"]
        else:
            lambda_seg = 1.0
        results = train_model_multi_task(model, train_loader, test_loader, num_epochs, device, learning_rate, lambda_seg=lambda_seg)

    elif model_name == "vgg":
        print("importing train functions")
        from models.vgg import train_vgg, precompute_model
        print("done importing train functions")
        train_features = precompute_model(model, train_loader, device=device)
        test_features = precompute_model(model, test_loader, device=device)
        train_loader = DataLoader(train_features, batch_size=cfg.batch_size, shuffle=True)
        test_loader = DataLoader(test_features, batch_size=cfg.batch_size, shuffle=False)
        resuls = train_vgg(model, train_loader, test_loader, device=device, num_epochs=num_epochs, learning_rate=learning_rate, criterion=criterion)
    
    elif model_name == "MFCN":
        from models.MFCN import train_model_multi_task
        results = train_model_multi_task(model, train_loader, test_loader, num_epochs, device, learning_rate, lambda_seg=1.0)


    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return results

        

def save_model(model, path):
    """Saves the model to the specified path.
    Only for pyTorch models.
    Args:
        model (torch.nn.Module): The model to save.
        path (str): The path to save the model to.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
