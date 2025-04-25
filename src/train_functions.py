# trainers/__init__.py
import trainers
    
def train(model_name, model, train_loader, test_loader,
          num_epochs, device, learning_rate, *,
          cfg=None,
          **extra):
    """
    Trains the model using the specified parameters and returns the results (log).
    Args:
        model_name (str): The name of the model to train.
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): The DataLoader for the training set.
        test_loader (torch.utils.data.DataLoader): The DataLoader for the test set.
        num_epochs (int): The number of epochs to train for.
        device (str): The device to train on ('cpu' or 'cuda')."""

    trainer_fn = trainers.get_trainer(model_name)

    # Common kwargs can now be pushed in one call
    return trainer_fn(model,
                      train_loader,
                      test_loader,
                      num_epochs=num_epochs,
                      device=device,
                      learning_rate=learning_rate,
                      cfg=cfg,
                      **extra)


import torch


        

def save_model(model, path):
    """Saves the model to the specified path.
    Only for pyTorch models.
    Args:
        model (torch.nn.Module): The model to save.
        path (str): The path to save the model to.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
