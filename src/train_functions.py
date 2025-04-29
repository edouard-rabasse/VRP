# trainers/__init__.py
import src.trainers as trainers
import wandb
from omegaconf import OmegaConf
    
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

    # Initialize Weights & Biases run
    # if cfg is not None:
    #     # convert OmegaConf to plain python dict
    #     wandb_config = OmegaConf.to_container(cfg, resolve=True)
    # else:
    #     wandb_config = {}
    # project = wandb_config.pop('project_name', None) or 'default_project'
    wandb.init(project="VRP", name=cfg.experiment_name)
    wandb.run.name = f"{model_name}_{wandb.run.id}"
    # Watch model for gradients and parameters
    wandb.watch(model)

    trainer_fn = trainers.get_trainer(model_name)

    # Execute training and capture per-epoch metrics
    metrics = trainer_fn(model,
                         train_loader,
                         test_loader,
                         num_epochs=num_epochs,
                         device=device,
                         learning_rate=learning_rate,
                         cfg=cfg,
                         **extra)
    # Log metrics to W&B
    for epoch_metrics in metrics:
        wandb.log(epoch_metrics)
    wandb.finish()
    return metrics


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
