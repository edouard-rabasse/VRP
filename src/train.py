from torch.utils.data import DataLoader


def train(model_name,model, train_loader, test_loader, num_epochs, device, learning_rate,criterion=None,cfg=None):
    if model_name == 'VisualScoringModel':
        from models.VisualScoringModel import train_model
        train_model(
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
        from models.deit_tiny import precompute_deit_tiny_features, train_deit
        train_features = precompute_deit_tiny_features(model, train_loader, device=device)
        test_features = precompute_deit_tiny_features(model, test_loader, device=device)
        train_loader = DataLoader(train_features, batch_size=cfg.batch_size, shuffle=True)
        test_loader = DataLoader(test_features, batch_size=cfg.batch_size, shuffle=False)

        train_deit(model, train_loader, test_loader, device=device, num_epochs=num_epochs, learning_rate=learning_rate, criterion=criterion)

        
