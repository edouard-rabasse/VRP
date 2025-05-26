# benchmark.py: Train and evaluate a simple MLP on VRP graph-derived features

import hydra
import os
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from VRPgraph import VRPGraphDataset


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=None, num_classes: int = 2):
        super(SimpleMLP, self).__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def collate_fn(batch):
    features, labels = zip(*batch)
    xs = []
    for f in features:
        feats = f["features"]
        xs.append(
            [
                feats["number_of_routes"],
                feats["average_points_per_route"],
                feats["max_walked_segment"],
            ]
        )
    x_tensor = torch.tensor(xs, dtype=torch.float)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    return x_tensor, y_tensor


def train(cfg: DictConfig):
    # set random seeds
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    # Load dataset
    dataset = VRPGraphDataset(cfg.arcs_original, cfg.arcs_modified, cfg.coords)
    # Group indices by instance ID to keep original and modified together
    name_to_indices = {}
    for idx, inst_id in enumerate(dataset.names):
        name_to_indices.setdefault(inst_id, []).append(idx)
    group_ids = list(name_to_indices.keys())
    random.shuffle(group_ids)
    n_train_groups = int(cfg.train_split * len(group_ids))
    train_groups = set(group_ids[:n_train_groups])
    train_indices, val_indices = [], []
    for inst_id, idxs in name_to_indices.items():
        if inst_id in train_groups:
            train_indices.extend(idxs)
        else:
            val_indices.extend(idxs)
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMLP(input_dim=3, hidden_dims=cfg.hidden_dims).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x_batch.size(0)
        avg_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                logits = model(x_val)
                preds = logits.argmax(dim=1)
                correct += (preds == y_val).sum().item()
        val_acc = correct / len(val_loader.dataset)

        print(
            f"Epoch {epoch}/{cfg.epochs} - Train Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    print("Training complete.")
    return model


@hydra.main(version_base=None, config_path="config", config_name="benchmark")
def main(cfg: DictConfig):
    # restore working directory to project root
    os.chdir(get_original_cwd())
    model = train(cfg)
    print("weights for the model are :", model.state_dict())


if __name__ == "__main__":
    main()
