"""Train a model on feature/target tensors."""

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from .build import loss_for_task


def train_model(model, features, target, task, epochs=50, batch_size=32,
                lr=1e-3, val_split=0.2, seed=42):
    """Train the model and return (model, history)."""
    generator = torch.Generator().manual_seed(seed)

    dataset = TensorDataset(features, target)
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    if n_val > 0:
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    else:
        train_ds, val_ds = dataset, None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    criterion = loss_for_task(task)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
        train_loss = running / n_train

        val_loss = _evaluate(model, val_ds, criterion, batch_size) if val_ds else None
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            line = f"Epoch {epoch:3d} | train_loss {train_loss:.4f}"
            if val_loss is not None:
                line += f" | val_loss {val_loss:.4f}"
            print(line)

    return model, history


def _evaluate(model, dataset, criterion, batch_size):
    """Compute the average loss over a dataset without updating weights."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    total = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            total += criterion(model(xb), yb).item() * len(xb)
    return total / len(dataset)
