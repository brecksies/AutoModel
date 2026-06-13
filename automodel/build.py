"""Build a neural network sized for the data and task."""

import torch.nn as nn


def build_model(input_dim, output_dim, hidden=(64, 32)):
    """Construct a simple feed-forward MLP."""
    layers = []
    prev = input_dim
    for width in hidden:
        layers.append(nn.Linear(prev, width))
        layers.append(nn.ReLU())
        prev = width
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


def loss_for_task(task):
    """Return the loss function appropriate for the task."""
    if task == "classification":
        return nn.CrossEntropyLoss()
    return nn.MSELoss()
