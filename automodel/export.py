"""Export a trained model and its metadata for reuse."""

import torch


def export_model(model, path, metadata):
    """
    Save the model's weights together with the metadata needed to rebuild and
    interpret it later (task type, column layout, class mapping, etc.).
    """
    payload = {"state_dict": model.state_dict(), "metadata": metadata}
    torch.save(payload, path)
    print(f"Saved trained model to: {path}")
