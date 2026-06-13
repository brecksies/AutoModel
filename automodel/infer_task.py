"""Infer whether the problem is regression or classification, and encode targets."""

import pandas as pd
import torch


# A numeric target with at most this many unique integer values is treated as
# classification rather than regression.
MAX_CLASSES_FOR_CLASSIFICATION = 20


def infer_task(target):
    """
    Decide 'regression' or 'classification' from a target DataFrame.

    Rules:
      - More than one target column            -> regression (multi-output).
      - A single non-numeric column            -> classification.
      - A single numeric column whose values
        are all integers and few in number     -> classification.
      - Anything else                          -> regression.
    """
    if target.shape[1] > 1:
        return "regression"

    series = target.iloc[:, 0]
    if not pd.api.types.is_numeric_dtype(series):
        return "classification"

    nonnull = series.dropna()
    looks_integer = bool((nonnull % 1 == 0).all())
    if looks_integer and series.nunique() <= MAX_CLASSES_FOR_CLASSIFICATION:
        return "classification"
    return "regression"


def encode_target(target, task):
    """
    Convert the target DataFrame into a tensor plus metadata.

    Returns (tensor, meta). For classification the tensor holds class indices
    (dtype long) and meta records the class label mapping; for regression the
    tensor is float with one column per target.
    """
    if task == "classification":
        series = target.iloc[:, 0]
        classes = sorted(series.unique(), key=str)
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        indices = series.map(class_to_idx).to_numpy()
        tensor = torch.tensor(indices, dtype=torch.long)
        meta = {
            "output_dim": len(classes),
            "classes": [str(c) for c in classes],
            "class_to_idx": {str(k): v for k, v in class_to_idx.items()},
        }
        return tensor, meta

    tensor = torch.tensor(target.to_numpy(dtype="float32"))
    meta = {"output_dim": target.shape[1]}
    return tensor, meta
