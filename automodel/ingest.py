"""CSV ingestion: validation, feature/target splitting, and feature encoding."""

import argparse
import os

import pandas as pd
import torch


def csv_file(path):
    """argparse type that validates the path exists and is a .csv file."""
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"File does not exist: {path}")
    if os.path.splitext(path)[1].lower() != ".csv":
        raise argparse.ArgumentTypeError(f"File must be a .csv file: {path}")
    return path


def load_csv(path):
    """Read a CSV file into a DataFrame."""
    return pd.read_csv(path)


def validate_columns(df, predicted, ignore):
    """Ensure every named column actually exists in the DataFrame."""
    requested = set(predicted) | set(ignore)
    missing = requested - set(df.columns)
    if missing:
        raise ValueError(f"Column(s) not found in CSV: {', '.join(sorted(missing))}")


def split_columns(df, predicted, ignore):
    """Drop ignored columns, then return (features_df, target_df)."""
    if ignore:
        df = df.drop(columns=list(ignore))
    target = df[list(predicted)]
    features = df.drop(columns=list(predicted))
    return features, target


def features_to_tensor(features):
    """
    One-hot encode any non-numeric feature columns and convert to a float
    tensor. Returns (tensor, feature_names) so the column layout can be
    reproduced at inference time.
    """
    encoded = pd.get_dummies(features)
    if encoded.shape[1] == 0:
        raise ValueError("No feature columns remain after preparing the data.")
    array = encoded.to_numpy(dtype="float32")
    return torch.tensor(array), list(encoded.columns)
