#!/usr/bin/env python3
"""
AutoModel: input any CSV and the column(s) to predict, get a trained model back.

Pipeline: ingest -> infer task -> build -> train -> export.
"""

import argparse
import sys

from automodel import build, export, infer_task, ingest, train


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Ingest a CSV, train a model to predict the chosen "
                    "column(s), and export the trained model."
    )
    parser.add_argument(
        "input",
        type=ingest.csv_file,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "predicted",
        nargs="+",
        help="Name(s) of the column(s) to predict.",
    )
    parser.add_argument(
        "-o", "--output",
        default="model.pt",
        help="Path where the trained model is saved (default: model.pt).",
    )
    parser.add_argument(
        "-i", "--ignore",
        nargs="+",
        default=[],
        help="Name(s) of the column(s) to ignore.",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Training batch size (default: 32).",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3).",
    )
    parser.add_argument(
        "--hidden", type=int, nargs="+", default=[64, 32],
        help="Hidden layer sizes (default: 64 32).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # 1. Ingest --------------------------------------------------------------
    df = ingest.load_csv(args.input)
    try:
        ingest.validate_columns(df, args.predicted, args.ignore)
    except ValueError as exc:
        sys.exit(f"Error: {exc}")

    features_df, target_df = ingest.split_columns(df, args.predicted, args.ignore)
    feature_tensor, feature_names = ingest.features_to_tensor(features_df)

    # 2. Infer task and encode the target -----------------------------------
    task = infer_task.infer_task(target_df)
    target_tensor, target_meta = infer_task.encode_target(target_df, task)

    print(f"Inferred task: {task}")
    print(f"Samples: {feature_tensor.shape[0]} | "
          f"Features: {feature_tensor.shape[1]} | "
          f"Output dim: {target_meta['output_dim']}")

    # 3. Build ---------------------------------------------------------------
    model = build.build_model(
        input_dim=feature_tensor.shape[1],
        output_dim=target_meta["output_dim"],
        hidden=tuple(args.hidden),
    )

    # 4. Train ---------------------------------------------------------------
    model, _ = train.train_model(
        model, feature_tensor, target_tensor, task,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    )

    # 5. Export --------------------------------------------------------------
    metadata = {
        "task": task,
        "predicted": args.predicted,
        "ignored": args.ignore,
        "feature_names": feature_names,
        "hidden": list(args.hidden),
        **target_meta,
    }
    export.export_model(model, args.output, metadata)


if __name__ == "__main__":
    main()
