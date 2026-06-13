# AutoModel
**Input any CSV, choose what to predict, and get a trained model back.**

![License](https://img.shields.io/github/license/brecksies/AutoModel)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

AutoModel ingests an arbitrary CSV file, lets you choose the target column(s) to predict and any columns to ignore, automatically infers the problem type (regression vs. classification), builds and trains a neural network, and exports the trained model for reuse.

## Project structure

```
AutoModel/
├── AutoModel.py            # CLI entry point: parses args and runs the pipeline
├── automodel/              # pipeline package
│   ├── __init__.py
│   ├── ingest.py           # CSV validation, column splitting, feature encoding
│   ├── infer_task.py       # regression vs. classification detection + target encoding
│   ├── build.py            # builds the MLP and selects the loss function
│   ├── train.py            # training loop with a validation split
│   └── export.py           # saves model weights + metadata
├── requirements.txt
├── README.md
└── LICENSE
```

`AutoModel.py` is a thin orchestrator. Each step of the workflow lives in its own module inside the `automodel` package and is called in sequence: **ingest → infer task → build → train → export**.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ with pandas and PyTorch.

## Usage

```bash
python AutoModel.py <input.csv> <predicted...> [options]
```

### Arguments

| Argument | Required | Description |
| --- | --- | --- |
| `input` | yes | Path to the input CSV file (validated to be a `.csv`). |
| `predicted` | yes | One or more column names to predict. |
| `-o`, `--output` | no | Path to save the trained model (default: `model.pt`). |
| `-i`, `--ignore` | no | One or more column names to drop before training. |
| `--epochs` | no | Number of training epochs (default: `50`). |
| `--batch-size` | no | Training batch size (default: `32`). |
| `--lr` | no | Learning rate (default: `0.001`). |
| `--hidden` | no | Hidden layer sizes (default: `[64, 32]`). |

### Examples

Train a classifier, ignoring an ID column:

```bash
python AutoModel.py data.csv label --ignore id --output model.pt
```

Train a regressor with custom architecture and training settings:

```bash
python AutoModel.py houses.csv price --ignore id address --hidden 128 64 --epochs 100 --lr 0.0005
```

## How it works

1. **Ingest** — Validates the file is a CSV and that the named columns exist, normalizes the header (strips BOM and surrounding whitespace), drops ignored columns, and one-hot encodes any non-numeric feature columns before converting them to a tensor.
2. **Infer task** — Decides between regression and classification. Non-numeric or few-valued integer targets are treated as classification; everything else is regression. The target is encoded accordingly (class indices or floats).
3. **Build** — Assembles a feed-forward MLP sized to the input/output dimensions and selects `CrossEntropyLoss` (classification) or `MSELoss` (regression).
4. **Train** — Runs an Adam training loop with a validation split and reports progress.
5. **Export** — Saves a single file containing the model weights and the metadata needed to rebuild and interpret it later.

## Output

The exported `.pt` file is a dictionary with two keys:

- `state_dict` — the trained model's weights.
- `metadata` — the task type, predicted/ignored columns, encoded feature layout, hidden sizes, output dimension, and (for classification) the class label mapping.

This is everything required to reconstruct the network and map predictions back to their original labels.

## Roadmap

- Feature normalization / standardization
- `predict.py` for running a saved model on new data
- Time-budgeted model search — try architectures/hyperparameters within a
  user-specified time limit and return the best by validation score
- Package for distribution (pip-installable with a CLI entry point)
- Multi-target classification support

## License

Distributed under the GPL-3.0 License.