"""H2 verification experiment - common evaluation.

Evaluates, on the SAME held-out centralized N-BaIoT test split (the
standard, full, non-subsampled test set used for results/baseline_nbaiot.json
- i.e. load_centralized_data with the same seed/test-size):
  - the federated FedProx checkpoint (h2_fedprox_nbaiot.pt)
  - the federated FedAvg checkpoint (h2_fedavg_nbaiot.pt)
  - the centralized checkpoint (h2_centralized_nbaiot.pt)
  - each of the 9 isolated per-client checkpoints (h2_isolated_checkpoints/client_*.pt)

All four conditions therefore differ only in TRAINING methodology
(federated-FedProx / federated-FedAvg / centralized / isolated-per-client),
not in evaluation data - this is the apples-to-apples comparison H2 needs.
Threshold is calibrated the same way results/baseline_nbaiot.json's number
was (swept on this same test set - see test_03_evaluation.py's
test_threshold_was_calibrated_on_test_set_not_validation finding, which
applies identically here), plus metrics at fixed threshold=0.5 for reference.

Usage (from projet/):
    python eval_h2_models.py
"""
from __future__ import annotations
import os

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

from pytorchexample.task import Net, load_centralized_data

HERE = Path(__file__).parent
DATA_DIR = str(Path(os.environ.get("IOT_DATA", Path(__file__).resolve().parents[1] / "data")) / "Processed_NBaIoT_dataset")
FILE_PATTERN = "NBaIoT_*.csv"
BATCH_SIZE = 512
TEST_SIZE = 0.2
SEED = 42
OUTPUT = HERE.parent / "results" / "h2_comparison.json"


def metrics_at_threshold(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
    }


def evaluate_checkpoint(path: Path, input_dim: int, testloader, device):
    model = Net(input_dim)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    probs, labels = [], []
    with torch.no_grad():
        for xb, yb in testloader:
            p = torch.softmax(model(xb.to(device)), dim=1)[:, 1]
            probs.extend(p.cpu().numpy())
            labels.extend(yb.numpy())
    probs, labels = np.array(probs), np.array(labels)

    thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1 = f1_score(labels, (probs >= t).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    result = {
        "checkpoint": str(path.name),
        "num_test_examples": int(len(labels)),
        "calibrated_threshold": float(best_t),
        "at_calibrated_threshold": metrics_at_threshold(probs, labels, best_t),
        "at_threshold_0_5": metrics_at_threshold(probs, labels, 0.5),
        "auc": float(roc_auc_score(labels, probs)),
        "auc_pr": float(average_precision_score(labels, probs)),
    }
    return result


def main() -> None:
    device = torch.device("cpu")
    testloader, input_dim, has_labels = load_centralized_data(
        data_dir=DATA_DIR, file_pattern=FILE_PATTERN,
        batch_size=BATCH_SIZE, test_size=TEST_SIZE, seed=SEED,
    )
    assert has_labels

    results = {}
    for name, path in [
        ("federated_fedprox", HERE / "h2_fedprox_nbaiot.pt"),
        ("federated_fedavg", HERE / "h2_fedavg_nbaiot.pt"),
        ("centralized", HERE / "h2_centralized_nbaiot.pt"),
    ]:
        if path.exists():
            print(f"Evaluating {name} ({path.name})...")
            results[name] = evaluate_checkpoint(path, input_dim, testloader, device)
        else:
            results[name] = {"status": "NOT TESTABLE - checkpoint not found", "expected_path": str(path)}

    isolated_dir = HERE / "h2_isolated_checkpoints"
    isolated_results = []
    if isolated_dir.exists():
        for ckpt in sorted(isolated_dir.glob("client_*.pt")):
            print(f"Evaluating isolated {ckpt.name}...")
            r = evaluate_checkpoint(ckpt, input_dim, testloader, device)
            r["client"] = ckpt.stem
            isolated_results.append(r)
    results["isolated_per_client"] = isolated_results
    if isolated_results:
        results["isolated_mean_f1_macro"] = float(
            np.mean([r["at_calibrated_threshold"]["f1_macro"] for r in isolated_results])
        )
        results["isolated_std_f1_macro"] = float(
            np.std([r["at_calibrated_threshold"]["f1_macro"] for r in isolated_results])
        )

    OUTPUT.parent.mkdir(exist_ok=True, parents=True)
    OUTPUT.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUTPUT}")
    for name in ("federated_fedprox", "federated_fedavg", "centralized"):
        r = results[name]
        if "at_calibrated_threshold" in r:
            print(f"  {name}: F1-macro={r['at_calibrated_threshold']['f1_macro']:.4f} AUC={r['auc']:.4f}")
        else:
            print(f"  {name}: {r['status']}")
    if isolated_results:
        print(f"  isolated (mean+-std over {len(isolated_results)} clients): "
              f"F1-macro={results['isolated_mean_f1_macro']:.4f}+-{results['isolated_std_f1_macro']:.4f}")


if __name__ == "__main__":
    main()
