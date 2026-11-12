"""Step 6.0 — establish a documented TON-IoT baseline for final_model.pt.

Loads the checkpoint produced by the federated run, re-evaluates it on the
centralized held-out test split with the optimized-threshold classification
report, and persists F1 / AUC / accuracy / FPR / FNR to results/baseline_ton_iot.json
so later N-BaIoT / BoT-IoT runs have a citable reference row to compare against.
"""

import os
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report

from pytorchexample.task import Net, evaluate, load_centralized_data

HERE = Path(__file__).parent
CHECKPOINT = HERE / "final_model.pt"
RESULTS_DIR = HERE.parent / "results"
RESULTS_PATH = RESULTS_DIR / "baseline_ton_iot.json"

DATA_DIR = str(Path(os.environ.get("IOT_DATA", Path(__file__).resolve().parents[1] / "data")) / "Processed_IoT_dataset")
FILE_PATTERN = "IoT_*.csv"
BATCH_SIZE = 512
TEST_SIZE = 0.2
SEED = 42


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    testloader, input_dim, has_labels = load_centralized_data(
        data_dir=DATA_DIR,
        file_pattern=FILE_PATTERN,
        batch_size=BATCH_SIZE,
        test_size=TEST_SIZE,
        seed=SEED,
    )
    if not has_labels:
        raise RuntimeError("Centralized test split has no labels; cannot evaluate.")

    model = Net(input_dim)
    state_dict = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    loss, num_examples, metrics = evaluate(
        model, testloader, device, threshold=None, log_report=True
    )

    # Recompute predictions once more to get the confusion matrix / FPR / FNR,
    # using the same threshold-sweep logic evaluate() already ran internally.
    all_probs, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in testloader:
            features, targets = batch
            features = features.to(device)
            logits = model(features)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(targets.numpy())
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    thresholds = np.linspace(0.05, 0.95, 91)
    from sklearn.metrics import f1_score
    best_thresh, best_f1 = 0.5, 0.0
    for t in thresholds:
        preds_t = (all_probs >= t).astype(int)
        f1_t = f1_score(all_labels, preds_t, average="macro", zero_division=0)
        if f1_t > best_f1:
            best_f1, best_thresh = f1_t, t

    preds = (all_probs >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    report = classification_report(all_labels, preds, zero_division=0, output_dict=True)

    result = {
        "dataset": "TON-IoT (IoT device telemetry sub-dataset)",
        "checkpoint": str(CHECKPOINT.name),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "num_test_examples": num_examples,
        "decision_threshold": best_thresh,
        "loss": loss,
        "f1_macro": metrics["f1"],
        "auc": metrics["auc"],
        "accuracy": metrics["accuracy"],
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "classification_report": report,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2))

    print(f"\nBaseline saved to {RESULTS_PATH}")
    print(
        f"F1-macro={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}  "
        f"Acc={metrics['accuracy']:.4f}  FPR={fpr:.4f}  FNR={fnr:.4f}"
    )


if __name__ == "__main__":
    main()
