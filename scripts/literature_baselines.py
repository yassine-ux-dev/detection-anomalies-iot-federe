"""Step 7 — Literature-baseline comparison (Isolation Forest, One-Class SVM, LOF,
Random Forest, XGBoost, CNN, LSTM) against the federated model's results.

Uses the exact same feature-engineering pipeline as task.py (get_feature_columns /
_load_engineered_files / _prepare_features_and_labels) so features match what the
federated model was trained on for each dataset, then works on a stratified
subsample (default 100k rows) for computational feasibility -- One-Class SVM and
LOF in particular do not scale to the multi-million-row full datasets.

Unsupervised anomaly detectors (Isolation Forest, One-Class SVM, LOF) are fit on
normal-only training rows, per the standard semi-supervised novelty-detection
convention. One-Class SVM and LOF additionally cap that normal-only fit set at
--unsupervised-fit-cap (default 8000) since both remain expensive even at 80k rows.

All 7 models are evaluated with the same threshold-sweep-for-macro-F1 procedure
used by task.py's evaluate()/evaluate_checkpoint.py, so results are directly
comparable to results/baseline_<dataset>.json.

Usage (from projet/):
    python literature_baselines.py --dataset ton_iot \
        --data-dir $IOT_DATA/Processed_IoT_dataset --file-pattern "IoT_*.csv" \
        --output-dir results/literature_baselines/ton_iot

    python literature_baselines.py --dataset nbaiot \
        --data-dir $IOT_DATA/Processed_NBaIoT_dataset --file-pattern "NBaIoT_*.csv" \
        --output-dir results/literature_baselines/nbaiot

    python literature_baselines.py --dataset botiot \
        --data-dir $IOT_DATA/Processed_BoT_IoT_dataset --file-pattern "BoT_IoT_processed.csv" \
        --output-dir results/literature_baselines/botiot
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from pytorchexample.task import (
    list_csv_files,
    get_feature_columns,
    _load_engineered_files,
    _prepare_features_and_labels,
    _split_and_scale,
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Shared metric computation -- mirrors task.py's evaluate() threshold sweep so
# every baseline is scored the same way as the federated checkpoints.
# ---------------------------------------------------------------------------

def _metrics_from_scores(y_true: np.ndarray, scores: np.ndarray) -> Dict:
    """scores: higher = more likely attack, in any range (will be sweep-thresholded)."""
    lo, hi = scores.min(), scores.max()
    norm = (scores - lo) / (hi - lo) if hi > lo else np.zeros_like(scores)

    thresholds = np.linspace(0.05, 0.95, 91)
    best_thresh, best_f1 = 0.5, 0.0
    for t in thresholds:
        preds_t = (norm >= t).astype(int)
        f1_t = f1_score(y_true, preds_t, average="macro", zero_division=0)
        if f1_t > best_f1:
            best_f1, best_thresh = f1_t, t

    preds = (norm >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    accuracy = float((preds == y_true).mean())
    try:
        auc = float(roc_auc_score(y_true, norm))
    except ValueError:
        auc = 0.5
    report = classification_report(y_true, preds, zero_division=0, output_dict=True)

    return {
        "decision_threshold": float(best_thresh),
        "f1_macro": float(best_f1),
        "auc": auc,
        "accuracy": accuracy,
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "classification_report": report,
    }


# ---------------------------------------------------------------------------
# Small PyTorch CNN / LSTM baselines
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x.unsqueeze(1))
        return self.fc(x.squeeze(-1))


class SimpleLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x.unsqueeze(-1))
        return self.fc(h_n[-1])


def _train_torch_model(
    model: nn.Module, x_train: np.ndarray, y_train: np.ndarray, epochs: int = 15, batch_size: int = 512
) -> nn.Module:
    model.to(DEVICE)
    x_t = torch.from_numpy(x_train).float()
    y_t = torch.from_numpy(y_train).long()

    class_counts = torch.bincount(y_t, minlength=2).clamp(min=1)
    class_weights = class_counts.sum().float() / (2 * class_counts.float())
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    dataset = torch.utils.data.TensorDataset(x_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    return model


def _predict_proba_torch_model(model: nn.Module, x_test: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    probs = []
    x_t = torch.from_numpy(x_test).float()
    with torch.no_grad():
        for i in range(0, len(x_t), batch_size):
            batch = x_t[i:i + batch_size].to(DEVICE)
            probs.append(torch.softmax(model(batch), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


# ---------------------------------------------------------------------------
# Data loading + subsampling
# ---------------------------------------------------------------------------

def _load_subsample(
    data_dir: str, file_pattern: str, subsample_size: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    files = list_csv_files(data_dir, file_pattern)
    feature_columns = get_feature_columns(files)
    df_all = pd.concat(_load_engineered_files(files), ignore_index=True)
    features, labels = _prepare_features_and_labels(df_all, feature_columns)
    print(f"[OK] Loaded full dataset: {len(features)} rows, {len(feature_columns)} features")

    if len(features) > subsample_size:
        sample_idx, _ = train_test_split(
            np.arange(len(features)), train_size=subsample_size, stratify=labels, random_state=seed
        )
        features = features.iloc[sample_idx].reset_index(drop=True)
        labels = labels[sample_idx]
    print(f"[OK] Subsampled to {len(features)} rows (stratified, seed={seed})")

    x_train, x_test, y_train, y_test = _split_and_scale(features, labels, test_size=0.2, seed=seed)
    return x_train, x_test, y_train, y_test


def _cap_normal_rows(x_train: np.ndarray, y_train: np.ndarray, cap: int, seed: int) -> np.ndarray:
    normal = x_train[y_train == 0]
    if len(normal) > cap:
        rng = np.random.default_rng(seed)
        normal = normal[rng.choice(len(normal), size=cap, replace=False)]
    return normal


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Label used in output, e.g. 'TON-IoT'")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--file-pattern", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subsample-size", type=int, default=100_000)
    parser.add_argument("--unsupervised-fit-cap", type=int, default=8_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test = _load_subsample(
        str(ROOT / args.data_dir), args.file_pattern, args.subsample_size, args.seed
    )
    print(f"[OK] Train: {len(x_train)} rows  Test: {len(x_test)} rows  input_dim={x_train.shape[1]}")

    normal_train_full = x_train[y_train == 0]
    normal_train_capped = _cap_normal_rows(x_train, y_train, args.unsupervised_fit_cap, args.seed)
    print(f"[OK] Unsupervised fit sets: IsolationForest={len(normal_train_full)} rows, "
          f"OC-SVM/LOF={len(normal_train_capped)} rows (capped)")

    results: Dict[str, Dict] = {}

    def run_model(name: str, fit_and_score_fn) -> None:
        print(f"\n--- {name} ---")
        t0 = time.time()
        try:
            scores = fit_and_score_fn()
            metrics = _metrics_from_scores(y_test, scores)
            metrics["dataset"] = args.dataset
            metrics["model"] = name
            metrics["num_train_examples"] = int(len(x_train))
            metrics["num_test_examples"] = int(len(x_test))
            metrics["input_dim"] = int(x_train.shape[1])
            metrics["train_seconds"] = round(time.time() - t0, 1)
            metrics["evaluated_at"] = datetime.now(timezone.utc).isoformat()
            results[name] = metrics
            print(f"F1-macro={metrics['f1_macro']:.4f}  AUC={metrics['auc']:.4f}  "
                  f"Acc={metrics['accuracy']:.4f}  ({metrics['train_seconds']}s)")
        except Exception as exc:
            print(f"[ERROR] {name} failed: {exc}")

    # --- Unsupervised anomaly detectors (fit on normal-only rows) ---
    def isolation_forest():
        clf = IsolationForest(n_estimators=200, contamination="auto", random_state=args.seed, n_jobs=-1)
        clf.fit(normal_train_full)
        return -clf.decision_function(x_test)  # higher = more anomalous

    def one_class_svm():
        clf = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")
        clf.fit(normal_train_capped)
        return -clf.decision_function(x_test)

    def local_outlier_factor():
        clf = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1)
        clf.fit(normal_train_capped)
        return -clf.decision_function(x_test)

    # --- Supervised classifiers ---
    def random_forest():
        clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=args.seed, n_jobs=-1,
                                      class_weight="balanced")
        clf.fit(x_train, y_train)
        return clf.predict_proba(x_test)[:, 1]

    def xgboost_clf():
        if not HAS_XGBOOST:
            raise RuntimeError("xgboost not installed")
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        clf = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1, n_jobs=-1,
            random_state=args.seed, eval_metric="logloss", scale_pos_weight=scale_pos_weight,
        )
        clf.fit(x_train, y_train)
        return clf.predict_proba(x_test)[:, 1]

    def cnn():
        model = SimpleCNN(x_train.shape[1])
        _train_torch_model(model, x_train, y_train)
        return _predict_proba_torch_model(model, x_test)

    def lstm():
        model = SimpleLSTM(x_train.shape[1])
        _train_torch_model(model, x_train, y_train)
        return _predict_proba_torch_model(model, x_test)

    run_model("IsolationForest", isolation_forest)
    run_model("OneClassSVM", one_class_svm)
    run_model("LOF", local_outlier_factor)
    run_model("RandomForest", random_forest)
    run_model("XGBoost", xgboost_clf)
    run_model("CNN", cnn)
    run_model("LSTM", lstm)

    for name, metrics in results.items():
        path = output_dir / f"{name}.json"
        path.write_text(json.dumps(metrics, indent=2))

    summary_path = output_dir / "_summary.csv"
    with open(summary_path, "w") as f:
        f.write("model,f1_macro,auc,accuracy,fpr,fnr,train_seconds\n")
        for name, m in results.items():
            f.write(f"{name},{m['f1_macro']:.4f},{m['auc']:.4f},{m['accuracy']:.4f},"
                    f"{m['false_positive_rate']:.4f},{m['false_negative_rate']:.4f},{m['train_seconds']}\n")

    print(f"\n[SAVED] {len(results)} model results to {output_dir}")
    print(f"[SAVED] {summary_path}")


if __name__ == "__main__":
    main()
