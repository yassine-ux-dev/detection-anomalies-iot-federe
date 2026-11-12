"""H1 verification experiment - leave-one-attack-out, TON-IoT.

For each of 3 representative attack types (ddos, scanning, injection):
  1. Build the full engineered TON-IoT dataset (same pipeline as training:
     task.get_feature_columns / _encode_known_categoricals /
     _add_engineered_features / _prepare_features_and_labels).
  2. Remove EVERY row of that attack type before training (so the model
     never sees it, at all, in any form).
  3. Train the SAME architecture (task.Net), same hyperparameters as the
     federated runs (centralized here - leave-one-out doesn't need to be
     federated to test generalization to an unseen attack TYPE, which is a
     property of the model/training data, not of the aggregation scheme).
  4. Evaluate detection rate (recall) on ONLY that attack type's rows
     (which were never trained on in any form).
  5. Compare against the EXISTING supervised baseline (final_model.pt,
     trained WITH this attack type) evaluated on the exact same rows.

Usage (from projet/):
    python train_h1_leave_one_out.py
"""
from __future__ import annotations
import os

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler

import pytorchexample.task as task
from pytorchexample.task import Net, train as train_fn

HERE = Path(__file__).parent
DATA_DIR = Path(os.environ.get("IOT_DATA", Path(__file__).resolve().parents[1] / "data")) / "Processed_IoT_dataset"
FILE_PATTERN = "IoT_*.csv"
ATTACK_TYPES = ["ddos", "scanning", "injection"]
EPOCHS = 20
LR = 0.001
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 512
SEED = 42
TEST_SIZE = 0.2
CHECKPOINT_DIR = HERE / "h1_leave_one_out_checkpoints"
OUTPUT = HERE.parent / "results" / "h1_leave_one_out.json"
EXISTING_SUPERVISED_CHECKPOINT = HERE / "final_model.pt"


def build_full_df():
    files = task.list_csv_files(str(DATA_DIR), FILE_PATTERN)
    feature_columns = task.get_feature_columns(files)
    all_source_names = [task._source_name_from_path(p) for p in files]
    dfs = []
    for p in files:
        df = pd.read_csv(p, low_memory=False)
        df = task._encode_known_categoricals(df)
        df = task._add_engineered_features(df, task._source_name_from_path(p), all_source_names)
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    return df_all, feature_columns


def main() -> None:
    device = torch.device("cpu")
    print("Building full TON-IoT dataset (all 7 devices, engineered features)...")
    df_all, feature_columns = build_full_df()
    print(f"Total rows: {len(df_all)}, features: {len(feature_columns)}")

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    results = {}

    # Load the existing supervised baseline ONCE (trained WITH all attack types)
    existing_model = None
    if EXISTING_SUPERVISED_CHECKPOINT.exists():
        existing_model = Net(len(feature_columns))
        existing_model.load_state_dict(torch.load(EXISTING_SUPERVISED_CHECKPOINT, map_location=device))
        existing_model.to(device).eval()

    for attack in ATTACK_TYPES:
        t0 = time.time()
        print(f"\n=== Leave-one-out: {attack} ===")
        held_out_mask = df_all["type"].astype(str).str.lower() == attack
        n_held_out = int(held_out_mask.sum())
        if n_held_out == 0:
            results[attack] = {"status": f"NOT TESTABLE - no rows of type={attack!r} found"}
            continue

        training_pool = df_all[~held_out_mask].reset_index(drop=True)
        held_out_rows = df_all[held_out_mask].reset_index(drop=True)

        features_train_pool, labels_train_pool = task._prepare_features_and_labels(
            training_pool, feature_columns
        )
        features_held_out, labels_held_out = task._prepare_features_and_labels(
            held_out_rows, feature_columns
        )
        assert (labels_held_out == 1).all(), "held-out attack rows must all be label=1"

        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(features_train_pool.values).astype(np.float32)
        x_held_out = scaler.transform(features_held_out.values).astype(np.float32)

        from torch.utils.data import DataLoader, TensorDataset
        train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(labels_train_pool).long())
        trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        trainloader.dataset.labels = torch.from_numpy(labels_train_pool).long()

        model = Net(len(feature_columns))
        for epoch in range(EPOCHS):
            loss = train_fn(
                model, trainloader, epochs=1, lr=LR, device=device,
                proximal_mu=0.0, global_params=None, weight_decay=WEIGHT_DECAY,
                use_class_weights=True,
            )
        print(f"  trained {EPOCHS} epochs on {len(x_train)} rows (excl. {attack}), final loss={loss:.4f}")

        torch.save(model.state_dict(), CHECKPOINT_DIR / f"no_{attack}.pt")

        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(x_held_out).to(device))
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        detection_rate_loo = float((probs >= 0.5).mean())  # recall on this attack type, threshold=0.5

        existing_detection_rate = None
        if existing_model is not None:
            with torch.no_grad():
                logits_existing = existing_model(torch.from_numpy(x_held_out).to(device))
                probs_existing = torch.softmax(logits_existing, dim=1)[:, 1].cpu().numpy()
            existing_detection_rate = float((probs_existing >= 0.5).mean())

        results[attack] = {
            "n_held_out_rows": n_held_out,
            "n_train_rows_excluding_attack": int(len(x_train)),
            "leave_one_out_model_detection_rate_at_0_5": detection_rate_loo,
            "existing_supervised_baseline_detection_rate_at_0_5": existing_detection_rate,
            "existing_baseline_was_trained_on_this_attack_type": True,
            "wall_time_sec": time.time() - t0,
        }
        print(f"  leave-one-out detection rate: {detection_rate_loo:.4f}  "
              f"existing-supervised-baseline detection rate: {existing_detection_rate}")

    OUTPUT.parent.mkdir(exist_ok=True, parents=True)
    OUTPUT.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
