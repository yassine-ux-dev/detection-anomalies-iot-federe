"""H2 verification experiment - centralized baseline.

Trains the SAME architecture (task.Net) on the UNION of all 9 N-BaIoT
clients' own (15%-subsampled, partition-by-file) training data - i.e. the
exact same data the federated FedProx/FedAvg runs train on, just pooled
into one process instead of kept per-client - for the same total number of
epochs (num-server-rounds * local-epochs, matching the federated runs'
total local-epoch budget per client). No aggregation, no clients: this is
the "what if we just centralized the data" reference point for H2.

Usage (from projet/):
    python train_h2_centralized.py
"""
from __future__ import annotations
import os

import json
import time
from pathlib import Path

import torch

from pytorchexample.task import Net, load_partition_data, train as train_fn

HERE = Path(__file__).parent
DATA_DIR = str(Path(os.environ.get("IOT_DATA", Path(__file__).resolve().parents[1] / "data")) / "Processed_NBaIoT_dataset")
FILE_PATTERN = "NBaIoT_*.csv"
NUM_CLIENTS = 9
CLIENT_SAMPLE_FRAC = 0.15
TOTAL_EPOCHS = 30 * 2  # num-server-rounds * local-epochs, matches run_config_h2_*.toml
BATCH_SIZE = 512
TEST_SIZE = 0.2
SEED = 42
LR = 0.001
WEIGHT_DECAY = 1e-4
CHECKPOINT = HERE / "h2_centralized_nbaiot.pt"
LOG_FILE = HERE.parent / "results" / "fl_logs" / "h2_centralized_nbaiot.jsonl"


def main() -> None:
    t0 = time.time()
    device = torch.device("cpu")

    print(f"Loading and pooling {NUM_CLIENTS} clients' local training data "
          f"(client-sample-frac={CLIENT_SAMPLE_FRAC})...")
    all_train_features, all_train_labels = [], []
    input_dim = None
    for client_id in range(NUM_CLIENTS):
        trainloader, _, dim, _ = load_partition_data(
            partition_id=client_id, num_partitions=NUM_CLIENTS,
            data_dir=DATA_DIR, file_pattern=FILE_PATTERN,
            batch_size=BATCH_SIZE, test_size=TEST_SIZE, seed=SEED,
            partition_by_file=True, client_sample_frac=CLIENT_SAMPLE_FRAC,
        )
        input_dim = dim
        for xb, yb in trainloader:
            all_train_features.append(xb)
            all_train_labels.append(yb)
        print(f"  client {client_id}: {len(trainloader.dataset)} rows pooled")

    import torch as T
    x_train = T.cat(all_train_features, dim=0)
    y_train = T.cat(all_train_labels, dim=0)
    from torch.utils.data import DataLoader, TensorDataset
    pooled_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    # train_fn expects .dataset.labels (NumpyDataset interface) for class weighting
    pooled_loader.dataset.labels = y_train

    print(f"Pooled training set: {len(x_train)} rows, input_dim={input_dim}")
    model = Net(input_dim)

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as logf:
        for epoch in range(1, TOTAL_EPOCHS + 1):
            epoch_t0 = time.time()
            loss = train_fn(
                model, pooled_loader, epochs=1, lr=LR, device=device,
                proximal_mu=0.0, global_params=None, weight_decay=WEIGHT_DECAY,
                use_class_weights=True,
            )
            epoch_time = time.time() - epoch_t0
            logf.write(json.dumps({
                "epoch": epoch, "loss": loss, "epoch_wall_time_sec": epoch_time,
                "elapsed_since_run_start_sec": time.time() - t0,
            }) + "\n")
            logf.flush()
            if epoch % 5 == 0 or epoch == 1:
                print(f"  epoch {epoch}/{TOTAL_EPOCHS}: loss={loss:.4f} ({epoch_time:.1f}s)")

    torch.save(model.state_dict(), CHECKPOINT)
    total_time = time.time() - t0
    print(f"\nSaved {CHECKPOINT} in {total_time:.1f}s total")


if __name__ == "__main__":
    main()
