"""H2 verification experiment - isolated local models (no aggregation).

Trains 9 INDEPENDENT task.Net models, one per N-BaIoT client, each only on
its own (15%-subsampled, partition-by-file) local data, for the same total
local-epoch budget as one client in the federated runs (num-server-rounds *
local-epochs). No communication/aggregation between clients at all - this
is the "what if every device just learns on its own" reference point for H2.

Usage (from projet/):
    python train_h2_isolated.py
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
TOTAL_EPOCHS = 30 * 2  # matches run_config_h2_*.toml's num-server-rounds * local-epochs
BATCH_SIZE = 512
TEST_SIZE = 0.2
SEED = 42
LR = 0.001
WEIGHT_DECAY = 1e-4
CHECKPOINT_DIR = HERE / "h2_isolated_checkpoints"
LOG_FILE = HERE.parent / "results" / "fl_logs" / "h2_isolated_nbaiot.jsonl"


def main() -> None:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    device = torch.device("cpu")

    with open(LOG_FILE, "w", encoding="utf-8") as logf:
        for client_id in range(NUM_CLIENTS):
            client_t0 = time.time()
            trainloader, _, input_dim, _ = load_partition_data(
                partition_id=client_id, num_partitions=NUM_CLIENTS,
                data_dir=DATA_DIR, file_pattern=FILE_PATTERN,
                batch_size=BATCH_SIZE, test_size=TEST_SIZE, seed=SEED,
                partition_by_file=True, client_sample_frac=CLIENT_SAMPLE_FRAC,
            )
            model = Net(input_dim)
            print(f"Client {client_id}: {len(trainloader.dataset)} rows, training {TOTAL_EPOCHS} epochs...")
            for epoch in range(1, TOTAL_EPOCHS + 1):
                epoch_t0 = time.time()
                loss = train_fn(
                    model, trainloader, epochs=1, lr=LR, device=device,
                    proximal_mu=0.0, global_params=None, weight_decay=WEIGHT_DECAY,
                    use_class_weights=True,
                )
                logf.write(json.dumps({
                    "client_id": client_id, "epoch": epoch, "loss": loss,
                    "epoch_wall_time_sec": time.time() - epoch_t0,
                    "elapsed_since_run_start_sec": time.time() - t0,
                }) + "\n")
            logf.flush()
            ckpt_path = CHECKPOINT_DIR / f"client_{client_id}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  saved {ckpt_path} ({time.time() - client_t0:.1f}s)")

    print(f"\nAll {NUM_CLIENTS} isolated models trained in {time.time() - t0:.1f}s total")


if __name__ == "__main__":
    main()
