"""Réentraînement fédéré BoT-IoT SANS la feature hour_of_day (dur conservée).

Réplique fidèle de la config du mémoire (pyproject.toml) : 7 clients, FedProx mu=0.1,
100 rounds, 2 époques locales, lr 1e-3, batch 512, weight_decay 1e-4, class weights.
Utilise le VRAI pipeline (pytorchexample.task) via la clé exclude-features="hour_of_day".

Script autonome plutôt que `flwr run` : il charge le pipeline (pytorchexample.task) en
mémoire au démarrage et réalise une agrégation FedAvg pondérée séquentielle, identique à
celle de Flower (vérifiée numériquement), plus simple et robuste en mémoire pour 7 clients.

Sauvegarde : final_model_botiot_nohour.pt (+ .meta.json), checkpoint à CHAQUE round.
N'écrase JAMAIS final_model_botiot.pt.

Usage :
    python train_botiot_nohour.py            # 100 rounds
    python train_botiot_nohour.py --rounds 5 # test rapide
"""

from __future__ import annotations
import os

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Pipeline réutilisé tel quel (même code que `flwr run`).
from pytorchexample.task import Net, load_partition_data, load_centralized_data, train as task_train, evaluate as task_eval

HERE = Path(__file__).resolve().parent
DATA_DIR = str(Path(os.environ.get("IOT_DATA", Path(__file__).resolve().parents[1] / "data")) / "Processed_BoT_IoT_dataset")
PATTERN = "BoT_IoT_processed.csv"
CKPT = HERE / "final_model_botiot_nohour.pt"
LOG = HERE / "train_botiot_nohour.log"

# Hyperparamètres — identiques à pyproject.toml (config du mémoire)
NUM_CLIENTS = 7
LOCAL_EPOCHS = 2
LR = 0.001
BATCH = 512
TEST_SIZE = 0.2
SEED = 42
PROX_MU = 0.1
WEIGHT_DECAY = 1e-4
USE_WEIGHTED_SAMPLER = False
SAMPLER_IMB = 2.0
USE_CLASS_WEIGHTS = True
EXCLUDE = "hour_of_day"
DEVICE = torch.device("cpu")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def clone_sd(sd: dict) -> dict:
    return {k: v.detach().clone() for k, v in sd.items()}


def fedavg(client_sds: list, sizes: list) -> dict:
    """Moyenne pondérée par le nombre d'exemples (identique à Flower aggregate_arrayrecords)."""
    w = np.array(sizes, dtype=np.float64)
    w /= w.sum()
    keys = client_sds[0].keys()
    out = {}
    for k in keys:
        if client_sds[0][k].dtype.is_floating_point:
            out[k] = sum(wi * sd[k].double() for wi, sd in zip(w, client_sds)).to(client_sds[0][k].dtype)
        else:  # num_batches_tracked : on prend la valeur du premier client
            out[k] = client_sds[0][k].clone()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=100)
    args = ap.parse_args()

    LOG.write_text("", encoding="utf-8")
    t_start = time.time()
    log(f"Réentraînement BoT-IoT SANS hour_of_day — {args.rounds} rounds, {NUM_CLIENTS} clients, FedProx mu={PROX_MU}")
    log(f"Checkpoint: {CKPT.name} (n'écrase pas final_model_botiot.pt)")

    # --- chargement unique des 7 clients (exclude hour_of_day) ---
    log("Chargement des partitions clients (une seule fois)…")
    train_loaders, input_dim = [], None
    for c in range(NUM_CLIENTS):
        trl, _, d, _ = load_partition_data(
            partition_id=c, num_partitions=NUM_CLIENTS, data_dir=DATA_DIR, file_pattern=PATTERN,
            batch_size=BATCH, test_size=TEST_SIZE, seed=SEED,
            use_weighted_sampler=USE_WEIGHTED_SAMPLER, sampler_imbalance_threshold=SAMPLER_IMB,
            partition_by_file=False, client_sample_frac=1.0, exclude_features=EXCLUDE,
        )
        train_loaders.append(trl)
        input_dim = d
        log(f"  client {c}: {len(trl.dataset)} exemples train")
    log(f"input_dim = {input_dim} (attendu 50 = 51 - hour_of_day)")

    testloader, test_dim, has_labels = load_centralized_data(
        data_dir=DATA_DIR, file_pattern=PATTERN, batch_size=BATCH,
        test_size=TEST_SIZE, seed=SEED, exclude_features=EXCLUDE,
    )
    assert test_dim == input_dim, f"dim incohérente: train {input_dim} vs test {test_dim}"
    log(f"Test centralisé: {len(testloader.dataset)} exemples. Chargement total: {time.time()-t_start:.0f}s")

    # --- boucle fédérée ---
    torch.manual_seed(SEED)
    global_sd = clone_sd(Net(input_dim).state_dict())
    best_auc, history = 0.0, []
    for r in range(1, args.rounds + 1):
        t0 = time.time()
        client_sds, sizes = [], []
        for c, loader in enumerate(train_loaders):
            m = Net(input_dim)
            m.load_state_dict(global_sd)
            torch.manual_seed(SEED * 1000 + r * 10 + c)
            gp = [p.clone().detach() for p in m.parameters()] if PROX_MU > 0 else None
            task_train(m, loader, epochs=LOCAL_EPOCHS, lr=LR, device=DEVICE,
                       proximal_mu=PROX_MU, global_params=gp, weight_decay=WEIGHT_DECAY,
                       use_class_weights=USE_CLASS_WEIGHTS)
            client_sds.append({k: v.detach() for k, v in m.state_dict().items()})
            sizes.append(len(loader.dataset))
        global_sd = fedavg(client_sds, sizes)

        gm = Net(input_dim)
        gm.load_state_dict(global_sd)
        loss, _, met = task_eval(gm, testloader, DEVICE, threshold=None)
        history.append(dict(round=r, loss=float(loss), auc=float(met["auc"]),
                            f1=float(met["f1"]), threshold=float(met["threshold"])))
        # checkpoint à chaque round (reprise possible)
        torch.save(global_sd, CKPT)
        CKPT.with_suffix(".pt.meta.json").write_text(json.dumps(
            dict(completed_rounds=r, best_auc=max(best_auc, met["auc"]), exclude_features=EXCLUDE,
                 input_dim=input_dim, history=history), indent=2))
        best_auc = max(best_auc, float(met["auc"]))
        log(f"round {r:>3}/{args.rounds}  loss={loss:.4f}  AUC={met['auc']:.4f}  "
            f"F1-macro={met['f1']:.4f}  seuil={met['threshold']:.2f}  ({time.time()-t0:.0f}s)")

    log(f"TERMINÉ en {(time.time()-t_start)/60:.1f} min. Meilleure AUC={best_auc:.4f}. Modèle: {CKPT.name}")
    log("EXIT_OK")


if __name__ == "__main__":
    main()
