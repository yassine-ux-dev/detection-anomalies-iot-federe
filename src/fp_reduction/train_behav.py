"""Réentraînement fédéré BoT-IoT SANS hour_of_day, + feature octets_par_paquet (priorité 1).

Deux variantes (même config que le mémoire : 7 clients, FedProx µ=0,1, 100 rounds, 2 époques,
lr 1e-3, batch 512, weight_decay 1e-4, class weights ; scaling par client comme task.py) :
  --variant behav  -> final_model_botiot_behav.pt  : exclut hour_of_day, ajoute octets_par_paquet.
  --variant deploy -> final_model_botiot_deploy.pt : exclut hour_of_day ET les statistiques Argus
     agrégées (mean, stddev, min, max, sum), les drapeaux Argus (flgs_*) et src_bot_processed
     (constant) ; ajoute octets_par_paquet. Ne garde que des features qu'un extracteur standard
     reproduit (comptes, débits, protocole, état de connexion, octets/paquet).

Script autonome (helpers chargés en mémoire au démarrage). N'écrase aucun modèle existant.
Checkpoint à chaque round.

    python train_behav.py --variant behav --rounds 100
    python train_behav.py --variant deploy --rounds 100
    python train_behav.py --variant behav --rounds 1   # smoke test
"""

from __future__ import annotations
import os

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pytorchexample.task import (NumpyDataset, Net, _load_engineered_files,
                                 _prepare_features_and_labels, _split_and_scale,
                                 get_feature_columns, list_csv_files,
                                 train as task_train, evaluate as task_eval)

HERE = Path(__file__).resolve().parent
QS = HERE.parent
DATA_DIR = str(Path(os.environ.get("IOT_DATA", Path(__file__).resolve().parents[2] / "data")) / "Processed_BoT_IoT_dataset")
PATTERN = "BoT_IoT_processed.csv"
NUM_CLIENTS, LOCAL_EPOCHS, LR, BATCH, TEST_SIZE, SEED = 7, 2, 0.001, 512, 0.2, 42
PROX_MU, WEIGHT_DECAY = 0.1, 1e-4
DEVICE = torch.device("cpu")
DERIVED = "octets_par_paquet"

ARGUS_AGG = ["mean", "stddev", "min", "max", "sum"]
FLGS = ["flgs_e", "flgs_eU", "flgs_e_D", "flgs_e_F", "flgs_e_amp", "flgs_e_d", "flgs_e_dD",
        "flgs_e_dS", "flgs_e_g", "flgs_e_r", "flgs_e_s", "flgs_e_star", "flgs_e_t"]


def feature_spec(variant: str, base_cols: list) -> list:
    if variant == "behav":
        excl = {"hour_of_day"}
    else:  # deploy : seulement des features reproductibles par un extracteur standard
        excl = {"hour_of_day", "src_bot_processed", *ARGUS_AGG, *FLGS}
    return [c for c in base_cols if c not in excl] + [DERIVED]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df[DERIVED] = df["bytes"] / df["pkts"].clip(lower=1)
    return df


def clone_sd(sd):
    return {k: v.detach().clone() for k, v in sd.items()}


def fedavg(client_sds, sizes):
    w = np.array(sizes, dtype=np.float64); w /= w.sum()
    out = {}
    for k in client_sds[0]:
        if client_sds[0][k].dtype.is_floating_point:
            out[k] = sum(wi * sd[k].double() for wi, sd in zip(w, client_sds)).to(client_sds[0][k].dtype)
        else:
            out[k] = client_sds[0][k].clone()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["behav", "deploy"], required=True)
    ap.add_argument("--rounds", type=int, default=100)
    a = ap.parse_args()
    ckpt = QS / f"final_model_botiot_{a.variant}.pt"
    logf = QS / f"train_botiot_{a.variant}.log"
    assert ckpt.name not in ("final_model_botiot.pt", "final_model_botiot_nohour.pt")
    logf.write_text("", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(logf, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    t0 = time.time()
    files = list_csv_files(DATA_DIR, PATTERN)
    base_cols = get_feature_columns(files)
    cols = feature_spec(a.variant, base_cols)
    log(f"Variante {a.variant} : {len(cols)} features ({DERIVED} ajouté). {a.rounds} rounds, "
        f"{NUM_CLIENTS} clients, FedProx µ={PROX_MU} -> {ckpt.name}")
    log("Chargement + feature engineering…")
    df_all = add_derived(pd.concat(_load_engineered_files(files), ignore_index=True))
    assert not df_all[DERIVED].isna().any() and np.isfinite(df_all[DERIVED]).all()
    df_all = df_all.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    n = len(df_all); size = n // NUM_CLIENTS
    train_loaders = []
    for c in range(NUM_CLIENTS):
        sl = df_all.iloc[c * size:(c + 1) * size if c < NUM_CLIENTS - 1 else n]
        feats, y = _prepare_features_and_labels(sl, cols)
        x_tr, _, y_tr, _ = _split_and_scale(feats, y, test_size=TEST_SIZE, seed=SEED)  # scaler par client
        train_loaders.append(DataLoader(NumpyDataset(x_tr, y_tr), batch_size=BATCH, shuffle=True))
        log(f"  client {c}: {len(x_tr)} exemples train")
    input_dim = len(cols)

    feats, y = _prepare_features_and_labels(df_all, cols)
    _, x_te, _, y_te = _split_and_scale(feats, y, test_size=TEST_SIZE, seed=SEED)
    testloader = DataLoader(NumpyDataset(x_te, y_te), batch_size=BATCH, shuffle=False)
    log(f"input_dim={input_dim} | test centralisé={len(x_te)} | chargement {time.time()-t0:.0f}s")

    torch.manual_seed(SEED)
    global_sd = clone_sd(Net(input_dim).state_dict())
    best_auc, history = 0.0, []
    for r in range(1, a.rounds + 1):
        tr = time.time()
        sds, sizes = [], []
        for c, loader in enumerate(train_loaders):
            m = Net(input_dim); m.load_state_dict(global_sd)
            torch.manual_seed(SEED * 1000 + r * 10 + c)
            gp = [p.clone().detach() for p in m.parameters()]
            task_train(m, loader, epochs=LOCAL_EPOCHS, lr=LR, device=DEVICE, proximal_mu=PROX_MU,
                       global_params=gp, weight_decay=WEIGHT_DECAY, use_class_weights=True)
            sds.append({k: v.detach() for k, v in m.state_dict().items()}); sizes.append(len(loader.dataset))
        global_sd = fedavg(sds, sizes)
        gm = Net(input_dim); gm.load_state_dict(global_sd)
        loss, _, met = task_eval(gm, testloader, DEVICE, threshold=None)
        history.append(dict(round=r, loss=float(loss), auc=float(met["auc"]), f1=float(met["f1"]),
                            threshold=float(met["threshold"])))
        torch.save(global_sd, ckpt)
        ckpt.with_suffix(".pt.meta.json").write_text(json.dumps(dict(
            completed_rounds=r, best_auc=max(best_auc, met["auc"]), variant=a.variant,
            feature_columns=cols, derived=DERIVED, input_dim=input_dim, history=history),
            indent=2, ensure_ascii=False), encoding="utf-8")
        best_auc = max(best_auc, float(met["auc"]))
        log(f"round {r:>3}/{a.rounds}  loss={loss:.4f}  AUC={met['auc']:.4f}  F1-macro={met['f1']:.4f}  "
            f"seuil={met['threshold']:.2f}  ({time.time()-tr:.0f}s)")
    log(f"TERMINÉ en {(time.time()-t0)/60:.1f} min. Meilleure AUC={best_auc:.4f}. Modèle: {ckpt.name}")
    log("EXIT_OK")


if __name__ == "__main__":
    main()
