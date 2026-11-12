"""Réentraînement fédéré BoT-IoT avec une pondération renforcée de la classe normale.

Le modèle d'origine utilise DÉJÀ des class weights « balanced » (task.train :
w_c = N / (2 * n_c), soit normal ≈ 53 x attaque). Cette variante multiplie en plus le
poids de la classe normale par --normal-mult (5 par défaut), pour que les flux
normaux, très minoritaires (0,94 %), comptent encore davantage.

Tout le reste est identique à la config du mémoire et à train_botiot_nohour.py :
7 clients, FedProx mu=0.1, 100 rounds, 2 époques locales, lr 1e-3, batch 512,
weight_decay 1e-4, même pipeline task.load_partition_data, mêmes graines, modèle
final = dernier round (aucune sélection sur la validation ni sur le test).

La perte suivie à chaque round est mesurée sur la VALIDATION de l'étude
(common.Split) — à titre informatif uniquement.

Sauvegarde : final_model_botiot_cw<mult>.pt (+ .meta.json), checkpoint à chaque round.
N'écrase aucun modèle existant.

Usage :
    python train_botiot_classweight.py                     # 100 rounds, x5
    python train_botiot_classweight.py --rounds 2 --tag smoke
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

from common import QS, DATA_DIR, PATTERN, NUM_CLIENTS, SEED, TEST_SIZE, Split, scores
from pytorchexample.task import Net, _unpack_batch, load_partition_data

LOCAL_EPOCHS, LR, BATCH, PROX_MU, WEIGHT_DECAY = 2, 0.001, 512, 0.1, 1e-4
DEVICE = torch.device("cpu")


def train_weighted(model, trainloader, epochs, lr, proximal_mu, global_params, weight_decay, normal_mult):
    """Copie conforme de pytorchexample.task.train, sauf le poids de la classe 0
    multiplié par normal_mult (normal_mult=1 -> strictement identique à task.train)."""
    train_labels = trainloader.dataset.labels
    class_counts = torch.clamp(torch.bincount(train_labels.long(), minlength=2), min=1)
    class_weights = class_counts.sum().float() / (len(class_counts) * class_counts.float())
    class_weights[0] *= normal_mult
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        for batch in trainloader:
            features, targets = _unpack_batch(batch)
            noise = torch.randn_like(features) * 0.01
            features = (features + noise).clamp(0.0, 1.0)
            optimizer.zero_grad()
            loss = criterion(model(features), targets.long())
            if proximal_mu > 0.0 and global_params is not None:
                prox = 0.0
                for lw, gw in zip(model.parameters(), global_params):
                    prox += (lw - gw).norm(2) ** 2
                loss = loss + (proximal_mu / 2.0) * prox
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()


def fedavg(client_sds, sizes):
    """Moyenne pondérée par le nombre d'exemples (identique à train_botiot_nohour.py)."""
    w = np.array(sizes, dtype=np.float64)
    w /= w.sum()
    out = {}
    for k in client_sds[0]:
        if client_sds[0][k].dtype.is_floating_point:
            out[k] = sum(wi * sd[k].double() for wi, sd in zip(w, client_sds)).to(client_sds[0][k].dtype)
        else:
            out[k] = client_sds[0][k].clone()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--normal-mult", type=float, default=5.0)
    ap.add_argument("--tag", default=None, help="suffixe du nom de fichier (défaut : cw<mult>)")
    a = ap.parse_args()
    tag = a.tag or f"cw{a.normal_mult:g}"
    ckpt = QS / f"final_model_botiot_{tag}.pt"
    logf = QS / f"train_botiot_{tag}.log"
    assert ckpt.name not in ("final_model_botiot.pt", "final_model_botiot_nohour.pt")
    logf.write_text("", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(logf, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    t0 = time.time()
    log(f"BoT-IoT class weights renforcés : poids normal x{a.normal_mult:g} (en plus du balanced), "
        f"{a.rounds} rounds, {NUM_CLIENTS} clients, FedProx mu={PROX_MU} -> {ckpt.name}")
    loaders = []
    for c in range(NUM_CLIENTS):
        trl, _, d, _ = load_partition_data(
            partition_id=c, num_partitions=NUM_CLIENTS, data_dir=str(DATA_DIR), file_pattern=PATTERN,
            batch_size=BATCH, test_size=TEST_SIZE, seed=SEED, use_weighted_sampler=False,
            sampler_imbalance_threshold=2.0, partition_by_file=False, client_sample_frac=1.0)
        loaders.append(trl)
        input_dim = d
    s = Split("")
    xv, yv = s.x(s.val_idx), s.y[s.val_idx]
    del s
    log(f"input_dim={input_dim}, validation={len(yv)} (dont {int((yv == 0).sum())} normaux). "
        f"Chargement {time.time() - t0:.0f}s")

    from sklearn.metrics import roc_auc_score
    torch.manual_seed(SEED)
    global_sd = {k: v.clone() for k, v in Net(input_dim).state_dict().items()}
    history = []
    for r in range(1, a.rounds + 1):
        tr = time.time()
        sds, sizes = [], []
        for c, loader in enumerate(loaders):
            m = Net(input_dim)
            m.load_state_dict(global_sd)
            torch.manual_seed(SEED * 1000 + r * 10 + c)
            gp = [p.clone().detach() for p in m.parameters()]
            train_weighted(m, loader, LOCAL_EPOCHS, LR, PROX_MU, gp, WEIGHT_DECAY, a.normal_mult)
            sds.append({k: v.detach() for k, v in m.state_dict().items()})
            sizes.append(len(loader.dataset))
        global_sd = fedavg(sds, sizes)
        gm = Net(input_dim)
        gm.load_state_dict(global_sd)
        pv = scores(gm.eval(), xv)
        auc = float(roc_auc_score(yv, pv))
        fpr_05 = float((pv[yv == 0] >= 0.5).mean())
        rec_05 = float((pv[yv == 1] >= 0.5).mean())
        history.append(dict(round=r, val_auc=auc, val_fpr_at_0_5=fpr_05, val_recall_at_0_5=rec_05))
        torch.save(global_sd, ckpt)
        ckpt.with_suffix(".pt.meta.json").write_text(json.dumps(dict(
            completed_rounds=r, normal_mult=a.normal_mult, input_dim=input_dim,
            note="modèle = dernier round ; métriques = validation de l'étude fp_reduction (info)",
            history=history), indent=2), encoding="utf-8")
        log(f"round {r:>3}/{a.rounds}  val AUC={auc:.4f}  à 0.5 : FPR={fpr_05:.3f} rappel={rec_05:.4f}  "
            f"({time.time() - tr:.0f}s)")
    log(f"TERMINÉ en {(time.time() - t0) / 60:.1f} min -> {ckpt.name}")
    log("EXIT_OK")


if __name__ == "__main__":
    main()
