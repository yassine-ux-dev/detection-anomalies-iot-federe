"""N-BaIoT — autoencodeur fédéré auto-supervisé (étape 1, points 3-5).

Entraînement UNIQUEMENT sur le trafic NORMAL de chaque appareil (aucune étiquette). Score
d'anomalie = erreur de reconstruction (MSE). Scaler LOCAL par client. Seuil choisi sur l'erreur
du NORMAL de VALIDATION (percentile 99), figé avant le test. Les attaques ne servent qu'au test.

Modes (un entraînement à la fois) :
  fed       : autoencodeur fédéré (FedAvg) sur les 9 appareils — méthode principale ;
  local     : un autoencodeur par appareil (entraîné seul sur son normal) — contrôle ;
  central   : un autoencodeur sur le normal de tous les appareils réuni — référence ;
  untrained : poids aléatoires, non entraîné — contrôle (doit avoir une AUC bien plus basse).

Seuils comparés : GLOBAL (un seul, percentile 99 du normal val de tous) et LOCAL (par appareil).

    python nbaiot_ae.py --mode fed --rounds 20 --cap 50000
    python nbaiot_ae.py --mode fed --rounds 1 --cap 20000 --smoke   # mesure du temps
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

import nbaiot_data as D

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "nbaiot_fed"
DEVICE = torch.device("cpu")
SEED = 42
torch.set_num_threads(4)


class AE(nn.Module):
    """Autoencodeur compact (architecture inspirée de Meidan et al. 2018 : 0.75/0.5/0.33/0.25·L)."""

    def __init__(self, d: int = 115):
        super().__init__()
        h = [d, int(0.75 * d), int(0.5 * d), int(0.33 * d), int(0.25 * d)]
        enc, dec = [], []
        for a, b in zip(h[:-1], h[1:]):
            enc += [nn.Linear(a, b), nn.ReLU()]
        for a, b in zip(h[::-1][:-1], h[::-1][1:]):
            dec += [nn.Linear(a, b), nn.ReLU()]
        dec[-1] = nn.Sigmoid()  # sortie dans [0,1] (features MinMax)
        self.net = nn.Sequential(*enc, *dec)

    def forward(self, x):
        return self.net(x)


def recon_error(model: nn.Module, x: np.ndarray, batch: int = 8192) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i + batch]).to(DEVICE)
            out.append(((model(xb) - xb) ** 2).mean(1).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def train_local(model: nn.Module, x: np.ndarray, epochs: int, lr: float = 1e-3, batch: int = 256) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    dl = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch, shuffle=True)
    model.train()
    for _ in range(epochs):
        for (xb,) in dl:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(xb), xb).backward()
            opt.step()


def fedavg(sds, sizes):
    w = np.array(sizes, float); w /= w.sum()
    return {k: sum(wi * sd[k].double() for wi, sd in zip(w, sds)).to(sds[0][k].dtype) for k in sds[0]}


def wilson(k, n, z=1.96):
    import math
    if n == 0:
        return [float("nan")] * 2
    ph, d = k / n, 1 + z * z / n
    c, h = (ph + z * z / (2 * n)) / d, z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


PCTS = [95, 99, 99.9]


def _m(pred, y, fam):
    """Rappel/FPR + rappel par famille à partir de prédictions déjà calculées."""
    tp = int(pred[y == 1].sum()); fp = int(pred[y == 0].sum())
    n0 = int((y == 0).sum())
    return dict(rappel=tp / max((y == 1).sum(), 1), fpr=fp / max(n0, 1), fpr_ic95=wilson(fp, n0),
                rappel_mirai=float(pred[fam == "mirai"].mean()) if (fam == "mirai").any() else None,
                rappel_bashlite=float(pred[fam == "bashlite"].mean()) if (fam == "bashlite").any() else None)


def evaluate(per_dev: dict, thr_global: dict) -> dict:
    """per_dev[dev] = dict(err_norm_test, err_att, fam, thr_local={p:val}).
    thr_global = {p: val}. Calcule, aux percentiles 95/99/99.9, en seuil GLOBAL et LOCAL,
    l'AUC + rappel/FPR + rappel Mirai/BASHLITE, par appareil et en global."""
    res = {"par_appareil": {}, "global": {}}
    pool_err, pool_y, pool_fam = [], [], []
    # pour le cas LOCAL global : prédictions par appareil (chacun son seuil) puis mises bout à bout
    local_pred = {p: [] for p in PCTS}
    local_y, local_fam = [], []
    for dev, d in per_dev.items():
        en, ea = d["err_norm_test"], d["err_att"]
        y = np.r_[np.zeros(len(en)), np.ones(len(ea))]
        err = np.r_[en, ea]
        fam = np.r_[np.array(["normal"] * len(en)), d["fam"]]
        row = dict(n_normal_test=int(len(en)), n_attaque=int(len(ea)),
                   auc=float(roc_auc_score(y, err)) if len(ea) and len(set(y)) > 1 else float("nan"))
        for p in PCTS:
            row[f"p{p}"] = {"global": dict(seuil=float(thr_global[p]), **_m(err >= thr_global[p], y, fam)),
                            "local": dict(seuil=float(d["thr_local"][p]), **_m(err >= d["thr_local"][p], y, fam))}
            local_pred[p].append(err >= d["thr_local"][p])
        res["par_appareil"][dev] = row
        pool_err.append(err); pool_y.append(y); pool_fam.append(fam)
        local_y.append(y); local_fam.append(fam)
    err = np.concatenate(pool_err); y = np.concatenate(pool_y); fam = np.concatenate(pool_fam)
    ly = np.concatenate(local_y); lf = np.concatenate(local_fam)
    g = dict(n_normal_test=int((y == 0).sum()), n_attaque=int((y == 1).sum()), auc=float(roc_auc_score(y, err)))
    for p in PCTS:
        g[f"p{p}"] = {"global": dict(seuil=float(thr_global[p]), **_m(err >= thr_global[p], y, fam)),
                      "local": _m(np.concatenate(local_pred[p]), ly, lf)}
    res["global"] = g
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fed", "local", "central", "untrained"], required=True)
    ap.add_argument("--rounds", type=int, default=20, help="rounds fédérés (mode fed)")
    ap.add_argument("--epochs", type=int, default=2, help="époques locales par round (fed) ou totales (local/central)")
    ap.add_argument("--cap", type=int, default=50000, help="plafond de normal TRAIN par client")
    ap.add_argument("--smoke", action="store_true", help="essai court : mesure le temps, n'écrit pas les résultats finaux")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    devs = D.devices()
    print(f"[{a.mode}] {len(devs)} appareils, cap={a.cap}", flush=True)
    data, scalers, leak_total, dedup_total = {}, {}, 0, 0
    for dev in devs:
        dd = D.load_device(dev, cap=a.cap)
        sc = D.fit_local_scaler(dd["train"])                 # scaler LOCAL (normal train du client)
        leak_total += (D.check_no_leak(dd["train"], dd["test"]) + D.check_no_leak(dd["train"], dd["val"])
                       + D.check_no_leak(dd["val"], dd["test"]))
        dedup_total += dd["dedup"]["val_retirees"] + dd["dedup"]["test_retirees"]
        # échantillon d'attaques : autant que le normal de test de l'appareil
        xatt, fam, sub = D.sample_attacks(dd, n_target=len(dd["test"]))
        data[dev] = dict(
            xtr=np.clip(sc.transform(dd["train"]), 0, 1).astype(np.float32),
            xva=np.clip(sc.transform(dd["val"]), 0, 1).astype(np.float32),
            xte=np.clip(sc.transform(dd["test"]), 0, 1).astype(np.float32),
            xatt=np.clip(sc.transform(xatt), 0, 1).astype(np.float32) if len(xatt) else xatt,
            fam=fam, sub=sub)
        scalers[dev] = sc
    print(f"Chargement + scaling {time.time()-t0:.0f}s | doublons de contenu retirés de val/test : "
          f"{dedup_total} | lignes identiques restantes train/val/test : {leak_total}", flush=True)

    torch.manual_seed(SEED)
    models = {}
    if a.mode == "untrained":
        g = AE().to(DEVICE)
        models = {dev: g for dev in devs}
    elif a.mode == "central":
        xall = np.vstack([data[d]["xtr"] for d in devs])
        m = AE().to(DEVICE)
        for e in range(a.epochs):
            te = time.time(); train_local(m, xall, epochs=1); print(f"  central époque {e+1}/{a.epochs} ({time.time()-te:.0f}s)", flush=True)
        models = {dev: m for dev in devs}
    elif a.mode == "local":
        for dev in devs:
            m = AE().to(DEVICE); train_local(m, data[dev]["xtr"], epochs=a.epochs * a.rounds)
            models[dev] = m
            print(f"  local {dev} entraîné", flush=True)
    else:  # fed
        torch.manual_seed(SEED)
        gsd = {k: v.clone() for k, v in AE().state_dict().items()}
        val_hist = []
        n_params = sum(p.numel() for p in AE().parameters())
        bytes_round = n_params * 4 * 2 * len(devs)  # download+upload, float32, par round
        for r in range(1, a.rounds + 1):
            tr = time.time(); sds, sizes = [], []
            for dev in devs:
                m = AE().to(DEVICE); m.load_state_dict(gsd)
                train_local(m, data[dev]["xtr"], epochs=a.epochs)
                sds.append({k: v.detach() for k, v in m.state_dict().items()}); sizes.append(len(data[dev]["xtr"]))
            gsd = fedavg(sds, sizes)
            # suivi de convergence : loss de reconstruction sur le NORMAL de VALIDATION uniquement
            # (aucune attaque). C'est le SEUL signal utilisé pour juger du nombre de rounds.
            gm = AE().to(DEVICE); gm.load_state_dict(gsd)
            val_loss = float(np.mean([recon_error(gm, data[d]["xva"]).mean() for d in devs]))
            val_hist.append(val_loss)
            if r == 1 or r % 5 == 0 or r == a.rounds:
                print(f"  round {r}/{a.rounds}  loss_val(normal)={val_loss:.6f}  ({time.time()-tr:.0f}s)", flush=True)
        gm = AE().to(DEVICE); gm.load_state_dict(gsd)
        models = {dev: gm for dev in devs}

    # TROIS seuils figés sur le NORMAL de VALIDATION uniquement (percentiles 95, 99, 99.9).
    # Aucune attaque n'intervient dans le choix des seuils.
    per_dev = {}
    val_err_all = []
    for dev in devs:
        ev = recon_error(models[dev], data[dev]["xva"])
        per_dev[dev] = dict(err_norm_test=recon_error(models[dev], data[dev]["xte"]),
                            err_att=recon_error(models[dev], data[dev]["xatt"]) if len(data[dev]["xatt"]) else np.empty(0),
                            fam=data[dev]["fam"],
                            thr_local={p: float(np.percentile(ev, p)) for p in PCTS})
        val_err_all.append(ev)
    pooled_val = np.concatenate(val_err_all)
    thr_global = {p: float(np.percentile(pooled_val, p)) for p in PCTS}
    res = evaluate(per_dev, thr_global)
    res["meta"] = dict(mode=a.mode, rounds=a.rounds, epochs=a.epochs, cap=a.cap,
                       doublons_retires=dedup_total, lignes_identiques_restantes=leak_total,
                       seuils_global=thr_global, percentiles=PCTS,
                       temps_total_s=round(time.time() - t0, 1))
    if a.mode == "fed":
        res["meta"]["loss_val_par_round"] = val_hist
        res["ressources"] = dict(parametres=n_params, taille_ko=round(n_params * 4 / 1024, 1),
                                 octets_par_round=bytes_round, octets_total=bytes_round * a.rounds,
                                 note="float32, download+upload, 9 clients")
    g = res["global"]
    print(f"\n[{a.mode}] AUC global={g['auc']:.4f}")
    for p in PCTS:
        gl, lo = g[f"p{p}"]["global"], g[f"p{p}"]["local"]
        print(f"  p{p}: GLOBAL rappel={gl['rappel']:.3f} FPR={gl['fpr']:.3f} "
              f"(Mirai {gl['rappel_mirai']:.3f} / BASHLITE {gl['rappel_bashlite']:.3f}) | "
              f"LOCAL rappel={lo['rappel']:.3f} FPR={lo['fpr']:.3f} "
              f"(Mirai {lo['rappel_mirai']:.3f} / BASHLITE {lo['rappel_bashlite']:.3f})")
    if not a.smoke:
        (OUT / f"ae_{a.mode}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"-> results/nbaiot_fed/ae_{a.mode}.json")
    print(f"TEMPS TOTAL {a.mode}: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
