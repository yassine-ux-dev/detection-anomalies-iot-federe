"""Étape 4 — explicabilité de l'autoencodeur fédéré (N-BaIoT, méthode FedProx + fine-tuning).

Deux explications complémentaires de l'anomalie (erreur de reconstruction élevée) :
  (1) erreur de reconstruction PAR FEATURE : (x_i - AE(x)_i)^2 — directe et rapide ; dit quelles
      features sont mal reconstruites (donc anormales) ;
  (2) SHAP (GradientExplainer) sur l'erreur de reconstruction scalaire moyenne — attribue l'erreur
      aux features d'entrée.
On vérifie que les deux classements concordent (Spearman), pour 3 anomalies et en global.

Tests de validité (comme pour le classifieur) :
  - additivité SHAP : base + somme des SHAP ≈ sortie du modèle ;
  - randomisation : un AE NON entraîné donne des explications sans structure (importance ~plate) ;
  - suppression (deletion) : mettre à 0 les k features les plus importantes réduit fortement
    l'erreur de reconstruction des anomalies (vs features aléatoires).
Artefacts : N-BaIoT n'a ni heure ni identifiant (115 statistiques de flux) — on vérifie que le
top des explications est bien composé de ces statistiques.

Réentraîne FedProx+FT (poids non sauvegardés) puis explique. ~15-20 min.
    python nbaiot_xai.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

import nbaiot_data as D
from nbaiot_ae import AE, recon_error
from nbaiot_hetero import OUT, EPOCHS, finetune, load_scaled, train_fed

MU, ROUNDS, SEED = 0.01, 20, 42
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
torch.set_num_threads(4)


def feature_names():
    f = D.RAW / "Danmini_Doorbell" / "benign_traffic.csv"
    import csv
    with open(f) as fh:
        return next(csv.reader(fh))


class ErrModule(nn.Module):
    """Sortie scalaire = erreur de reconstruction moyenne par flux (pour SHAP)."""
    def __init__(self, ae):
        super().__init__()
        self.ae = ae

    def forward(self, x):
        return ((self.ae(x) - x) ** 2).mean(1, keepdim=True)


def per_feature_error(ae, x):
    with torch.no_grad():
        return ((ae(torch.from_numpy(x)) - torch.from_numpy(x)) ** 2).numpy()


def main():
    t0 = time.time()
    names = feature_names()
    print(f"{len(names)} features (aucune temporelle/identifiant)", flush=True)
    data, _, devs = load_scaled(cap=50000, scaler_mode="local")
    clients = [data[d]["xtr"] for d in devs]
    vals = [data[d]["xva"] for d in devs]
    print("Réentraînement FedProx + fine-tuning (méthode d)…", flush=True)
    prox_sd, _ = train_fed(clients, vals, ROUNDS, EPOCHS, MU, "FedProx", seed=SEED)
    models, _ = finetune(prox_sd, data, devs)   # un AE par appareil (personnalisé)
    print(f"  entraînement {time.time()-t0:.0f}s", flush=True)

    rng = np.random.default_rng(SEED)
    res = {"features": names, "global": {}, "exemples": [], "validite": {}}

    # --- global : agrégation sur un échantillon d'attaques par appareil ---
    pfe_all, shap_all = [], []
    import shap
    for dev in devs:
        ae = models[dev]
        xa = data[dev]["xatt"]
        if len(xa) == 0:
            continue
        sel = rng.choice(len(xa), min(300, len(xa)), replace=False)
        xa = xa[sel]
        pfe_all.append(per_feature_error(ae, xa).mean(0))
        bg = data[dev]["xtr"][rng.choice(len(data[dev]["xtr"]), 100, replace=False)]
        ex = shap.GradientExplainer(ErrModule(ae).eval(), torch.from_numpy(bg))
        torch.manual_seed(SEED)
        sv = np.asarray(ex.shap_values(torch.from_numpy(xa[:150])))
        sv = sv.reshape(sv.shape[0], sv.shape[1])  # (n, 115)
        shap_all.append(np.abs(sv).mean(0))
    pfe_glob = np.mean(pfe_all, 0)
    shap_glob = np.mean(shap_all, 0)
    order_pfe = np.argsort(-pfe_glob)[:10]
    order_shap = np.argsort(-shap_glob)[:10]
    rho = float(spearmanr(pfe_glob, shap_glob).statistic)
    res["global"] = dict(top10_erreur_par_feature=[names[i] for i in order_pfe],
                         top10_shap=[names[i] for i in order_shap],
                         spearman_pfe_vs_shap=rho,
                         communes_top10=len(set(order_pfe) & set(order_shap)))
    print(f"\nGlobal — Spearman(erreur/feature vs SHAP) = {rho:.3f} ; "
          f"{res['global']['communes_top10']}/10 features communes au top-10", flush=True)
    print("  top-10 erreur/feature :", ", ".join(res["global"]["top10_erreur_par_feature"]))
    print("  top-10 SHAP           :", ", ".join(res["global"]["top10_shap"]))

    # --- 3 anomalies exemples (une Mirai, une BASHLITE, une d'un appareil difficile) ---
    picks = []
    for dev, fam_want in (("Danmini_Doorbell", "mirai"), ("Ennio_Doorbell", "bashlite"),
                          ("Samsung_SNH_1011_N_Webcam", "bashlite")):
        fam = data[dev]["fam"]
        idx = np.flatnonzero(fam == fam_want)
        if len(idx):
            picks.append((dev, fam_want, int(idx[0])))
    for dev, fam_want, i in picks:
        ae = models[dev]
        x = data[dev]["xatt"][i:i + 1]
        pfe = per_feature_error(ae, x)[0]
        bg = data[dev]["xtr"][rng.choice(len(data[dev]["xtr"]), 100, replace=False)]
        ex = shap.GradientExplainer(ErrModule(ae).eval(), torch.from_numpy(bg))
        torch.manual_seed(SEED)
        sv = np.asarray(ex.shap_values(torch.from_numpy(x))).reshape(-1)
        res["exemples"].append(dict(appareil=dev, famille=fam_want,
                                    erreur_totale=float(pfe.sum()),
                                    top5_erreur_par_feature=[names[j] for j in np.argsort(-pfe)[:5]],
                                    top5_shap=[names[j] for j in np.argsort(-np.abs(sv))[:5]]))

    # --- validité : additivité, randomisation, suppression ---
    dev = "Danmini_Doorbell"; ae = models[dev]
    xa = data[dev]["xatt"][:200]
    bg = data[dev]["xtr"][rng.choice(len(data[dev]["xtr"]), 200, replace=False)]
    em = ErrModule(ae).eval()
    base = float(em(torch.from_numpy(bg)).mean())
    ex = shap.GradientExplainer(em, torch.from_numpy(bg))
    torch.manual_seed(SEED)
    sv = np.asarray(ex.shap_values(torch.from_numpy(xa))).reshape(len(xa), -1)
    with torch.no_grad():
        fx = em(torch.from_numpy(xa)).numpy().reshape(-1)
    add_err = float(np.abs(base + sv.sum(1) - fx).mean())
    # randomisation : AE non entraîné
    torch.manual_seed(999); rae = AE().eval()
    rem = ErrModule(rae).eval()
    exr = shap.GradientExplainer(rem, torch.from_numpy(bg))
    torch.manual_seed(SEED)
    svr = np.abs(np.asarray(exr.shap_values(torch.from_numpy(xa))).reshape(len(xa), -1)).mean(0)
    # concentration (part du top-10 dans l'importance totale) : entraîné vs aléatoire
    conc_train = float(np.sort(np.abs(sv).mean(0))[::-1][:10].sum() / (np.abs(sv).mean(0).sum() + 1e-12))
    conc_rand = float(np.sort(svr)[::-1][:10].sum() / (svr.sum() + 1e-12))
    # suppression : mettre à 0 les k=10 top features (erreur/feature global) vs 10 aléatoires
    def err_after_zero(cols):
        xz = xa.copy(); xz[:, cols] = 0.0
        return float(per_feature_error(ae, xz).sum(1).mean())
    base_err = float(per_feature_error(ae, xa).sum(1).mean())
    drop_top = base_err - err_after_zero(order_pfe)
    drop_rand = base_err - err_after_zero(rng.choice(len(names), 10, replace=False))
    res["validite"] = dict(additivite_erreur_moyenne=add_err,
                           concentration_top10_entraine=conc_train, concentration_top10_aleatoire=conc_rand,
                           suppression_baisse_top=drop_top, suppression_baisse_aleatoire=drop_rand)
    print(f"\nValidité : additivité={add_err:.4f} | concentration top-10 entraîné={conc_train:.2f} "
          f"vs aléatoire={conc_rand:.2f} | suppression erreur -{drop_top:.3f} (top) vs -{drop_rand:.3f} (aléa)")

    # figure : top-10 erreur/feature global
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    yy = np.arange(10)
    ax.barh(yy, pfe_glob[order_pfe], color=BLUE)
    ax.set_yticks(yy, [names[i] for i in order_pfe], fontsize=7); ax.invert_yaxis()
    ax.set_xlabel("erreur de reconstruction moyenne par feature (attaques)")
    ax.set_title("AE fédéré (FedProx+FT) — features les plus anormales (N-BaIoT)", loc="left",
                 fontsize=10, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.savefig(OUT / "xai_feature_error.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    res["meta"] = dict(temps_min=round((time.time() - t0) / 60, 1))
    (OUT / "xai_ae.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> results/nbaiot_fed/xai_ae.json ; temps {res['meta']['temps_min']} min")


if __name__ == "__main__":
    main()
