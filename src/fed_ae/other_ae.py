"""Étape 2 — AE fédéré auto-supervisé sur TON-IoT et BoT-IoT (méthode principale, évaluation 3 jeux).

RÉSULTAT (2026-10-14) : l'AE auto-supervisé N'EST PAS efficace sur ces deux jeux (contrairement
à N-BaIoT) — AUC ~0,5. TON-IoT : features = relevés de capteurs discrets (door_state, motion...),
qui ne reflètent PAS les attaques réseau -> l'erreur de reconstruction ne sépare pas (AUC 0,40-0,62 ;
3/7 appareils dégénérés, sans normal unique). BoT-IoT : seulement 9 543 flux normaux (clients
Dirichlet aussi petits que 41 lignes) -> trop peu pour apprendre le normal (AUC 0,45-0,67). Résultat
négatif honnête : la méthode excelle là où le normal est abondant et les features comportementales
(N-BaIoT), pas ici. Plafond normal-train = 50 000/client (CAP_TRAIN) pour la vitesse.


Règles : entraînement sur le NORMAL uniquement ; pas de feature temporelle ni d'identifiant
(exclusion de hour_of_day et des src_* d'appareil) ; découpage CHRONOLOGIQUE 60/20/20 par client
avec dédup ; scaler LOCAL par client ; attaques au test seulement ; seuil p99 GLOBAL sur le
normal-val poolé. Modèles : FedAvg, FedProx, FedProx + fine-tuning local, AE centralisé.

Partition non-IID :
  - TON-IoT : un client = un appareil (7 fichiers IoT_*.csv).
  - BoT-IoT : normal (9 543 flux) réparti en 7 clients par Dirichlet(α=0.5) (quantité non-IID).

Robustesse : la télémétrie TON-IoT a des capteurs discrets très répétitifs ; après dédup, certains
appareils n'ont AUCUN normal unique en val/test. L'évaluation est donc POOLÉE (seuil et FPR sur
les appareils ayant du normal unique ; rappel d'attaque sur tous). Les appareils dégénérés sont
rapportés.

    python other_ae.py --dataset ton
    python other_ae.py --dataset botiot
"""

from __future__ import annotations
import os

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from nbaiot_ae import AE, recon_error
from nbaiot_hetero import EPOCHS, finetune, train_fed, train_client
from nbaiot_data import row_hash
from pytorchexample.task import (_apply_exclude, _load_engineered_files, _prepare_features_and_labels,
                                 get_feature_columns, list_csv_files)

QS = Path(__file__).resolve().parents[2]
OUT = QS / "results" / "nbaiot_fed"
MU, ROUNDS, SEED = 0.01, 20, 42
torch.set_num_threads(4)

CFG = {
    "ton": dict(dir="Processed_IoT_dataset", pattern="IoT_*.csv", exclude="hour_of_day",
                drop_prefix="src_", type_col="type", label="TON-IoT", partition="device",
                types=["backdoor", "ddos", "injection", "password", "ransomware", "scanning", "xss"]),
    "botiot": dict(dir="Processed_BoT_IoT_dataset", pattern="BoT_IoT_processed.csv", exclude="hour_of_day",
                   drop_prefix=None, type_col="subcategory", label="BoT-IoT", partition="dirichlet",
                   types=["UDP", "TCP", "HTTP", "Service_Scan", "OS_Fingerprint", "Keylogging",
                          "Data_Exfiltration"]),
}


CAP_TRAIN = 50000   # plafond de normal-train par client (comme N-BaIoT), pour la vitesse


def chrono_split(x):
    n = len(x); a, b = int(0.6 * n), int(0.8 * n)
    tr, va, te = x[:a], x[a:b], x[b:]
    if len(tr) > CAP_TRAIN:   # sous-échantillon régulier (garde l'ordre chronologique)
        tr = tr[np.linspace(0, len(tr) - 1, CAP_TRAIN).astype(int)]
    htr = set(row_hash(tr).tolist())
    va = va[~np.isin(row_hash(va), list(htr))]
    te = te[~np.isin(row_hash(te), list(htr | set(row_hash(va).tolist())))]
    return tr, va, te


def _mk(sd):
    m = AE(sd["net.0.weight"].shape[1]); m.load_state_dict(sd); return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", choices=list(CFG), required=True)
    a = ap.parse_args(); t0 = time.time()
    c = CFG[a.dataset]
    files = list_csv_files(str(Path(os.environ.get("IOT_DATA", Path(__file__).resolve().parents[2] / "data")) / c["dir"]), c["pattern"])
    cols = _apply_exclude(get_feature_columns(files), c["exclude"])
    if c["drop_prefix"]:
        cols = [k for k in cols if not k.startswith(c["drop_prefix"])]
    d = len(cols)
    dfs = _load_engineered_files(files)
    tr_raw, va_raw, te_raw, att_raw, att_typ = [], [], [], [], []
    for df in dfs:
        feats, y = _prepare_features_and_labels(df, cols)
        X = feats.to_numpy(np.float32)
        typ = df[c["type_col"]].astype(str).str.strip().to_numpy() if c["type_col"] in df.columns else np.array([""] * len(df))
        t, v, e = chrono_split(X[y == 0])
        tr_raw.append(t); va_raw.append(v); te_raw.append(e)
        att_raw.append(X[y == 1]); att_typ.append(typ[y == 1])

    if c["partition"] == "dirichlet":  # BoT-IoT : 7 clients par Dirichlet sur le normal-train réuni
        pooled = np.vstack(tr_raw); rng = np.random.default_rng(SEED); rng.shuffle(pooled)
        sizes = np.maximum((rng.dirichlet([0.5] * 7) * len(pooled)).astype(int), 1); sizes[-1] = len(pooled) - sizes[:-1].sum()
        tr_raw = np.split(pooled, np.cumsum(sizes)[:-1])
        # val/test/attaques restent globaux (un "client" logique = tout)
        va_raw = [np.vstack([v for v in va_raw if len(v)])]; te_raw = [np.vstack([e for e in te_raw if len(e)])]
        att_raw = [np.vstack(att_raw)]; att_typ = [np.concatenate(att_typ)]
        central_scaler_src = 0
    K = len(tr_raw)
    sc = [MinMaxScaler().fit(x) for x in tr_raw]
    clip = lambda z: np.clip(z, 0, 1).astype(np.float32)
    Xtr = [clip(sc[i].transform(tr_raw[i])) for i in range(K)]
    # pour l'éval poolée, chaque source normal/attaque est scalée par SON scaler (device) ou le client 0 (dirichlet)
    def scaled(src_list, i_scaler):
        return [clip(sc[i_scaler(i)].transform(x)) if len(x) else np.empty((0, d), np.float32) for i, x in enumerate(src_list)]
    i_sc = (lambda i: 0) if c["partition"] == "dirichlet" else (lambda i: i)
    Xva = scaled(va_raw, i_sc); Xte = scaled(te_raw, i_sc)
    Xatt = scaled(att_raw, i_sc); typ_src = att_typ
    deg = [i for i in range(len(Xva)) if len(Xva[i]) == 0]
    print(f"[{c['label']}] {d} features, {K} clients train {[len(x) for x in Xtr]} ; "
          f"appareils sans normal unique (dégénérés) : {deg} ; chargement {time.time()-t0:.0f}s", flush=True)

    # entraînements
    val_fed = [Xva[i] for i in range(len(Xva)) if len(Xva[i])] or [Xtr[0][:200]]
    avg_sd, _ = train_fed(Xtr, val_fed, ROUNDS, EPOCHS, 0.0, f"{a.dataset} FedAvg", seed=SEED)
    prox_sd, _ = train_fed(Xtr, val_fed, ROUNDS, EPOCHS, MU, f"{a.dataset} FedProx", seed=SEED)
    torch.manual_seed(SEED); cen = AE(d); train_client(cen, np.vstack(Xtr), epochs=ROUNDS * EPOCHS)

    # FedProx + fine-tuning : par client (device) si val non vide ; sinon on garde le modèle global
    if c["partition"] == "device":
        dft = {i: dict(xtr=Xtr[i], xva=Xva[i]) for i in range(K) if len(Xva[i])}
        ftm, _ = finetune(prox_sd, dft, list(dft))
        ft_model_for = lambda i: ftm[i] if i in ftm else _mk(prox_sd)
    else:
        ft_model_for = lambda i: _mk(prox_sd)

    def pooled_eval(model_for):
        """model_for(i) = modèle pour la source i. Seuil p99 sur val poolé ; FPR sur test normal
        poolé (sources non dégénérées) ; rappel/AUC sur attaques poolées (toutes sources)."""
        ev_va = np.concatenate([recon_error(model_for(i), Xva[i]) for i in range(len(Xva)) if len(Xva[i])])
        thr = float(np.percentile(ev_va, 99))
        ev_te = [recon_error(model_for(i), Xte[i]) for i in range(len(Xte)) if len(Xte[i])]
        err_n = np.concatenate(ev_te) if ev_te else np.array([])
        err_a = np.concatenate([recon_error(model_for(i), Xatt[i]) for i in range(len(Xatt)) if len(Xatt[i])])
        typ = np.concatenate([typ_src[i] for i in range(len(Xatt)) if len(Xatt[i])])
        y = np.r_[np.zeros(len(err_n)), np.ones(len(err_a))]; err = np.r_[err_n, err_a]
        auc = float(roc_auc_score(y, err)) if len(err_n) and len(err_a) else float("nan")
        pa = err_a >= thr
        per = {t: float(pa[typ == t].mean()) if (typ == t).any() else None for t in c["types"]}
        return dict(seuil=thr, auc=auc, rappel=float(pa.mean()),
                    fpr=float((err_n >= thr).mean()) if len(err_n) else float("nan"),
                    n_normal_test=int(len(err_n)), n_attaque=int(len(err_a)), par_type=per)

    res = {"dataset": c["label"], "n_clients": K, "features": d, "appareils_degeneres": deg,
           "note_degeneres": "appareils sans aucun normal unique en val/test après dédup (capteurs "
           "discrets très répétitifs) : exclus du seuil et du FPR, mais leurs attaques comptent dans le rappel"}
    for name, mf in (("fedavg", lambda i: _mk(avg_sd)), ("fedprox", lambda i: _mk(prox_sd)),
                     ("fedprox_ft", ft_model_for), ("central", lambda i: cen)):
        res[name] = pooled_eval(mf)
        g = res[name]
        print(f"  {name:<11} AUC={g['auc']:.4f} rappel={100*g['rappel']:.1f} FPR={100*g['fpr']:.1f}", flush=True)
    res["temps_min"] = round((time.time() - t0) / 60, 1)
    (OUT / f"ae_other_{a.dataset}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> ae_other_{a.dataset}.json ({res['temps_min']} min)")


if __name__ == "__main__":
    main()
