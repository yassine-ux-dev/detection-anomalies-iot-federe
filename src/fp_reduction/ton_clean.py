"""Protocole d'évaluation propre pour TON-IoT (seuil choisi sur la validation).

Même principe que pour BoT-IoT (voir README.md), appliqué au modèle TON-IoT
`final_model.pt` (fédéré, 7 clients, partition_by_file=False, sous-dataset IoT telemetry) :

- TRAIN      = union des splits train des 7 clients (reconstruits à l'index près) ;
- HELD-OUT   = union des splits test des 7 clients (jamais vus à l'entraînement) ;
- VALIDATION = 50 % du held-out (stratifié par type), TOUS les choix (seuil) ;
- TEST       = l'autre 50 %, utilisé une seule fois.

Le scaler MinMax est ajusté sur TRAIN. Seuils candidats choisis sur la validation :
origine (0,67, seuil du mémoire), F1-macro max, FPR <= 10 %, FPR <= 5 %.

    python ton_clean.py              # validation + test (le test sert une seule fois)
    python ton_clean.py --force      # réécrire le résultat de test (à éviter)
"""

from __future__ import annotations
import os

import argparse
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_recall_curve, roc_auc_score, roc_curve)

from common import NUM_CLIENTS, QS, RESULTS, SEED, TEST_SIZE, load_model, scores
from evaluate_variants import AQUA, BLUE, GRID, INK, INK2, ORANGE, wilson
from pytorchexample.task import (_load_engineered_files, _prepare_features_and_labels,
                                 get_feature_columns, list_csv_files)

torch.set_num_threads(4)
DATA_DIR = Path(os.environ.get("IOT_DATA", Path(__file__).resolve().parents[2] / "data")) / "Processed_IoT_dataset"
PATTERN = "IoT_*.csv"
CHECKPOINT = "final_model.pt"
ORIGINAL_THRESHOLD = 0.67
OUT = QS / "results" / "fp_reduction_ton"
MIN_GAP = 1e-3
ATTACK_TYPES = ["backdoor", "ddos", "injection", "password", "ransomware", "scanning", "xss"]
OP_LABELS = {"origine_0.67": "Origine (0,67)", "f1_macro_max": "F1-macro max",
             "fpr_10": "FPR ≤ 10 %", "fpr_5": "FPR ≤ 5 %"}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "lines.linewidth": 2, "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight"})


def build_split():
    files = list_csv_files(str(DATA_DIR), PATTERN)
    columns = get_feature_columns(files)
    df = pd.concat(_load_engineered_files(files), ignore_index=True)
    feats, y = _prepare_features_and_labels(df, columns)
    raw = feats.to_numpy(dtype=np.float32)
    y = y.astype(int)
    typ = df["type"].astype(str).str.strip().str.lower().to_numpy()
    typ[y == 0] = "normal"
    n = len(df)
    del df, feats

    perm = pd.Series(np.arange(n)).sample(frac=1.0, random_state=SEED).to_numpy()
    size = n // NUM_CLIENTS
    client_train, client_test = [], []
    for k in range(NUM_CLIENTS):
        part = perm[k * size:(k + 1) * size] if k < NUM_CLIENTS - 1 else perm[k * size:]
        a, b = train_test_split(part, test_size=TEST_SIZE, random_state=SEED, shuffle=True,
                                stratify=y[part])
        client_train.append(a)
        client_test.append(b)
    train_idx = np.concatenate(client_train)
    held = np.concatenate(client_test)
    val_idx, test_idx = train_test_split(held, test_size=0.5, random_state=SEED, shuffle=True,
                                         stratify=typ[held])
    scaler = MinMaxScaler().fit(raw[train_idx])
    return columns, raw, y, typ, scaler, train_idx, val_idx, test_idx


def sweep(y, p):
    order = np.argsort(-p, kind="mergesort")
    ps, ys = p[order], y[order]
    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    last = np.r_[ps[1:] != ps[:-1], True]
    t, tp, fp = ps[last], tp[last], fp[last]
    P, N = (y == 1).sum(), (y == 0).sum()
    fn, tn = P - tp, N - fp
    f1a = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    f1n = 2 * tn / np.maximum(2 * tn + fn + fp, 1)
    return t, fp / N, tp / P, (f1a + f1n) / 2


def choose_thresholds(y, p):
    """Mêmes seuils candidats que pour BoT-IoT. Le filtre d'écart minimal (MIN_GAP), utile quand
    beaucoup de flux ont un score identique (amas DNS de BoT-IoT), est ici une préférence : les
    scores TON-IoT sont continus, donc on ne l'applique que s'il laisse des candidats."""
    t, fpr, rec, f1m = sweep(y, p)
    gap = np.r_[t[:-1] - t[1:], 0.0]
    t = t - gap / 2
    valide = (t > 0.0) & (t < 1.0)
    ok = valide & (gap >= MIN_GAP)
    if not ok.any():
        ok = valide
    out = {"origine_0.67": ORIGINAL_THRESHOLD}
    out["f1_macro_max"] = float(t[ok][np.argmax(f1m[ok])])
    for name, lim in (("fpr_10", 0.10), ("fpr_5", 0.05)):
        c = ok & (fpr <= lim)
        out[name] = float(t[c].min()) if c.any() else None
    return out


def metrics(y, p, t, typ):
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    per = {}
    for a in ATTACK_TYPES:
        m = typ == a
        nn = int(m.sum())
        per[a] = dict(rappel=float(pred[m].mean()) if nn else math.nan, n=nn, ic95=wilson(int(pred[m].sum()), nn))
    return dict(seuil=t, f1_macro=float(f1_score(y, pred, average="macro")), auc=float(roc_auc_score(y, p)),
                ap_attaque=float(average_precision_score(y, p)), ap_normal=float(average_precision_score(1 - y, 1 - p)),
                mcc=float(matthews_corrcoef(y, pred)), balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                rappel=tp / max(tp + fn, 1), fpr=fp / max(fp + tn, 1), fpr_ic95=wilson(fp, fp + tn),
                precision=tp / max(tp + fp, 1), tn=tn, fp=fp, fn=fn, tp=tp, par_type=per)


def fig_threshold(y, p, th, path):
    t, fpr, rec, _ = sweep(y, p)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.3, 4.6), sharex=True, gridspec_kw=dict(hspace=0.12))
    a1.plot(t, 100 * fpr, color=BLUE); a1.set_ylabel("Faux positifs (%)"); a1.set_ylim(0, 100)
    a2.plot(t, 100 * rec, color=ORANGE); a2.set_ylabel("Rappel attaques (%)")
    a2.set_xlabel("Seuil de décision (attaque si P(attaque) ≥ seuil)"); a2.set_xlim(0, 0.95)
    for i, (k, v) in enumerate(th.items()):
        if v is None:
            continue
        for a in (a1, a2):
            a.axvline(v, color=INK2, lw=0.8, ls="--")
        a1.annotate(f"{OP_LABELS[k]} = {v:.3f}", (v, 92 - 13 * (i % 4)), xytext=(4, 0),
                    textcoords="offset points", fontsize=7, color=INK, va="top",
                    bbox=dict(fc="white", ec="none", pad=0.5))
    a1.set_title("Validation TON-IoT — faux positifs et rappel selon le seuil", loc="left", fontsize=10, color=INK)
    fig.savefig(path); plt.close(fig)


def fig_curves(y, p, path, title, ops):
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.1))
    f, r, _ = roc_curve(y, p)
    axes[0].plot(100 * f, 100 * r, color=BLUE, label=f"AUC = {roc_auc_score(y, p):.4f}")
    axes[0].plot([0, 100], [0, 100], color=GRID, lw=1)
    axes[0].set(xlabel="Taux de faux positifs (%)", ylabel="Rappel attaques (%)", title="ROC")
    for (name, (fx, ry)), mk in zip(ops.items(), "osD^"):
        axes[0].plot(100 * fx, 100 * ry, mk, ms=7, mfc="white", mec=INK, mew=1.2,
                     label=f"{OP_LABELS[name]} : FPR {100 * fx:.1f} %, rappel {100 * ry:.1f} %")
    axes[0].legend(loc="lower right", fontsize=6.5)
    pr, rc, _ = precision_recall_curve(y, p)
    axes[1].plot(100 * rc, 100 * pr, color=ORANGE, label=f"AP = {average_precision_score(y, p):.4f}")
    axes[1].set(xlabel="Rappel (%)", ylabel="Précision (%)", title="Précision-rappel — attaque")
    axes[1].legend(loc="lower left")
    pr0, rc0, _ = precision_recall_curve(1 - y, 1 - p)
    axes[2].plot(100 * rc0, 100 * pr0, color=AQUA, label=f"AP = {average_precision_score(1 - y, 1 - p):.4f}")
    axes[2].set(xlabel="Rappel (%)", ylabel="Précision (%)", title="Précision-rappel — normal")
    axes[2].legend(loc="lower left")
    for a in axes:
        a.title.set_fontsize(9); a.title.set_color(INK)
    for a in axes[1:]:
        a.set_xlim(-2, 102); a.set_ylim(-2, 102)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK); fig.tight_layout()
    fig.savefig(path); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    columns, raw, y, typ, scaler, train_idx, val_idx, test_idx = build_split()
    model = load_model(CHECKPOINT, len(columns))
    xf = lambda idx: np.clip(scaler.transform(raw[idx]), 0.0, 1.0).astype(np.float32)

    # VALIDATION : choix des seuils
    yv, pv = y[val_idx], scores(model, xf(val_idx))
    th = choose_thresholds(yv, pv)
    (OUT / "thresholds.json").write_text(json.dumps(th, indent=2), encoding="utf-8")
    val_res = {k: (metrics(yv, pv, v, typ[val_idx]) if v is not None else None) for k, v in th.items()}
    (OUT / "val_metrics.json").write_text(json.dumps(val_res, indent=2, ensure_ascii=False), encoding="utf-8")
    fig_threshold(yv, pv, th, OUT / "val_threshold_curve.png")
    fig_curves(yv, pv, OUT / "val_roc_pr.png", "Validation — TON-IoT",
               {k: (r["fpr"], r["rappel"]) for k, r in val_res.items() if r})

    # TEST : une seule fois
    tf = OUT / "test_metrics.json"
    if tf.exists() and not a.force:
        print(f"{tf} existe déjà : le test a déjà servi (règle : une seule fois). --force pour outrepasser.")
    else:
        yt, pt = y[test_idx], scores(model, xf(test_idx))
        test_res = {k: (metrics(yt, pt, v, typ[test_idx]) if v is not None else None) for k, v in th.items()}
        tf.write_text(json.dumps(test_res, indent=2, ensure_ascii=False), encoding="utf-8")
        fig_curves(yt, pt, OUT / "test_roc_pr.png", "Test — TON-IoT",
                   {k: (r["fpr"], r["rappel"]) for k, r in test_res.items() if r})

    info = dict(dataset="TON-IoT (IoT telemetry)", checkpoint=CHECKPOINT, features=len(columns),
                train=len(train_idx), validation=len(val_idx), test=len(test_idx),
                normaux_validation=int((yv == 0).sum()), normaux_test=int((y[test_idx] == 0).sum()))
    (OUT / "protocol_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, indent=1, ensure_ascii=False))
    for stage, res in [("VALIDATION", val_res), ("TEST", json.loads(tf.read_text(encoding="utf-8")))]:
        print(f"\n[{stage}]")
        for k, r in res.items():
            if r is None:
                print(f"  {k}: aucun seuil ne respecte la contrainte"); continue
            pt = "  ".join(f"{n}={v['rappel']:.3f}" for n, v in r["par_type"].items())
            print(f"  {k:<13} t={r['seuil']:.4f} F1m={r['f1_macro']:.4f} AUC={r['auc']:.4f} MCC={r['mcc']:.4f} "
                  f"BA={r['balanced_accuracy']:.4f} rappel={r['rappel']:.4f} FPR={r['fpr']:.4f}\n      {pt}")


if __name__ == "__main__":
    main()
