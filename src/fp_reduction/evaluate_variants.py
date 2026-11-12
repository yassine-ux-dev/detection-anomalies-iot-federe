"""Évaluation détaillée + réglage du seuil (étapes 1-2), en deux temps séparés.

    python evaluate_variants.py --stage val  --model origine
        -> scores de validation, courbes seuil/FPR/rappel, choix des seuils
           (origine 0.09, F1-macro max, FPR <= 10 %, FPR <= 5 %), figés dans
           results/fp_reduction/<model>/thresholds.json
    python evaluate_variants.py --stage test --model origine
        -> applique UNE SEULE FOIS les seuils figés au test ; refuse de tourner si
           test_metrics.json existe déjà (--force pour passer outre, à éviter).

Les seuils sont toujours choisis sur la validation. Le test ne sert à rien d'autre
qu'à la mesure finale.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_recall_curve, roc_auc_score, roc_curve)

from common import MODELS, RESULTS, Split, load_model, scores, type_masks

torch.set_num_threads(4)

# Palette de référence (mode clair, impression) — voir skill dataviz
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})

ORIGINAL_THRESHOLD = 0.09
OP_LABELS = {"origine_0.09": "Origine", "f1_macro_max": "F1-macro max",
             "fpr_10": "FPR ≤ 10 %", "fpr_5": "FPR ≤ 5 %"}


def sweep(y: np.ndarray, p: np.ndarray):
    """Pour chaque seuil candidat t (= valeurs distinctes du score, règle p >= t) :
    FPR, rappel attaque, F1-macro. Vectorisé (tri + sommes cumulées)."""
    order = np.argsort(-p, kind="mergesort")
    ps, ys = p[order], y[order]
    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    last = np.r_[ps[1:] != ps[:-1], True]  # dernier élément de chaque groupe d'égalité
    t, tp, fp = ps[last], tp[last], fp[last]
    P, N = (y == 1).sum(), (y == 0).sum()
    fn, tn = P - tp, N - fp
    f1_att = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    f1_nor = 2 * tn / np.maximum(2 * tn + fn + fp, 1)
    return t, fp / N, tp / P, (f1_att + f1_nor) / 2


MIN_GAP = 1e-3  # un seuil doit tomber dans un « trou » d'au moins 0.001 entre deux scores


def choose_thresholds(y: np.ndarray, p: np.ndarray) -> dict:
    """Seuils candidats = milieux des écarts >= MIN_GAP entre scores de validation
    consécutifs. Évite de couper au milieu d'un amas de flux quasi identiques
    (ex. les flux DNS normaux à 0.115598/0.115599, mêlés à des attaques UDP) :
    un tel seuil changerait de dizaines de flux pour une variation de 1e-6."""
    t, fpr, rec, f1m = sweep(y, p)  # t décroissant, valeurs distinctes
    gap = np.r_[t[:-1] - t[1:], 0.0]  # écart avec le score distinct immédiatement inférieur
    t = t - gap / 2  # même partition (p >= t) mais placée au milieu du trou
    ok = (t > 0.0) & (t < 1.0) & (gap >= MIN_GAP)
    out = {"origine_0.09": ORIGINAL_THRESHOLD}
    out["f1_macro_max"] = float(t[ok][np.argmax(f1m[ok])])
    for name, limit in (("fpr_10", 0.10), ("fpr_5", 0.05)):
        cand = ok & (fpr <= limit)
        # le plus petit seuil qui respecte la contrainte = le rappel maximal
        out[name] = float(t[cand].min()) if cand.any() else None
    return out


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (math.nan, math.nan)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def metrics(y, p, t, masks) -> dict:
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    per_type = {}
    for name, m in masks.items():
        n = int(m.sum()); k = int(pred[m].sum())
        per_type[name] = dict(rappel=k / n if n else math.nan, n=n, ic95=wilson(k, n))
    return dict(
        seuil=t, f1_macro=float(f1_score(y, pred, average="macro")),
        auc=float(roc_auc_score(y, p)), ap_attaque=float(average_precision_score(y, p)),
        ap_normal=float(average_precision_score(1 - y, 1 - p)),
        mcc=float(matthews_corrcoef(y, pred)), balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        rappel=tp / (tp + fn), fpr=fp / (fp + tn), fpr_ic95=wilson(fp, fp + tn),
        precision=tp / max(tp + fp, 1), tn=tn, fp=fp, fn=fn, tp=tp, par_type=per_type)


def fig_threshold(y, p, th: dict, path, title):
    t, fpr, rec, _ = sweep(y, p)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.3, 4.6), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1, 1], hspace=0.12))
    a1.plot(t, 100 * fpr, color=BLUE)
    a1.set_ylabel("Faux positifs (%)")
    a1.set_ylim(0, 100)
    a2.plot(t, 100 * rec, color=ORANGE)
    a2.set_ylabel("Rappel attaques (%)")
    lo = max(90.0, 100 * rec[(t > 0) & (t < 0.95)].min() - 0.5) if ((t > 0) & (t < 0.95)).any() else 90
    a2.set_ylim(lo, 100.05)
    a2.set_xlabel("Seuil de décision (attaque si P(attaque) ≥ seuil)")
    a2.set_xlim(0, 0.30)
    labels = OP_LABELS
    for i, (k, v) in enumerate(th.items()):
        if v is None:
            continue
        for a in (a1, a2):
            a.axvline(v, color=INK2, lw=0.8, ls="--")
        a1.annotate(f"{labels[k]} = {v:.3f}", (v, 97 - 13 * (i % 4)), xytext=(4, 0),
                    textcoords="offset points", fontsize=7, color=INK, va="top",
                    bbox=dict(fc="white", ec="none", pad=0.5))
    a1.set_title(title, loc="left", fontsize=10, color=INK)
    fig.savefig(path)
    plt.close(fig)


def fig_curves(y, p, path, title, ops: dict | None = None):
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.1))
    f, r, _ = roc_curve(y, p)
    axes[0].plot(100 * f, 100 * r, color=BLUE, label=f"AUC = {roc_auc_score(y, p):.4f}")
    axes[0].set(xlabel="Taux de faux positifs (%)", ylabel="Rappel attaques (%)",
                title="ROC (zoom)", xlim=(-1, 30), ylim=(95, 100.2))
    if ops:
        for (name, (fx, ry)), mk in zip(ops.items(), "osD^"):
            axes[0].plot(100 * fx, 100 * ry, mk, ms=7, mfc="white", mec=INK, mew=1.2,
                         label=f"{OP_LABELS.get(name, name)} : FPR {100 * fx:.1f} %, rappel {100 * ry:.2f} %")
    axes[0].legend(loc="lower right", fontsize=6.5)
    pr, rc, _ = precision_recall_curve(y, p)
    axes[1].plot(100 * rc, 100 * pr, color=ORANGE, label=f"AP = {average_precision_score(y, p):.4f}")
    axes[1].set(xlabel="Rappel (%)", ylabel="Précision (%)", title="Précision-rappel — classe attaque")
    axes[1].legend(loc="lower left")
    pr0, rc0, _ = precision_recall_curve(1 - y, 1 - p)
    axes[2].plot(100 * rc0, 100 * pr0, color=AQUA,
                 label=f"AP = {average_precision_score(1 - y, 1 - p):.4f}")
    axes[2].set(xlabel="Rappel (%)", ylabel="Précision (%)", title="Précision-rappel — classe normale")
    axes[2].legend(loc="lower left")
    for a in axes:
        a.title.set_fontsize(9); a.title.set_color(INK)
    for a in axes[1:]:
        a.set_xlim(-2, 102); a.set_ylim(-2, 102)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["val", "test"], required=True)
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    ckpt, exclude = MODELS[a.model]
    out = RESULTS / a.model
    out.mkdir(parents=True, exist_ok=True)

    s = Split(exclude)
    model = load_model(ckpt, len(s.columns))
    print(f"{a.model}: {ckpt}, {len(s.columns)} features, val {len(s.val_idx)}, test {len(s.test_idx)}")

    if a.stage == "val":
        y, p = s.y[s.val_idx], scores(model, s.x(s.val_idx))
        masks = type_masks(s.category[s.val_idx], s.subcategory[s.val_idx])
        th = choose_thresholds(y, p)
        (out / "thresholds.json").write_text(json.dumps(th, indent=2), encoding="utf-8")
        res = {k: (metrics(y, p, v, masks) if v is not None else None) for k, v in th.items()}
        (out / "val_metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        fig_threshold(y, p, th, out / "val_threshold_curve.png",
                      f"Validation — faux positifs et rappel selon le seuil ({a.model})")
        ops = {k: (r["fpr"], r["rappel"]) for k, r in res.items() if r}
        fig_curves(y, p, out / "val_roc_pr.png", f"Validation — {a.model}", ops)
        np.savez_compressed(out / "val_scores.npz", p=p.astype(np.float32), y=y.astype(np.int8))
    else:
        tf = out / "test_metrics.json"
        if tf.exists() and not a.force:
            sys.exit(f"{tf} existe déjà : le test a déjà été utilisé. Refus (règle : test une seule fois).")
        th = json.loads((out / "thresholds.json").read_text(encoding="utf-8"))
        y, p = s.y[s.test_idx], scores(model, s.x(s.test_idx))
        masks = type_masks(s.category[s.test_idx], s.subcategory[s.test_idx])
        res = {k: (metrics(y, p, v, masks) if v is not None else None) for k, v in th.items()}
        tf.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        ops = {k: (r["fpr"], r["rappel"]) for k, r in res.items() if r}
        fig_curves(y, p, out / "test_roc_pr.png", f"Test — {a.model}", ops)
        np.savez_compressed(out / "test_scores.npz", p=p.astype(np.float32), y=y.astype(np.int8))

    for k, r in res.items():
        if r is None:
            print(f"  {k}: aucun seuil ne respecte la contrainte")
            continue
        pt = "  ".join(f"{n}={v['rappel']:.3f}" for n, v in r["par_type"].items())
        print(f"  [{a.stage}] {k:<13} t={r['seuil']:.4f}  F1m={r['f1_macro']:.4f}  MCC={r['mcc']:.4f}  "
              f"BA={r['balanced_accuracy']:.4f}  rappel={r['rappel']:.4f}  FPR={r['fpr']:.4f} "
              f"(FP={r['fp']}/{r['fp'] + r['tn']})\n      {pt}")


if __name__ == "__main__":
    main()
