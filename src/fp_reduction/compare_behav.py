"""Compare, sur le protocole propre, le modèle actuel et les deux modèles sans heure.

Variantes :
  - origine : final_model_botiot.pt, 51 features (incl. hour_of_day), seuil figé 0,167 ;
  - behav   : final_model_botiot_behav.pt (sans hour, + octets_par_paquet) ;
  - deploy  : final_model_botiot_deploy.pt (sans hour ni stats Argus, + octets_par_paquet).

Pour behav/deploy : seuil choisi sur la VALIDATION (FPR <= 5 %, règle d'écart minimal).
Le modèle recommandé est choisi sur la validation AVANT le test ; le test sert une seule fois
pour les trois. Sortie : results/fp_reduction/behav/ (val/test metrics, décision, SHAP, tableau).

    python compare_behav.py --stage val     # scores de validation + choix du modèle recommandé
    python compare_behav.py --stage test     # test unique (refuse si déjà fait)
    python compare_behav.py --stage shap     # top-10 SHAP de behav et deploy (validation)
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch

from common import QS, RESULTS, Split, load_model, scores, type_masks
from evaluate_variants import wilson
from pytorchexample.task import Net

torch.set_num_threads(4)
OUT = RESULTS / "behav"
DERIVED = "octets_par_paquet"
MIN_GAP = 1e-3
ORIGINE_SEUIL = 0.16687420010566711  # decision_validation.json

# nom -> (checkpoint, meta.json ou None pour origine)
VARIANTS = {
    "origine": ("final_model_botiot.pt", None),
    "behav": ("final_model_botiot_behav.pt", "final_model_botiot_behav.pt.meta.json"),
    "deploy": ("final_model_botiot_deploy.pt", "final_model_botiot_deploy.pt.meta.json"),
}


def build_matrix(s: Split, cols: list) -> np.ndarray:
    """Matrice brute (lignes = toutes) pour une liste de features, avec octets_par_paquet dérivé."""
    base = {n: s.columns.index(n) for n in s.columns}
    bytes_i, pkts_i = base["bytes"], base["pkts"]
    out = np.empty((len(s.raw), len(cols)), dtype=np.float64)
    for j, c in enumerate(cols):
        if c == DERIVED:
            out[:, j] = s.raw[:, bytes_i] / np.clip(s.raw[:, pkts_i], 1, None)
        else:
            out[:, j] = s.raw[:, base[c]]
    return out


def variant_cols(name: str) -> list:
    _, meta = VARIANTS[name]
    if meta is None:
        return None  # origine = Split.columns (51)
    m = json.loads((QS / meta).read_text(encoding="utf-8"))
    return m["feature_columns"]


def scaler_fit(mat, train_idx):
    from sklearn.preprocessing import MinMaxScaler
    return MinMaxScaler().fit(mat[train_idx])


def scores_for(s: Split, name: str, idx):
    cols = variant_cols(name)
    if cols is None:
        model = load_model(VARIANTS[name][0], len(s.columns))
        return scores(model, s.x(idx))
    mat = build_matrix(s, cols)
    sc = scaler_fit(mat, s.train_idx)
    model = load_model(VARIANTS[name][0], len(cols))
    x = np.clip(sc.transform(mat[idx]), 0.0, 1.0).astype(np.float32)
    return scores(model, x)


def sweep(y, p):
    order = np.argsort(-p, kind="mergesort")
    ps, ys = p[order], y[order]
    tp = np.cumsum(ys == 1); fp = np.cumsum(ys == 0)
    last = np.r_[ps[1:] != ps[:-1], True]
    return ps[last], (fp[last] / (y == 0).sum())


def choose_fpr5(y, p):
    t, fpr = sweep(y, p)
    gap = np.r_[t[:-1] - t[1:], 0.0]
    tmid = t - gap / 2
    ok = (tmid > 0) & (tmid < 1) & (gap >= MIN_GAP) & (fpr <= 0.05)
    if not ok.any():
        ok = (tmid > 0) & (tmid < 1) & (fpr <= 0.05)
    return float(tmid[ok].min())


def metrics(y, p, t, typ):
    from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    per = {}
    for a, m in type_masks(typ[0], typ[1]).items():
        nn = int(m.sum()); per[a] = dict(rappel=float(pred[m].mean()) if nn else math.nan, n=nn)
    return dict(seuil=float(t), f1_macro=float(f1_score(y, pred, average="macro")),
                auc=float(roc_auc_score(y, p)), mcc=float(matthews_corrcoef(y, pred)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                rappel=tp / max(tp + fn, 1), fpr=fp / max(fp + tn, 1), fpr_ic95=wilson(fp, fp + tn),
                tn=tn, fp=fp, fn=fn, tp=tp, par_type=per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["val", "test", "shap"], required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    s = Split("")
    typv = (s.category[s.val_idx], s.subcategory[s.val_idx])
    typt = (s.category[s.test_idx], s.subcategory[s.test_idx])

    if a.stage == "shap":
        import shap
        import torch.nn as nn

        class SM(nn.Module):
            def __init__(s_, m): super().__init__(); s_.m = m
            def forward(s_, x): return torch.softmax(s_.m(x), 1)
        rng = np.random.default_rng(0)
        res = {}
        for name in ("behav", "deploy"):
            cols = variant_cols(name)
            mat = build_matrix(s, cols); sc = scaler_fit(mat, s.train_idx)
            xf = lambda idx: np.clip(sc.transform(mat[idx]), 0, 1).astype(np.float32)
            model = load_model(VARIANTS[name][0], len(cols))
            bg = xf(rng.choice(s.train_idx, 200, replace=False))
            xs = xf(s.val_idx[rng.choice(len(s.val_idx), 2000, replace=False)])
            ex = shap.GradientExplainer(SM(model).eval(), torch.from_numpy(bg))
            torch.manual_seed(0)
            sv = np.asarray(ex.shap_values(torch.from_numpy(xs), nsamples=200, rseed=0))[..., 1]
            imp = np.abs(sv).mean(0)
            order = np.argsort(-imp)[:10]
            res[name] = [dict(feature=cols[i], importance=float(imp[i])) for i in order]
            print(f"\n[{name}] top-10 SHAP (|valeur| moyenne, validation) :")
            for i in order:
                print(f"  {cols[i]:<22} {imp[i]:.4f}")
        (OUT / "shap_top10.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        return

    if a.stage == "val":
        th = {"origine": ORIGINE_SEUIL}
        val = {}
        for name in VARIANTS:
            pv = scores_for(s, name, s.val_idx)
            if name != "origine":
                th[name] = choose_fpr5(s.y[s.val_idx], pv)
            val[name] = metrics(s.y[s.val_idx], pv, th[name], typv)
            print(f"[val] {name:<8} t={th[name]:.4f}  F1m={val[name]['f1_macro']:.4f}  AUC={val[name]['auc']:.4f}  "
                  f"BA={val[name]['balanced_accuracy']:.4f}  rappel={val[name]['rappel']:.4f}  FPR={val[name]['fpr']:.4f}")
        (OUT / "thresholds.json").write_text(json.dumps(th, indent=2), encoding="utf-8")
        (OUT / "val_metrics.json").write_text(json.dumps(val, indent=2, ensure_ascii=False), encoding="utf-8")
        # recommandation : meilleure balanced accuracy sur la validation à FPR comparable (<=5%)
        reco = max(("behav", "deploy", "origine"), key=lambda n: val[n]["balanced_accuracy"])
        (OUT / "decision.json").write_text(json.dumps(dict(
            recommande_sur_validation=reco, seuils=th,
            note="choix sur la validation avant tout usage du test ; critère = balanced accuracy"), indent=2,
            ensure_ascii=False), encoding="utf-8")
        print(f"\nRecommandé sur la validation : {reco}")
        return

    # test
    tf = OUT / "test_metrics.json"
    if tf.exists() and not a.force:
        raise SystemExit(f"{tf} existe déjà : test déjà utilisé (une seule fois).")
    th = json.loads((OUT / "thresholds.json").read_text())
    test = {}
    for name in VARIANTS:
        pt = scores_for(s, name, s.test_idx)
        test[name] = metrics(s.y[s.test_idx], pt, th[name], typt)
        print(f"[test] {name:<8} t={th[name]:.4f}  F1m={test[name]['f1_macro']:.4f}  AUC={test[name]['auc']:.4f}  "
              f"BA={test[name]['balanced_accuracy']:.4f}  rappel={test[name]['rappel']:.4f}  FPR={test[name]['fpr']:.4f}")
    tf.write_text(json.dumps(test, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
