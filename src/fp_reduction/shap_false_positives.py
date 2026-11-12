"""Étape 5 — Les flux de type DNS sont-ils toujours des faux positifs ? (SHAP)

Travaille UNIQUEMENT sur la validation (le test reste réservé à la mesure finale).

Profil « type DNS » = proto_udp=1, state_INT=1, pkts <= 4 (les ports ne font pas
partie des features BoT-IoT harmonisées : le port 53 ne peut pas être vérifié ici,
d'où le terme « type DNS »).

Pour chaque modèle demandé :
- composition des faux positifs (type DNS / autres) à chaque seuil figé dans
  results/fp_reduction/<model>/thresholds.json ;
- SHAP (GradientExplainer, fond = 200 lignes du TRAIN) sur tous les flux normaux de
  type DNS de la validation : features qui les poussent vers « attaque » ;
- comparaison avec les attaques UDP (DoS/DDoS) de même profil, qui partagent l'amas.

    python shap_false_positives.py --model origine sans_hour class_weights_x5
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
import torch.nn as nn

from common import MODELS, RESULTS, Split, load_model, scores
from evaluate_variants import BLUE, INK, INK2, ORANGE, OP_LABELS

torch.set_num_threads(4)
TOP = 10


class Softmax(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return torch.softmax(self.m(x), dim=1)


def dns_mask(raw, cols):
    c = {n: cols.index(n) for n in ("proto_udp", "state_INT", "pkts")}
    return (raw[:, c["proto_udp"]] == 1) & (raw[:, c["state_INT"]] == 1) & (raw[:, c["pkts"]] <= 4)


def shap_attack(model, background, x):
    ex = shap.GradientExplainer(Softmax(model).eval(), torch.from_numpy(background))
    torch.manual_seed(0)
    sv = ex.shap_values(torch.from_numpy(x), nsamples=2000, rseed=0)
    sv = np.asarray(sv[1]) if isinstance(sv, list) else np.asarray(sv)[..., 1]
    return sv


def run(name: str) -> dict:
    ckpt, exclude = MODELS[name]
    out = RESULTS / name
    th = json.loads((out / "thresholds.json").read_text(encoding="utf-8"))
    s = Split(exclude)
    model = load_model(ckpt, len(s.columns))
    raw_v, y = s.raw[s.val_idx], s.y[s.val_idx]
    p = scores(model, s.x(s.val_idx))
    dns = dns_mask(raw_v, s.columns)
    sub = s.subcategory[s.val_idx]

    res = {"modele": name, "normaux_validation": int((y == 0).sum()),
           "normaux_type_dns": int((dns & (y == 0)).sum()),
           "attaques_type_dns": {k: int(v) for k, v in zip(*np.unique(sub[dns & (y == 1)], return_counts=True))},
           "score_median_normaux_dns": float(np.median(p[dns & (y == 0)])),
           "score_median_attaques_dns": float(np.median(p[dns & (y == 1)])),
           "faux_positifs": {}}
    for k, t in th.items():
        if t is None:
            continue
        fp = (p >= t) & (y == 0)
        res["faux_positifs"][k] = dict(seuil=t, total=int(fp.sum()), type_dns=int((fp & dns).sum()),
                                       autres=int((fp & ~dns).sum()),
                                       attaques_dns_detectees=float((p[dns & (y == 1)] >= t).mean()))

    rng = np.random.default_rng(0)
    bg = s.x(rng.choice(s.train_idx, 200, replace=False))
    x_norm = s.x(s.val_idx[dns & (y == 0)])
    att_idx = s.val_idx[dns & (y == 1)]
    x_att = s.x(rng.choice(att_idx, min(len(att_idx), len(x_norm)), replace=False))
    sv_n, sv_a = shap_attack(model, bg, x_norm), shap_attack(model, bg, x_att)
    mean_n, mean_a = sv_n.mean(0), sv_a.mean(0)
    order = np.argsort(-np.abs(mean_n))[:TOP]
    res["shap_normaux_dns_top"] = [dict(feature=s.columns[i], shap_moyen=float(mean_n[i]),
                                        shap_moyen_attaques_dns=float(mean_a[i])) for i in order]
    # contrôle d'additivité (GradientExplainer = approximation)
    base = float(Softmax(model).eval()(torch.from_numpy(bg)).detach()[:, 1].mean())
    res["shap_additivite_erreur_moyenne"] = float(np.abs(base + sv_n.sum(1) - scores(model, x_norm)).mean())

    # Figure 1 : SHAP moyen, normaux type DNS vs attaques UDP de même profil
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    yy = np.arange(TOP)
    ax.barh(yy - 0.2, mean_n[order], height=0.38, color=BLUE, label="Normaux type DNS")
    ax.barh(yy + 0.2, mean_a[order], height=0.38, color=ORANGE, label="Attaques UDP de même profil")
    ax.set_yticks(yy, [s.columns[i] for i in order])
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("SHAP moyen vers « attaque » (> 0 pousse vers attaque)")
    ax.set_title(f"Flux de type DNS — contributions SHAP ({name}, validation)", loc="left", fontsize=10, color=INK)
    ax.legend(loc="lower right", fontsize=7)
    fig.savefig(out / "shap_dns_flows.png")
    plt.close(fig)

    # Figure 2 : composition des faux positifs par seuil
    ks = list(res["faux_positifs"])
    d = np.array([res["faux_positifs"][k]["type_dns"] for k in ks])
    o = np.array([res["faux_positifs"][k]["autres"] for k in ks])
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    lab = [f"{OP_LABELS.get(k, k)}\n(t = {res['faux_positifs'][k]['seuil']:.3f})" for k in ks]
    ax.bar(lab, d, color=BLUE, label="Type DNS (UDP, INT, ≤ 4 paquets)", width=0.6, edgecolor="white", linewidth=2)
    ax.grid(axis="x", visible=False)
    ax.bar(lab, o, bottom=d, color=ORANGE, label="Autres", width=0.6, edgecolor="white", linewidth=2)
    for i, (a_, b_) in enumerate(zip(d, o)):
        ax.text(i, a_ + b_ + 2, f"{a_ + b_}", ha="center", fontsize=8, color=INK)
    ax.set_ylabel(f"Faux positifs (sur {res['normaux_validation']})")
    ax.set_title(f"Composition des faux positifs selon le seuil ({name}, validation)",
                 loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=7)
    ax.set_ylim(0, (d + o).max() * 1.2)
    fig.savefig(out / "fp_composition.png")
    plt.close(fig)

    (out / "shap_dns.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", choices=list(MODELS), required=True)
    for m in ap.parse_args().model:
        print(json.dumps(run(m), indent=2, ensure_ascii=False))
