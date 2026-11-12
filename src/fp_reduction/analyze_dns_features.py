"""Priorité 1, points 1-2 : qu'est-ce qui sépare les flux DNS normaux des attaques UDP
sans hour_of_day ? (validation uniquement)

1. Statistiques des features Argus brutes pour DNS normal vs DoS/DDoS UDP.
2. SHAP (modèle sans hour_of_day) sur ces deux groupes.
3. Propose <= 5 features dérivées (sans heure ni identifiant) et mesure, sur la validation,
   leur pouvoir de séparation (AUC univariée DNS-normal vs UDP-attaque).

Sortie : results/fp_reduction/dns_feature_analysis.json (+ texte console).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import MODELS, RESULTS, Split, load_model, scores

pd.set_option("display.width", 200)
RAW_FEATS = ["pkts", "spkts", "dpkts", "bytes", "sbytes", "dbytes", "dur", "rate", "srate",
             "drate", "mean", "stddev", "min", "max", "sum"]


def groups(s: Split, idx):
    raw, y, sub = s.raw[idx], s.y[idx], s.subcategory[idx]
    c = {n: s.columns.index(n) for n in s.columns}
    udp = raw[:, c["proto_udp"]] == 1
    intst = raw[:, c["state_INT"]] == 1
    dns = (y == 0) & udp & intst & (raw[:, c["pkts"]] <= 4)
    att = (y == 1) & udp & (sub == "UDP")
    return raw, c, dns, att


def sep_auc(x_dns, x_att):
    """AUC univariée : séparer DNS-normal (0) de UDP-attaque (1) avec une feature."""
    y = np.r_[np.zeros(len(x_dns)), np.ones(len(x_att))]
    v = np.r_[x_dns, x_att]
    if len(np.unique(v)) < 2:
        return 0.5
    a = roc_auc_score(y, v)
    return max(a, 1 - a)  # orientation indifférente


def main():
    s = Split("")
    raw, c, dns, att = groups(s, s.val_idx)
    out = {"n_dns_normal": int(dns.sum()), "n_attaque_udp": int(att.sum()), "raw_stats": {}, "derivees": {}}
    print(f"Validation — DNS normaux : {dns.sum()} | attaques UDP : {att.sum()}\n")

    # 1. features brutes
    rows = []
    for f in RAW_FEATS:
        a, b = raw[dns, c[f]], raw[att, c[f]]
        rows.append(dict(feature=f, dns_med=np.median(a), udp_med=np.median(b),
                         auc_sep=sep_auc(a, b)))
        out["raw_stats"][f] = dict(dns_median=float(np.median(a)), udp_median=float(np.median(b)),
                                   dns_mean=float(a.mean()), udp_mean=float(b.mean()), auc_sep=float(sep_auc(a, b)))
    df = pd.DataFrame(rows).sort_values("auc_sep", ascending=False)
    print("Pouvoir de séparation des features BRUTES (AUC DNS-normal vs UDP-attaque) :")
    print(df.round(4).to_string(index=False))

    # 2. features dérivées candidates (sans heure ni identifiant), toutes reproductibles par
    #    n'importe quel extracteur (bytes, pkts, spkts, dpkts, dbytes, sbytes, dur mesurables).
    eps = 1e-6
    sb, db = raw[:, c["sbytes"]], raw[:, c["dbytes"]]
    sp, dp = raw[:, c["spkts"]], raw[:, c["dpkts"]]
    byt, pk, dur = raw[:, c["bytes"]], raw[:, c["pkts"]], raw[:, c["dur"]]
    deriv = {
        "octets_par_paquet": byt / (pk + eps),               # taille moyenne de paquet (bytes/pkts)
        "octets_par_paquet_aller": sb / (sp + eps),          # idem côté initiateur (sbytes/spkts)
        "duree_par_paquet": dur / (pk + eps),                # espacement moyen (dur/pkts)
        "octets_par_seconde": byt / (dur + eps),             # débit en octets
        "a_reponse": (dp > 0).astype(float),                 # au moins un paquet en retour (test idée)
        "ratio_dbytes_sbytes": db / (sb + eps),              # volume de réponse (test idée)
    }
    print("\nPouvoir de séparation des features DÉRIVÉES :")
    drows = []
    for name, v in deriv.items():
        a, b = v[dns], v[att]
        drows.append(dict(derivee=name, dns_med=np.median(a), udp_med=np.median(b), auc_sep=sep_auc(a, b)))
        out["derivees"][name] = dict(dns_median=float(np.median(a)), udp_median=float(np.median(b)),
                                     dns_mean=float(a.mean()), udp_mean=float(b.mean()), auc_sep=float(sep_auc(a, b)))
    print(pd.DataFrame(drows).sort_values("auc_sep", ascending=False).round(4).to_string(index=False))

    # percentiles du meilleur séparateur déployable
    opp = deriv["octets_par_paquet"]
    out["octets_par_paquet_percentiles"] = dict(
        dns={f"p{q}": float(np.percentile(opp[dns], q)) for q in (10, 25, 50, 75, 90)},
        udp={f"p{q}": float(np.percentile(opp[att], q)) for q in (10, 25, 50, 75, 90)})

    # combinaison : les 3 dérivées déployables non dégénérées, sur la validation
    from sklearn.linear_model import LogisticRegression
    best = ["octets_par_paquet", "duree_par_paquet", "octets_par_seconde"]
    X = np.column_stack([np.clip(deriv[n], -1e9, 1e9) for n in best])
    yy = np.r_[np.zeros(int(dns.sum())), np.ones(int(att.sum()))]
    Xg = np.vstack([X[dns], X[att]])
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xg, yy)
    auc_combo = roc_auc_score(yy, lr.predict_proba(Xg)[:, 1])
    out["auc_combinee_3features_validation"] = float(auc_combo)
    out["features_combinees"] = best
    print(f"\nAUC des 3 dérivées déployables combinées (régression logistique, validation) : {auc_combo:.4f}")

    # figure : distribution de octets_par_paquet
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from evaluate_variants import BLUE, GRID, INK, ORANGE
    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    bins = np.linspace(0, 200, 41)
    ax.hist(np.clip(opp[dns], 0, 200), bins=bins, density=True, color=BLUE, alpha=0.75,
            label=f"DNS normaux (n={int(dns.sum())})", edgecolor="white", linewidth=0.5)
    ax.hist(np.clip(opp[att], 0, 200), bins=bins, density=True, color=ORANGE, alpha=0.75,
            label=f"Attaques UDP (n={int(att.sum())})", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("octets par paquet (bytes / pkts)"); ax.set_ylabel("densité")
    ax.set_title("Séparateur déployable : octets/paquet (validation BoT-IoT)", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
    fig.savefig(RESULTS / "dns_octets_par_paquet.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    # SHAP sans hour_of_day sur les deux groupes : importance moyenne
    try:
        import shap, torch
        sh = Split("hour_of_day")
        rh, ch, dnsh, atth = groups(sh, sh.val_idx)
        model = load_model(MODELS["sans_hour"][0], len(sh.columns))
        import torch.nn as nn

        class SM(nn.Module):
            def __init__(s_, m): super().__init__(); s_.m = m
            def forward(s_, x): return torch.softmax(s_.m(x), 1)
        rng = np.random.default_rng(0)
        bg = sh.x(rng.choice(sh.train_idx, 200, replace=False))
        xd = sh.x(sh.val_idx[dnsh]); xa = sh.x(sh.val_idx[atth][rng.choice(int(atth.sum()), min(int(atth.sum()), len(xd)), replace=False)])
        ex = shap.GradientExplainer(SM(model).eval(), torch.from_numpy(bg))
        torch.manual_seed(0)
        sd = np.asarray(ex.shap_values(torch.from_numpy(xd), nsamples=400, rseed=0))[..., 1]
        sa = np.asarray(ex.shap_values(torch.from_numpy(xa), nsamples=400, rseed=0))[..., 1]
        diff = np.abs(sd.mean(0) - sa.mean(0))
        order = np.argsort(-diff)[:10]
        out["shap_sans_hour_diff_dns_vs_udp"] = [
            dict(feature=sh.columns[i], shap_dns=float(sd.mean(0)[i]), shap_udp=float(sa.mean(0)[i])) for i in order]
        print("\nSHAP (sans hour_of_day) — features qui DIFFÈRENT le plus entre DNS normal et UDP attaque :")
        for i in order:
            print(f"  {sh.columns[i]:<12} DNS={sd.mean(0)[i]:+.3f}  UDP={sa.mean(0)[i]:+.3f}  |écart|={diff[i]:.3f}")
    except Exception as e:
        out["shap_error"] = str(e)
        print("SHAP ignoré :", e)

    (RESULTS / "dns_feature_analysis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
