"""Étape 4 — Tableau comparatif final (TEST) : CSV, Markdown, LaTeX + figure de compromis.

Lit results/fp_reduction/<model>/test_metrics.json (produits une seule fois par
evaluate_variants.py --stage test). Les seuils ont tous été choisis sur la validation.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from common import ATTACK_TYPES, MODELS, RESULTS
from evaluate_variants import AQUA, BLUE, INK, INK2, ORANGE, OP_LABELS

MODEL_LABELS = {"origine": "Origine", "sans_hour": "Sans hour\\_of\\_day",
                "class_weights_x5": "Class weights $\\times$5"}
MODEL_PLAIN = {"origine": "Origine", "sans_hour": "Sans hour_of_day", "class_weights_x5": "Class weights ×5"}
COLORS = {"origine": BLUE, "sans_hour": ORANGE, "class_weights_x5": AQUA}
# 0.09 = seuil d'origine du modèle d'origine ; sans signification pour les autres modèles
ROWS = {"origine": ["origine_0.09", "f1_macro_max", "fpr_10", "fpr_5"]}
DEFAULT_ROWS = ["f1_macro_max", "fpr_10", "fpr_5"]


def main() -> None:
    rows, ns = [], {}
    for m in MODELS:
        f = RESULTS / m / "test_metrics.json"
        if not f.exists():
            print(f"(absent : {m})")
            continue
        res = json.loads(f.read_text(encoding="utf-8"))
        for k in ROWS.get(m, DEFAULT_ROWS):
            r = res.get(k)
            if r is None:
                continue
            row = dict(modele=m, seuil_type=k, seuil=r["seuil"], f1_macro=r["f1_macro"], auc=r["auc"],
                       mcc=r["mcc"], balanced_accuracy=r["balanced_accuracy"], rappel=r["rappel"],
                       fpr=r["fpr"], fpr_ic95_bas=r["fpr_ic95"][0], fpr_ic95_haut=r["fpr_ic95"][1],
                       fp=r["fp"], normaux=r["fp"] + r["tn"], fn=r["fn"])
            for t, v in r["par_type"].items():
                row[f"rappel_{t}"] = v["rappel"]
                ns[t] = v["n"]
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "comparison_test.csv", index=False)

    types = list(ATTACK_TYPES)
    pct = lambda v: f"{100 * v:.2f}"
    # Markdown
    head = ["Variante", "Seuil", "F1-macro", "AUC", "MCC", "Bal. acc.", "Rappel %", "FPR % [IC95]"] + \
           [f"{t} %" for t in types]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in df.iterrows():
        md.append("| " + " | ".join([
            MODEL_PLAIN[r.modele], f"{OP_LABELS[r.seuil_type]} ({r.seuil:.3f})", f"{r.f1_macro:.4f}",
            f"{r.auc:.4f}", f"{r.mcc:.4f}", f"{r.balanced_accuracy:.4f}", pct(r.rappel),
            f"{pct(r.fpr)} [{100 * r.fpr_ic95_bas:.1f}–{100 * r.fpr_ic95_haut:.1f}]"]
            + [pct(r[f"rappel_{t}"]) for t in types]) + " |")
    md.append("")
    md.append("Effectifs du test par type : " + ", ".join(f"{t} n={ns[t]}" for t in types)
              + f" ; normaux n={int(df.normaux.iloc[0])}.")
    (RESULTS / "comparison_test.md").write_text("\n".join(md), encoding="utf-8")

    # LaTeX (booktabs)
    tex_types = {"DDoS": "DDoS", "DoS": "DoS", "Scan": "Scan", "Vol de données": "Vol",
                 "Keylogging": "Keylog."}
    tl = [r"% Généré par projet/fp_reduction/build_table.py — nécessite \usepackage{booktabs,graphicx}",
          r"\begin{table}[htbp]", r"\centering",
          r"\caption{BoT-IoT -- comparaison des variantes sur le jeu de test (utilisé une seule fois). "
          r"Tous les seuils sont choisis sur la validation. FPR = taux de faux positifs sur "
          f"{int(df.normaux.iloc[0])} flux normaux (IC 95 \\% de Wilson). Rappels par type en \\%"
          f" (effectifs : " + ", ".join(f"{tex_types[t]} {ns[t]}" for t in types) + r"). "
          r"Le seuil 0.090 est celui du modèle d'origine, choisi à l'époque sur l'ancien test ; il est "
          r"donné pour référence.}",
          r"\label{tab:botiot-fp-variantes}", r"\resizebox{\textwidth}{!}{%",
          r"\begin{tabular}{ll" + "r" * 6 + "r" * len(types) + "}", r"\toprule",
          r" & & & & & & & & \multicolumn{" + str(len(types)) + r"}{c}{Rappel par type (\%)} \\",
          r"\cmidrule(lr){9-" + str(8 + len(types)) + "}",
          r"Variante & Seuil & F1-macro & AUC & MCC & Bal.\ acc. & Rappel (\%) & FPR (\%) & "
          + " & ".join(tex_types[t] for t in types) + r" \\", r"\midrule"]
    prev = None
    for _, r in df.iterrows():
        if prev is not None and r.modele != prev:
            tl.append(r"\midrule")
        prev = r.modele
        lab = OP_LABELS[r.seuil_type].replace("≤", r"$\leq$").replace("%", r"\%")
        tl.append(" & ".join([
            MODEL_LABELS[r.modele], f"{lab} ({r.seuil:.3f})", f"{r.f1_macro:.3f}", f"{r.auc:.4f}",
            f"{r.mcc:.3f}", f"{r.balanced_accuracy:.3f}", f"{100 * r.rappel:.2f}",
            f"{100 * r.fpr:.1f} [{100 * r.fpr_ic95_bas:.1f}--{100 * r.fpr_ic95_haut:.1f}]"]
            + [f"{100 * r[f'rappel_{t}']:.2f}" for t in types]) + r" \\")
    tl += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (RESULTS / "comparison_test.tex").write_text("\n".join(tl) + "\n", encoding="utf-8")

    # Figure : courbes ROC de test (zoom) + points de fonctionnement choisis sur la validation
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    markers = {"origine_0.09": "o", "f1_macro_max": "s", "fpr_10": "D", "fpr_5": "^"}
    for m, g in df.groupby("modele", sort=False):
        z = np.load(RESULTS / m / "test_scores.npz")
        f, r, _ = roc_curve(z["y"], z["p"])
        ax.plot(100 * f, 100 * r, color=COLORS[m], lw=1.6,
                label=f"{MODEL_PLAIN[m]} (AUC {g.auc.iloc[0]:.4f})")
        for _, row in g.iterrows():
            ax.plot(100 * row.fpr, 100 * row.rappel, markers[row.seuil_type], color=COLORS[m], ms=8,
                    mec="white", mew=1.2, zorder=3)
    for k, mk in markers.items():
        ax.plot([], [], mk, color=INK2, ms=7, ls="none",
                label="Seuil 0.09 d'origine" if k == "origine_0.09" else OP_LABELS[k])
    ax.axvline(10, color=INK2, lw=0.8, ls="--")
    ax.text(10.4, 99.75, "objectif FPR 10 %", fontsize=7, color=INK2)
    ax.set(xlim=(0, 40), ylim=(94, 100.2))
    ax.set_xlabel("Taux de faux positifs (%) — test")
    ax.set_ylabel("Rappel attaques (%) — test")
    ax.set_title("Courbes ROC de test (zoom) et seuils choisis sur la validation", loc="left",
                 fontsize=10, color=INK)
    ax.legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    fig.savefig(RESULTS / "tradeoff_test.png")
    plt.close(fig)
    print("\n".join(md))


if __name__ == "__main__":
    main()
