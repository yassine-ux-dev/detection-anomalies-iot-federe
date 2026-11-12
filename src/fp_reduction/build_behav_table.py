"""Tableau comparatif (test) origine / behav / deploy + synthèse. Markdown + LaTeX.
Lit results/fp_reduction/behav/{val_metrics,test_metrics,shap_top10,decision}.json.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "results" / "fp_reduction" / "behav"
val = json.loads((OUT / "val_metrics.json").read_text(encoding="utf-8"))
test = json.loads((OUT / "test_metrics.json").read_text(encoding="utf-8"))
shap = json.loads((OUT / "shap_top10.json").read_text(encoding="utf-8"))
dec = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
TYPES = ["DDoS", "DoS", "Scan", "Vol de données", "Keylogging"]
LABELS = {"origine": "Origine (0,167)", "behav": "Sans heure + oct/paquet (behav)",
          "deploy": "100 % déployable (deploy)"}
NFEAT = {"origine": 51, "behav": 51, "deploy": 32}


def pct(v):
    return f"{100*v:.1f}"


def row(name, r):
    return [LABELS[name], str(NFEAT[name]), f"{r['seuil']:.3f}", f"{r['f1_macro']:.3f}", f"{r['auc']:.4f}",
            f"{r['mcc']:.3f}", f"{r['balanced_accuracy']:.3f}", pct(r['rappel']),
            f"{pct(r['fpr'])} [{pct(r['fpr_ic95'][0])}–{pct(r['fpr_ic95'][1])}]"] + \
           [pct(r['par_type'][t]['rappel']) for t in TYPES]

head = ["Variante", "Feat.", "Seuil", "F1-macro", "AUC", "MCC", "Bal. acc.", "Rappel %", "FPR % [IC95]"] + \
       [f"{t} %" for t in TYPES]
md = ["# BoT-IoT — modèles sans heure vs modèle actuel (test propre)", "",
      f"Protocole propre (test 101 115 flux, 955 normaux). Seuils choisis sur la validation "
      f"(FPR ≤ 5 %) ; test unique. Modèle recommandé sur la validation : **{dec['recommande_sur_validation']}**.",
      "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
for n in ("origine", "behav", "deploy"):
    md.append("| " + " | ".join(row(n, test[n])) + " |")
md += ["", f"Effectifs test par type : DDoS {test['origine']['par_type']['DDoS']['n']}, "
       f"DoS {test['origine']['par_type']['DoS']['n']}, Scan {test['origine']['par_type']['Scan']['n']}, "
       f"Vol de données {test['origine']['par_type']['Vol de données']['n']}, "
       f"Keylogging {test['origine']['par_type']['Keylogging']['n']}.", ""]
md += ["## Top-10 SHAP (validation)", ""]
for n in ("behav", "deploy"):
    feats = ", ".join(f"{x['feature']} ({x['importance']:.3f})" for x in shap[n][:10])
    md.append(f"- **{n}** : {feats}")
(OUT / "comparison_behav.md").write_text("\n".join(md), encoding="utf-8")

# LaTeX
esc = lambda x: x.replace("%", r"\%").replace("≤", r"$\leq$").replace("–", "--")
tl = [r"% fp_reduction/build_behav_table.py — \usepackage{booktabs,graphicx}",
      r"\begin{table}[htbp]\centering",
      r"\caption{BoT-IoT -- modèles sans \texttt{hour\_of\_day} comparés au modèle actuel, sur le jeu "
      r"de test propre (utilisé une seule fois). Seuils choisis sur la validation (FPR $\leq$ 5\,\%). "
      r"« deploy » n'utilise que des variables qu'un extracteur standard reproduit.}",
      r"\label{tab:botiot-sans-heure}", r"\resizebox{\textwidth}{!}{%",
      r"\begin{tabular}{lr" + "r" * 7 + "r" * len(TYPES) + "}", r"\toprule",
      " & & & & & & & & & \\multicolumn{%d}{c}{Rappel par type (\\%%)} \\\\" % len(TYPES),
      r"\cmidrule(lr){10-%d}" % (9 + len(TYPES)),
      "Variante & Feat. & Seuil & F1-m. & AUC & MCC & Bal.\\ acc. & Rappel (\\%) & FPR (\\%) & "
      + " & ".join(["DDoS", "DoS", "Scan", "Vol", "Keyl."]) + r" \\", r"\midrule"]
for n in ("origine", "behav", "deploy"):
    cells = row(n, test[n])
    cells[0] = {"origine": "Origine (0,167)", "behav": "Sans heure + oct./paq.",
                "deploy": r"\textbf{100 \% déployable}"}[n]
    cells[8] = test[n]['fpr']; cells[8] = f"{100*test[n]['fpr']:.1f}"  # drop CI in latex for width
    tl.append(" & ".join(esc(str(c)) for c in cells) + r" \\")
tl += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
(OUT / "comparison_behav.tex").write_text("\n".join(tl) + "\n", encoding="utf-8")
print("\n".join(md))
