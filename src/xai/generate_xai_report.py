"""Final XAI reporting stage -- assembles a single, self-contained PDF report
per dataset from the artifacts already produced by explain_shap_generic.py
and evaluate_checkpoint.py, instead of presenting separate PNGs/CSVs by hand.

Reads (all relative to this file):
    results/baseline_<dataset>.json        -- performance metrics
    results/shap_<dataset>/shap_bar.png    -- global feature importance
    results/shap_<dataset>/shap_summary.png
    results/shap_<dataset>/shap_waterfall.png / shap_force_attack.png
    results/shap_<dataset>/shap_waterfall_normal.png / shap_force_normal.png
    results/shap_<dataset>/shap_report_data.json -- per-example SHAP contributions
                                                     (written by explain_shap_generic.py)
    results/comparison_table.csv           -- cross-dataset federated metrics (optional)

Writes:
    results/xai_report_<dataset>.pdf

All narrative text about "why this flow was flagged" is generated from the
actual numbers in shap_report_data.json, not hardcoded -- rerunning
explain_shap_generic.py with a different sample/seed and then this script
regenerates a report that matches the new numbers.

Usage:
    python generate_xai_report.py --dataset botiot --dataset-label "BoT-IoT" \
        --baseline-json results/baseline_botiot.json --shap-dir results/shap_botiot \
        --output results/xai_report_botiot.pdf
"""
from __future__ import annotations

import argparse
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[2]
PAGE_SIZE = (8.27, 11.69)  # A4 portrait, inches


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="Key used for file lookup, e.g. 'botiot'")
    p.add_argument("--dataset-label", required=True, help="Display name, e.g. 'BoT-IoT'")
    p.add_argument("--baseline-json", required=True)
    p.add_argument("--shap-dir", required=True)
    p.add_argument("--comparison-csv", default="results/comparison_table.csv")
    p.add_argument("--output", required=True)
    return p.parse_args()


def title_page(pdf: PdfPages, dataset_label: str, checkpoint: str, baseline: dict) -> None:
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")

    ax.text(0.5, 0.88, "Rapport d'explicabilite (XAI)", ha="center", fontsize=22, fontweight="bold")
    ax.text(0.5, 0.83, f"Detecteur d'anomalies IoT federe -- {dataset_label}", ha="center", fontsize=15, color="#333")
    ax.text(0.5, 0.79, f"Checkpoint : {checkpoint}", ha="center", fontsize=10, color="#666", family="monospace")
    ax.text(0.5, 0.76, f"Genere le {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            ha="center", fontsize=9, color="#999")

    ax.text(0.08, 0.66, "Metriques de performance (jeu de test complet, held-out)", fontsize=12, fontweight="bold")
    rows = [
        ("Accuracy", f"{baseline.get('accuracy', float('nan')):.4f}"),
        ("F1-macro", f"{baseline.get('f1_macro', float('nan')):.4f}"),
        ("AUC", f"{baseline.get('auc', float('nan')):.4f}"),
        ("Seuil de decision", f"{baseline.get('decision_threshold', float('nan')):.4f}"),
        ("Taux de faux positifs", f"{baseline.get('false_positive_rate', float('nan')):.4f}"),
        ("Taux de faux negatifs", f"{baseline.get('false_negative_rate', float('nan')):.4f}"),
        ("Exemples de test", f"{baseline.get('num_test_examples', 'n/a'):,}" if isinstance(baseline.get("num_test_examples"), int) else "n/a"),
    ]
    y = 0.60
    for label, value in rows:
        ax.text(0.10, y, label, fontsize=10.5, color="#333")
        ax.text(0.60, y, value, fontsize=10.5, family="monospace", fontweight="bold")
        y -= 0.035

    cm = baseline.get("confusion_matrix")
    if cm:
        ax.text(0.08, y - 0.02, "Matrice de confusion", fontsize=12, fontweight="bold")
        y -= 0.06
        ax.text(0.10, y, f"TN={cm.get('tn')}   FP={cm.get('fp')}   FN={cm.get('fn')}   TP={cm.get('tp')}",
                fontsize=10.5, family="monospace")

    if baseline.get("_obsolete"):
        ax.text(0.08, y - 0.07,
                "Note : ces metriques utilisent un seuil balaye sur le jeu de test, qui recouvre\n"
                "l'entrainement (voir results/OBSOLETE.md). Chiffres de deploiement honnetes, seuil\n"
                "choisi sur une validation separee : results/fp_reduction/ et CHIFFRES_OFFICIELS.md.\n"
                "L'analyse SHAP ci-apres (importance et sens des caracteristiques) reste valable.",
                fontsize=8.5, color="#b00", va="top")

    ax.text(0.5, 0.04,
            "Ce rapport combine : (1) l'importance globale des caracteristiques (SHAP), "
            "(2) le sens de leur effet,\n(3) une explication detaillee d'un exemple ATTAQUE et d'un exemple NORMAL, "
            "et (4) une mise en perspective\nentre les trois jeux de donnees du projet.",
            ha="center", fontsize=8.5, color="#666")
    pdf.savefig(fig)
    plt.close(fig)


def image_page(pdf: PdfPages, image_path: Path, title: str, caption: str) -> None:
    if not image_path.exists():
        return
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.97)
    ax_img = fig.add_axes((0.05, 0.18, 0.9, 0.75))
    ax_img.imshow(mpimg.imread(image_path))
    ax_img.axis("off")
    ax_text = fig.add_axes((0.07, 0.02, 0.86, 0.14))
    ax_text.axis("off")
    ax_text.text(0, 1, caption, fontsize=9.5, va="top", wrap=True)
    pdf.savefig(fig)
    plt.close(fig)


def format_contribution(c: dict) -> str:
    arrow = "-> ATTAQUE" if c["direction"] == "attack" else "-> NORMAL"
    return f"    - {c['feature']} = {c['value']:.4f}   SHAP = {c['shap_value']:+.4f}   {arrow}"


def example_page(pdf: PdfPages, shap_dir: Path, example: dict, label_fr: str,
                  waterfall_name: str, dataset_label: str) -> None:
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.suptitle(f"Pourquoi ce flux est classe {label_fr} -- {dataset_label}", fontsize=13.5, fontweight="bold", y=0.97)

    image_path = shap_dir / waterfall_name
    if image_path.exists():
        ax_img = fig.add_axes((0.06, 0.42, 0.88, 0.52))
        ax_img.imshow(mpimg.imread(image_path))
        ax_img.axis("off")

    ax_text = fig.add_axes((0.07, 0.03, 0.86, 0.35))
    ax_text.axis("off")

    f_x = example["f_x"]
    base = example["base_value"]
    true_class = example.get("true_class")
    true_label_fr = {1: "attaque (verite terrain)", 0: "normal (verite terrain)", None: "inconnue"}[true_class]
    predicted_fr = "ATTAQUE" if example["predicted_class"] == "attack" else "NORMAL"
    match = "coherente" if (example["predicted_class"] == "attack") == (true_class == 1) else "en desaccord"

    top = example["top_contributions"][:5]
    toward_attack = [c for c in top if c["direction"] == "attack"]
    toward_normal = [c for c in top if c["direction"] == "normal"]

    wrap_width = 92
    lines = textwrap.wrap(
        f"Score du modele f(x) = {f_x:.4f} (valeur de base = {base:.4f}) -> prediction = {predicted_fr}",
        wrap_width,
    )
    lines += textwrap.wrap(
        f"Etiquette reelle : {true_label_fr} -> la prediction est {match} avec la verite terrain.",
        wrap_width,
    )
    lines += [
        "",
        "Caracteristiques ayant le plus influence cette decision (par |valeur SHAP|) :",
    ]
    lines += [format_contribution(c) for c in top]
    lines.append("")
    if toward_attack and toward_normal:
        interp = (
            f"Interpretation : {toward_attack[0]['feature']} pousse le score vers ATTAQUE tandis que "
            f"{toward_normal[0]['feature']} tire vers NORMAL ; le resultat final ({predicted_fr}) reflete "
            "lequel de ces effets domine pour ce flux precis."
        )
    elif toward_attack:
        interp = (
            f"Interpretation : toutes les caracteristiques dominantes "
            f"({', '.join(c['feature'] for c in toward_attack[:3])}) poussent dans le meme sens (ATTAQUE), "
            "d'ou une decision tres confiante."
        )
    elif toward_normal:
        interp = (
            f"Interpretation : toutes les caracteristiques dominantes "
            f"({', '.join(c['feature'] for c in toward_normal[:3])}) poussent dans le meme sens (NORMAL), "
            "d'ou une decision tres confiante."
        )
    else:
        interp = ""
    lines += textwrap.wrap(interp, wrap_width)

    ax_text.text(0, 1, "\n".join(lines), fontsize=9, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def cross_dataset_page(pdf: PdfPages, comparison_csv: Path, dataset_label: str,
                        shap_dir: Path) -> None:
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.suptitle("Mise en perspective inter-jeux de donnees", fontsize=14, fontweight="bold", y=0.97)
    ax = fig.add_axes((0.07, 0.55, 0.86, 0.35))
    ax.axis("off")

    if comparison_csv.exists():
        import csv
        with open(comparison_csv) as f:
            rows = list(csv.reader(f))
        header, body = rows[0], rows[1:]
        table = ax.table(cellText=body, colLabels=header, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)
    else:
        ax.text(0, 0.5, "(comparison_table.csv introuvable)", fontsize=9)

    ax2 = fig.add_axes((0.07, 0.06, 0.86, 0.42))
    ax2.axis("off")

    other_shap_dirs = {
        "TON-IoT": ROOT / "results" / "shap_ton_iot" / "shap_importance.csv",
        "N-BaIoT": ROOT / "results" / "shap_nbaiot" / "shap_importance.csv",
        "BoT-IoT": ROOT / "results" / "shap_botiot" / "shap_importance.csv",
    }
    lines = [
        "Caracteristique dominante (mean |SHAP|) par jeu de donnees :",
    ]
    import pandas as pd
    for name, path in other_shap_dirs.items():
        if path.exists():
            df = pd.read_csv(path).sort_values("mean_abs_shap", ascending=False)
            top_feat, top_val = df.iloc[0]["feature"], df.iloc[0]["mean_abs_shap"]
            second_val = df.iloc[1]["mean_abs_shap"] if len(df) > 1 else float("nan")
            ratio = top_val / second_val if second_val else float("nan")
            marker = "  <-- ce rapport" if name.lower().replace("-", "") in dataset_label.lower().replace("-", "") else ""
            lines.append(f"  - {name}: {top_feat} ({top_val:.4f}, {ratio:.1f}x la 2e caracteristique){marker}")
        else:
            lines.append(f"  - {name}: (rapport SHAP non genere)")

    lines += [
        "",
        "Constat pour la discussion : la dominance de 'hour_of_day' observee sur TON-IoT est",
        "specifique a ce jeu de donnees, pas une propriete generale du modele -- sur N-BaIoT et",
        "BoT-IoT, ce sont des statistiques de flux reseau (paquets/timing/etat TCP) qui dominent.",
        "Ceci est coherent avec la nature des trois jeux : TON-IoT melange de la telemetrie de",
        "capteurs (l'heure de la journee est informative pour un comportement domestique), tandis",
        "que N-BaIoT/BoT-IoT sont du trafic reseau pur, ou la forme statistique du flux importe plus",
        "que le moment de la journee.",
    ]
    ax2.text(0, 1, "\n".join(lines), fontsize=9.5, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    shap_dir = ROOT / args.shap_dir
    baseline_path = ROOT / args.baseline_json
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(baseline_path) as f:
        baseline = json.load(f)

    report_data_path = shap_dir / "shap_report_data.json"
    report_data = {}
    if report_data_path.exists():
        with open(report_data_path) as f:
            report_data = json.load(f)
    checkpoint = report_data.get("checkpoint", baseline.get("checkpoint", "?"))

    with PdfPages(output_path) as pdf:
        title_page(pdf, args.dataset_label, checkpoint, baseline)

        image_page(
            pdf, shap_dir / "shap_bar.png",
            f"Importance globale des caracteristiques -- {args.dataset_label}",
            "Classement de toutes les caracteristiques par impact moyen absolu sur la sortie du "
            "modele (mean |SHAP value|). Plus une barre est longue, plus cette caracteristique "
            "pese, en moyenne sur l'ensemble de test, dans la decision ATTAQUE/NORMAL.",
        )
        image_page(
            pdf, shap_dir / "shap_summary.png",
            f"Sens de l'effet de chaque caracteristique -- {args.dataset_label}",
            "Chaque point est un flux du jeu de test. Position horizontale = impact sur la "
            "prediction (droite = pousse vers ATTAQUE, gauche = pousse vers NORMAL). Couleur = "
            "valeur de la caracteristique (rouge = elevee, bleu = faible). Un nuage rouge a droite "
            "signifie qu'une valeur elevee de cette caracteristique est associee a une attaque.",
        )

        examples = report_data.get("examples", {})
        if "attack" in examples:
            waterfall = "shap_waterfall.png" if (shap_dir / "shap_waterfall.png").exists() else "shap_force_attack.png"
            example_page(pdf, shap_dir, examples["attack"], "ATTAQUE", waterfall, args.dataset_label)
        if "normal" in examples:
            waterfall = "shap_waterfall_normal.png" if (shap_dir / "shap_waterfall_normal.png").exists() else "shap_force_normal.png"
            example_page(pdf, shap_dir, examples["normal"], "NORMAL", waterfall, args.dataset_label)

        cross_dataset_page(pdf, ROOT / args.comparison_csv, args.dataset_label, shap_dir)

    print(f"[SAVED] {output_path}")


if __name__ == "__main__":
    main()
