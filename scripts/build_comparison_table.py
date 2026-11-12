"""Step 6.5 — Build the cross-dataset comparison table (TON-IoT / N-BaIoT / BoT-IoT).

Reads the three baseline_*.json files in results/ and produces a single
markdown table plus a CSV, both saved to results/comparison_table.*.

Usage (from projet/):
    python build_comparison_table.py
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"

FILES = {
    "TON-IoT": "baseline_ton_iot.json",
    "N-BaIoT": "baseline_nbaiot.json",
    "BoT-IoT": "baseline_botiot.json",
}

METRICS = [
    ("F1-macro", "f1_macro", "{:.4f}"),
    ("AUC", "auc", "{:.4f}"),
    ("Accuracy", "accuracy", "{:.4f}"),
    ("FPR", "false_positive_rate", "{:.4f}"),
    ("FNR", "false_negative_rate", "{:.4f}"),
    ("Test examples", "num_test_examples", "{:,}"),
]


def main() -> None:
    rows = {}
    for label, filename in FILES.items():
        path = RESULTS / filename
        data = json.loads(path.read_text())
        rows[label] = data

    header = "| Metric | " + " | ".join(FILES.keys()) + " |"
    sep = "|---" * (len(FILES) + 1) + "|"
    lines = [header, sep]
    for label, key, fmt in METRICS:
        cells = [fmt.format(rows[ds][key]) for ds in FILES]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    note = (
        "\n> **Note (2026-10-10).** Chaque colonne utilise le seuil qui maximise le F1-macro,\n"
        "> balayé sur le jeu de test de chaque dataset (méthodologie commune aux trois pour une\n"
        "> comparaison relative). Pour BoT-IoT, ces chiffres viennent de `baseline_botiot.json`,\n"
        "> marqué obsolète (test recouvrant l'entraînement) ; l'effet mesuré est nul et l'AUC est\n"
        "> inchangée. Chiffres de déploiement honnêtes (seuil choisi sur une validation séparée) :\n"
        "> `results/fp_reduction/` (BoT-IoT) et `results/fp_reduction_ton/` (TON-IoT). Voir\n"
        "> `results/OBSOLETE.md` et `CHIFFRES_OFFICIELS.md`.\n"
    )
    md = "\n".join(lines) + "\n" + note
    (RESULTS / "comparison_table.md").write_text(md, encoding="utf-8")

    csv_lines = ["metric," + ",".join(FILES.keys())]
    for label, key, _ in METRICS:
        cells = [str(rows[ds][key]) for ds in FILES]
        csv_lines.append(f"{label}," + ",".join(cells))
    (RESULTS / "comparison_table.csv").write_text("\n".join(csv_lines))

    print(md)
    print(f"\nSaved to {RESULTS / 'comparison_table.md'} and {RESULTS / 'comparison_table.csv'}")


if __name__ == "__main__":
    main()
