"""Step 7.2 -- Merge the federated model's results with the 7 literature
baselines into one comparison table across all three datasets.

Usage (from projet/):
    python build_literature_comparison.py
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"

DATASETS = {
    "ton_iot": ("TON-IoT", RESULTS / "baseline_ton_iot.json"),
    "nbaiot": ("N-BaIoT", RESULTS / "baseline_nbaiot.json"),
    "botiot": ("BoT-IoT", RESULTS / "baseline_botiot.json"),
}

BASELINE_MODELS = ["IsolationForest", "OneClassSVM", "LOF", "RandomForest", "XGBoost", "CNN", "LSTM"]


def main() -> None:
    rows = []  # (dataset_label, model_label, f1, auc, acc, fpr, fnr)

    for key, (label, path) in DATASETS.items():
        data = json.loads(path.read_text())
        rows.append((
            label, "Federated (Ours)",
            data["f1_macro"], data["auc"], data["accuracy"],
            data["false_positive_rate"], data["false_negative_rate"],
        ))
        lit_dir = RESULTS / "literature_baselines" / key
        for model in BASELINE_MODELS:
            model_path = lit_dir / f"{model}.json"
            if not model_path.exists():
                continue
            data = json.loads(model_path.read_text())
            rows.append((
                label, model,
                data["f1_macro"], data["auc"], data["accuracy"],
                data["false_positive_rate"], data["false_negative_rate"],
            ))

    # Sort: dataset order as declared, then F1-macro descending within each dataset
    dataset_order = {label: i for i, (label, _) in enumerate(DATASETS.values())}
    rows.sort(key=lambda r: (dataset_order[r[0]], -r[2]))

    header = "| Dataset | Model | F1-macro | AUC | Accuracy | FPR | FNR |"
    sep = "|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for ds, model, f1, auc, acc, fpr, fnr in rows:
        model_label = f"**{model}**" if model == "Federated (Ours)" else model
        lines.append(f"| {ds} | {model_label} | {f1:.4f} | {auc:.4f} | {acc:.4f} | {fpr:.4f} | {fnr:.4f} |")
    note = (
        "\n> **Note (2026-10-10).** Toutes les méthodes utilisent le seuil qui maximise le F1-macro,\n"
        "> balayé sur le jeu de test (méthodologie commune pour une comparaison équitable). La ligne\n"
        "> Federated BoT-IoT vient de `baseline_botiot.json`, marqué obsolète (test recouvrant\n"
        "> l'entraînement, effet mesuré nul, AUC inchangée). Chiffres de déploiement honnêtes :\n"
        "> `results/fp_reduction/`. Voir `results/OBSOLETE.md` et `CHIFFRES_OFFICIELS.md`.\n"
    )
    md = "\n".join(lines) + "\n" + note
    (RESULTS / "literature_comparison_table.md").write_text(md, encoding="utf-8")

    csv_lines = ["dataset,model,f1_macro,auc,accuracy,fpr,fnr"]
    for ds, model, f1, auc, acc, fpr, fnr in rows:
        csv_lines.append(f"{ds},{model},{f1:.4f},{auc:.4f},{acc:.4f},{fpr:.4f},{fnr:.4f}")
    (RESULTS / "literature_comparison_table.csv").write_text("\n".join(csv_lines))

    print(md)
    print(f"\nSaved to {RESULTS / 'literature_comparison_table.md'} "
          f"and {RESULTS / 'literature_comparison_table.csv'}")


if __name__ == "__main__":
    main()
