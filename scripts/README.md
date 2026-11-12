# `scripts/` — Expériences du classifieur de référence

Scripts lançables du classifieur supervisé. À exécuter depuis la racine du projet avec
`PYTHONPATH=src/classifieur` (ils importent le pipeline `pytorchexample`).

| Fichier | Rôle |
|---|---|
| `train_h2_centralized.py`, `train_h2_isolated.py` | Comparaison fédéré / centralisé / isolé (hypothèse H2). |
| `train_h1_leave_one_out.py` | *Leave-one-device-out* pour le classifieur. |
| `train_botiot_nohour.py` | Réentraînement BoT-IoT sans `hour_of_day`. |
| `eval_h2_models.py`, `evaluate_baseline.py` | Évaluation des modèles. |
| `literature_baselines.py` | Baselines classiques (IF, One-Class SVM, LOF, RF, XGBoost…) — *supplémentaires, hors mémoire final*. |
| `build_comparison_table.py`, `build_literature_comparison.py` | Tableaux comparatifs. |

Détails : [`../docs/EXPLICATION_DU_CODE.md`](../docs/EXPLICATION_DU_CODE.md).
