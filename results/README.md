# `results/` — Résultats

Tous les résultats chiffrés et les figures cités dans le mémoire.

| Dossier / fichier | Contenu | Mémoire |
|---|---|---|
| `nbaiot_fed/` | **Auto-encodeur fédéré (méthode principale)** : `ae_fed/central/local.json`, `seeds_summary` (5 graines), `hetero_variants` (FedAvg/FedProx/affinage), `lodo`, `xai_ae`, `ae_other_ton/botiot`, `synthese_3jeux.md` | Tableaux 4.1–4.3, Figure 4.5 |
| `fp_reduction/` | **Classifieur BoT-IoT, protocole propre** : `origine/`, `behav/`, `sans_hour/`, `class_weights_x5/` (métriques test/val, seuils, courbes ROC/PR, SHAP), `leak_check`, `decision_validation` | §4.2.2, Tableau 4.1 |
| `fp_reduction_ton/` | TON-IoT, protocole propre (métriques, seuils, courbes) | TON-IoT |
| `shap_nbaiot/`, `shap_botiot/`, `shap_ton_iot/` | Explications SHAP (bar, summary, *waterfall*, *force*, `shap_importance.csv`) | Figure 4.6 |
| `literature_baselines/` | Baselines classiques (IF, One-Class SVM, LOF, RF, XGBoost, CNN, LSTM) — *supplémentaires, non repris dans le mémoire final* | — |
| `baseline_*.json`, `comparison_table.*` | Récapitulatif des performances par jeu | Figure 4.2, Tableau 4.1 |
| `xai_report_*.pdf` | Rapports SHAP complets (PDF) par jeu | §4.5 |

Le **modèle entraîné réutilisable** est dans [`../artifacts/nbaiot/`](../artifacts) (rechargeable via
`src/fed_ae/charger_modele.py`).
