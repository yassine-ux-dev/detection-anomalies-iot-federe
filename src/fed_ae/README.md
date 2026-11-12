# `src/fed_ae/` — Auto-encodeur fédéré auto-supervisé (méthode principale)

Chaque client apprend **uniquement son trafic normal** (sans étiquette) ; l'**erreur de
reconstruction** sert de score d'anomalie. Entraînement **fédéré** (FedProx + affinage local).

| Fichier | Rôle |
|---|---|
| `nbaiot_data.py` | Chargement par appareil (découpage chronologique, normaliseur local, attaques au test). |
| `nbaiot_ae.py` | Modèle auto-encodeur + erreur de reconstruction. |
| `nbaiot_hetero.py` | Entraînement fédéré (FedAvg/FedProx) + affinage local + évaluation. |
| `nbaiot_noniid.py` | FedAvg → FedProx → affinage sur partition non-IID (**Figure 4.5**). |
| `diagnostic_noniid.py` | Mesure de l'hétérogénéité sans entraînement. |
| `nbaiot_seeds.py` | Robustesse sur 5 graines (**Tableau 4.2**). |
| `nbaiot_lodo.py` | Appareil jamais vu (*leave-one-device-out*). |
| `nbaiot_xai.py` | Explications SHAP de l'auto-encodeur (**Tableau 4.3**). |
| `other_ae.py` | Auto-encodeur sur TON-IoT et BoT-IoT (limites). |
| `sauvegarder_artifacts.py` | Entraîne et **sauvegarde** le modèle dans `artifacts/nbaiot/`. |
| `charger_modele.py` | **Recharge** le modèle et fait l'inférence sans réentraîner. |
| `build_hetero_report.py`, `build_nbaiot_report.py` | Rapports Markdown des résultats. |

Exécution : se placer dans ce dossier (`cd src/fed_ae`) puis `python <script>.py`.
Détails : [`../../docs/EXPLICATION_DU_CODE.md`](../../docs/EXPLICATION_DU_CODE.md).
