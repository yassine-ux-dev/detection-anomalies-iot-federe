# `src/classifieur/` — Classifieur supervisé de référence (Flower)

Perceptron multicouche (MLP) entraîné de façon **fédérée** avec le cadriciel **Flower**, à partir des
étiquettes. Sert de point de comparaison à l'auto-encodeur.

| Fichier | Rôle |
|---|---|
| `pytorchexample/task.py` | Pipeline : lecture CSV, prétraitement, modèle `Net`, entraînement/évaluation. |
| `pytorchexample/client_app.py` | Côté client : entraînement local. |
| `pytorchexample/server_app.py` | Côté serveur : agrégation FedProx, suivi par tour. |
| `pyproject.toml` | Hyperparamètres (100 tours, FedProx μ = 0,1, lot 512…). |
| `run_config_*.toml` | Configurations par jeu (N-BaIoT, BoT-IoT) et comparaison FedAvg/FedProx. |

Exécution : `cd src/classifieur && flwr run . --run-config run_config_nbaiot.toml`.
Détails : [`../../docs/EXPLICATION_DU_CODE.md`](../../docs/EXPLICATION_DU_CODE.md).
