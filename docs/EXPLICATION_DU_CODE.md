# Explication du code

Ce document décrit, partie par partie, le rôle de chaque dossier et de chaque script, et comment
ils s'enchaînent pour produire les résultats du mémoire. Il s'adresse à un lecteur qui veut
comprendre l'organisation sans exécuter le code.

## 1. Vue d'ensemble

Le système détecte des anomalies (attaques) dans le trafic de dispositifs IoT. Il repose sur deux
modèles complémentaires :

- **Méthode principale — auto-encodeur fédéré auto-supervisé** (dossier `src/fed_ae/`). Chaque client
  (un appareil, une passerelle) apprend **uniquement son trafic normal**, sans étiquette. En
  exploitation, une observation mal reconstruite (erreur de reconstruction élevée) est jugée
  anormale. L'entraînement est **fédéré** (les données ne sont pas centralisées), avec l'algorithme
  **FedProx** et un **affinage local** par client.
- **Classifieur supervisé de référence** (dossier `src/classifieur/`). Un perceptron multicouche
  (MLP) entraîné avec les étiquettes, via le cadriciel **Flower**. Il sert de point de comparaison.

Les décisions sont expliquées par **SHAP** (dossier `src/xai/`). L'évaluation porte sur trois jeux
publics : **TON-IoT, N-BaIoT, BoT-IoT**.

**Chaîne de traitement :**

```
données brutes ──(src/preprocessing)──► données harmonisées
      │
      ├──► src/fed_ae/        auto-encodeur fédéré (méthode principale)   ──► results/nbaiot_fed/
      ├──► src/classifieur/   classifieur MLP fédéré (référence, Flower)  ──► results/baseline_*.json
      ├──► src/fp_reduction/  protocole d'évaluation propre (BoT-IoT/TON) ──► results/fp_reduction*/
      └──► src/xai/           explications SHAP                           ──► results/shap_*/
```

## 2. `src/preprocessing/` — préparation des données

Transforme les jeux bruts en un format commun (un CSV par source, une colonne d'étiquette binaire).

| Fichier | Rôle |
|---|---|
| `harmonize_nbaiot.py` | Réunit, pour chaque appareil N-BaIoT, le trafic bénin et les fichiers d'attaques (Mirai, BASHLITE) en un seul fichier étiqueté. |
| `harmonize_botiot.py` | Lit les 75 fichiers bruts de BoT-IoT, sous-échantillonne par catégorie, retire les colonnes d'identification (IP, MAC, ports) et produit un fichier unique. |

## 3. `src/fed_ae/` — auto-encodeur fédéré (MÉTHODE PRINCIPALE)

Le cœur du mémoire. Les modules se lisent dans cet ordre.

| Fichier | Rôle |
|---|---|
| `nbaiot_data.py` | Chargement par appareil : découpage **chronologique** 60/20/20 du trafic normal, retrait des doublons, normaliseur **local** par client, attaques réservées au test. |
| `nbaiot_ae.py` | Définition de l'auto-encodeur (encodeur/décodeur) et de l'**erreur de reconstruction** (le score d'anomalie). |
| `nbaiot_hetero.py` | Entraînement **fédéré** (FedAvg / FedProx) + **affinage local** ; évaluation (AUC, rappel, FPR par seuil). |
| `nbaiot_noniid.py` | Compare FedAvg → FedProx → FedProx + affinage sur la partition non-IID (un client = un appareil). → **Figure 4.5**. |
| `diagnostic_noniid.py` | Mesure l'hétérogénéité **sans entraînement** (ex. part des attaques plus proches d'un autre appareil que du sien). |
| `nbaiot_seeds.py` | Répète l'expérience sur **5 graines** pour la robustesse (moyenne ± écart-type). → **Tableau 4.2**. |
| `nbaiot_lodo.py` | *Leave-one-device-out* : performance sur un appareil **jamais vu** à l'entraînement. |
| `nbaiot_xai.py` | Explications de l'auto-encodeur (SHAP sur l'erreur de reconstruction) + tests de validité. → **Tableau 4.3**. |
| `other_ae.py` | Applique l'auto-encodeur à **TON-IoT** et **BoT-IoT** (où la méthode atteint ses limites). |
| `sauvegarder_artifacts.py` | Entraîne l'AE N-BaIoT et **sauvegarde** le modèle global, les modèles locaux, les normaliseurs et les seuils dans `artifacts/nbaiot/`. |
| `charger_modele.py` | **Recharge** le modèle sauvegardé et fait l'inférence **sans réentraîner** (utile pour vérifier rapidement). |
| `build_hetero_report.py`, `build_nbaiot_report.py` | Construisent les rapports Markdown de `results/nbaiot_fed/`. |

## 4. `src/classifieur/` — classifieur de référence (Flower)

Application d'apprentissage fédéré standard, basée sur l'exemple *quickstart-pytorch* de Flower.

| Fichier | Rôle |
|---|---|
| `pytorchexample/task.py` | Le pipeline : lecture des CSV, prétraitement commun, définition du **MLP** (`Net`), entraînement et évaluation locaux. |
| `pytorchexample/client_app.py` | Côté **client** Flower : entraîne le modèle sur la partition locale. |
| `pytorchexample/server_app.py` | Côté **serveur** Flower : agrégation **FedProx** et suivi des performances par tour. |
| `pyproject.toml` | Hyperparamètres de l'entraînement fédéré (100 tours, FedProx μ = 0,1, lot 512, etc.). |
| `run_config_*.toml` | Variantes de configuration par jeu (N-BaIoT, BoT-IoT) et pour la comparaison FedAvg/FedProx. |

Exécution : `cd src/classifieur && flwr run . --run-config run_config_nbaiot.toml`.

## 5. `src/fp_reduction/` — protocole d'évaluation propre

Corrige deux biais d'évaluation du classifieur (recouvrement test/entraînement, seuil choisi sur le
test) et étudie la réduction des faux positifs sur BoT-IoT et TON-IoT.

| Fichier | Rôle |
|---|---|
| `common.py` | Fonctions partagées : chargement du modèle, calcul des scores, chemins des résultats. |
| `check_leak.py`, `check_protocol.py` | Vérifient l'absence de fuite entre entraînement et test. |
| `evaluate_variants.py` | Évalue les variantes du modèle avec un **seuil fixé sur la validation** (jamais sur le test). |
| `ton_clean.py` | Même protocole propre appliqué à **TON-IoT**. |
| `train_behav.py`, `train_botiot_classweight.py` | Réentraînent des variantes (sans heure, pondération des classes). |
| `analyze_dns_features.py`, `shap_false_positives.py` | Analysent les **faux positifs** (features DNS, SHAP). |
| `compare_behav.py`, `build_table.py`, `build_behav_table.py`, `export_scaler.py` | Comparaisons et construction des tableaux de résultats. |

## 6. `src/xai/` — explicabilité (SHAP)

| Fichier | Rôle |
|---|---|
| `explain_shap_generic.py` | Calcule les valeurs **SHAP** du classifieur pour un jeu donné et produit les figures (importance globale, *waterfall*, *force*). → **Figure 4.6**. |
| `generate_xai_report.py` | Assemble les figures SHAP en un **rapport PDF** par jeu (`results/xai_report_*.pdf`). |

## 7. `scripts/` — expériences du classifieur

Scripts lançables du classifieur de référence (à exécuter avec `PYTHONPATH=src/classifieur`).

| Fichier | Rôle |
|---|---|
| `train_h2_centralized.py`, `train_h2_isolated.py` | Comparaison fédéré vs centralisé vs isolé (hypothèse H2). |
| `train_h1_leave_one_out.py` | *Leave-one-device-out* pour le classifieur. |
| `train_botiot_nohour.py` | Réentraînement BoT-IoT sans la caractéristique `hour_of_day`. |
| `eval_h2_models.py`, `evaluate_baseline.py` | Évaluation des modèles entraînés. |
| `literature_baselines.py` | Baselines classiques (Isolation Forest, One-Class SVM, LOF, RF, XGBoost…) — *résultats supplémentaires, non repris dans le mémoire final.* |
| `build_comparison_table.py`, `build_literature_comparison.py` | Construction des tableaux comparatifs. |

## 8. `results/` et `artifacts/`

- **`results/`** : tous les résultats chiffrés et figures cités dans le mémoire (voir `results/README.md`
  pour le détail). En particulier `results/nbaiot_fed/` (auto-encodeur) et `results/fp_reduction*/`
  (classifieur, protocole propre).
- **`artifacts/nbaiot/`** : l'auto-encodeur N-BaIoT entraîné (modèle global + 9 modèles locaux +
  normaliseurs + seuils), **rechargeable sans réentraînement** via `src/fed_ae/charger_modele.py`.

## 9. Où regarder en priorité

1. `src/fed_ae/nbaiot_hetero.py` et `nbaiot_noniid.py` — la méthode principale et l'effet du fédéré.
2. `results/nbaiot_fed/synthese_3jeux.md` — la synthèse des résultats sur les trois jeux.
3. `src/classifieur/pytorchexample/` — le classifieur de référence.
4. `README.md` (racine) — installation, données et correspondance script → figure/tableau du mémoire.
