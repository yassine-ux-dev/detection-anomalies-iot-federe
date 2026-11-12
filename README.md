# Détection d'anomalies IoT par apprentissage fédéré auto-supervisé et explicable

**Auteur :** Yassine Ben Charrada
**Encadrant :** Dr. Zied Jaoua
**Établissement :** Institut Supérieur d'Informatique et de Mathématiques de Monastir (ISIMM), Université de Monastir
**Année universitaire :** 2026 / 2027

## Description

Code du mémoire de mastère de recherche. Le système détecte des anomalies dans des environnements
IoT hétérogènes. La **méthode principale** est un **auto-encodeur fédéré auto-supervisé** : chaque
client apprend uniquement son trafic normal (sans étiquette) et signale les écarts par l'erreur de
reconstruction ; l'entraînement fédéré utilise FedProx et un affinage local. Un **classifieur
supervisé** (perceptron multicouche, via Flower) sert de référence. Les décisions sont expliquées
par **SHAP**. L'évaluation porte sur trois jeux publics : **TON-IoT, N-BaIoT, BoT-IoT**.

## Structure

```
projet/
├── README.md               ce fichier
├── requirements.txt        versions exactes
├── data/README.md          comment obtenir et placer les données (non incluses)
├── src/
│   ├── preprocessing/      harmonisation des données (harmonize_*.py)
│   ├── fed_ae/             AUTO-ENCODEUR fédéré auto-supervisé (méthode principale)
│   ├── classifieur/        classifieur de référence (application Flower + configs)
│   ├── fp_reduction/       protocole d'évaluation propre BoT-IoT / TON-IoT
│   └── xai/                explications SHAP
├── scripts/                scripts d'expériences du classifieur de référence
├── results/                résultats (CSV / JSON) cités dans le mémoire
└── artifacts/nbaiot/       auto-encodeur N-BaIoT sauvegardé (rechargeable sans réentraîner)
```

## Installation

```bash
python -m venv .venv
# Linux/macOS : source .venv/bin/activate   |  Windows : .venv\Scripts\activate
pip install -r requirements.txt
```
Python 3.11 recommandé. Les versions sont celles de l'environnement ayant produit les résultats.

## Données

Non incluses (~32 Go). Voir **`data/README.md`** : téléchargement, arborescence attendue,
harmonisation, et la variable `IOT_DATA` qui indique l'emplacement des données.

```bash
export IOT_DATA=/chemin/vers/data        # PowerShell : $env:IOT_DATA="C:\chemin\data"
```

## Exécuter les expériences

> Les scripts de l'auto-encodeur importent leurs modules voisins : les lancer **depuis
> `src/fed_ae/`**. Les scripts du classifieur importent le pipeline Flower : définir
> `PYTHONPATH=src/classifieur` (depuis la racine `projet/`).

### A. Auto-encodeur fédéré — méthode principale (N-BaIoT)

```bash
cd src/fed_ae
python sauvegarder_artifacts.py       # entraîne l'AE fédéré N-BaIoT + sauvegarde artifacts/nbaiot/
python nbaiot_seeds.py                # robustesse sur 5 graines
python nbaiot_noniid.py               # FedAvg vs FedProx vs affinage local (non-IID)
python nbaiot_lodo.py                 # appareil jamais vu (leave-one-device-out)
python nbaiot_xai.py                  # validité des explications SHAP de l'AE
python other_ae.py                    # AE sur TON-IoT et BoT-IoT (limites)
python charger_modele.py              # recharge l'AE sauvegardé, sans réentraîner (démo)
```

### B. Classifieur de référence (Flower)

```bash
cd src/classifieur
flwr run . --run-config run_config_nbaiot.toml      # N-BaIoT
flwr run . --run-config run_config_botiot.toml      # BoT-IoT
```

### C. Protocole propre BoT-IoT / TON-IoT (classifieur)

```bash
# depuis projet/ :
PYTHONPATH=src/classifieur python src/fp_reduction/evaluate_variants.py
PYTHONPATH=src/classifieur python src/fp_reduction/ton_clean.py
```

### D. Explications SHAP (classifieur)

```bash
PYTHONPATH=src/classifieur python src/xai/explain_shap_generic.py --data-dir "$IOT_DATA/Processed_BoT_IoT_dataset" --file-pattern "BoT_*.csv"
```

## Correspondance script → figure / tableau du mémoire

| Script | Produit | Mémoire |
|---|---|---|
| `src/fed_ae/sauvegarder_artifacts.py` | AE fédéré N-BaIoT (AUC 0,9998) + `artifacts/nbaiot/` | Tableau 4.1 (N-BaIoT), §4.2.1 |
| `src/fed_ae/other_ae.py` | AE auto-supervisé sur TON-IoT et BoT-IoT (limites) | Tableau 4.1 (TON-IoT, BoT-IoT) |
| `src/fed_ae/nbaiot_noniid.py`, `diagnostic_noniid.py` | FedAvg → FedProx → affinage local (non-IID) | Figure 4.5, §4.3 |
| `src/fed_ae/nbaiot_seeds.py` | robustesse sur 5 graines | Tableau 4.2, §4.4 |
| `src/fed_ae/nbaiot_lodo.py` | leave-one-device-out | §4.4 |
| `src/fed_ae/nbaiot_xai.py` | validité des explications de l'AE | Tableau 4.3, §4.5.1 |
| `src/classifieur/` (`flwr run …`) | classifieur fédéré de référence | Tableau 4.1 (référence), Figure 4.2 |
| `src/fp_reduction/evaluate_variants.py`, `ton_clean.py`, `check_leak.py` | protocole propre, modèle BoT-IoT retenu | §4.2.2, Tableau 4.1 (BoT-IoT référence) |
| `src/xai/explain_shap_generic.py` | importance SHAP globale du classifieur | Figure 4.6, §4.5.2 |
| `src/preprocessing/harmonize_nbaiot.py`, `harmonize_botiot.py` | harmonisation des jeux de données | §3.2 (préparation des données) |

Les chiffres de référence sont rassemblés dans `results/` (notamment `results/nbaiot_fed/` pour
l'auto-encodeur et `results/fp_reduction/` pour le classifieur).

## Temps d'exécution et matériel

- Matériel utilisé : **CPU** (aucun GPU requis).
- Auto-encodeur fédéré N-BaIoT (20 tours, 9 clients) : **~6 min**.
- Classifieur fédéré (100 tours) : de quelques minutes (N-BaIoT) à ~1 h (BoT-IoT), selon la taille du jeu.
- La lecture/harmonisation des données brutes (volumineuses) peut dominer le temps total.

## Reproductibilité

Graine fixée à **42**. L'auto-encodeur N-BaIoT peut être rechargé sans réentraînement via
`src/fed_ae/charger_modele.py` à partir de `artifacts/nbaiot/` (modèle global, modèles locaux,
normaliseurs, seuils p95/p99/p99,9).
