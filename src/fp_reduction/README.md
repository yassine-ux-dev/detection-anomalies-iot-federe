# `src/fp_reduction/` — Protocole d'évaluation propre & réduction des faux positifs

Corrige les biais d'évaluation du classifieur (recouvrement test/entraînement ; seuil choisi sur le
test) et étudie la réduction des faux positifs sur **BoT-IoT** et **TON-IoT**. Seuil toujours fixé
sur la **validation**, jamais sur le test.

| Fichier | Rôle |
|---|---|
| `common.py` | Fonctions partagées (chargement du modèle, scores, chemins). |
| `check_leak.py`, `check_protocol.py` | Vérifient l'absence de fuite entre entraînement et test. |
| `evaluate_variants.py` | Évalue les variantes (seuil fixé sur la validation). |
| `ton_clean.py` | Protocole propre appliqué à TON-IoT. |
| `train_behav.py`, `train_botiot_classweight.py` | Variantes réentraînées (sans heure, pondération). |
| `analyze_dns_features.py`, `shap_false_positives.py` | Analyse des faux positifs (DNS, SHAP). |
| `compare_behav.py`, `build_table.py`, `build_behav_table.py`, `export_scaler.py` | Comparaisons et tableaux. |

Résultats produits dans `results/fp_reduction/` et `results/fp_reduction_ton/`.
Détails : [`../../docs/EXPLICATION_DU_CODE.md`](../../docs/EXPLICATION_DU_CODE.md).
