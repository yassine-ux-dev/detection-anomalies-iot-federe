# TON-IoT — protocole d'évaluation propre (seuil choisi sur la validation)

Produit par `fp_reduction/ton_clean.py`. Même protocole que pour BoT-IoT : validation et test
sont les deux moitiés (50/50, stratifiées par type) de l'union des splits test des 7 clients,
soit des lignes jamais vues à l'entraînement. Scaler ajusté sur l'entraînement. Tous les seuils
sont choisis sur la validation ; le test sert une seule fois.

Modèle `final_model.pt` (25 features). Train 2 884 903 flux ; validation 360 615 (308 699
normaux) ; test 360 616 (308 700 normaux).

## Résultats (test) — seuils choisis sur la validation

| Seuil (validation) | F1-macro | AUC | MCC | Bal. acc. | Rappel | FPR |
|---|---|---|---|---|---|---|
| Origine 0,67 | 0,650 | 0,814 | 0,328 | 0,701 | 59,1 % | 18,8 % |
| F1-macro max (0,672) | 0,650 | 0,814 | 0,327 | 0,700 | 58,4 % | 18,5 % |
| FPR ≤ 10 % (0,715) | 0,640 | 0,814 | 0,279 | 0,638 | 37,6 % | 10,0 % |
| FPR ≤ 5 % (0,750) | 0,610 | 0,814 | 0,246 | 0,592 | 23,2 % | 4,8 % |

Rappel par type au seuil 0,67 (test) : ddos 72,8 %, password 66,3 %, backdoor 58,7 %,
injection 53,3 %, scanning 42,7 %, **ransomware 0 %, xss 0 %**.

## Lecture

- **Le protocole propre confirme le baseline publié.** Au seuil 0,67, le test propre donne
  F1-macro 0,650, AUC 0,814, FPR 18,8 %, à comparer au `baseline_ton_iot.json` (0,649 / 0,813 /
  18,9 %). Comme le montrait déjà `leak_check_ton_iot.json`, le recouvrement de 80 % entre
  l'ancien test et l'entraînement n'a pas d'effet mesurable : les chiffres TON-IoT du mémoire
  restent valables.
- **Le seuil du mémoire (0,67) est justifié a posteriori** : le seuil qui maximise le F1-macro
  sur la validation est 0,672, soit la même valeur.
- **Contrairement à BoT-IoT, on ne peut pas descendre sous 10 % de faux positifs sans ruiner le
  rappel** : à FPR ≤ 5 %, le rappel tombe à 23 %. TON-IoT est un problème bien plus difficile
  (AUC 0,81 contre 0,996 pour BoT-IoT) ; ransomware et xss ne sont jamais détectés, quel que soit
  le seuil.

Fichiers : `thresholds.json`, `val_metrics.json`, `test_metrics.json`, `protocol_info.json`,
`val_threshold_curve.png`, `val_roc_pr.png`, `test_roc_pr.png`.
