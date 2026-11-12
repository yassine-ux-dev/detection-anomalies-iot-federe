| Metric | TON-IoT | N-BaIoT | BoT-IoT |
|---|---|---|---|
| F1-macro | 0.6487 | 0.9977 | 0.9099 |
| AUC | 0.8131 | 0.9998 | 0.9962 |
| Accuracy | 0.7792 | 0.9993 | 0.9967 |
| FPR | 0.1890 | 0.0056 | 0.1985 |
| FNR | 0.4102 | 0.0003 | 0.0014 |
| Test examples | 721,227 | 1,412,522 | 202,228 |

> **Note (2026-10-10).** Chaque colonne utilise le seuil qui maximise le F1-macro,
> balayé sur le jeu de test de chaque dataset (méthodologie commune aux trois pour une
> comparaison relative). Pour BoT-IoT, ces chiffres viennent de `baseline_botiot.json`,
> marqué obsolète (test recouvrant l'entraînement) ; l'effet mesuré est nul et l'AUC est
> inchangée. Chiffres de déploiement honnêtes (seuil choisi sur une validation séparée) :
> `results/fp_reduction/` (BoT-IoT) et `results/fp_reduction_ton/` (TON-IoT). Voir
> `results/OBSOLETE.md` et `CHIFFRES_OFFICIELS.md`.
