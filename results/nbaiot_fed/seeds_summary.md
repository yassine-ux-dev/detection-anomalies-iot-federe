# N-BaIoT — robustesse sur 5 graines (moyenne ± écart-type)

Mêmes données/seuils ; graines [42, 49, 56, 63, 70] (init + ordre des lots). μ=0.01, 20 rounds. Aucun réglage sur les attaques.

| Modèle | AUC | rappel p99 | FPR p99 | Mirai p99 | BASHLITE p99 |
|---|--:|--:|--:|--:|--:|
| AE local (standard) | 0.9900 ± 0.0067 | 88.9 ± 3.1 | 0.4 ± 0.1 | 100.0 ± 0.0 | 80.9 ± 5.4 |
| AE local équitable | 0.9878 ± 0.0058 | 89.5 ± 1.1 | 0.5 ± 0.0 | 100.0 ± 0.0 | 81.9 ± 1.8 |
| AE centralisé | 0.9893 ± 0.0059 | 77.6 ± 2.1 | 0.3 ± 0.0 | 98.8 ± 2.5 | 62.3 ± 2.2 |
| FedAvg | 0.8433 ± 0.0154 | 67.5 ± 7.0 | 1.1 ± 0.2 | 87.0 ± 13.5 | 53.4 ± 3.4 |
| FedProx (μ=0.01) | 0.9781 ± 0.0010 | 73.3 ± 0.2 | 0.8 ± 0.0 | 99.7 ± 0.1 | 54.3 ± 0.3 |
| FedAvg + FT | 0.9850 ± 0.0036 | 76.9 ± 0.8 | 0.5 ± 0.0 | 99.5 ± 0.9 | 60.5 ± 1.1 |
| FedProx + FT | 0.9998 ± 0.0000 | 100.0 ± 0.0 | 0.4 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 |

Rappel/FPR/Mirai/BASHLITE en % au seuil p99 global. « AE local (standard) » = époques fixes sans early-stopping (comme l'étape 1) ; « AE local équitable » = jusqu'à 45 époques avec early-stopping sur normal-val (même budget que fédéré+fine-tuning).

## Comparaisons appariées (l'écart dépasse-t-il la variabilité entre graines ?)

| Comparaison | écart moyen | σ des écarts | significatif ? |
|---|--:|--:|---|
| FedProx+FT vs AE local équitable (AUC) | +0.0120 | 0.0058 | oui, net (>2σ) |
| FedProx+FT vs AE local équitable (BASHLITE_p99) | +0.1805 | 0.0183 | oui, net (>2σ) |
| FedProx vs FedAvg (AUC) | +0.1348 | 0.0144 | oui, net (>2σ) |
| FedAvg+FT vs FedAvg (AUC) | +0.1417 | 0.0164 | oui, net (>2σ) |

Écart apparié (même graine des deux côtés). « significatif » : >σ ou >2σ des écarts entre graines ; « non » = l'écart tient dans la variabilité (donc non concluant).