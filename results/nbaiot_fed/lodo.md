# N-BaIoT — « nouvel appareil » (leave-one-device-out)

Appareil exclu de la fédération ; les 8 autres entraînent FedProx (μ=0.01). L'appareil exclu ne dispose que de ses N premières lignes normales (chronologique) ; normal-val et test inchangés. Seuils p95/p99/p99.9 sur son normal-val. Comparaison : AE local de zéro vs modèle fédéré + fine-tuning. (Mirai absent chez Ennio et Samsung : exclu de la moyenne Mirai.)

## N = 500 lignes normales

| Méthode | AUC | rappel p99 | FPR p99 | Mirai p99 | BASHLITE p99 |
|---|--:|--:|--:|--:|--:|
| AE local de zéro | 0.9920 ± 0.0096 | 83.5 ± 20.5 | 1.1 ± 1.5 | 96.5 ± 6.4 | 74.1 ± 32.6 |
| Fédéré + fine-tuning | 0.9887 ± 0.0114 | 79.5 ± 19.9 | 1.1 ± 1.5 | 89.7 ± 15.2 | 71.6 ± 26.2 |

Détail par appareil (AUC · rappel p99 · BASHLITE p99) :

| Appareil | local de zéro | fédéré + FT |
|---|--:|--:|
| Danmini_Doorbell | 1.000 · 100% · 100% | 1.000 · 100% · 100% |
| Ecobee_Thermostat | 0.993 · 80% · 60% | 0.993 · 80% · 60% |
| Ennio_Doorbell | 0.995 · 60% · 60% | 0.970 · 59% · 59% |
| Philips_B120N10_Baby_Monitor | 0.999 · 100% · 99% | 0.997 · 50% · 42% |
| Provision_PT_737E_Security_Camera | 0.969 · 43% · 4% | 0.972 · 59% · 40% |
| Provision_PT_838_Security_Camera | 0.986 · 69% · 44% | 0.980 · 68% · 44% |
| Samsung_SNH_1011_N_Webcam | 1.000 · 100% · 100% | 1.000 · 100% · 100% |
| SimpleHome_XCS7_1002_WHT_Security_Camera | 0.988 · 100% · 100% | 0.988 · 100% · 100% |
| SimpleHome_XCS7_1003_WHT_Security_Camera | 1.000 · 100% · 100% | 0.999 · 100% · 100% |

## N = 2000 lignes normales

| Méthode | AUC | rappel p99 | FPR p99 | Mirai p99 | BASHLITE p99 |
|---|--:|--:|--:|--:|--:|
| AE local de zéro | 0.9971 ± 0.0035 | 95.0 ± 9.4 | 0.8 ± 0.7 | 99.9 ± 0.2 | 90.0 ± 18.7 |
| Fédéré + fine-tuning | 0.9882 ± 0.0130 | 74.9 ± 25.2 | 1.2 ± 1.6 | 89.3 ± 23.9 | 62.6 ± 31.2 |

Détail par appareil (AUC · rappel p99 · BASHLITE p99) :

| Appareil | local de zéro | fédéré + FT |
|---|--:|--:|
| Danmini_Doorbell | 1.000 · 100% · 100% | 1.000 · 100% · 100% |
| Ecobee_Thermostat | 0.998 · 100% · 100% | 0.994 · 80% · 60% |
| Ennio_Doorbell | 0.994 · 100% · 100% | 0.971 · 59% · 59% |
| Philips_B120N10_Baby_Monitor | 0.999 · 79% · 59% | 0.998 · 72% · 47% |
| Provision_PT_737E_Security_Camera | 1.000 · 100% · 100% | 0.981 · 15% · 0% |
| Provision_PT_838_Security_Camera | 0.990 · 76% · 52% | 0.986 · 72% · 48% |
| Samsung_SNH_1011_N_Webcam | 1.000 · 100% · 100% | 1.000 · 100% · 100% |
| SimpleHome_XCS7_1002_WHT_Security_Camera | 0.993 · 100% · 100% | 0.964 · 75% · 50% |
| SimpleHome_XCS7_1003_WHT_Security_Camera | 1.000 · 100% · 100% | 1.000 · 100% · 100% |

## Lecture
Sur un appareil **jamais vu** de la fédération, l'AE local entraîné de zéro est **aussi bon ou meilleur** que le modèle fédéré + fine-tuning, et l'écart se creuse à N=2000 (rappel p99 moyen 95 % vs 75 %, BASHLITE 90 % vs 63 %). FedProx+FT ne s'effondre pas en AUC (aucun appareil < 0,90) mais **son rappel au seuil p99 chute** sur plusieurs appareils non vus (Provision-737E 15 %, Philips, SimpleHome-1002). L'avantage du fédéré+fine-tuning valait pour les appareils **membres** (voir seeds_summary) ; pour un **nouvel** appareil disposant de ses propres données, un AE local suffit. Limite honnête et défendable.