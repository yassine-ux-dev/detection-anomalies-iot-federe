# Synthèse — détection d'anomalies IoT fédérée, auto-supervisée et explicable (3 jeux)

Méthode **principale** : autoencodeur fédéré auto-supervisé (entraîné sur le NORMAL seul).
Baseline : classifieur supervisé (utilise les étiquettes). Tous les chiffres d'AE sont au seuil
p99 global sur le normal-val ; aucun réglage sur les attaques ; pas de feature temporelle ni
d'identifiant.

## Tableau comparatif (AUC ; rappel/FPR au seuil p99 quand pertinent)

| Jeu | AE **fédéré** (FedProx+FT) | AE local | AE centralisé | Classifieur supervisé (baseline) |
|---|---|---|---|---|
| **N-BaIoT** | **0,9998** (rappel 100 %, FPR 0,4 %) | 0,990 | 0,989 | 0,9998 (FPR 0,56 %) |
| **TON-IoT** | 0,623 (rappel 16 %) | — | 0,402 | 0,814 (rappel 59 %, FPR 18,8 %) |
| **BoT-IoT** | 0,454 | — | 0,675 | 0,989 (rappel 97,5 %, FPR 3,7 %) |

(FedAvg : N-BaIoT 0,843 · TON-IoT 0,477 · BoT-IoT 0,575. Source : `seeds_summary.json`,
`ae_other_ton.json`, `ae_other_botiot.json`, `fp_reduction/behav` et `CHIFFRES_OFFICIELS.md`.)

## Lecture honnête — où la méthode marche, où elle échoue

- **N-BaIoT : succès net.** 115 statistiques de flux (débits, jitter), normal abondant. L'AE
  fédéré FedProx + fine-tuning atteint AUC 0,9998 (robuste sur 5 graines, écart-type 0), **égal au
  classifieur supervisé** mais **sans étiquettes**. FedProx corrige l'effet non-IID de FedAvg
  (0,843 → 0,978 → 0,9998). Explicable (SHAP + erreur par feature cohérents, tests de validité OK).
- **TON-IoT : échec de l'AE.** Les 17 features retenues sont des relevés de capteurs discrets
  (porte, mouvement, lumière…) qui **ne reflètent pas les attaques réseau** : l'erreur de
  reconstruction ne sépare pas (AUC 0,40–0,62). De plus, 3 appareils sur 7 (capteurs on/off) n'ont
  aucun état normal unique en val/test. Le classifieur supervisé fait mieux (0,81), mais en partie
  grâce à des artefacts (`hour_of_day`), et reste modeste.
- **BoT-IoT : échec de l'AE par manque de normal.** Seulement 9 543 flux normaux (clients fédérés
  aussi petits que 41 lignes) : insuffisant pour apprendre le comportement normal (AUC 0,45–0,68).
  Le classifieur supervisé, lui, exploite les 99 % d'attaques et atteint 0,989.

## Conclusion pour le mémoire
La contribution centrale — **AE fédéré auto-supervisé, non-IID, explicable** — est **démontrée et
robuste sur N-BaIoT**, le jeu adapté (features comportementales de flux, normal abondant). Sur
TON-IoT et BoT-IoT, l'approche auto-supervisée **ne suffit pas** : les features de TON-IoT ne
portent pas l'attaque, et BoT-IoT manque de trafic normal. C'est un **résultat honnête et
instructif** : il délimite précisément le domaine de validité de la méthode et justifie, pour ces
deux jeux, soit un classifieur supervisé, soit des features de flux réseau (et non de télémétrie
capteur) avec davantage de trafic normal. Ressources : AE ~36 k paramètres, ~2,5 Mo échangés/round.
