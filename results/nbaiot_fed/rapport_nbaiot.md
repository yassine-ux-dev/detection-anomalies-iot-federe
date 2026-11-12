# N-BaIoT — autoencodeur fédéré auto-supervisé : résultats

Protocole : entraînement sur le trafic NORMAL uniquement (auto-supervisé), découpage **chronologique** 60/20/20 par appareil, scaler **local** par client, doublons de contenu retirés de val/test (**31484** lignes ; identiques restantes : 0). Attaques réservées au test (échantillon stratifié par appareil et famille). Seuils figés sur le NORMAL de validation (percentiles 95/99/99.9), en GLOBAL et LOCAL. Test exécuté une seule fois pour les 4 modèles.

> Note : un essai court préalable (1 round) avait affiché l'AUC/Mirai/BASHLITE du test ; aucun réglage n'en a découlé. Le nombre de rounds (20) et d'époques (2) a été jugé uniquement sur la loss de reconstruction du normal de validation (voir plus bas).

## 1. Comparaison globale des 4 modèles (test, seuil GLOBAL puis LOCAL)

| Modèle | AUC | p95 rap/FPR (G) | p99 rap/FPR (G) | p99.9 rap/FPR (G) | p99 rap/FPR (L) |
|---|--:|--:|--:|--:|--:|
| AE fédéré (principal) | 0.8661 | 77.7/4.5 | 73.6/1.1 | 23.0/0.1 | 67.1/0.8 |
| AE non entraîné (contrôle) | 0.7539 | 56.1/5.3 | 14.4/1.1 | 9.4/0.1 | 60.8/1.2 |
| AE centralisé (référence) | 0.9935 | 88.4/3.0 | 78.3/0.3 | 71.9/0.0 | 76.8/0.5 |
| AE local par appareil (contrôle) | 0.9929 | 93.6/3.3 | 86.7/0.4 | 67.2/0.0 | 86.7/0.4 |

Rappel = attaques détectées, FPR = faux positifs sur le normal de test (%). (G) seuil global, (L) seuil local par appareil.

## 2. Rappel par famille — AE fédéré, seuil p99

| Seuil | Rappel global | Mirai | BASHLITE | FPR |
|---|--:|--:|--:|--:|
| p95 (global) | 77.7 | 95.8 | 64.6 | 4.5 |
| p95 (local) | 79.3 | 99.6 | 64.6 | 3.9 |
| p99 (global) | 73.6 | 93.8 | 59.0 | 1.1 |
| p99 (local) | 67.1 | 87.7 | 52.1 | 0.8 |
| p99.9 (global) | 23.0 | 45.7 | 6.6 | 0.1 |
| p99.9 (local) | 28.2 | 54.8 | 8.9 | 0.1 |

## 3. AE fédéré par appareil (test, seuil p99 global)

| Appareil | AUC | Rappel | FPR | Mirai | BASHLITE |
|---|--:|--:|--:|--:|--:|
| Danmini_Doorbell | 0.8000 | 79.9 | 0.1 | 100.0 | 59.8 |
| Ecobee_Thermostat | 0.8075 | 71.4 | 3.8 | 82.9 | 59.9 |
| Ennio_Doorbell | 0.9954 | 60.0 | 0.1 | — | 60.0 |
| Philips_B120N10_Baby_Monitor | 0.9898 | 73.3 | 0.1 | 89.0 | 57.6 |
| Provision_PT_737E_Security_Camera | 0.8318 | 79.2 | 3.5 | 99.0 | 59.3 |
| Provision_PT_838_Security_Camera | 0.8591 | 77.8 | 0.8 | 96.9 | 58.7 |
| Samsung_SNH_1011_N_Webcam | 0.6147 | 59.6 | 0.0 | — | 59.6 |
| SimpleHome_XCS7_1002_WHT_Security_Camera | 0.7810 | 79.7 | 5.2 | 99.8 | 59.7 |
| SimpleHome_XCS7_1003_WHT_Security_Camera | 0.8000 | 73.2 | 0.0 | 86.6 | 59.8 |

## 4. Convergence (loss de reconstruction, normal de validation)

Round 1 : 0.030132 · round 5 : 0.017520 · round 10 : 0.013504 · round 20 : 0.012303. La baisse devient marginale après ~11 rounds (critère : amélioration < 1 % entre rounds), ce qui justifie 20 rounds.

## 5. Ressources (AE fédéré)

- Paramètres : 36 377 (~142.1 Ko par mise à jour de modèle)
- Octets échangés par round (9 clients  download+upload  float32) : 2 619 144 o (~2.5 Mo)
- Total sur 20 rounds : ~50.0 Mo
- Temps d'entraînement fédéré : 6.4 min (CPU)

## 6. Comparaison avec Meidan et al. (2018)

Meidan et al. (*N-BaIoT*, IEEE Pervasive Computing 2018) utilisent aussi des autoencodeurs entraînés sur le trafic normal, mais avec un protocole différent :

| Aspect | Meidan et al. 2018 | Ce travail |
|---|---|---|
| Modèle | **un AE par appareil** (local) | **AE fédéré** (un modèle commun, données non partagées) |
| Seuil | moyenne + écart-type de l'erreur sur val, + **vote majoritaire** sur une fenêtre de paquets | percentiles 95/99/99.9 par flux, **sans fenêtrage** |
| Découpage | train = début du normal, reste en test | **chronologique 60/20/20** |
| Doublons | non mentionné | **retirés** entre train/val/test |
| Résultat rapporté | TPR ≈ 100 %, FPR ≈ 0,7 % (après vote majoritaire) | voir tableaux ci-dessus (par flux) |

Notre AE **fédéré** atteint une AUC globale de 0.8661 et, au seuil p99 local, un rappel de 67.1 % pour 0.8 % de faux positifs, **par flux et sans vote majoritaire**, avec un seul modèle partagé plutôt qu'un modèle par appareil. Le vote majoritaire de Meidan et al. augmente mécaniquement le rappel et réduit le FPR ; nos chiffres par flux ne sont donc pas directement comparables, mais l'AE local (contrôle), qui reprend leur cadre (un modèle par appareil), atteint une AUC de 0.9929, cohérente avec leurs très bons résultats.

## 7. Lecture honnête

- **Contrôle validé** : l'AE non entraîné (AUC 0.754) est nettement moins bon que les AE entraînés — mais il sépare déjà à 0.75 car les attaques N-BaIoT (Mirai/BASHLITE) ont des valeurs extrêmes, faciles à mal reconstruire même avec des poids aléatoires.
- **L'AE local (AUC 0.993) et centralisé (AUC 0.994) sont quasi parfaits** : un modèle spécialisé par appareil (ou voyant toutes les données) sépare très bien normal et attaque.
- **L'AE fédéré est en retrait (AUC 0.866)**, et surtout très inégal entre appareils (0,61 pour Samsung, 0,80 pour plusieurs caméras, contre ~0,99 pour Ennio et Philips). La loss de reconstruction du normal, elle, converge bien : le problème n'est pas l'optimisation mais l'**hétérogénéité** — un modèle global unique (FedAvg) moyenne des comportements normaux très différents d'un appareil à l'autre, et perd le pouvoir discriminant sur les appareils atypiques. C'est un résultat honnête, pas un échec technique (pas de divergence, doublons retirés, seuils sur validation normale uniquement).
- **Conséquence** : cela motive directement l'étape 3 (FedProx vs FedAvg, IID vs non-IID) : FedProx est conçu pour limiter cette divergence sur données non-IID. Une personnalisation par appareil (ou un scaler fédéré partagé) est une piste complémentaire.
