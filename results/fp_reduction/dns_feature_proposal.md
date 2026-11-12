# Priorité 1, points 1-2 : séparer DNS normal / attaque UDP sans hour_of_day

Analyse sur la **validation** BoT-IoT (`fp_reduction/analyze_dns_features.py`,
`dns_feature_analysis.json`). 336 flux DNS normaux (UDP, état INT, ≤ 4 paquets) contre
44 591 flux d'attaque UDP (DoS/DDoS).

## Point 1 — Ce qui les distingue aujourd'hui, et pourquoi ce n'est pas déployable

SHAP sur le modèle **sans hour_of_day** : les features qui séparent le plus DNS normal et
attaque UDP sont `max`, `mean`, `stddev`, `min` — c'est-à-dire les **statistiques de durée des
« enregistrements agrégés » d'Argus**. Pour un flux DNS normal elles valent 0 (un seul
enregistrement), pour une attaque UDP elles sont élevées (médianes : max 4,3 ; mean 2,9 ; sum 9,6).

| Feature brute | médiane DNS | médiane UDP | AUC séparation |
|---|---|---|---|
| sum, max, mean | 0 | 2,9–9,6 | 0,946 |
| stddev | 0 | 1,3 | 0,903 |
| spkts / pkts | 2 | 7 | 0,931 |
| dur | 5,0 | 21,6 | 0,880 |
| bytes / sbytes | 176 | 420 | 0,869 |
| **dbytes, dpkts, drate** | **0** | **0** | **0,500** |

**C'est précisément le problème du mémoire** : les meilleurs séparateurs (max/mean/stddev/sum)
sont les features Argus agrégées, non reproductibles par un autre extracteur (paramètres
d'agrégation non publiés) — la cause du décalage de domaine observé au labo.

**Les idées « présence de réponse » et « ratio dbytes/sbytes » ne marchent PAS ici** : dans
BoT-IoT, les flux DNS normaux comme les floods UDP ont **dbytes = dpkts = 0** (aucune réponse
capturée). Ces features ont une AUC de 0,500. À retenir pour le mémoire : ce n'est pas que le DNS
normal a une réponse et pas le flood — les deux sont unidirectionnels dans ce jeu.

## Point 2 — Features dérivées déployables (reproductibles par tout extracteur)

Toutes calculées à partir de grandeurs directement mesurables (bytes, pkts, spkts, dpkts, dur).

| Dérivée | médiane DNS | médiane UDP | AUC séparation |
|---|---|---|---|
| **`octets_par_paquet` = bytes / pkts** | **88** | **60** | **0,991** |
| `octets_par_paquet_aller` = sbytes / spkts | 88 | 60 | 0,991 (identique : dbytes=0) |
| `octets_par_seconde` = bytes / dur | 34,9 | 20,4 | 0,796 |
| `duree_par_paquet` = dur / pkts | 2,52 | 2,94 | 0,658 |
| `a_reponse`, `ratio_dbytes_sbytes` | 0 | 0 | 0,500 |

**`octets_par_paquet` sépare quasi parfaitement les deux groupes** (AUC 0,991, figure
`dns_octets_par_paquet.png`) : les attaques UDP sont serrées autour de 57 octets/paquet, les flux
DNS normaux autour de 88 et au-dessus. Un seuil vers 70 octets/paquet les sépare. C'est une
feature **comportementale et reproductible**, contrairement à hour_of_day et aux statistiques
Argus agrégées.

La combinaison linéaire des trois dérivées non dégénérées donne AUC 0,967 — moins bien que
`octets_par_paquet` seule : cette feature suffit, les autres ajoutent du bruit en linéaire.

### Proposition pour le point 3 (à valider)

Réentraîner le modèle fédéré BoT-IoT :
- **sans `hour_of_day`** ;
- **en ajoutant `octets_par_paquet = bytes / pkts`** (séparateur principal, AUC 0,991) ;
- optionnellement **`octets_par_seconde = bytes / dur`** (complément faible mais déployable).

Même config (7 clients, FedProx µ=0,1, 100 rounds, 2 époques, class weights), nouveau nom
`final_model_botiot_behav.pt`. Seuil choisi sur la validation (FPR ≤ 5 %). Comparaison au modèle
actuel (seuil 0,167) : rappel, FPR, AUC, balanced accuracy, rappel par type, top-10 SHAP.

**Réserve honnête à inscrire au mémoire** : `octets_par_paquet` sépare le DNS normal des floods
UDP *de BoT-IoT*. Un attaquant peut choisir la taille de ses paquets ; cette feature n'est donc
pas une parade absolue, mais elle supprime la dépendance à un artefact (l'heure) au profit d'une
grandeur comportementale mesurable partout. Sa généralisation réelle relève de la priorité 3 (labo).

**Estimation de temps du réentraînement** : ~75–90 min sur ce CPU (100 rounds, 7 clients ;
référence : le modèle sans hour_of_day a pris ~75 min hors charge concurrente). > 30 min → en
attente de votre accord.
