# BoT-IoT — modèles sans heure vs modèle actuel (test propre)

Protocole propre (test 101 115 flux, 955 normaux). Seuils choisis sur la validation (FPR ≤ 5 %) ; test unique. Modèle recommandé sur la validation : **deploy**.

| Variante | Feat. | Seuil | F1-macro | AUC | MCC | Bal. acc. | Rappel % | FPR % [IC95] | DDoS % | DoS % | Scan % | Vol de données % | Keylogging % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Origine (0,167) | 51 | 0.167 | 0.711 | 0.9962 | 0.511 | 0.966 | 97.7 | 4.5 [3.4–6.0] | 100.0 | 94.6 | 99.2 | 36.4 | 94.7 |
| Sans heure + oct/paquet (behav) | 51 | 0.601 | 0.621 | 0.9960 | 0.375 | 0.952 | 95.1 | 4.6 [3.4–6.1] | 93.4 | 94.6 | 99.3 | 100.0 | 98.0 |
| 100 % déployable (deploy) | 32 | 0.699 | 0.704 | 0.9891 | 0.502 | 0.969 | 97.5 | 3.7 [2.6–5.1] | 98.7 | 97.5 | 95.2 | 36.4 | 92.8 |

Effectifs test par type : DDoS 39922, DoS 39952, Scan 20123, Vol de données 11, Keylogging 152.

## Top-10 SHAP (validation)

- **behav** : mean (0.121), state_INT (0.108), max (0.073), min (0.070), state_RST (0.062), stddev (0.061), proto_tcp (0.050), state_REQ (0.032), proto_udp (0.017), flgs_e_s (0.013)
- **deploy** : state_REQ (0.081), proto_udp (0.078), state_RST (0.075), proto_tcp (0.066), state_INT (0.012), state_CON (0.006), dur (0.005), octets_par_paquet (0.003), state_FIN (0.001), proto_icmp (0.000)
## Conclusion honnête

- **Supprimer l'heure seule ne suffit pas (behav).** Le modèle `behav` (sans `hour_of_day`,
  + `octets_par_paquet`, mais gardant les stats Argus) remplace simplement l'heure par les
  statistiques agrégées `mean`/`max`/`min`/`stddev` (SHAP) : il reste non déployable et son rappel
  baisse à 95,1 %. `octets_par_paquet` n'apparaît pas dans son top-10.
- **Le modèle 100 % déployable (deploy) est le bon compromis — et n'en est presque pas un.** En
  retirant l'heure ET toutes les features non reproductibles (stats Argus agrégées, drapeaux
  `flgs_*`), en gardant comptes, débits, protocole, état de connexion et `octets_par_paquet`, le
  modèle `deploy` obtient sur le test : rappel 97,5 %, FPR 3,7 %, balanced accuracy 0,969 — soit
  l'équivalent du modèle d'origine (97,7 % / 4,5 % / 0,966), avec une AUC à peine plus basse
  (0,989 vs 0,996). Son top-10 SHAP ne contient **aucune feature non reproductible**.
- **Objectif atteint.** On peut retirer la dépendance à un artefact (l'heure) ET aux features
  Argus non reproductibles sans perte de performance au point de fonctionnement. C'est un argument
  fort pour un déploiement réel : le modèle `deploy` est entraînable et utilisable avec le même
  extracteur partout. Réserve : la généralisation réelle reste à mesurer sur les captures du labo
  (priorité 3), et `octets_par_paquet` sépare le DNS des floods *de BoT-IoT* (un attaquant peut
  changer la taille de ses paquets).
- **Recommandation (sur la validation) : `deploy`.** Reste à confirmer par l'utilisateur avant de
  basculer la vidéo, `RESUME_FINAL.md` et le mémoire sur ce modèle.
