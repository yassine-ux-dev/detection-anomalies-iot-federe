# `src/preprocessing/` — Préparation des données

Transforme les jeux bruts en un format commun (un CSV par source, colonne d'étiquette binaire).

| Fichier | Rôle |
|---|---|
| `harmonize_nbaiot.py` | Réunit, par appareil, le trafic bénin et les attaques (Mirai, BASHLITE) de N-BaIoT. |
| `harmonize_botiot.py` | Lit les fichiers bruts de BoT-IoT, sous-échantillonne, retire les identifiants (IP/MAC/ports). |

Emplacement des données via la variable `IOT_DATA` (voir [`../../data/README.md`](../../data/README.md)).
Détails : [`../../docs/EXPLICATION_DU_CODE.md`](../../docs/EXPLICATION_DU_CODE.md).
