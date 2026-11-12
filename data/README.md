# Données

Les jeux de données ne sont **pas** inclus (volumineux : ~32 Go bruts). Ils sont publics et
doivent être téléchargés puis placés dans un dossier pointé par la variable d'environnement
`IOT_DATA` (par défaut : ce dossier `data/`).

## 1. Télécharger les jeux de données

| Jeu | Source | Contenu utilisé |
|---|---|---|
| **N-BaIoT** | UCI — *Detection of IoT Botnet Attacks N-BaIoT* : <https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot> | trafic bénin + attaques (Mirai, BASHLITE), 9 appareils |
| **TON-IoT** | UNSW Canberra — *TON_IoT datasets* (partie **télémétrie**) : <https://research.unsw.edu.au/projects/toniot-datasets> | 7 services IoT/IIoT |
| **BoT-IoT** | UNSW Canberra — *BoT-IoT dataset* : <https://research.unsw.edu.au/projects/bot-iot-dataset> | flux réseau (Argus) |

## 2. Arborescence attendue sous `IOT_DATA`

```
IOT_DATA/
├── Raw_NBaIoT_dataset/            # brut UCI : un dossier par appareil
│   └── <Appareil>/
│       ├── benign_traffic.csv
│       ├── gafgyt_attacks/*.csv
│       └── mirai_attacks/*.csv
├── Raw_BoT_IoT_dataset/           # brut BoT-IoT : data_1.csv ... data_75.csv
├── Processed_IoT_dataset/         # TON-IoT télémétrie : IoT_*.csv (un par service)
├── Processed_NBaIoT_dataset/      # sortie de harmonize_nbaiot.py : NBaIoT_*.csv
└── Processed_BoT_IoT_dataset/     # sortie de harmonize_botiot.py : BoT_*.csv
```

- L'**auto-encodeur** (méthode principale, N-BaIoT) lit directement `Raw_NBaIoT_dataset/`.
- Le **classifieur de référence** (Flower) lit les dossiers `Processed_*` (CSV harmonisés).
- TON-IoT : utiliser les fichiers de **télémétrie** prétraités fournis par les auteurs, renommés
  `IoT_<service>.csv` dans `Processed_IoT_dataset/`.

## 3. Harmoniser (N-BaIoT et BoT-IoT)

Produit les dossiers `Processed_NBaIoT_dataset/` et `Processed_BoT_IoT_dataset/` à partir des bruts :

```bash
export IOT_DATA=/chemin/vers/data        # Windows PowerShell : $env:IOT_DATA="C:\chemin\data"
python src/preprocessing/harmonize_nbaiot.py
python src/preprocessing/harmonize_botiot.py
```

## 4. Indiquer l'emplacement des données

- Scripts Python (auto-encodeur, prétraitement, fp_reduction) : variable `IOT_DATA`.
- Application Flower (classifieur) : clé `data-dir` des fichiers `src/classifieur/run_config_*.toml`
  (chemin relatif `../../data/Processed_<jeu>_dataset`, à adapter si besoin).
