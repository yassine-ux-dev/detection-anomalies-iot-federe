# N-BaIoT — répartition non-IID (un client = un appareil)

Comptage des flux bruts par appareil (9 clients fédérés).

| Client (appareil) | Normal | BASHLITE (gafgyt) | Mirai | % attaques |
|---|--:|--:|--:|--:|
| Danmini_Doorbell | 49 548 | 316 650 | 652 100 | 95.1 |
| Ecobee_Thermostat | 13 113 | 310 630 | 512 133 | 98.4 |
| Ennio_Doorbell | 39 100 | 316 400 | **0** | 89.0 |
| Philips_B120N10_Baby_Monitor | 175 240 | 312 723 | 610 714 | 84.0 |
| Provision_PT_737E_Security_Camera | 62 154 | 330 096 | 436 010 | 92.5 |
| Provision_PT_838_Security_Camera | 98 514 | 309 040 | 429 337 | 88.2 |
| Samsung_SNH_1011_N_Webcam | 52 150 | 323 072 | **0** | 86.1 |
| SimpleHome_XCS7_1002_WHT_Security_Camera | 46 585 | 303 223 | 513 248 | 94.6 |
| SimpleHome_XCS7_1003_WHT_Security_Camera | 19 528 | 316 438 | 514 860 | 97.7 |
| **Total** | 555 932 | 2 838 272 | 3 668 402 | — |

**Caractère non-IID :**
- **Familles d'attaque absentes chez certains clients** : 2 appareils (Ennio_Doorbell, Samsung_SNH_1011_N_Webcam) n'ont **aucun flux Mirai** — ils ne voient que BASHLITE.
- **Déséquilibre de quantité** : le trafic normal va de 13 113 (Ecobee_Thermostat) à 175 240 (Philips_B120N10_Baby_Monitor) flux  soit un facteur 13.
- **Part d'attaques élevée partout** : 84.0 % à 98.4 %. L'entraînement auto-supervisé n'utilisera que le trafic NORMAL de chaque client.