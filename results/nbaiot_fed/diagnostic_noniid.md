# N-BaIoT — diagnostic du non-IID (sans entraînement)

Espace commun (scaler global, pour ce diagnostic uniquement).

## A) Les attaques d'un appareil ressemblent-elles au normal d'un autre ?

| Appareil | dist. moy. à son propre normal | dist. moy. au normal d'un autre (plus proche) | % attaques plus proches d'un AUTRE appareil |
|---|--:|--:|--:|
| Danmini_Doorbell | 10.127 | 10.020 | 100.0 % |
| Ecobee_Thermostat | 10.400 | 10.242 | 100.0 % |
| Ennio_Doorbell | 8.178 | 8.095 | 60.0 % |
| Philips_B120N10_Baby_Monitor | 10.147 | 10.137 | 85.9 % |
| Provision_PT_737E_Security_Camera | 9.455 | 9.383 | 100.0 % |
| Provision_PT_838_Security_Camera | 9.512 | 9.505 | 44.2 % |
| Samsung_SNH_1011_N_Webcam | 8.277 | 8.117 | 99.9 % |
| SimpleHome_XCS7_1002_WHT_Security_Camera | 10.349 | 10.269 | 100.0 % |
| SimpleHome_XCS7_1003_WHT_Security_Camera | 10.370 | 10.237 | 100.0 % |

En moyenne, **87.8 %** des attaques d'un appareil sont plus proches du normal d'un AUTRE appareil que du sien. Appareil « attracteur » le plus fréquent par source : Danmini_Doorbell→Provision_PT_838_Security_Camera, Ecobee_Thermostat→Provision_PT_838_Security_Camera, Ennio_Doorbell→Provision_PT_838_Security_Camera, Philips_B120N10_Baby_Monitor→Provision_PT_838_Security_Camera, Provision_PT_737E_Security_Camera→Provision_PT_838_Security_Camera, Provision_PT_838_Security_Camera→Philips_B120N10_Baby_Monitor, Samsung_SNH_1011_N_Webcam→Ennio_Doorbell, SimpleHome_XCS7_1002_WHT_Security_Camera→Provision_PT_838_Security_Camera, SimpleHome_XCS7_1003_WHT_Security_Camera→Provision_PT_838_Security_Camera.

## B) Effet du scaler local

Part des valeurs qui tombent hors [0,1] quand on applique à un appareil le scaler d'un autre (moyenne sur les 8 autres) — mesure l'incohérence d'échelle vue par un modèle global :

| Appareil | part hors [0,1] avec le scaler d'autrui |
|---|--:|
| Danmini_Doorbell | 0.1 % |
| Ecobee_Thermostat | 4.1 % |
| Ennio_Doorbell | 0.4 % |
| Philips_B120N10_Baby_Monitor | 7.8 % |
| Provision_PT_737E_Security_Camera | 4.8 % |
| Provision_PT_838_Security_Camera | 6.1 % |
| Samsung_SNH_1011_N_Webcam | 2.9 % |
| SimpleHome_XCS7_1002_WHT_Security_Camera | 4.6 % |
| SimpleHome_XCS7_1003_WHT_Security_Camera | 1.1 % |

**Lecture.** (A) Une part notable des attaques d'un appareil se logent dans la région du normal d'autres appareils : le modèle fédéré, qui a appris le normal de TOUS, les reconstruit bien et les manque. (B) Le scaler local place le même comportement à des échelles très différentes selon l'appareil ; un modèle global reçoit donc des entrées incohérentes d'un client à l'autre. Les deux effets expliquent l'AUC fédérée en retrait et très inégale, et motivent FedProx / la personnalisation (fine-tuning local).