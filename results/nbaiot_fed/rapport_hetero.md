# N-BaIoT — hétérogénéité : FedAvg vs FedProx (étape 3)

Entraînement sur le NORMAL uniquement ; seuils p95/p99/p99.9 sur le normal de validation ; aucune décision fondée sur les attaques. Variantes a–d : **scaler local** par client. μ et nb d'époques de fine-tuning choisis sur la loss du normal de validation.

## 1. Variantes a–d (global, test)

| Variante | AUC | p95 rap/FPR | p99 rap/FPR | Mirai p99 | BASHLITE p99 |
|---|--:|--:|--:|--:|--:|
| a) FedAvg | 0.8314 | 76.7/3.6 | 66.1/0.9 | 89.2 | 49.3 |
| b) FedProx (μ*) | 0.9778 | 78.3/4.6 | 73.3/0.8 | 99.7 | 54.1 |
| c) FedAvg + fine-tuning local | 0.9880 | 91.4/3.3 | 76.8/0.4 | 100.0 | 60.0 |
| d) FedProx + fine-tuning local | 0.9998 | 100.0/3.2 | 100.0/0.5 | 100.0 | 100.0 |

**μ\*** = 0.01 (choisi sur la loss normal-val ; μ=0.01:0.02320, μ=0.1:0.02597, μ=1.0:0.15911).
Fine-tuning local : 4–5 époques par appareil (early-stopping sur la loss normal-val).

## 2. AUC par appareil (effet de FedProx et du fine-tuning)

| Appareil | FedAvg | FedProx | FedAvg+FT | FedProx+FT |
|---|--:|--:|--:|--:|
| Danmini_Doorbell | 0.800 | 1.000 | 0.999 | 1.000 |
| Ecobee_Thermostat | 0.751 | 0.996 | 0.975 | 0.999 |
| Ennio_Doorbell | 0.700 | 0.996 | 0.964 | 1.000 |
| Philips_B120N10_Baby_Monitor | 0.838 | 0.998 | 0.998 | 1.000 |
| Provision_PT_737E_Security_Camera | 0.869 | 0.978 | 0.990 | 1.000 |
| Provision_PT_838_Security_Camera | 0.840 | 0.986 | 0.942 | 1.000 |
| Samsung_SNH_1011_N_Webcam | 0.600 | 0.827 | 0.997 | 1.000 |
| SimpleHome_XCS7_1002_WHT_Security_Camera | 0.781 | 0.881 | 0.996 | 1.000 |
| SimpleHome_XCS7_1003_WHT_Security_Camera | 0.800 | 0.999 | 1.000 | 1.000 |

## 3. Scénario « peu de données » (plafond normal-train par appareil)

Idée : avec peu de normal par appareil, un AE local seul dispose de peu d'exemples ; le fédéré + fine-tuning met en commun l'apprentissage. **Nuance honnête** : sur N-BaIoT, le normal est très régulier, donc un AE local atteint déjà une bonne AUC avec 1 000 lignes ; l'apport du fédéré se voit surtout sur le **rappel à FPR fixe** et pour BASHLITE, pas tant sur l'AUC.

| Plafond | Modèle | AUC | p99 rap/FPR | Mirai | BASHLITE |
|---|---|--:|--:|--:|--:|
| 1000 | AE local seul | 0.9907 | 64.2/0.7 | 92.3 | 44.1 |
| 1000 | FedAvg | 0.9474 | 59.2/0.6 | 80.3 | 44.1 |
| 1000 | FedAvg + FT | 0.9763 | 67.9/0.4 | 92.6 | 50.1 |
| 1000 | FedProx + FT | 0.9772 | 67.6/0.4 | 92.1 | 50.0 |
| 5000 | AE local seul | 0.9994 | 97.3/0.3 | 100.0 | 95.4 |
| 5000 | FedAvg | 0.9899 | 69.9/0.5 | 97.0 | 50.5 |
| 5000 | FedAvg + FT | 0.9994 | 98.8/0.3 | 98.7 | 98.9 |
| 5000 | FedProx + FT | 0.9936 | 70.3/0.7 | 98.2 | 50.3 |

## 4. IID vs non-IID — ablation contrôlée (non déployable)

⚠️ **Ablation contrôlée, non déployable** : cette expérience utilise un **scaler global** (ajusté sur le normal de tous les clients réuni) dans les DEUX bras, uniquement pour isoler l'effet de la répartition des données. Les variantes a–d ci-dessus gardent le scaler local (méthode déployable). Ici, seule change la répartition : par appareil (non-IID) vs normal mélangé et redistribué au hasard (IID).

| Répartition | AUC | p95 rap/FPR | p99 rap/FPR | Mirai | BASHLITE |
|---|--:|--:|--:|--:|--:|
| non-IID (par appareil) | 0.8346 | 76.8/3.4 | 76.8/0.6 | 100.0 | 60.0 |
| IID (mélangé) | 0.9998 | 100.0/2.3 | 100.0/0.2 | 100.0 | 100.0 |

Écart d'AUC IID − non-IID : **+0.1652**. Un écart positif net confirme que l'hétérogénéité (non-IID) dégrade le modèle fédéré, toutes choses égales par ailleurs.

## 5. Ressources (identiques pour FedAvg/FedProx — même modèle)

- 36 377 paramètres (~142.1 Ko/màj) ; ~2.5 Mo échangés/round ; FedProx ajoute un terme proximal (coût CPU local  0 octet de plus).

## 6. Lecture

- FedAvg 0.831 → FedProx 0.978 : effet de μ sur l'AUC globale (+0.146).
- Fine-tuning local : FedAvg+FT 0.988, FedProx+FT 1.000 (personnalisation par appareil ; gain +0.168 vs FedAvg).
- Voir §2 pour l'effet par appareil (notamment les appareils faibles comme Samsung), §3 pour l'intérêt du fédéré à faible volume, §4 pour l'effet isolé de l'hétérogénéité.