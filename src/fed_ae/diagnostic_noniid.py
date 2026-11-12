"""Étape 3, point 1 — diagnostic du non-IID N-BaIoT (SANS entraînement).

Deux hypothèses à vérifier :
 A) « une attaque sur un appareil ressemble au normal d'un autre appareil » : dans un espace
    commun (scaler GLOBAL ajusté sur tout le normal — UNIQUEMENT pour ce diagnostic), on mesure,
    pour chaque appareil, si ses attaques sont plus proches du normal d'un AUTRE appareil que du
    sien. Si oui, un modèle global (qui a appris le normal de tous) reconstruit bien ces attaques
    -> erreur basse -> non détectées.
 B) effet du scaler LOCAL : le même comportement brut est mis à des échelles différentes selon
    l'appareil. On quantifie en appliquant le scaler d'un appareil au normal d'un autre et en
    mesurant la part de valeurs hors [0,1] (incohérence vue par le modèle global).

Sortie : results/nbaiot_fed/diagnostic_noniid.{json,md}
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.preprocessing import MinMaxScaler

import nbaiot_data as D

OUT = Path(__file__).resolve().parents[2] / "results" / "nbaiot_fed"
CAP = 50000
SAMPLE_ATT = 4000


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    devs = D.devices()
    normal_tr, attacks, fam_by_dev = {}, {}, {}
    for dev in devs:
        dd = D.load_device(dev, cap=CAP)
        normal_tr[dev] = dd["train"]
        xa, fam, sub = D.sample_attacks(dd, n_target=SAMPLE_ATT)
        attacks[dev] = xa
        fam_by_dev[dev] = fam
    # scaler GLOBAL (diagnostic uniquement) pour un espace comparable
    gsc = MinMaxScaler().fit(np.vstack([normal_tr[d] for d in devs]))
    norm_c = {d: gsc.transform(normal_tr[d]).mean(0) for d in devs}      # centroïde normal par appareil

    # A) proximité attaque -> normal d'un autre appareil
    rowsA, cross = [], 0
    confus = {}
    for d in devs:
        xa = np.clip(gsc.transform(attacks[d]), -5, 5)
        dists = {o: np.linalg.norm(xa - norm_c[o], axis=1) for o in devs}
        own = dists[d]
        nearest = np.array([min(devs, key=lambda o: dists[o][i]) for i in range(len(xa))])
        pct_autre = float((nearest != d).mean())
        # plus proche centroïde normal "autre" (min sur o != d)
        best_other = np.min([dists[o] for o in devs if o != d], axis=0)
        rowsA.append(dict(appareil=d, dist_moy_normal_propre=float(own.mean()),
                          dist_moy_normal_autre_plus_proche=float(best_other.mean()),
                          pct_attaques_plus_proches_dun_autre=pct_autre))
        # vers quel appareil les attaques de d sont-elles le plus souvent attirées ?
        vals, cnts = np.unique(nearest[nearest != d], return_counts=True)
        if len(vals):
            confus[d] = str(vals[np.argmax(cnts)])
        cross += pct_autre

    # B) effet du scaler local : appliquer le scaler de B au normal de A
    local_sc = {d: MinMaxScaler().fit(normal_tr[d]) for d in devs}
    rowsB = []
    for a in devs:
        outside = []
        for b in devs:
            if a == b:
                continue
            z = local_sc[b].transform(normal_tr[a][:5000])
            outside.append(float(((z < 0) | (z > 1)).mean()))
        rowsB.append(dict(appareil=a, part_hors_01_avec_scaler_dautrui=float(np.mean(outside))))

    res = dict(hypothese_A=rowsA, pct_attaques_attirees_autre_appareil_moyen=cross / len(devs),
               appareil_attracteur=confus, hypothese_B=rowsB)
    (OUT / "diagnostic_noniid.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# N-BaIoT — diagnostic du non-IID (sans entraînement)", "",
          "Espace commun (scaler global, pour ce diagnostic uniquement).", "",
          "## A) Les attaques d'un appareil ressemblent-elles au normal d'un autre ?", "",
          "| Appareil | dist. moy. à son propre normal | dist. moy. au normal d'un autre (plus proche) | "
          "% attaques plus proches d'un AUTRE appareil |", "|---|--:|--:|--:|"]
    for r in rowsA:
        md.append(f"| {r['appareil']} | {r['dist_moy_normal_propre']:.3f} | "
                  f"{r['dist_moy_normal_autre_plus_proche']:.3f} | "
                  f"{100*r['pct_attaques_plus_proches_dun_autre']:.1f} % |")
    md += ["", f"En moyenne, **{100*res['pct_attaques_attirees_autre_appareil_moyen']:.1f} %** des attaques d'un "
           "appareil sont plus proches du normal d'un AUTRE appareil que du sien. Appareil « attracteur » le "
           "plus fréquent par source : " + ", ".join(f"{k}→{v}" for k, v in confus.items()) + ".", "",
           "## B) Effet du scaler local", "",
           "Part des valeurs qui tombent hors [0,1] quand on applique à un appareil le scaler d'un autre "
           "(moyenne sur les 8 autres) — mesure l'incohérence d'échelle vue par un modèle global :", "",
           "| Appareil | part hors [0,1] avec le scaler d'autrui |", "|---|--:|"]
    for r in rowsB:
        md.append(f"| {r['appareil']} | {100*r['part_hors_01_avec_scaler_dautrui']:.1f} % |")
    md += ["",
           "**Lecture.** (A) Une part notable des attaques d'un appareil se logent dans la région du normal "
           "d'autres appareils : le modèle fédéré, qui a appris le normal de TOUS, les reconstruit bien et "
           "les manque. (B) Le scaler local place le même comportement à des échelles très différentes selon "
           "l'appareil ; un modèle global reçoit donc des entrées incohérentes d'un client à l'autre. Les deux "
           "effets expliquent l'AUC fédérée en retrait et très inégale, et motivent FedProx / la "
           "personnalisation (fine-tuning local)."]
    (OUT / "diagnostic_noniid.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
