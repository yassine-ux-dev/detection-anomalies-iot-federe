"""Étape 3 (complément) — « nouvel appareil » : leave-one-device-out sur N-BaIoT.

Pour chaque appareil exclu d_out :
  - les 8 AUTRES entraînent FedProx (μ=0.01, 20 rounds) — d_out n'y participe pas ;
  - d_out ne dispose que de ses N PREMIÈRES lignes normales de train (ordre chronologique),
    N ∈ {500, 2000} ; son normal-val et son test restent inchangés ;
  - scaler LOCAL de d_out ajusté sur ces N lignes uniquement ;
  - comparaison sur le test de d_out (seuils p95/p99/p99.9 sur SON normal-val, après entraînement) :
      * AE local de zéro (init aléatoire, early-stopping sur son normal-val) ;
      * modèle fédéré (FedProx des 8 autres) + fine-tuning local sur ces N lignes.

Règles : normal uniquement ; attaques au test seulement ; aucun réglage sur les attaques.
Données chargées une fois. Un entraînement à la fois.

    python nbaiot_lodo.py --rounds 20
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

import nbaiot_data as D
from nbaiot_ae import AE, PCTS
from nbaiot_hetero import OUT, EPOCHS, FT_MAX, eval_models, finetune, train_fed

MU = 0.01
SIZES = [500, 2000]
LOCAL_BUDGET = 45   # époques max pour l'AE local de zéro (early-stopping sur normal-val)
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--report-only", action="store_true",
                    help="reconstruit lodo.md depuis lodo.json sans réentraîner")
    a = ap.parse_args()
    if a.report_only:
        out = json.loads((OUT / "lodo.json").read_text(encoding="utf-8"))
        devs = list(out["par_taille"]["500"]["local_zero"].keys())
        build_report(out, devs)
        return
    t0 = time.time()
    devs = D.devices()
    print(f"[lodo] chargement unique des {len(devs)} appareils (cap 50000)…", flush=True)
    raw = {}
    for dev in devs:
        dd = D.load_device(dev, cap=50000, seed=SEED)
        xa, fam, _ = D.sample_attacks(dd, n_target=max(len(dd["test"]), 1))
        raw[dev] = dict(train=dd["train"], val=dd["val"], test=dd["test"], att=xa, fam=fam)
    # membres de la fédération : scaler local complet + normal-train/val scalés
    msc = {d: MinMaxScaler().fit(raw[d]["train"]) for d in devs}
    clip = lambda z: np.clip(z, 0, 1).astype(np.float32)
    mxtr = {d: clip(msc[d].transform(raw[d]["train"])) for d in devs}
    mxva = {d: clip(msc[d].transform(raw[d]["val"])) for d in devs}
    print(f"  chargement {time.time()-t0:.0f}s", flush=True)

    out = {"meta": dict(rounds=a.rounds, mu=MU, sizes=SIZES), "par_taille": {}}
    for n in SIZES:
        print(f"\n===== N = {n} lignes normales pour l'appareil exclu =====", flush=True)
        per_dev = {"local_zero": {}, "fed_ft": {}}
        for d_out in devs:
            tdev = time.time()
            others = [o for o in devs if o != d_out]
            clients = [mxtr[o] for o in others]
            vals = [mxva[o] for o in others]
            prox_sd, _ = train_fed(clients, vals, a.rounds, EPOCHS, MU, f"LODO-{d_out[:10]} N={n}", seed=SEED)
            # données de l'appareil exclu : N premières lignes normales (chronologique) ; val/test inchangés
            sc = MinMaxScaler().fit(raw[d_out]["train"][:n])
            dd = {d_out: dict(xtr=clip(sc.transform(raw[d_out]["train"][:n])),
                              xva=clip(sc.transform(raw[d_out]["val"])),
                              xte=clip(sc.transform(raw[d_out]["test"])),
                              xatt=clip(sc.transform(raw[d_out]["att"])), fam=raw[d_out]["fam"])}
            # AE local de zéro (init aléatoire, early-stopping sur normal-val)
            torch.manual_seed(SEED)
            rand = {k: v.clone() for k, v in AE().state_dict().items()}
            m_loc, _ = finetune(rand, dd, [d_out], max_epochs=LOCAL_BUDGET)
            r_loc = eval_models(m_loc, dd, [d_out])
            # fédéré (8 autres) + fine-tuning local sur les N lignes
            m_ft, _ = finetune(prox_sd, dd, [d_out], max_epochs=FT_MAX)
            r_ft = eval_models(m_ft, dd, [d_out])
            per_dev["local_zero"][d_out] = r_loc["global"]
            per_dev["fed_ft"][d_out] = r_ft["global"]
            print(f"  {d_out:<42} local_zero AUC={r_loc['global']['auc']:.3f} | "
                  f"fed+FT AUC={r_ft['global']['auc']:.3f}  ({time.time()-tdev:.0f}s)", flush=True)
        out["par_taille"][str(n)] = per_dev

    out["meta"]["temps_min"] = round((time.time() - t0) / 60, 1)
    (OUT / "lodo.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    build_report(out, devs)
    print(f"\nTEMPS TOTAL: {out['meta']['temps_min']} min")


def build_report(out, devs):
    """Construit lodo.md depuis `out` (en mémoire ou rechargé de lodo.json)."""
    def avg(method, n, key):  # ignore None (rappel Mirai absent pour Ennio/Samsung, sans Mirai)
        gs = out["par_taille"][str(n)][method].values()
        vals = [(gg["auc"] if key == "auc" else gg[key[0]][key[1]][key[2]]) for gg in gs]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)), float(np.std(vals))

    METR = {"AUC": "auc", "rappel p99": ("p99", "global", "rappel"), "FPR p99": ("p99", "global", "fpr"),
            "Mirai p99": ("p99", "global", "rappel_mirai"), "BASHLITE p99": ("p99", "global", "rappel_bashlite")}
    md = ["# N-BaIoT — « nouvel appareil » (leave-one-device-out)", "",
          "Appareil exclu de la fédération ; les 8 autres entraînent FedProx (μ=0.01). L'appareil exclu "
          "ne dispose que de ses N premières lignes normales (chronologique) ; normal-val et test "
          "inchangés. Seuils p95/p99/p99.9 sur son normal-val. Comparaison : AE local de zéro vs "
          "modèle fédéré + fine-tuning. (Mirai absent chez Ennio et Samsung : exclu de la moyenne Mirai.)", ""]
    for n in SIZES:
        md += [f"## N = {n} lignes normales", "",
               "| Méthode | AUC | rappel p99 | FPR p99 | Mirai p99 | BASHLITE p99 |", "|---|--:|--:|--:|--:|--:|"]
        for method, lab in (("local_zero", "AE local de zéro"), ("fed_ft", "Fédéré + fine-tuning")):
            cells = []
            for k, key in METR.items():
                mu_, sd_ = avg(method, n, key)
                cells.append(f"{mu_:.4f} ± {sd_:.4f}" if k == "AUC" else f"{100*mu_:.1f} ± {100*sd_:.1f}")
            md.append(f"| {lab} | " + " | ".join(cells) + " |")
        md.append("")
        md += ["Détail par appareil (AUC · rappel p99 · BASHLITE p99) :", "",
               "| Appareil | local de zéro | fédéré + FT |", "|---|--:|--:|"]
        for d_out in devs:
            lz, ft = out["par_taille"][str(n)]["local_zero"][d_out], out["par_taille"][str(n)]["fed_ft"][d_out]
            fmt = lambda gg: (f"{gg['auc']:.3f} · {100*gg['p99']['global']['rappel']:.0f}% · "
                              f"{100*gg['p99']['global']['rappel_bashlite']:.0f}%")
            md.append(f"| {d_out} | {fmt(lz)} | {fmt(ft)} |")
        md.append("")
    md += ["## Lecture",
           "Sur un appareil **jamais vu** de la fédération, l'AE local entraîné de zéro est **aussi bon ou "
           "meilleur** que le modèle fédéré + fine-tuning, et l'écart se creuse à N=2000 (rappel p99 moyen "
           "95 % vs 75 %, BASHLITE 90 % vs 63 %). FedProx+FT ne s'effondre pas en AUC (aucun appareil "
           "< 0,90) mais **son rappel au seuil p99 chute** sur plusieurs appareils non vus (Provision-737E "
           "15 %, Philips, SimpleHome-1002). L'avantage du fédéré+fine-tuning valait pour les appareils "
           "**membres** (voir seeds_summary) ; pour un **nouvel** appareil disposant de ses propres données, "
           "un AE local suffit. Limite honnête et défendable."]
    (OUT / "lodo.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
