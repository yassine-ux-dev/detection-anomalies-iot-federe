"""Étape 3 (robustesse) — 5 graines pour a,b,c,d + AE local équitable + AE centralisé.

Mêmes données (cap 50000, scaler LOCAL, attaques d'évaluation FIXÉES à seed 42), mêmes seuils
(p95/p99/p99.9 sur normal-val), aucun réglage fondé sur les attaques. La graine fait varier
l'initialisation et l'ordre des lots. μ fixé à 0.01 (valeur choisie sur val à l'étape 3 ;
non re-sélectionnée ici pour ne pas multiplier les runs). Données chargées UNE fois.

« AE local équitable » : AE par appareil, init aléatoire, entraîné jusqu'à FT_BUDGET époques avec
early-stopping sur SON normal-val — même mécanisme d'arrêt que le fine-tuning, pour comparer d
(FedProx+FT) à un local qui a eu autant d'occasions d'apprendre.

Sortie : results/nbaiot_fed/seeds_summary.json (+ .md). Un entraînement à la fois.

    python nbaiot_seeds.py --seeds 5 --rounds 20
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import nbaiot_data as D
from nbaiot_ae import AE, recon_error
from nbaiot_hetero import OUT, EPOCHS, finetune, eval_models, load_scaled, train_fed, train_client

MU = 0.01
FT_BUDGET = 45   # budget d'époques pour le local équitable (>= fed local + fine-tuning)


def local_fair(data, devs, seed):
    """AE par appareil, init aléatoire (graine), early-stopping sur normal-val, budget FT_BUDGET."""
    torch.manual_seed(seed)
    rand = {k: v.clone() for k, v in AE().state_dict().items()}
    models, _ = finetune(rand, data, devs, max_epochs=FT_BUDGET)
    return models


def local_std(data, devs, seed, rounds, epochs):
    """AE local « standard » (comme l'étape 1) : par appareil, init aléatoire, rounds*epochs
    époques FIXES sur son normal-train, sans early-stopping."""
    models = {}
    for d in devs:
        torch.manual_seed(seed)
        m = AE()
        train_client(m, data[d]["xtr"], epochs=rounds * epochs, mu=0.0)
        models[d] = m
    return models


def central(data, devs, seed, rounds, epochs):
    """AE centralisé : normal de tous réuni (déjà scalé localement), budget équivalent au fédéré."""
    torch.manual_seed(seed)
    m = AE()
    xall = np.vstack([data[d]["xtr"] for d in devs])
    train_client(m, xall, epochs=rounds * epochs, mu=0.0)   # ~ même total d'époques que le fédéré
    return {d: m for d in devs}


def g(metrics, key):
    gg = metrics["global"]
    if key == "auc":
        return gg["auc"]
    p, scope, field = key
    return gg[p][scope][field]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=20)
    a = ap.parse_args()
    t0 = time.time()
    print(f"[seeds] chargement unique (cap 50000, scaler local, attaques fixées seed 42)…", flush=True)
    data, _, devs = load_scaled(cap=50000, scaler_mode="local")   # attaques échantillonnées à seed 42
    clients = [data[d]["xtr"] for d in devs]
    vals = [data[d]["xva"] for d in devs]
    seeds = [42 + 7 * i for i in range(a.seeds)]

    MODELS = ["local_std", "local_fair", "central", "fedavg", "fedprox", "fedavg_ft", "fedprox_ft"]
    METRICS = {"AUC": "auc", "rappel_p99": ("p99", "global", "rappel"), "FPR_p99": ("p99", "global", "fpr"),
               "Mirai_p99": ("p99", "global", "rappel_mirai"), "BASHLITE_p99": ("p99", "global", "rappel_bashlite")}
    acc = {m: {k: [] for k in METRICS} for m in MODELS}

    for si, seed in enumerate(seeds):
        ts = time.time()
        print(f"\n=== graine {si+1}/{len(seeds)} (seed={seed}) ===", flush=True)
        res = {}
        res["local_std"] = eval_models(local_std(data, devs, seed, a.rounds, EPOCHS), data, devs)
        res["local_fair"] = eval_models(local_fair(data, devs, seed), data, devs)
        res["central"] = eval_models(central(data, devs, seed, a.rounds, EPOCHS), data, devs)
        avg_sd, _ = train_fed(clients, vals, a.rounds, EPOCHS, 0.0, "FedAvg", seed=seed)
        res["fedavg"] = eval_models({d: _l(avg_sd) for d in devs}, data, devs)
        prox_sd, _ = train_fed(clients, vals, a.rounds, EPOCHS, MU, f"FedProx μ={MU}", seed=seed)
        res["fedprox"] = eval_models({d: _l(prox_sd) for d in devs}, data, devs)
        res["fedavg_ft"] = eval_models(finetune(avg_sd, data, devs)[0], data, devs)
        res["fedprox_ft"] = eval_models(finetune(prox_sd, data, devs)[0], data, devs)
        for m in MODELS:
            for k, key in METRICS.items():
                acc[m][k].append(g(res[m], key))
        print("  " + " | ".join(f"{m} AUC={acc[m]['AUC'][-1]:.4f}" for m in MODELS)
              + f"  ({time.time()-ts:.0f}s)", flush=True)

    summary = {"seeds": seeds, "mu": MU, "rounds": a.rounds, "ft_budget_local": FT_BUDGET,
               "models": {}}
    for m in MODELS:
        summary["models"][m] = {k: dict(moyenne=float(np.mean(v)), ecart_type=float(np.std(v)),
                                        valeurs=[float(x) for x in v]) for k, v in acc[m].items()}
    summary["temps_min"] = round((time.time() - t0) / 60, 1)
    (OUT / "seeds_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # comparaisons appariées (par graine) : l'écart dépasse-t-il la variabilité entre graines ?
    def paired(ma, mb, key="AUC"):
        da = np.array(acc[ma][key]) - np.array(acc[mb][key])
        md_, sd_ = float(da.mean()), float(da.std())
        verdict = ("oui, net (>2σ)" if abs(md_) > 2 * sd_ + 1e-9 else
                   "oui (>σ)" if abs(md_) > sd_ + 1e-9 else "non (dans la variabilité)")
        return dict(ecart_moyen=md_, ecart_type_des_diff=sd_, significatif=verdict,
                    diffs=[float(x) for x in da])
    summary["comparaisons"] = {
        "FedProx+FT vs AE local équitable (AUC)": paired("fedprox_ft", "local_fair"),
        "FedProx+FT vs AE local équitable (BASHLITE_p99)": paired("fedprox_ft", "local_fair", "BASHLITE_p99"),
        "FedProx vs FedAvg (AUC)": paired("fedprox", "fedavg"),
        "FedAvg+FT vs FedAvg (AUC)": paired("fedavg_ft", "fedavg"),
    }
    (OUT / "seeds_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # markdown
    lab = {"local_std": "AE local (standard)", "local_fair": "AE local équitable", "central": "AE centralisé",
           "fedavg": "FedAvg", "fedprox": "FedProx (μ=0.01)", "fedavg_ft": "FedAvg + FT",
           "fedprox_ft": "FedProx + FT"}
    md = [f"# N-BaIoT — robustesse sur {len(seeds)} graines (moyenne ± écart-type)", "",
          f"Mêmes données/seuils ; graines {seeds} (init + ordre des lots). μ={MU}, {a.rounds} rounds. "
          "Aucun réglage sur les attaques.", "",
          "| Modèle | AUC | rappel p99 | FPR p99 | Mirai p99 | BASHLITE p99 |", "|---|--:|--:|--:|--:|--:|"]
    for m in MODELS:
        s = summary["models"][m]
        def c(k, p=False):
            mu_, sd_ = s[k]["moyenne"], s[k]["ecart_type"]
            return f"{100*mu_:.1f} ± {100*sd_:.1f}" if p or k != "AUC" else f"{mu_:.4f} ± {sd_:.4f}"
        md.append(f"| {lab[m]} | {s['AUC']['moyenne']:.4f} ± {s['AUC']['ecart_type']:.4f} | "
                  f"{c('rappel_p99',1)} | {c('FPR_p99',1)} | {c('Mirai_p99',1)} | {c('BASHLITE_p99',1)} |")
    md += ["", "Rappel/FPR/Mirai/BASHLITE en % au seuil p99 global. « AE local (standard) » = époques "
           f"fixes sans early-stopping (comme l'étape 1) ; « AE local équitable » = jusqu'à {FT_BUDGET} "
           "époques avec early-stopping sur normal-val (même budget que fédéré+fine-tuning).", "",
           "## Comparaisons appariées (l'écart dépasse-t-il la variabilité entre graines ?)", "",
           "| Comparaison | écart moyen | σ des écarts | significatif ? |", "|---|--:|--:|---|"]
    for name, cmp in summary["comparaisons"].items():
        md.append(f"| {name} | {cmp['ecart_moyen']:+.4f} | {cmp['ecart_type_des_diff']:.4f} | "
                  f"{cmp['significatif']} |")
    md += ["", "Écart apparié (même graine des deux côtés). « significatif » : >σ ou >2σ des écarts "
           "entre graines ; « non » = l'écart tient dans la variabilité (donc non concluant)."]
    (OUT / "seeds_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n" + "\n".join(md))


def _l(sd):
    m = AE(); m.load_state_dict(sd); return m


if __name__ == "__main__":
    main()
