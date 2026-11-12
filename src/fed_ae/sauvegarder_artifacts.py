"""Sauvegarde des artefacts de l'autoencodeur fédéré (méthode principale du mémoire).

Réutilise EXACTEMENT les fonctions qui ont produit les résultats du chapitre 4
(nbaiot_data.load_device, nbaiot_hetero.train_fed / finetune / eval_models), avec la
même configuration et la même graine, puis sauvegarde :
  - le modèle global FedProx  (AVANT fine-tuning)            -> global_fedprox.pt
  - les modèles locaux fine-tunés (un par appareil)          -> local/<appareil>.pt
  - le MinMax de chaque client                               -> scalers/<appareil>.pkl
  - les seuils p95 / p99 / p99,9 de chaque client + globaux  -> thresholds.json
  - la configuration (graine, hyperparamètres, features…)    -> config.json
  - les métriques + comparaison avec le chapitre 4           -> metrics.json

Dossiers : artifacts/nbaiot/ (et artifacts/toniot/, artifacts/botiot/ créés pour la suite).

    python sauvegarder_artifacts.py --dataset nbaiot
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

import nbaiot_data as D
from nbaiot_ae import AE, recon_error, PCTS
from nbaiot_hetero import EPOCHS, eval_models, finetune, train_fed

QS = Path(__file__).resolve().parents[2]          # projet/
ART = QS / "artifacts"
REF = QS / "results" / "nbaiot_fed"                  # résultats du chapitre 4 (référence)

# Configuration IDENTIQUE au chapitre 4 (tab:robustesse / hetero_variants, graine 42)
CFG = {"nbaiot": dict(cap=50000, mu=0.01, rounds=20, epochs=EPOCHS, seed=42, label="N-BaIoT", dir="nbaiot")}


def _mk(sd):
    m = AE(sd["net.0.weight"].shape[1]); m.load_state_dict(sd); return m


def feature_names_nbaiot() -> list[str]:
    f = D.RAW / D.devices()[0] / "benign_traffic.csv"
    with open(f) as fh:
        return next(csv.reader(fh))


def build_nbaiot(cap, seed):
    """Reproduit nbaiot_hetero.load_scaled (scaler LOCAL) en conservant les objets scaler."""
    devs = D.devices()
    clip = lambda z: np.clip(z, 0, 1).astype(np.float32)
    data, scalers = {}, {}
    for dev in devs:
        dd = D.load_device(dev, cap=cap, seed=seed)
        sc = MinMaxScaler().fit(dd["train"])
        xa, fam, _ = D.sample_attacks(dd, n_target=max(len(dd["test"]), 1), seed=seed)
        d_feat = dd["train"].shape[1]
        data[dev] = dict(
            xtr=clip(sc.transform(dd["train"])), xva=clip(sc.transform(dd["val"])),
            xte=clip(sc.transform(dd["test"])),
            xatt=clip(sc.transform(xa)) if len(xa) else np.empty((0, d_feat), np.float32),
            fam=fam)
        scalers[dev] = sc
    return devs, data, scalers


def layer_dims(sd):
    """Dimensions des couches linéaires (pour tracer l'architecture dans config.json)."""
    dims = []
    for k, v in sd.items():
        if k.endswith(".weight") and v.ndim == 2:
            dims.append([int(v.shape[1]), int(v.shape[0])])
    return dims


def charger_reference():
    """AUC du chapitre 4 (graine 42 : hetero_variants ; moyenne 5 graines : seeds_summary)."""
    ref = {}
    hv = REF / "hetero_variants.json"
    if hv.exists():
        d = json.loads(hv.read_text(encoding="utf-8"))
        ref["graine42"] = {"fedprox": d["b_fedprox"]["global"]["auc"],
                           "fedprox_ft": d["d_fedprox_ft"]["global"]["auc"],
                           "fedavg": d["a_fedavg"]["global"]["auc"]}
        ref["par_appareil_ft_42"] = {k: d["d_fedprox_ft"]["par_appareil"][k]["auc"]
                                     for k in d["d_fedprox_ft"]["par_appareil"]}
    ss = REF / "seeds_summary.json"
    if ss.exists():
        s = json.loads(ss.read_text(encoding="utf-8"))
        ref["moyenne5"] = {"fedprox": s["models"]["fedprox"]["AUC"]["moyenne"],
                           "fedprox_ft": s["models"]["fedprox_ft"]["AUC"]["moyenne"],
                           "fedavg": s["models"]["fedavg"]["AUC"]["moyenne"]}
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["nbaiot"], default="nbaiot",
                    help="N-BaIoT uniquement (TON-IoT/BoT-IoT : voir other_ae.py, dossiers créés).")
    a = ap.parse_args(); c = CFG[a.dataset]; t0 = time.time()

    for d in ("nbaiot", "toniot", "botiot"):
        (ART / d).mkdir(parents=True, exist_ok=True)
    outdir = ART / c["dir"]
    (outdir / "local").mkdir(exist_ok=True); (outdir / "scalers").mkdir(exist_ok=True)

    print(f"[{c['label']}] chargement + entraînement (graine {c['seed']}, cap {c['cap']}, "
          f"FedProx μ={c['mu']}, {c['rounds']} rounds × {c['epochs']} époques)…", flush=True)
    devs, data, scalers = build_nbaiot(c["cap"], c["seed"])
    feats = feature_names_nbaiot()
    clients = [data[d]["xtr"] for d in devs]
    vals = [data[d]["xva"] for d in devs]

    # 1) modèle global FedProx (AVANT fine-tuning)
    prox_sd, vh = train_fed(clients, vals, c["rounds"], c["epochs"], c["mu"], "FedProx", seed=c["seed"])
    # 2) modèles locaux fine-tunés (un par appareil)
    ft_models, ft_ep = finetune(prox_sd, data, devs)

    # 3) évaluation (même code que le chapitre 4)
    res_global = eval_models({d: _mk(prox_sd) for d in devs}, data, devs)   # FedProx global partout
    res_ft = eval_models(ft_models, data, devs)                            # FedProx + fine-tuning

    # 4) seuils par appareil (sur le NORMAL de validation, modèle fine-tuné) + global poolé
    thr = {"local": {}, "global": {}, "note": "percentiles de l'erreur de reconstruction du normal-val"}
    val_all = []
    for d in devs:
        ev = recon_error(ft_models[d], data[d]["xva"])
        thr["local"][d] = {str(p): float(np.percentile(ev, p)) for p in PCTS}
        val_all.append(ev)
    allev = np.concatenate(val_all)
    thr["global"] = {str(p): float(np.percentile(allev, p)) for p in PCTS}

    # 5) SAUVEGARDE des artefacts
    torch.save(prox_sd, outdir / "global_fedprox.pt")
    for d in devs:
        torch.save(ft_models[d].state_dict(), outdir / "local" / f"{d}.pt")
        with open(outdir / "scalers" / f"{d}.pkl", "wb") as fh:
            pickle.dump(scalers[d], fh)
    (outdir / "thresholds.json").write_text(json.dumps(thr, indent=2, ensure_ascii=False), encoding="utf-8")
    config = dict(
        dataset="N-BaIoT", seed=c["seed"], cap_normal_train=c["cap"], aggregation="FedProx", mu=c["mu"],
        rounds=c["rounds"], local_epochs=c["epochs"], fine_tuning="early-stopping sur normal-val",
        fine_tuning_epochs=ft_ep, percentiles=PCTS, input_dim=len(feats), n_features=len(feats),
        feature_names=feats, devices=devs, encoder_decoder_layer_dims=layer_dims(prox_sd),
        activation="ReLU (couches cachées) + Sigmoid (sortie)", scaler="MinMax [0,1] local par client",
        anomaly_score="erreur de reconstruction MSE par caractéristique",
        files=dict(global_model="global_fedprox.pt", local_models="local/<appareil>.pt",
                   scalers="scalers/<appareil>.pkl", thresholds="thresholds.json"))
    (outdir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    # 6) métriques + comparaison avec le chapitre 4
    ref = charger_reference()
    auc_fedprox = res_global["global"]["auc"]; auc_ft = res_ft["global"]["auc"]
    metrics = dict(
        rerun=dict(fedprox_global_auc=auc_fedprox, fedprox_ft_global_auc=auc_ft,
                   fedprox_ft_rappel_p99=res_ft["global"]["p99"]["global"]["rappel"],
                   fedprox_ft_fpr_p99=res_ft["global"]["p99"]["global"]["fpr"],
                   par_appareil_ft_auc={d: res_ft["par_appareil"][d]["auc"] for d in devs}),
        reference_chapitre4=ref, temps_min=round((time.time() - t0) / 60, 1))
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    # 7) comparaison affichée + garde-fou (écart AUC > 0,01 -> arrêt)
    print(f"\n=== Artefacts sauvegardés dans {outdir} ===")
    print(f"  global_fedprox.pt · local/*.pt ({len(devs)}) · scalers/*.pkl ({len(devs)}) · thresholds.json · config.json")
    print("\n=== Comparaison avec le chapitre 4 (AUC) ===")
    tol, depasse = 0.01, False
    def ligne(nom, obtenu, attendu):
        nonlocal depasse
        if attendu is None:
            print(f"  {nom:<34} obtenu={obtenu:.4f}  (référence absente)"); return
        delta = obtenu - attendu
        flag = "OK" if abs(delta) <= tol else ">>> ÉCART > 0,01 <<<"
        if abs(delta) > tol: depasse = True
        print(f"  {nom:<34} obtenu={obtenu:.4f}  chap.4={attendu:.4f}  Δ={delta:+.4f}  {flag}")
    r42 = ref.get("graine42", {}); r5 = ref.get("moyenne5", {})
    ligne("FedProx global (graine 42)", auc_fedprox, r42.get("fedprox"))
    ligne("FedProx+FT (graine 42)", auc_ft, r42.get("fedprox_ft"))
    ligne("FedProx+FT (moyenne 5 graines)", auc_ft, r5.get("fedprox_ft"))
    if depasse:
        print("\n!!! Écart d'AUC supérieur à 0,01 : voir metrics.json et l'explication.")
    else:
        print("\nTous les écarts d'AUC sont ≤ 0,01 : reproduction conforme au chapitre 4.")
    print(f"\nTemps total : {metrics['temps_min']} min")


if __name__ == "__main__":
    main()
