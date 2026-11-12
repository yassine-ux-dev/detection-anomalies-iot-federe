"""Étape 3 — hétérogénéité : FedAvg vs FedProx (+ fine-tuning, peu de données, IID vs non-IID).

Variantes déclarées à l'avance, TOUTES rapportées sur le test (aucune sélection a posteriori).
Règles : entraînement sur le NORMAL uniquement ; découpage chronologique + dédup (nbaiot_data) ;
seuils p95/p99/p99.9 sur le NORMAL de validation (global + local) ; μ et nb d'époques de
fine-tuning choisis UNIQUEMENT sur la loss de reconstruction du normal de validation.

Parties (séquentielles, un entraînement à la fois) :
  variants : a) FedAvg  b) FedProx (μ sur val)  c) FedAvg+fine-tuning  d) FedProx+fine-tuning
             -- scaler LOCAL par client (méthode principale), cap 50000.
  lowdata  : plafond 1000 puis 5000 lignes normales/appareil, pour local seul, FedAvg,
             FedAvg+fine-tuning, FedProx+fine-tuning (fine-tuning sur les mêmes données plafonnées).
  iid      : ABLATION CONTRÔLÉE (non déployable) — scaler GLOBAL dans les deux bras, FedAvg sur
             partition par appareil (non-IID) vs normal mélangé/redistribué au hasard (IID).

    python nbaiot_hetero.py --part variants
    python nbaiot_hetero.py --part lowdata
    python nbaiot_hetero.py --part iid
    python nbaiot_hetero.py --part all
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

import nbaiot_data as D
from nbaiot_ae import AE, PCTS, evaluate, fedavg, recon_error

OUT = Path(__file__).resolve().parents[2] / "results" / "nbaiot_fed"
DEVICE = torch.device("cpu")
SEED = 42
ROUNDS, EPOCHS = 20, 2
MU_GRID = [0.01, 0.1, 1.0]
FT_MAX = 5           # époques de fine-tuning maxi ; early-stop sur loss normal-val
torch.set_num_threads(4)


def train_client(model, x, epochs, lr=1e-3, batch=256, mu=0.0, gparams=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    dl = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch, shuffle=True)
    model.train()
    for _ in range(epochs):
        for (xb,) in dl:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            if mu > 0 and gparams is not None:
                loss = loss + (mu / 2.0) * sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), gparams))
            loss.backward()
            opt.step()


def train_fed(clients: list[np.ndarray], val_arrays: list[np.ndarray], rounds: int, epochs: int,
              mu: float, label: str, seed: int = SEED) -> tuple[dict, list]:
    """FedAvg (mu=0) ou FedProx (mu>0). clients = liste de matrices normal-train (par client).
    seed : graine d'initialisation + ordre des lots (robustesse)."""
    torch.manual_seed(seed)
    dim = clients[0].shape[1]   # dimension déduite des données (115 N-BaIoT, 17 TON-IoT, 51 BoT-IoT)
    gsd = {k: v.clone() for k, v in AE(dim).state_dict().items()}
    val_hist = []
    for r in range(1, rounds + 1):
        tr = time.time(); sds, sizes = [], []
        for x in clients:
            m = AE(dim).to(DEVICE); m.load_state_dict(gsd)
            gp = [p.clone().detach() for p in m.parameters()] if mu > 0 else None
            train_client(m, x, epochs, mu=mu, gparams=gp)
            sds.append({k: v.detach() for k, v in m.state_dict().items()}); sizes.append(len(x))
        gsd = fedavg(sds, sizes)
        gm = AE(dim).to(DEVICE); gm.load_state_dict(gsd)
        vl = float(np.mean([recon_error(gm, v).mean() for v in val_arrays]))
        val_hist.append(vl)
        if r == 1 or r % 5 == 0 or r == rounds:
            print(f"    [{label}] round {r}/{rounds} loss_val={vl:.6f} ({time.time()-tr:.0f}s)", flush=True)
    return gsd, val_hist


def finetune(gsd: dict, data: dict, devs: list, max_epochs: int = FT_MAX) -> tuple[dict, dict]:
    """Depuis le modèle global, chaque appareil affine sur SON normal-train ; nb d'époques choisi
    par early-stopping sur SA loss normal-val. Retourne {dev: modèle} et {dev: nb d'époques}."""
    models, ep_used = {}, {}
    dim = gsd["net.0.weight"].shape[1]
    for dev in devs:
        m = AE(dim).to(DEVICE); m.load_state_dict(gsd)
        best = (float("inf"), 0, copy.deepcopy(m.state_dict()))
        for e in range(1, max_epochs + 1):
            train_client(m, data[dev]["xtr"], epochs=1, mu=0.0)
            vl = float(recon_error(m, data[dev]["xva"]).mean())
            if vl < best[0]:
                best = (vl, e, copy.deepcopy(m.state_dict()))
        m.load_state_dict(best[2]); models[dev] = m; ep_used[dev] = best[1]
    return models, ep_used


def eval_models(models: dict, data: dict, devs: list) -> dict:
    """models[dev] = modèle à utiliser pour l'appareil (global répété, ou per-device)."""
    per_dev, val_all = {}, []
    for dev in devs:
        ev = recon_error(models[dev], data[dev]["xva"])
        per_dev[dev] = dict(err_norm_test=recon_error(models[dev], data[dev]["xte"]),
                            err_att=recon_error(models[dev], data[dev]["xatt"]) if len(data[dev]["xatt"]) else np.empty(0),
                            fam=data[dev]["fam"], thr_local={p: float(np.percentile(ev, p)) for p in PCTS})
        val_all.append(ev)
    thr_global = {p: float(np.percentile(np.concatenate(val_all), p)) for p in PCTS}
    return evaluate(per_dev, thr_global)


def load_scaled(cap: int, scaler_mode: str = "local", seed: int = SEED):
    """Charge les 9 appareils. scaler_mode='local' (par appareil) ou 'global' (ablation IID).
    Retourne data[dev]={xtr,xva,xte,xatt,fam}, raw_tr[dev] (normal-train brut), devs."""
    devs = D.devices()
    raw_tr, raw_va, raw_te, raw_att, fam = {}, {}, {}, {}, {}
    for dev in devs:
        dd = D.load_device(dev, cap=cap, seed=seed)
        raw_tr[dev], raw_va[dev], raw_te[dev] = dd["train"], dd["val"], dd["test"]
        xa, f, _ = D.sample_attacks(dd, n_target=max(len(dd["test"]), 1))
        raw_att[dev], fam[dev] = xa, f
    if scaler_mode == "global":
        gsc = MinMaxScaler().fit(np.vstack([raw_tr[d] for d in devs]))
        sc = {d: gsc for d in devs}
    else:
        sc = {d: MinMaxScaler().fit(raw_tr[d]) for d in devs}
    clip = lambda z: np.clip(z, 0, 1).astype(np.float32)
    data = {d: dict(xtr=clip(sc[d].transform(raw_tr[d])), xva=clip(sc[d].transform(raw_va[d])),
                    xte=clip(sc[d].transform(raw_te[d])),
                    xatt=clip(sc[d].transform(raw_att[d])) if len(raw_att[d]) else raw_att[d],
                    fam=fam[d]) for d in devs}
    return data, raw_tr, devs


def pick_mu(data, devs, label=""):
    """FedProx : μ choisi sur la loss de reconstruction du normal de validation (jamais attaques)."""
    clients = [data[d]["xtr"] for d in devs]
    vals = [data[d]["xva"] for d in devs]
    results = {}
    for mu in MU_GRID:
        gsd, vh = train_fed(clients, vals, ROUNDS, EPOCHS, mu, f"{label}FedProx μ={mu}")
        results[mu] = dict(gsd=gsd, val_loss=vh[-1], val_hist=vh)
        print(f"    μ={mu}: loss_val finale={vh[-1]:.6f}", flush=True)
    mu_star = min(results, key=lambda m: results[m]["val_loss"])
    print(f"    -> μ* = {mu_star} (loss normal-val minimale)", flush=True)
    return mu_star, results


def run_variants():
    t0 = time.time()
    print("[variants] chargement (cap 50000, scaler local)…", flush=True)
    data, _, devs = load_scaled(cap=50000, scaler_mode="local")
    clients = [data[d]["xtr"] for d in devs]
    vals = [data[d]["xva"] for d in devs]
    out = {"meta": dict(cap=50000, rounds=ROUNDS, epochs=EPOCHS, scaler="local", mu_grid=MU_GRID)}

    print("  a) FedAvg", flush=True)
    avg_sd, avg_vh = train_fed(clients, vals, ROUNDS, EPOCHS, 0.0, "FedAvg")
    out["a_fedavg"] = eval_models({d: _load(avg_sd) for d in devs}, data, devs)
    out["a_fedavg"]["val_hist"] = avg_vh

    print("  b) FedProx (recherche μ sur val)", flush=True)
    mu_star, mu_res = pick_mu(data, devs)
    prox_sd = mu_res[mu_star]["gsd"]
    out["b_fedprox"] = eval_models({d: _load(prox_sd) for d in devs}, data, devs)
    out["b_fedprox"]["mu_star"] = mu_star
    out["b_fedprox"]["mu_val_loss"] = {str(m): mu_res[m]["val_loss"] for m in MU_GRID}

    print("  c) FedAvg + fine-tuning local", flush=True)
    c_models, c_ep = finetune(avg_sd, data, devs)
    out["c_fedavg_ft"] = eval_models(c_models, data, devs)
    out["c_fedavg_ft"]["ft_epochs"] = c_ep

    print("  d) FedProx + fine-tuning local", flush=True)
    d_models, d_ep = finetune(prox_sd, data, devs)
    out["d_fedprox_ft"] = eval_models(d_models, data, devs)
    out["d_fedprox_ft"]["ft_epochs"] = d_ep

    out["meta"]["temps_min"] = round((time.time() - t0) / 60, 1)
    (OUT / "hetero_variants.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_summary("VARIANTS", out, ["a_fedavg", "b_fedprox", "c_fedavg_ft", "d_fedprox_ft"])


def run_lowdata():
    t0 = time.time()
    out = {"meta": dict(rounds=ROUNDS, epochs=EPOCHS, scaler="local", caps=[1000, 5000])}
    for cap in (1000, 5000):
        print(f"[lowdata cap={cap}] chargement…", flush=True)
        data, _, devs = load_scaled(cap=cap, scaler_mode="local")
        clients = [data[d]["xtr"] for d in devs]; vals = [data[d]["xva"] for d in devs]
        block = {}
        # local seul = fine-tuning depuis des poids aléatoires (chaque appareil, son normal)
        torch.manual_seed(SEED)
        rand_sd = {k: v.clone() for k, v in AE().state_dict().items()}
        loc_models, loc_ep = finetune(rand_sd, data, devs, max_epochs=ROUNDS * EPOCHS)
        block["local_seul"] = eval_models(loc_models, data, devs)
        avg_sd, _ = train_fed(clients, vals, ROUNDS, EPOCHS, 0.0, f"cap{cap} FedAvg")
        block["fedavg"] = eval_models({d: _load(avg_sd) for d in devs}, data, devs)
        cmod, _ = finetune(avg_sd, data, devs); block["fedavg_ft"] = eval_models(cmod, data, devs)
        # μ repris de la recherche principale (hetero_variants.json) pour tenir le budget temps ;
        # à défaut, 0.1. (μ reste un hyperparamètre choisi sur la loss normal-val, pas sur attaques.)
        vf = OUT / "hetero_variants.json"
        mu_star = json.loads(vf.read_text(encoding="utf-8"))["b_fedprox"]["mu_star"] if vf.exists() else 0.1
        prox_sd, _ = train_fed(clients, vals, ROUNDS, EPOCHS, mu_star, f"cap{cap} FedProx μ={mu_star}")
        dmod, _ = finetune(prox_sd, data, devs)
        block["fedprox_ft"] = eval_models(dmod, data, devs); block["mu_star"] = mu_star
        out[f"cap_{cap}"] = block
        print(f"  cap {cap}: " + " | ".join(
            f"{k} AUC={block[k]['global']['auc']:.3f}" for k in ("local_seul", "fedavg", "fedavg_ft", "fedprox_ft")),
            flush=True)
    out["meta"]["temps_min"] = round((time.time() - t0) / 60, 1)
    (OUT / "hetero_lowdata.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


def run_iid():
    t0 = time.time()
    print("[iid] ablation contrôlée (scaler GLOBAL, non déployable)…", flush=True)
    data, raw_tr, devs = load_scaled(cap=50000, scaler_mode="global")
    vals = [data[d]["xva"] for d in devs]
    out = {"meta": dict(cap=50000, rounds=ROUNDS, epochs=EPOCHS, scaler="global (ablation)",
                        note="ablation contrôlée non déployable : scaler global dans les deux bras")}
    # non-IID : clients = appareils
    non_clients = [data[d]["xtr"] for d in devs]
    sd_non, vh_non = train_fed(non_clients, vals, ROUNDS, EPOCHS, 0.0, "nonIID")
    out["non_iid"] = eval_models({d: _load(sd_non) for d in devs}, data, devs); out["non_iid"]["val_hist"] = vh_non
    # IID : normal réuni, mélangé, redistribué au hasard en 9 parts
    pooled = np.vstack([data[d]["xtr"] for d in devs])
    rng = np.random.default_rng(SEED); rng.shuffle(pooled)
    shards = np.array_split(pooled, len(devs))
    sd_iid, vh_iid = train_fed(list(shards), vals, ROUNDS, EPOCHS, 0.0, "IID")
    out["iid"] = eval_models({d: _load(sd_iid) for d in devs}, data, devs); out["iid"]["val_hist"] = vh_iid
    out["meta"]["temps_min"] = round((time.time() - t0) / 60, 1)
    (OUT / "hetero_iid.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  non-IID AUC={out['non_iid']['global']['auc']:.4f} | IID AUC={out['iid']['global']['auc']:.4f}", flush=True)


def _load(sd):
    m = AE().to(DEVICE); m.load_state_dict(sd); return m


def _print_summary(title, out, keys):
    print(f"\n=== {title} (global) ===")
    for k in keys:
        g = out[k]["global"]
        print(f"  {k:<14} AUC={g['auc']:.4f} | p95 rap/FPR={100*g['p95']['global']['rappel']:.1f}/"
              f"{100*g['p95']['global']['fpr']:.1f} | p99 rap/FPR={100*g['p99']['global']['rappel']:.1f}/"
              f"{100*g['p99']['global']['fpr']:.1f} | Mirai {100*g['p99']['global']['rappel_mirai']:.1f} "
              f"BASHLITE {100*g['p99']['global']['rappel_bashlite']:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["variants", "lowdata", "iid", "all"], required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.part in ("variants", "all"):
        run_variants()
    if a.part in ("lowdata", "all"):
        run_lowdata()
    if a.part in ("iid", "all"):
        run_iid()
    print("DONE")


if __name__ == "__main__":
    main()
