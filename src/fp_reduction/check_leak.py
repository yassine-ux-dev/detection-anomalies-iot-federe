"""Le test « centralisé » d'origine recouvre-t-il les données d'entraînement des clients ?

Pour un dataset entraîné avec load_partition_data(partition_by_file=False) — TON-IoT,
BoT-IoT — reconstruit (code actuel de task.py) :
  - le test d'origine (load_centralized_data : split 80/20 stratifié du jeu concaténé) ;
  - les lignes vues à l'entraînement (union des splits train des 7 clients).
Mesure le chevauchement d'index et de contenu, puis compare les performances du modèle
sur les lignes de test DÉJÀ VUES et JAMAIS VUES : un écart net montre que la fuite gonfle
réellement les chiffres (et que la reconstruction correspond à l'entraînement réel).

    python check_leak.py --dataset ton_iot
"""

from __future__ import annotations
import os

import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from common import NUM_CLIENTS, QS, RESULTS, SEED, TEST_SIZE, load_model, scores
from pytorchexample.task import (_load_engineered_files, _prepare_features_and_labels,
                                 get_feature_columns, list_csv_files)

torch.set_num_threads(4)
CFG = {
    "ton_iot": dict(dir="Processed_IoT_dataset", pattern="IoT_*.csv", ckpt="final_model.pt",
                    baseline="baseline_ton_iot.json"),
    "botiot": dict(dir="Processed_BoT_IoT_dataset", pattern="BoT_IoT_processed.csv",
                   ckpt="final_model_botiot.pt", baseline="baseline_botiot.json"),
}


def perf(y, p, t):
    pred = p >= t
    return dict(n=int(len(y)), normaux=int((y == 0).sum()), auc=float(roc_auc_score(y, p)),
                f1_macro=float(f1_score(y, pred, average="macro")),
                fpr=float((pred & (y == 0)).sum() / max((y == 0).sum(), 1)),
                rappel=float((pred & (y == 1)).sum() / max((y == 1).sum(), 1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(CFG), required=True)
    name = ap.parse_args().dataset
    c = CFG[name]
    files = list_csv_files(str(Path(os.environ.get("IOT_DATA", Path(__file__).resolve().parents[2] / "data")) / c["dir"]), c["pattern"])
    cols = get_feature_columns(files)
    df = pd.concat(_load_engineered_files(files), ignore_index=True)
    feats, y = _prepare_features_and_labels(df, cols)
    raw = feats.to_numpy(dtype=np.float64)
    hashes = pd.util.hash_pandas_object(feats.round(8), index=False).to_numpy()
    n = len(y)
    del df, feats

    idx = np.arange(n)
    c_tr, c_te = train_test_split(idx, test_size=TEST_SIZE, random_state=SEED, shuffle=True, stratify=y)
    perm = pd.Series(idx).sample(frac=1.0, random_state=SEED).to_numpy()
    size = n // NUM_CLIENTS
    seen = []
    for k in range(NUM_CLIENTS):
        part = perm[k * size:(k + 1) * size] if k < NUM_CLIENTS - 1 else perm[k * size:]
        a, _ = train_test_split(part, test_size=TEST_SIZE, random_state=SEED, shuffle=True, stratify=y[part])
        seen.append(a)
    seen = np.concatenate(seen)
    in_train = np.isin(c_te, seen)
    dup = np.isin(hashes[c_te], np.unique(hashes[seen]))

    base = json.loads((QS / "results" / c["baseline"]).read_text(encoding="utf-8"))
    thr = float(base["decision_threshold"])
    sc = MinMaxScaler().fit(raw[c_tr])  # comme evaluate_checkpoint.py
    p = scores(load_model(c["ckpt"], len(cols)), sc.transform(raw[c_te]).astype(np.float32))
    yt = y[c_te]
    out = dict(
        dataset=c["dir"], lignes=n, test_origine=int(len(c_te)), seuil=thr,
        test_vu_en_train_index=float(in_train.mean()),
        test_normaux_vus_en_train_index=float(in_train[yt == 0].mean()),
        test_contenu_identique_a_une_ligne_train=float(dup.mean()),
        test_jamais_vu_ni_doublon=float((~in_train & ~dup).mean()),
        perf_test_complet=perf(yt, p, thr), perf_lignes_vues=perf(yt[in_train], p[in_train], thr),
        perf_lignes_jamais_vues=perf(yt[~in_train], p[~in_train], thr),
        baseline_publie=dict(f1_macro=base["f1_macro"], auc=base["auc"], fpr=base["false_positive_rate"]),
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))
    (RESULTS / f"leak_check_{name}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
