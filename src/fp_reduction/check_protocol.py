"""Contrôles préalables de l'étude (à lancer avant toute évaluation).

1. La reconstruction des splits clients est identique à task.load_partition_data.
2. Le modèle d'origine reproduit les chiffres du mémoire sur le test d'origine.
3. Recouvrement test d'origine / train des clients (fuite).
4. Les faux positifs d'origine ont-ils des « jumeaux » identiques étiquetés attaque
   dans le TRAIN ? (si oui : aucun seuil ni pondération ne peut les séparer)
"""

from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from common import (DATA_DIR, NUM_CLIENTS, PATTERN, RESULTS, SEED, TEST_SIZE, Split,
                    load_model, scores)
from pytorchexample.task import load_partition_data

torch.set_num_threads(4)
RESULTS.mkdir(parents=True, exist_ok=True)
out = {}

s = Split("")
print("NaN dans les features brutes :", int(np.isnan(s.raw).sum()))

# 1. reconstruction client 0 vs DataLoader réel
trl, _, d, _ = load_partition_data(0, NUM_CLIENTS, str(DATA_DIR), PATTERN, 512, TEST_SIZE, SEED,
                                   use_weighted_sampler=False)
same = (np.array_equal(trl.dataset.labels.numpy(), s.y[s.client_train[0]])
        and np.allclose(trl.dataset.features.numpy(), s.client_x(0), atol=1e-6))
out["reconstruction_client0_identique"] = bool(same)
print("1. reconstruction client 0 identique :", same)

# 2. reproduction du baseline (test d'origine = split centralisé)
idx = np.arange(len(s.y))
c_tr, c_te = train_test_split(idx, test_size=TEST_SIZE, random_state=SEED, shuffle=True, stratify=s.y)
sc = MinMaxScaler().fit(s.raw[c_tr])
m = load_model("final_model_botiot.pt", s.raw.shape[1])
p = scores(m, sc.transform(s.raw[c_te]).astype(np.float32))
yt = s.y[c_te]
pred = p >= 0.09
out["baseline_reproduit"] = dict(f1_macro=float(f1_score(yt, pred, average="macro")),
                                 auc=float(roc_auc_score(yt, p)),
                                 fp=int((pred & (yt == 0)).sum()), normaux=int((yt == 0).sum()))
print("2. baseline reproduit :", out["baseline_reproduit"])

# 3. fuite
leak = np.isin(c_te, s.train_idx)
out["test_origine_vu_en_train"] = float(leak.mean())
out["test_origine_normaux_vus_en_train"] = float(leak[yt == 0].mean())
print("3. part du test d'origine vue en entraînement :", out["test_origine_vu_en_train"])

# 4. jumeaux des faux positifs (sur la VALIDATION uniquement)
pv = scores(m, s.x(s.val_idx))
yv = s.y[s.val_idx]
fp_rows = s.val_idx[(pv >= 0.09) & (yv == 0)]
key = lambda rows: [r.tobytes() for r in s.raw[rows]]
from collections import Counter
tr_att = Counter(key(s.train_idx[s.y[s.train_idx] == 1]))
tr_nor = Counter(key(s.train_idx[s.y[s.train_idx] == 0]))
k_fp = key(fp_rows)
twins_att = np.array([tr_att.get(k, 0) for k in k_fp])
twins_nor = np.array([tr_nor.get(k, 0) for k in k_fp])
out["validation_fp_seuil_0.09"] = int(len(fp_rows))
out["fp_avec_jumeau_attaque_en_train"] = int((twins_att > 0).sum())
out["fp_avec_jumeau_normal_en_train"] = int((twins_nor > 0).sum())

# Profil des FP : flux UDP courts type DNS ?
cols = s.columns
c = {n: cols.index(n) for n in ["proto_udp", "state_INT", "pkts", "bytes", "dur", "hour_of_day"]}
R = s.raw[fp_rows]
dns_like = (R[:, c["proto_udp"]] == 1) & (R[:, c["state_INT"]] == 1) & (R[:, c["pkts"]] <= 4)
out["fp_profil_udp_int_<=4pkts"] = int(dns_like.sum())
out["fp_scores_les_plus_frequents"] = {f"{v:.3f}": int(n) for v, n in
                                       Counter(np.round(pv[(pv >= 0.09) & (yv == 0)], 3)).most_common(5)}
out["fp_pkts_bytes_les_plus_frequents"] = {f"{int(a)} pkts / {int(b)} o": int(n) for (a, b), n in
                                           Counter(map(tuple, R[:, [c['pkts'], c['bytes']]])).most_common(5)}
# Combien d'attaques du TRAIN ont exactement le profil « UDP, INT, 2 paquets » ? et de quels types ?
T = s.raw[s.train_idx]
prof = (T[:, c["proto_udp"]] == 1) & (T[:, c["state_INT"]] == 1) & (T[:, c["pkts"]] == 2)
out["train_profil_udp_int_2pkts"] = {
    "normal": int((prof & (s.y[s.train_idx] == 0)).sum()),
    **{f"attaque/{k}": int(v) for k, v in Counter(
        s.subcategory[s.train_idx][prof & (s.y[s.train_idx] == 1)]).most_common()}}
print(json.dumps(out, indent=2, ensure_ascii=False))
(RESULTS / "check_protocol.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
