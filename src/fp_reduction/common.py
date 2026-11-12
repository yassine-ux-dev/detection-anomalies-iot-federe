"""Données et protocole communs à l'étude « réduction des faux positifs » BoT-IoT.

Pourquoi un protocole propre ? Le test d'origine (load_centralized_data, 202 228
lignes) est tiré indépendamment des partitions clients : ~80 % de ses lignes ont
servi à l'entraînement d'un client, et le seuil 0.09 a été choisi en balayant ce
même ensemble (evaluate_checkpoint.py). Ici :

- TRAIN      = union des splits train des 7 clients (exactement les lignes vues par
               le modèle fédéré, reconstruites à l'index près depuis task.py) ;
- HELD-OUT   = union des splits test des 7 clients (jamais vus à l'entraînement) ;
- VALIDATION = 50 % du held-out (stratifié par sous-catégorie, seed 42) -> TOUS les
               choix (seuil, poids, variante) ;
- TEST       = l'autre 50 % -> utilisé une seule fois, à la fin.

Le scaler MinMax est ajusté sur TRAIN uniquement.
"""

from __future__ import annotations
import os

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

HERE = Path(__file__).resolve().parent
QS = HERE.parents[1]
sys.path.insert(0, str(QS))

from pytorchexample.task import (  # noqa: E402  (chargé en mémoire une seule fois)
    Net,
    _apply_exclude,
    _load_engineered_files,
    _prepare_features_and_labels,
    get_feature_columns,
    list_csv_files,
)

DATA_DIR = Path(os.environ.get("IOT_DATA", Path(__file__).resolve().parents[2] / "data")) / "Processed_BoT_IoT_dataset"
PATTERN = "BoT_IoT_processed.csv"
RESULTS = QS / "results" / "fp_reduction"
NUM_CLIENTS = 7
TEST_SIZE = 0.2
SEED = 42

# Types d'attaque demandés (catégorie/sous-catégorie BoT-IoT)
ATTACK_TYPES = {
    "DDoS": lambda c, s: c == "DDoS",
    "DoS": lambda c, s: c == "DoS",
    "Scan": lambda c, s: c == "Reconnaissance",
    "Vol de données": lambda c, s: s == "Data_Exfiltration",
    "Keylogging": lambda c, s: s == "Keylogging",
}

# Modèles évalués : nom -> (checkpoint, features exclues)
MODELS = {
    "origine": ("final_model_botiot.pt", ""),
    "sans_hour": ("final_model_botiot_nohour.pt", "hour_of_day"),
    "class_weights_x5": ("final_model_botiot_cw5.pt", ""),  # train_botiot_classweight.py
}


class Split:
    """Tout ce qu'il faut pour un jeu de features donné (avec ou sans hour_of_day)."""

    def __init__(self, exclude: str = "") -> None:
        files = list_csv_files(str(DATA_DIR), PATTERN)
        self.columns = _apply_exclude(get_feature_columns(files), exclude)
        df = pd.concat(_load_engineered_files(files), ignore_index=True)
        feats, y = _prepare_features_and_labels(df, self.columns)
        self.raw = feats.to_numpy(dtype=np.float64)
        self.y = y.astype(int)
        self.category = df["category"].astype(str).to_numpy()
        self.subcategory = df["subcategory"].astype(str).to_numpy()
        n = len(df)

        # Reconstruction exacte de load_partition_data (partition_by_file=False)
        perm = df.sample(frac=1.0, random_state=SEED).index.to_numpy()
        size = n // NUM_CLIENTS
        self.client_train, client_test = [], []
        for k in range(NUM_CLIENTS):
            part = perm[k * size:(k + 1) * size] if k < NUM_CLIENTS - 1 else perm[k * size:]
            a, b = train_test_split(part, test_size=TEST_SIZE, random_state=SEED,
                                    shuffle=True, stratify=self.y[part])
            self.client_train.append(a)
            client_test.append(b)
        self.train_idx = np.concatenate(self.client_train)
        held = np.concatenate(client_test)
        strat = np.char.add(self.category[held].astype(str), "/" + self.subcategory[held])
        self.val_idx, self.test_idx = train_test_split(
            held, test_size=0.5, random_state=SEED, shuffle=True, stratify=strat)
        assert not np.intersect1d(self.train_idx, held).size

        self.scaler = MinMaxScaler().fit(self.raw[self.train_idx])

    def x(self, idx: np.ndarray) -> np.ndarray:
        return self.scaler.transform(self.raw[idx]).astype(np.float32)

    def client_x(self, k: int) -> np.ndarray:
        """Features du client k mises à l'échelle comme dans task.py (scaler du client)."""
        idx = self.client_train[k]
        return MinMaxScaler().fit_transform(self.raw[idx]).astype(np.float32)


def load_model(checkpoint: str, input_dim: int) -> Net:
    m = Net(input_dim)
    m.load_state_dict(torch.load(QS / checkpoint, map_location="cpu"))
    return m.eval()


@torch.no_grad()
def scores(model: Net, x: np.ndarray, batch: int = 8192) -> np.ndarray:
    out = [torch.softmax(model(torch.from_numpy(x[i:i + batch])), dim=1)[:, 1].numpy()
           for i in range(0, len(x), batch)]
    return np.concatenate(out).astype(np.float64)


def type_masks(cat: np.ndarray, sub: np.ndarray) -> dict:
    return {name: f(cat, sub) for name, f in ATTACK_TYPES.items()}
