"""N-BaIoT — chargement auto-supervisé par appareil (données + découpage + scaler + attaques).

Contraintes (voir demande) :
- découpage CHRONOLOGIQUE du normal : 60 % train / 20 % val / 20 % test, dans l'ordre des lignes
  (les features sont des fenêtres glissantes ; un découpage aléatoire créerait une fuite) ;
- scaler LOCAL par client, ajusté sur le NORMAL TRAIN du client uniquement ;
- les attaques ne servent QU'au test ; échantillonnage stratifié par appareil et par famille ;
- sous-échantillonnage optionnel du normal (plafond par client), en gardant les écarts de quantité.
"""

from __future__ import annotations
import os

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

RAW = Path(os.environ.get("IOT_DATA", Path(__file__).resolve().parents[2] / "data")) / "Raw_NBaIoT_dataset"
GAFGYT = ["combo", "junk", "scan", "tcp", "udp"]          # BASHLITE
MIRAI = ["ack", "scan", "syn", "udp", "udpplain"]
SEED = 42


def devices() -> list[str]:
    return sorted(p.name for p in RAW.iterdir() if p.is_dir() and (p / "benign_traffic.csv").exists())


def _read(path: Path) -> np.ndarray:
    return pd.read_csv(path).to_numpy(dtype=np.float32)


def row_hash(a: np.ndarray) -> np.ndarray:
    """Empreinte par ligne (pour contrôler l'absence de doublon train/test)."""
    return pd.util.hash_pandas_object(pd.DataFrame(a).round(6), index=False).to_numpy()


def load_device(dev: str, cap: int | None, seed: int = SEED) -> dict:
    """Charge un appareil : normal découpé chronologiquement + attaques par famille (pour test)."""
    ddir = RAW / dev
    normal = _read(ddir / "benign_traffic.csv")          # ordre = ordre de capture (chronologique)
    n = len(normal)
    n_tr, n_va = int(0.6 * n), int(0.2 * n)
    tr = normal[:n_tr]
    va = normal[n_tr:n_tr + n_va]
    te = normal[n_tr + n_va:]
    # sous-échantillonnage du normal TRAIN uniquement (plafond), en gardant l'ordre chronologique
    if cap is not None and len(tr) > cap:
        idx = np.linspace(0, len(tr) - 1, cap).astype(int)   # sous-échantillon régulier (garde l'ordre)
        tr = tr[idx]
    # Déduplication de contenu : le trafic IoT au repos produit des fenêtres statistiques
    # IDENTIQUES à des instants différents. On retire de val/test toute ligne dont le contenu
    # apparaît dans train (et de test ce qui est dans val), pour qu'AUCUNE ligne identique ne soit
    # partagée entre train, val et test (contrainte). Ce n'est pas une fuite temporelle (découpage
    # chronologique) mais des doublons de contenu ; les retirer rend le taux de faux positifs honnête.
    htr = set(row_hash(tr).tolist())
    va = va[~np.isin(row_hash(va), list(htr))]
    hva = set(row_hash(va).tolist())
    te = te[~np.isin(row_hash(te), list(htr | hva))]
    dedup = dict(val_retirees=int(n_va - len(va)), test_retirees=int((n - n_tr - n_va) - len(te)))

    attacks = {}
    for fam, subs in (("bashlite", GAFGYT), ("mirai", MIRAI)):
        adir = ddir / f"{'gafgyt' if fam == 'bashlite' else 'mirai'}_attacks"
        if not adir.is_dir():
            continue
        for s in subs:
            f = adir / f"{s}.csv"
            if f.exists():
                attacks[(fam, s)] = _read(f)
    return dict(dev=dev, train=tr, val=va, test=te, attacks=attacks, n_normal=n, dedup=dedup)


def check_no_leak(tr: np.ndarray, te: np.ndarray) -> int:
    """Nombre de lignes de test identiques (contenu) à une ligne de train."""
    htr = set(row_hash(tr).tolist())
    return int(np.isin(row_hash(te), list(htr)).sum())


def fit_local_scaler(train_normal: np.ndarray) -> MinMaxScaler:
    """Scaler LOCAL : ajusté sur le NORMAL TRAIN du client seulement (jamais inter-clients)."""
    return MinMaxScaler().fit(train_normal)


def sample_attacks(dev_data: dict, n_target: int, seed: int = SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Échantillon stratifié d'attaques pour le test d'un appareil : ~n_target lignes réparties
    également entre familles présentes, puis entre sous-types. Retourne (X, famille, sous_type)."""
    rng = np.random.default_rng(seed)
    fams = sorted({f for (f, _) in dev_data["attacks"]})
    if not fams:
        return np.empty((0, 115), np.float32), np.array([]), np.array([])
    per_fam = max(n_target // len(fams), 1)
    X, famcol, subcol = [], [], []
    for fam in fams:
        subs = [(s, a) for (f, s), a in dev_data["attacks"].items() if f == fam]
        per_sub = max(per_fam // len(subs), 1)
        for s, a in subs:
            k = min(per_sub, len(a))
            sel = rng.choice(len(a), k, replace=False)
            X.append(a[sel]); famcol += [fam] * k; subcol += [s] * k
    return np.vstack(X), np.array(famcol), np.array(subcol)
