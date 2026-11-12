"""charger_modele.py — recharge les artefacts de l'autoencodeur SANS réentraîner.

Utilisation en bibliothèque :
    from charger_modele import ModeleAE
    m = ModeleAE.charger("nbaiot")                      # ou chemin vers artifacts/nbaiot
    err = m.score("Danmini_Doorbell", X_brut)           # erreur de reconstruction par ligne
    est_anomalie, err = m.predire("Danmini_Doorbell", X_brut, percentile="99")

Démonstration (lecture seule, aucune donnée réentraînée) :
    python charger_modele.py --dataset nbaiot
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from nbaiot_ae import AE, recon_error

import sys
try:  # affichage UTF-8 (symboles mu, accents) meme sur une console Windows (cp1252)
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

QS = Path(__file__).resolve().parents[2]
ART = QS / "artifacts"


class ModeleAE:
    """Charge le modèle global FedProx, les modèles locaux fine-tunés, les scalers et les seuils."""

    def __init__(self, config, global_model, locaux, scalers, seuils, racine):
        self.config = config
        self.global_model = global_model        # modèle global FedProx (avant fine-tuning)
        self.locaux = locaux                    # {appareil: AE fine-tuné}
        self.scalers = scalers                  # {appareil: MinMaxScaler}
        self.seuils = seuils                    # thresholds.json (local + global)
        self.racine = racine
        self.features = config.get("feature_names")
        self.appareils = config.get("devices")

    @classmethod
    def charger(cls, dataset_ou_chemin: str) -> "ModeleAE":
        p = Path(dataset_ou_chemin)
        racine = p if (p / "config.json").exists() else ART / dataset_ou_chemin
        config = json.loads((racine / "config.json").read_text(encoding="utf-8"))
        seuils = json.loads((racine / "thresholds.json").read_text(encoding="utf-8"))

        gsd = torch.load(racine / "global_fedprox.pt", map_location="cpu")
        gm = AE(gsd["net.0.weight"].shape[1]); gm.load_state_dict(gsd); gm.eval()

        locaux, scalers = {}, {}
        for dev in config["devices"]:
            sd = torch.load(racine / "local" / f"{dev}.pt", map_location="cpu")
            m = AE(sd["net.0.weight"].shape[1]); m.load_state_dict(sd); m.eval()
            locaux[dev] = m
            with open(racine / "scalers" / f"{dev}.pkl", "rb") as fh:
                scalers[dev] = pickle.load(fh)
        return cls(config, gm, locaux, scalers, seuils, racine)

    def _prep(self, dev: str, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return np.clip(self.scalers[dev].transform(X), 0, 1).astype(np.float32)

    def score(self, dev: str, X_brut, modele: str = "local") -> np.ndarray:
        """Erreur de reconstruction (score d'anomalie) par ligne, avec le scaler de l'appareil."""
        m = self.locaux[dev] if modele == "local" else self.global_model
        return recon_error(m, self._prep(dev, X_brut))

    def seuil(self, dev: str, percentile="99", portee: str = "local") -> float:
        src = self.seuils["local"][dev] if portee == "local" else self.seuils["global"]
        return float(src[str(percentile)])

    def predire(self, dev: str, X_brut, percentile="99", portee: str = "local", modele: str = "local"):
        """Retourne (anomalie: bool[], erreur: float[]) : anomalie si erreur >= seuil(percentile)."""
        err = self.score(dev, X_brut, modele=modele)
        return err >= self.seuil(dev, percentile, portee), err


def _demo(dataset: str):
    m = ModeleAE.charger(dataset)
    print(f"Artefacts chargés depuis : {m.racine}")
    print(f"  dataset={m.config['dataset']} · graine={m.config['seed']} · μ={m.config['mu']} · "
          f"{len(m.appareils)} appareils · {m.config['n_features']} caractéristiques")
    print(f"  modèle global FedProx + {len(m.locaux)} modèles locaux + {len(m.scalers)} scalers chargés (sans réentraînement).")
    dev = m.appareils[0]
    print(f"  seuils {dev} (local) : " + ", ".join(f"p{p}={m.seuil(dev, p):.4f}" for p in ("95", "99", "99.9")))

    # Démonstration d'inférence si les données brutes sont disponibles (lecture seule).
    try:
        import nbaiot_data as D
        ddir = D.RAW / dev
        import pandas as pd
        normal = pd.read_csv(ddir / "benign_traffic.csv").to_numpy(np.float32)[-500:]   # fin = jamais vue
        atk_file = next((ddir / f"gafgyt_attacks" / f"{s}.csv" for s in D.GAFGYT
                         if (ddir / "gafgyt_attacks" / f"{s}.csv").exists()), None)
        atk = pd.read_csv(atk_file).to_numpy(np.float32)[:500] if atk_file else None
        an_n, _ = m.predire(dev, normal, percentile="99")
        print(f"  [demo] {dev} : normal signalé anomalie = {100*an_n.mean():.1f} % (attendu ~1 % au seuil p99)")
        if atk is not None:
            an_a, _ = m.predire(dev, atk, percentile="99")
            print(f"  [demo] {dev} : attaques détectées = {100*an_a.mean():.1f} %")
    except Exception as e:
        print(f"  (démonstration d'inférence ignorée : {e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nbaiot")
    _demo(ap.parse_args().dataset)


if __name__ == "__main__":
    main()
