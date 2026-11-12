"""Exporte les scalers MinMax du protocole propre (ajustés sur l'union des splits train des
clients, common.Split), pour les appliquer à des données externes (test labo) avec la même
normalisation que celle sous laquelle les seuils ont été choisis.

  - scaler_train_clients.pkl         : 51 features (modèle final_model_botiot.pt)
  - scaler_train_clients_nohour.pkl  : 50 features sans hour_of_day (final_model_botiot_nohour.pt)

    python export_scaler.py
"""

import json

import joblib

from common import RESULTS, Split

for exclude, suffix in [("", ""), ("hour_of_day", "_nohour")]:
    s = Split(exclude)
    joblib.dump(s.scaler, RESULTS / f"scaler_train_clients{suffix}.pkl")
    (RESULTS / f"scaler_train_clients{suffix}.json").write_text(json.dumps(dict(
        features=s.columns, data_min=s.scaler.data_min_.tolist(), data_max=s.scaler.data_max_.tolist(),
        lignes_train=int(len(s.train_idx))), indent=1), encoding="utf-8")
    print(f"scaler{suffix} : {len(s.columns)} features, ajusté sur {len(s.train_idx)} lignes d'entraînement")
