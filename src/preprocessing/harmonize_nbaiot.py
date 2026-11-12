"""Harmonize the raw N-BaIoT dataset into the same per-device CSV convention
already used for TON-IoT (Processed_IoT_dataset/IoT_<device>.csv), so the
existing federated pipeline (projet/pytorchexample/task.py) can
ingest it unchanged: point data-dir at the output folder and file-pattern at
"NBaIoT_*.csv" (task.py auto-discovers numeric features, the `label` column,
and derives one src_<device> flag per file already).

N-BaIoT's 115 features are float statistics from Kitsune's feature
extractor (weights/means/variances/etc. of network flow behaviour) and are
already numeric, so no categorical encoding is needed like it was for
TON-IoT's temp_condition/door_state/etc.
"""

import os
from pathlib import Path

import pandas as pd

RAW_DIR = Path(os.environ.get("IOT_DATA", Path(__file__).resolve().parents[2] / "data")) / "Raw_NBaIoT_dataset"
OUT_DIR = Path(__file__).parent / "Processed_NBaIoT_dataset"

DEVICES = [
    "Danmini_Doorbell",
    "Ecobee_Thermostat",
    "Ennio_Doorbell",
    "Philips_B120N10_Baby_Monitor",
    "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera",
    "Samsung_SNH_1011_N_Webcam",
    "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera",
]


def _load_device(device_dir: Path) -> pd.DataFrame:
    frames = []

    benign_path = device_dir / "benign_traffic.csv"
    df_benign = pd.read_csv(benign_path)
    df_benign["label"] = 0
    df_benign["attack_family"] = "none"
    df_benign["attack_type"] = "benign"
    frames.append(df_benign)

    for family in ("gafgyt_attacks", "mirai_attacks"):
        family_dir = device_dir / family
        if not family_dir.is_dir():
            continue
        for csv_path in sorted(family_dir.glob("*.csv")):
            df_attack = pd.read_csv(csv_path)
            df_attack["label"] = 1
            df_attack["attack_family"] = family.replace("_attacks", "")
            df_attack["attack_type"] = csv_path.stem
            frames.append(df_attack)

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    summary = []

    for device in DEVICES:
        device_dir = RAW_DIR / device
        if not device_dir.is_dir():
            print(f"[skip] {device}: folder not found")
            continue

        df = _load_device(device_dir)
        out_path = OUT_DIR / f"NBaIoT_{device}.csv"
        df.to_csv(out_path, index=False)

        n_benign = int((df["label"] == 0).sum())
        n_attack = int((df["label"] == 1).sum())
        summary.append((device, len(df), n_benign, n_attack))
        print(f"[ok] {device}: {len(df)} rows (benign={n_benign}, attack={n_attack}) -> {out_path.name}")

    print("\n--- Summary ---")
    total = sum(s[1] for s in summary)
    for device, n, n_benign, n_attack in summary:
        print(f"  {device:45s} rows={n:>8d}  benign={n_benign:>7d}  attack={n_attack:>7d}")
    print(f"  {'TOTAL':45s} rows={total:>8d}")


if __name__ == "__main__":
    main()
