"""Harmonize the raw BoT-IoT dataset into a single processed CSV that the
existing federated pipeline (projet/pytorchexample/task.py) can
ingest unchanged.

What this does, and why
------------------------
The raw drop (`Raw_BoT_IoT_dataset/data_1.csv` ... `data_75.csv`) is the
**full** BoT-IoT flow export, not the commonly-cited 5% subset: 73,370,443
rows total, confirmed by a full scan. Class balance is extreme even before
subsampling (attack=73,360,900 vs benign=9,543 — a real reflection of the
"anomalies are rare" premise in the thesis objective, not a bug):

    category        rows
    DDoS            38,532,480
    DoS             33,005,194
    Reconnaissance   1,821,639
    Normal               9,543
    Theft                 1,587   (Keylogging 1,469 + Data_Exfiltration 118)

Training federated rounds on 73M rows is not tractable for a thesis-scale
laptop simulation, and 75 per-shard CSVs are each dominated by a single
attack category (e.g. data_1.csv is 99.8% Reconnaissance) — so shard
identity is a near-perfect label proxy. Both problems are solved by:

1. Streaming through all 75 shards (never holding more than one in memory)
   and stratified-subsampling each category down towards a fixed cap, with
   the two rarest classes (Normal, Theft) kept at 100%. Sampling fraction
   per category is fixed in advance from the global counts above, so it
   applies uniformly across shards (seed=42, reproducible).
2. Writing ONE consolidated output file, not one-per-shard, so task.py's
   existing src_<filename> one-hot logic (built for TON-IoT's per-device
   files) degrades harmlessly to a constant column instead of leaking which
   shard (~= which attack campaign) a row came from.

Column handling
----------------
- Dropped as identifiers / near-empty artifacts of the flow exporter:
  pkSeqID, ltime, seq, saddr, sport, daddr, dport, smac, dmac, soui, doui,
  sco, dco. (IP/MAC/port fields are high-cardinality and specific to this
  testbed's simulated topology — keeping them would let the model
  memorize "which host" instead of learning attack behaviour.)
- `stime` (unix epoch) is converted to a `time` "HH:MM:SS" string column and
  dropped — this lets task.py's existing hour_of_day feature-engineering
  (built for TON-IoT in Step 2) apply here too, unchanged, which is exactly
  the cross-dataset hypothesis Step 5 flagged: does hour_of_day still
  dominate SHAP importance outside TON-IoT?
- `proto` / `flgs` / `state` are one-hot encoded here (task.py's numeric-only
  column auto-discovery would otherwise silently drop these string columns
  and lose real signal, the same failure mode Step 2 fixed for TON-IoT's
  temp_condition/door_state columns).
- `attack` (0/1) is kept as-is — task.py already recognizes "attack" as a
  label column.
- `category` / `subcategory` are kept in the output file as metadata (not
  auto-picked-up as features, since they're non-numeric strings) for the
  Step 6 per-attack-type detection-rate breakdown.
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RAW_DIR = Path(os.environ.get("IOT_DATA", Path(__file__).resolve().parents[2] / "data")) / "Raw_BoT_IoT_dataset"
OUT_DIR = Path(__file__).parent / "Processed_BoT_IoT_dataset"
OUT_PATH = OUT_DIR / "BoT_IoT_processed.csv"

SEED = 42

# Global counts from the full 73.37M-row scan (see docstring) drive the
# per-category sampling fraction so every shard is subsampled consistently.
GLOBAL_CATEGORY_COUNTS = {
    "DDoS": 38_532_480,
    "DoS": 33_005_194,
    "Reconnaissance": 1_821_639,
    "Normal": 9_543,
    "Theft": 1_587,
}
CATEGORY_CAPS = {
    "DDoS": 400_000,
    "DoS": 400_000,
    "Reconnaissance": 200_000,
    "Normal": 9_543,   # keep all
    "Theft": 1_587,    # keep all
}
SAMPLE_FRAC = {
    cat: min(1.0, CATEGORY_CAPS[cat] / GLOBAL_CATEGORY_COUNTS[cat])
    for cat in GLOBAL_CATEGORY_COUNTS
}

DROP_COLUMNS = [
    "pkSeqID", "ltime", "seq", "saddr", "sport", "daddr", "dport",
    "smac", "dmac", "soui", "doui", "sco", "dco",
]

PROTO_VALUES = ["arp", "icmp", "igmp", "ipv6-icmp", "rarp", "tcp", "udp"]
STATE_VALUES = [
    "ACC", "CLO", "CON", "ECO", "FIN", "INT", "MAS", "NRS",
    "PAR", "REQ", "RSP", "RST", "TST", "URP",
]
# flgs values contain literal spaces; map each to a filesystem/CSV-safe suffix.
FLGS_VALUES = {
    "e": "e",
    "e    F": "e_F",
    "e   t": "e_t",
    "e  D": "e_D",
    "e &": "e_amp",
    "e *": "e_star",
    "e d": "e_d",
    "e dD": "e_dD",
    "e dS": "e_dS",
    "e g": "e_g",
    "e r": "e_r",
    "e s": "e_s",
    "eU": "eU",
}


def _stime_to_hhmmss(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(float(epoch_seconds), tz=timezone.utc).strftime("%H:%M:%S")


def _process_shard(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # Stratified subsample per category (see module docstring for rationale).
    sampled = []
    for category, group in df.groupby("category"):
        frac = SAMPLE_FRAC.get(category, 1.0)
        sampled.append(group.sample(frac=frac, random_state=SEED) if frac < 1.0 else group)
    df = pd.concat(sampled, ignore_index=True)

    df["time"] = df["stime"].apply(_stime_to_hhmmss)
    df = df.drop(columns=["stime"] + DROP_COLUMNS)

    for value in PROTO_VALUES:
        df[f"proto_{value}"] = (df["proto"] == value).astype(int)
    for value in STATE_VALUES:
        df[f"state_{value}"] = (df["state"] == value).astype(int)
    for value, suffix in FLGS_VALUES.items():
        df[f"flgs_{suffix}"] = (df["flgs"] == value).astype(int)
    df = df.drop(columns=["proto", "state", "flgs"])

    return df


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    shard_paths = sorted(
        (p for p in RAW_DIR.glob("data_*.csv") if p.stem.split("_")[1].isdigit()),
        key=lambda p: int(p.stem.split("_")[1]),
    )

    total_rows = 0
    category_totals: dict = {}
    for i, path in enumerate(shard_paths, start=1):
        df = _process_shard(path)
        df.to_csv(OUT_PATH, mode="w" if i == 1 else "a", header=(i == 1), index=False)
        total_rows += len(df)
        for cat, n in df["category"].value_counts().items():
            category_totals[cat] = category_totals.get(cat, 0) + int(n)
        print(f"[{i}/{len(shard_paths)}] {path.name}: {len(df)} rows sampled (running total={total_rows})")

    print(f"\nWrote {total_rows} rows -> {OUT_PATH}")
    print("Category breakdown:")
    for cat, n in sorted(category_totals.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:20s} {n}")


if __name__ == "__main__":
    main()
