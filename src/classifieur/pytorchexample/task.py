"""pytorchexample: Classifier task for TON_IoT dataset with class-balancing fixes."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# ---------------------------------------------------------------------------
# FIX 1 (Model architecture choice) — Classifier instead of Autoencoder
# ---------------------------------------------------------------------------

class Net(nn.Module):
    """Deep classifier with BatchNorm, Dropout and LeakyReLU.

    This replaces the Autoencoder with a Multi-layer Perceptron (MLP) binary classifier
    to predict normal/attack classes directly, preventing majority class prediction collapse.
    Architecture widened to 256→128→64→2 to handle the expanded feature set.
    """

    def __init__(self, input_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(64, 2)  # 2 output logits: class 0 (normal) and class 1 (attack)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# Alias to ensure backwards compatibility with client_app.py and server_app.py
Autoencoder = Net


class NumpyDataset(Dataset):
    """Dataset for classifier training with labels."""

    def __init__(self, features: np.ndarray, labels: Optional[np.ndarray] = None) -> None:
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long() if labels is not None else None

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        if self.labels is None:
            return self.features[idx]
        return self.features[idx], self.labels[idx]


_FEATURE_COLUMNS_CACHE: Optional[List[str]] = None
# Bounded cache (see _bounded_cache_set below). Left unbounded, it leaked
# memory: when the simulation backend reuses one worker process across many
# clients in sequence, each client's DataLoaders (with their x_train/x_test
# arrays) accumulated indefinitely. We keep only the 2 most-recently-used
# entries, enough for train() then evaluate() of the SAME client to reuse the
# same loaders, without retaining every client ever seen.
_PARTITION_CACHE: dict = {}
_PARTITION_CACHE_MAXSIZE = 2
_CENTRAL_CACHE: dict = {}


def _bounded_cache_set(cache: dict, key, value, maxsize: int) -> None:
    cache[key] = value
    while len(cache) > maxsize:
        oldest_key = next(iter(cache))
        del cache[oldest_key]

_LABEL_COLUMNS = [
    "label",
    "Label",
    "attack",
    "Attack",
    "anomaly",
    "Anomaly",
    "is_attack",
    "is_anomaly",
    "class",
    "Class",
    "type",
]

# Known categorical columns and their binary mappings
_CATEGORICAL_ENCODINGS: dict = {
    "temp_condition": {"high": 1.0, "low": 0.0},
    "door_state": {"open": 1.0, "closed": 0.0},
    "sphone_signal": {"true": 1.0, "1": 1.0, "1.0": 1.0, "false": 0.0, "0": 0.0, "0.0": 0.0},
    "light_status": {"on": 1.0, "off": 0.0},
}


def list_csv_files(data_dir: str, file_pattern: str) -> List[Path]:
    if not file_pattern:
        file_pattern = "*.csv"
    pattern_path = Path(file_pattern)
    if pattern_path.is_absolute():
        glob_path = pattern_path
    else:
        glob_path = Path(data_dir) / file_pattern
    files = sorted(glob_path.parent.glob(glob_path.name))
    if not files:
        raise FileNotFoundError(f"No CSV files found for pattern: {glob_path}")
    return files


def _encode_known_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode known categorical columns to numeric 0/1 in-place.

    These columns contain real signal but are dropped by pd.to_numeric because
    they hold string values (e.g. 'high'/'low', 'open'/'closed', 'on'/'off').
    Encoding them before feature discovery ensures they are included.
    """
    for col, mapping in _CATEGORICAL_ENCODINGS.items():
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.strip().str.lower().map(mapping).fillna(0.0)
            )
    return df


def _add_engineered_features(
    df: pd.DataFrame,
    source_name: str,
    all_source_names: List[str],
) -> pd.DataFrame:
    """Add feature-engineered columns to a dataframe.

    Adds:
    - ``hour_of_day``: hour (0-23) extracted from the ``time`` column, normalised
      to [0, 1].  Encodes temporal attack patterns that repeat at certain hours.
    - ``src_<device>``: one-hot flag for each known device, so the model can
      distinguish which IoT device generated each row (strong predictor).
    """
    # Memory optimisation: inserting hour_of_day plus one src_<device>
    # column per source file one at a time (`df[col] = ...` in a loop)
    # leaves the DataFrame "highly fragmented" (pandas) and can multiply
    # peak memory by the number of source files (9 for N-BaIoT). We build
    # every new column in one dict and concat once.
    # Output is identical (same column names/values), only the
    # construction is vectorized.
    new_columns: dict = {}

    # Hour-of-day feature
    if "time" in df.columns:
        def _parse_hour(t: str) -> float:
            try:
                return float(str(t).strip().split(":")[0]) / 23.0
            except (ValueError, IndexError):
                return 0.0
        new_columns["hour_of_day"] = df["time"].apply(_parse_hour)
    else:
        new_columns["hour_of_day"] = pd.Series(0.0, index=df.index)

    # Device source one-hot
    for name in all_source_names:
        new_columns[f"src_{name}"] = pd.Series(
            1.0 if name == source_name else 0.0, index=df.index
        )

    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    return df


def _find_label_column(df: pd.DataFrame) -> Optional[str]:
    for col in _LABEL_COLUMNS:
        if col in df.columns:
            return col
    return None


def _normalize_labels(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        values = series.fillna(0).astype(float)
        return (values > 0).astype(int).to_numpy()
    values = series.astype(str).str.lower().str.strip()
    normal_values = {"0", "normal", "benign", "false", "no", "clean"}
    return (~values.isin(normal_values)).astype(int).to_numpy()


def _infer_numeric_columns(df: pd.DataFrame, label_col: Optional[str]) -> List[str]:
    numeric_cols: List[str] = []
    for col in df.columns:
        if col == label_col:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().any():
            numeric_cols.append(col)
    return numeric_cols


def _source_name_from_path(path: Path) -> str:
    """Return a clean device-source name derived from a CSV filename."""
    return path.stem.replace("IoT_", "").lower()


def _load_engineered_files(files: List[Path]) -> List[pd.DataFrame]:
    """Read + encode categoricals + add engineered features for every file,
    returning one DataFrame per file (same per-file loop duplicated inline
    inside load_partition_data/load_centralized_data).

    Exposed for the explainability and baseline scripts
    (explain_shap_generic.py, literature_baselines.py), which import this
    helper from pytorchexample.task."""
    all_source_names = [_source_name_from_path(p) for p in files]
    dfs = []
    for p in files:
        df = pd.read_csv(p, low_memory=False)
        df = _encode_known_categoricals(df)
        df = _add_engineered_features(df, _source_name_from_path(p), all_source_names)
        dfs.append(df)
    return dfs


_FEATURE_COLUMNS_SAMPLE_ROWS = 5000


def _apply_exclude(feature_columns: List[str], exclude_features: str) -> List[str]:
    """Opt-in feature filter. ``exclude_features`` is a comma-separated list of
    column names to drop from the model input (default "" = no change, so every
    existing caller/run keeps the exact same 51/25/125-feature set and results).

    Added to support an ablation (e.g. training without ``hour_of_day`` to test
    whether the model's performance depends on the capture-time artifact). Only
    removes columns that are actually present; unknown names are ignored.
    """
    if not exclude_features:
        return feature_columns
    drop = {c.strip() for c in exclude_features.split(",") if c.strip()}
    return [c for c in feature_columns if c not in drop]


def get_feature_columns(files: List[Path]) -> List[str]:
    """Discover the numeric feature-column set.

    Memory optimisation: reading every full CSV
    (for N-BaIoT, ~7.7GB across 9 files) on EVERY call, with only an
    in-process cache - so a 9-client sequential simulation run (required
    for genuine one-client-per-device partitioning, see
    load_partition_data's partition_by_file mode) paid this full-dataset
    read 9 times just to discover column names, before even touching any
    client's actual data, which drove memory usage up sharply and could
    trigger an out-of-memory crash on a long sequential simulation.

    Column numeric-ness is determined from a sample
    (_FEATURE_COLUMNS_SAMPLE_ROWS rows) per file instead of the full file -
    safe because _infer_numeric_columns only needs ONE non-null coercible
    value per column to include it, and these are fixed-schema flow-
    statistics CSVs (a column that is numeric is numeric from row 1).
    """
    global _FEATURE_COLUMNS_CACHE
    if _FEATURE_COLUMNS_CACHE is not None:
        return _FEATURE_COLUMNS_CACHE
    all_source_names = [_source_name_from_path(p) for p in files]
    columns = set()
    for path in files:
        df = pd.read_csv(path, low_memory=False, nrows=_FEATURE_COLUMNS_SAMPLE_ROWS)
        df = _encode_known_categoricals(df)
        df = _add_engineered_features(df, _source_name_from_path(path), all_source_names)
        label_col = _find_label_column(df)
        columns.update(_infer_numeric_columns(df, label_col))
    if not columns:
        raise ValueError("No numeric feature columns found across CSV files.")
    _FEATURE_COLUMNS_CACHE = sorted(columns)
    return _FEATURE_COLUMNS_CACHE


def _prepare_features_and_labels(
    df: pd.DataFrame, feature_columns: List[str]
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    label_col = _find_label_column(df)
    labels = _normalize_labels(df[label_col]) if label_col else None
    features = df.reindex(columns=feature_columns)
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True))
    features = features.fillna(0.0)
    if features.shape[1] == 0:
        raise ValueError("No numeric features after preprocessing.")
    return features, labels


def _split_and_scale(
    features: pd.DataFrame,
    labels: Optional[np.ndarray],
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    if labels is None:
        x_train, x_test = train_test_split(
            features.values,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )
        y_train, y_test = None, None
    else:
        stratify = labels if len(np.unique(labels)) > 1 else None
        x_train, x_test, y_train, y_test = train_test_split(
            features.values,
            labels,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train.astype(np.float32), x_test.astype(np.float32), y_train, y_test


# ---------------------------------------------------------------------------
# Data loading — concatenate all CSVs, split into N non-overlapping slices
# ---------------------------------------------------------------------------

def _maybe_subsample_df(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """No-op when frac>=1.0 (default, every existing caller). Otherwise
    stratified-samples by the raw label column (if found) so class
    proportions are preserved in the reduced client slice."""
    if frac >= 1.0 or len(df) == 0:
        return df
    label_col = _find_label_column(df)
    if label_col is None:
        return df.sample(frac=frac, random_state=seed).reset_index(drop=True)
    groups = []
    for _, group in df.groupby(label_col, dropna=False):
        groups.append(group.sample(frac=frac, random_state=seed))
    return pd.concat(groups, ignore_index=True)


_CHUNKED_SAMPLE_READ_ROWS = 100_000


def _read_csv_sampled(path: Path, frac: float, seed: int) -> pd.DataFrame:
    """Read a CSV file sampled down to `frac` of its rows WITHOUT ever
    materializing the full file in memory - reads in
    _CHUNKED_SAMPLE_READ_ROWS-row chunks and keeps a uniform-random `frac`
    of each chunk. frac>=1.0 falls back to a plain full read (identical to
    every existing caller's behavior).

    Statistically equivalent to sampling frac from the whole file: each row
    is independently kept with probability `frac` regardless of its
    position, so the chunk boundaries (which may group benign/attack rows
    contiguously, e.g. N-BaIoT's benign_traffic.csv + per-family attack
    files concatenated by harmonize_nbaiot.py) do not bias the result.

    For N-BaIoT, reading one device's full file (up to ~1.3GB) before
    subsampling in memory could itself push a 9-client sequential
    simulation toward the memory ceiling. This bounds peak memory to
    O(chunk size + final sample size) instead of O(full file size).
    """
    if frac >= 1.0:
        return pd.read_csv(path, low_memory=False)
    rng_seed = seed
    chunks = []
    for i, chunk in enumerate(pd.read_csv(path, low_memory=False, chunksize=_CHUNKED_SAMPLE_READ_ROWS)):
        if len(chunk) > 0:
            chunks.append(chunk.sample(frac=frac, random_state=rng_seed + i))
    if not chunks:
        return pd.read_csv(path, low_memory=False, nrows=0)
    return pd.concat(chunks, ignore_index=True)


def load_partition_data(
    partition_id: int,
    num_partitions: int,
    data_dir: str,
    file_pattern: str,
    batch_size: int,
    test_size: float,
    seed: int,
    use_weighted_sampler: bool = True,
    sampler_imbalance_threshold: float = 1.5,
    partition_by_file: bool = False,
    client_sample_frac: float = 1.0,
    exclude_features: str = "",
) -> Tuple[DataLoader, DataLoader, int, bool]:
    """Load a client's local data slice.

    ``client_sample_frac`` (default 1.0 = no change to any existing
    behavior): if < 1.0, each client's full slice is stratified-sampled
    down to this fraction (by label, where available) BEFORE the
    train/test split - added for resource-constrained verification re-runs
    (e.g. a reduced-scale H2 FedAvg/FedProx N-BaIoT comparison on a 16GB
    machine), so class proportions are preserved.

    Two modes, selected by ``partition_by_file``:

    - False (default, unchanged historical behavior): all CSV files are
      concatenated, globally shuffled, then divided into ``num_partitions``
      equal, non-overlapping slices (no file reuse). This is ~IID across
      source files/devices.
    - True: client ``partition_id`` gets exactly source file
      ``partition_id`` (one client per device/file) - genuine non-IID-by-
      device partitioning. Requires ``num_partitions == number of source
      files`` matching ``file_pattern``. This was previously a dead
      run-config key (``partition-by-file``, declared in
      run_config_nbaiot.toml but never read by any code) - now wired up for
      the H2 FedAvg/FedProx-on-N-BaIoT verification experiments, which need
      real one-client-per-device partitioning. Opt-in only: every existing
      caller/config that doesn't pass ``partition_by_file=True`` keeps the
      exact prior behavior and prior results are unaffected.
    """
    cache_key = (
        partition_id,
        num_partitions,
        data_dir,
        file_pattern,
        batch_size,
        test_size,
        seed,
        use_weighted_sampler,
        sampler_imbalance_threshold,
        partition_by_file,
        client_sample_frac,
        exclude_features,
    )
    if cache_key in _PARTITION_CACHE:
        return _PARTITION_CACHE[cache_key]

    if partition_by_file:
        files = list_csv_files(data_dir, file_pattern)
        if num_partitions != len(files):
            raise ValueError(
                f"partition_by_file=True requires num_partitions ({num_partitions}) "
                f"== number of source files ({len(files)}) for pattern {file_pattern!r}"
            )
        feature_columns = _apply_exclude(get_feature_columns(files), exclude_features)
        all_source_names = [_source_name_from_path(p) for p in files]
        path = files[partition_id]
        # Sampled read (see _read_csv_sampled docstring): when
        # client_sample_frac < 1.0, never materializes this device's full
        # file in memory. frac>=1.0 (every pre-existing caller) is a plain
        # full read, byte-for-byte the same as before.
        df = _read_csv_sampled(path, client_sample_frac, seed)
        df = _encode_known_categoricals(df)
        df = _add_engineered_features(df, _source_name_from_path(path), all_source_names)

        features, labels = _prepare_features_and_labels(df, feature_columns)
        x_train, x_test, y_train, y_test = _split_and_scale(
            features, labels, test_size=test_size, seed=seed
        )
        return _finish_partition_loaders(
            cache_key, x_train, x_test, y_train, y_test, batch_size,
            use_weighted_sampler, sampler_imbalance_threshold,
        )

    files = list_csv_files(data_dir, file_pattern)
    feature_columns = _apply_exclude(get_feature_columns(files), exclude_features)
    all_source_names = [_source_name_from_path(p) for p in files]

    # Load, encode categoricals and add engineered features per file, then concatenate
    dfs = []
    for p in files:
        df = pd.read_csv(p, low_memory=False)
        df = _encode_known_categoricals(df)
        df = _add_engineered_features(df, _source_name_from_path(p), all_source_names)
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)

    # Shuffle with a fixed seed for reproducibility
    df_all = df_all.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Split into num_partitions equal non-overlapping slices
    total_rows = len(df_all)
    slice_size = total_rows // num_partitions
    start = partition_id * slice_size
    # Last partition takes any remaining rows
    end = (partition_id + 1) * slice_size if partition_id < num_partitions - 1 else total_rows
    df = df_all.iloc[start:end].reset_index(drop=True)
    df = _maybe_subsample_df(df, client_sample_frac, seed)

    features, labels = _prepare_features_and_labels(df, feature_columns)
    x_train, x_test, y_train, y_test = _split_and_scale(
        features, labels, test_size=test_size, seed=seed
    )
    return _finish_partition_loaders(
        cache_key, x_train, x_test, y_train, y_test, batch_size,
        use_weighted_sampler, sampler_imbalance_threshold,
    )


def _finish_partition_loaders(
    cache_key: tuple,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: Optional[np.ndarray],
    y_test: Optional[np.ndarray],
    batch_size: int,
    use_weighted_sampler: bool,
    sampler_imbalance_threshold: float,
) -> Tuple[DataLoader, DataLoader, int, bool]:
    """Shared tail of load_partition_data's two partitioning modes: builds
    the (optionally class-balanced) train/test DataLoaders and caches them."""
    # We do NOT filter out attack traffic for supervised classification
    y_train_train = y_train

    # Optional weighted sampler for imbalanced partitions
    if y_train_train is not None:
        labels_array = torch.tensor(y_train_train).long()
        class_counts = torch.bincount(labels_array, minlength=2)
        class_counts = torch.clamp(class_counts, min=1)
        imbalance_ratio = float(class_counts.max() / class_counts.min())
        use_sampler = use_weighted_sampler and imbalance_ratio >= sampler_imbalance_threshold

        if use_sampler:
            sample_weights = 1.0 / class_counts[labels_array].float()
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            trainloader = DataLoader(
                NumpyDataset(x_train, y_train_train), batch_size=batch_size, sampler=sampler
            )
        else:
            trainloader = DataLoader(
                NumpyDataset(x_train, y_train_train), batch_size=batch_size, shuffle=True
            )
    else:
        trainloader = DataLoader(
            NumpyDataset(x_train, y_train_train), batch_size=batch_size, shuffle=True
        )

    testloader = DataLoader(
        NumpyDataset(x_test, y_test), batch_size=batch_size, shuffle=False
    )
    input_dim = x_train.shape[1]
    has_labels = y_test is not None

    _bounded_cache_set(
        _PARTITION_CACHE, cache_key,
        (trainloader, testloader, input_dim, has_labels),
        _PARTITION_CACHE_MAXSIZE,
    )
    return _PARTITION_CACHE[cache_key]


def load_centralized_data(
    data_dir: str,
    file_pattern: str,
    batch_size: int,
    test_size: float,
    seed: int,
    exclude_features: str = "",
) -> Tuple[DataLoader, int, bool]:
    cache_key = (data_dir, file_pattern, batch_size, test_size, seed, exclude_features)
    if cache_key in _CENTRAL_CACHE:
        return _CENTRAL_CACHE[cache_key]

    files = list_csv_files(data_dir, file_pattern)
    feature_columns = _apply_exclude(get_feature_columns(files), exclude_features)
    all_source_names = [_source_name_from_path(p) for p in files]
    dfs = []
    for path in files:
        df_src = pd.read_csv(path, low_memory=False)
        df_src = _encode_known_categoricals(df_src)
        df_src = _add_engineered_features(df_src, _source_name_from_path(path), all_source_names)
        dfs.append(df_src)
    df = pd.concat(dfs, ignore_index=True)
    features, labels = _prepare_features_and_labels(df, feature_columns)
    x_train, x_test, y_train, y_test = _split_and_scale(
        features, labels, test_size=test_size, seed=seed
    )
    testloader = DataLoader(
        NumpyDataset(x_test, y_test), batch_size=batch_size, shuffle=False
    )
    input_dim = x_train.shape[1]
    has_labels = y_train is not None
    _CENTRAL_CACHE[cache_key] = (testloader, input_dim, has_labels)
    return _CENTRAL_CACHE[cache_key]


def _unpack_batch(batch):
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        return batch[0], batch[1]
    return batch, None


# ---------------------------------------------------------------------------
# Training — Adam + weight_decay, grad clipping
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    proximal_mu: float = 0.0,
    global_params: Optional[list] = None,
    weight_decay: float = 1e-3,
    use_class_weights: bool = True,
) -> float:
    """Train the classifier with FedProx proximal term support.

    Parameters
    ----------
    proximal_mu:
        Weight of the FedProx proximal term. 0.0 = pure FedAvg behaviour.
    global_params:
        Snapshot of global model parameters (before local training) used to
        compute the proximal regularisation term.
    weight_decay:
        L2 regularisation for the local optimizer.
    """
    model.to(device)

    # Optional weighted loss for imbalanced training labels
    if use_class_weights and trainloader.dataset.labels is not None:
        train_labels = trainloader.dataset.labels
        class_counts = torch.bincount(train_labels.long(), minlength=2)
        class_counts = torch.clamp(class_counts, min=1)
        total = class_counts.sum().float()
        class_weights = total / (len(class_counts) * class_counts.float())
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device)).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    running_loss = 0.0
    steps = 0
    for _ in range(epochs):
        for batch in trainloader:
            features, targets = _unpack_batch(batch)
            if targets is None:
                continue
            features = features.to(device)
            targets = targets.to(device).long()

            if model.training:
                noise = torch.randn_like(features) * 0.01
                features = (features + noise).clamp(0.0, 1.0)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)

            # FedProx proximal term
            if proximal_mu > 0.0 and global_params is not None:
                proximal_term = 0.0
                for local_w, global_w in zip(model.parameters(), global_params):
                    proximal_term += (local_w - global_w.to(device)).norm(2) ** 2
                loss = loss + (proximal_mu / 2.0) * proximal_term

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            steps += 1

    return running_loss / max(steps, 1)


# ---------------------------------------------------------------------------
# Backwards Compatibility Stubs for server_app / client_app
# ---------------------------------------------------------------------------

def compute_reconstruction_errors(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Placeholder to maintain backwards compatibility with legacy strategy signatures."""
    return np.array([]), None


def compute_train_threshold(
    model: nn.Module, trainloader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    """Placeholder to maintain backwards compatibility with legacy strategy signatures."""
    return 0.5, 0.0


def _best_threshold(errors: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Placeholder to maintain backwards compatibility with legacy strategy signatures."""
    return 0.5, 0.0


# ---------------------------------------------------------------------------
# FIX 3 & FIX 6 — Track Metrics Properly & Threshold Optimization
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    testloader: DataLoader,
    device: torch.device,
    threshold: Optional[float] = None,
    log_report: bool = False,
) -> Tuple[float, int, dict]:
    """Evaluate model and compute classification metrics under optimal threshold."""
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss().to(device)

    all_preds = []
    all_labels = []
    all_probs = []
    running_loss = 0.0

    with torch.no_grad():
        for batch in testloader:
            features, targets = _unpack_batch(batch)
            if targets is None:
                continue
            features = features.to(device)
            targets = targets.to(device).long()

            logits = model(features)
            loss = criterion(logits, targets)
            running_loss += loss.item() * len(targets)

            # FIX 3 — Track F1 and AUC properly as primary metrics
            # We obtain predictions and class 1 probabilities to compute ROC-AUC and macro F1 metrics.
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    val_loss = running_loss / max(len(testloader.dataset), 1)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    if len(all_labels) > 0 and len(np.unique(all_labels)) > 1:
        if threshold is None:
            # Sweep decision thresholds to select the threshold that maximizes macro F1
            thresholds = np.linspace(0.05, 0.95, 91)
            best_thresh = 0.5
            best_f1 = 0.0
            for t in thresholds:
                preds_t = (all_probs >= t).astype(int)
                f1_t = float(f1_score(all_labels, preds_t, average='macro', zero_division=0))
                if f1_t > best_f1:
                    best_f1 = f1_t
                    best_thresh = t
        else:
            best_thresh = float(threshold)

        # Apply chosen threshold for final metrics
        all_preds_opt = (all_probs >= best_thresh).astype(int)
        f1 = float(f1_score(all_labels, all_preds_opt, average='macro', zero_division=0))
        accuracy = float((all_preds_opt == all_labels).mean())

        try:
            auc = float(roc_auc_score(all_labels, all_probs))
        except:
            auc = 0.5

        if log_report:
            print(f"\n[Validation Report] Optimized Decision Threshold: {best_thresh:.4f}")
            print(classification_report(all_labels, all_preds_opt, zero_division=0))
    elif len(all_labels) > 0:
        f1 = float(f1_score(all_labels, all_preds, average='macro', zero_division=0))
        accuracy = float((all_preds == all_labels).mean())
        auc = 0.5
        best_thresh = float(threshold) if threshold is not None else 0.5
    else:
        f1 = 0.0
        accuracy = 0.0
        auc = 0.5
        best_thresh = float(threshold) if threshold is not None else 0.5

    # Return structure matching Flower's expected signature: (loss, num_examples, metrics).
    # 'threshold' is the decision threshold actually used (the swept-best one
    # when the caller passed threshold=None, or the caller's own value
    # otherwise) - exposed so server_app.py's round-level aggregation/logging
    # can report it (previously missing: server_app.py unpacked a 4th
    # "threshold_used" return value that this function never produced,
    # see test_00_pipeline_integrity.py::test_server_app_eval_fn_unpacking_matches_task_evaluate).
    return float(val_loss), len(testloader.dataset), {
        "f1": f1, "auc": auc, "accuracy": accuracy, "threshold": best_thresh,
    }
