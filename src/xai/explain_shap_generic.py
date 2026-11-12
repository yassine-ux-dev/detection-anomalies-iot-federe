"""Step 6.6 — Generic per-dataset SHAP explainability.

Unlike explain_shap.py (which hardcodes TON-IoT's own feature-engineering
rules for its data loading), this script reuses task.py's actual
get_feature_columns/_load_engineered_files/_prepare_features_and_labels/
_split_and_scale pipeline -- the exact same code path used for training and
for evaluate_checkpoint.py -- so the feature set/order always matches what
each checkpoint was trained on, whichever of TON-IoT/N-BaIoT/BoT-IoT it is.

Usage (from projet/):
    python explain_shap_generic.py --dataset nbaiot \
        --checkpoint final_model_nbaiot.pt \
        --data-dir $IOT_DATA/Processed_NBaIoT_dataset \
        --file-pattern "NBaIoT_*.csv" \
        --output-dir results/shap_nbaiot

    python explain_shap_generic.py --dataset botiot \
        --checkpoint final_model_botiot.pt \
        --data-dir $IOT_DATA/Processed_BoT_IoT_dataset \
        --file-pattern "BoT_IoT_processed.csv" \
        --output-dir results/shap_botiot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn

from pytorchexample.task import (
    Net,
    list_csv_files,
    get_feature_columns,
    _load_engineered_files,
    _prepare_features_and_labels,
    _split_and_scale,
)

ROOT = Path(__file__).resolve().parents[2]


class SoftmaxWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model(x), dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Label used in printed output, e.g. 'N-BaIoT'")
    parser.add_argument("--checkpoint", required=True, help="Path to the .pt state_dict, relative to this file")
    parser.add_argument("--data-dir", required=True, help="Path to the harmonized CSVs, relative to this file")
    parser.add_argument("--file-pattern", required=True, help="Glob pattern for the CSVs")
    parser.add_argument("--output-dir", required=True, help="Directory to write SHAP plots/CSV, relative to this file")
    parser.add_argument("--background-size", type=int, default=200)
    parser.add_argument("--test-sample-size", type=int, default=200)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_csv_files(str(ROOT / args.data_dir), args.file_pattern)
    feature_columns = get_feature_columns(files)
    df_all = pd.concat(_load_engineered_files(files), ignore_index=True)
    df_all = df_all.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    features, labels = _prepare_features_and_labels(df_all, feature_columns)
    if labels is None:
        raise ValueError("Could not find a label column in the harmonized CSVs.")

    _, x_test, _, y_test = _split_and_scale(
        features, labels, test_size=args.test_size, seed=args.seed
    )
    input_dim = x_test.shape[1]
    print(f"[OK] {args.dataset}: input_dim={input_dim}, test rows={len(x_test)}")

    model = Net(input_dim=input_dim)
    state_dict = torch.load(ROOT / args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    rng = np.random.default_rng(args.seed)
    sample_count = len(x_test)
    background_size = min(args.background_size, sample_count)
    test_sample_size = min(args.test_sample_size, max(sample_count - background_size, 1))

    permutation = rng.permutation(sample_count)
    background_indices = permutation[:background_size]
    test_indices = permutation[background_size:background_size + test_sample_size]
    if len(test_indices) == 0:
        test_indices = permutation[:test_sample_size]

    x_background = torch.tensor(x_test[background_indices], dtype=torch.float32, device=device)
    x_samples = torch.tensor(x_test[test_indices], dtype=torch.float32, device=device)
    y_samples = y_test[test_indices] if y_test is not None else None

    wrapped_model = SoftmaxWrapper(model).to(device)
    wrapped_model.eval()
    baseline_values = wrapped_model(x_background).detach().mean(dim=0).cpu().numpy()

    try:
        explainer = shap.DeepExplainer(wrapped_model, x_background)
        shap_values = explainer.shap_values(x_samples)
        print("[OK] DeepExplainer completed successfully")
    except Exception as exc:
        print(f"DeepExplainer failed: {exc}")
        print("Switching to GradientExplainer...")
        explainer = shap.GradientExplainer(wrapped_model, x_background)
        shap_values = explainer.shap_values(x_samples)

    def class_values(values: object, class_index: int = 1) -> np.ndarray:
        if isinstance(values, list):
            return np.asarray(values[class_index])
        arr = np.asarray(values)
        if arr.ndim == 3:
            return arr[:, :, class_index]
        return arr

    def expected_value(explainer_obj: object, class_index: int = 1):
        value = getattr(explainer_obj, "expected_value", None)
        if isinstance(value, (list, np.ndarray)):
            return value[class_index] if len(value) > class_index else value[0]
        if value is None:
            return float(baseline_values[class_index])
        return value

    sv = class_values(shap_values, 1)
    test_array = x_samples.detach().cpu().numpy()

    try:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(sv, test_array, feature_names=feature_columns, show=False)
        plt.title(f"SHAP Summary Plot - {args.dataset}")
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("[SAVED] shap_summary.png")
    except Exception as exc:
        print(f"[ERROR] Summary plot failed: {exc}")

    try:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv, test_array, feature_names=feature_columns, plot_type="bar", show=False)
        plt.title(f"Top Features by Mean |SHAP Value| - {args.dataset}")
        plt.tight_layout()
        plt.savefig(output_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("[SAVED] shap_bar.png")
    except Exception as exc:
        print(f"[ERROR] Bar plot failed: {exc}")

    base_value_1 = float(expected_value(explainer, 1))

    def explain_one_example(idx: int, label: str) -> dict:
        """Build a JSON-serializable, data-driven explanation for one sample
        (label is 'attack' or 'normal') -- feeds generate_xai_report.py's
        narrative text so it's computed from real SHAP output, not hardcoded."""
        feature_values = test_array[idx]
        shap_vals = sv[idx]
        f_x = base_value_1 + float(shap_vals.sum())
        order = np.argsort(-np.abs(shap_vals))
        contributions = [
            {
                "feature": feature_columns[i],
                "value": float(feature_values[i]),
                "shap_value": float(shap_vals[i]),
                "direction": "attack" if shap_vals[i] > 0 else "normal",
            }
            for i in order[:10]
        ]
        return {
            "label": label,
            "true_class": int(y_samples[idx]) if y_samples is not None else None,
            "base_value": base_value_1,
            "f_x": f_x,
            "predicted_class": "attack" if f_x >= 0.5 else "normal",
            "top_contributions": contributions,
        }

    attack_indices = []
    normal_indices = []
    if y_samples is not None:
        attack_indices = np.where(np.asarray(y_samples) == 1)[0].tolist()
        normal_indices = np.where(np.asarray(y_samples) == 0)[0].tolist()

    examples_data = {}

    for label, indices, force_file, waterfall_file in (
        ("attack", attack_indices, "shap_force_attack.png", "shap_waterfall.png"),
        ("normal", normal_indices, "shap_force_normal.png", "shap_waterfall_normal.png"),
    ):
        if not indices:
            print(f"[WARN] No {label} sample found in the selected SHAP test subset")
            continue
        idx = int(indices[0])
        examples_data[label] = explain_one_example(idx, label)

        try:
            plt.figure(figsize=(14, 4))
            shap.force_plot(
                base_value_1,
                sv[idx],
                test_array[idx],
                feature_names=feature_columns,
                matplotlib=True,
                show=False,
            )
            plt.title(f"Force Plot - {label.capitalize()} Sample Explanation ({args.dataset})")
            plt.tight_layout()
            plt.savefig(output_dir / force_file, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[SAVED] {force_file}")
        except Exception as exc:
            print(f"[ERROR] Force plot ({label}) failed: {exc}")

        try:
            explanation = shap.Explanation(
                values=sv[idx],
                base_values=base_value_1,
                data=test_array[idx],
                feature_names=feature_columns,
            )
            plt.figure(figsize=(12, 6))
            shap.waterfall_plot(explanation, show=False)
            plt.title(f"Waterfall Plot - {label.capitalize()} Sample ({args.dataset})")
            plt.tight_layout()
            plt.savefig(output_dir / waterfall_file, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[SAVED] {waterfall_file}")
        except Exception as exc:
            print(f"[ERROR] Waterfall plot ({label}) failed: {exc}")

    if examples_data:
        report_data = {"dataset": args.dataset, "checkpoint": args.checkpoint, "examples": examples_data}
        with open(output_dir / "shap_report_data.json", "w") as f:
            json.dump(report_data, f, indent=2)
        print("[SAVED] shap_report_data.json")

    mean_shap = np.abs(sv).mean(axis=0)
    importance_df = pd.DataFrame(
        {"feature": feature_columns, "mean_abs_shap": mean_shap}
    ).sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv(output_dir / "shap_importance.csv", index=False)

    print(f"\n=== Top 10 features for {args.dataset} ===")
    print(importance_df.head(10).to_string(index=False))
    print(f"\n[SAVED] {output_dir / 'shap_importance.csv'}")


if __name__ == "__main__":
    main()
