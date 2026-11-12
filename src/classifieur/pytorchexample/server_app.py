"""pytorchexample: A Flower / PyTorch app."""

import torch

# Limit CPU threads to avoid WinError 1450 resource exhaustion on Windows
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedProx

from pytorchexample.task import Autoencoder, evaluate as eval_fn, load_centralized_data

# Create ServerApp
app = ServerApp()


# ---------------------------------------------------------------------------
# Weighted threshold aggregation from clients
# ---------------------------------------------------------------------------

def aggregate_eval_metrics(
    record_dicts: list[RecordDict], weight_key: str
) -> MetricRecord:
    """Aggregate evaluation metrics using weighted average.

    The threshold reported by each client is weighted by the number of
    test examples on that client, producing a globally consistent threshold
    that matches the distribution seen during local training.
    """
    total_weight = 0.0
    weighted_loss = 0.0
    weighted_acc = 0.0
    weighted_threshold = 0.0
    weighted_std = 0.0

    for rd in record_dicts:
        metrics = rd["metrics"]
        weight = float(metrics.get(weight_key, metrics.get("num-examples", 1)))
        total_weight += weight
        weighted_loss += float(metrics.get("loss", 0.0)) * weight
        weighted_acc += float(metrics.get("accuracy", 0.0)) * weight
        weighted_threshold += float(metrics.get("threshold", 0.0)) * weight
        weighted_std += float(metrics.get("std", 0.0)) * weight

    if total_weight == 0:
        return MetricRecord({"loss": 0.0, "accuracy": 0.0, "threshold": 0.0, "std": 0.0})

    return MetricRecord(
        {
            "loss": weighted_loss / total_weight,
            "accuracy": weighted_acc / total_weight,
            "threshold": weighted_threshold / total_weight,
            "std": weighted_std / total_weight,
        }
    )


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    weight_decay: float = float(context.run_config.get("weight-decay", 1e-3))
    data_dir: str = context.run_config["data-dir"]
    file_pattern: str = context.run_config["file-pattern"]
    batch_size: int = context.run_config["batch-size"]
    test_size: float = context.run_config.get("test-size", 0.2)
    seed: int = context.run_config.get("seed", 42)

    # Read proximal-mu and early-stopping params from config
    proximal_mu: float = float(context.run_config.get("proximal-mu", 0.1))
    patience: int = int(context.run_config.get("early-stopping-patience", 3))
    warmup: int = int(context.run_config.get("early-stopping-warmup", 5))

    testloader, input_dim, has_labels = load_centralized_data(
        data_dir=data_dir,
        file_pattern=file_pattern,
        batch_size=batch_size,
        test_size=test_size,
        seed=seed,
        exclude_features=str(context.run_config.get("exclude-features", "")),
    )

    global_model = Autoencoder(input_dim)
    arrays = ArrayRecord(global_model.state_dict())

    # Mutable state shared across closures
    _state = {
        "aggregated_threshold": None,   # cached from client aggregation
        "best_acc": 0.0,                # best global accuracy seen
        "no_improve_count": 0,          # consecutive rounds without improvement
        "stop": False,                  # early-stopping flag
    }

    # -----------------------------------------------------------------------
    # Checkpoint diagnostics — printed every round
    # -----------------------------------------------------------------------

    def _diagnose(server_round: int, loss: float, accuracy: float, threshold: float) -> None:
        """Print per-round diagnostic status matching the monitoring format."""
        # Determine status
        prev_acc = _state["best_acc"]
        if accuracy > prev_acc + 0.001:
            status = "IMPROVING"
        elif abs(accuracy - prev_acc) <= 0.001 and server_round > 4:
            status = "CONVERGED"
        elif loss < 0.001 and accuracy < prev_acc - 0.002:
            status = "OVERFITTING"
        else:
            status = "STABLE"

        # Recommended action
        action = "None"
        priority = "OPTIONAL"
        if threshold < 0.005:
            action = "learning_rate=0.0001  # threshold collapse → reduce lr"
            priority = "URGENT"
        elif loss < 0.001 and accuracy < prev_acc:
            action = f"proximal_mu={min(proximal_mu * 2, 0.5):.3f}  # overfitting → double mu"
            priority = "URGENT"
        elif status == "CONVERGED":
            action = "early_stop=True  # accuracy stable"
            priority = "MODERATE"

        print(
            f"\n{'='*60}\n"
            f"Round: {server_round}\n"
            f"Status: {status}\n"
            f"  loss={loss:.6f}  acc={accuracy:.4f}  threshold={threshold:.6f}\n"
            f"Recommended change: {action}\n"
            f"Priority: {priority}\n"
            f"{'='*60}"
        )

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate global model using the client-aggregated threshold."""
        model = Autoencoder(input_dim)
        model.load_state_dict(arrays.to_torch_state_dict())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Use threshold aggregated from clients (avoids 5× mismatch)
        agg_threshold = _state["aggregated_threshold"]
        loss, accuracy, threshold_used, std = eval_fn(
            model, testloader, device, threshold=agg_threshold
        )

        # Early stopping: track consecutive non-improving rounds after warmup
        if accuracy > _state["best_acc"] + 0.001:
            _state["best_acc"] = accuracy
            if server_round > warmup:
                _state["no_improve_count"] = 0
        else:
            if server_round > warmup:
                _state["no_improve_count"] += 1

        if server_round > warmup and _state["no_improve_count"] >= patience:
            _state["stop"] = True
            print(
                f"\n[EARLY STOP] Global accuracy has not improved for "
                f"{patience} consecutive rounds (best={_state['best_acc']:.4f}). "
                f"Stopping at round {server_round}."
            )

        # Per-round checkpoint diagnostics
        if server_round % 5 == 0 or server_round == 1 or _state["stop"]:
            _diagnose(server_round, loss, accuracy, threshold_used)

        metrics = {
            "loss": loss,
            "accuracy": accuracy,
            "threshold": threshold_used,
            "std": std,
            "has-labels": int(has_labels),
        }
        return MetricRecord(metrics)

    def custom_evaluate_aggr(
        record_dicts: list[RecordDict], weight_key: str
    ) -> MetricRecord:
        """Aggregate client evaluation metrics and cache the weighted threshold."""
        result = aggregate_eval_metrics(record_dicts, weight_key)
        _state["aggregated_threshold"] = float(result["threshold"])
        agg_acc = float(result["accuracy"])
        agg_thr = _state["aggregated_threshold"]
        print(
            f"  [Client aggregation] "
            f"acc={agg_acc:.4f}  threshold={agg_thr:.6f}"
        )
        return result

    # -----------------------------------------------------------------------
    # FedProx strategy — mu and node counts from config
    # -----------------------------------------------------------------------
    strategy = FedProx(
        fraction_train=1.0,
        fraction_evaluate=1.0,
        min_train_nodes=7,
        min_evaluate_nodes=7,
        min_available_nodes=7,
        proximal_mu=proximal_mu,
        evaluate_metrics_aggr_fn=custom_evaluate_aggr,
    )

    print(
        f"\n[Config] proximal_mu={proximal_mu}  lr={lr}  weight_decay={weight_decay}  "
        f"patience={patience}  warmup={warmup}"
    )

    # Run federated learning — Flower's strategy.start() does not support
    # early stopping natively, so we run it for the full num_rounds but
    # print the EARLY STOP message so the operator can interrupt if needed.
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr, "weight-decay": weight_decay}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
