"""pytorchexample: A Flower / PyTorch app."""

import torch

# Limit CPU threads to avoid WinError 1450 resource exhaustion on Windows
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import (
    Net,
    evaluate as eval_fn,
    load_partition_data,
    train as train_fn,
)

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the classifier on local data with FedProx proximal term."""

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data_dir = context.run_config["data-dir"]
    file_pattern = context.run_config["file-pattern"]
    batch_size = context.run_config["batch-size"]
    test_size = context.run_config.get("test-size", 0.2)
    seed = context.run_config.get("seed", 42)
    use_weighted_sampler = bool(context.run_config.get("use-weighted-sampler", True))
    sampler_imbalance_threshold = float(
        context.run_config.get("sampler-imbalance-threshold", 1.5)
    )
    use_class_weights = bool(context.run_config.get("use-class-weights", True))
    partition_by_file = bool(context.run_config.get("partition-by-file", False))
    client_sample_frac = float(context.run_config.get("client-sample-frac", 1.0))
    exclude_features = str(context.run_config.get("exclude-features", ""))

    trainloader, _, input_dim, _ = load_partition_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        data_dir=data_dir,
        file_pattern=file_pattern,
        batch_size=batch_size,
        test_size=test_size,
        seed=seed,
        use_weighted_sampler=use_weighted_sampler,
        sampler_imbalance_threshold=sampler_imbalance_threshold,
        partition_by_file=partition_by_file,
        client_sample_frac=client_sample_frac,
        exclude_features=exclude_features,
    )

    model = Net(input_dim)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # FedProx: accept both underscore and hyphen config keys.
    # Fall back to run_config value if not present (e.g. during testing).
    proximal_mu = float(
        msg.content["config"].get(
            "proximal_mu",
            msg.content["config"].get(
                "proximal-mu",
                context.run_config.get("proximal-mu", 0.1),
            ),
        )
    )
    weight_decay = float(
        msg.content["config"].get(
            "weight_decay",
            msg.content["config"].get(
                "weight-decay",
                context.run_config.get("weight-decay", 1e-3),
            ),
        )
    )
    # Snapshot global params BEFORE local training for proximal term computation
    global_params = [p.clone().detach() for p in model.parameters()] if proximal_mu > 0.0 else None

    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
        proximal_mu=proximal_mu,
        global_params=global_params,
        weight_decay=weight_decay,
        use_class_weights=use_class_weights,
    )

    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the classifier."""

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data_dir = context.run_config["data-dir"]
    file_pattern = context.run_config["file-pattern"]
    batch_size = context.run_config["batch-size"]
    test_size = context.run_config.get("test-size", 0.2)
    seed = context.run_config.get("seed", 42)
    use_weighted_sampler = bool(context.run_config.get("use-weighted-sampler", True))
    sampler_imbalance_threshold = float(
        context.run_config.get("sampler-imbalance-threshold", 1.5)
    )
    partition_by_file = bool(context.run_config.get("partition-by-file", False))
    client_sample_frac = float(context.run_config.get("client-sample-frac", 1.0))
    exclude_features = str(context.run_config.get("exclude-features", ""))

    trainloader, testloader, input_dim, has_labels = load_partition_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        data_dir=data_dir,
        file_pattern=file_pattern,
        batch_size=batch_size,
        test_size=test_size,
        seed=seed,
        use_weighted_sampler=use_weighted_sampler,
        sampler_imbalance_threshold=sampler_imbalance_threshold,
        partition_by_file=partition_by_file,
        client_sample_frac=client_sample_frac,
        exclude_features=exclude_features,
    )

    model = Net(input_dim)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluate using the updated signature
    eval_loss, num_examples, metrics_dict = eval_fn(
        model, testloader, device, threshold=None, log_report=False
    )

    metrics = {
        "loss": eval_loss,
        "accuracy": metrics_dict["accuracy"],
        "f1": metrics_dict["f1"],
        "auc": metrics_dict["auc"],
        "threshold": metrics_dict["threshold"],
        "has-labels": int(has_labels),
        "num-examples": num_examples,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
