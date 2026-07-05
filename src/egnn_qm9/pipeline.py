import argparse
import json

from .config import ExperimentConfig
from .data import prepare_splits, summarize_batch_targets, write_split_metadata
from .plotting import plot_energy_parity, plot_rotation_invariance, plot_training_curve
from .training import train_model, write_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QM9 energy prediction with e3nn tensor-product message passing")
    subparsers = parser.add_subparsers(dest="command", required=False)
    subparsers.add_parser("prepare-data", help="Prepare QM9 subset splits for energy prediction")
    subparsers.add_parser("train", help="Train the tensor-product energy model")
    subparsers.add_parser("full-pipeline", help="Prepare splits, train, and write plots/metrics")
    return parser


def bundle_with_stats(config: ExperimentConfig) -> dict:
    bundle = prepare_splits(config.dataset)
    bundle["target_stats"] = summarize_batch_targets(bundle["loaders"]["train"], config.dataset.target_index)
    return bundle


def command_prepare_data(config: ExperimentConfig) -> None:
    bundle = bundle_with_stats(config)
    metadata = {**bundle["metadata"], "target_stats": bundle["target_stats"]}
    write_split_metadata(config.dataset, metadata)
    print(json.dumps(metadata, indent=2))


def command_train(config: ExperimentConfig) -> None:
    bundle = bundle_with_stats(config)
    metrics = train_model(config, bundle)
    write_metrics(config.output.metrics_path, metrics)
    plot_training_curve(metrics["history"], config.output.figures_dir / config.output.training_curve_filename)
    plot_energy_parity(metrics["references"], metrics["predictions"], config.output.figures_dir / config.output.parity_filename)
    plot_rotation_invariance(metrics["rotation_errors"], config.output.figures_dir / config.output.equivariance_filename)
    print(f"Wrote metrics to {config.output.metrics_path}")


def command_full_pipeline(config: ExperimentConfig) -> None:
    command_prepare_data(config)
    command_train(config)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = ExperimentConfig()
    command = args.command or "full-pipeline"

    if command == "prepare-data":
        command_prepare_data(config)
    elif command == "train":
        command_train(config)
    elif command == "full-pipeline":
        command_full_pipeline(config)
    else:
        raise ValueError(f"Unknown command {command}")
