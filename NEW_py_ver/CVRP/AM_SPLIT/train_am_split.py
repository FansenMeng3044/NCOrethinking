#!/usr/bin/env python
"""Configured training entry point for demand-blind AM + Split."""

import argparse
from pathlib import Path

from options import get_options
from run import run


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DATASETS = {
    50: THIS_DIR.parents[2]
    / "reproduction_data"
    / "fixed_testsets"
    / "cvrp50_n10000_seed1234.pt",
    100: THIS_DIR.parent / "vrp100_test_seed1234.pt",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the official Attention Model as a customer giant-tour policy"
    )
    parser.add_argument("--graph-size", type=int, choices=(20, 50, 100), default=100)
    parser.add_argument("--reward-mode", choices=("split", "raw"), default="split")
    parser.add_argument("--capacity", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--epoch-size", type=int, default=1280000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--val-size", type=int, default=10000)
    parser.add_argument("--val-dataset", type=str, default=None)
    parser.add_argument("--checkpoint-epochs", type=int, default=10)
    parser.add_argument("--baseline", choices=("rollout", "exponential", "none"), default="rollout")
    parser.add_argument("--run-name", default="am_split")
    parser.add_argument("--output-dir", default=str(THIS_DIR / "outputs"))
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--metrics-log-interval", type=int, default=1)
    parser.add_argument("--metrics-flush-interval", type=int, default=50)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    graph_size = args.graph_size
    epochs = args.epochs
    epoch_size = args.epoch_size
    batch_size = args.batch_size
    val_size = args.val_size
    checkpoint_epochs = args.checkpoint_epochs
    baseline = None if args.baseline == "none" else args.baseline
    no_cuda = args.no_cuda
    no_tensorboard = args.no_tensorboard

    if args.smoke:
        graph_size = 20
        epochs = 1
        epoch_size = 8
        batch_size = 4
        val_size = 8
        checkpoint_epochs = 1
        baseline = "exponential"
        no_cuda = True
        no_tensorboard = True

    val_dataset = args.val_dataset
    if val_dataset is None and graph_size in DEFAULT_DATASETS:
        candidate = DEFAULT_DATASETS[graph_size]
        if candidate.exists():
            val_dataset = str(candidate)

    cli = [
        "--problem", "am_split",
        "--graph_size", str(graph_size),
        "--train_reward", args.reward_mode,
        "--capacity", str(args.capacity),
        "--n_epochs", str(epochs),
        "--epoch_size", str(epoch_size),
        "--batch_size", str(batch_size),
        "--val_size", str(val_size),
        "--eval_batch_size", str(min(1024, val_size)),
        "--checkpoint_epochs", str(checkpoint_epochs),
        "--metrics_log_interval", str(args.metrics_log_interval),
        "--metrics_flush_interval", str(args.metrics_flush_interval),
        "--run_name", "{}_{}".format(args.run_name, args.reward_mode),
        "--output_dir", args.output_dir,
        "--seed", str(args.seed),
        "--normalization", "batch",
        "--n_encode_layers", "3",
    ]
    if baseline is not None:
        cli.extend(("--baseline", baseline))
    if val_dataset is not None:
        cli.extend(("--val_dataset", val_dataset))
    if args.resume is not None:
        cli.extend(("--resume", args.resume))
    if no_cuda:
        cli.append("--no_cuda")
    if no_tensorboard:
        cli.append("--no_tensorboard")
    if args.smoke or args.no_progress_bar:
        cli.append("--no_progress_bar")

    run(get_options(cli))


if __name__ == "__main__":
    main()
