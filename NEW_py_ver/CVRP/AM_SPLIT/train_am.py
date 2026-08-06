#!/usr/bin/env python
"""Train the official capacity-aware Attention Model on CVRP."""

import argparse
from pathlib import Path

from options import get_options
from run import run


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DATASETS = {
    50: THIS_DIR.parents[2] / "reproduction_data" / "fixed_testsets" / "cvrp50_n10000_seed1234.pt",
    100: THIS_DIR.parent / "vrp100_test_seed1234.pt",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-size", type=int, choices=(20, 50, 100), default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--epoch-size", type=int, default=1280000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--val-size", type=int, default=10000)
    parser.add_argument("--val-dataset", default=None)
    parser.add_argument("--checkpoint-epochs", type=int, default=10)
    parser.add_argument("--baseline", choices=("rollout", "exponential", "none"), default="rollout")
    parser.add_argument("--run-name", default="am")
    parser.add_argument("--output-dir", default=str(THIS_DIR / "outputs"))
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--metrics-log-interval", type=int, default=1)
    parser.add_argument("--metrics-flush-interval", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    val_dataset = args.val_dataset
    if val_dataset is None and args.graph_size in DEFAULT_DATASETS:
        candidate = DEFAULT_DATASETS[args.graph_size]
        if candidate.exists():
            val_dataset = str(candidate)

    baseline = None if args.baseline == "none" else args.baseline
    cli = [
        "--problem", "cvrp",
        "--graph_size", str(args.graph_size),
        "--n_epochs", str(args.epochs),
        "--epoch_size", str(args.epoch_size),
        "--batch_size", str(args.batch_size),
        "--val_size", str(args.val_size),
        "--eval_batch_size", str(min(1024, args.val_size)),
        "--checkpoint_epochs", str(args.checkpoint_epochs),
        "--metrics_log_interval", str(args.metrics_log_interval),
        "--metrics_flush_interval", str(args.metrics_flush_interval),
        "--run_name", args.run_name,
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
    if args.no_cuda:
        cli.append("--no_cuda")
    if args.no_tensorboard:
        cli.append("--no_tensorboard")
    if args.no_progress_bar:
        cli.append("--no_progress_bar")
    run(get_options(cli))


if __name__ == "__main__":
    main()
