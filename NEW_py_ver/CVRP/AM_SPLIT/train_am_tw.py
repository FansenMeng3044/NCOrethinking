#!/usr/bin/env python
"""Train capacity/TW-aware Attention Model on multi-vehicle CVRPTW."""

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

from options import get_options
from run import run


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-size", type=int, choices=(20, 50, 100, 200), default=100)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--epoch-size", type=int, default=1280000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--val-size", type=int, default=10000)
    parser.add_argument("--val-dataset", default=None)
    parser.add_argument("--checkpoint-epochs", type=int, default=10)
    parser.add_argument("--baseline", choices=("rollout", "exponential", "none"), default="rollout")
    parser.add_argument("--depot-end", type=float, default=3.0)
    parser.add_argument("--service-duration", type=float, default=0.2)
    parser.add_argument("--run-name", default="am_tw")
    parser.add_argument("--output-dir", default=str(THIS_DIR / "outputs"))
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.graph_size, args.epochs, args.epoch_size = 20, 1, 8
        args.batch_size, args.val_size, args.checkpoint_epochs = 4, 8, 1
        args.baseline, args.no_cuda, args.no_tensorboard = "exponential", True, True
        args.no_progress_bar = True
    cli = [
        "--problem", "cvrptw",
        "--graph_size", str(args.graph_size),
        "--n_epochs", str(args.epochs),
        "--epoch_size", str(args.epoch_size),
        "--batch_size", str(args.batch_size),
        "--val_size", str(args.val_size),
        "--eval_batch_size", str(min(1024, args.val_size)),
        "--checkpoint_epochs", str(args.checkpoint_epochs),
        "--depot_end", str(args.depot_end),
        "--service_duration", str(args.service_duration),
        "--run_name", args.run_name,
        "--output_dir", args.output_dir,
        "--seed", str(args.seed),
        "--normalization", "batch",
        "--n_encode_layers", "3",
    ]
    if args.baseline != "none":
        cli.extend(("--baseline", args.baseline))
    if args.val_dataset:
        cli.extend(("--val_dataset", args.val_dataset))
    if args.resume:
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
