#!/usr/bin/env python
"""Evaluate an AM giant-tour checkpoint using exact hard-capacity Split."""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from nets.attention_model import set_decode_type
from problems.am_split.problem_am_split import AMSplit
from utils.functions import load_model, move_to


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = THIS_DIR.parent / "vrp100_test_seed1234.pt"


def augment_xy_by_8(xy):
    x = xy[..., [0]]
    y = xy[..., [1]]
    return torch.cat(
        (
            torch.cat((x, y), dim=-1),
            torch.cat((1 - x, y), dim=-1),
            torch.cat((x, 1 - y), dim=-1),
            torch.cat((1 - x, 1 - y), dim=-1),
            torch.cat((y, x), dim=-1),
            torch.cat((1 - y, x), dim=-1),
            torch.cat((y, 1 - x), dim=-1),
            torch.cat((1 - y, 1 - x), dim=-1),
        ),
        dim=0,
    )


def augment_batch(batch, factor):
    if factor == 1:
        return batch
    if factor != 8:
        raise ValueError("augmentation must be 1 or 8")
    return {
        "depot": augment_xy_by_8(batch["depot"][:, None, :])[:, 0],
        "loc": augment_xy_by_8(batch["loc"]),
        "demand": batch["demand"].repeat(8, 1),
    }


def decode(model, batch, strategy, width, sample_chunk):
    if strategy == "greedy":
        set_decode_type(model, "greedy")
        costs, _, tours = model(batch, return_pi=True)
        return costs, tours

    set_decode_type(model, "sampling")
    best_cost = None
    best_tour = None
    remaining = width
    while remaining > 0:
        current_width = min(sample_chunk, remaining)
        tours, costs = model.sample_many(batch, batch_rep=current_width, iter_rep=1)
        if best_cost is None:
            best_cost, best_tour = costs, tours
        else:
            improve = costs < best_cost
            best_cost = torch.where(improve, costs, best_cost)
            best_tour = torch.where(improve[:, None], tours, best_tour)
        remaining -= current_width
    return best_cost, best_tour


def evaluate(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model, model_args = load_model(args.model, epoch=args.epoch)
    capacity = args.capacity if args.capacity is not None else model_args.get("capacity", 1.0)
    AMSplit.configure(capacity=capacity, train_reward="split")
    model.problem.configure(capacity=capacity, train_reward="split")
    model.to(device)
    model.eval()

    dataset_path = Path(args.dataset).expanduser().resolve() if args.dataset else None
    dataset = AMSplit.make_dataset(
        filename=str(dataset_path) if dataset_path else None,
        size=model_args["graph_size"],
        num_samples=args.num_instances,
        offset=args.offset,
    )
    if len(dataset) == 0:
        raise ValueError("evaluation selection is empty")

    loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)
    no_aug_sum = 0.0
    aug_sum = 0.0
    no_aug_sq_sum = 0.0
    aug_sq_sum = 0.0
    count = 0
    started = time.time()

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            original_batch_size = batch["loc"].size(0)
            batch = augment_batch(move_to(batch, device), args.augmentation)
            costs, _ = decode(
                model,
                batch,
                args.decode_strategy,
                args.width,
                args.sample_chunk,
            )
            cost_matrix = costs.view(args.augmentation, original_batch_size)
            no_aug = cost_matrix[0]
            aug = cost_matrix.min(dim=0).values
            no_aug_sum += no_aug.sum().item()
            aug_sum += aug.sum().item()
            no_aug_sq_sum += no_aug.double().square().sum().item()
            aug_sq_sum += aug.double().square().sum().item()
            count += original_batch_size
            if args.log_every and (batch_index % args.log_every == 0 or count == len(dataset)):
                print(
                    "evaluated {}/{}: no_aug_split={:.6f}, aug_split={:.6f}".format(
                        count, len(dataset), no_aug_sum / count, aug_sum / count
                    )
                )

    no_aug_mean = no_aug_sum / count
    aug_mean = aug_sum / count
    no_aug_se = math.sqrt(max(no_aug_sq_sum / count - no_aug_mean ** 2, 0.0) / count)
    aug_se = math.sqrt(max(aug_sq_sum / count - aug_mean ** 2, 0.0) / count)
    result = {
        "model": str(Path(args.model).expanduser().resolve()),
        "epoch": args.epoch,
        "dataset": str(dataset_path) if dataset_path else "generated",
        "instances": count,
        "decode_strategy": args.decode_strategy,
        "width": 1 if args.decode_strategy == "greedy" else args.width,
        "augmentation": args.augmentation,
        "capacity": capacity,
        "no_aug_split_score": no_aug_mean,
        "no_aug_standard_error": no_aug_se,
        "aug_split_score": aug_mean,
        "aug_standard_error": aug_se,
        "elapsed_seconds": time.time() - started,
    }
    print(json.dumps(result, indent=2))

    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print("saved", output)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", default=str(DEFAULT_DATASET))
    parser.add_argument("--model", required=True, help="Checkpoint file or run directory")
    parser.add_argument("--epoch", type=int, default=None, help="Epoch when --model is a directory")
    parser.add_argument("--num-instances", type=int, default=10000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--decode-strategy", choices=("greedy", "sampling"), default="sampling")
    parser.add_argument("--width", type=int, default=100, help="Sample count per augmentation")
    parser.add_argument("--sample-chunk", type=int, default=20)
    parser.add_argument("--augmentation", type=int, choices=(1, 8), default=8)
    parser.add_argument("--capacity", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output", default=None, help="Optional JSON summary path")
    args = parser.parse_args()
    if args.width <= 0 or args.sample_chunk <= 0:
        parser.error("width and sample-chunk must be positive")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
