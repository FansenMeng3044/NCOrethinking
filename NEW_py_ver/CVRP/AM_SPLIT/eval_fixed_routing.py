#!/usr/bin/env python
"""Evaluate standard AM or AM-Split on a fixed POMO-format CVRP dataset."""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from nets.attention_model import set_decode_type
from utils.functions import load_model, move_to


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
        costs, _ = model(batch)
        return costs

    set_decode_type(model, "sampling")
    best_cost = None
    remaining = width
    while remaining > 0:
        current_width = min(sample_chunk, remaining)
        _, costs = model.sample_many(batch, batch_rep=current_width, iter_rep=1)
        best_cost = costs if best_cost is None else torch.minimum(best_cost, costs)
        remaining -= current_width
    return best_cost


def evaluate(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model, model_args = load_model(args.model, epoch=args.epoch)
    problem_name = model_args["problem"]
    if problem_name == "am_split":
        capacity = model_args.get("capacity", 1.0)
        model.problem.configure(capacity=capacity, train_reward="split")
    model.to(device)
    model.eval()

    dataset_path = Path(args.dataset).expanduser().resolve()
    dataset = model.problem.make_dataset(
        filename=str(dataset_path),
        size=model_args["graph_size"],
        num_samples=args.num_instances,
        offset=args.offset,
    )
    if not dataset:
        raise ValueError("evaluation selection is empty")
    loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)

    total = 0.0
    square_total = 0.0
    count = 0
    started = time.time()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            original_batch_size = batch["loc"].size(0)
            batch = augment_batch(move_to(batch, device), args.augmentation)
            costs = decode(model, batch, args.decode_strategy, args.width, args.sample_chunk)
            costs = costs.view(args.augmentation, original_batch_size).min(dim=0).values
            costs_double = costs.double()
            total += costs_double.sum().item()
            square_total += costs_double.square().sum().item()
            count += original_batch_size
            if args.log_every and (batch_index % args.log_every == 0 or count == len(dataset)):
                print("evaluated {}/{}: score={:.6f}".format(count, len(dataset), total / count), flush=True)

    mean = total / count
    standard_error = math.sqrt(max(square_total / count - mean ** 2, 0.0) / count)
    result = {
        "problem": problem_name,
        "model": str(Path(args.model).expanduser().resolve()),
        "epoch": args.epoch,
        "dataset": str(dataset_path),
        "instances": count,
        "decode_strategy": args.decode_strategy,
        "width": 1 if args.decode_strategy == "greedy" else args.width,
        "augmentation": args.augmentation,
        "score": mean,
        "standard_error": standard_error,
        "elapsed_seconds": time.time() - started,
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print("saved", output, flush=True)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--model", required=True, help="Checkpoint file or run directory")
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--num-instances", type=int, default=10000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--augmentation", type=int, choices=(1, 8), required=True)
    parser.add_argument("--decode-strategy", choices=("greedy", "sampling"), default="greedy")
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--sample-chunk", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.width <= 0 or args.sample_chunk <= 0:
        parser.error("width and sample-chunk must be positive")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
