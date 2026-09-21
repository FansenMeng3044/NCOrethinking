#!/usr/bin/env python
"""Evaluate AM-TW or AM-Split-TW with 1x/8x geometric augmentation."""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path(os.environ["NCORETHINKING_REPO"]).expanduser().resolve()
CVRP_DIR = REPO / "NEW_py_ver" / "CVRP"
AM_ROOT = CVRP_DIR / "AM_SPLIT"
for import_root in (AM_ROOT, CVRP_DIR, CVRP_DIR.parent):
    sys.path.insert(0, str(import_root))

from CVRPTWCore import (
    augment_xy_by_8,
    replay_cvrptw_actions,
    split_routes_to_actions,
)
from nets.attention_model import set_decode_type
from utils.functions import load_model, move_to


def augment_batch(batch, factor):
    if factor == 1:
        return batch
    if factor != 8:
        raise ValueError("augmentation must be 1 or 8")
    output = {
        "depot": augment_xy_by_8(batch["depot"][:, None])[:, 0],
        "loc": augment_xy_by_8(batch["loc"]),
    }
    for key, value in batch.items():
        if key not in output:
            output[key] = value.repeat((8,) + (1,) * (value.dim() - 1))
    return output


def decode(model, batch, strategy, width):
    if strategy == "greedy":
        set_decode_type(model, "greedy")
        costs, _, tours = model(batch, return_pi=True)
        return costs, tours
    set_decode_type(model, "sampling")
    return tuple(reversed(model.sample_many(batch, batch_rep=width, iter_rep=1)))


def strict_route_counts(problem, batch, tours):
    if problem.NAME == "am_split_tw":
        split = problem.split_costs(batch, tours, return_predecessors=True)
        actions = split_routes_to_actions(tours, split.predecessors)
    else:
        split = None
        actions = tours
    replay = replay_cvrptw_actions(
        batch["depot"][:, None], batch["loc"], batch["demand"],
        batch["service_time"], batch["tw_start"], batch["tw_end"], actions,
        capacity=problem.VEHICLE_CAPACITY,
        depot_start=batch["depot_start"], depot_end=batch["depot_end"],
        speed=problem.SPEED,
        loc_scaler=problem.LOC_SCALER,
        epsilon=problem.EPSILON,
    )
    feasible = replay.feasible
    if split is not None:
        feasible &= torch.isclose(
            replay.distances, split.costs, rtol=1e-5, atol=1e-5
        )
        feasible &= replay.route_counts == split.route_counts
    return replay.route_counts, feasible


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_batch(model, batch, args, device):
    original_size = batch["loc"].size(0)
    augmented_batch = augment_batch(move_to(batch, device), args.augmentation)

    synchronize(device)
    inference_started = time.perf_counter()
    costs, tours = decode(model, augmented_batch, args.strategy, args.width)
    cost_matrix = costs.reshape(args.augmentation, original_size)
    no_aug = cost_matrix[0]
    best_cost, best_aug = cost_matrix.min(dim=0)

    tour_matrix = tours.reshape(args.augmentation, original_size, -1)
    rows = torch.arange(original_size, device=device)
    best_tours = tour_matrix[best_aug, rows]
    selected_batch = {
        key: value.reshape(
            args.augmentation, original_size, *value.shape[1:]
        )[best_aug, rows]
        for key, value in augmented_batch.items()
    }
    synchronize(device)
    inference_seconds = time.perf_counter() - inference_started

    synchronize(device)
    replay_started = time.perf_counter()
    counts, feasible = strict_route_counts(
        model.problem, selected_batch, best_tours
    )
    synchronize(device)
    replay_seconds = time.perf_counter() - replay_started
    return {
        "distance_1x": no_aug.cpu(),
        "distance_aug": best_cost.cpu(),
        "route_counts": counts.cpu(),
        "feasible": feasible.cpu(),
        "inference_seconds": inference_seconds,
        "replay_seconds": replay_seconds,
    }


def evaluate(args):
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    model, model_args = load_model(args.model, epoch=args.epoch)
    if model.problem.NAME not in ("cvrptw", "am_split_tw"):
        raise ValueError("checkpoint must be AM-TW or AM-Split-TW")
    model.to(device).eval()
    dataset = model.problem.make_dataset(
        filename=str(Path(args.dataset).expanduser().resolve()),
        size=model_args["graph_size"],
        num_samples=args.num_instances,
        offset=args.offset,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    if args.timing_warmup_batches:
        warmup_batch = next(iter(loader))
        with torch.no_grad():
            for _ in range(args.timing_warmup_batches):
                evaluate_batch(model, warmup_batch, args, device)
        synchronize(device)

    no_aug_values, aug_values, route_values, feasible_values = [], [], [], []
    all_feasible = True
    inference_seconds = 0.0
    replay_seconds = 0.0
    synchronize(device)
    wall_started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            details = evaluate_batch(model, batch, args, device)
            no_aug_values.append(details["distance_1x"])
            aug_values.append(details["distance_aug"])
            route_values.append(details["route_counts"])
            feasible_values.append(details["feasible"])
            inference_seconds += details["inference_seconds"]
            replay_seconds += details["replay_seconds"]
            all_feasible &= bool(details["feasible"].all())
    synchronize(device)
    audited_wall_seconds = time.perf_counter() - wall_started

    no_aug = torch.cat(no_aug_values).double()
    augmented = torch.cat(aug_values).double()
    routes = torch.cat(route_values).double()
    feasible = torch.cat(feasible_values).bool()

    if args.per_instance_output:
        output = Path(args.per_instance_output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ("instance", "distance_1x", "distance_aug", "vehicle_count", "feasible")
            )
            for index, values in enumerate(zip(no_aug, augmented, routes, feasible)):
                writer.writerow((
                    args.offset + index, values[0].item(), values[1].item(),
                    int(values[2].item()), int(values[3].item()),
                ))

    def stats(values):
        return values.mean().item(), (
            values.std(unbiased=True) / math.sqrt(values.numel())
        ).item() if values.numel() > 1 else 0.0

    no_aug_mean, no_aug_se = stats(no_aug)
    aug_mean, aug_se = stats(augmented)
    result = {
        "problem": model.problem.NAME,
        "model": str(Path(args.model).expanduser().resolve()),
        "dataset": str(Path(args.dataset).expanduser().resolve()),
        "instances": int(augmented.numel()),
        "augmentation": args.augmentation,
        "decode_strategy": args.strategy,
        "width": 1 if args.strategy == "greedy" else args.width,
        "timing_batch_size": args.batch_size,
        "timing_warmup_batches": args.timing_warmup_batches,
        "mean_distance_1x": no_aug_mean,
        "standard_error_1x": no_aug_se,
        "mean_distance_aug": aug_mean,
        "standard_error_aug": aug_se,
        "mean_vehicle_count": routes.mean().item(),
        "strict_replay_instances": int(augmented.numel()),
        "strict_replay_feasible_instances": int(feasible.sum().item()),
        "all_selected_solutions_feasible": all_feasible,
        "inference_seconds": inference_seconds,
        "strict_replay_seconds": replay_seconds,
        "audited_wall_seconds": audited_wall_seconds,
        "inference_instances_per_second": (
            augmented.numel() / inference_seconds
            if inference_seconds > 0 else None
        ),
        "per_instance_output": (
            str(Path(args.per_instance_output).expanduser().resolve())
            if args.per_instance_output else None
        ),
    }
    print(json.dumps(result, indent=2))
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--model", required=True)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--num-instances", type=int, default=10000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--augmentation", type=int, choices=(1, 8), default=8)
    parser.add_argument("--strategy", choices=("greedy", "sampling"), default="greedy")
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--per-instance-output", default=None)
    parser.add_argument("--timing-warmup-batches", type=int, default=1)
    args = parser.parse_args()
    if args.num_instances <= 0:
        parser.error("--num-instances must be positive")
    if args.offset < 0:
        parser.error("--offset must be non-negative")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.width <= 0:
        parser.error("width must be positive")
    if args.timing_warmup_batches < 0:
        parser.error("--timing-warmup-batches must be non-negative")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
