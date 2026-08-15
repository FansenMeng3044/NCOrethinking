#!/usr/bin/env python
"""Evaluate AM-TW or AM-Split-TW with 1x/8x geometric augmentation."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

from CVRPTWCore import augment_xy_by_8, replay_cvrptw_actions
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
        result = problem.split_costs(batch, tours)
        return result.route_counts, torch.ones_like(result.route_counts, dtype=torch.bool)
    replay = replay_cvrptw_actions(
        batch["depot"][:, None], batch["loc"], batch["demand"],
        batch["service_time"], batch["tw_start"], batch["tw_end"], tours,
        capacity=problem.VEHICLE_CAPACITY,
        depot_start=batch["depot_start"], depot_end=batch["depot_end"],
        speed=problem.SPEED,
        loc_scaler=problem.LOC_SCALER,
        epsilon=problem.EPSILON,
    )
    return replay.route_counts, replay.feasible


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
    no_aug_values, aug_values, route_values = [], [], []
    all_feasible = True
    started = time.time()
    with torch.no_grad():
        for batch in loader:
            original_size = batch["loc"].size(0)
            augmented = augment_batch(move_to(batch, device), args.augmentation)
            costs, tours = decode(model, augmented, args.strategy, args.width)
            cost_matrix = costs.reshape(args.augmentation, original_size)
            no_aug_values.append(cost_matrix[0].cpu())
            best_cost, best_aug = cost_matrix.min(dim=0)
            aug_values.append(best_cost.cpu())

            tour_matrix = tours.reshape(args.augmentation, original_size, -1)
            rows = torch.arange(original_size, device=device)
            best_tours = tour_matrix[best_aug, rows]
            selected_batch = {
                key: value.reshape(args.augmentation, original_size, *value.shape[1:])[
                    best_aug, rows
                ]
                for key, value in augmented.items()
            }
            counts, feasible = strict_route_counts(
                model.problem, selected_batch, best_tours
            )
            route_values.append(counts.cpu())
            all_feasible &= bool(feasible.all())

    no_aug = torch.cat(no_aug_values).double()
    augmented = torch.cat(aug_values).double()
    routes = torch.cat(route_values).double()

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
        "mean_distance_1x": no_aug_mean,
        "standard_error_1x": no_aug_se,
        "mean_distance_aug": aug_mean,
        "standard_error_aug": aug_se,
        "mean_vehicle_count": routes.mean().item(),
        "strict_replay_instances": int(augmented.numel()),
        "all_selected_solutions_feasible": all_feasible,
        "elapsed_seconds": time.time() - started,
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
    args = parser.parse_args()
    if args.width <= 0:
        parser.error("width must be positive")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
