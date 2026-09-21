#!/usr/bin/env python
"""Evaluate AM Direct or AM Split on one frozen POMO-format CVRP data set."""

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def augment_xy_by_8(xy):
    x, y = xy[..., [0]], xy[..., [1]]
    return torch.cat((
        torch.cat((x, y), -1), torch.cat((1 - x, y), -1),
        torch.cat((x, 1 - y), -1), torch.cat((1 - x, 1 - y), -1),
        torch.cat((y, x), -1), torch.cat((1 - y, x), -1),
        torch.cat((y, 1 - x), -1), torch.cat((1 - y, 1 - x), -1),
    ), 0)


def augment_batch(batch, factor):
    if factor == 1:
        return batch
    return {
        "depot": augment_xy_by_8(batch["depot"][:, None])[:, 0],
        "loc": augment_xy_by_8(batch["loc"]),
        "demand": batch["demand"].repeat(8, 1),
    }


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class FrozenCVRPDataset(torch.utils.data.Dataset):
    """Expose the official POMO tuple without rewriting any frozen tensor."""

    def __init__(self, path, count):
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict):
            self.depot = raw["depot_xy"]
            self.loc = raw["node_xy"]
            self.demand = raw["node_demand"]
        elif isinstance(raw, (tuple, list)) and len(raw) == 3:
            self.depot, self.loc, self.demand = raw
        else:
            raise TypeError("unsupported frozen CVRP container")
        self.depot = self.depot[:count]
        self.loc = self.loc[:count]
        self.demand = self.demand[:count]

    def __len__(self):
        return self.loc.size(0)

    def __getitem__(self, index):
        return {
            "depot": self.depot[index, 0],
            "loc": self.loc[index],
            "demand": self.demand[index],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--architecture", choices=("am", "am_split"), required=True)
    parser.add_argument("--instances", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--augmentation", type=int, choices=(1, 8), default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-instance-output", required=True)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    cvrp = repo / "NEW_py_ver" / "CVRP"
    am_root = cvrp / "AM_SPLIT"
    for path in (am_root, cvrp, cvrp.parent):
        sys.path.insert(0, str(path))
    from nets.attention_model import set_decode_type
    from utils.functions import load_model, move_to

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = Path(args.checkpoint).resolve()
    dataset_path = Path(args.dataset).resolve()
    model, model_args = load_model(str(checkpoint))
    expected = "cvrp" if args.architecture == "am" else "am_split"
    if model.problem.NAME != expected:
        raise ValueError(f"expected {expected}, got {model.problem.NAME}")
    if int(model_args["graph_size"]) != 50:
        raise ValueError("this formal run requires an n=50 checkpoint")
    if args.architecture == "am_split":
        model.problem.configure(capacity=1.0, train_reward="split")
    model.to(device).eval()
    set_decode_type(model, "greedy")

    dataset = FrozenCVRPDataset(dataset_path, args.instances)
    if len(dataset) != args.instances:
        raise ValueError(f"expected {args.instances} instances, found {len(dataset)}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    one_values, aug_values = [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            original = batch["loc"].size(0)
            batch = augment_batch(move_to(batch, device), args.augmentation)
            costs, _ = model(batch)
            matrix = costs.reshape(args.augmentation, original)
            one_values.append(matrix[0].double().cpu())
            aug_values.append(matrix.min(0).values.double().cpu())
    elapsed = time.perf_counter() - started
    one = torch.cat(one_values)
    aug = torch.cat(aug_values)
    if not torch.isfinite(one).all() or not torch.isfinite(aug).all():
        raise ValueError("non-finite cost in formal evaluation")

    per_instance = Path(args.per_instance_output).resolve()
    per_instance.parent.mkdir(parents=True, exist_ok=True)
    with per_instance.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("instance", "distance_1x", "distance_aug"))
        for index, (first, best) in enumerate(zip(one.tolist(), aug.tolist())):
            writer.writerow((index, first, best))

    def stats(values):
        mean = values.mean().item()
        se = values.std(unbiased=True).item() / math.sqrt(values.numel())
        return mean, se

    one_mean, one_se = stats(one)
    aug_mean, aug_se = stats(aug)
    result = {
        "dataset": str(dataset_path),
        "dataset_sha256": digest(dataset_path),
        "model": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "architecture": args.architecture,
        "training_size": 50,
        "instances": int(aug.numel()),
        "decoding": "greedy",
        "augmentation": args.augmentation,
        "mean_distance_1x": one_mean,
        "standard_error_1x": one_se,
        "mean_distance_aug": aug_mean,
        "standard_error_aug": aug_se,
        "elapsed_seconds": elapsed,
        "per_instance_output": str(per_instance),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
