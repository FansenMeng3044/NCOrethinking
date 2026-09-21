#!/usr/bin/env python
"""Evaluate POMO Direct or POMO Split on one frozen POMO-format CVRP data set."""

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch


MODEL_PARAMS = {
    "embedding_dim": 128,
    "sqrt_embedding_dim": 128 ** 0.5,
    "encoder_layer_num": 6,
    "qkv_dim": 16,
    "head_num": 8,
    "logit_clipping": 10,
    "ff_hidden_dim": 512,
    "eval_type": "argmax",
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def pomo_tensor_device(device):
    if device.type != "cuda":
        yield
        return
    torch.cuda.set_device(device)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    try:
        yield
    finally:
        torch.set_default_tensor_type(torch.FloatTensor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--architecture", choices=("pomo", "pomo_split"), required=True)
    parser.add_argument("--instances", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--augmentation", type=int, choices=(1, 8), default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-instance-output", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    repo = Path(args.repo).resolve()
    cvrp = repo / "NEW_py_ver" / "CVRP"
    sys.path.insert(0, str(cvrp))
    if args.architecture == "pomo":
        from POMO.CVRPEnv import CVRPEnv as Env
        from POMO.CVRPModel import CVRPModel as Model
    else:
        from POMO_SPLIT.GiantTourEnv import GiantTourEnv as Env
        from POMO_SPLIT.GiantTourModel import GiantTourModel as Model

    device = torch.device(args.device)
    checkpoint = Path(args.checkpoint).resolve()
    dataset_path = Path(args.dataset).resolve()
    raw = torch.load(dataset_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        saved = {
            "depot_xy": raw["depot_xy"],
            "node_xy": raw["node_xy"],
            "node_demand": raw["node_demand"],
        }
    elif isinstance(raw, (tuple, list)) and len(raw) == 3:
        saved = dict(zip(("depot_xy", "node_xy", "node_demand"), raw))
    else:
        raise TypeError("unsupported frozen CVRP container")
    if saved["node_xy"].ndim != 3 or tuple(saved["node_xy"].shape[1:]) != (50, 2):
        raise ValueError(f"unexpected node tensor shape {tuple(saved['node_xy'].shape)}")
    if saved["node_xy"].size(0) < args.instances:
        raise ValueError(
            f"requested {args.instances} instances from a set of {saved['node_xy'].size(0)}"
        )

    model_params = dict(MODEL_PARAMS)
    if args.architecture == "pomo_split":
        model_params["node_feature_dim"] = 3
    model = Model(**model_params).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if int(state.get("epoch", 2000)) != 2000:
        raise ValueError(f"expected epoch 2000, found {state.get('epoch')}")
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()

    one_values, aug_values = [], []
    started = time.perf_counter()
    for begin in range(0, args.instances, args.batch_size):
        end = min(begin + args.batch_size, args.instances)
        batch = end - begin
        env = Env(problem_size=50, pomo_size=50, capacity=1.0, device=device)
        env.FLAG__use_saved_problems = True
        env.saved_depot_xy = saved["depot_xy"][begin:end].to(device)
        env.saved_node_xy = saved["node_xy"][begin:end].to(device)
        env.saved_node_demand = saved["node_demand"][begin:end].to(device)
        env.saved_index = 0
        with pomo_tensor_device(device), torch.inference_mode():
            env.load_problems(batch, aug_factor=args.augmentation)
            reset, _, _ = env.reset()
            model.pre_forward(reset)
            step, reward, done = env.pre_step()
            while not done:
                selected, _ = model(step)
                step, reward, done = env.step(selected)
        costs = (-reward).reshape(args.augmentation, batch, 50)
        best_pomo = costs.min(2).values
        one_values.append(best_pomo[0].double().cpu())
        aug_values.append(best_pomo.min(0).values.double().cpu())
    elapsed = time.perf_counter() - started
    one, aug = torch.cat(one_values), torch.cat(aug_values)
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
        "decoding": "greedy POMO with 50 starts",
        "pomo_size": 50,
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
