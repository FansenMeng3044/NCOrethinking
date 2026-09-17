#!/usr/bin/env python3
"""Evaluate one final n=100 AM/POMO Direct or Split checkpoint on larger CVRP.

Protocol: deterministic greedy decoding, no geometric augmentation, one AM
trajectory, and full POMO multi-start (one start per customer).  Every selected
solution is replayed independently for coverage, capacity and Euclidean cost.
Per-instance JSON files make the run losslessly resumable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


MODEL_PARAMS = {
    "embedding_dim": 128,
    "sqrt_embedding_dim": 128**0.5,
    "encoder_layer_num": 6,
    "qkv_dim": 16,
    "head_num": 8,
    "logit_clipping": 10,
    "ff_hidden_dim": 512,
    "eval_type": "argmax",
}
EXPECTED = {
    "pomo": (2000, "74bf36630f179b69d69ceae4997d6b03994815677c0c6bf4000b4936feb29af8"),
    "pomo_split": (2000, "887e1e7a24df1cdb4877b11a188ecda2648e006d8cea7f849866e51332b6fbac"),
    "am": (100, "450c46de4a257605d66b5705cb38f0fd265ec3587a2c7bd866d0605ee463dfce"),
    "am_split": (100, "b669a5251c8aba3ba97d74df3c6ea1fc8e0cda0066be1bb76e9855d9d1ce2f44"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@contextmanager
def pomo_cuda_defaults(device: torch.device):
    if device.type != "cuda":
        yield
        return
    torch.cuda.set_device(device)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    try:
        yield
    finally:
        torch.set_default_tensor_type(torch.FloatTensor)


def load_dataset(path: Path) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    required = ("depot_xy", "node_xy", "node_demand")
    if any(key not in data for key in required):
        raise ValueError(f"{path}: missing one of {required}")
    depot = data["depot_xy"].float().contiguous()
    nodes = data["node_xy"].float().contiguous()
    demand = data["node_demand"].float().contiguous()
    if depot.ndim != 3 or depot.shape[1:] != (1, 2):
        raise ValueError("depot_xy must be (instances,1,2)")
    if nodes.ndim != 3 or nodes.shape[2] != 2:
        raise ValueError("node_xy must be (instances,n,2)")
    if demand.shape != nodes.shape[:2]:
        raise ValueError("node_demand must be (instances,n)")
    if not torch.isfinite(depot).all() or not torch.isfinite(nodes).all() or not torch.isfinite(demand).all():
        raise ValueError("dataset contains non-finite values")
    if (demand <= 0).any() or (demand > 1 + 1e-7).any():
        raise ValueError("demands must be in (0,1]")
    return {"depot_xy": depot, "node_xy": nodes, "node_demand": demand, **{
        key: value for key, value in data.items() if key not in required
    }}


def validate_checkpoint(path: Path, architecture: str, repo: Path) -> dict[str, Any]:
    expected_epoch, expected_hash = EXPECTED[architecture]
    digest = sha256_file(path)
    if digest != expected_hash:
        raise ValueError(f"checkpoint SHA-256 mismatch: {digest} != {expected_hash}")
    if architecture.startswith("am"):
        cvrp = repo / "NEW_py_ver" / "CVRP"
        am_root = cvrp / "AM_SPLIT"
        for import_root in (am_root, cvrp, cvrp.parent):
            if str(import_root) not in sys.path:
                sys.path.insert(0, str(import_root))
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError("checkpoint is not a dict")
    epoch = state.get("epoch")
    if epoch is None:
        token = path.stem.rsplit("-", 1)[-1]
        epoch = int(token) if token.isdigit() else None
    if int(epoch) != expected_epoch:
        raise ValueError(f"checkpoint epoch {epoch} != {expected_epoch}")
    model_key = "model_state_dict" if architecture.startswith("pomo") else "model"
    if model_key not in state or not isinstance(state[model_key], dict):
        raise ValueError(f"checkpoint missing {model_key}")
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "epoch": int(epoch),
        "model_state_key_count": len(state[model_key]),
    }


def actions_to_routes(actions: list[int], n: int) -> list[list[int]]:
    routes: list[list[int]] = []
    current: list[int] = []
    for action in actions:
        value = int(action)
        if value == 0:
            if current:
                routes.append(current)
                current = []
        elif 1 <= value <= n:
            current.append(value)
        else:
            raise ValueError(f"out-of-range action {value}")
    if current:
        routes.append(current)
    return routes


def replay_routes(
    depot: np.ndarray,
    nodes: np.ndarray,
    demand: np.ndarray,
    routes: list[list[int]],
) -> tuple[float, int]:
    n = len(nodes)
    flat = [customer for route in routes for customer in route]
    if len(flat) != n or sorted(flat) != list(range(1, n + 1)):
        raise ValueError("coverage/duplicate verification failed")
    cost = 0.0
    for route in routes:
        if not route:
            raise ValueError("empty route")
        indices = np.asarray(route, dtype=np.int64) - 1
        if float(demand[indices].sum(dtype=np.float64)) > 1.0 + 1e-6:
            raise ValueError("capacity verification failed")
        coords = nodes[indices]
        cost += float(np.linalg.norm(coords[0] - depot))
        if len(coords) > 1:
            cost += float(np.linalg.norm(coords[1:] - coords[:-1], axis=1).sum(dtype=np.float64))
        cost += float(np.linalg.norm(coords[-1] - depot))
    return cost, len(routes)


def reconstruct_split(tour: torch.Tensor, predecessor: torch.Tensor) -> list[list[int]]:
    order = [int(value) for value in tour.detach().cpu().tolist()]
    pred = [int(value) for value in predecessor.detach().cpu().tolist()]
    routes: list[list[int]] = []
    end = len(order)
    while end > 0:
        begin = pred[end]
        if begin < 0 or begin >= end:
            raise ValueError("invalid predecessor chain")
        routes.append(order[begin:end])
        end = begin
    routes.reverse()
    return routes


class POMOEvaluator:
    def __init__(self, repo: Path, checkpoint: Path, architecture: str, device: torch.device):
        cvrp = repo / "NEW_py_ver" / "CVRP"
        sys.path.insert(0, str(cvrp))
        self.architecture = architecture
        self.device = device
        if architecture == "pomo":
            from POMO.CVRPEnv import CVRPEnv as Env
            from POMO.CVRPModel import CVRPModel as Model
        else:
            from POMO_SPLIT.GiantTourEnv import GiantTourEnv as Env
            from POMO_SPLIT.GiantTourModel import GiantTourModel as Model
        self.Env = Env
        self.model = Model(**MODEL_PARAMS).to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"], strict=True)
        self.model.eval()

    def evaluate(self, depot: torch.Tensor, nodes: torch.Tensor, demand: torch.Tensor):
        batch, n, _ = nodes.shape
        env = self.Env(
            problem_size=n,
            pomo_size=n,
            capacity=1.0,
            device=self.device,
        )
        env.FLAG__use_saved_problems = True
        env.saved_depot_xy = depot
        env.saved_node_xy = nodes
        env.saved_node_demand = demand
        env.saved_index = 0
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with pomo_cuda_defaults(self.device), torch.inference_mode():
            env.load_problems(batch, aug_factor=1)
            reset_state, _, _ = env.reset()
            self.model.pre_forward(reset_state)
            step_state, _, done = env.pre_step()
            while not done:
                selected, _ = self.model(step_state)
                step_state, reward, done = env.step(selected)
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        # One best POMO trajectory per physical instance.
        best = reward.argmax(dim=1)
        outputs = []
        for row in range(batch):
            index = int(best[row].item())
            decoder_cost = float((-reward[row, index]).item())
            if self.architecture == "pomo":
                actions = env.selected_node_list[row, index].detach().cpu().tolist()
                routes = actions_to_routes(actions, n)
            else:
                routes = reconstruct_split(
                    env.selected_node_list[row, index],
                    env.last_split_result.predecessors[row, index],
                )
            outputs.append((decoder_cost, routes))
        return outputs, elapsed / batch, int(torch.cuda.max_memory_allocated(self.device))


class AMEvaluator:
    def __init__(self, repo: Path, checkpoint: Path, architecture: str, device: torch.device):
        cvrp = repo / "NEW_py_ver" / "CVRP"
        am_root = cvrp / "AM_SPLIT"
        for path in (am_root, cvrp, cvrp.parent):
            sys.path.insert(0, str(path))
        from nets.attention_model import AttentionModel, set_decode_type
        from utils.functions import load_problem

        self.architecture = architecture
        self.device = device
        expected = "cvrp" if architecture == "am" else "am_split"
        model_args = json.loads((checkpoint.parent / "args.json").read_text(encoding="utf-8"))
        if model_args["problem"] != expected:
            raise ValueError(f"checkpoint problem {model_args['problem']} != {expected}")
        if int(model_args["graph_size"]) != 100:
            raise ValueError("AM checkpoint is not the declared n=100 model")
        problem = load_problem(expected)
        if architecture == "am_split":
            problem.configure(capacity=1.0, train_reward="split")
        self.model = AttentionModel(
            model_args["embedding_dim"],
            model_args["hidden_dim"],
            problem,
            n_encode_layers=model_args["n_encode_layers"],
            mask_inner=True,
            mask_logits=True,
            normalization=model_args["normalization"],
            tanh_clipping=model_args["tanh_clipping"],
            checkpoint_encoder=model_args.get("checkpoint_encoder", False),
            shrink_size=model_args.get("shrink_size"),
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"], strict=True)
        self.model.to(device).eval()
        set_decode_type(self.model, "greedy")

    def evaluate(self, depot: torch.Tensor, nodes: torch.Tensor, demand: torch.Tensor):
        batch = {"depot": depot[:, 0], "loc": nodes, "demand": demand}
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            embeddings, _ = self.model.embedder(self.model._init_embed(batch))
            _, tours = self.model._inner(batch, embeddings)
            if self.architecture == "am_split":
                split = self.model.problem.split_costs(batch, tours, return_predecessors=True)
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        outputs = []
        n = nodes.size(1)
        for row in range(nodes.size(0)):
            if self.architecture == "am":
                routes = actions_to_routes(tours[row].detach().cpu().tolist(), n)
                decoder_cost = math.nan  # replay below is authoritative
            else:
                routes = reconstruct_split(tours[row], split.predecessors[row])
                decoder_cost = float(split.costs[row].item())
            outputs.append((decoder_cost, routes))
        return outputs, elapsed / nodes.size(0), int(torch.cuda.max_memory_allocated(self.device))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def consolidate(output: Path, expected_count: int) -> list[dict[str, Any]]:
    rows = []
    for index in range(expected_count):
        path = output / "instances" / f"{index:04d}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({key: value for key, value in payload.items() if key != "routes"})
        rows[-1]["route_file"] = path.relative_to(output).as_posix()
    csv_path = output / "results.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    if rows:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        temporary.write_text("", encoding="utf-8")
    temporary.replace(csv_path)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--architecture", choices=tuple(EXPECTED), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    seed_all(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this formal evaluation requires CUDA")
    data = load_dataset(args.dataset)
    total = data["node_xy"].size(0)
    if args.limit is not None:
        total = min(total, args.limit)
    n = data["node_xy"].size(1)
    checkpoint_meta = validate_checkpoint(args.checkpoint, args.architecture, args.repo)
    dataset_hash = sha256_file(args.dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "protocol": "cvrp_size_generalization_n100_checkpoint_greedy_noaug_full_pomo_v1",
        "architecture": args.architecture,
        "training_size": 100,
        "problem_size": n,
        "instances": total,
        "seed": args.seed,
        "pomo_starts": n if args.architecture.startswith("pomo") else 1,
        "augmentation": 1,
        "demand_capacity_semantics": "integer Uniform{1,...,9}, capacity=50, normalized capacity=1",
        "dataset_path": str(args.dataset.resolve()),
        "dataset_sha256": dataset_hash,
        "checkpoint": checkpoint_meta,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
    }
    atomic_json(args.output / "run_manifest.json", manifest)
    evaluator = (
        POMOEvaluator(args.repo, args.checkpoint, args.architecture, device)
        if args.architecture.startswith("pomo")
        else AMEvaluator(args.repo, args.checkpoint, args.architecture, device)
    )

    existing = {
        int(path.stem)
        for path in (args.output / "instances").glob("*.json")
        if path.stem.isdigit()
    } if (args.output / "instances").is_dir() else set()
    for start in range(0, total, args.batch_size):
        indices = [index for index in range(start, min(total, start + args.batch_size)) if index not in existing]
        if not indices:
            continue
        depot = data["depot_xy"][indices].to(device)
        nodes = data["node_xy"][indices].to(device)
        demand = data["node_demand"][indices].to(device)
        outputs, inference_seconds, peak_memory = evaluator.evaluate(depot, nodes, demand)
        for offset, index in enumerate(indices):
            decoder_cost, routes = outputs[offset]
            replay_cost, vehicles = replay_routes(
                data["depot_xy"][index, 0].numpy().astype(np.float64),
                data["node_xy"][index].numpy().astype(np.float64),
                data["node_demand"][index].numpy().astype(np.float64),
                routes,
            )
            if math.isfinite(decoder_cost):
                tolerance = max(2e-4, abs(replay_cost) * 2e-5)
                if abs(decoder_cost - replay_cost) > tolerance:
                    raise AssertionError(
                        f"instance {index}: decoder/replay mismatch {decoder_cost} vs {replay_cost}"
                    )
            payload = {
                "schema_version": 1,
                "timestamp_utc": utc_now(),
                "protocol": manifest["protocol"],
                "architecture": args.architecture,
                "training_size": 100,
                "problem_size": n,
                "instance_index": index,
                "status": "ok",
                "cost": replay_cost,
                "vehicles": vehicles,
                "inference_seconds_amortized": inference_seconds,
                "peak_cuda_memory_bytes": peak_memory,
                "candidates": n if args.architecture.startswith("pomo") else 1,
                "dataset_sha256": dataset_hash,
                "checkpoint_sha256": checkpoint_meta["sha256"],
                "routes": routes,
            }
            atomic_json(args.output / "instances" / f"{index:04d}.json", payload)
        rows = consolidate(args.output, total)
        completed = len(rows)
        mean_cost = sum(float(row["cost"]) for row in rows) / completed
        print(
            f"{utc_now()} architecture={args.architecture} n={n} "
            f"completed={completed}/{total} mean_cost={mean_cost:.8f} "
            f"last_seconds={inference_seconds:.4f} peak_gib={peak_memory / 2**30:.3f}",
            flush=True,
        )

    rows = consolidate(args.output, total)
    if len(rows) != total:
        raise RuntimeError(f"incomplete result: {len(rows)}/{total}")
    summary = {
        **manifest,
        "completed_at_utc": utc_now(),
        "success": len(rows),
        "mean_cost": sum(float(row["cost"]) for row in rows) / len(rows),
        "mean_vehicles": sum(float(row["vehicles"]) for row in rows) / len(rows),
        "mean_inference_seconds": sum(float(row["inference_seconds_amortized"]) for row in rows) / len(rows),
        "max_peak_cuda_memory_bytes": max(int(row["peak_cuda_memory_bytes"]) for row in rows),
    }
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
