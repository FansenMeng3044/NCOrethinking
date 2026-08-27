#!/usr/bin/env python3
"""Evaluate local POMO/AM direct and Split checkpoints on official XML100.

The decoder follows the public MVMoE benchmark protocol (greedy, optional
8-fold geometric augmentation, POMO size n), but every candidate is selected
and finally verified under the official per-edge CVRPLIB EUC_2D objective.
The 10,000 official solution files are never exposed to the model and are read
only after decoding to obtain the proven optimum and the optimality gap.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from xml100_io import (
    CVRPInstance,
    actions_to_routes,
    gap_percent,
    parse_instance,
    parse_solution,
    reconstruct_split_routes,
    split_giant_tour_dp,
    verify_routes,
)


SCHEMA_VERSION = 2
PROTOCOL = "xml100_mvmoe_greedy_8fold_official_euc2d_v2"
MVMOE_UPSTREAM_COMMIT = "af29e5af0595f94f3ecc3bc46d72df1089a62682"
MVMOE_UPSTREAM_URL = "https://github.com/RoyalSkye/Routing-MVMoE.git"
XML100_ARCHIVE_SHA256 = "ef3814ee4c26e7f1d09cf33a4c0da564109de04ee393afa3b46c7eb33473b24b"
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
ARCHITECTURES = {"pomo", "pomo_split", "am", "am_split"}


@dataclass(frozen=True)
class CandidateBatch:
    kind: str  # depot_actions or giant_tours
    sequences: list[np.ndarray]
    augmentation_indices: np.ndarray
    policy_indices: np.ndarray
    inference_seconds_amortized: float
    physical_batch_size: int


@dataclass(frozen=True)
class Selection:
    routes: list[list[int]]
    cost: int
    vehicles: int
    candidate_index: int
    augmentation_index: int
    policy_index: int
    candidates_generated: int
    candidates_checked_until_selection: int


class NoFeasibleCandidateError(RuntimeError):
    """The decoder ran, but none of its candidates was a valid CVRP solution."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_optimal_costs(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 10_000 or set(rows[0] if rows else ()) != {"instance", "optimum"}:
        raise ValueError(f"{path}: expected a 10,000-row instance/optimum CSV")
    costs: dict[str, int] = {}
    for row in rows:
        name = row["instance"]
        if name in costs:
            raise ValueError(f"{path}: duplicate optimum for {name}")
        value = int(row["optimum"])
        if value <= 0:
            raise ValueError(f"{path}: non-positive optimum for {name}")
        costs[name] = value
    return costs


def evaluator_source_hashes(directory: Path) -> dict[str, str]:
    names = (
        "evaluate_xml100.py",
        "xml100_io.py",
        "prepare_xml100.py",
        "verify_xml100_results.py",
        "summarize_xml100.py",
        "plot_xml100_solution.py",
        "run_xml100.sh",
        "README.md",
        "AUDIT.md",
    )
    return {name: sha256_file(directory / name) for name in names}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if not rows:
        temporary.write_text("", encoding="utf-8")
    else:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def git_metadata(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "status_porcelain": run("status", "--porcelain").splitlines(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "status_porcelain": None}


def resolve_checkpoint(models_path: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = models_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def checkpoint_metadata(path: Path, spec: dict[str, Any], repo: Path) -> dict[str, Any]:
    if not spec["architecture"].startswith("pomo"):
        # AM checkpoints pickle baseline/model-related classes from the local
        # ``nets``/``utils`` packages.  Establish the exact audited source
        # import roots before the metadata-only CPU load as well as before
        # constructing the runner.
        cvrp = repo / "NEW_py_ver" / "CVRP"
        am_root = cvrp / "AM_SPLIT"
        for import_root in (am_root, cvrp, cvrp.parent):
            if str(import_root) not in sys.path:
                sys.path.insert(0, str(import_root))
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(f"{path}: checkpoint must be a dictionary")
    architecture = spec["architecture"]
    if architecture.startswith("pomo"):
        if "model_state_dict" not in state or not isinstance(state["model_state_dict"], dict):
            raise ValueError(f"{path}: missing POMO model_state_dict")
        state_key_count = len(state["model_state_dict"])
    else:
        if "model" not in state or not isinstance(state["model"], dict):
            raise ValueError(f"{path}: missing AM model state")
        state_key_count = len(state["model"])
        if not (path.parent / "args.json").is_file():
            raise FileNotFoundError(f"{path}: AM checkpoint requires sibling args.json")
    epoch = state.get("epoch")
    if epoch is None:
        stem_number = path.stem.rsplit("-", 1)[-1]
        epoch = int(stem_number) if stem_number.isdigit() else None
    expected_epoch = spec.get("expected_epoch")
    if expected_epoch is not None and int(epoch) != int(expected_epoch):
        raise ValueError(f"{path}: expected epoch {expected_epoch}, found {epoch}")
    digest = sha256_file(path)
    expected_digest = spec.get("expected_sha256")
    if expected_digest and digest.lower() != str(expected_digest).lower():
        raise ValueError(f"{path}: checkpoint SHA-256 mismatch")
    return {
        "path": str(path),
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": digest,
        "epoch": epoch,
        "top_level_keys": sorted(str(key) for key in state),
        "model_state_key_count": state_key_count,
    }


def normalized_problem_batch(
    instances: Sequence[CVRPInstance], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # XML100 generator coordinates are integer values in [0, 1000].  Matching
    # MVMoE/POMO input semantics requires a fixed 1000 scaler, not per-instance
    # max scaling (which would change the learned demand/distance relationship).
    coords = np.stack([instance.coords for instance in instances]).astype(np.float32) / 1000.0
    demands = np.stack(
        [instance.demands[1:].astype(np.float32) / float(instance.capacity) for instance in instances]
    )
    if np.any(coords < 0) or np.any(coords > 1):
        raise ValueError("XML100 coordinates must remain within [0, 1000]")
    depot = torch.from_numpy(coords[:, :1]).to(device)
    nodes = torch.from_numpy(coords[:, 1:]).to(device)
    demand = torch.from_numpy(demands).to(device)
    return depot, nodes, demand


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def pomo_tensor_device(device: torch.device):
    """Match the original POMO scripts' global CUDA tensor convention.

    The upstream POMO model and direct environment create a few tensors without
    an explicit device.  Their official test entry point consequently sets the
    default tensor type to CUDA.  Keep that behavior tightly scoped so it cannot
    leak into AM loading or CPU-side official scoring.
    """
    if device.type != "cuda":
        yield
        return
    if torch.empty(0).device.type != "cpu":
        raise RuntimeError("unexpected non-CPU default tensor device before POMO rollout")
    torch.cuda.set_device(device)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    try:
        yield
    finally:
        torch.set_default_tensor_type(torch.FloatTensor)


class POMORunner:
    def __init__(
        self,
        repo: Path,
        spec: dict[str, Any],
        checkpoint: Path,
        device: torch.device,
        augmentation: int,
    ) -> None:
        self.architecture = spec["architecture"]
        self.device = device
        self.augmentation = augmentation
        cvrp = repo / "NEW_py_ver" / "CVRP"
        if str(cvrp) not in sys.path:
            sys.path.insert(0, str(cvrp))
        if self.architecture == "pomo":
            module_root = cvrp / "POMO"
            if str(module_root) not in sys.path:
                sys.path.insert(0, str(module_root))
            from POMO.CVRPEnv import CVRPEnv as Env
            from POMO.CVRPModel import CVRPModel as Model
        elif self.architecture == "pomo_split":
            module_root = cvrp / "POMO_SPLIT"
            if str(module_root) not in sys.path:
                sys.path.insert(0, str(module_root))
            from POMO_SPLIT.GiantTourEnv import GiantTourEnv as Env
            from POMO_SPLIT.GiantTourModel import GiantTourModel as Model
        else:
            raise ValueError(self.architecture)
        self.Env = Env
        params = dict(MODEL_PARAMS)
        params.update(spec.get("model_params", {}))
        params["eval_type"] = "argmax"
        self.model = Model(**params).to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"], strict=True)
        self.model.eval()

    def decode(self, instances: Sequence[CVRPInstance]) -> CandidateBatch:
        batch_size = len(instances)
        depot, nodes, demand = normalized_problem_batch(instances, self.device)
        env_kwargs = {
            "problem_size": 100,
            "pomo_size": 100,
            "capacity": 1.0,
            "device": self.device,
        }
        env = self.Env(**env_kwargs)
        env.FLAG__use_saved_problems = True
        env.saved_depot_xy = depot
        env.saved_node_xy = nodes
        env.saved_node_demand = demand
        env.saved_index = 0
        synchronize(self.device)
        started = time.perf_counter()
        with pomo_tensor_device(self.device), torch.inference_mode():
            env.load_problems(batch_size, aug_factor=self.augmentation)
            reset, _, _ = env.reset()
            self.model.pre_forward(reset)
            step, _, done = env.pre_step()
            while not done:
                selected, _ = self.model(step)
                if (
                    self.architecture == "pomo_split"
                    and env.selected_count + 1 == env.problem_size
                ):
                    # The upstream Split environment computes a floating-point
                    # CUDA Split reward inside the final env.step().  Evaluation
                    # does not use that reward: this evaluator re-splits every
                    # completed giant tour with its exact integer EUC_2D CPU DP.
                    # Append the already-decoded final action directly, matching
                    # env.step() through the point at which the full tour exists,
                    # and avoid executing an irrelevant nondeterministic CUDA
                    # cumsum in the discarded training reward path.
                    env.selected_count += 1
                    env.current_node = selected
                    env.selected_node_list = torch.cat(
                        (env.selected_node_list, selected[:, :, None]), dim=2
                    )
                    done = True
                else:
                    step, _, done = env.step(selected)
        synchronize(self.device)
        elapsed = time.perf_counter() - started
        selected = env.selected_node_list.detach().cpu().numpy().astype(np.int64, copy=False)
        sequences = [
            np.concatenate(
                [selected[augmentation * batch_size + index] for augmentation in range(self.augmentation)],
                axis=0,
            )
            for index in range(batch_size)
        ]
        augmentations = np.repeat(np.arange(self.augmentation, dtype=np.int64), 100)
        policies = np.tile(np.arange(100, dtype=np.int64), self.augmentation)
        kind = "depot_actions" if self.architecture == "pomo" else "giant_tours"
        return CandidateBatch(kind, sequences, augmentations, policies, elapsed / batch_size, batch_size)


def augment_am_batch(batch: dict[str, torch.Tensor], factor: int) -> dict[str, torch.Tensor]:
    if factor == 1:
        return batch
    if factor != 8:
        raise ValueError("augmentation must be 1 or 8")

    def augment_xy(xy: torch.Tensor) -> torch.Tensor:
        x, y = xy[..., [0]], xy[..., [1]]
        return torch.cat(
            (
                torch.cat((x, y), -1),
                torch.cat((1 - x, y), -1),
                torch.cat((x, 1 - y), -1),
                torch.cat((1 - x, 1 - y), -1),
                torch.cat((y, x), -1),
                torch.cat((1 - y, x), -1),
                torch.cat((y, 1 - x), -1),
                torch.cat((1 - y, 1 - x), -1),
            ),
            dim=0,
        )

    output = {
        "depot": augment_xy(batch["depot"][:, None])[:, 0],
        "loc": augment_xy(batch["loc"]),
    }
    for key, value in batch.items():
        if key not in output:
            output[key] = value.repeat((factor,) + (1,) * (value.dim() - 1))
    return output


class AMRunner:
    def __init__(
        self,
        repo: Path,
        spec: dict[str, Any],
        checkpoint: Path,
        device: torch.device,
        augmentation: int,
    ) -> None:
        self.architecture = spec["architecture"]
        self.device = device
        self.augmentation = augmentation
        cvrp = repo / "NEW_py_ver" / "CVRP"
        am_root = cvrp / "AM_SPLIT"
        for path in (am_root, cvrp, cvrp.parent):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        from nets.attention_model import set_decode_type
        from utils.functions import load_model

        self.model, model_args = load_model(str(checkpoint))
        expected = "cvrp" if self.architecture == "am" else "am_split"
        if self.model.problem.NAME != expected:
            raise ValueError(
                f"{checkpoint}: expected non-TW AM problem {expected}, got {self.model.problem.NAME}"
            )
        training_size = int(spec["training_size"])
        if int(model_args["graph_size"]) != training_size:
            raise ValueError(
                f"{checkpoint}: args.json graph_size={model_args['graph_size']} does not match "
                f"declared training_size={training_size}"
            )
        if self.architecture == "am_split":
            self.model.problem.configure(capacity=1.0, train_reward="split")
        self.model.to(device).eval()
        set_decode_type(self.model, "greedy")

    def decode(self, instances: Sequence[CVRPInstance]) -> CandidateBatch:
        batch_size = len(instances)
        depot, nodes, demand = normalized_problem_batch(instances, self.device)
        batch = {"depot": depot[:, 0], "loc": nodes, "demand": demand}
        batch = augment_am_batch(batch, self.augmentation)
        synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            # AttentionModel.forward(return_pi=True) decodes ``pi`` and then
            # computes a training cost/log-likelihood before returning it.  In
            # AM-Split that discarded cost invokes a floating CUDA Split with a
            # nondeterministic cumsum.  Use the model's own greedy decoding
            # kernel directly; both direct and Split sequences are subsequently
            # scored and verified by the evaluator's independent exact integer
            # EUC_2D implementation.
            embeddings, _ = self.model.embedder(self.model._init_embed(batch))
            _, tours = self.model._inner(batch, embeddings)
        synchronize(self.device)
        elapsed = time.perf_counter() - started
        selected = tours.detach().cpu().numpy().astype(np.int64, copy=False)
        sequences = [
            np.stack(
                [selected[augmentation * batch_size + index] for augmentation in range(self.augmentation)]
            )
            for index in range(batch_size)
        ]
        augmentations = np.arange(self.augmentation, dtype=np.int64)
        policies = np.zeros(self.augmentation, dtype=np.int64)
        kind = "depot_actions" if self.architecture == "am" else "giant_tours"
        return CandidateBatch(kind, sequences, augmentations, policies, elapsed / batch_size, batch_size)


def select_depot_actions(
    instance: CVRPInstance,
    sequences: np.ndarray,
    augmentation_indices: np.ndarray,
    policy_indices: np.ndarray,
    augmentation_filter: int | None = None,
) -> Selection:
    if sequences.ndim != 2 or sequences.shape[0] == 0:
        raise ValueError("empty or malformed direct candidate tensor")
    if np.any(sequences < 0) or np.any(sequences > instance.customers):
        raise ValueError("direct candidate contains an out-of-range action")
    matrix = instance.distance_matrix
    # Decoder action lists omit the implicit initial depot and may also omit a
    # final depot.  Closing the raw sequence with np.roll would incorrectly add
    # last-customer -> first-customer whenever both endpoints are customers.
    # Explicit depot padding exactly represents depot -> actions -> depot; any
    # depot zeros already present in the actions naturally delimit routes.
    depot_column = np.zeros((sequences.shape[0], 1), dtype=np.int64)
    previous = np.concatenate((depot_column, sequences), axis=1)
    following = np.concatenate((sequences, depot_column), axis=1)
    costs = matrix[previous, following].sum(axis=1, dtype=np.int64)
    pool = np.arange(len(costs), dtype=np.int64)
    if augmentation_filter is not None:
        pool = pool[augmentation_indices == augmentation_filter]
    if len(pool) == 0:
        raise ValueError(f"no candidates for augmentation {augmentation_filter}")
    order = pool[np.lexsort((pool, costs[pool]))]
    errors: list[str] = []
    for checked, candidate_index in enumerate(order, start=1):
        routes = actions_to_routes(sequences[candidate_index])
        verification = verify_routes(instance, routes)
        if verification.feasible and verification.cost == int(costs[candidate_index]):
            return Selection(
                routes=routes,
                cost=int(verification.cost),
                vehicles=verification.vehicles,
                candidate_index=int(candidate_index),
                augmentation_index=int(augmentation_indices[candidate_index]),
                policy_index=int(policy_indices[candidate_index]),
                candidates_generated=len(pool),
                candidates_checked_until_selection=checked,
            )
        if len(errors) < 5:
            errors.extend(verification.errors[: 5 - len(errors)])
    raise NoFeasibleCandidateError(
        f"no feasible direct candidate; first errors: {errors[:5]}"
    )


def select_giant_tours(
    instance: CVRPInstance,
    sequences: np.ndarray,
    augmentation_indices: np.ndarray,
    policy_indices: np.ndarray,
    augmentation_filter: int | None = None,
) -> Selection:
    tours, costs, predecessor = split_giant_tour_dp(instance, sequences)
    return select_precomputed_split(
        instance,
        tours,
        costs,
        predecessor,
        augmentation_indices,
        policy_indices,
        augmentation_filter,
    )


def select_precomputed_split(
    instance: CVRPInstance,
    tours: np.ndarray,
    costs: np.ndarray,
    predecessor: np.ndarray,
    augmentation_indices: np.ndarray,
    policy_indices: np.ndarray,
    augmentation_filter: int | None = None,
) -> Selection:
    if len(tours) != len(costs) or len(costs) != len(augmentation_indices):
        raise ValueError("precomputed Split candidate metadata length mismatch")
    pool = np.arange(len(costs), dtype=np.int64)
    if augmentation_filter is not None:
        pool = pool[augmentation_indices == augmentation_filter]
    if len(pool) == 0:
        raise ValueError(f"no candidates for augmentation {augmentation_filter}")
    local_index = min(
        range(len(pool)), key=lambda index: (int(costs[pool[index]]), int(pool[index]))
    )
    candidate_index = int(pool[local_index])
    selected_routes = reconstruct_split_routes(
        instance,
        tours,
        costs,
        predecessor,
        [candidate_index],
    )[0]
    selected_cost = int(costs[candidate_index])
    verification = verify_routes(instance, selected_routes)
    if not verification.feasible or verification.cost != selected_cost:
        raise AssertionError("selected Split solution failed independent verification")
    return Selection(
        routes=selected_routes,
        cost=selected_cost,
        vehicles=verification.vehicles,
        candidate_index=int(candidate_index),
        augmentation_index=int(augmentation_indices[candidate_index]),
        policy_index=int(policy_indices[candidate_index]),
        candidates_generated=len(pool),
        candidates_checked_until_selection=len(pool),
    )


def result_path(output: Path, model_name: str, instance_name: str) -> Path:
    return output / "solutions" / model_name / f"{instance_name}.json"


def validate_existing_result(
    path: Path,
    spec: dict[str, Any],
    instance_name: str,
    checkpoint_sha256: str,
) -> None:
    """Fail closed instead of silently resuming from a mixed/corrupt result."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot resume from invalid result {path}") from exc
    expected = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "CVRPLIB XML100",
        "protocol": PROTOCOL,
        "model": spec["name"],
        "architecture": spec["architecture"],
        "training_size": int(spec["training_size"]),
        "instance": instance_name,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{path}: resume mismatch for {key}: {payload.get(key)!r} != {value!r}")
    if payload.get("status") not in {"ok", "no_feasible_candidate"}:
        raise ValueError(f"{path}: unsupported result status {payload.get('status')!r}")
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint.get("sha256") != checkpoint_sha256:
        raise ValueError(f"{path}: resume checkpoint hash mismatch")
    if payload.get("status") == "ok" and not payload.get("routes"):
        raise ValueError(f"{path}: successful result has no routes")
    if payload.get("no_aug_status") not in {"ok", "no_feasible_candidate"}:
        raise ValueError(f"{path}: missing or invalid no-augmentation status")
    if payload.get("no_aug_status") == "ok" and not payload.get("no_aug_routes"):
        raise ValueError(f"{path}: successful no-augmentation result has no routes")


def flatten_result(
    payload: dict[str, Any], route_path: Path, output_root: Path
) -> dict[str, Any]:
    features = payload["features"]
    checkpoint = payload["checkpoint"]
    return {
        "schema_version": payload["schema_version"],
        "dataset": payload["dataset"],
        "protocol": payload["protocol"],
        "model": payload["model"],
        "architecture": payload["architecture"],
        "training_size": payload["training_size"],
        "instance": payload["instance"],
        "group": features["group"],
        "depot_code": features["depot_code"],
        "depot_type": features["depot_type"],
        "customer_code": features["customer_code"],
        "customer_type": features["customer_type"],
        "demand_code": features["demand_code"],
        "demand_type": features["demand_type"],
        "route_size_code": features["route_size_code"],
        "route_size_type": features["route_size_type"],
        "status": payload["status"],
        "failure_reason": payload.get("failure_reason", ""),
        "cost": payload.get("cost", ""),
        "optimum": payload.get("optimum", ""),
        "optimum_sources_agree": payload.get("optimum_sources_agree", ""),
        "absolute_error": payload.get("absolute_error", ""),
        "gap_percent": payload.get("gap_percent", ""),
        "vehicles": payload.get("vehicles", ""),
        "no_aug_status": payload.get("no_aug_status", ""),
        "no_aug_failure_reason": payload.get("no_aug_failure_reason", ""),
        "no_aug_cost": payload.get("no_aug_cost", ""),
        "no_aug_absolute_error": payload.get("no_aug_absolute_error", ""),
        "no_aug_gap_percent": payload.get("no_aug_gap_percent", ""),
        "no_aug_vehicles": payload.get("no_aug_vehicles", ""),
        "no_aug_candidate_index": payload.get("no_aug_candidate_index", ""),
        "no_aug_policy_index": payload.get("no_aug_policy_index", ""),
        "no_aug_candidates_generated": payload.get("no_aug_candidates_generated", ""),
        "candidates_generated": payload.get("candidates_generated", ""),
        "candidate_index": payload.get("candidate_index", ""),
        "augmentation_index": payload.get("augmentation_index", ""),
        "policy_index": payload.get("policy_index", ""),
        "inference_seconds_amortized": payload.get("inference_seconds_amortized", ""),
        "postprocess_seconds": payload.get("postprocess_seconds", ""),
        "physical_batch_size": payload.get("physical_batch_size", ""),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_path": checkpoint["path"],
        "routes_json": route_path.resolve().relative_to(output_root.resolve()).as_posix(),
    }


def consolidate_results(
    output: Path,
    specs: Sequence[dict[str, Any]],
    instance_names: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_names = set(instance_names)
    for spec in specs:
        directory = output / "solutions" / spec["name"]
        actual_names = {path.stem for path in directory.glob("XML100_*.json")}
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)[:5]
            extra = sorted(actual_names - expected_names)[:5]
            raise ValueError(
                f"{spec['name']}: result-set contamination/missing rows; "
                f"missing={missing}, extra={extra}"
            )
        for instance_name in sorted(expected_names):
            path = directory / f"{instance_name}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append(flatten_result(payload, path, output))
    rows.sort(key=lambda row: (row["model"], row["instance"]))
    atomic_csv(output / "xml100_results.csv", rows)
    return rows


def build_runner(
    repo: Path,
    spec: dict[str, Any],
    checkpoint: Path,
    device: torch.device,
    augmentation: int,
):
    if spec["architecture"].startswith("pomo"):
        return POMORunner(repo, spec, checkpoint, device, augmentation)
    return AMRunner(repo, spec, checkpoint, device, augmentation)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="NCOrethinking repository")
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=here / "data")
    parser.add_argument("--dataset-manifest", type=Path, default=here / "xml100_dataset_manifest.json")
    parser.add_argument("--optimal-costs", type=Path, default=here / "xml100_optimal_costs.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--augmentation", type=int, choices=(1, 8), default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch-size-pomo", type=int, default=8)
    parser.add_argument("--batch-size-am", type=int, default=64)
    parser.add_argument("--only-model", action="append", default=[])
    parser.add_argument("--only-instance", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None, help="smoke test only; not a complete benchmark")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expected-repo-commit", default=None)
    parser.add_argument("--expected-repo-branch", default=None)
    parser.add_argument("--require-clean-repo", action="store_true")
    parser.add_argument("--require-checkpoint-hashes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in ("repo", "models", "data_root", "dataset_manifest", "optimal_costs", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.batch_size_pomo <= 0 or args.batch_size_am <= 0:
        raise ValueError("batch sizes must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    data_manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    if data_manifest.get("instances") != 10_000 or data_manifest.get("solutions") != 10_000:
        raise ValueError("dataset manifest is not a complete validated XML100 release")
    if data_manifest.get("archive_sha256") != XML100_ARCHIVE_SHA256:
        raise ValueError("dataset manifest does not identify the pinned official XML100 archive")
    if data_manifest.get("optimal_cost_crosscheck") != (
        "all .sol declared costs exactly match OptimalCosts.ods"
    ):
        raise ValueError("dataset optimum cross-check is absent or unexpected")
    expected_optima_hash = data_manifest.get("optimal_costs_file_sha256")
    if expected_optima_hash and sha256_file(args.optimal_costs) != expected_optima_hash:
        raise ValueError("optimal-cost CSV hash does not match the validated dataset manifest")
    optimal_costs = read_optimal_costs(args.optimal_costs)
    instances_dir = args.data_root / "XML" / "instances"
    solutions_dir = args.data_root / "XML" / "solutions"
    all_instance_paths = sorted(instances_dir.glob("XML100_*.vrp"))
    if len(all_instance_paths) != 10_000:
        raise ValueError(f"expected 10,000 XML100 instances, found {len(all_instance_paths)}")
    if set(optimal_costs) != {path.stem for path in all_instance_paths}:
        raise ValueError("optimal-cost CSV names do not exactly match the XML100 instances")
    requested_instances = {name.upper() for name in args.only_instance}
    instance_paths = [
        path for path in all_instance_paths if not requested_instances or path.stem.upper() in requested_instances
    ]
    if requested_instances - {path.stem.upper() for path in instance_paths}:
        raise ValueError("one or more --only-instance names were not found")
    if args.limit is not None:
        instance_paths = instance_paths[: args.limit]

    config = json.loads(args.models.read_text(encoding="utf-8"))
    enabled_config_names = [spec["name"] for spec in config["models"] if spec.get("enabled", True)]
    if len(enabled_config_names) != len(set(enabled_config_names)):
        raise ValueError("enabled model names in the configuration must be unique")
    unknown_requested_models = set(args.only_model) - set(enabled_config_names)
    if unknown_requested_models:
        raise ValueError(f"unknown or disabled --only-model values: {sorted(unknown_requested_models)}")
    specs = [
        spec
        for spec in config["models"]
        if spec.get("enabled", True) and (not args.only_model or spec["name"] in args.only_model)
    ]
    if not specs:
        raise ValueError("no enabled models selected")
    names = [spec["name"] for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("model names must be unique")
    for spec in specs:
        if spec.get("architecture") not in ARCHITECTURES:
            raise ValueError(f"unsupported architecture in {spec}")
        training_size = int(spec.get("training_size", 0))
        if training_size not in (50, 100):
            raise ValueError(
                f"{spec['name']}: training_size must be explicitly declared as 50 or 100"
            )
        if args.require_checkpoint_hashes and not spec.get("expected_sha256"):
            raise ValueError(f"{spec['name']}: formal run requires expected_sha256 in models JSON")
        if spec.get("expected_sha256") and re.fullmatch(
            r"[0-9a-fA-F]{64}", str(spec["expected_sha256"])
        ) is None:
            raise ValueError(f"{spec['name']}: expected_sha256 must be exactly 64 hexadecimal digits")

    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing to overwrite non-empty output {args.output}; use --resume")
    args.output.mkdir(parents=True, exist_ok=True)

    checkpoint_by_model: dict[str, Path] = {}
    metadata_by_model: dict[str, dict[str, Any]] = {}
    for spec in specs:
        checkpoint = resolve_checkpoint(args.models, spec["checkpoint"])
        checkpoint_by_model[spec["name"]] = checkpoint
        metadata_by_model[spec["name"]] = checkpoint_metadata(checkpoint, spec, args.repo)

    source_repo = git_metadata(args.repo)
    if args.expected_repo_commit and source_repo.get("commit") != args.expected_repo_commit:
        raise ValueError(
            f"repository commit {source_repo.get('commit')} != expected {args.expected_repo_commit}"
        )
    if args.expected_repo_branch and source_repo.get("branch") != args.expected_repo_branch:
        raise ValueError(
            f"repository branch {source_repo.get('branch')} != expected {args.expected_repo_branch}"
        )
    if args.require_clean_repo and source_repo.get("status_porcelain") != []:
        raise ValueError("repository is unavailable or not clean")

    selected_instance_names = [path.stem for path in instance_paths]
    full_model_set_selected = set(names) == set(enabled_config_names)
    effective_batch_sizes = {
        spec["name"]: int(
            spec.get(
                "batch_size",
                args.batch_size_pomo
                if spec["architecture"].startswith("pomo")
                else args.batch_size_am,
            )
        )
        for spec in specs
    }
    if any(value <= 0 for value in effective_batch_sizes.values()):
        raise ValueError("all effective model batch sizes must be positive")

    # Validate every architecture/checkpoint pairing before emitting any result.
    # This prevents a later mislabeled checkpoint from leaving a plausible but
    # incomplete formal table produced by earlier models.
    preflight_load_seconds: dict[str, float] = {}
    for spec in specs:
        started = time.perf_counter()
        preflight_runner = build_runner(
            args.repo,
            spec,
            checkpoint_by_model[spec["name"]],
            device,
            args.augmentation,
        )
        preflight_load_seconds[spec["name"]] = time.perf_counter() - started
        del preflight_runner
        if device.type == "cuda":
            torch.cuda.empty_cache()

    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "CVRPLIB XML100",
        "protocol": PROTOCOL,
        "protocol_provenance": {
            "upstream_url": MVMOE_UPSTREAM_URL,
            "upstream_commit": MVMOE_UPSTREAM_COMMIT,
            "upstream_tester_sha256": sha256_file(
                Path(__file__).resolve().parent / "upstream" / "Routing-MVMoE" / "Tester.py"
            ),
            "ported_elements": ["greedy/argmax", "8-fold", "POMO size n", "fixed XML scaler 1000"],
            "scoring_hardening": (
                "upstream CVRPEnv already rounds each edge when loc_scaler is set; this port "
                "recomputes the same EUC_2D objective with CPU integer arithmetic, preserves "
                "routes, and independently verifies every reported result"
            ),
        },
        "complete_benchmark_requested": len(instance_paths) == 10_000 and full_model_set_selected,
        "instances_requested": len(instance_paths),
        "instance_names_sha256": sha256_lines(selected_instance_names),
        "dataset_identity": {
            "archive_sha256": data_manifest.get("archive_sha256"),
            "aggregate_file_sha256": data_manifest.get("aggregate_file_sha256"),
            "optimal_costs_file_sha256": sha256_file(args.optimal_costs),
        },
        "models_requested": names,
        "models_config_sha256": sha256_file(args.models),
        "model_protocol_identity": [
            {
                "name": spec["name"],
                "architecture": spec["architecture"],
                "training_size": int(spec.get("training_size", 100)),
                "model_params": spec.get("model_params", {}),
                "checkpoint_sha256": metadata_by_model[spec["name"]]["sha256"],
            }
            for spec in specs
        ],
        "effective_batch_sizes": effective_batch_sizes,
        "checkpoint_architecture_preflight": {
            "status": "all_selected_models_loaded_strictly_before_result_emission",
            "load_seconds": preflight_load_seconds,
        },
        "augmentation": args.augmentation,
        "augmentation_transforms": (
            ["(x,y)"]
            if args.augmentation == 1
            else [
                "(x,y)",
                "(1-x,y)",
                "(x,1-y)",
                "(1-x,1-y)",
                "(y,x)",
                "(1-y,x)",
                "(y,1-x)",
                "(1-y,1-x)",
            ]
        ),
        "decode_type": "greedy/argmax",
        "pomo_size": 100,
        "candidate_selection": "minimum official integer EUC_2D total distance",
        "postprocessing": "none for direct; exact capacity Split for Split architectures",
        "forbidden_methods": ["sampling", "beam_search", "repair", "local_search", "TTO"],
        "coordinate_normalization": "fixed divisor 1000 for model input only",
        "official_scoring": "nint Euclidean distance per edge, then sum",
        "official_optimum_source": (
            "OptimalCosts.ods exported to a hash-locked CSV and cross-checked against every .sol Cost"
        ),
        "official_solutions_used_during_decode": False,
        "seed": args.seed,
        "device": str(device),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "determinism": {
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "execution_identity": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
        },
        "dataset_manifest": data_manifest,
        "source_repo": source_repo,
        "evaluator_source_sha256": evaluator_source_hashes(Path(__file__).resolve().parent),
        "models": [
            {**spec, "checkpoint": metadata_by_model[spec["name"]]} for spec in specs
        ],
        "started_at_utc": utc_now(),
    }
    existing_manifest = args.output / "run_manifest.json"
    if args.resume and existing_manifest.exists():
        previous = json.loads(existing_manifest.read_text(encoding="utf-8"))
        immutable = (
            "dataset",
            "protocol",
            "augmentation",
            "seed",
            "models_requested",
            "models_config_sha256",
            "instances_requested",
            "instance_names_sha256",
            "dataset_identity",
            "model_protocol_identity",
            "effective_batch_sizes",
            "execution_identity",
            "cuda_visible_devices",
            "determinism",
            "evaluator_source_sha256",
        )
        for key in immutable:
            if previous.get(key) != run_manifest.get(key):
                raise ValueError(f"resume manifest mismatch for {key}")
        previous_hashes = {
            item["name"]: item["checkpoint"]["sha256"] for item in previous["models"]
        }
        current_hashes = {
            item["name"]: item["checkpoint"]["sha256"] for item in run_manifest["models"]
        }
        if previous_hashes != current_hashes:
            raise ValueError("resume checkpoint hashes do not match")
        run_manifest["started_at_utc"] = previous["started_at_utc"]
        run_manifest["resumed_at_utc"] = utc_now()
    atomic_json(existing_manifest, run_manifest)

    wall_started = time.perf_counter()
    model_invocation_timings: dict[str, dict[str, Any]] = {}
    for spec in specs:
        model_wall_started = time.perf_counter()
        model_name = spec["name"]
        checkpoint = checkpoint_by_model[model_name]
        checkpoint_meta = metadata_by_model[model_name]
        pending: list[Path] = []
        for path in instance_paths:
            existing = result_path(args.output, model_name, path.stem)
            if existing.exists():
                validate_existing_result(
                    existing,
                    spec,
                    path.stem,
                    checkpoint_meta["sha256"],
                )
            else:
                pending.append(path)
        if not pending:
            print(f"{model_name}: already complete for requested instances", flush=True)
            model_invocation_timings[model_name] = {
                "new_instances": 0,
                "model_load_seconds": 0.0,
                "wall_seconds": time.perf_counter() - model_wall_started,
                "note": "all requested result JSONs already existed and were validated",
            }
            continue
        load_started = time.perf_counter()
        runner = build_runner(args.repo, spec, checkpoint, device, args.augmentation)
        load_seconds = time.perf_counter() - load_started
        batch_size = effective_batch_sizes[model_name]
        print(
            f"{model_name}: loaded {checkpoint.name} in {load_seconds:.2f}s; "
            f"pending={len(pending)} batch={batch_size}",
            flush=True,
        )
        for offset in range(0, len(pending), batch_size):
            paths = pending[offset : offset + batch_size]
            instances = [parse_instance(path) for path in paths]
            candidate_batch = runner.decode(instances)
            for instance, sequences in zip(instances, candidate_batch.sequences):
                post_started = time.perf_counter()
                split_tours: np.ndarray | None = None
                split_costs: np.ndarray | None = None
                split_predecessor: np.ndarray | None = None
                try:
                    if candidate_batch.kind == "depot_actions":
                        selection = select_depot_actions(
                            instance,
                            sequences,
                            candidate_batch.augmentation_indices,
                            candidate_batch.policy_indices,
                        )
                    else:
                        split_tours, split_costs, split_predecessor = split_giant_tour_dp(
                            instance,
                            sequences,
                        )
                        selection = select_precomputed_split(
                            instance,
                            split_tours,
                            split_costs,
                            split_predecessor,
                            candidate_batch.augmentation_indices,
                            candidate_batch.policy_indices,
                        )
                except NoFeasibleCandidateError as error:
                    official = parse_solution(solutions_dir / f"{instance.name}.sol")
                    optimum = optimal_costs[instance.name]
                    if official.declared_cost != optimum:
                        raise ValueError(
                            f"{instance.name}: current .sol Cost {official.declared_cost} "
                            f"does not match hash-locked optimum {optimum}"
                        )
                    failure = {
                        "schema_version": SCHEMA_VERSION,
                        "dataset": "CVRPLIB XML100",
                        "protocol": PROTOCOL,
                        "status": "no_feasible_candidate",
                        "failure_reason": str(error),
                        "model": model_name,
                        "architecture": spec["architecture"],
                        "training_size": int(spec["training_size"]),
                        "instance": instance.name,
                        "features": instance.features.as_dict(),
                        "capacity": instance.capacity,
                        "customers": instance.customers,
                        "optimum": optimum,
                        "optimum_sources_agree": True,
                        "no_aug_status": "no_feasible_candidate",
                        "no_aug_failure_reason": "augmented pool has no feasible candidate",
                        "no_aug_routes": [],
                        "candidates_generated": int(sequences.shape[0]),
                        "inference_seconds_amortized": candidate_batch.inference_seconds_amortized,
                        "postprocess_seconds": time.perf_counter() - post_started,
                        "physical_batch_size": candidate_batch.physical_batch_size,
                        "routes": [],
                        "checkpoint": checkpoint_meta,
                        "created_at_utc": utc_now(),
                    }
                    atomic_json(result_path(args.output, model_name, instance.name), failure)
                    continue
                if args.augmentation == 1:
                    no_aug_selection = selection
                    no_aug_error: str | None = None
                else:
                    try:
                        if candidate_batch.kind == "depot_actions":
                            no_aug_selection = select_depot_actions(
                                instance,
                                sequences,
                                candidate_batch.augmentation_indices,
                                candidate_batch.policy_indices,
                                augmentation_filter=0,
                            )
                        else:
                            if (
                                split_tours is None
                                or split_costs is None
                                or split_predecessor is None
                            ):
                                raise AssertionError("missing precomputed Split candidates")
                            no_aug_selection = select_precomputed_split(
                                instance,
                                split_tours,
                                split_costs,
                                split_predecessor,
                                candidate_batch.augmentation_indices,
                                candidate_batch.policy_indices,
                                augmentation_filter=0,
                            )
                        no_aug_error = None
                    except NoFeasibleCandidateError as error:
                        no_aug_selection = None
                        no_aug_error = str(error)
                # Read the proven optimum only after candidate generation and selection.
                official = parse_solution(solutions_dir / f"{instance.name}.sol")
                optimum = optimal_costs[instance.name]
                if official.declared_cost != optimum:
                    raise ValueError(
                        f"{instance.name}: current .sol Cost {official.declared_cost} "
                        f"does not match hash-locked optimum {optimum}"
                    )
                final_verification = verify_routes(instance, selection.routes)
                if not final_verification.feasible or final_verification.cost != selection.cost:
                    raise AssertionError(f"{model_name}/{instance.name}: final route verification failed")
                if selection.cost < optimum:
                    raise ValueError(
                        f"{model_name}/{instance.name}: cost {selection.cost} is below proven optimum "
                        f"{optimum}; scoring or route indexing is wrong"
                    )
                post_seconds = time.perf_counter() - post_started
                payload = {
                    "schema_version": SCHEMA_VERSION,
                    "dataset": "CVRPLIB XML100",
                    "protocol": PROTOCOL,
                    "status": "ok",
                    "model": model_name,
                    "architecture": spec["architecture"],
                    "training_size": int(spec["training_size"]),
                    "instance": instance.name,
                    "features": instance.features.as_dict(),
                    "capacity": instance.capacity,
                    "customers": instance.customers,
                    "cost": selection.cost,
                    "optimum": optimum,
                    "optimum_sources_agree": True,
                    "absolute_error": selection.cost - optimum,
                    "gap_percent": gap_percent(selection.cost, optimum),
                    "vehicles": selection.vehicles,
                    "routes": selection.routes,
                    "no_aug_status": (
                        "ok" if no_aug_selection is not None else "no_feasible_candidate"
                    ),
                    "no_aug_failure_reason": no_aug_error,
                    "no_aug_cost": no_aug_selection.cost if no_aug_selection is not None else None,
                    "no_aug_absolute_error": (
                        no_aug_selection.cost - optimum if no_aug_selection is not None else None
                    ),
                    "no_aug_gap_percent": (
                        gap_percent(no_aug_selection.cost, optimum)
                        if no_aug_selection is not None
                        else None
                    ),
                    "no_aug_vehicles": (
                        no_aug_selection.vehicles if no_aug_selection is not None else None
                    ),
                    "no_aug_routes": (
                        no_aug_selection.routes if no_aug_selection is not None else []
                    ),
                    "no_aug_candidate_index": (
                        no_aug_selection.candidate_index if no_aug_selection is not None else None
                    ),
                    "no_aug_policy_index": (
                        no_aug_selection.policy_index if no_aug_selection is not None else None
                    ),
                    "no_aug_candidates_generated": (
                        no_aug_selection.candidates_generated if no_aug_selection is not None else None
                    ),
                    "candidate_index": selection.candidate_index,
                    "augmentation_index": selection.augmentation_index,
                    "policy_index": selection.policy_index,
                    "candidates_generated": selection.candidates_generated,
                    "candidates_checked_until_selection": selection.candidates_checked_until_selection,
                    "candidate_feasibility_scope": "selected candidate independently verified",
                    "inference_seconds_amortized": candidate_batch.inference_seconds_amortized,
                    "postprocess_seconds": post_seconds,
                    "physical_batch_size": candidate_batch.physical_batch_size,
                    "checkpoint": checkpoint_meta,
                    "created_at_utc": utc_now(),
                }
                atomic_json(result_path(args.output, model_name, instance.name), payload)
            done = min(offset + len(paths), len(pending))
            print(f"{model_name}: {done}/{len(pending)} newly evaluated", flush=True)
        del runner
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model_invocation_timings[model_name] = {
            "new_instances": len(pending),
            "model_load_seconds": load_seconds,
            "wall_seconds": time.perf_counter() - model_wall_started,
        }

    rows = consolidate_results(args.output, specs, selected_instance_names)
    expected_rows = len(specs) * len(instance_paths)
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} result rows, found {len(rows)}")
    run_manifest["completed_at_utc"] = utc_now()
    run_manifest["wall_seconds_this_invocation"] = time.perf_counter() - wall_started
    run_manifest["model_invocation_timings"] = model_invocation_timings
    run_manifest["result_rows"] = len(rows)
    run_manifest["status"] = (
        "complete" if len(instance_paths) == 10_000 and full_model_set_selected else "partial"
    )
    atomic_json(existing_manifest, run_manifest)
    print(f"saved {len(rows)} rows to {args.output / 'xml100_results.csv'}", flush=True)


if __name__ == "__main__":
    main()
