#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from solomon_io import (
    SolomonInstance,
    actions_to_routes,
    normalized_tensors,
    parse_instance,
    split_giant_tour_double,
    verify_routes,
)


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

PROTOCOL_MVMOE = "mvmoe_table7_greedy_8fold_distance"
PROTOCOL_STRICT = "strict_fleet_lexicographic"


@dataclass
class Candidate:
    candidate_index: int
    augmentation_index: int
    policy_index: int
    routes: list[list[int]]
    vehicles: int
    distance_raw: float
    distance_2dp: str


class NoFeasibleCandidateError(RuntimeError):
    pass


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_candidates(
    instance: SolomonInstance,
    routes: list[tuple[int, int, list[list[int]]]],
    *,
    enforce_fleet_limit: bool,
) -> tuple[list[Candidate], Candidate, Candidate]:
    feasible: list[Candidate] = []
    for index, (aug, policy, candidate_routes) in enumerate(routes):
        verification = verify_routes(
            instance,
            candidate_routes,
            enforce_fleet_limit=enforce_fleet_limit,
        )
        if verification.feasible:
            feasible.append(
                Candidate(
                    candidate_index=index,
                    augmentation_index=aug,
                    policy_index=policy,
                    routes=candidate_routes,
                    vehicles=verification.vehicles,
                    distance_raw=verification.distance_raw,
                    distance_2dp=verification.distance_2dp,
                )
            )
    if not feasible:
        raise NoFeasibleCandidateError(f"{instance.name}: no externally feasible candidate")
    distance_only = min(feasible, key=lambda c: (c.distance_raw, c.vehicles, c.candidate_index))
    official = (
        min(feasible, key=lambda c: (c.vehicles, c.distance_raw, c.candidate_index))
        if enforce_fleet_limit
        else distance_only
    )
    return feasible, distance_only, official


def _pomo_imports(repo: Path, architecture: str):
    cvrp = repo / "NEW_py_ver" / "CVRP"
    work = cvrp / ("POMO_TW" if architecture == "pomo_tw" else "POMO_SPLIT_TW")
    for path in (work, cvrp, cvrp.parent):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    if architecture == "pomo_tw":
        from POMO_TW.VRPTWEnv import VRPTWEnv as Env
        from POMO_TW.VRPTWModel import VRPTWModel as Model
        return Env, Model, None
    from CVRPTWCore import reconstruct_routes
    from POMO_SPLIT.GiantTourModel import GiantTourModel as Model
    from POMO_SPLIT_TW.GiantTourTWEnv import GiantTourTWEnv as Env
    return Env, Model, reconstruct_routes


def decode_pomo(
    repo: Path,
    architecture: str,
    checkpoint: Path,
    instance: SolomonInstance,
    device: torch.device,
    augmentation: int,
    scale: float,
    enforce_fleet_limit: bool,
) -> tuple[list[Candidate], Candidate, Candidate]:
    Env, Model, _ = _pomo_imports(repo, architecture)
    env = Env(
        problem_size=instance.customers,
        pomo_size=instance.customers,
        capacity=1.0,
        speed=1.0,
        depot_start=instance.depot.ready / scale,
        depot_end=instance.depot.due / scale,
        service_duration=0.0,
        epsilon=1e-7,
        loc_scaler=None,
        device=device,
    )
    model_params = dict(MODEL_PARAMS)
    if architecture == "pomo_split_tw":
        model_params["node_feature_dim"] = 6
    model = Model(**model_params).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    problems = tuple(t.to(device) for t in normalized_tensors(instance, scale))
    with torch.no_grad():
        env.load_problems(1, aug_factor=augmentation, problems=problems)
        reset, _, _ = env.reset()
        model.pre_forward(reset)
        step, reward, done = env.pre_step()
        dead_rollouts = torch.zeros(
            (augmentation, instance.customers), dtype=torch.bool, device=device
        )
        while not done:
            selected, _ = model(step)
            mask = step.ninf_mask
            invalid = mask[env.BATCH_IDX, env.POMO_IDX, selected].isneginf()
            newly_dead = invalid & ~dead_rollouts
            if newly_dead.any():
                dead_rollouts |= newly_dead
                selected = selected.clone()
                if architecture == "pomo_tw":
                    # One dead POMO rollout must not discard the other 399/799
                    # candidates. Mark it finished and exclude it below.
                    env.visited_ninf_flag[newly_dead] = float("-inf")
                    env.finished[newly_dead] = True
                    env.ninf_mask[newly_dead] = float("-inf")
                    env.ninf_mask[:, :, 0][newly_dead] = 0
                    selected[newly_dead] = 0
                    env._update_step_state()
                else:
                    finite = torch.isfinite(mask)
                    if bool((newly_dead & ~finite.any(dim=2)).any()):
                        raise RuntimeError("dead giant-tour rollout has no unvisited fallback")
                    fallback = finite.to(torch.int64).argmax(dim=2)
                    selected[newly_dead] = fallback[newly_dead]
            step, reward, done = env.step(selected)

    candidates: list[tuple[int, int, list[list[int]]]] = []
    for aug in range(augmentation):
        for policy in range(instance.customers):
            if bool(dead_rollouts[aug, policy]):
                continue
            if architecture == "pomo_tw":
                actions = env.selected_node_list[aug, policy].detach().cpu().tolist()
                routes = actions_to_routes(actions)
            else:
                tour = env.selected_node_list[aug, policy].detach().cpu().tolist()
                try:
                    routes = split_giant_tour_double(
                        instance,
                        tour,
                        enforce_fleet_limit=enforce_fleet_limit,
                    )
                except ValueError:
                    routes = [tour]  # retained as an externally rejected candidate
            candidates.append((aug, policy, routes))
    return choose_candidates(
        instance,
        candidates,
        enforce_fleet_limit=enforce_fleet_limit,
    )


def augment_am_batch(batch: dict[str, torch.Tensor], factor: int) -> dict[str, torch.Tensor]:
    if factor == 1:
        return batch
    if factor != 8:
        raise ValueError("AM augmentation must be 1 or 8")

    def aug_xy(xy: torch.Tensor) -> torch.Tensor:
        x, y = xy[..., [0]], xy[..., [1]]
        return torch.cat(
            (
                torch.cat((x, y), -1), torch.cat((1 - x, y), -1),
                torch.cat((x, 1 - y), -1), torch.cat((1 - x, 1 - y), -1),
                torch.cat((y, x), -1), torch.cat((1 - y, x), -1),
                torch.cat((y, 1 - x), -1), torch.cat((1 - y, 1 - x), -1),
            ),
            0,
        )

    output = {"depot": aug_xy(batch["depot"][:, None])[:, 0], "loc": aug_xy(batch["loc"])}
    for key, value in batch.items():
        if key not in output:
            output[key] = value.repeat((8,) + (1,) * (value.dim() - 1))
    return output


def decode_am(
    repo: Path,
    architecture: str,
    checkpoint: Path,
    instance: SolomonInstance,
    device: torch.device,
    augmentation: int,
    scale: float,
    enforce_fleet_limit: bool,
) -> tuple[list[Candidate], Candidate, Candidate]:
    cvrp = repo / "NEW_py_ver" / "CVRP"
    am = cvrp / "AM_SPLIT"
    for path in (am, cvrp, cvrp.parent):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.chdir(am)
    from nets.attention_model import set_decode_type
    from utils.functions import load_model

    model, model_args = load_model(str(checkpoint))
    expected = "cvrptw" if architecture == "am_tw" else "am_split_tw"
    if model.problem.NAME != expected:
        raise ValueError(f"{checkpoint}: expected {expected}, got {model.problem.NAME}")
    if int(model_args["graph_size"]) != instance.customers:
        raise ValueError(
            f"{checkpoint}: graph_size={model_args['graph_size']} does not match {instance.customers}"
        )
    model.problem.configure(
        capacity=1.0,
        depot_start=instance.depot.ready / scale,
        depot_end=instance.depot.due / scale,
        speed=1.0,
        service_duration=0.0,
        loc_scaler=None,
        epsilon=1e-7,
        train_reward="split",
    )
    tensors = tuple(t.to(device) for t in normalized_tensors(instance, scale))
    batch = {
        "depot": tensors[0][:, 0],
        "loc": tensors[1],
        "demand": tensors[2],
        "service_time": tensors[3],
        "tw_start": tensors[4],
        "tw_end": tensors[5],
        "depot_start": torch.tensor([instance.depot.ready / scale], device=device),
        "depot_end": torch.tensor([instance.depot.due / scale], device=device),
        "speed": torch.tensor([1.0], device=device),
    }
    batch = augment_am_batch(batch, augmentation)
    model.to(device).eval()
    set_decode_type(model, "greedy")
    with torch.no_grad():
        _, _, tours = model(batch, return_pi=True)

    candidates: list[tuple[int, int, list[list[int]]]] = []
    for aug in range(augmentation):
        if architecture == "am_tw":
            routes = actions_to_routes(tours[aug].detach().cpu().tolist())
        else:
            tour = tours[aug].detach().cpu().tolist()
            try:
                routes = split_giant_tour_double(
                    instance,
                    tour,
                    enforce_fleet_limit=enforce_fleet_limit,
                )
            except ValueError:
                routes = [tour]  # retained as an externally rejected candidate
        candidates.append((aug, 0, routes))
    return choose_candidates(
        instance,
        candidates,
        enforce_fleet_limit=enforce_fleet_limit,
    )


def load_reference(path: Path | None) -> dict[tuple[int, str], dict[str, Any]]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (int(row["size"]), row["instance"].upper()): row
            for row in csv.DictReader(handle)
        }


def safe_gap(value: float, reference: float) -> float:
    return 100.0 * (value - reference) / reference


def mvmoe_coordinate_scale(instance: SolomonInstance) -> float:
    """Match the public MVMoE Solomon loader's instance-specific scaling."""
    coordinate_max = max(max(node.x, node.y) for node in instance.nodes)
    return max(float(coordinate_max), float(instance.depot.due) / 3.0)


def result_row(
    model_spec: dict[str, Any],
    instance: SolomonInstance,
    checkpoint: Path,
    feasible_count: int,
    candidate_count: int,
    distance: Candidate,
    official: Candidate,
    sintef: dict[str, Any] | None,
    legacy: dict[str, Any] | None,
    elapsed: float,
    protocol: str,
    coordinate_scale: float,
) -> dict[str, Any]:
    sintef_vehicles = int(sintef["vehicles"]) if sintef else None
    sintef_distance = float(sintef["distance"]) if sintef else None
    legacy_vehicles = int(legacy["vehicles"]) if legacy else None
    legacy_distance = float(legacy["distance"]) if legacy else None
    return {
        "protocol": protocol,
        "model": model_spec["name"],
        "architecture": model_spec["architecture"],
        "trained_size": model_spec["size"],
        "checkpoint": str(checkpoint.resolve()),
        "instance": instance.name,
        "family": (instance.name[:3] if instance.name.startswith("RC") else instance.name[:2]),
        "customers": instance.customers,
        "max_fleet": instance.max_vehicles,
        "fleet_limit_enforced": protocol == PROTOCOL_STRICT,
        "coordinate_scale": f"{coordinate_scale:.12f}",
        "candidates_total": candidate_count,
        "candidates_externally_feasible": feasible_count,
        "selected_vehicles": distance.vehicles,
        "selected_distance_raw": f"{distance.distance_raw:.12f}",
        "selected_distance_2dp": distance.distance_2dp,
        "selected_exceeds_reported_fleet": distance.vehicles > instance.max_vehicles,
        "distance_only_vehicles": distance.vehicles,
        "distance_only_raw": f"{distance.distance_raw:.12f}",
        "distance_only_2dp": distance.distance_2dp,
        "official_lex_vehicles": official.vehicles,
        "official_lex_raw": f"{official.distance_raw:.12f}",
        "official_lex_2dp": official.distance_2dp,
        "same_candidate": distance.candidate_index == official.candidate_index,
        "sintef_double_bks_vehicles": sintef_vehicles,
        "sintef_double_bks_distance": sintef_distance,
        "sintef_vehicle_gap": official.vehicles - sintef_vehicles if sintef else None,
        "sintef_official_distance_gap_pct": (
            safe_gap(official.distance_raw, sintef_distance)
            if sintef and official.vehicles == sintef_vehicles else None
        ),
        "sintef_nonofficial_distance_only_gap_pct": (
            safe_gap(distance.distance_raw, sintef_distance) if sintef else None
        ),
        "legacy_reference_vehicles": legacy_vehicles,
        "legacy_reference_distance": legacy_distance,
        "legacy_distance_only_gap_pct_noncomparable": (
            safe_gap(distance.distance_raw, legacy_distance) if legacy else None
        ),
        "elapsed_seconds": elapsed,
        "status": "ok",
    }


def failure_row(
    model_spec: dict[str, Any],
    instance: SolomonInstance,
    checkpoint: Path,
    candidate_count: int,
    sintef: dict[str, Any] | None,
    legacy: dict[str, Any] | None,
    elapsed: float,
    status: str = "no_externally_feasible_candidate",
    protocol: str = PROTOCOL_MVMOE,
    coordinate_scale: float = 100.0,
) -> dict[str, Any]:
    sintef_vehicles = int(sintef["vehicles"]) if sintef else None
    sintef_distance = float(sintef["distance"]) if sintef else None
    legacy_vehicles = int(legacy["vehicles"]) if legacy else None
    legacy_distance = float(legacy["distance"]) if legacy else None
    return {
        "protocol": protocol,
        "model": model_spec["name"],
        "architecture": model_spec["architecture"],
        "trained_size": model_spec["size"],
        "checkpoint": str(checkpoint.resolve()),
        "instance": instance.name,
        "family": (instance.name[:3] if instance.name.startswith("RC") else instance.name[:2]),
        "customers": instance.customers,
        "max_fleet": instance.max_vehicles,
        "fleet_limit_enforced": protocol == PROTOCOL_STRICT,
        "coordinate_scale": f"{coordinate_scale:.12f}",
        "candidates_total": candidate_count,
        "candidates_externally_feasible": 0,
        "selected_vehicles": None,
        "selected_distance_raw": None,
        "selected_distance_2dp": None,
        "selected_exceeds_reported_fleet": None,
        "distance_only_vehicles": None,
        "distance_only_raw": None,
        "distance_only_2dp": None,
        "official_lex_vehicles": None,
        "official_lex_raw": None,
        "official_lex_2dp": None,
        "same_candidate": None,
        "sintef_double_bks_vehicles": sintef_vehicles,
        "sintef_double_bks_distance": sintef_distance,
        "sintef_vehicle_gap": None,
        "sintef_official_distance_gap_pct": None,
        "sintef_nonofficial_distance_only_gap_pct": None,
        "legacy_reference_vehicles": legacy_vehicles,
        "legacy_reference_distance": legacy_distance,
        "legacy_distance_only_gap_pct_noncomparable": None,
        "elapsed_seconds": elapsed,
        "status": status,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate AM/POMO TW models with the MVMoE Table 7 or strict Solomon protocol."
    )
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sintef-bks", type=Path, default=None)
    parser.add_argument("--legacy-reference", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--augmentation", type=int, choices=(1, 8), default=8)
    parser.add_argument("--coordinate-scale", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--protocol",
        choices=("mvmoe", "strict"),
        default="mvmoe",
        help="mvmoe: greedy distance with no fleet-limit filtering; strict: fleet-limited lexicographic",
    )
    parser.add_argument("--only-model", action="append", default=[])
    parser.add_argument("--only-instance", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    # AM's upstream load_model changes the process working directory.  Freeze
    # every caller-provided path before loading any model so outputs and data
    # remain anchored to this evaluation batch rather than the source tree.
    for name in ("repo", "models", "data_root", "output", "sintef_bks", "legacy_reference"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    seed_all(args.seed)
    protocol = PROTOCOL_MVMOE if args.protocol == "mvmoe" else PROTOCOL_STRICT
    enforce_fleet_limit = args.protocol == "strict"
    device = torch.device(args.device)
    specs = json.loads(args.models.read_text(encoding="utf-8"))["models"]
    sintef = load_reference(args.sintef_bks)
    legacy = load_reference(args.legacy_reference)
    output_csv = args.output / "solomon_results.csv"
    rows: list[dict[str, Any]] = []
    completed = set()
    if args.resume and output_csv.exists():
        with output_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        completed = {(row["model"], row["instance"], int(row["customers"])) for row in rows}

    solution_dir = args.output / "solutions"
    solution_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        if not spec.get("enabled", True) or (args.only_model and spec["name"] not in args.only_model):
            continue
        size = int(spec["size"])
        checkpoint = Path(spec["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        instance_dir = args.data_root / f"instances_{size}"
        for instance_path in sorted(instance_dir.glob("*.txt")):
            instance = parse_instance(instance_path)
            coordinate_scale = (
                mvmoe_coordinate_scale(instance)
                if args.protocol == "mvmoe"
                else args.coordinate_scale
            )
            if args.only_instance and instance.name not in {value.upper() for value in args.only_instance}:
                continue
            key = (spec["name"], instance.name, size)
            if key in completed:
                continue
            started = time.time()
            architecture = spec["architecture"]
            candidate_total = args.augmentation * size if architecture.startswith("pomo") else args.augmentation
            try:
                if architecture.startswith("pomo"):
                    feasible, distance, official = decode_pomo(
                        args.repo, architecture, checkpoint, instance, device,
                        args.augmentation, coordinate_scale, enforce_fleet_limit,
                    )
                else:
                    feasible, distance, official = decode_am(
                        args.repo, architecture, checkpoint, instance, device,
                        args.augmentation, coordinate_scale, enforce_fleet_limit,
                    )
            except NoFeasibleCandidateError:
                row = failure_row(
                    spec, instance, checkpoint, candidate_total,
                    sintef.get((size, instance.name)), legacy.get((size, instance.name)),
                    time.time() - started, protocol=protocol,
                    coordinate_scale=coordinate_scale,
                )
                rows.append(row)
                write_csv(output_csv, rows)
                print(f"{spec['name']} {instance.name} n={size} no externally feasible candidate", flush=True)
                continue
            except ValueError as error:
                status = f"decoder_error_{type(error).__name__}"
                row = failure_row(
                    spec, instance, checkpoint, candidate_total,
                    sintef.get((size, instance.name)), legacy.get((size, instance.name)),
                    time.time() - started, status=status, protocol=protocol,
                    coordinate_scale=coordinate_scale,
                )
                rows.append(row)
                write_csv(output_csv, rows)
                print(
                    f"{spec['name']} {instance.name} n={size} {status}: {error}",
                    flush=True,
                )
                continue
            row = result_row(
                spec, instance, checkpoint, len(feasible), candidate_total,
                distance, official, sintef.get((size, instance.name)),
                legacy.get((size, instance.name)), time.time() - started,
                protocol, coordinate_scale,
            )
            rows.append(row)
            write_csv(output_csv, rows)
            solution_path = solution_dir / f"{spec['name']}__{instance.name}__n{size}.json"
            solution_path.write_text(
                json.dumps(
                    {
                        "model": spec,
                        "instance": instance.name,
                        "customers": size,
                        "protocol": protocol,
                        "selected_distance": asdict(distance),
                        "reported_max_fleet": instance.max_vehicles,
                        "fleet_limit_enforced": enforce_fleet_limit,
                        "coordinate_scale": coordinate_scale,
                        "selection_note": (
                            "MVMoE Table 7 protocol: greedy inference, 8-fold augmentation, "
                            "distance-only selection, parsed Solomon fleet count is not enforced."
                            if not enforce_fleet_limit else
                            "Strict protocol: fleet limit is enforced and lexicographic selection is retained."
                        ),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                f"{spec['name']} {instance.name} n={size} "
                f"selected={distance.vehicles}/{distance.distance_2dp} "
                f"protocol={args.protocol}",
                flush=True,
            )
    print(f"saved {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
