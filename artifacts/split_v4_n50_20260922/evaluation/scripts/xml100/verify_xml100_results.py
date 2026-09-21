#!/usr/bin/env python3
"""Independently verify XML100 result CSV rows and every saved route."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from xml100_io import gap_percent, parse_instance, parse_solution


XML100_ARCHIVE_SHA256 = "ef3814ee4c26e7f1d09cf33a4c0da564109de04ee393afa3b46c7eb33473b24b"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_optimal_costs(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    costs = {row["instance"]: int(row["optimum"]) for row in rows}
    if len(rows) != 10_000 or len(costs) != 10_000 or any(value <= 0 for value in costs.values()):
        raise ValueError(f"{path}: invalid official optimum table")
    return costs


def independent_nint_euclidean(a: Any, b: Any) -> int:
    """Exact integer implementation, deliberately separate from xml100_io."""
    squared = (int(a[0]) - int(b[0])) ** 2 + (int(a[1]) - int(b[1])) ** 2
    lower = math.isqrt(squared)
    return lower + int(4 * squared >= (2 * lower + 1) ** 2)


def independently_verify_routes(instance: Any, routes: Any) -> tuple[bool, int, int, list[str]]:
    errors: list[str] = []
    seen: list[int] = []
    total = 0
    expected = set(range(1, instance.customers + 1))
    if not isinstance(routes, list):
        return False, 0, 0, ["routes is not a list"]
    for route_index, route in enumerate(routes, start=1):
        if not isinstance(route, list) or not route:
            errors.append(f"route {route_index} is empty or not a list")
            continue
        load = 0
        previous = 0
        for raw_customer in route:
            if isinstance(raw_customer, bool) or not isinstance(raw_customer, int):
                errors.append(f"route {route_index} has non-integer customer")
                continue
            customer = raw_customer
            if customer not in expected:
                errors.append(f"route {route_index} has out-of-range customer {customer}")
                continue
            seen.append(customer)
            load += int(instance.demands[customer])
            total += independent_nint_euclidean(
                instance.coords[previous], instance.coords[customer]
            )
            previous = customer
        total += independent_nint_euclidean(instance.coords[previous], instance.coords[0])
        if load > instance.capacity:
            errors.append(f"route {route_index} exceeds capacity")
    counts = Counter(seen)
    missing = sorted(expected - set(counts))
    duplicates = sorted(customer for customer, count in counts.items() if count > 1)
    if missing:
        errors.append(f"missing customers: {missing}")
    if duplicates:
        errors.append(f"duplicate customers: {duplicates}")
    return not errors, total, len(routes), errors


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=here / "data")
    parser.add_argument("--dataset-manifest", type=Path, default=here / "xml100_dataset_manifest.json")
    parser.add_argument("--optimal-costs", type=Path, default=here / "xml100_optimal_costs.csv")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in ("results", "models", "data_root", "dataset_manifest", "optimal_costs"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    report_path = (
        args.report.expanduser().resolve()
        if args.report
        else args.results.parent / "xml100_verification.json"
    )
    dataset_manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    if dataset_manifest.get("instances") != 10_000 or dataset_manifest.get("solutions") != 10_000:
        raise ValueError("dataset manifest is incomplete")
    expected_optima_hash = dataset_manifest.get("optimal_costs_file_sha256")
    if expected_optima_hash and sha256_file(args.optimal_costs) != expected_optima_hash:
        raise ValueError("optimal-cost CSV hash does not match the dataset manifest")
    optimal_costs = read_optimal_costs(args.optimal_costs)
    config = json.loads(args.models.read_text(encoding="utf-8"))
    model_specs = {
        spec["name"]: spec for spec in config["models"] if spec.get("enabled", True)
    }
    model_names = sorted(model_specs)
    with args.results.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("results CSV is empty")
    keys = [(row["model"], row["instance"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate (model, instance) result rows")
    unexpected = sorted(set(row["model"] for row in rows) - set(model_names))
    if unexpected:
        raise ValueError(f"results contain models absent from config: {unexpected}")
    counts = Counter(row["model"] for row in rows)
    if not args.allow_partial:
        if sorted(counts) != model_names:
            raise ValueError("not every enabled model appears in results")
        bad_counts = {model: counts[model] for model in model_names if counts[model] != 10_000}
        if bad_counts:
            raise ValueError(f"full XML100 requires 10,000 rows per model: {bad_counts}")

    output_root = args.results.parent.resolve()
    run_manifest_path = output_root / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise FileNotFoundError(run_manifest_path)
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("dataset") != "CVRPLIB XML100":
        raise ValueError("run manifest dataset mismatch")
    dataset_identity = run_manifest.get("dataset_identity", {})
    if dataset_identity.get("archive_sha256") != XML100_ARCHIVE_SHA256:
        raise ValueError("run manifest does not identify the pinned official archive")
    if dataset_identity.get("aggregate_file_sha256") != dataset_manifest.get(
        "aggregate_file_sha256"
    ) or dataset_identity.get("optimal_costs_file_sha256") != sha256_file(
        args.optimal_costs
    ):
        raise ValueError("run/dataset manifest identity mismatch")
    source_root = Path(__file__).resolve().parent
    recorded_source_hashes = run_manifest.get("evaluator_source_sha256", {})
    if not recorded_source_hashes:
        raise ValueError("run manifest has no evaluator source hashes")
    for filename, recorded_hash in recorded_source_hashes.items():
        current_path = source_root / filename
        if not current_path.is_file() or sha256_file(current_path) != recorded_hash:
            raise ValueError(f"evaluation source changed since the run: {filename}")
    instances_dir = args.data_root / "XML" / "instances"
    solutions_dir = args.data_root / "XML" / "solutions"
    instance_cache = {}
    solution_cost_cache: dict[str, int] = {}
    status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    checkpoint_hashes: dict[str, set[str]] = defaultdict(set)
    maximum_gap = 0.0
    maximum_no_aug_gap = 0.0
    for index, row in enumerate(rows, start=1):
        model = row["model"]
        name = row["instance"]
        spec = model_specs[model]
        if row.get("architecture") != spec.get("architecture"):
            raise ValueError(f"{model}/{name}: architecture does not match models config")
        if int(row.get("training_size", 0)) != int(spec.get("training_size", 0)):
            raise ValueError(f"{model}/{name}: training_size does not match models config")
        route_path = (output_root / row["routes_json"]).resolve()
        if output_root not in route_path.parents:
            raise ValueError(f"route path escapes output root: {row['routes_json']}")
        if not route_path.is_file():
            raise FileNotFoundError(route_path)
        payload = json.loads(route_path.read_text(encoding="utf-8"))
        for key, expected in (("model", model), ("instance", name), ("status", row["status"])):
            if payload.get(key) != expected:
                raise ValueError(f"{model}/{name}: JSON/CSV mismatch for {key}")
        if payload.get("protocol") != row["protocol"]:
            raise ValueError(f"{model}/{name}: protocol mismatch")
        checkpoint_hash = payload["checkpoint"]["sha256"]
        if checkpoint_hash != row["checkpoint_sha256"]:
            raise ValueError(f"{model}/{name}: checkpoint hash mismatch")
        if len(checkpoint_hash) != 64 or any(character not in "0123456789abcdef" for character in checkpoint_hash.lower()):
            raise ValueError(f"{model}/{name}: malformed checkpoint SHA-256")
        checkpoint_hashes[model].add(checkpoint_hash)
        expected_checkpoint_hash = spec.get("expected_sha256")
        if expected_checkpoint_hash and checkpoint_hash.lower() != expected_checkpoint_hash.lower():
            raise ValueError(f"{model}/{name}: checkpoint hash does not match models config")
        expected_epoch = spec.get("expected_epoch")
        if expected_epoch is not None and int(float(row["checkpoint_epoch"])) != int(expected_epoch):
            raise ValueError(f"{model}/{name}: checkpoint epoch does not match models config")
        status_counts[model][row["status"]] += 1
        group_counts[model][row["group"]] += 1
        for timing_key in ("inference_seconds_amortized", "postprocess_seconds"):
            timing_value = float(row[timing_key])
            if not math.isfinite(timing_value) or timing_value < 0:
                raise ValueError(f"{model}/{name}: invalid {timing_key}")
            if not math.isclose(
                timing_value, float(payload[timing_key]), rel_tol=0.0, abs_tol=1e-15
            ):
                raise ValueError(f"{model}/{name}: JSON/CSV mismatch for {timing_key}")
        physical_batch_size = int(float(row["physical_batch_size"]))
        if physical_batch_size <= 0 or physical_batch_size != int(payload["physical_batch_size"]):
            raise ValueError(f"{model}/{name}: invalid physical batch size")
        candidates_generated = int(float(row["candidates_generated"]))
        if candidates_generated <= 0 or candidates_generated != int(payload["candidates_generated"]):
            raise ValueError(f"{model}/{name}: invalid candidate count")
        base_candidates = 100 if row["architecture"].startswith("pomo") else 1
        expected_candidates = base_candidates * int(run_manifest["augmentation"])
        if candidates_generated != expected_candidates:
            raise ValueError(
                f"{model}/{name}: candidate count {candidates_generated} != {expected_candidates}"
            )

        if name not in instance_cache:
            instance_cache[name] = parse_instance(instances_dir / f"{name}.vrp")
            solution_cost_cache[name] = parse_solution(
                solutions_dir / f"{name}.sol"
            ).declared_cost
        instance = instance_cache[name]
        optimum = optimal_costs[name]
        if solution_cost_cache[name] != optimum:
            raise ValueError(f"{model}/{name}: .sol and ODS-derived optimum disagree")
        if int(float(row["optimum"])) != optimum or int(payload["optimum"]) != optimum:
            raise ValueError(f"{model}/{name}: optimum mismatch")
        if row.get("optimum_sources_agree") != "True" or payload.get("optimum_sources_agree") is not True:
            raise ValueError(f"{model}/{name}: optimum source cross-check not recorded")
        if row["status"] == "ok":
            feasible, verified_cost, vehicles, errors = independently_verify_routes(
                instance, payload["routes"]
            )
            if not feasible:
                raise ValueError(f"{model}/{name}: infeasible routes: {errors}")
            cost = int(float(row["cost"]))
            if verified_cost != cost or payload["cost"] != cost:
                raise ValueError(f"{model}/{name}: official route cost mismatch")
            if cost < optimum:
                raise ValueError(f"{model}/{name}: result is below proven optimum")
            expected_gap = gap_percent(cost, optimum)
            reported_gap = float(row["gap_percent"])
            if not math.isfinite(reported_gap) or not math.isclose(
                reported_gap, expected_gap, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"{model}/{name}: gap mismatch")
            if int(float(row["vehicles"])) != vehicles:
                raise ValueError(f"{model}/{name}: vehicle count mismatch")
            maximum_gap = max(maximum_gap, reported_gap)
        elif row["status"] != "no_feasible_candidate":
            raise ValueError(f"{model}/{name}: unsupported status {row['status']!r}")
        elif payload.get("routes"):
            raise ValueError(f"{model}/{name}: failure result must not contain routes")

        no_aug_status = row.get("no_aug_status")
        if no_aug_status != payload.get("no_aug_status"):
            raise ValueError(f"{model}/{name}: no-augmentation status mismatch")
        if no_aug_status == "ok":
            no_aug_candidates = int(float(row["no_aug_candidates_generated"]))
            if no_aug_candidates <= 0 or no_aug_candidates != int(
                payload["no_aug_candidates_generated"]
            ):
                raise ValueError(f"{model}/{name}: invalid no-augmentation candidate count")
            if no_aug_candidates != base_candidates:
                raise ValueError(f"{model}/{name}: unexpected no-augmentation candidate budget")
            feasible, no_aug_cost, no_aug_vehicles, errors = independently_verify_routes(
                instance, payload.get("no_aug_routes")
            )
            if not feasible:
                raise ValueError(f"{model}/{name}: infeasible no-augmentation routes: {errors}")
            reported_no_aug_cost = int(float(row["no_aug_cost"]))
            if no_aug_cost != reported_no_aug_cost or payload.get("no_aug_cost") != no_aug_cost:
                raise ValueError(f"{model}/{name}: no-augmentation cost mismatch")
            no_aug_gap = gap_percent(no_aug_cost, optimum)
            if not math.isclose(
                float(row["no_aug_gap_percent"]), no_aug_gap, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"{model}/{name}: no-augmentation gap mismatch")
            if int(float(row["no_aug_vehicles"])) != no_aug_vehicles:
                raise ValueError(f"{model}/{name}: no-augmentation vehicle count mismatch")
            if no_aug_cost < optimum:
                raise ValueError(f"{model}/{name}: no-augmentation result is below optimum")
            if row["status"] == "ok" and int(float(row["cost"])) > no_aug_cost:
                raise ValueError(
                    f"{model}/{name}: augmented candidate pool is worse than its no-aug subset"
                )
            maximum_no_aug_gap = max(maximum_no_aug_gap, no_aug_gap)
        elif no_aug_status == "no_feasible_candidate":
            if payload.get("no_aug_routes"):
                raise ValueError(f"{model}/{name}: failed no-augmentation result has routes")
        else:
            raise ValueError(f"{model}/{name}: unsupported no-augmentation status")
        if index % 5000 == 0:
            print(f"verified {index}/{len(rows)}", flush=True)

    if any(len(hashes) != 1 for hashes in checkpoint_hashes.values()):
        raise ValueError("a model was evaluated with more than one checkpoint")
    protocols = {row["protocol"] for row in rows}
    if len(protocols) != 1 or run_manifest.get("protocol") != next(iter(protocols)):
        raise ValueError("result/run-manifest protocol mismatch")
    if int(run_manifest.get("result_rows", -1)) != len(rows):
        raise ValueError("result/run-manifest row-count mismatch")
    manifest_hashes = {
        item["name"]: item["checkpoint"]["sha256"] for item in run_manifest.get("models", [])
    }
    verified_hashes = {
        model: next(iter(hashes)) for model, hashes in checkpoint_hashes.items()
    }
    if manifest_hashes != verified_hashes:
        raise ValueError("result/run-manifest model checkpoint mismatch")
    if not args.allow_partial:
        if run_manifest.get("status") != "complete" or not run_manifest.get(
            "complete_benchmark_requested"
        ):
            raise ValueError("run manifest does not describe a completed full benchmark")
        expected_distribution = Counter({27: 172, 26: 206})
        for model in model_names:
            cardinalities = Counter(group_counts[model].values())
            if cardinalities != expected_distribution:
                raise ValueError(
                    f"{model}: unexpected XML100 group cardinalities {dict(cardinalities)}"
                )

    report = {
        "schema_version": 1,
        "dataset": "CVRPLIB XML100",
        "results": str(args.results),
        "run_manifest": str(run_manifest_path),
        "evaluator_source_sha256": recorded_source_hashes,
        "rows": len(rows),
        "unique_model_instance_rows": len(set(keys)),
        "models": model_names,
        "rows_per_model": dict(sorted(counts.items())),
        "status_counts": {
            model: dict(sorted(values.items())) for model, values in sorted(status_counts.items())
        },
        "checkpoint_sha256": {
            model: next(iter(hashes)) for model, hashes in sorted(checkpoint_hashes.items())
        },
        "route_verification": (
            "every status=ok route was checked by an implementation separate from the evaluator; "
            "it covers every customer exactly once, obeys capacity, and exactly matches an "
            "integer-arithmetic per-edge EUC_2D cost"
        ),
        "maximum_gap_percent": maximum_gap,
        "maximum_no_aug_gap_percent": maximum_no_aug_gap,
        "complete": not args.allow_partial and len(rows) == len(model_names) * 10_000,
    }
    atomic_json(report_path, report)
    print(f"verified {len(rows)} XML100 result rows; report={report_path}")


if __name__ == "__main__":
    main()
