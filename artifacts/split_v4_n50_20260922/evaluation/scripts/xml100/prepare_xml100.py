#!/usr/bin/env python3
"""Download and exhaustively validate the official CVRPLIB XML100 release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

from xml100_io import parse_instance, parse_solution, route_cost, verify_routes


OFFICIAL_ARCHIVE_URL = (
    "https://galgos.inf.puc-rio.br/cvrplib/uploads/files/xml100/XML.7z"
)
OFFICIAL_ARCHIVE_SHA256 = (
    "ef3814ee4c26e7f1d09cf33a4c0da564109de04ee393afa3b46c7eb33473b24b"
)
EXPECTED_INSTANCES = 10_000


def read_optimal_costs_ods(path: Path) -> dict[str, int]:
    """Read the official ODS without adding a spreadsheet dependency."""
    namespaces = {
        "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
    }
    table_namespace = namespaces["table"]
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("content.xml"))
    rows: list[list[str]] = []
    for row in root.findall(".//table:table-row", namespaces):
        values: list[str] = []
        for cell in row.findall("table:table-cell", namespaces):
            repeat = int(cell.attrib.get(f"{{{table_namespace}}}number-columns-repeated", "1"))
            value = "".join(cell.itertext()).strip()
            values.extend([value] * repeat)
        rows.append(values)
    header_index = next(
        index for index, values in enumerate(rows) if values[:2] == ["Instance", "Optimal cost"]
    )
    costs: dict[str, int] = {}
    for values in rows[header_index + 1 :]:
        if not values or not values[0].startswith("XML100_"):
            continue
        if len(values) < 2 or not values[1]:
            raise ValueError(f"{path}: missing optimal cost for {values[0]}")
        if values[0] in costs:
            raise ValueError(f"{path}: duplicate instance {values[0]}")
        numeric = float(values[1])
        if not math.isfinite(numeric) or not numeric.is_integer() or numeric <= 0:
            raise ValueError(f"{path}: invalid integer optimal cost for {values[0]}: {values[1]!r}")
        costs[values[0]] = int(numeric)
    if len(costs) != EXPECTED_INSTANCES:
        raise ValueError(f"{path}: expected {EXPECTED_INSTANCES} optimum rows, got {len(costs)}")
    return costs


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=destination.name + ".", suffix=".partial", dir=destination.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with urllib.request.urlopen(url) as response, temporary_path.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary_path.replace(destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def extract_archive(archive: Path, data_root: Path) -> None:
    try:
        import py7zr
    except ImportError as exc:
        raise RuntimeError("py7zr is required to extract XML.7z") from exc
    data_root.mkdir(parents=True, exist_ok=True)
    if any(data_root.iterdir()):
        raise FileExistsError(
            f"refusing to extract into non-empty directory {data_root}; preserve or move it first"
        )
    with py7zr.SevenZipFile(archive, mode="r") as archive_handle:
        names = archive_handle.getnames()
        for name in names:
            candidate = (data_root / name).resolve()
            if data_root.resolve() not in candidate.parents and candidate != data_root.resolve():
                raise ValueError(f"unsafe archive member: {name}")
        archive_handle.extractall(path=data_root)


def locate_data(data_root: Path) -> tuple[Path, Path]:
    candidates = [data_root / "XML", data_root]
    for candidate in candidates:
        instances = candidate / "instances"
        solutions = candidate / "solutions"
        if instances.is_dir() and solutions.is_dir():
            return instances, solutions
    raise FileNotFoundError(f"could not find XML/instances and XML/solutions below {data_root}")


def validate_dataset(data_root: Path, archive: Path | None = None) -> dict:
    instances_dir, solutions_dir = locate_data(data_root)
    instance_paths = sorted(instances_dir.glob("XML100_*.vrp"))
    solution_paths = sorted(solutions_dir.glob("XML100_*.sol"))
    instance_names = {path.stem for path in instance_paths}
    solution_names = {path.stem for path in solution_paths}
    if len(instance_paths) != EXPECTED_INSTANCES or len(solution_paths) != EXPECTED_INSTANCES:
        raise ValueError(
            f"expected {EXPECTED_INSTANCES} instance/solution files, got "
            f"{len(instance_paths)}/{len(solution_paths)}"
        )
    if instance_names != solution_names:
        raise ValueError(
            f"unpaired files: missing solutions={sorted(instance_names-solution_names)[:10]}, "
            f"missing instances={sorted(solution_names-instance_names)[:10]}"
        )
    ods_path = instances_dir.parent / "OptimalCosts.ods"
    if not ods_path.is_file():
        raise FileNotFoundError(ods_path)
    ods_costs = read_optimal_costs_ods(ods_path)
    if set(ods_costs) != instance_names:
        raise ValueError("OptimalCosts.ods names do not exactly match the XML100 instances")

    group_counts: Counter[str] = Counter()
    aggregate = hashlib.sha256()
    total_optimum = 0
    min_optimum: int | None = None
    max_optimum: int | None = None
    reference_route_anomalies: list[dict] = []
    feasible_reference_routes = 0
    for index, instance_path in enumerate(instance_paths, start=1):
        name = instance_path.stem
        solution_path = solutions_dir / f"{name}.sol"
        instance = parse_instance(instance_path)
        solution = parse_solution(solution_path)
        verification = verify_routes(instance, solution.routes)
        recomputed_route_cost = route_cost(instance, solution.routes)
        if solution.declared_cost != ods_costs[name]:
            raise ValueError(
                f"{name}: .sol cost {solution.declared_cost} != ODS optimum {ods_costs[name]}"
            )
        anomaly: dict | None = None
        if not verification.feasible:
            anomaly = {
                "instance": name,
                "errors": list(verification.errors),
                "declared_cost": solution.declared_cost,
                "route_cost_even_if_infeasible": recomputed_route_cost,
            }
        elif verification.cost != solution.declared_cost:
            anomaly = {
                "instance": name,
                "errors": ["feasible reference route cost does not match declared optimum"],
                "declared_cost": solution.declared_cost,
                "recomputed_cost": verification.cost,
            }
        else:
            feasible_reference_routes += 1
        if anomaly is not None:
            reference_route_anomalies.append(anomaly)
        group_counts[instance.features.group] += 1
        optimum = solution.declared_cost
        total_optimum += optimum
        min_optimum = optimum if min_optimum is None else min(min_optimum, optimum)
        max_optimum = optimum if max_optimum is None else max(max_optimum, optimum)
        for path in (instance_path, solution_path):
            file_hash = sha256_file(path)
            aggregate.update(path.name.encode("utf-8"))
            aggregate.update(b"\0")
            aggregate.update(file_hash.encode("ascii"))
            aggregate.update(b"\n")
        if index % 1000 == 0:
            print(f"validated {index}/{EXPECTED_INSTANCES}", flush=True)

    if len(group_counts) != 378:
        raise ValueError(f"expected 378 XML100 groups, got {len(group_counts)}")
    counts_of_counts = Counter(group_counts.values())
    if counts_of_counts != Counter({27: 172, 26: 206}):
        raise ValueError(f"unexpected group cardinalities: {dict(counts_of_counts)}")

    return {
        "schema_version": 1,
        "dataset": "CVRPLIB XML100",
        "official_url": OFFICIAL_ARCHIVE_URL,
        "archive": str(archive.resolve()) if archive else None,
        "archive_sha256": sha256_file(archive) if archive else None,
        "expected_archive_sha256": OFFICIAL_ARCHIVE_SHA256,
        "instances_dir": str(instances_dir.resolve()),
        "solutions_dir": str(solutions_dir.resolve()),
        "instances": len(instance_paths),
        "solutions": len(solution_paths),
        "groups": len(group_counts),
        "group_cardinality_distribution": {
            str(size): groups for size, groups in sorted(counts_of_counts.items())
        },
        "aggregate_file_sha256": aggregate.hexdigest(),
        "total_optimum": total_optimum,
        "minimum_optimum": min_optimum,
        "maximum_optimum": max_optimum,
        "optimal_cost_crosscheck": "all .sol declared costs exactly match OptimalCosts.ods",
        "feasible_reference_route_files": feasible_reference_routes,
        "reference_route_anomaly_count": len(reference_route_anomalies),
        "reference_route_anomalies": reference_route_anomalies,
        "model_route_verification_policy": (
            "reference-route anomalies are never accepted for model output; every model route "
            "must be independently feasible and exactly match per-edge EUC_2D cost"
        ),
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=here / "downloads" / "XML.7z")
    parser.add_argument("--data-root", type=Path, default=here / "data")
    parser.add_argument("--manifest", type=Path, default=here / "xml100_dataset_manifest.json")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--print-full-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = args.archive.resolve()
    data_root = args.data_root.resolve()
    if not archive.exists():
        if args.no_download:
            raise FileNotFoundError(archive)
        print(f"downloading {OFFICIAL_ARCHIVE_URL} -> {archive}")
        download(OFFICIAL_ARCHIVE_URL, archive)
    archive_hash = sha256_file(archive)
    if archive_hash != OFFICIAL_ARCHIVE_SHA256:
        raise ValueError(
            f"official archive hash mismatch: expected {OFFICIAL_ARCHIVE_SHA256}, got {archive_hash}"
        )
    try:
        locate_data(data_root)
    except FileNotFoundError:
        if args.no_extract:
            raise
        extract_archive(archive, data_root)
    manifest = validate_dataset(data_root, archive)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    instances_dir, _ = locate_data(data_root)
    optimal_costs = read_optimal_costs_ods(instances_dir.parent / "OptimalCosts.ods")
    optima_path = args.manifest.with_name("xml100_optimal_costs.csv")
    optima_temporary = optima_path.with_suffix(optima_path.suffix + ".tmp")
    with optima_temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["instance", "optimum"])
        writer.writeheader()
        for name in sorted(optimal_costs):
            writer.writerow({"instance": name, "optimum": optimal_costs[name]})
    optima_temporary.replace(optima_path)
    manifest["optimal_costs_file"] = str(optima_path.resolve())
    manifest["optimal_costs_file_sha256"] = sha256_file(optima_path)
    anomaly_path = args.manifest.with_name("xml100_reference_route_anomalies.json")
    anomaly_temporary = anomaly_path.with_suffix(anomaly_path.suffix + ".tmp")
    anomaly_temporary.write_text(
        json.dumps(
            {
                "dataset": "CVRPLIB XML100",
                "archive_sha256": manifest["archive_sha256"],
                "important": (
                    "These are anomalies in released reference-route text only. Official optimal "
                    "values remain usable because every .sol Cost matches OptimalCosts.ods. "
                    "No official file was modified."
                ),
                "count": manifest["reference_route_anomaly_count"],
                "anomalies": manifest["reference_route_anomalies"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    anomaly_temporary.replace(anomaly_path)
    manifest["reference_route_anomaly_file"] = str(anomaly_path.resolve())
    manifest["reference_route_anomaly_file_sha256"] = sha256_file(anomaly_path)
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.manifest)
    if args.print_full_manifest:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "manifest": str(args.manifest.resolve()),
                    "archive_sha256": manifest["archive_sha256"],
                    "instances": manifest["instances"],
                    "solutions": manifest["solutions"],
                    "groups": manifest["groups"],
                    "optimal_cost_crosscheck": manifest["optimal_cost_crosscheck"],
                    "reference_route_anomaly_count": manifest["reference_route_anomaly_count"],
                    "reference_route_anomaly_file": manifest["reference_route_anomaly_file"],
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
