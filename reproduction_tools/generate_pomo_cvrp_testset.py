"""Generate an auditable fixed CVRP test set with POMO's official generator."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


DEFAULT_POMO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_RELATIVE_PATH = Path("NEW_py_ver/CVRP/CVRProblemDef.py")
LOADER_RELATIVE_PATH = Path("NEW_py_ver/CVRP/POMO/CVRPEnv.py")
DEMAND_SCALERS = {20: 30, 50: 40, 100: 50}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata(pomo_root):
    rev = subprocess.run(
        ["git", "-C", str(pomo_root), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    status = subprocess.run(
        ["git", "-C", str(pomo_root), "status", "--porcelain"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    return {
        "git_commit": rev.stdout.strip() if rev.returncode == 0 else "unavailable",
        "git_dirty": status.returncode != 0 or bool(status.stdout.strip()),
    }


def validate_pomo_root(pomo_root):
    generator_source = pomo_root / GENERATOR_RELATIVE_PATH
    loader_source = pomo_root / LOADER_RELATIVE_PATH
    missing = [path for path in (generator_source, loader_source) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "POMO root is missing required official source file(s): {}".format(
                ", ".join(str(path) for path in missing)
            )
        )
    return generator_source, loader_source


def add_import_paths(pomo_root):
    cvrp_root = pomo_root / "NEW_py_ver" / "CVRP"
    pomo_dir = cvrp_root / "POMO"
    for path in reversed((cvrp_root, pomo_dir)):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def validate_tensors(data, num_instances, problem_size):
    expected_shapes = {
        "depot_xy": (num_instances, 1, 2),
        "node_xy": (num_instances, problem_size, 2),
        "node_demand": (num_instances, problem_size),
    }
    if set(data) != set(expected_shapes):
        raise ValueError(
            "CVRP test set keys are {}, expected {}.".format(
                sorted(data), sorted(expected_shapes)
            )
        )

    for name, expected_shape in expected_shapes.items():
        tensor = data[name]
        if tuple(tensor.shape) != expected_shape:
            raise RuntimeError(
                "{} has shape {}, expected {}.".format(
                    name, tuple(tensor.shape), expected_shape
                )
            )
        if tensor.device.type != "cpu":
            raise RuntimeError("{} was not generated on CPU.".format(name))
        if not torch.is_floating_point(tensor):
            raise TypeError("{} must be floating-point.".format(name))
        if not torch.isfinite(tensor).all().item():
            raise ValueError("{} contains NaN or Inf.".format(name))

    coordinate_min = min(
        data["depot_xy"].min().item(), data["node_xy"].min().item()
    )
    coordinate_max = max(
        data["depot_xy"].max().item(), data["node_xy"].max().item()
    )
    if coordinate_min < 0.0 or coordinate_max > 1.0:
        raise ValueError(
            "CVRP coordinates are outside [0,1]: min={}, max={}.".format(
                coordinate_min, coordinate_max
            )
        )

    demand = data["node_demand"]
    demand_scaler = DEMAND_SCALERS[problem_size]
    demand_min = demand.min().item()
    demand_max = demand.max().item()
    if demand_min < 1.0 / demand_scaler or demand_max > 9.0 / demand_scaler:
        raise ValueError(
            "CVRP demands are outside the official range: min={}, max={}.".format(
                demand_min, demand_max
            )
        )
    scaled_demand = demand * demand_scaler
    demand_grid_valid = torch.allclose(
        scaled_demand, scaled_demand.round(), rtol=0.0, atol=1e-6
    )
    if not demand_grid_valid:
        raise ValueError("CVRP demands are not on the official integer/scaler grid.")

    return {
        "shapes": {name: list(tensor.shape) for name, tensor in data.items()},
        "dtypes": {name: str(tensor.dtype) for name, tensor in data.items()},
        "coordinate_min": coordinate_min,
        "coordinate_max": coordinate_max,
        "demand_min": demand_min,
        "demand_max": demand_max,
        "demand_scaler": demand_scaler,
        "demand_grid_valid": demand_grid_valid,
    }


def tensors_equal(left, right):
    return all(torch.equal(left[name], right[name]) for name in left)


def main():
    parser = argparse.ArgumentParser(
        description="Generate fixed CVRP data with POMO NEW_py_ver's official generator."
    )
    parser.add_argument(
        "--pomo-root",
        default=str(DEFAULT_POMO_ROOT),
        help="POMO repository root (default: repository containing this script).",
    )
    parser.add_argument(
        "--problem-size", type=int, required=True, choices=sorted(DEMAND_SCALERS)
    )
    parser.add_argument("--num-instances", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing test set and its metadata sidecar.",
    )
    args = parser.parse_args()

    if args.num_instances <= 0:
        raise ValueError("--num-instances must be positive.")

    pomo_root = Path(args.pomo_root).expanduser().resolve()
    generator_source, loader_source = validate_pomo_root(pomo_root)
    add_import_paths(pomo_root)
    from CVRProblemDef import get_random_problems
    from CVRPEnv import CVRPEnv

    output = Path(args.output).expanduser().resolve()
    sidecar = output.with_name(output.name + ".metadata.json")
    existing = [path for path in (output, sidecar) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing file(s): {}. Pass --overwrite explicitly.".format(
                ", ".join(str(path) for path in existing)
            )
        )

    # The official generator uses the global RNG and has no device argument.
    # Force the default tensor type to CPU immediately before seeding/generation.
    torch.set_default_tensor_type("torch.FloatTensor")
    torch.manual_seed(args.seed)
    depot_xy, node_xy, node_demand = get_random_problems(
        args.num_instances, args.problem_size
    )
    generated = {
        "depot_xy": depot_xy,
        "node_xy": node_xy,
        "node_demand": node_demand,
    }
    validation = validate_tensors(
        generated, args.num_instances, args.problem_size
    )

    # Verify determinism against a second call to the same official generator.
    torch.manual_seed(args.seed)
    regenerated_tuple = get_random_problems(args.num_instances, args.problem_size)
    regenerated = dict(zip(("depot_xy", "node_xy", "node_demand"), regenerated_tuple))
    regeneration_equal = tensors_equal(generated, regenerated)
    if not regeneration_equal:
        raise RuntimeError("Official generator was not reproducible after reseeding.")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(generated, str(output))
    testset_sha256 = sha256_file(output)

    reloaded = torch.load(str(output), map_location=torch.device("cpu"))
    validate_tensors(reloaded, args.num_instances, args.problem_size)
    roundtrip_equal = tensors_equal(generated, reloaded)
    if not roundtrip_equal:
        raise RuntimeError("Saved and reloaded CVRP tensors differ under torch.equal.")

    # Exercise the official fixed-data loader without running model inference.
    env = CVRPEnv(problem_size=args.problem_size, pomo_size=args.problem_size)
    env.use_saved_problems(str(output), torch.device("cpu"))
    loader_data = {
        "depot_xy": env.saved_depot_xy,
        "node_xy": env.saved_node_xy,
        "node_demand": env.saved_node_demand,
    }
    loader_roundtrip_equal = tensors_equal(generated, loader_data)
    if not loader_roundtrip_equal:
        raise RuntimeError("CVRPEnv.use_saved_problems changed one or more tensors.")

    metadata = {
        "format_version": 1,
        "problem": "CVRP",
        "problem_size": args.problem_size,
        "num_instances": args.num_instances,
        "seed": args.seed,
        "testset_path": str(output),
        "testset_sha256": testset_sha256,
        "size_bytes": output.stat().st_size,
        "generator": "POMO NEW_py_ver/CVRP/CVRProblemDef.get_random_problems",
        "generator_source": str(GENERATOR_RELATIVE_PATH),
        "generator_source_sha256": sha256_file(generator_source),
        "loader": "POMO NEW_py_ver/CVRP/POMO/CVRPEnv.use_saved_problems",
        "loader_source": str(LOADER_RELATIVE_PATH),
        "loader_source_sha256": sha256_file(loader_source),
        "distribution": (
            "independent uniform depot/customer coordinates in [0,1]^2; "
            "integer customer demands 1..9 divided by {}"
        ).format(DEMAND_SCALERS[args.problem_size]),
        "provenance": (
            "Reproduction test set generated with the official POMO generator; "
            "not the paper authors' original test instances."
        ),
        "generation_device": "cpu",
        "loaded_tensor_device": "cpu",
        "regeneration_equal_torch_equal": regeneration_equal,
        "roundtrip_equal_torch_equal": roundtrip_equal,
        "official_loader_roundtrip_equal_torch_equal": loader_roundtrip_equal,
        "validation_method": (
            "Generate twice after torch.manual_seed(seed), save/load on CPU, "
            "then load through official CVRPEnv.use_saved_problems; compare every "
            "tensor with torch.equal."
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    metadata.update(validation)
    metadata.update(git_metadata(pomo_root))

    with sidecar.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    print("Metadata sidecar: {}".format(sidecar), flush=True)
    print("Metadata SHA256: {}".format(sha256_file(sidecar)), flush=True)


if __name__ == "__main__":
    main()
