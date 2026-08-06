"""Demonstrate whether a giant-tour checkpoint can react to customer demand.

Coordinates are held fixed while demands are reassigned among customers. The
policy is rolled out twice with argmax decoding, then both tours are evaluated
with the same optimal Split decoder.
"""

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


DEFAULT_MODEL_PARAMS = {
    "embedding_dim": 128,
    "sqrt_embedding_dim": 128 ** 0.5,
    "encoder_layer_num": 6,
    "qkv_dim": 16,
    "head_num": 8,
    "logit_clipping": 10,
    "ff_hidden_dim": 512,
    "eval_type": "argmax",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(source, module_name):
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def configure_device(name, cuda_device):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        torch.cuda.set_device(cuda_device)
        return torch.device("cuda", cuda_device)
    return torch.device("cpu")


def rollout(model, depot_xy, node_xy, node_demand, pomo_size):
    batch_size, problem_size, _ = node_xy.shape
    state = SimpleNamespace()
    state.BATCH_IDX = torch.arange(device=node_xy.device, end=batch_size)[
        :, None
    ].expand(batch_size, pomo_size)
    state.POMO_IDX = torch.arange(device=node_xy.device, end=pomo_size)[
        None, :
    ].expand(batch_size, pomo_size)
    state.current_node = None
    state.ninf_mask = torch.zeros(
        batch_size,
        pomo_size,
        problem_size + 1,
        device=node_xy.device,
        dtype=node_xy.dtype,
    )
    state.ninf_mask[:, :, 0] = float("-inf")

    reset_state = SimpleNamespace(
        depot_xy=depot_xy, node_xy=node_xy, node_demand=node_demand
    )
    model.pre_forward(reset_state)
    encoded = model.encoded_nodes.detach().clone()
    selected_list = []
    for _ in range(problem_size):
        selected, _ = model(state)
        if (selected < 1).any() or (selected > problem_size).any():
            raise AssertionError("Policy selected depot or an invalid customer ID")
        selected_list.append(selected)
        state.current_node = selected
        state.ninf_mask[state.BATCH_IDX, state.POMO_IDX, selected] = float("-inf")
    tours = torch.stack(selected_list, dim=2)

    expected = torch.arange(
        1, problem_size + 1, device=node_xy.device, dtype=torch.long
    ).view(1, 1, problem_size)
    if not torch.equal(tours.sort(dim=2).values, expected.expand_as(tours)):
        raise AssertionError("A rollout is not a permutation of all customers")
    return tours, encoded


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test demand sensitivity of a giant-tour POMO checkpoint."
    )
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--model-class", default="GiantTourModel")
    parser.add_argument("--split-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--pomo-size", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = {
        "model_source": args.model_source.resolve(),
        "split_source": args.split_source.resolve(),
        "checkpoint": args.checkpoint.resolve(),
        "test_data": args.test_data.resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")

    device = configure_device(args.device, args.cuda_device)
    checkpoint_data = torch.load(str(paths["checkpoint"]), map_location=device)
    test_data = torch.load(str(paths["test_data"]), map_location=device)
    depot_xy = test_data["depot_xy"][: args.episodes]
    node_xy = test_data["node_xy"][: args.episodes]
    demand_a = test_data["node_demand"][: args.episodes]
    if node_xy.size(1) < args.pomo_size:
        raise ValueError("pomo_size cannot exceed problem_size")

    # Preserve each instance's demand multiset while changing which customer owns
    # each demand. Coordinates and depot stay bit-for-bit identical.
    demand_b = torch.roll(demand_a, shifts=1, dims=1)
    changed_demand_elements = int((demand_a != demand_b).sum().item())
    if changed_demand_elements == 0:
        raise AssertionError("Demand reassignment did not change the selected cases")

    model_module = load_module(paths["model_source"], "giant_tour_model_under_audit")
    if not hasattr(model_module, args.model_class):
        raise AttributeError(
            f"{paths['model_source']} does not define {args.model_class}"
        )
    model_class = getattr(model_module, args.model_class)
    model_params = dict(checkpoint_data.get("model_params", DEFAULT_MODEL_PARAMS))
    model_params["eval_type"] = "argmax"
    model = model_class(**model_params).to(device)
    load_result = model.load_state_dict(checkpoint_data["model_state_dict"], strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise AssertionError(
            f"Checkpoint mismatch: missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )
    model.eval()

    with torch.no_grad():
        tours_a, encoded_a = rollout(
            model, depot_xy, node_xy, demand_a, args.pomo_size
        )
        tours_b, encoded_b = rollout(
            model, depot_xy, node_xy, demand_b, args.pomo_size
        )

    encoded_difference = (encoded_a - encoded_b).abs()
    tour_difference_count = int((tours_a != tours_b).sum().item())

    split_module = load_module(paths["split_source"], "split_for_demand_audit")
    with torch.no_grad():
        costs_a = split_module.split_giant_tours(
            depot_xy, node_xy, demand_a, tours_a, capacity=1.0
        ).costs
        costs_b = split_module.split_giant_tours(
            depot_xy, node_xy, demand_b, tours_b, capacity=1.0
        ).costs
    cost_difference = (costs_a - costs_b).abs()
    best_a = costs_a.min(dim=1).values
    best_b = costs_b.min(dim=1).values

    report = {
        "status": "passed",
        "interpretation": (
            "Demands changed while coordinates stayed fixed. Zero encoded/tour "
            "difference proves this checkpoint cannot condition its customer "
            "ordering on demand; Split costs may still change afterward."
        ),
        "model_source": str(paths["model_source"]),
        "model_source_sha256": sha256_file(paths["model_source"]),
        "model_class": args.model_class,
        "split_source": str(paths["split_source"]),
        "split_source_sha256": sha256_file(paths["split_source"]),
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": sha256_file(paths["checkpoint"]),
        "test_data": str(paths["test_data"]),
        "test_data_sha256": sha256_file(paths["test_data"]),
        "episodes": args.episodes,
        "problem_size": int(node_xy.size(1)),
        "pomo_size": args.pomo_size,
        "changed_demand_elements": changed_demand_elements,
        "coordinates_bitwise_equal_between_runs": True,
        "max_encoded_abs_difference": float(encoded_difference.max().item()),
        "different_tour_elements": tour_difference_count,
        "tours_bitwise_equal": bool(tour_difference_count == 0),
        "split_cost_elements_changed": int((cost_difference > 1e-7).sum().item()),
        "mean_absolute_split_cost_change": float(cost_difference.float().mean().item()),
        "mean_best_pomo_cost_demand_a": float(best_a.float().mean().item()),
        "mean_best_pomo_cost_demand_b": float(best_b.float().mean().item()),
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    if report["max_encoded_abs_difference"] != 0 or not report["tours_bitwise_equal"]:
        raise AssertionError(
            "Policy changed after demand-only reassignment; inspect model inputs "
            "instead of labeling this checkpoint demand-blind"
        )

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
