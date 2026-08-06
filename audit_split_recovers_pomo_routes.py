"""Check Split against routes produced by the official POMO CVRP model.

For each official POMO trajectory, depot visits are removed to form a giant
customer tour. The original depot-separated routes are one feasible contiguous
partition of that tour, so an optimal Split cost must be no larger than the
official trajectory cost (up to floating-point tolerance).
"""

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
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


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_import_paths(pomo_root):
    new_py = pomo_root / "NEW_py_ver"
    for path in reversed(
        [new_py, new_py / "CVRP", new_py / "CVRP" / "POMO"]
    ):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def load_split_module(source):
    module_name = "split_decoder_for_pomo_invariant"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import SplitDecoder from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "split_giant_tours"):
        raise AttributeError(f"{source} does not define split_giant_tours")
    return module


def configure_device(name, cuda_device):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        torch.cuda.set_device(cuda_device)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        return torch.device("cuda", cuda_device)
    torch.set_default_tensor_type("torch.FloatTensor")
    return torch.device("cpu")


def customer_only_tours(selected_node_list, problem_size):
    batch_size, pomo_size, _ = selected_node_list.shape
    tours = torch.empty(
        batch_size,
        pomo_size,
        problem_size,
        dtype=torch.long,
        device=selected_node_list.device,
    )
    expected = torch.arange(
        1, problem_size + 1, dtype=torch.long, device=selected_node_list.device
    )
    for batch_index in range(batch_size):
        for pomo_index in range(pomo_size):
            selected = selected_node_list[batch_index, pomo_index]
            customers = selected[selected != 0]
            if customers.numel() != problem_size:
                raise AssertionError(
                    f"Trajectory ({batch_index},{pomo_index}) contains "
                    f"{customers.numel()} customer visits, expected {problem_size}"
                )
            if not torch.equal(customers.sort().values, expected):
                raise AssertionError(
                    f"Trajectory ({batch_index},{pomo_index}) is not a permutation"
                )
            tours[batch_index, pomo_index] = customers
    return tours


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify that optimal Split recovers no-worse POMO route costs."
    )
    parser.add_argument("--pomo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--split-source", type=Path, required=True)
    parser.add_argument("--problem-size", type=int, default=100)
    parser.add_argument("--pomo-size", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=2e-5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    pomo_root = args.pomo_root.resolve()
    checkpoint = args.checkpoint.resolve()
    test_data = args.test_data.resolve()
    split_source = args.split_source.resolve()
    for required in (checkpoint, test_data, split_source):
        if not required.is_file():
            raise FileNotFoundError(required)
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")

    device = configure_device(args.device, args.cuda_device)
    add_import_paths(pomo_root)
    from CVRPEnv import CVRPEnv
    from CVRPModel import CVRPModel

    split_module = load_split_module(split_source)
    env = CVRPEnv(problem_size=args.problem_size, pomo_size=args.pomo_size)
    env.use_saved_problems(str(test_data), device)
    env.load_problems(args.episodes, aug_factor=1)
    reset_state, _, _ = env.reset()

    model = CVRPModel(**MODEL_PARAMS).to(device)
    checkpoint_data = torch.load(str(checkpoint), map_location=device)
    load_result = model.load_state_dict(checkpoint_data["model_state_dict"], strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise AssertionError(
            f"Checkpoint mismatch: missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )
    model.eval()
    model.pre_forward(reset_state)

    state, reward, done = env.pre_step()
    with torch.no_grad():
        while not done:
            selected, _ = model(state)
            state, reward, done = env.step(selected)

    official_costs = -reward
    giant_tours = customer_only_tours(env.selected_node_list, args.problem_size)
    split_result = split_module.split_giant_tours(
        reset_state.depot_xy,
        reset_state.node_xy,
        reset_state.node_demand,
        giant_tours,
        capacity=1.0,
        return_predecessors=False,
    )
    split_costs = split_result.costs
    if not torch.isfinite(official_costs).all() or not torch.isfinite(split_costs).all():
        raise AssertionError("Non-finite route cost encountered")

    difference = split_costs - official_costs
    violation_mask = difference > args.tolerance
    violation_count = int(violation_mask.sum().item())
    if violation_count:
        worst_flat = int(difference.reshape(-1).argmax().item())
        worst_batch = worst_flat // args.pomo_size
        worst_pomo = worst_flat % args.pomo_size
        raise AssertionError(
            f"Split exceeded an original feasible POMO partition in "
            f"{violation_count} trajectories. Worst ({worst_batch},{worst_pomo}): "
            f"split={float(split_costs[worst_batch,worst_pomo]):.12f}, "
            f"official={float(official_costs[worst_batch,worst_pomo]):.12f}, "
            f"difference={float(difference[worst_batch,worst_pomo]):.3g}"
        )

    savings = official_costs - split_costs
    report = {
        "status": "passed",
        "meaning": "For every checked POMO trajectory, optimal Split was no worse than the original feasible route partition.",
        "pomo_root": str(pomo_root),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "test_data": str(test_data),
        "test_data_sha256": sha256_file(test_data),
        "split_source": str(split_source),
        "split_source_sha256": sha256_file(split_source),
        "problem_size": args.problem_size,
        "pomo_size": args.pomo_size,
        "episodes": args.episodes,
        "trajectories_checked": args.episodes * args.pomo_size,
        "tolerance": args.tolerance,
        "max_split_minus_official": float(difference.max().item()),
        "minimum_saving": float(savings.min().item()),
        "mean_saving": float(savings.float().mean().item()),
        "maximum_saving": float(savings.max().item()),
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
