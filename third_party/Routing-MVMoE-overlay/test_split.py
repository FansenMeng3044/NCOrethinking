"""Evaluate one B/L-decoder + C/TW-Split checkpoint on official environments."""

import argparse
import csv
import os

import torch

from split import ALL_PROBLEMS, verify_routes
from split_envs import GiantTourEnv
from split_models import get_split_model
from utils import get_env, seed_everything


def evaluate_batch(model, source, pomo_size, augmentation):
    env = GiantTourEnv.from_official_env(source, pomo_size=pomo_size)
    actual_pomo_size = env.pomo_size
    reset, _, _ = env.reset()
    model.pre_forward(reset)
    state, reward, done = env.pre_step()
    while not done:
        selected, _ = model(state)
        state, reward, done = env.step(selected)

    aug_batch = env.batch_size
    if aug_batch % augmentation:
        raise ValueError("augmented batch is not divisible by augmentation")
    batch = aug_batch // augmentation
    costs = env.last_split_result.costs.reshape(augmentation, batch, actual_pomo_size)
    feasible = env.last_split_result.feasible.reshape(augmentation, batch, actual_pomo_size)
    masked = costs.masked_fill(~feasible, float("inf"))
    flat = masked.permute(1, 0, 2).reshape(batch, -1)
    best_cost, best_flat = flat.min(dim=1)

    records = []
    for index in range(batch):
        if not torch.isfinite(best_cost[index]):
            records.append(("no_feasible_candidate", None, None))
            continue
        augmentation_index = int(best_flat[index].item()) // actual_pomo_size
        pomo_index = int(best_flat[index].item()) % actual_pomo_size
        source_batch_index = augmentation_index * batch + index
        routes = env.get_routes(source_batch_index, pomo_index)
        replay = verify_routes(
            env.depot_xy,
            env.node_xy,
            routes,
            env.instance.split_view,
            batch_index=source_batch_index,
        )
        if not replay.feasible:
            raise AssertionError(f"independent replay failed: {replay.reason}")
        if abs(replay.cost - float(best_cost[index].item())) > 2e-5:
            raise AssertionError("independent replay cost disagrees with Split")
        records.append(("ok", replay.cost, replay.route_count))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_type", required=True, choices=[
        "MTL_SPLIT", "MOE_SPLIT", "MOE_LIGHT_SPLIT"
    ])
    parser.add_argument("--problem", default="ALL", choices=["ALL", *ALL_PROBLEMS])
    parser.add_argument("--problem_size", type=int, choices=[50, 100, 200], default=100)
    parser.add_argument("--pomo_size", type=int)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--augmentation", type=int, choices=[1, 8], default=8)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--output", default="split_evaluation.csv")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    seed_everything(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    xy_encoder_only = checkpoint.get("xy_encoder_only", checkpoint.get("xy_only"))
    if not xy_encoder_only or not checkpoint.get("split_reward"):
        raise ValueError("checkpoint is not marked as XY-encoded/post-Split training")
    if checkpoint.get("decoder_constraints") != ["B", "L"]:
        raise ValueError("checkpoint does not use the B/L-aware decoder")
    if checkpoint.get("split_constraints") != ["C", "TW"]:
        raise ValueError("checkpoint does not use the C/TW-only Split protocol")
    if checkpoint.get("model_type") != args.model_type:
        raise ValueError("checkpoint model type mismatch")
    model_params = dict(checkpoint["model_params"])
    model_params["device"] = device
    model = get_split_model(args.model_type)(**model_params).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    model.set_eval_type("argmax")

    problems = ALL_PROBLEMS if args.problem == "ALL" else (args.problem,)
    pomo_size = args.pomo_size or args.problem_size
    rows = []
    with torch.no_grad():
        for problem in problems:
            source_cls = get_env(problem)[0]
            source = source_cls(
                problem_size=args.problem_size, pomo_size=pomo_size, device=device
            )
            dataset_path = os.path.join(
                args.data_root, problem, f"{problem.lower()}{args.problem_size}_uniform.pkl"
            )
            offset = 0
            while offset < args.episodes:
                count = min(args.batch_size, args.episodes - offset)
                data = source.load_dataset(dataset_path, offset=offset, num_samples=count)
                if isinstance(data, torch.Tensor):
                    data = data.to(device)
                else:
                    data = tuple(value.to(device) for value in data)
                source.load_problems(count, problems=data, aug_factor=args.augmentation)
                records = evaluate_batch(model, source, pomo_size, args.augmentation)
                for local, (status, cost, vehicles) in enumerate(records):
                    rows.append({
                        "problem": problem,
                        "instance": offset + local,
                        "status": status,
                        "cost": "" if cost is None else f"{cost:.10f}",
                        "vehicles": "" if vehicles is None else vehicles,
                    })
                offset += count
            ok = [row for row in rows if row["problem"] == problem and row["status"] == "ok"]
            mean = sum(float(row["cost"]) for row in ok) / len(ok) if ok else float("nan")
            print(f"{problem}: {len(ok)}/{args.episodes} feasible, mean cost={mean:.6f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["problem", "instance", "status", "cost", "vehicles"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
