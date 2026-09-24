"""Evaluate an official Direct checkpoint and save per-instance costs."""

import argparse
import csv
import os

import torch

from utils import get_env, get_model, seed_everything


ALL_PROBLEMS = (
    "CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW",
    "OVRPB", "OVRPL", "VRPBL", "VRPBTW", "VRPLTW",
    "OVRPBL", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW",
)


def build_model(checkpoint_path, device, model_type):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    params = {
        "embedding_dim": 128,
        "sqrt_embedding_dim": 128 ** 0.5,
        "encoder_layer_num": 6,
        "decoder_layer_num": 1,
        "qkv_dim": 16,
        "head_num": 8,
        "logit_clipping": 10,
        "ff_hidden_dim": 512,
        "num_experts": 4,
        "eval_type": "argmax",
        "norm": "instance",
        "norm_loc": "norm_last",
        "expert_loc": ["Enc0", "Enc1", "Enc2", "Enc3", "Enc4", "Enc5", "Dec"],
        "problem": checkpoint["problem"],
        "topk": 2,
        "routing_level": "node",
        "routing_method": "input_choice",
    }
    model = get_model(model_type)(**params).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    model.set_eval_type("argmax")
    return model, int(checkpoint["epoch"])


def evaluate_batch(model, env, data, augmentation):
    batch = data.size(0) if isinstance(data, torch.Tensor) else data[-1].size(0)
    env.load_problems(batch, problems=data, aug_factor=augmentation)
    reset, _, _ = env.reset()
    model.pre_forward(reset)
    state, reward, done = env.pre_step()
    while not done:
        selected, _ = model(state)
        state, reward, done = env.step(selected)
    rewards = reward.reshape(augmentation, batch, env.pomo_size)
    best_per_aug = rewards.max(dim=2).values
    no_aug = -best_per_aug[0].float()
    augmented = -best_per_aug.max(dim=0).values.float()
    return no_aug.cpu(), augmented.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_type", required=True, choices=("MTL", "MOE", "MOE_LIGHT"))
    parser.add_argument("--problem", default="ALL", choices=("ALL", *ALL_PROBLEMS))
    parser.add_argument("--problem_size", type=int, choices=(50, 100, 200), required=True)
    parser.add_argument("--pomo_size", type=int)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--augmentation", type=int, choices=(1, 8), default=8)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, epoch = build_model(args.checkpoint, device, args.model_type)
    problems = ALL_PROBLEMS if args.problem == "ALL" else (args.problem,)
    pomo_size = args.pomo_size or args.problem_size
    rows = []

    with torch.no_grad():
        for problem in problems:
            env_class = get_env(problem)[0]
            env = env_class(problem_size=args.problem_size, pomo_size=pomo_size, device=device)
            path = os.path.join(
                args.data_root,
                problem,
                f"{problem.lower()}{args.problem_size}_uniform.pkl",
            )
            offset = 0
            while offset < args.episodes:
                count = min(args.batch_size, args.episodes - offset)
                data = env.load_dataset(path, offset=offset, num_samples=count)
                if isinstance(data, torch.Tensor):
                    data = data.to(device)
                else:
                    data = tuple(value.to(device) for value in data)
                no_aug, augmented = evaluate_batch(model, env, data, args.augmentation)
                for local in range(count):
                    rows.append(
                        {
                            "problem_size": args.problem_size,
                            "problem": problem,
                            "instance": offset + local,
                            "status": "ok",
                            "no_aug_cost": f"{float(no_aug[local]):.10f}",
                            "cost": f"{float(augmented[local]):.10f}",
                        }
                    )
                offset += count
            current = [row for row in rows if row["problem"] == problem]
            mean = sum(float(row["cost"]) for row in current) / len(current)
            print(f"{problem}: {len(current)}/{args.episodes} ok, mean cost={mean:.6f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("problem_size", "problem", "instance", "status", "no_aug_cost", "cost"),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"checkpoint epoch={epoch}; wrote {len(rows)} rows to {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
