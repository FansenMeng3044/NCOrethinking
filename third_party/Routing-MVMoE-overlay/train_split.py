import argparse
import math
import os
from datetime import datetime, timedelta

import torch
import torch.distributed as dist

from SplitTrainer import SplitTrainer
from utils import seed_everything


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Train XY-encoded POMO-MTL-Split / MVMoE-Split with B-aware "
            "decoding and B/L/C/TW Split on official MVMoE tasks"
        )
    )
    parser.add_argument("--problem", default="Train_ALL", choices=[
        "Train_ALL", "CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW"
    ])
    parser.add_argument("--model_type", default="MTL_SPLIT", choices=[
        "MTL_SPLIT", "MOE_SPLIT", "MOE_LIGHT_SPLIT"
    ])
    parser.add_argument("--problem_size", type=int, default=100)
    parser.add_argument("--pomo_size", type=int, default=100)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--encoder_layer_num", type=int, default=6)
    parser.add_argument("--qkv_dim", type=int, default=16)
    parser.add_argument("--head_num", type=int, default=8)
    parser.add_argument("--logit_clipping", type=float, default=10.0)
    parser.add_argument("--ff_hidden_dim", type=int, default=512)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--eval_type", choices=["argmax", "softmax"], default="argmax")
    parser.add_argument("--norm", choices=["batch", "batch_no_track", "instance", "layer", "rezero", "none"], default="instance")
    parser.add_argument("--norm_loc", choices=["norm_first", "norm_last"], default="norm_last")
    parser.add_argument("--expert_loc", nargs="+", default=["Enc0", "Enc1", "Enc2", "Enc3", "Enc4", "Enc5", "Dec"])
    parser.add_argument("--routing_level", choices=["node", "instance"], default="node")
    parser.add_argument("--routing_method", choices=["input_choice", "expert_choice", "soft_moe", "random"], default="input_choice")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--milestones", type=int, nargs="+", default=[4501])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--train_episodes", type=int, default=20000)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--model_save_interval", type=int, default=2500)
    parser.add_argument(
        "--split_backend",
        choices=["reference", "triton"],
        default="reference",
        help=(
            "exact Split implementation to use; triton is a fused CUDA backend "
            "validated against the reference dynamic program"
        ),
    )
    parser.add_argument(
        "--metrics_log_interval",
        type=int,
        default=1,
        help="write one batch row to training_metrics.csv every N optimizer steps",
    )
    parser.add_argument(
        "--metrics_flush_interval",
        type=int,
        default=50,
        help="flush training_metrics.csv after this many rows",
    )
    parser.add_argument("--max_grad_norm", type=float, default=math.inf)
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--log_dir", default="./results_split")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="enable DistributedDataParallel; launch with torchrun",
    )
    parser.add_argument(
        "--expected_world_size",
        type=int,
        default=0,
        help="fail unless the torchrun world size matches this value (0 disables the check)",
    )
    parser.add_argument(
        "--dist_backend",
        choices=["nccl", "gloo"],
        default=None,
        help="distributed backend (default: nccl on CUDA, otherwise gloo)",
    )
    return parser


def _setup_runtime(args):
    launched_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if launched_world_size > 1 and not args.ddp:
        raise RuntimeError("torchrun detected but --ddp was not specified")
    if args.ddp and launched_world_size <= 1:
        raise RuntimeError("--ddp requires torchrun with more than one process")

    if args.ddp:
        use_cuda = torch.cuda.is_available() and not args.no_cuda
        backend = args.dist_backend or ("nccl" if use_cuda else "gloo")
        if backend == "nccl" and not use_cuda:
            raise RuntimeError("NCCL DDP requires CUDA")
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=30),
        )
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()
        args.local_rank = int(os.environ.get("LOCAL_RANK", str(args.rank)))
        if args.expected_world_size and args.world_size != args.expected_world_size:
            raise RuntimeError(
                f"expected world size {args.expected_world_size}, got {args.world_size}"
            )
        if use_cuda:
            if args.local_rank >= torch.cuda.device_count():
                raise RuntimeError("LOCAL_RANK exceeds the visible CUDA device count")
            torch.cuda.set_device(args.local_rank)
            args.device = torch.device("cuda", args.local_rank)
        else:
            args.device = torch.device("cpu")
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = args.gpu_id
        args.device = torch.device(
            f"cuda:{args.gpu_id}"
            if torch.cuda.is_available() and not args.no_cuda
            else "cpu"
        )
        if args.device.type == "cuda":
            torch.cuda.set_device(args.gpu_id)


def _shared_log_path(args):
    if args.rank == 0:
        run_name = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
            f"{args.model_type.lower()}_n{args.problem_size}"
        )
    else:
        run_name = None
    if args.ddp:
        values = [run_name]
        dist.broadcast_object_list(values, src=0)
        run_name = values[0]
    return os.path.join(args.log_dir, run_name)


def main():
    args = build_parser().parse_args()
    if args.pomo_size > args.problem_size:
        raise ValueError("pomo_size cannot exceed problem_size")
    try:
        _setup_runtime(args)
        if args.split_backend == "triton" and args.device.type != "cuda":
            raise RuntimeError("the Triton Split backend requires CUDA")
        os.environ["NCO_SPLIT_BACKEND"] = args.split_backend
        if args.ddp and (
            args.routing_method != "input_choice"
            or args.routing_level not in ("node", "instance")
        ):
            raise ValueError(
                "rigorous DDP training currently supports input-choice MoE "
                "with node- or instance-level routing"
            )
        if args.train_batch_size % args.world_size != 0:
            raise ValueError("global train_batch_size must be divisible by world size")
        if args.train_episodes % args.world_size != 0:
            raise ValueError("train_episodes must be divisible by world size")

        seed_everything(args.seed + args.rank)
        args.log_path = _shared_log_path(args)
        model_params = {
            "embedding_dim": args.embedding_dim,
            "sqrt_embedding_dim": args.embedding_dim ** 0.5,
            "encoder_layer_num": args.encoder_layer_num,
            "decoder_layer_num": 1,
            "qkv_dim": args.qkv_dim,
            "head_num": args.head_num,
            "logit_clipping": args.logit_clipping,
            "ff_hidden_dim": args.ff_hidden_dim,
            "num_experts": args.num_experts,
            "eval_type": args.eval_type,
            "norm": args.norm,
            "norm_loc": args.norm_loc,
            "expert_loc": args.expert_loc,
            "problem": args.problem,
            "topk": args.topk,
            "routing_level": args.routing_level,
            "routing_method": args.routing_method,
            "device": args.device,
        }
        trainer = SplitTrainer(
            args=args,
            env_params={"problem_size": args.problem_size, "pomo_size": args.pomo_size},
            model_params=model_params,
            optimizer_params={
                "optimizer": {"lr": args.lr, "weight_decay": args.weight_decay},
                "scheduler": {"milestones": args.milestones, "gamma": args.gamma},
            },
            trainer_params={
                "epochs": args.epochs,
                "train_episodes": args.train_episodes,
                "train_batch_size": args.train_batch_size,
                "model_save_interval": args.model_save_interval,
                "metrics_log_interval": args.metrics_log_interval,
                "metrics_flush_interval": args.metrics_flush_interval,
                "max_grad_norm": args.max_grad_norm,
                "split_backend": args.split_backend,
                "distributed": args.ddp,
                "world_size": args.world_size,
                "global_batch_size": args.train_batch_size,
                "local_batch_size": args.train_batch_size // args.world_size,
            },
        )
        trainer.run()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
