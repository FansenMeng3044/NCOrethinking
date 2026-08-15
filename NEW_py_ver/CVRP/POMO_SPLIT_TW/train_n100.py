#!/usr/bin/env python
import argparse
import os
import sys


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(THIS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, "../..")))

from POMO_SPLIT_TW.GiantTourTWTrainer import GiantTourTWTrainer
from utils.utils import create_logger


def main():
    parser = argparse.ArgumentParser(description="Train XY-only POMO + hard-TW Split")
    parser.add_argument("--problem-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--train-episodes", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.problem_size, args.epochs, args.train_episodes, args.batch_size = 20, 1, 4, 2
        args.no_cuda = True
    env_params = {
        "problem_size": args.problem_size,
        "pomo_size": args.problem_size,
        "capacity": 1.0,
        "speed": 1.0,
        "depot_start": 0.0,
        "depot_end": 3.0,
        "service_duration": 0.2,
    }
    model_params = {
        "embedding_dim": 128,
        "sqrt_embedding_dim": 128 ** 0.5,
        "encoder_layer_num": 6,
        "qkv_dim": 16,
        "head_num": 8,
        "logit_clipping": 10,
        "ff_hidden_dim": 512,
        "eval_type": "argmax",
    }
    optimizer_params = {
        "optimizer": {"lr": 1e-4, "weight_decay": 1e-6},
        "scheduler": {"milestones": [180, 190], "gamma": 0.1},
    }
    trainer_params = {
        "use_cuda": not args.no_cuda,
        "cuda_device_num": args.cuda_device,
        "epochs": args.epochs,
        "train_episodes": args.train_episodes,
        "train_batch_size": args.batch_size,
        "logging": {
            "model_save_interval": 200,
            "metrics_log_interval": 1,
            "metrics_flush_interval": 50,
        },
        "model_load": {"enable": False},
    }
    create_logger(
        log_file={
            "desc": "pomo_split_cvrptw_n{}".format(args.problem_size),
            "filename": "run_log.txt",
        }
    )
    GiantTourTWTrainer(
        env_params, model_params, optimizer_params, trainer_params
    ).run()


if __name__ == "__main__":
    main()
