#!/usr/bin/env python
import argparse
import os
import sys


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(THIS_DIR)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, "../..")))

from POMO_TW.VRPTWTrainer import VRPTWTrainer
from utils.utils import copy_all_src, create_logger


def main():
    parser = argparse.ArgumentParser(description="Train POMO on multi-vehicle CVRPTW")
    parser.add_argument("--problem-size", type=int, default=100)
    parser.add_argument("--pomo-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--train-episodes", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--depot-end", type=float, default=3.0)
    parser.add_argument("--service-duration", type=float, default=0.2)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--metrics-log-interval", type=int, default=1)
    parser.add_argument("--metrics-flush-interval", type=int, default=50)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.problem_size, args.epochs, args.train_episodes, args.batch_size = 20, 1, 4, 2
        args.no_cuda = True
    pomo_size = args.pomo_size or args.problem_size
    env_params = {
        "problem_size": args.problem_size,
        "pomo_size": pomo_size,
        "capacity": 1.0,
        "speed": 1.0,
        "depot_start": 0.0,
        "depot_end": args.depot_end,
        "service_duration": args.service_duration,
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
            "metrics_log_interval": args.metrics_log_interval,
            "metrics_flush_interval": args.metrics_flush_interval,
        },
        "model_load": {"enable": False},
    }
    create_logger(
        log_file={
            "desc": "pomo_cvrptw_n{}".format(args.problem_size),
            "filename": "run_log.txt",
        }
    )
    trainer = VRPTWTrainer(env_params, model_params, optimizer_params, trainer_params)
    copy_all_src(trainer.result_folder)
    trainer.run()


if __name__ == "__main__":
    main()
