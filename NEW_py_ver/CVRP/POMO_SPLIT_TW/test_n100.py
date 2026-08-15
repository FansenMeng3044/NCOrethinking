#!/usr/bin/env python
import argparse
import os
import sys


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(THIS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, "../..")))

from POMO_SPLIT_TW.GiantTourTWTester import GiantTourTWTester
from utils.utils import copy_all_src, create_logger


def main():
    parser = argparse.ArgumentParser(description="Evaluate POMO-Split-TW")
    parser.add_argument("dataset")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--problem-size", type=int, default=100)
    parser.add_argument("--instances", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--augmentation", type=int, choices=(1, 8), default=8)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()
    env_params = {
        "problem_size": args.problem_size, "pomo_size": args.problem_size,
        "capacity": 1.0, "speed": 1.0, "depot_start": 0.0, "depot_end": 3.0,
    }
    model_params = {
        "embedding_dim": 128, "sqrt_embedding_dim": 128 ** 0.5,
        "encoder_layer_num": 6, "qkv_dim": 16, "head_num": 8,
        "logit_clipping": 10, "ff_hidden_dim": 512, "eval_type": "argmax",
    }
    tester_params = {
        "use_cuda": not args.no_cuda, "cuda_device_num": args.cuda_device,
        "model_load": {"path": args.checkpoint_dir, "epoch": args.epoch},
        "test_episodes": args.instances, "test_batch_size": args.batch_size,
        "augmentation_enable": args.augmentation == 8, "aug_factor": args.augmentation,
        "test_data_load": {"enable": True, "filename": args.dataset},
    }
    create_logger(log_file={"desc": "test_pomo_split_cvrptw", "filename": "log.txt"})
    tester = GiantTourTWTester(env_params, model_params, tester_params)
    copy_all_src(tester.result_folder)
    tester.run()


if __name__ == "__main__":
    main()
