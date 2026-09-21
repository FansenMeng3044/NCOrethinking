#!/usr/bin/env python
"""Evaluate POMO TW Direct or Split with independent strict route replay."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def load_class(path, module_name, class_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--architecture", choices=("pomo_tw", "pomo_split_tw"), required=True)
    parser.add_argument("--instances", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--augmentation", type=int, choices=(1, 8), default=8)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-instance-output", required=True)
    parser.add_argument("--timing-warmup-batches", type=int, default=1)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    cvrp = repo / "NEW_py_ver" / "CVRP"
    for path in (cvrp, cvrp.parent):
        sys.path.insert(0, str(path))
    from utils.utils import create_logger

    here = Path(__file__).resolve().parent
    if args.architecture == "pomo_tw":
        Tester = load_class(here / "VRPTWTester_strict.py", "strict_pomo_tw", "VRPTWTester")
        model_params = {
            "embedding_dim": 128, "sqrt_embedding_dim": 128 ** 0.5,
            "encoder_layer_num": 6, "qkv_dim": 16, "head_num": 8,
            "logit_clipping": 10, "ff_hidden_dim": 512, "eval_type": "argmax",
        }
    else:
        Tester = load_class(
            here / "GiantTourTWTester_strict.py", "strict_pomo_split_tw", "GiantTourTWTester"
        )
        model_params = {
            "node_feature_dim": 6,
            "embedding_dim": 128, "sqrt_embedding_dim": 128 ** 0.5,
            "encoder_layer_num": 6, "qkv_dim": 16, "head_num": 8,
            "logit_clipping": 10, "ff_hidden_dim": 512, "eval_type": "argmax",
        }

    env_params = {
        "problem_size": 50, "pomo_size": 50, "capacity": 1.0,
        "speed": 1.0, "depot_start": 0.0, "depot_end": 3.0,
    }
    tester_params = {
        "use_cuda": True,
        "cuda_device_num": args.cuda_device,
        "model_load": {"path": str(Path(args.checkpoint_dir).resolve()), "epoch": 2000},
        "test_episodes": args.instances,
        "test_batch_size": args.batch_size,
        "augmentation_enable": args.augmentation == 8,
        "aug_factor": args.augmentation,
        "test_data_load": {"enable": True, "filename": str(Path(args.dataset).resolve())},
        "per_instance_output": str(Path(args.per_instance_output).resolve()),
        "timing_warmup_batches": args.timing_warmup_batches,
    }
    create_logger(log_file={"desc": f"fixed_{args.architecture}", "filename": "log.txt"})
    tester = Tester(env_params, model_params, tester_params)
    result = tester.run()
    text = json.dumps(result, indent=2, allow_nan=False)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
