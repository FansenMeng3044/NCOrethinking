import argparse
import logging
import os
import shutil
import sys


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(THIS_DIR)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, "../..")))

from GiantTourTrainer import GiantTourTrainer
from utils.utils import create_logger


ENV_PARAMS = {
    "problem_size": 100,
    "pomo_size": 100,
    "capacity": 1.0,
}

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

OPTIMIZER_PARAMS = {
    "optimizer": {"lr": 1e-4, "weight_decay": 1e-6},
    "scheduler": {"milestones": [8001, 8051], "gamma": 0.1},
}

TRAINER_PARAMS = {
    "use_cuda": True,
    "cuda_device_num": 0,
    "epochs": 8100,
    "train_episodes": 10 * 1000,
    "train_batch_size": 64,
    "logging": {"model_save_interval": 200},
    "model_load": {"enable": False},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke", action="store_true", help="run 1 epoch with a tiny CPU workload"
    )
    args = parser.parse_args()

    trainer_params = dict(TRAINER_PARAMS)
    trainer_params["logging"] = dict(TRAINER_PARAMS["logging"])
    if args.smoke:
        trainer_params.update(
            {"use_cuda": False, "epochs": 1, "train_episodes": 4, "train_batch_size": 2}
        )
        # Keep the smoke run small enough for CPU verification.
        env_params = {"problem_size": 20, "pomo_size": 20, "capacity": 1.0}
    else:
        env_params = dict(ENV_PARAMS)

    create_logger(
        log_file={
            "desc": "pomo_giant_tour_noplit_train_n{}".format(env_params["problem_size"]),
            "filename": "run_log.txt",
        }
    )
    logger = logging.getLogger("root")
    logger.info("env_params=%s", env_params)
    logger.info("model_params=%s", MODEL_PARAMS)
    logger.info("optimizer_params=%s", OPTIMIZER_PARAMS)
    logger.info("trainer_params=%s", trainer_params)

    trainer = GiantTourTrainer(
        env_params, MODEL_PARAMS, OPTIMIZER_PARAMS, trainer_params
    )
    _snapshot_experiment_sources(trainer.result_folder)
    trainer.run()


def _snapshot_experiment_sources(result_folder):
    """Copy only this experiment's stable sources into the run directory."""
    destination = os.path.join(result_folder, "src")
    os.makedirs(destination, exist_ok=True)
    for filename in os.listdir(THIS_DIR):
        if filename.endswith(".py") or filename == "README.md":
            shutil.copy2(os.path.join(THIS_DIR, filename), os.path.join(destination, filename))


if __name__ == "__main__":
    main()
