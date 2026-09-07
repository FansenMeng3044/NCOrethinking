import csv
import math
from argparse import Namespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from SplitTrainer import SplitTrainer
from utils import seed_everything


def _model_params(device):
    return {
        "embedding_dim": 16,
        "sqrt_embedding_dim": 4.0,
        "encoder_layer_num": 1,
        "decoder_layer_num": 1,
        "qkv_dim": 4,
        "head_num": 2,
        "logit_clipping": 10.0,
        "ff_hidden_dim": 32,
        "num_experts": 4,
        "eval_type": "argmax",
        "norm": "layer",
        "norm_loc": "norm_last",
        "expert_loc": ["Enc0", "Dec"],
        "problem": "CVRP",
        "topk": 2,
        "routing_level": "node",
        "routing_method": "input_choice",
        "device": device,
    }


def _ddp_worker(
    rank, world_size, rendezvous_uri, output, model_type, checkpoint, epochs
):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous_uri,
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device("cpu")
        seed_everything(2023 + rank)
        args = Namespace(
            device=device,
            log_path=output,
            problem="CVRP",
            model_type=model_type,
            checkpoint=checkpoint,
            ddp=True,
            rank=rank,
            world_size=world_size,
            seed=2023,
        )
        trainer = SplitTrainer(
            args=args,
            env_params={"problem_size": 20, "pomo_size": 5},
            model_params=_model_params(device),
            optimizer_params={
                "optimizer": {"lr": 1e-4, "weight_decay": 1e-6},
                "scheduler": {"milestones": [2], "gamma": 0.1},
            },
            trainer_params={
                "epochs": epochs,
                "train_episodes": 4,
                "train_batch_size": 4,
                "model_save_interval": 1,
                "metrics_log_interval": 1,
                "metrics_flush_interval": 1,
                "max_grad_norm": math.inf,
                "distributed": True,
                "world_size": world_size,
                "global_batch_size": 4,
                "local_batch_size": 2,
            },
        )
        trainer.run()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available(), reason="distributed unavailable")
@pytest.mark.parametrize("model_type", ["MTL_SPLIT", "MOE_SPLIT", "MOE_LIGHT_SPLIT"])
def test_two_rank_ddp_preserves_global_batch_and_writes_one_checkpoint(
    tmp_path, model_type
):
    output = tmp_path / "ddp"
    rendezvous = tmp_path / "ddp_rendezvous"
    mp.spawn(
        _ddp_worker,
        args=(2, rendezvous.as_uri(), str(output), model_type, None, 1),
        nprocs=2,
        join=True,
    )

    checkpoints = list(output.glob("epoch-1.pt"))
    metrics_files = list(output.glob("training_metrics.csv"))
    assert len(checkpoints) == 1
    assert len(metrics_files) == 1

    checkpoint = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    assert checkpoint["distributed"] is True
    assert checkpoint["world_size"] == 2
    assert checkpoint["global_batch_size"] == 4
    assert checkpoint["local_batch_size"] == 2
    assert checkpoint["global_train_episodes"] == 4
    assert len(checkpoint["rng_states"]) == 2
    assert checkpoint["checkpoint_schema_version"] == 2
    assert checkpoint["resume_contract"]["world_size"] == 2
    assert checkpoint["resume_contract"]["global_train_batch_size"] == 4
    assert (output / "epoch-1.pt.resume.json").is_file()
    assert not any(key.startswith("module.") for key in checkpoint["model_state_dict"])

    with metrics_files[0].open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    batch = next(row for row in rows if row["record_type"] == "batch")
    epoch = next(row for row in rows if row["record_type"] == "epoch")
    assert int(batch["batch_size"]) == 4
    assert int(epoch["epoch_examples"]) == 4
    assert epoch["task_counts_json"] == '{"CVRP": 4}'


@pytest.mark.skipif(not torch.distributed.is_available(), reason="distributed unavailable")
def test_two_rank_checkpoint_resumes_with_per_rank_rng_state(tmp_path):
    first_output = tmp_path / "first"
    first_rendezvous = tmp_path / "first_rendezvous"
    mp.spawn(
        _ddp_worker,
        args=(
            2,
            first_rendezvous.as_uri(),
            str(first_output),
            "MOE_LIGHT_SPLIT",
            None,
            1,
        ),
        nprocs=2,
        join=True,
    )
    first_checkpoint = first_output / "epoch-1.pt"
    assert first_checkpoint.is_file()

    uninterrupted_output = tmp_path / "uninterrupted"
    uninterrupted_rendezvous = tmp_path / "uninterrupted_rendezvous"
    mp.spawn(
        _ddp_worker,
        args=(
            2,
            uninterrupted_rendezvous.as_uri(),
            str(uninterrupted_output),
            "MOE_LIGHT_SPLIT",
            None,
            2,
        ),
        nprocs=2,
        join=True,
    )

    resumed_output = tmp_path / "resumed"
    resumed_rendezvous = tmp_path / "resumed_rendezvous"
    mp.spawn(
        _ddp_worker,
        args=(
            2,
            resumed_rendezvous.as_uri(),
            str(resumed_output),
            "MOE_LIGHT_SPLIT",
            str(first_checkpoint),
            2,
        ),
        nprocs=2,
        join=True,
    )
    resumed_checkpoint = resumed_output / "epoch-2.pt"
    assert resumed_checkpoint.is_file()
    payload = torch.load(resumed_checkpoint, map_location="cpu", weights_only=False)
    assert payload["epoch"] == 2
    assert payload["world_size"] == 2
    assert len(payload["rng_states"]) == 2
    uninterrupted = torch.load(
        uninterrupted_output / "epoch-2.pt",
        map_location="cpu",
        weights_only=False,
    )
    for name, value in payload["model_state_dict"].items():
        assert torch.equal(value, uninterrupted["model_state_dict"][name]), name
    assert payload["optimizer_state_dict"]["param_groups"] == (
        uninterrupted["optimizer_state_dict"]["param_groups"]
    )
    assert payload["scheduler_state_dict"] == uninterrupted["scheduler_state_dict"]
