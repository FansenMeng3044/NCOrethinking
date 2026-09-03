import csv
import math
from argparse import Namespace

import pytest
import torch

from SplitTrainer import SplitTrainer


def _model_params(model_type, device):
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


@pytest.mark.parametrize("model_type", ["MTL_SPLIT", "MOE_SPLIT", "MOE_LIGHT_SPLIT"])
def test_all_three_trainers_write_structured_metrics(tmp_path, model_type):
    device = torch.device("cpu")
    output = tmp_path / model_type.lower()
    args = Namespace(
        device=device,
        log_path=str(output),
        problem="CVRP",
        model_type=model_type,
        checkpoint=None,
    )
    trainer = SplitTrainer(
        args=args,
        env_params={"problem_size": 20, "pomo_size": 5},
        model_params=_model_params(model_type, device),
        optimizer_params={
            "optimizer": {"lr": 1e-4, "weight_decay": 1e-6},
            "scheduler": {"milestones": [2], "gamma": 0.1},
        },
        trainer_params={
            "epochs": 1,
            "train_episodes": 2,
            "train_batch_size": 2,
            "model_save_interval": 1,
            "metrics_log_interval": 1,
            "metrics_flush_interval": 1,
            "max_grad_norm": math.inf,
        },
    )
    trainer.run()

    metrics_path = output / "training_metrics.csv"
    checkpoint_path = output / "epoch-1.pt"
    assert metrics_path.is_file()
    assert checkpoint_path.is_file()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["xy_only"] is True
    assert checkpoint["xy_encoder_only"] is True
    assert checkpoint["decoder_constraints"] == ["B", "L"]
    assert checkpoint["split_constraints"] == ["C", "TW"]
    assert checkpoint["constraint_factorization_version"] == 1
    with metrics_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["record_type"] for row in rows} == {
        "run_metadata", "event", "batch", "checkpoint", "epoch", "run_summary"
    }
    batch = next(row for row in rows if row["record_type"] == "batch")
    assert batch["problem"] == "CVRP"
    assert float(batch["learning_rate"]) == pytest.approx(1e-4)
    for field in (
        "score_mean", "loss_mean", "policy_loss_mean", "aux_loss_mean",
        "grad_norm", "valid_candidate_rate", "mean_routes", "step_seconds",
    ):
        assert math.isfinite(float(batch[field]))
    epoch = next(row for row in rows if row["record_type"] == "epoch")
    assert epoch["checkpoint_saved"] == "True"
    assert epoch["task_counts_json"] == '{"CVRP": 2}'
    assert float(epoch["learning_rate_start"]) == pytest.approx(1e-4)
