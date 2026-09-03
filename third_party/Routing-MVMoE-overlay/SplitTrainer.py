import os
import random
import time
import traceback
from dataclasses import dataclass

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from split_envs import GiantTourEnv
from split_models import get_split_model
from training_metrics import MVMoESplitTrainingMetrics
from utils import get_env, num_param


class NoFeasibleCandidateError(RuntimeError):
    pass


@dataclass
class BatchMetrics:
    score_mean: float
    score_std: float
    solution_cost_mean: float
    policy_loss_mean: float
    aux_loss_mean: float
    loss_mean: float
    loss_std: float
    loss_standard_error: float
    advantage_mean: float
    advantage_std: float
    log_prob_mean: float
    nll_mean: float
    valid_candidate_rate: float
    mean_routes: float
    grad_norm: float
    step_seconds: float
    throughput_instances_per_second: float
    gpu_memory_allocated_mb: float
    gpu_memory_reserved_mb: float
    gpu_peak_allocated_mb: float
    gpu_peak_reserved_mb: float
    _score_sum: float
    _score_sumsq: float
    _loss_sum: float
    _loss_sumsq: float


def valid_pomo_reinforce_loss(reward: torch.Tensor, log_prob: torch.Tensor):
    """POMO loss that never allows +/-inf from infeasible tours into an advantage."""
    if reward.shape != log_prob.shape:
        raise ValueError("reward and log_prob shapes must match")
    valid = torch.isfinite(reward)
    valid_count = valid.sum(dim=1, keepdim=True)
    if (valid_count == 0).any():
        bad = torch.nonzero(valid_count.squeeze(1) == 0).flatten().tolist()
        raise NoFeasibleCandidateError(
            f"no Split-feasible POMO candidate for batch rows {bad}; constraints were not relaxed"
        )
    safe_reward = torch.where(valid, reward, torch.zeros_like(reward))
    baseline = safe_reward.sum(dim=1, keepdim=True) / valid_count
    advantage = torch.where(valid, reward - baseline, torch.zeros_like(reward))
    per_instance = (-(advantage.detach() * log_prob) * valid).sum(dim=1) / valid_count.squeeze(1)
    return per_instance.mean(), valid


class SplitTrainer:
    """Six-task trainer for XY-encoded, B/L-aware giant-tour policies."""

    def __init__(self, args, env_params, model_params, optimizer_params, trainer_params):
        self.args = args
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        self.device = args.device
        self.log_path = args.log_path
        self.env_classes = get_env(args.problem)
        self.model = get_split_model(args.model_type)(**model_params).to(self.device)
        self.optimizer = Adam(self.model.parameters(), **optimizer_params["optimizer"])
        self.scheduler = MultiStepLR(self.optimizer, **optimizer_params["scheduler"])
        self.start_epoch = 1
        num_param(self.model)

        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=self.device, weights_only=False)
            if not checkpoint.get("xy_encoder_only", checkpoint.get("xy_only", False)):
                raise ValueError("refusing to load a checkpoint without XY-only static encoding")
            if checkpoint.get("decoder_constraints") != ["B", "L"]:
                raise ValueError("checkpoint does not use the B/L-aware decoder")
            if checkpoint.get("split_constraints") != ["C", "TW"]:
                raise ValueError("checkpoint does not use the C/TW-only Split protocol")
            if checkpoint.get("model_type") != args.model_type:
                raise ValueError("checkpoint model_type does not match the requested Split model")
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.start_epoch = int(checkpoint["epoch"]) + 1

        self.metrics_logger = MVMoESplitTrainingMetrics(
            self.log_path,
            args.model_type.lower(),
            self.env_params,
            self.model_params,
            self.optimizer_params,
            self.trainer_params,
            self.model,
            self.device,
            start_epoch=self.start_epoch,
        )

    def _official_batch(self, env_class, batch_size):
        params = dict(self.env_params)
        params["device"] = self.device
        source = env_class(**params)
        data = source.get_random_problems(batch_size, params["problem_size"], normalized=True)
        if isinstance(data, torch.Tensor):
            data = data.to(self.device)
        else:
            data = tuple(value.to(self.device) for value in data)
        source.load_problems(batch_size, problems=data, aug_factor=1)
        return source, GiantTourEnv.from_official_env(source, pomo_size=params["pomo_size"])

    def _train_one_batch(self, env_class, batch_size) -> BatchMetrics:
        batch_start = time.perf_counter()
        self.model.train()
        source, env = self._official_batch(env_class, batch_size)
        reset_state, _, _ = env.reset()
        self.model.pre_forward(reset_state)
        state, reward, done = env.pre_step()
        probabilities = []
        while not done:
            selected, prob = self.model(state)
            state, reward, done = env.step(selected)
            probabilities.append(prob)
        prob_tensor = torch.stack(probabilities, dim=2)
        if not torch.isfinite(prob_tensor).all() or (prob_tensor <= 0).any():
            raise FloatingPointError("policy probabilities are not finite and positive")
        log_prob = prob_tensor.log().sum(dim=2)
        policy_loss, valid = valid_pomo_reinforce_loss(reward, log_prob)
        aux = self.model.aux_loss
        if not isinstance(aux, torch.Tensor):
            aux = policy_loss.new_tensor(float(aux))
        aux = aux.mean()
        total_loss = policy_loss + aux
        if not torch.isfinite(total_loss):
            raise FloatingPointError("non-finite policy or MoE auxiliary loss")

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"non-finite gradient in {name}")
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.trainer_params.get("max_grad_norm", float("inf"))
        )
        self.optimizer.step()

        best_reward = reward.masked_fill(~valid, float("-inf")).max(dim=1).values
        best_index = reward.masked_fill(~valid, float("-inf")).argmax(dim=1)
        route_counts = env.last_split_result.route_counts[
            torch.arange(batch_size, device=self.device), best_index
        ].float()
        valid_count = valid.sum(dim=1, keepdim=True)
        safe_reward = torch.where(valid, reward, torch.zeros_like(reward))
        baseline = safe_reward.sum(dim=1, keepdim=True) / valid_count
        advantage = torch.where(valid, reward - baseline, torch.zeros_like(reward))
        per_instance_policy_loss = (
            (-(advantage.detach() * log_prob) * valid).sum(dim=1)
            / valid_count.squeeze(1)
        )
        aux_value = aux.detach()
        per_instance_total_loss = per_instance_policy_loss.detach() + aux_value
        score_values = -best_reward.detach().float()
        valid_costs = -reward.detach().float()[valid]
        valid_advantages = advantage.detach().float()[valid]
        valid_log_prob = log_prob.detach().float()[valid]
        loss_standard_error = (
            per_instance_total_loss.float().std(unbiased=True) / batch_size ** 0.5
            if batch_size > 1 else per_instance_total_loss.new_tensor(0.0)
        )
        packed = torch.stack([
            score_values.mean(),
            score_values.std(unbiased=False),
            valid_costs.mean(),
            policy_loss.detach(),
            aux_value,
            total_loss.detach(),
            per_instance_total_loss.float().std(unbiased=False),
            loss_standard_error,
            valid_advantages.mean(),
            valid_advantages.std(unbiased=False),
            valid_log_prob.mean(),
            -valid_log_prob.mean(),
            valid.float().mean(),
            route_counts.mean(),
            torch.as_tensor(grad_norm, device=self.device),
            score_values.sum(),
            score_values.square().sum(),
            per_instance_total_loss.float().sum(),
            per_instance_total_loss.float().square().sum(),
        ]).cpu().tolist()
        (
            score_mean,
            score_std,
            solution_cost_mean,
            policy_loss_mean,
            aux_loss_mean,
            loss_mean,
            loss_std,
            loss_standard_error_value,
            advantage_mean,
            advantage_std,
            log_prob_mean,
            nll_mean,
            valid_candidate_rate,
            mean_routes,
            grad_norm_value,
            score_sum,
            score_sumsq,
            loss_sum,
            loss_sumsq,
        ) = packed
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
            peak_allocated = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            peak_reserved = torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)
        else:
            allocated = reserved = peak_allocated = peak_reserved = 0.0
        step_seconds = time.perf_counter() - batch_start
        return BatchMetrics(
            score_mean=score_mean,
            score_std=score_std,
            solution_cost_mean=solution_cost_mean,
            policy_loss_mean=policy_loss_mean,
            aux_loss_mean=aux_loss_mean,
            loss_mean=loss_mean,
            loss_std=loss_std,
            loss_standard_error=loss_standard_error_value,
            advantage_mean=advantage_mean,
            advantage_std=advantage_std,
            log_prob_mean=log_prob_mean,
            nll_mean=nll_mean,
            valid_candidate_rate=valid_candidate_rate,
            mean_routes=mean_routes,
            grad_norm=grad_norm_value,
            step_seconds=step_seconds,
            throughput_instances_per_second=batch_size / max(step_seconds, 1e-12),
            gpu_memory_allocated_mb=allocated,
            gpu_memory_reserved_mb=reserved,
            gpu_peak_allocated_mb=peak_allocated,
            gpu_peak_reserved_mb=peak_reserved,
            _score_sum=score_sum,
            _score_sumsq=score_sumsq,
            _loss_sum=loss_sum,
            _loss_sumsq=loss_sumsq,
        )

    def run(self):
        epochs = self.trainer_params["epochs"]
        episodes = self.trainer_params["train_episodes"]
        batch_size = self.trainer_params["train_batch_size"]
        batches_per_epoch = (episodes + batch_size - 1) // batch_size
        status = "completed"
        error = None
        try:
            for epoch in range(self.start_epoch, epochs + 1):
                epoch_start = time.perf_counter()
                self.metrics_logger.start_epoch(epoch)
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)
                learning_rate_start = self.optimizer.param_groups[0]["lr"]
                training_start = time.perf_counter()
                totals = {
                    "score_mean": 0.0,
                    "loss_mean": 0.0,
                    "valid_candidate_rate": 0.0,
                    "mean_routes": 0.0,
                    "grad_norm": 0.0,
                }
                seen = 0
                batch_id = 0
                while seen < episodes:
                    current = min(batch_size, episodes - seen)
                    env_class = random.choice(self.env_classes)
                    metrics = self._train_one_batch(env_class, current)
                    problem = env_class.__name__.removesuffix("Env")
                    row = vars(metrics).copy()
                    row.update({
                        "epoch": epoch,
                        "batch_id": batch_id,
                        "global_step": (epoch - 1) * batches_per_epoch + batch_id,
                        "batch_size": current,
                        "problem": problem,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    })
                    self.metrics_logger.log_batch(row)
                    for key in totals:
                        totals[key] += getattr(metrics, key) * current
                    seen += current
                    batch_id += 1
                training_seconds = time.perf_counter() - training_start
                self.scheduler.step()
                means = {key: value / seen for key, value in totals.items()}

                checkpoint_path = ""
                checkpoint_seconds = 0.0
                interval = self.trainer_params["model_save_interval"]
                if epoch == epochs or epoch % interval == 0:
                    checkpoint_start = time.perf_counter()
                    checkpoint_path = self.save(epoch)
                    checkpoint_seconds = time.perf_counter() - checkpoint_start
                    self.metrics_logger.log_checkpoint(
                        epoch, checkpoint_path, checkpoint_seconds
                    )

                epoch_seconds = time.perf_counter() - epoch_start
                self.metrics_logger.log_epoch({
                    "epoch": epoch,
                    "global_step_end": epoch * batches_per_epoch,
                    "learning_rate_start": learning_rate_start,
                    "learning_rate_end": self.optimizer.param_groups[0]["lr"],
                    "training_seconds": training_seconds,
                    "checkpoint_seconds": checkpoint_seconds,
                    "epoch_total_seconds": epoch_seconds,
                    "throughput_instances_per_second": episodes / max(training_seconds, 1e-12),
                    "checkpoint_saved": bool(checkpoint_path),
                    "checkpoint_path": checkpoint_path,
                })
                print(
                    f"Epoch {epoch:4d}/{epochs}: score={means['score_mean']:.6f} "
                    f"loss={means['loss_mean']:.6f} "
                    f"valid={means['valid_candidate_rate']:.4f} "
                    f"routes={means['mean_routes']:.3f} "
                    f"grad={means['grad_norm']:.4f} seconds={epoch_seconds:.1f}"
                )
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            error = traceback.format_exc()
            raise
        finally:
            self.metrics_logger.close(status=status, error=error)

    def save(self, epoch):
        os.makedirs(self.log_path, exist_ok=True)
        path = os.path.join(self.log_path, f"epoch-{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_type": self.args.model_type,
                "xy_only": True,
                "xy_encoder_only": True,
                "decoder_constraints": ["B", "L"],
                "split_constraints": ["C", "TW"],
                "constraint_factorization_version": 1,
                "split_reward": True,
                "training_problems": [cls.__name__.removesuffix("Env") for cls in self.env_classes],
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "env_params": self.env_params,
                "model_params": self.model_params,
                "optimizer_params": self.optimizer_params,
                "trainer_params": self.trainer_params,
                "training_metrics_path": self.metrics_logger.path,
            },
            path,
        )
        return path
