import os
import random
import time
import traceback
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
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
    _count: float
    _valid_cost_sum: float
    _valid_cost_sumsq: float
    _valid_cost_count: float
    _policy_loss_sum: float
    _advantage_sum: float
    _advantage_sumsq: float
    _advantage_count: float
    _log_prob_sum: float
    _log_prob_sumsq: float
    _log_prob_count: float
    _valid_sum: float
    _candidate_count: float
    _route_sum: float


def _moments(total: float, total_squares: float, count: float):
    if count <= 0:
        raise ValueError("moment count must be positive")
    mean = total / count
    variance = max(0.0, total_squares / count - mean * mean)
    if count > 1:
        sample_variance = max(
            0.0, (total_squares - total * total / count) / (count - 1)
        )
        standard_error = (sample_variance / count) ** 0.5
    else:
        standard_error = 0.0
    return mean, variance ** 0.5, standard_error


class PolicyRollout(torch.nn.Module):
    """Keep the entire autoregressive graph inside one DDP forward call."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, reset_state, env):
        self.policy.pre_forward(reset_state)
        state, reward, done = env.pre_step()
        probabilities = []
        while not done:
            selected, prob = self.policy(state)
            state, reward, done = env.step(selected)
            probabilities.append(prob)
        return torch.stack(probabilities, dim=2), reward, self.policy.aux_loss


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
    """Six-task trainer for XY-encoded, B-aware giant-tour policies."""

    def __init__(self, args, env_params, model_params, optimizer_params, trainer_params):
        self.args = args
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        self.device = args.device
        self.log_path = args.log_path
        self.distributed = bool(getattr(args, "ddp", False))
        self.rank = int(getattr(args, "rank", 0))
        self.world_size = int(getattr(args, "world_size", 1))
        self.is_main = self.rank == 0
        self.env_classes = get_env(args.problem)
        self.model = get_split_model(args.model_type)(**model_params).to(self.device)
        self.optimizer = Adam(self.model.parameters(), **optimizer_params["optimizer"])
        self.scheduler = MultiStepLR(self.optimizer, **optimizer_params["scheduler"])
        self.start_epoch = 1
        pending_rng_state = None
        if self.is_main:
            num_param(self.model)

        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=self.device, weights_only=False)
            if not checkpoint.get("xy_encoder_only", checkpoint.get("xy_only", False)):
                raise ValueError("refusing to load a checkpoint without XY-only static encoding")
            if checkpoint.get("decoder_constraints") != ["B"]:
                raise ValueError("checkpoint does not use the B-only decoder")
            if checkpoint.get("split_constraints") != ["B", "L", "C", "TW"]:
                raise ValueError("checkpoint does not use the B/L/C/TW Split protocol")
            if checkpoint.get("constraint_factorization_version") != 2:
                raise ValueError("checkpoint uses an incompatible constraint factorization")
            if checkpoint.get("model_type") != args.model_type:
                raise ValueError("checkpoint model_type does not match the requested Split model")
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.start_epoch = int(checkpoint["epoch"]) + 1
            checkpoint_world_size = int(checkpoint.get("world_size", 1))
            if checkpoint.get("distributed", False) and checkpoint_world_size != self.world_size:
                raise ValueError(
                    "checkpoint world size differs from the requested DDP world size"
                )
            rng_states = checkpoint.get("rng_states")
            if rng_states is not None:
                if self.rank >= len(rng_states):
                    raise ValueError("checkpoint does not contain RNG state for this rank")
                pending_rng_state = rng_states[self.rank]

        rollout = PolicyRollout(self.model)
        if self.distributed:
            ddp_kwargs = {
                "broadcast_buffers": False,
                "find_unused_parameters": args.model_type != "MTL_SPLIT",
            }
            if self.device.type == "cuda":
                ddp_kwargs.update({
                    "device_ids": [self.device.index],
                    "output_device": self.device.index,
                })
            self.rollout = DistributedDataParallel(rollout, **ddp_kwargs)
        else:
            self.rollout = rollout

        if pending_rng_state is not None:
            self._restore_rng_state(pending_rng_state)

        self.metrics_logger = None
        if self.is_main:
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

    def _local_rng_state(self):
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state(self.device)
                if self.device.type == "cuda" else None
            ),
        }

    def _restore_rng_state(self, state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        if self.device.type == "cuda" and state.get("torch_cuda") is not None:
            torch.cuda.set_rng_state(state["torch_cuda"], self.device)

    def _synchronized_check(self, failed: bool, stage: str, detail: str):
        if not self.distributed:
            if failed:
                raise RuntimeError(f"{stage}: {detail}")
            return
        flag = torch.tensor(int(failed), device=self.device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if not flag.item():
            return
        details = [None for _ in range(self.world_size)]
        dist.all_gather_object(details, detail if failed else None)
        failures = "; ".join(
            f"rank {rank}: {message}"
            for rank, message in enumerate(details)
            if message is not None
        )
        raise RuntimeError(f"{stage} failed on at least one rank ({failures})")

    def _select_env(self):
        if not self.distributed:
            return random.choice(self.env_classes)
        if self.is_main:
            index = random.randrange(len(self.env_classes))
        else:
            index = 0
        value = torch.tensor(index, device=self.device, dtype=torch.long)
        dist.broadcast(value, src=0)
        return self.env_classes[int(value.item())]

    def _distributed_max(self, value: float) -> float:
        tensor = torch.tensor(float(value), device=self.device, dtype=torch.float64)
        if self.distributed:
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return float(tensor.item())

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
        self.rollout.train()
        source, env = self._official_batch(env_class, batch_size)
        reset_state, _, _ = env.reset()
        prob_tensor, reward, aux = self.rollout(reset_state, env)
        bad_probabilities = (
            not bool(torch.isfinite(prob_tensor).all().item())
            or bool((prob_tensor <= 0).any().item())
        )
        self._synchronized_check(
            bad_probabilities,
            "policy probability validation",
            "probabilities are non-finite or non-positive",
        )
        log_prob = prob_tensor.log().sum(dim=2)
        valid_before_loss = torch.isfinite(reward)
        empty_rows = torch.nonzero(
            valid_before_loss.sum(dim=1) == 0
        ).flatten().tolist()
        self._synchronized_check(
            bool(empty_rows),
            "Split candidate validation",
            f"no feasible POMO candidate for local rows {empty_rows}",
        )
        policy_loss, valid = valid_pomo_reinforce_loss(reward, log_prob)
        if not isinstance(aux, torch.Tensor):
            aux = policy_loss.new_tensor(float(aux))
        aux = aux.mean()
        total_loss = policy_loss + aux
        self._synchronized_check(
            bool(not torch.isfinite(total_loss)),
            "loss validation",
            "policy or MoE auxiliary loss is non-finite",
        )

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        bad_gradients = []
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                bad_gradients.append(name)
        self._synchronized_check(
            bool(bad_gradients),
            "gradient validation",
            f"non-finite gradients in {bad_gradients}",
        )
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
        raw = torch.stack([
            score_values.sum(),
            score_values.square().sum(),
            score_values.new_tensor(float(score_values.numel())),
            valid_costs.sum(),
            valid_costs.square().sum(),
            valid_costs.new_tensor(float(valid_costs.numel())),
            per_instance_policy_loss.detach().float().sum(),
            per_instance_total_loss.float().sum(),
            per_instance_total_loss.float().square().sum(),
            valid_advantages.sum(),
            valid_advantages.square().sum(),
            valid_advantages.new_tensor(float(valid_advantages.numel())),
            valid_log_prob.sum(),
            valid_log_prob.square().sum(),
            valid_log_prob.new_tensor(float(valid_log_prob.numel())),
            valid.float().sum(),
            valid.new_tensor(float(valid.numel()), dtype=torch.float32),
            route_counts.sum(),
            aux_value.float(),
            torch.as_tensor(grad_norm, device=self.device).float(),
            score_values.new_tensor(1.0),
        ]).to(dtype=torch.float64)
        if self.distributed:
            dist.all_reduce(raw, op=dist.ReduceOp.SUM)
        values = raw.cpu().tolist()
        score_sum, score_sumsq, count = values[0:3]
        valid_cost_sum, valid_cost_sumsq, valid_cost_count = values[3:6]
        policy_loss_sum = values[6]
        loss_sum, loss_sumsq = values[7:9]
        advantage_sum, advantage_sumsq, advantage_count = values[9:12]
        log_prob_sum, log_prob_sumsq, log_prob_count = values[12:15]
        valid_sum, candidate_count, route_sum = values[15:18]
        aux_sum, grad_norm_sum, participants = values[18:21]

        score_mean, score_std, _ = _moments(score_sum, score_sumsq, count)
        loss_mean, loss_std, loss_standard_error_value = _moments(
            loss_sum, loss_sumsq, count
        )
        solution_cost_mean, _, _ = _moments(
            valid_cost_sum, valid_cost_sumsq, valid_cost_count
        )
        advantage_mean, advantage_std, _ = _moments(
            advantage_sum, advantage_sumsq, advantage_count
        )
        log_prob_mean, _, _ = _moments(
            log_prob_sum, log_prob_sumsq, log_prob_count
        )
        policy_loss_mean = policy_loss_sum / count
        aux_loss_mean = aux_sum / participants
        valid_candidate_rate = valid_sum / candidate_count
        mean_routes = route_sum / count
        grad_norm_value = grad_norm_sum / participants

        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
            peak_allocated = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            peak_reserved = torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)
        else:
            allocated = reserved = peak_allocated = peak_reserved = 0.0
        memory = torch.tensor(
            [allocated, reserved, peak_allocated, peak_reserved],
            device=self.device,
            dtype=torch.float64,
        )
        if self.distributed:
            dist.all_reduce(memory, op=dist.ReduceOp.MAX)
        allocated, reserved, peak_allocated, peak_reserved = memory.cpu().tolist()
        step_seconds = self._distributed_max(time.perf_counter() - batch_start)
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
            nll_mean=-log_prob_mean,
            valid_candidate_rate=valid_candidate_rate,
            mean_routes=mean_routes,
            grad_norm=grad_norm_value,
            step_seconds=step_seconds,
            throughput_instances_per_second=count / max(step_seconds, 1e-12),
            gpu_memory_allocated_mb=allocated,
            gpu_memory_reserved_mb=reserved,
            gpu_peak_allocated_mb=peak_allocated,
            gpu_peak_reserved_mb=peak_reserved,
            _score_sum=score_sum,
            _score_sumsq=score_sumsq,
            _loss_sum=loss_sum,
            _loss_sumsq=loss_sumsq,
            _count=count,
            _valid_cost_sum=valid_cost_sum,
            _valid_cost_sumsq=valid_cost_sumsq,
            _valid_cost_count=valid_cost_count,
            _policy_loss_sum=policy_loss_sum,
            _advantage_sum=advantage_sum,
            _advantage_sumsq=advantage_sumsq,
            _advantage_count=advantage_count,
            _log_prob_sum=log_prob_sum,
            _log_prob_sumsq=log_prob_sumsq,
            _log_prob_count=log_prob_count,
            _valid_sum=valid_sum,
            _candidate_count=candidate_count,
            _route_sum=route_sum,
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
                if self.is_main:
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
                    if current % self.world_size != 0:
                        raise RuntimeError(
                            "final global batch is not divisible by the DDP world size"
                        )
                    local_current = current // self.world_size
                    env_class = self._select_env()
                    metrics = self._train_one_batch(env_class, local_current)
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
                    if self.is_main:
                        self.metrics_logger.log_batch(row)
                    for key in totals:
                        totals[key] += getattr(metrics, key) * current
                    seen += current
                    batch_id += 1
                training_seconds = self._distributed_max(
                    time.perf_counter() - training_start
                )
                self.scheduler.step()
                means = {key: value / seen for key, value in totals.items()}

                checkpoint_path = ""
                checkpoint_seconds = 0.0
                interval = self.trainer_params["model_save_interval"]
                if epoch == epochs or epoch % interval == 0:
                    checkpoint_start = time.perf_counter()
                    checkpoint_path = self.save(epoch)
                    checkpoint_seconds = self._distributed_max(
                        time.perf_counter() - checkpoint_start
                    )
                    if self.is_main:
                        self.metrics_logger.log_checkpoint(
                            epoch, checkpoint_path, checkpoint_seconds
                        )

                epoch_seconds = self._distributed_max(time.perf_counter() - epoch_start)
                if self.is_main:
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
            if self.is_main and self.metrics_logger is not None:
                self.metrics_logger.close(status=status, error=error)

    def save(self, epoch):
        path = os.path.join(self.log_path, f"epoch-{epoch}.pt")
        local_rng_state = self._local_rng_state()
        if self.distributed:
            rng_states = [None for _ in range(self.world_size)]
            dist.all_gather_object(rng_states, local_rng_state)
        else:
            rng_states = [local_rng_state]

        save_error = None
        if self.is_main:
            try:
                os.makedirs(self.log_path, exist_ok=True)
                payload = {
                    "epoch": epoch,
                    "model_type": self.args.model_type,
                    "xy_only": True,
                    "xy_encoder_only": True,
                    "decoder_constraints": ["B"],
                    "split_constraints": ["B", "L", "C", "TW"],
                    "constraint_factorization_version": 2,
                    "split_reward": True,
                    "training_problems": [
                        cls.__name__.removesuffix("Env")
                        for cls in self.env_classes
                    ],
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "env_params": self.env_params,
                    "model_params": self.model_params,
                    "optimizer_params": self.optimizer_params,
                    "trainer_params": self.trainer_params,
                    "training_metrics_path": os.path.join(
                        self.log_path, "training_metrics.csv"
                    ),
                    "distributed": self.distributed,
                    "world_size": self.world_size,
                    "global_batch_size": self.trainer_params["train_batch_size"],
                    "local_batch_size": (
                        self.trainer_params["train_batch_size"] // self.world_size
                    ),
                    "global_train_episodes": self.trainer_params["train_episodes"],
                    "rng_states": rng_states,
                }
                temporary_path = f"{path}.tmp"
                torch.save(payload, temporary_path)
                os.replace(temporary_path, path)
            except BaseException:
                save_error = traceback.format_exc()

        if self.distributed:
            values = [save_error]
            dist.broadcast_object_list(values, src=0)
            save_error = values[0]
        if save_error is not None:
            raise RuntimeError(f"checkpoint save failed on rank 0:\n{save_error}")
        return path
