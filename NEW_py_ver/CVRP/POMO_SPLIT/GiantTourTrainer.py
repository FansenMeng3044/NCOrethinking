import math
import os
import time
import traceback
from logging import getLogger

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

try:
    from .GiantTourEnv import GiantTourEnv
    from .GiantTourModel import GiantTourModel
except ImportError:  # Keep the original standalone-script entry points working.
    from GiantTourEnv import GiantTourEnv
    from GiantTourModel import GiantTourModel
from utils.utils import AverageMeter, LogData, TimeEstimator, get_result_folder
from utils.training_metrics import POMOTrainingMetrics


class GiantTourTrainer:
    ENV_CLASS = GiantTourEnv
    MODEL_CLASS = GiantTourModel
    METRICS_NAME = "pomo_split_cvrp"

    def __init__(self, env_params, model_params, optimizer_params, trainer_params):
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        self.logger = getLogger("trainer")
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        use_cuda = trainer_params["use_cuda"] and torch.cuda.is_available()
        if trainer_params["use_cuda"] and not use_cuda:
            self.logger.warning("CUDA was requested but is unavailable; falling back to CPU")
        self.device = torch.device(
            "cuda", trainer_params["cuda_device_num"]
        ) if use_cuda else torch.device("cpu")
        if use_cuda:
            torch.cuda.set_device(self.device)

        self.model = self.MODEL_CLASS(**model_params).to(self.device)
        env_params_with_device = dict(env_params)
        env_params_with_device["device"] = self.device
        self.env = self.ENV_CLASS(**env_params_with_device)
        self.optimizer = Adam(self.model.parameters(), **optimizer_params["optimizer"])
        self.scheduler = MultiStepLR(self.optimizer, **optimizer_params["scheduler"])

        self.start_epoch = 1
        model_load = trainer_params.get("model_load", {"enable": False})
        if model_load.get("enable", False):
            checkpoint_name = os.path.join(
                model_load["path"], "checkpoint-{}.pt".format(model_load["epoch"])
            )
            checkpoint = torch.load(checkpoint_name, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if "result_log" in checkpoint:
                self.result_log.set_raw_data(checkpoint["result_log"])
            self.start_epoch = checkpoint["epoch"] + 1

        self.time_estimator = TimeEstimator()
        self.metrics_logger = POMOTrainingMetrics(
            self.result_folder, self.METRICS_NAME, self.env_params, self.model_params,
            self.optimizer_params, self.trainer_params, self.model, self.device
        )

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        epochs = self.trainer_params["epochs"]
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
                score, loss, global_step_end = self._train_one_epoch(epoch)
                training_seconds = time.perf_counter() - training_start
                self.result_log.append("train_score", epoch, score)
                self.result_log.append("train_loss", epoch, loss)
                self.scheduler.step()

                elapsed, remaining = self.time_estimator.get_est_string(epoch, epochs)
                self.logger.info(
                    "Epoch %d/%d: split_score=%.6f loss=%.6f elapsed=%s remaining=%s",
                    epoch, epochs, score, loss, elapsed, remaining,
                )

                checkpoint_path = ""
                checkpoint_seconds = 0.0
                save_interval = self.trainer_params["logging"]["model_save_interval"]
                if epoch == epochs or epoch % save_interval == 0:
                    checkpoint_start = time.perf_counter()
                    checkpoint_path = self._save_checkpoint(epoch)
                    checkpoint_seconds = time.perf_counter() - checkpoint_start
                    self.metrics_logger.log_checkpoint(epoch, checkpoint_path, checkpoint_seconds)

                self.metrics_logger.log_epoch({
                    "epoch": epoch,
                    "global_step_end": global_step_end,
                    "learning_rate_start": learning_rate_start,
                    "learning_rate_end": self.optimizer.param_groups[0]["lr"],
                    "training_seconds": training_seconds,
                    "checkpoint_seconds": checkpoint_seconds,
                    "epoch_total_seconds": time.perf_counter() - epoch_start,
                    "throughput_instances_per_second": self.trainer_params["train_episodes"] / max(training_seconds, 1e-12),
                    "checkpoint_saved": bool(checkpoint_path),
                    "checkpoint_path": checkpoint_path,
                })
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            error = traceback.format_exc()
            raise
        finally:
            self.metrics_logger.close(status=status, error=error)

    def _train_one_epoch(self, epoch):
        score_meter = AverageMeter()
        loss_meter = AverageMeter()
        episode = 0
        total = self.trainer_params["train_episodes"]
        batch_id = 0
        batches_per_epoch = (total + self.trainer_params["train_batch_size"] - 1) // self.trainer_params["train_batch_size"]
        while episode < total:
            batch_size = min(self.trainer_params["train_batch_size"], total - episode)
            metrics = self._train_one_batch(batch_size)
            metrics.update({
                "epoch": epoch,
                "batch_id": batch_id,
                "global_step": (epoch - 1) * batches_per_epoch + batch_id,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            })
            self.metrics_logger.log_batch(metrics)
            score, loss = metrics["score_mean"], metrics["loss_mean"]
            score_meter.update(score, batch_size)
            loss_meter.update(loss, batch_size)
            episode += batch_size
            batch_id += 1
        return score_meter.avg, loss_meter.avg, epoch * batches_per_epoch

    def _train_one_batch(self, batch_size):
        batch_start = time.perf_counter()
        self.model.train()
        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        prob_list = torch.empty(
            batch_size, self.env.pomo_size, 0, device=self.device
        )
        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = self.model(state)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        advantage = reward - reward.mean(dim=1, keepdim=True)
        log_prob = prob_list.log().sum(dim=2)
        loss = (-advantage * log_prob).mean()

        best_reward = reward.max(dim=1).values
        score = -best_reward.mean()

        self.optimizer.zero_grad()
        loss.backward()
        grad_parts = [p.grad.detach().float().norm(2) for p in self.model.parameters() if p.grad is not None]
        grad_norm = torch.stack(grad_parts).norm(2) if grad_parts else loss.new_tensor(0.0)
        self.optimizer.step()

        score_values = -best_reward.detach().float()
        loss_values = (-advantage * log_prob).detach().float().mean(dim=1)
        loss_standard_error = (
            loss_values.std(unbiased=True) / math.sqrt(batch_size)
            if batch_size > 1 else loss_values.new_tensor(0.0)
        )
        packed = torch.stack([
            score_values.mean(), score_values.std(unbiased=False), (-reward.detach().float()).mean(),
            loss_values.mean(), loss_values.std(unbiased=False), loss_standard_error,
            advantage.detach().float().mean(),
            advantage.detach().float().std(unbiased=False), log_prob.detach().float().mean(),
            grad_norm, score_values.sum(), score_values.square().sum(), loss_values.sum(),
            loss_values.square().sum(),
        ]).cpu().tolist()
        (score_value, score_std, solution_cost_mean, loss_value, loss_std,
         loss_standard_error_value, advantage_mean,
         advantage_std, log_prob_mean, grad_norm_value, score_sum, score_sumsq,
         loss_sum, loss_sumsq) = packed
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
            peak_allocated = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            peak_reserved = torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)
        else:
            allocated = reserved = peak_allocated = peak_reserved = 0.0
        step_seconds = time.perf_counter() - batch_start
        return {
            "batch_size": batch_size, "score_mean": score_value, "score_std": score_std,
            "solution_cost_mean": solution_cost_mean, "loss_mean": loss_value,
            "loss_std": loss_std, "loss_standard_error": loss_standard_error_value,
            "advantage_mean": advantage_mean,
            "advantage_std": advantage_std, "log_prob_mean": log_prob_mean,
            "nll_mean": -log_prob_mean, "grad_norm": grad_norm_value,
            "step_seconds": step_seconds,
            "throughput_instances_per_second": batch_size / max(step_seconds, 1e-12),
            "gpu_memory_allocated_mb": allocated, "gpu_memory_reserved_mb": reserved,
            "gpu_peak_allocated_mb": peak_allocated, "gpu_peak_reserved_mb": peak_reserved,
            "_score_sum": score_sum, "_score_sumsq": score_sumsq,
            "_loss_sum": loss_sum, "_loss_sumsq": loss_sumsq,
        }

    def _save_checkpoint(self, epoch):
        os.makedirs(self.result_folder, exist_ok=True)
        checkpoint_path = os.path.join(
            self.result_folder, "checkpoint-{}.pt".format(epoch)
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "result_log": self.result_log.get_raw_data(),
                "env_params": self.env_params,
                "model_params": self.model_params,
            },
            checkpoint_path,
        )
        return checkpoint_path
