"""Single-file, paper-friendly metrics for Attention Model training runs."""

import csv
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

import torch


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _git_value(cwd, *args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=cwd, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _unique(*groups):
    return list(dict.fromkeys(item for group in groups for item in group))


class MetricsLogger:
    """Writes metadata, batches, epochs, checkpoints and summary to one CSV."""

    COMMON_FIELDS = [
        "record_type", "event", "timestamp_utc", "elapsed_seconds", "status", "error",
        "details_json",
    ]
    BATCH_FIELDS = [
        "epoch", "batch_id", "global_step", "samples_seen", "batch_size", "learning_rate",
        "cost_mean", "cost_std", "reinforce_loss", "reinforce_loss_std",
        "reinforce_loss_standard_error", "baseline_loss", "total_loss",
        "total_loss_std", "total_loss_standard_error",
        "log_likelihood_mean", "nll_mean", "baseline_value_mean", "advantage_mean",
        "advantage_std", "grad_norm", "grad_norm_clipped", "critic_grad_norm",
        "critic_grad_norm_clipped", "step_seconds", "throughput_instances_per_second",
        "gpu_memory_allocated_mb", "gpu_memory_reserved_mb", "gpu_peak_allocated_mb",
        "gpu_peak_reserved_mb",
    ]
    EPOCH_FIELDS = [
        "global_step_end", "epoch_examples", "cumulative_examples", "train_batches",
        "learning_rate_start", "learning_rate_end", "train_cost_mean", "train_cost_std",
        "reinforce_loss_mean", "reinforce_loss_std", "reinforce_loss_standard_error",
        "baseline_loss_mean", "total_loss_mean", "total_loss_std",
        "total_loss_standard_error", "nll_mean",
        "advantage_mean", "advantage_std", "grad_norm_mean", "grad_norm_max",
        "grad_norm_clipped_mean", "data_preparation_seconds", "training_seconds",
        "checkpoint_seconds", "validation_seconds", "baseline_callback_seconds",
        "epoch_total_seconds", "cumulative_elapsed_seconds",
        "throughput_instances_per_second", "validation_count", "validation_cost_mean",
        "validation_cost_std", "validation_standard_error", "best_validation_cost_mean",
        "best_validation_epoch", "gpu_peak_allocated_mb", "gpu_peak_reserved_mb",
        "checkpoint_saved", "checkpoint_path", "checkpoint_size_bytes",
        "baseline_candidate_mean", "baseline_reference_mean", "baseline_difference",
        "baseline_p_value", "baseline_updated", "baseline_epoch_before",
        "baseline_epoch_after", "warmup_alpha", "last_completed_epoch",
    ]
    FIELDS = _unique(COMMON_FIELDS, BATCH_FIELDS, EPOCH_FIELDS)

    def __init__(self, save_dir, opts, model, baseline):
        self.save_dir = save_dir
        self.opts = opts
        self.start_perf = time.perf_counter()
        self.start_utc = _utc_now()
        self.samples_seen = int(opts.epoch_start) * int(opts.epoch_size)
        self.log_interval = max(1, int(opts.metrics_log_interval))
        self.flush_interval = max(1, int(opts.metrics_flush_interval))
        self._pending_rows = 0
        self._closed = False
        self._epoch = None
        self.best_val = math.inf
        self.best_val_epoch = None
        self.last_epoch = int(opts.epoch_start)

        self.metrics_path = os.path.join(save_dir, "training_metrics.csv")
        self._metrics_file = open(self.metrics_path, "a", newline="", buffering=1)
        self._writer = csv.DictWriter(self._metrics_file, fieldnames=self.FIELDS)
        if os.path.getsize(self.metrics_path) == 0:
            self._writer.writeheader()

        metadata = self._collect_metadata(opts, model, baseline)
        self._write_row(
            "run_metadata", event="run_started",
            details_json=json.dumps(_jsonable(metadata), sort_keys=True),
        )
        self._metrics_file.flush()

    def _collect_metadata(self, opts, model, baseline):
        repository_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        inner_model = model.module if hasattr(model, "module") else model
        total_params = sum(p.numel() for p in inner_model.parameters())
        trainable_params = sum(p.numel() for p in inner_model.parameters() if p.requires_grad)
        gpu_info = []
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                gpu_info.append({
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": props.total_memory,
                    "compute_capability": "{}.{}".format(props.major, props.minor),
                })
        git_status = _git_value(repository_dir, "status", "--porcelain")
        return {
            "schema_version": 3,
            "started_at_utc": self.start_utc,
            "command": sys.argv,
            "working_directory": os.getcwd(),
            "repository_directory": repository_dir,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpus": gpu_info,
            "model_class": inner_model.__class__.__name__,
            "baseline_class": baseline.__class__.__name__,
            "model_parameters_total": total_params,
            "model_parameters_trainable": trainable_params,
            "git_commit": _git_value(repository_dir, "rev-parse", "HEAD"),
            "git_branch": _git_value(repository_dir, "rev-parse", "--abbrev-ref", "HEAD"),
            "git_is_dirty": bool(git_status),
            "git_status_porcelain": git_status.splitlines() if git_status else [],
            "arguments": _jsonable(vars(opts)),
            "metrics_file": os.path.basename(self.metrics_path),
        }

    def _write_row(self, record_type, **values):
        row = {field: "" for field in self.FIELDS}
        row["record_type"] = record_type
        row["timestamp_utc"] = values.pop("timestamp_utc", _utc_now())
        row["elapsed_seconds"] = values.pop(
            "elapsed_seconds", time.perf_counter() - self.start_perf
        )
        extras = {}
        for key, value in values.items():
            if key in row:
                row[key] = value
            else:
                extras[key] = value
        if extras:
            existing = row.get("details_json")
            details = {"extra": _jsonable(extras)}
            if existing:
                details["details"] = existing
            row["details_json"] = json.dumps(details, sort_keys=True)
        self._writer.writerow({key: _jsonable(value) for key, value in row.items()})
        self._pending_rows += 1
        if self._pending_rows >= self.flush_interval:
            self._metrics_file.flush()
            self._pending_rows = 0

    def log_event(self, event, **payload):
        self._write_row("event", event=event, **payload)
        self._metrics_file.flush()

    def start_epoch(self, epoch):
        self._epoch = {
            "epoch": int(epoch), "examples": 0, "batches": 0, "cost_sum": 0.0,
            "cost_sumsq": 0.0, "advantage_sum": 0.0, "advantage_sumsq": 0.0,
            "reinforce_loss_sum": 0.0, "reinforce_loss_sumsq": 0.0,
            "baseline_loss_sum": 0.0, "total_loss_sum": 0.0,
            "total_loss_sumsq": 0.0, "nll_sum": 0.0, "grad_norm_sum": 0.0,
            "grad_norm_clipped_sum": 0.0, "grad_norm_max": 0.0,
            "gpu_peak_allocated_mb": 0.0, "gpu_peak_reserved_mb": 0.0,
        }
        self.log_event("epoch_started", epoch=epoch)

    def log_batch(self, metrics):
        if self._epoch is None:
            raise RuntimeError("start_epoch must be called before log_batch")
        n = int(metrics["batch_size"])
        self.samples_seen += n
        acc = self._epoch
        acc["examples"] += n
        acc["batches"] += 1
        acc["cost_sum"] += float(metrics["_cost_sum"])
        acc["cost_sumsq"] += float(metrics["_cost_sumsq"])
        acc["advantage_sum"] += float(metrics["_advantage_sum"])
        acc["advantage_sumsq"] += float(metrics["_advantage_sumsq"])
        acc["reinforce_loss_sum"] += float(metrics["_reinforce_loss_sum"])
        acc["reinforce_loss_sumsq"] += float(metrics["_reinforce_loss_sumsq"])
        acc["baseline_loss_sum"] += float(metrics["baseline_loss"]) * n
        acc["total_loss_sum"] += float(metrics["_total_loss_sum"])
        acc["total_loss_sumsq"] += float(metrics["_total_loss_sumsq"])
        acc["nll_sum"] += float(metrics["nll_mean"]) * n
        acc["grad_norm_sum"] += float(metrics["grad_norm"])
        acc["grad_norm_clipped_sum"] += float(metrics["grad_norm_clipped"])
        acc["grad_norm_max"] = max(acc["grad_norm_max"], float(metrics["grad_norm"]))
        acc["gpu_peak_allocated_mb"] = max(
            acc["gpu_peak_allocated_mb"], float(metrics["gpu_peak_allocated_mb"])
        )
        acc["gpu_peak_reserved_mb"] = max(
            acc["gpu_peak_reserved_mb"], float(metrics["gpu_peak_reserved_mb"])
        )
        if int(metrics["global_step"]) % self.log_interval == 0:
            row = {key: value for key, value in metrics.items() if not key.startswith("_")}
            row["samples_seen"] = self.samples_seen
            self._write_row("batch", **row)

    def log_checkpoint(self, epoch, path, duration_seconds):
        self._write_row(
            "checkpoint", event="checkpoint_saved", epoch=epoch, checkpoint_path=path,
            checkpoint_size_bytes=os.path.getsize(path) if os.path.exists(path) else "",
            checkpoint_seconds=duration_seconds,
        )
        self._metrics_file.flush()

    def log_epoch(self, metrics, baseline_metrics=None):
        acc = self._epoch
        if acc is None:
            raise RuntimeError("start_epoch must be called before log_epoch")
        n = max(1, acc["examples"])
        batches = max(1, acc["batches"])
        cost_mean = acc["cost_sum"] / n
        cost_var = max(0.0, acc["cost_sumsq"] / n - cost_mean * cost_mean)
        adv_mean = acc["advantage_sum"] / n
        adv_var = max(0.0, acc["advantage_sumsq"] / n - adv_mean * adv_mean)
        reinforce_mean = acc["reinforce_loss_sum"] / n
        reinforce_var = max(
            0.0, acc["reinforce_loss_sumsq"] / n - reinforce_mean * reinforce_mean
        )
        reinforce_sample_var = (
            max(0.0, (
                acc["reinforce_loss_sumsq"] - acc["reinforce_loss_sum"] ** 2 / n
            ) / (n - 1)) if n > 1 else 0.0
        )
        total_mean = acc["total_loss_sum"] / n
        total_var = max(
            0.0, acc["total_loss_sumsq"] / n - total_mean * total_mean
        )
        total_sample_var = (
            max(0.0, (
                acc["total_loss_sumsq"] - acc["total_loss_sum"] ** 2 / n
            ) / (n - 1)) if n > 1 else 0.0
        )
        val_mean = float(metrics["validation_cost_mean"])
        epoch = int(metrics["epoch"])
        if val_mean < self.best_val:
            self.best_val = val_mean
            self.best_val_epoch = epoch
        row = dict(metrics)
        row.update({
            "event": "epoch_finished",
            "epoch_examples": acc["examples"],
            "cumulative_examples": self.samples_seen,
            "train_batches": acc["batches"],
            "train_cost_mean": cost_mean,
            "train_cost_std": math.sqrt(cost_var),
            "reinforce_loss_mean": reinforce_mean,
            "reinforce_loss_std": math.sqrt(reinforce_var),
            "reinforce_loss_standard_error": math.sqrt(reinforce_sample_var / n),
            "baseline_loss_mean": acc["baseline_loss_sum"] / n,
            "total_loss_mean": total_mean,
            "total_loss_std": math.sqrt(total_var),
            "total_loss_standard_error": math.sqrt(total_sample_var / n),
            "nll_mean": acc["nll_sum"] / n,
            "advantage_mean": adv_mean,
            "advantage_std": math.sqrt(adv_var),
            "grad_norm_mean": acc["grad_norm_sum"] / batches,
            "grad_norm_max": acc["grad_norm_max"],
            "grad_norm_clipped_mean": acc["grad_norm_clipped_sum"] / batches,
            "best_validation_cost_mean": self.best_val,
            "best_validation_epoch": self.best_val_epoch,
            "gpu_peak_allocated_mb": acc["gpu_peak_allocated_mb"],
            "gpu_peak_reserved_mb": acc["gpu_peak_reserved_mb"],
            "cumulative_elapsed_seconds": time.perf_counter() - self.start_perf,
        })
        if baseline_metrics:
            row.update(baseline_metrics)
        self._write_row("epoch", **row)
        self._metrics_file.flush()
        self.last_epoch = epoch
        self._epoch = None

    def close(self, status="completed", error=None):
        if self._closed:
            return
        elapsed = time.perf_counter() - self.start_perf
        summary = {
            "status": status,
            "error": error,
            "event": "run_finished",
            "elapsed_seconds": elapsed,
            "last_completed_epoch": self.last_epoch,
            "samples_seen": self.samples_seen,
            "best_validation_cost_mean": None if math.isinf(self.best_val) else self.best_val,
            "best_validation_epoch": self.best_val_epoch,
            "details_json": json.dumps({
                "started_at_utc": self.start_utc,
                "finished_at_utc": _utc_now(),
            }, sort_keys=True),
        }
        self._write_row("run_summary", **summary)
        self._metrics_file.flush()
        self._metrics_file.close()
        self._closed = True
