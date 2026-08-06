"""Single-CSV structured training metrics shared by POMO variants."""

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
    if isinstance(value, (list, tuple, set)):
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


class POMOTrainingMetrics:
    """Write metadata, batch/epoch metrics, checkpoints and summary to one CSV."""

    COMMON_FIELDS = [
        "record_type", "event", "timestamp_utc", "elapsed_seconds", "status", "error",
        "details_json",
    ]
    BATCH_FIELDS = [
        "epoch", "batch_id", "global_step", "samples_seen", "batch_size", "learning_rate",
        "score_mean", "score_std", "solution_cost_mean", "loss_mean", "loss_std",
        "advantage_mean", "advantage_std", "log_prob_mean", "nll_mean", "grad_norm",
        "step_seconds", "throughput_instances_per_second", "gpu_memory_allocated_mb",
        "gpu_memory_reserved_mb", "gpu_peak_allocated_mb", "gpu_peak_reserved_mb",
    ]
    EPOCH_FIELDS = [
        "global_step_end", "epoch_examples", "cumulative_examples", "train_batches",
        "learning_rate_start", "learning_rate_end", "train_score_mean", "train_score_std",
        "train_loss_mean", "train_loss_std", "grad_norm_mean", "grad_norm_max",
        "training_seconds", "checkpoint_seconds", "epoch_total_seconds",
        "cumulative_elapsed_seconds", "throughput_instances_per_second",
        "gpu_peak_allocated_mb", "gpu_peak_reserved_mb", "checkpoint_saved",
        "checkpoint_path", "checkpoint_size_bytes", "last_completed_epoch",
    ]
    FIELDS = _unique(COMMON_FIELDS, BATCH_FIELDS, EPOCH_FIELDS)

    def __init__(self, result_folder, run_type, env_params, model_params,
                 optimizer_params, trainer_params, model, device):
        self.result_folder = result_folder
        self.run_type = run_type
        self.device = device
        self.start_perf = time.perf_counter()
        self.start_utc = _utc_now()
        self.log_interval = max(
            1, int(trainer_params.get("logging", {}).get("metrics_log_interval", 1))
        )
        self.flush_interval = max(
            1, int(trainer_params.get("logging", {}).get("metrics_flush_interval", 50))
        )
        model_load = trainer_params.get("model_load", {})
        completed_epochs = int(model_load.get("epoch", 0)) if model_load.get("enable") else 0
        self.samples_seen = completed_epochs * int(trainer_params["train_episodes"])
        self.best_train_score = math.inf
        self.best_train_epoch = None
        self.last_epoch = int(model_load.get("epoch", 0)) if model_load.get("enable") else 0
        self._epoch = None
        self._pending_rows = 0
        self._closed = False

        os.makedirs(result_folder, exist_ok=True)
        self.path = os.path.join(result_folder, "training_metrics.csv")
        self._file = open(self.path, "a", newline="", buffering=1)
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        if os.path.getsize(self.path) == 0:
            self._writer.writeheader()
        metadata = self._collect_metadata(
            env_params, model_params, optimizer_params, trainer_params, model
        )
        self._write_row(
            "run_metadata", event="run_started",
            details_json=json.dumps(_jsonable(metadata), sort_keys=True),
        )
        self._file.flush()

    def _collect_metadata(self, env_params, model_params, optimizer_params,
                          trainer_params, model):
        repository_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        git_status = _git_value(repository_dir, "status", "--porcelain")
        gpus = []
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                gpus.append({
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": props.total_memory,
                    "compute_capability": "{}.{}".format(props.major, props.minor),
                })
        return {
            "schema_version": 1,
            "run_type": self.run_type,
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
            "device": str(self.device),
            "gpus": gpus,
            "model_class": model.__class__.__name__,
            "model_parameters_total": sum(p.numel() for p in model.parameters()),
            "model_parameters_trainable": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "git_commit": _git_value(repository_dir, "rev-parse", "HEAD"),
            "git_branch": _git_value(repository_dir, "rev-parse", "--abbrev-ref", "HEAD"),
            "git_is_dirty": bool(git_status),
            "git_status_porcelain": git_status.splitlines() if git_status else [],
            "env_params": env_params,
            "model_params": model_params,
            "optimizer_params": optimizer_params,
            "trainer_params": trainer_params,
            "metrics_file": os.path.basename(self.path),
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
            self._file.flush()
            self._pending_rows = 0

    def log_event(self, event, **values):
        self._write_row("event", event=event, **values)
        self._file.flush()

    def start_epoch(self, epoch):
        self._epoch = {
            "epoch": epoch, "examples": 0, "batches": 0, "score_sum": 0.0,
            "score_sumsq": 0.0, "loss_sum": 0.0, "loss_sumsq": 0.0,
            "grad_norm_sum": 0.0, "grad_norm_max": 0.0,
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
        acc["score_sum"] += float(metrics["_score_sum"])
        acc["score_sumsq"] += float(metrics["_score_sumsq"])
        acc["loss_sum"] += float(metrics["_loss_sum"])
        acc["loss_sumsq"] += float(metrics["_loss_sumsq"])
        acc["grad_norm_sum"] += float(metrics["grad_norm"])
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
        self._file.flush()

    def log_epoch(self, values):
        if self._epoch is None:
            raise RuntimeError("start_epoch must be called before log_epoch")
        acc = self._epoch
        n = max(1, acc["examples"])
        batches = max(1, acc["batches"])
        score_mean = acc["score_sum"] / n
        score_var = max(0.0, acc["score_sumsq"] / n - score_mean * score_mean)
        loss_mean = acc["loss_sum"] / n
        loss_var = max(0.0, acc["loss_sumsq"] / n - loss_mean * loss_mean)
        epoch = int(values["epoch"])
        if score_mean < self.best_train_score:
            self.best_train_score = score_mean
            self.best_train_epoch = epoch
        row = dict(values)
        row.update({
            "event": "epoch_finished",
            "epoch_examples": acc["examples"],
            "cumulative_examples": self.samples_seen,
            "train_batches": acc["batches"],
            "train_score_mean": score_mean,
            "train_score_std": math.sqrt(score_var),
            "train_loss_mean": loss_mean,
            "train_loss_std": math.sqrt(loss_var),
            "grad_norm_mean": acc["grad_norm_sum"] / batches,
            "grad_norm_max": acc["grad_norm_max"],
            "gpu_peak_allocated_mb": acc["gpu_peak_allocated_mb"],
            "gpu_peak_reserved_mb": acc["gpu_peak_reserved_mb"],
            "cumulative_elapsed_seconds": time.perf_counter() - self.start_perf,
        })
        self._write_row("epoch", **row)
        self._file.flush()
        self.last_epoch = epoch
        self._epoch = None

    def close(self, status="completed", error=None):
        if self._closed:
            return
        self._write_row(
            "run_summary", event="run_finished", status=status, error=error,
            elapsed_seconds=time.perf_counter() - self.start_perf,
            samples_seen=self.samples_seen, last_completed_epoch=self.last_epoch,
            details_json=json.dumps({
                "run_type": self.run_type,
                "started_at_utc": self.start_utc,
                "finished_at_utc": _utc_now(),
                "best_train_score": None if math.isinf(self.best_train_score)
                else self.best_train_score,
                "best_train_epoch": self.best_train_epoch,
            }, sort_keys=True),
        )
        self._file.flush()
        self._file.close()
        self._closed = True
