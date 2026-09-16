#!/usr/bin/env python3
"""Validate and summarize the final six MVMoE-Split training/evaluation runs.

The script uses only the Python standard library.  It streams the large
training CSV files, validates the six 16-environment result files, and writes
compact machine-readable and Markdown summaries next to the raw artifacts.
Compressed training records require the ``zstd`` command-line program.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import subprocess
from collections import Counter
from contextlib import contextmanager
from pathlib import Path


ENVIRONMENTS = [
    "CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW",
    "OVRPB", "OVRPL", "VRPBL", "VRPBTW", "VRPLTW", "OVRPBL",
    "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW",
]
IID = set(ENVIRONMENTS[:6])

MODELS = {
    "pomo_mtl_split_n50": ("POMO-MTL-Split", 50, "POMO-MTL"),
    "pomo_mtl_split_n100": ("POMO-MTL-Split", 100, "POMO-MTL"),
    "mvmoe_4e_split_n50": ("MVMoE/4E-Split", 50, "MVMoE/4E"),
    "mvmoe_4e_split_n100": ("MVMoE/4E-Split", 100, "MVMoE/4E"),
    "mvmoe_4el_split_n50": ("MVMoE/4E-L-Split", 50, "MVMoE/4E-L"),
    "mvmoe_4el_split_n100": ("MVMoE/4E-L-Split", 100, "MVMoE/4E-L"),
}

# Published direct-construction objectives from Tables 1 and 2 of:
# Zhou et al., "MVMoE: Multi-Task Vehicle Routing Solver with
# Mixture-of-Experts", ICML 2024.
PUBLISHED = {
    "POMO-MTL": {
        "CVRP": (10.437, 15.790), "OVRP": (6.671, 10.169),
        "VRPB": (8.182, 12.072), "VRPL": (10.513, 15.846),
        "VRPTW": (15.032, 25.610), "OVRPTW": (8.987, 15.008),
        "OVRPB": (6.116, 8.979), "OVRPL": (6.668, 10.126),
        "VRPBL": (8.188, 11.998), "VRPBTW": (16.055, 27.319),
        "VRPLTW": (14.961, 25.619), "OVRPBL": (6.104, 8.961),
        "OVRPBTW": (9.514, 15.879), "OVRPLTW": (8.987, 14.896),
        "VRPBLTW": (15.980, 27.247), "OVRPBLTW": (9.532, 15.738),
    },
    "MVMoE/4E": {
        "CVRP": (10.428, 15.760), "OVRP": (6.655, 10.138),
        "VRPB": (8.170, 12.027), "VRPL": (10.501, 15.812),
        "VRPTW": (14.999, 25.512), "OVRPTW": (8.964, 14.927),
        "OVRPB": (6.092, 8.959), "OVRPL": (6.650, 10.097),
        "VRPBL": (8.172, 11.945), "VRPBTW": (16.022, 27.236),
        "VRPLTW": (14.937, 25.514), "OVRPBL": (6.076, 8.942),
        "OVRPBTW": (9.486, 15.808), "OVRPLTW": (8.966, 14.828),
        "VRPBLTW": (15.945, 27.142), "OVRPBLTW": (9.503, 15.671),
    },
    "MVMoE/4E-L": {
        "CVRP": (10.434, 15.771), "OVRP": (6.665, 10.145),
        "VRPB": (8.176, 12.036), "VRPL": (10.506, 15.821),
        "VRPTW": (15.013, 25.519), "OVRPTW": (8.974, 14.940),
        "OVRPB": (6.122, 8.972), "OVRPL": (6.659, 10.106),
        "VRPBL": (8.180, 11.960), "VRPBTW": (16.041, 27.265),
        "VRPLTW": (14.953, 25.529), "OVRPBL": (6.104, 8.957),
        "OVRPBTW": (9.515, 15.841), "OVRPLTW": (8.974, 14.839),
        "VRPBLTW": (15.963, 27.177), "OVRPBLTW": (9.518, 15.706),
    },
}


def finite(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def open_training_csv(path: Path):
    """Open an original CSV or stream its lossless .zst archive as text."""
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            yield handle
        return

    archive = Path(f"{path}.zst")
    if not archive.is_file():
        raise FileNotFoundError(f"neither {path} nor {archive} exists")
    process = subprocess.Popen(
        ["zstd", "-dc", str(archive)],
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert process.stdout is not None
    try:
        yield process.stdout
    finally:
        process.stdout.close()
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"zstd failed for {archive} with exit code {return_code}")


def summarize_evaluation(path: Path, model_key: str):
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    if len(rows) != 16000:
        raise ValueError(f"{path}: expected 16000 rows, found {len(rows)}")
    keys = {(row["problem"], row["instance"]) for row in rows}
    if len(keys) != len(rows):
        raise ValueError(f"{path}: duplicate (problem, instance) keys")
    model, size, direct_name = MODELS[model_key]
    output = []
    for env in ENVIRONMENTS:
        env_rows = [row for row in rows if row["problem"] == env]
        if len(env_rows) != 1000:
            raise ValueError(f"{path}: {env} has {len(env_rows)} rows")
        ok = [row for row in env_rows if row["status"] == "ok"]
        failures = Counter(row["status"] for row in env_rows if row["status"] != "ok")
        for row in ok:
            if not finite(row["cost"]) or not finite(row["vehicles"]):
                raise ValueError(f"{path}: non-finite successful result: {row}")
        mean_cost = sum(float(row["cost"]) for row in ok) / len(ok) if ok else math.nan
        mean_vehicles = sum(float(row["vehicles"]) for row in ok) / len(ok) if ok else math.nan
        published = PUBLISHED[direct_name][env][0 if size == 50 else 1]
        output.append({
            "model_key": model_key,
            "model": model,
            "n": size,
            "environment": env,
            "split": "IID" if env in IID else "OOD",
            "success": len(ok),
            "total": len(env_rows),
            "success_rate": len(ok) / len(env_rows),
            "mean_cost_successes": mean_cost,
            "mean_vehicles_successes": mean_vehicles,
            "published_direct_cost": published,
            "split_vs_published_percent": (mean_cost / published - 1.0) * 100.0 if ok else math.nan,
            "full_set_comparable": len(ok) == len(env_rows),
            "failure_counts": ";".join(f"{k}:{v}" for k, v in sorted(failures.items())),
        })
    unexpected = sorted({row["problem"] for row in rows} - set(ENVIRONMENTS))
    if unexpected:
        raise ValueError(f"{path}: unexpected environments {unexpected}")
    return output


def summarize_training(path: Path, model_key: str):
    counts = Counter()
    row_count = 0
    max_epoch = 0
    learning_rates = []
    nonfinite = 0
    last_epoch_summary = None
    numeric_fields = (
        "learning_rate", "score_mean", "solution_cost_mean", "loss_mean",
        "grad_norm", "valid_candidate_rate", "mean_routes", "step_seconds",
        "throughput_instances_per_second", "train_score_mean",
        "train_loss_mean", "train_valid_candidate_rate", "train_mean_routes",
        "grad_norm_mean", "epoch_total_seconds",
    )
    with open_training_csv(path) as handle:
        for row in csv.DictReader(handle):
            row_count += 1
            counts[row.get("record_type", "")] += 1
            epoch_text = row.get("epoch", "") or row.get("last_completed_epoch", "")
            try:
                epoch = int(float(epoch_text)) if epoch_text else 0
            except ValueError:
                epoch = 0
            max_epoch = max(max_epoch, epoch)
            lr = row.get("learning_rate", "")
            if lr:
                if finite(lr):
                    learning_rates.append(float(lr))
                else:
                    nonfinite += 1
            for field in numeric_fields:
                value = row.get(field, "")
                if value and not finite(value):
                    nonfinite += 1
            if row.get("train_score_mean", "") and epoch >= (int(float(last_epoch_summary.get("epoch", 0))) if last_epoch_summary else -1):
                last_epoch_summary = row.copy()
    if max_epoch != 5000:
        raise ValueError(f"{path}: max epoch is {max_epoch}, expected 5000")
    if nonfinite:
        raise ValueError(f"{path}: found {nonfinite} non-finite populated metrics")
    checkpoint = path.parent.parent / "checkpoint" / "epoch-5000.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return {
        "model_key": model_key,
        "rows": row_count,
        "record_type_counts": ";".join(f"{k}:{v}" for k, v in sorted(counts.items())),
        "max_epoch": max_epoch,
        "learning_rate_min": min(learning_rates) if learning_rates else math.nan,
        "learning_rate_max": max(learning_rates) if learning_rates else math.nan,
        "final_train_score": (last_epoch_summary or {}).get("train_score_mean", ""),
        "final_train_loss": (last_epoch_summary or {}).get("train_loss_mean", ""),
        "final_valid_rate": (last_epoch_summary or {}).get("train_valid_candidate_rate", ""),
        "final_mean_routes": (last_epoch_summary or {}).get("train_mean_routes", ""),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256(checkpoint),
    }


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=4):
    if value == "" or value is None:
        return "--"
    value = float(value)
    return f"{value:.{digits}f}" if math.isfinite(value) else "--"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "reproduction_records" / "mvmoe_split_final_20260917",
    )
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    results_dir = root / "evaluations" / "official_16env" / "results"
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    eval_rows = []
    train_rows = []
    for key in MODELS:
        eval_rows.extend(summarize_evaluation(results_dir / f"{key}.csv", key))
        train_rows.append(summarize_training(root / "models" / key / "training" / "training_metrics.csv", key))

    per_env_fields = list(eval_rows[0])
    write_csv(summary_dir / "official_16env_per_environment.csv", eval_rows, per_env_fields)
    write_csv(summary_dir / "training_and_checkpoint_summary.csv", train_rows, list(train_rows[0]))

    aggregate = []
    for key, (model, size, _) in MODELS.items():
        selected = [row for row in eval_rows if row["model_key"] == key]
        for split_name, wanted in (("IID", IID), ("OOD", set(ENVIRONMENTS) - IID), ("ALL", set(ENVIRONMENTS))):
            part = [row for row in selected if row["environment"] in wanted]
            successful_instances = sum(row["success"] for row in part)
            total_instances = sum(row["total"] for row in part)
            aggregate.append({
                "model_key": key,
                "model": model,
                "n": size,
                "split": split_name,
                "complete_environments": sum(row["full_set_comparable"] for row in part),
                "environments": len(part),
                "successful_instances": successful_instances,
                "total_instances": total_instances,
                "success_rate": successful_instances / total_instances,
                "macro_mean_cost_successes": sum(row["mean_cost_successes"] for row in part) / len(part),
                "macro_mean_vehicles_successes": sum(row["mean_vehicles_successes"] for row in part) / len(part),
            })
    write_csv(summary_dir / "official_16env_aggregate.csv", aggregate, list(aggregate[0]))

    published_rows = []
    for method, by_env in PUBLISHED.items():
        for env in ENVIRONMENTS:
            for size, value in zip((50, 100), by_env[env]):
                published_rows.append({"method": method, "n": size, "environment": env, "published_direct_cost": value})
    write_csv(summary_dir / "published_direct_reference.csv", published_rows, list(published_rows[0]))

    lines = [
        "# 六个 Split 模型：官方 16 环境最终评测",
        "",
        "- 数据：MVMoE 官方生成测试集，每个环境 1,000 个实例。",
        "- 推理：greedy、POMO size = n、8-fold augmentation、seed = 2024。",
        "- IID：CVRP、OVRP、VRPB、VRPL、VRPTW、OVRPTW；其余 10 个为 zero-shot OOD。",
        "- `mean cost` 和 `vehicles` 只在 `status=ok` 的实例上取均值。若成功数不足 1,000，不能把该条件均值当作完整测试集结果。",
        "- 原文 Direct 参考值来自 Zhou et al. (ICML 2024) Tables 1--2；由于当前 Split 评测在含 B+TW 环境存在失败，只有 1,000/1,000 时才能作完整集的直接数值比较。",
        "",
        "## 汇总",
        "",
        "| 模型 | n | 划分 | 完整环境 | 成功实例 | 成功率 | 环境宏平均代价* | 环境宏平均车辆* |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['model']} | {row['n']} | {row['split']} | "
            f"{row['complete_environments']}/{row['environments']} | "
            f"{row['successful_instances']}/{row['total_instances']} | "
            f"{100 * row['success_rate']:.2f}% | {row['macro_mean_cost_successes']:.4f} | "
            f"{row['macro_mean_vehicles_successes']:.3f} |"
        )
    lines += [
        "",
        "\\* 宏平均先在每个环境的成功实例上求均值，再对环境等权平均；失败环境的数值是条件统计。",
        "",
        "## 逐环境",
        "",
        "| 模型 | n | 环境 | 划分 | 成功 | 代价* | 车辆* | 原文 Direct | 相对变化** |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in eval_rows:
        delta = fmt(row["split_vs_published_percent"], 2) + "%"
        if not row["full_set_comparable"]:
            delta += " (条件)"
        lines.append(
            f"| {row['model']} | {row['n']} | {row['environment']} | {row['split']} | "
            f"{row['success']}/{row['total']} | {fmt(row['mean_cost_successes'])} | "
            f"{fmt(row['mean_vehicles_successes'], 3)} | {fmt(row['published_direct_cost'], 3)} | {delta} |"
        )
    lines += [
        "",
        "\\* 仅统计成功实例。  ",
        "\\** Split 成功实例条件均值相对原文 Direct 完整集均值；仅当成功数为 1,000/1,000 时是同口径完整集比较。",
        "",
        "## 训练与最终权重",
        "",
        "| 模型键 | CSV 行数 | 最终 epoch | 最终训练 score | 最终训练 loss | checkpoint 字节 | SHA-256 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in train_rows:
        lines.append(
            f"| {row['model_key']} | {row['rows']} | {row['max_epoch']} | "
            f"{fmt(row['final_train_score'], 6)} | {fmt(row['final_train_loss'], 6)} | "
            f"{row['checkpoint_bytes']} | `{row['checkpoint_sha256']}` |"
        )
    lines += [
        "",
        "## 验证结论",
        "",
        "六个结果文件均包含 16,000 个唯一 `(problem, instance)`，每个环境恰好 1,000 条；所有 `status=ok` 的 cost/vehicles 均为有限数。六份训练 CSV 均到达 epoch 5,000，填充的关键训练数值均有限，六个最终 checkpoint 均存在且已记录 SHA-256。",
        "",
        "B+TW 联合约束的四个 OOD 环境（VRPBTW、OVRPBTW、VRPBLTW、OVRPBLTW）仍是所有模型的主要失败来源。它们不是评测进程故障，而是评测器在现有 giant tour 候选池中没有找到可由 Split 切出的可行候选。",
        "",
        "原文：<https://proceedings.mlr.press/v235/zhou24c.html>",
    ]
    (summary_dir / "FINAL_REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"validated {len(MODELS)} models; wrote summaries to {summary_dir}")


if __name__ == "__main__":
    main()
