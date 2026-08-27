#!/usr/bin/env python3
"""Render one saved XML100 solution as a publication-ready route figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from xml100_io import parse_instance, verify_routes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--solution-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("aug", "no_aug"), default="aug")
    parser.add_argument("--label-customers", action="store_true")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    instance = parse_instance(args.instance.expanduser().resolve())
    payload = json.loads(args.solution_json.expanduser().resolve().read_text(encoding="utf-8"))
    if payload.get("instance") != instance.name:
        raise ValueError("instance and solution JSON names do not match")
    if args.variant == "aug":
        status = payload.get("status")
        routes = payload.get("routes")
        reported_cost = payload.get("cost")
        label = "8-fold" if payload.get("protocol") else "augmented"
    else:
        status = payload.get("no_aug_status")
        routes = payload.get("no_aug_routes")
        reported_cost = payload.get("no_aug_cost")
        label = "no augmentation"
    if status != "ok":
        raise ValueError(f"cannot plot {args.variant} result with status {status!r}")
    verification = verify_routes(instance, routes)
    if not verification.feasible or verification.cost != int(reported_cost):
        raise ValueError(f"saved route failed verification: {verification}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required only for route visualization") from exc

    figure, axis = plt.subplots(figsize=(8.2, 8.2), constrained_layout=True)
    colors = plt.get_cmap("turbo")
    route_count = len(routes)
    for route_index, route in enumerate(routes):
        nodes = [0, *route, 0]
        xy = instance.coords[nodes]
        color = colors(route_index / max(route_count - 1, 1))
        axis.plot(xy[:, 0], xy[:, 1], color=color, linewidth=1.25, alpha=0.9)
        axis.scatter(xy[1:-1, 0], xy[1:-1, 1], color=[color], s=15, zorder=3)
    depot = instance.coords[0]
    axis.scatter(
        [depot[0]], [depot[1]], marker="*", s=190, c="black", edgecolors="white", linewidths=0.8,
        zorder=5, label="depot"
    )
    if args.label_customers:
        for customer, (x, y) in enumerate(instance.coords[1:], start=1):
            axis.annotate(str(customer), (x, y), xytext=(2, 2), textcoords="offset points", fontsize=5)
    model = payload.get("model", "model")
    gap = 100.0 * (verification.cost - int(payload["optimum"])) / int(payload["optimum"])
    axis.set_title(
        f"{instance.name} — {model} ({label})\n"
        f"cost={verification.cost:,}, gap={gap:.3f}%, vehicles={verification.vehicles}"
    )
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.15)
    axis.legend(loc="best")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.expanduser().resolve(), dpi=args.dpi)
    plt.close(figure)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
