"""Compare epoch-200 checkpoints on the fixed CVRP100 set.

- Original POMO: native CVRP decoder (no Split)
- POMO_SPLIT: giant tour + Bellman Split (split reward in training)
- POMO_SPLIT_EVALONLY: giant tour trained with raw-tour reward; Split only at eval

All use the same vrp100_test_seed1234.pt and x8 augmentation by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from run_pomo_reproduction import (
    CaseConfig,
    DEFAULT_POMO_ROOT,
    run_case,
    validate_pomo_root,
    write_csv,
)


DEFAULT_POMO_CKPT_DIR = (
    DEFAULT_POMO_ROOT
    / "NEW_py_ver"
    / "CVRP"
    / "POMO"
    / "result"
    / "20260712_222446_train_cvrp_n100_with_instNorm"
)
DEFAULT_SPLIT_CKPT_DIR = (
    DEFAULT_POMO_ROOT
    / "NEW_py_ver"
    / "CVRP"
    / "POMO_SPLIT"
    / "result"
    / "20260713_013205_pomo_giant_tour_split_n100"
)
DEFAULT_EVALONLY_CKPT_DIR = (
    DEFAULT_POMO_ROOT
    / "NEW_py_ver"
    / "CVRP"
    / "POMO_SPLIT_EVALONLY"
    / "result"
    / "20260713_042905_pomo_giant_tour_noplit_train_n100"
)
DEFAULT_TEST_DATA = (
    DEFAULT_POMO_ROOT / "NEW_py_ver" / "CVRP" / "vrp100_test_seed1234.pt"
)
DEFAULT_OUTPUT = (
    DEFAULT_POMO_ROOT
    / "reproduction_outputs"
    / "compare_pomo_vs_split_vs_evalonly_ep200.csv"
)


def build_cases(args):
    test_data = Path(args.test_data).expanduser().resolve()
    epoch = args.epoch
    size = args.size
    cases = []

    if args.pomo_checkpoint_dir:
        pomo_dir = Path(args.pomo_checkpoint_dir).expanduser().resolve()
        cases.append(
            CaseConfig(
                name="cvrp{}_pomo_ep{}".format(size, epoch),
                problem="CVRP",
                size=size,
                checkpoint_dir=pomo_dir,
                checkpoint_epoch=epoch,
                default_episodes=args.episodes,
                aug_batch_size=400,
                batch_size_basis="official NEW_py_ver/CVRP/POMO/test_n100.py x8 aug_batch_size",
                no_aug_batch_size=1000,
                no_aug_batch_size_basis=(
                    "official NEW_py_ver/CVRP/POMO/test_n100.py no-augmentation batch"
                ),
                saved_test_data=test_data,
                note="Trained original POMO checkpoint, native CVRP decode, no Split.",
                use_split=False,
            )
        )

    if args.split_checkpoint_dir:
        split_dir = Path(args.split_checkpoint_dir).expanduser().resolve()
        cases.append(
            CaseConfig(
                name="cvrp{}_split_ep{}".format(size, epoch),
                problem="CVRP",
                size=size,
                checkpoint_dir=split_dir,
                checkpoint_epoch=epoch,
                default_episodes=args.episodes,
                aug_batch_size=400,
                batch_size_basis=(
                    "giant tour + Split; x8 aug pre-expansion batch (matches cvrp{})".format(
                        size
                    )
                ),
                no_aug_batch_size=1000,
                no_aug_batch_size_basis="giant tour + Split no-augmentation batch",
                saved_test_data=test_data,
                note="Trained POMO_SPLIT checkpoint, giant tour + Bellman Split.",
                use_split=True,
            )
        )

    if args.evalonly_checkpoint_dir:
        evalonly_dir = Path(args.evalonly_checkpoint_dir).expanduser().resolve()
        cases.append(
            CaseConfig(
                name="cvrp{}_evalonly_ep{}".format(size, epoch),
                problem="CVRP",
                size=size,
                checkpoint_dir=evalonly_dir,
                checkpoint_epoch=epoch,
                default_episodes=args.episodes,
                aug_batch_size=400,
                batch_size_basis=(
                    "EVALONLY weights; giant tour + Split at eval "
                    "(same decoder path as POMO_SPLIT)"
                ),
                no_aug_batch_size=1000,
                no_aug_batch_size_basis="EVALONLY giant tour + Split no-augmentation batch",
                saved_test_data=test_data,
                note=(
                    "POMO_SPLIT_EVALONLY: raw-tour train reward; "
                    "Bellman Split only at evaluation."
                ),
                use_split=True,
            )
        )

    if not cases:
        raise ValueError("No checkpoint dirs enabled; pass at least one.")
    return cases


def print_summary(rows, output):
    print("\n========== Compare summary (epoch checkpoints) ==========")
    header = (
        f"{'case':<32} {'use_split':>9} {'primary':>10} "
        f"{'no_aug':>10} {'x8_aug':>10} {'secs':>8}"
    )
    print(header)
    print("-" * len(header))
    by_name = {}
    for row in rows:
        by_name[row["case"]] = row
        print(
            f"{row['case']:<32} {str(row['use_split']):>9} "
            f"{float(row['primary_score']):>10.4f} "
            f"{float(row['no_aug_diagnostic_score']):>10.4f} "
            f"{float(row['x8_aug_score']):>10.4f} "
            f"{float(row['elapsed_seconds']):>8.1f}"
        )

    pomo_key = next((k for k in by_name if "_pomo_ep" in k), None)
    split_key = next((k for k in by_name if "_split_ep" in k), None)
    evalonly_key = next((k for k in by_name if "_evalonly_ep" in k), None)
    if pomo_key:
        pomo = by_name[pomo_key]
        print("-" * len(header))
        if split_key:
            split = by_name[split_key]
            print(
                "primary delta (split - pomo)     = {:+.4f}".format(
                    float(split["primary_score"]) - float(pomo["primary_score"])
                )
            )
            print(
                "x8_aug  delta (split - pomo)     = {:+.4f}".format(
                    float(split["x8_aug_score"]) - float(pomo["x8_aug_score"])
                )
            )
        if evalonly_key:
            evalonly = by_name[evalonly_key]
            print(
                "primary delta (evalonly - pomo)  = {:+.4f}".format(
                    float(evalonly["primary_score"]) - float(pomo["primary_score"])
                )
            )
            print(
                "x8_aug  delta (evalonly - pomo)  = {:+.4f}".format(
                    float(evalonly["x8_aug_score"]) - float(pomo["x8_aug_score"])
                )
            )
    print("CSV: {}".format(output))
    print("=========================================================\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Eval POMO / POMO_SPLIT / POMO_SPLIT_EVALONLY epoch checkpoints "
            "on fixed CVRP100."
        )
    )
    parser.add_argument("--pomo-root", default=str(DEFAULT_POMO_ROOT))
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument(
        "--size",
        type=int,
        default=100,
        help="CVRP problem size (must match checkpoint and test set).",
    )
    parser.add_argument(
        "--pomo-checkpoint-dir",
        default=str(DEFAULT_POMO_CKPT_DIR),
        help="Dir containing original POMO checkpoint-<epoch>.pt",
    )
    parser.add_argument(
        "--split-checkpoint-dir",
        default=str(DEFAULT_SPLIT_CKPT_DIR),
        help="Dir containing POMO_SPLIT checkpoint-<epoch>.pt",
    )
    parser.add_argument(
        "--evalonly-checkpoint-dir",
        default=str(DEFAULT_EVALONLY_CKPT_DIR),
        help="Dir containing POMO_SPLIT_EVALONLY checkpoint-<epoch>.pt",
    )
    parser.add_argument(
        "--skip-pomo",
        action="store_true",
        help="Skip original POMO case.",
    )
    parser.add_argument(
        "--skip-split",
        action="store_true",
        help="Skip POMO_SPLIT case.",
    )
    parser.add_argument(
        "--skip-evalonly",
        action="store_true",
        help="Skip POMO_SPLIT_EVALONLY case.",
    )
    parser.add_argument("--test-data", default=str(DEFAULT_TEST_DATA))
    parser.add_argument(
        "--episodes",
        type=int,
        default=10000,
        help="Number of fixed-test instances (default: full 10000).",
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="full",
        help="smoke forces 10 episodes; full uses --episodes.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Use no-augmentation mode instead of x8.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--overwrite", action="store_true", default=True)
    args = parser.parse_args()

    if args.skip_pomo:
        args.pomo_checkpoint_dir = None
    if args.skip_split:
        args.split_checkpoint_dir = None
    if args.skip_evalonly:
        args.evalonly_checkpoint_dir = None

    pomo_root = Path(args.pomo_root).expanduser().resolve()
    validate_pomo_root(pomo_root)

    if args.mode == "smoke":
        args.episodes = 10

    cases = build_cases(args)
    for case in cases:
        ckpt = case.checkpoint_dir / "checkpoint-{}.pt".format(case.checkpoint_epoch)
        if not ckpt.is_file():
            raise FileNotFoundError("Missing checkpoint: {}".format(ckpt))
        print(
            "Will eval {} | use_split={} | {}".format(
                case.name, case.use_split, ckpt
            ),
            flush=True,
        )

    aug_factor = 1 if args.no_augmentation else 8
    rows = []
    for case in cases:
        print("\n>>> Running {} ...".format(case.name), flush=True)
        rows.append(
            run_case(
                case,
                pomo_root,
                args.episodes,
                args.device,
                args.cuda_device,
                args.seed,
                args.mode,
                aug_factor,
                args.batch_size,
            )
        )

    output = Path(args.output).expanduser().resolve()
    write_csv(rows, output, overwrite=True)
    print("Wrote {}".format(output), flush=True)
    print_summary(rows, output)


if __name__ == "__main__":
    # Allow `python compare_...py` from reproduction_tools/ or repo root.
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    main()
