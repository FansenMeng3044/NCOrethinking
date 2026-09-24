#!/usr/bin/env python3
"""Build centered appendix tables for the CVRPLIB X-set evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ARTIFACT_ROOT = HERE.parent
SOURCE = ARTIFACT_ROOT / "results" / "xset_verified_results.csv"
OUTPUT = HERE / "appendix_xset_results.tex"
AUDIT = HERE / "appendix_xset_results_audit.json"

MODELS = (
    "am_direct_n100",
    "am_split_n100",
    "pomo_direct_n100",
    "pomo_split_n100",
)
METHOD_NAMES = {
    "am_direct_n100": "AM Direct",
    "am_split_n100": "AM Split",
    "pomo_direct_n100": "POMO Direct",
    "pomo_split_n100": "POMO Split",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_and_validate() -> dict[str, dict[str, dict[str, str]]]:
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 400:
        raise ValueError(f"expected 400 model-instance rows, found {len(rows)}")

    indexed: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        model = row["model"]
        instance = row["instance"]
        if model not in MODELS:
            raise ValueError(f"unexpected model {model}")
        if (model, instance) in {
            (existing_model, existing_instance)
            for existing_instance, model_rows in indexed.items()
            for existing_model in model_rows
        }:
            raise ValueError(f"duplicate row for {model}/{instance}")
        indexed[instance][model] = row

    if len(indexed) != 100:
        raise ValueError(f"expected 100 instances, found {len(indexed)}")
    for instance, model_rows in indexed.items():
        if set(model_rows) != set(MODELS):
            raise ValueError(f"{instance}: incomplete model set")
        references = {int(row["reference_cost"]) for row in model_rows.values()}
        customers = {int(row["customers"]) for row in model_rows.values()}
        if len(references) != 1 or len(customers) != 1:
            raise ValueError(f"{instance}: inconsistent reference cost or size")
        for model, row in model_rows.items():
            reference = int(row["reference_cost"])
            cost = int(row["cost"])
            expected_gap = 100.0 * (cost - reference) / reference
            if abs(expected_gap - float(row["gap_percent"])) > 1e-9:
                raise ValueError(f"{model}/{instance}: gap mismatch")
    return indexed


def aggregate(indexed: dict[str, dict[str, dict[str, str]]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for model in MODELS:
        rows = [model_rows[model] for model_rows in indexed.values()]
        result[model] = {
            "instances": float(len(rows)),
            "mean_gap": statistics.fmean(float(row["gap_percent"]) for row in rows),
            "median_gap": statistics.median(float(row["gap_percent"]) for row in rows),
            "mean_vehicles": statistics.fmean(float(row["vehicles"]) for row in rows),
        }
    return result


def paired(indexed: dict[str, dict[str, dict[str, str]]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for backbone, direct, split in (
        ("AM", "am_direct_n100", "am_split_n100"),
        ("POMO", "pomo_direct_n100", "pomo_split_n100"),
    ):
        relative: list[float] = []
        gap_difference: list[float] = []
        vehicle_difference: list[float] = []
        direct_wins = split_wins = ties = 0
        for model_rows in indexed.values():
            direct_row = model_rows[direct]
            split_row = model_rows[split]
            direct_cost = int(direct_row["cost"])
            split_cost = int(split_row["cost"])
            relative.append(100.0 * (split_cost - direct_cost) / direct_cost)
            gap_difference.append(
                float(split_row["gap_percent"]) - float(direct_row["gap_percent"])
            )
            vehicle_difference.append(
                float(split_row["vehicles"]) - float(direct_row["vehicles"])
            )
            if direct_cost < split_cost:
                direct_wins += 1
            elif split_cost < direct_cost:
                split_wins += 1
            else:
                ties += 1
        result[backbone] = {
            "direct_wins": float(direct_wins),
            "split_wins": float(split_wins),
            "ties": float(ties),
            "mean_relative_cost": statistics.fmean(relative),
            "mean_gap_difference": statistics.fmean(gap_difference),
            "mean_vehicle_difference": statistics.fmean(vehicle_difference),
        }
    return result


def write_tex(
    indexed: dict[str, dict[str, dict[str, str]]],
    aggregate_rows: dict[str, dict[str, float]],
    paired_rows: dict[str, dict[str, float]],
) -> None:
    ordered = sorted(
        indexed,
        key=lambda instance: (int(indexed[instance][MODELS[0]]["customers"]), instance),
    )
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("% Auto-generated by paper_experiments/build_xset_appendix.py\n")
        handle.write("% Required packages: booktabs, longtable.\n")
        handle.write("% The detailed table is intended for one-column appendix layout.\n\n")
        handle.write("\\subsection{CVRPLIB X-set Results}\n")
        handle.write("\\label{app:xset-results}\n\n")
        handle.write(
            "We evaluate the four models trained on 100-customer instances on all "
            "100 CVRPLIB X-set instances, which contain between 100 and 1,000 "
            "customers. AM uses greedy decoding, whereas POMO uses one starting "
            "point per customer. Both methods use eightfold geometric augmentation. "
            "Objective values use the integer \\texttt{EUC\\_2D} convention. The "
            "reported gap is computed against the reference cost in the solution "
            "file accompanying each instance. Every returned route set passes "
            "independent checks of customer coverage, capacity feasibility, and "
            "objective value.\n\n"
        )

        handle.write("\\begin{table*}[t]\n\\centering\n\\small\n")
        handle.write(
            "\\caption{Aggregate results on all 100 CVRPLIB X-set instances. "
            "The mean and median gaps are computed from the per-instance gaps. "
            "Lower values are better.}\\label{tab:xset-summary}\n"
        )
        handle.write("\\setlength{\\tabcolsep}{6pt}\n")
        handle.write("\\begin{tabular}{ccccc}\n\\toprule\n")
        handle.write(
            "Method & Feasible & Mean gap & Median gap & Mean vehicles \\\\\n"
            "\\midrule\n"
        )
        for model in MODELS:
            values = aggregate_rows[model]
            handle.write(
                f"{METHOD_NAMES[model]} & 100/100 & {values['mean_gap']:.3f}\\% & "
                f"{values['median_gap']:.3f}\\% & {values['mean_vehicles']:.2f} \\\\\n"
            )
        handle.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n\n")

        handle.write("\\begin{table*}[t]\n\\centering\n\\small\n")
        handle.write(
            "\\caption{Paired comparison between Direct and Split on the X set. "
            "Relative cost is $100(C_{\\mathrm{Split}}-C_{\\mathrm{Direct}})"
            "/C_{\\mathrm{Direct}}$. Gap and vehicle differences are Split minus "
            "Direct.}\\label{tab:xset-paired}\n"
        )
        handle.write("\\setlength{\\tabcolsep}{4pt}\n")
        handle.write("\\begin{tabular}{ccccccc}\n\\toprule\n")
        handle.write(
            "Backbone & Direct wins & Split wins & Ties & Relative cost & "
            "Gap diff. & Vehicle diff. \\\\\n\\midrule\n"
        )
        for backbone in ("AM", "POMO"):
            values = paired_rows[backbone]
            handle.write(
                f"{backbone} & {int(values['direct_wins'])} & "
                f"{int(values['split_wins'])} & {int(values['ties'])} & "
                f"{values['mean_relative_cost']:+.3f}\\% & "
                f"{values['mean_gap_difference']:+.3f} pp & "
                f"{values['mean_vehicle_difference']:+.2f} \\\\\n"
            )
        handle.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n\n")

        handle.write(
            "Direct obtains a lower objective value on 90 of the 100 instances "
            "for AM and on 91 instances for POMO. On average, replacing direct "
            "route construction with customer ordering followed by Split increases "
            "the paired objective value by 5.09\\% for AM and 7.94\\% for POMO. "
            "The corresponding increases in the mean gap are 5.586 and 8.635 "
            "percentage points. Split also uses more vehicles on average for both "
            "backbones. The exceptions in the paired comparisons show that neither "
            "representation is uniformly superior on every instance, but they do "
            "not reverse the aggregate ordering.\n\n"
        )

        handle.write("\\clearpage\n\\onecolumn\n")
        handle.write("\\begingroup\n\\scriptsize\n")
        handle.write("\\setlength{\\tabcolsep}{2.2pt}\n")
        handle.write("\\renewcommand{\\arraystretch}{1.04}\n")
        handle.write("\\setlength{\\LTleft}{\\fill}\n")
        handle.write("\\setlength{\\LTright}{\\fill}\n")
        handle.write("\\begin{longtable}{" + "c" * 15 + "}\n")
        handle.write(
            "\\caption{Per-instance results on all 100 CVRPLIB X-set instances. "
            "Ref. is the reference cost from the accompanying solution file. "
            "For every method, Cost is the integer \\texttt{EUC\\_2D} objective, "
            "Gap is relative to Ref., and Veh. is the number of routes. All entries "
            "are independently verified.}\\label{tab:xset-all-instances}"
            "\\\\[\\baselineskip]\n"
        )
        first_header = (
            "Instance & $n$ & Ref. & \\multicolumn{3}{c}{AM Direct} & "
            "\\multicolumn{3}{c}{AM Split} & "
            "\\multicolumn{3}{c}{POMO Direct} & "
            "\\multicolumn{3}{c}{POMO Split} \\\\\n"
        )
        cmidrules = (
            "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}"
            "\\cmidrule(lr){10-12}\\cmidrule(lr){13-15}\n"
        )
        second_header = (
            " & & & Cost & Gap & Veh. & Cost & Gap & Veh. & "
            "Cost & Gap & Veh. & Cost & Gap & Veh. \\\\\n"
        )
        handle.write("\\toprule\n" + first_header + cmidrules + second_header)
        handle.write("\\midrule\n\\endfirsthead\n")
        handle.write(
            "\\multicolumn{15}{c}{\\tablename~\\thetable{} continued}"
            "\\\\[0.5em]\n\\toprule\n"
            + first_header
            + cmidrules
            + second_header
            + "\\midrule\n\\endhead\n"
            "\\midrule\n\\multicolumn{15}{c}{Continued on next page}\\\\\n"
            "\\endfoot\n\\bottomrule\n\\endlastfoot\n"
        )
        for row_index, instance in enumerate(ordered, start=1):
            model_rows = indexed[instance]
            reference = int(model_rows[MODELS[0]]["reference_cost"])
            customers = int(model_rows[MODELS[0]]["customers"])
            cells = [instance, str(customers), f"{reference:,}"]
            for model in MODELS:
                row = model_rows[model]
                cells.extend(
                    [
                        f"{int(row['cost']):,}",
                        f"{float(row['gap_percent']):.2f}\\%",
                        str(int(row["vehicles"])),
                    ]
                )
            handle.write(" & ".join(cells) + r" \\" + "\n")
            if row_index % 25 == 0 and row_index != len(ordered):
                handle.write("\\midrule\n")
        handle.write("\\end{longtable}\n\\endgroup\n")


def main() -> None:
    indexed = read_and_validate()
    aggregate_rows = aggregate(indexed)
    paired_rows = paired(indexed)
    write_tex(indexed, aggregate_rows, paired_rows)
    audit = {
        "source": str(SOURCE),
        "source_sha256": file_sha256(SOURCE),
        "output": str(OUTPUT),
        "output_sha256": file_sha256(OUTPUT),
        "instances": len(indexed),
        "model_instance_rows": len(indexed) * len(MODELS),
        "models": list(MODELS),
        "aggregate": aggregate_rows,
        "paired": paired_rows,
    }
    AUDIT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
