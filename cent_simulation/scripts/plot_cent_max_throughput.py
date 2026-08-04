#!/usr/bin/env python3
"""Plot decode-only maximum token throughput for CENT memory cases."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cent")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cent_capacity import select_capacity_constrained_best


ROOT = Path(__file__).resolve().parents[2]
CENT_SIM = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = CENT_SIM / "output"

DEFAULT_CASES = {
    "GDDR6": OUTPUT_ROOT / "GDDR6/simulation_results_decode_only_long_context_midpoint.csv",
    "LPDDR4X nCCD=2": OUTPUT_ROOT / "LPDDR4X/simulation_results_decode_only_long_context_midpoint_nCCD2.csv",
    "LPDDR4X nCCD=6": OUTPUT_ROOT / "LPDDR4X/simulation_results_decode_only_long_context_midpoint_nCCD6.csv",
}


def parse_contexts(raw: str) -> list[tuple[str, int]]:
    contexts: list[tuple[str, int]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            label, value = part.split("=", 1)
            contexts.append((label.strip(), int(value.strip())))
        else:
            value = int(part)
            contexts.append((f"{value // 1024}K", value))
    if not contexts:
        raise ValueError("--contexts must not be empty")
    return contexts


def parse_models(raw: str) -> list[str]:
    models = [part.strip() for part in raw.split(",") if part.strip()]
    if not models:
        raise ValueError("--models must not be empty")
    return models


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def collect_rows(
    models: list[str],
    contexts: list[tuple[str, int]],
    case_csvs: dict[str, Path],
    shard_kv_cache_across_tp: bool = True,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    missing: list[str] = []

    for case, csv_path in case_csvs.items():
        if not csv_path.exists():
            missing.append(display_path(csv_path))
            continue
        df = pd.read_csv(csv_path)
        context_column = "Context window" if "Context window" in df.columns else "Sequence length"
        for model in models:
            for context_label, context_window in contexts:
                subset = df[(df["Model"] == model) & (df[context_column] == context_window)]
                if subset.empty:
                    missing.append(f"{case}: {model} {context_label} ({context_window})")
                    continue
                best = select_capacity_constrained_best(
                    df,
                    model,
                    context_window,
                    shard_kv_cache_across_tp=shard_kv_cache_across_tp,
                )
                rows.append(
                    {
                        "Case": case,
                        "Model": model,
                        "Context": context_label,
                        "Context window": int(best[context_column]),
                        "Active sequence length": int(best["Sequence length"]),
                        "Trace throughput (tokens/s)": float(best["Trace throughput (tokens/s)"]),
                        "Max throughput (tokens/s)": float(best["Capacity-constrained throughput (tokens/s)"]),
                        "Token latency (ms)": float(best["Token latency (ms)"]),
                        "Token energy (mJ)": float(best["Token energy (mJ)"]),
                        "Pipeline parallelism": int(best["Pipeline parallelism"]),
                        "Tensor parallelism": int(best["Tensor parallelism"]),
                        "Max resident microbatch": int(best["Max resident microbatch"]),
                        "Microbatch used for throughput": int(best["Microbatch used for throughput"]),
                        "Pipeline fill ratio": float(best["Pipeline fill ratio"]),
                        "System power (W)": float(best["Capacity-constrained system power (W)"]),
                        "Source CSV": display_path(csv_path),
                    }
                )

    if missing:
        details = "\n".join(f"  - {item}" for item in missing)
        raise RuntimeError(f"missing input rows:\n{details}")

    return rows


def write_source_csv(rows: list[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(
    rows: list[dict[str, object]],
    models: list[str],
    contexts: list[tuple[str, int]],
    cases: list[str],
    output: Path,
) -> None:
    colors = {
        "GDDR6": "#3b6ea8",
        "LPDDR4X nCCD=2": "#2a9d8f",
        "LPDDR4X nCCD=6": "#d95f02",
    }
    hatch = {
        "GDDR6": "",
        "LPDDR4X nCCD=2": "//",
        "LPDDR4X nCCD=6": "\\\\",
    }

    lookup = {
        (str(row["Model"]), str(row["Context"]), str(row["Case"])): float(row["Max throughput (tokens/s)"])
        for row in rows
    }

    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 3.6), sharey=False)
    if len(models) == 1:
        axes = [axes]

    x = np.arange(len(contexts))
    width = 0.24

    for ax, model in zip(axes, models):
        for index, case in enumerate(cases):
            values = [lookup[(model, context_label, case)] for context_label, _ in contexts]
            offset = (index - (len(cases) - 1) / 2) * width
            ax.bar(
                x + offset,
                values,
                width,
                label=case,
                color=colors[case],
                hatch=hatch[case],
                edgecolor="black",
                linewidth=0.6,
            )
        ax.set_title(model)
        ax.set_xticks(x)
        ax.set_xticklabels([label for label, _ in contexts])
        ax.set_xlabel("Context length")
        ax.set_ylabel("Capacity-constrained decode throughput (tokens/s)")
        ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(cases), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="Llama2-7B,Llama2-70B")
    parser.add_argument("--contexts", default="4K=4096,32K=32768,128K=131072")
    parser.add_argument("--gddr6-csv", type=Path)
    parser.add_argument("--lpddr4x-nccd2-csv", type=Path)
    parser.add_argument("--lpddr4x-nccd6-csv", type=Path)
    parser.add_argument(
        "--master-attention",
        action="store_true",
        help="Use master-local KV-cache capacity instead of TP-sharded KV-cache capacity.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "figures/cent_max_decode_throughput.png")
    parser.add_argument("--source-csv", type=Path, default=OUTPUT_ROOT / "figures/cent_max_decode_throughput.csv")
    args = parser.parse_args()

    models = parse_models(args.models)
    contexts = parse_contexts(args.contexts)
    case_csvs = {
        "GDDR6": args.gddr6_csv or DEFAULT_CASES["GDDR6"],
        "LPDDR4X nCCD=2": args.lpddr4x_nccd2_csv or DEFAULT_CASES["LPDDR4X nCCD=2"],
        "LPDDR4X nCCD=6": args.lpddr4x_nccd6_csv or DEFAULT_CASES["LPDDR4X nCCD=6"],
    }
    rows = collect_rows(
        models,
        contexts,
        case_csvs,
        shard_kv_cache_across_tp=not args.master_attention,
    )
    write_source_csv(rows, args.source_csv)
    plot(rows, models, contexts, list(case_csvs), args.output)

    print(f"[data] {display_path(args.source_csv)}")
    print(f"[plot] {display_path(args.output)}")
    print(f"[plot] {display_path(args.output.with_suffix('.pdf'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
