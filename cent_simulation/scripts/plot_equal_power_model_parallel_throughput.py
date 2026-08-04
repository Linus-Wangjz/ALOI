#!/usr/bin/env python3
"""Compare DGX H100 and linearly scaled CENT model-parallel throughput."""

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

MODELS = {
    "Llama2-7B": {"gpu_model": "Llama-3.1-8B", "layers": 32, "power_per_gpu_w": 581.5},
    "Llama2-70B": {"gpu_model": "Llama-3.1-70B", "layers": 80, "power_per_gpu_w": 556.3},
}

CENT_CASES = {
    "GDDR6": {
        "csv": OUTPUT_ROOT / "GDDR6/simulation_results_decode_only_long_context_midpoint.csv",
        "target_cards": 128,
        "color": "#3B6EA8",
        "hatch": "",
    },
    "LPDDR4X nCCD=2": {
        "csv": OUTPUT_ROOT / "LPDDR4X/simulation_results_decode_only_long_context_midpoint_nCCD2.csv",
        "target_cards": 288,
        "color": "#2A9D8F",
        "hatch": "//",
    },
    "LPDDR4X nCCD=6": {
        "csv": OUTPUT_ROOT / "LPDDR4X/simulation_results_decode_only_long_context_midpoint_nCCD6.csv",
        "target_cards": 512,
        "color": "#D95F02",
        "hatch": "\\\\",
    },
}

GPU_CASE = "DGX H100 (8 GPUs)"
GPU_COLOR = "#707070"


def parse_contexts(raw: str) -> list[tuple[str, int]]:
    contexts: list[tuple[str, int]] = []
    for part in raw.split(","):
        if not part.strip():
            continue
        label, value = part.split("=", 1)
        contexts.append((label.strip(), int(value.strip())))
    if not contexts:
        raise ValueError("--contexts must not be empty")
    return contexts


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_h100_rows(profile_path: Path, contexts: list[tuple[str, int]]) -> list[dict[str, object]]:
    df = pd.read_csv(profile_path)
    rows: list[dict[str, object]] = []
    for model, metadata in MODELS.items():
        for context_label, context_window in contexts:
            subset = df[
                (df["model"] == metadata["gpu_model"])
                & (df["num_gpu"] == 8)
                & (df["stage"] == "Decoding")
                & (df["seqlen"] == context_window)
            ]
            if subset.empty:
                raise ValueError(f"missing DGX H100 row for {metadata['gpu_model']} context={context_window}")
            best = subset.loc[subset["throughput (tokens/s)"].idxmax()]
            system_power_w = float(metadata["power_per_gpu_w"]) * int(best["num_gpu"])
            rows.append(
                {
                    "Model": model,
                    "Context": context_label,
                    "Context window": context_window,
                    "System": GPU_CASE,
                    "Base throughput (tokens/s)": float(best["throughput (tokens/s)"]),
                    "Scale factor": 1,
                    "Equal-power throughput (tokens/s)": float(best["throughput (tokens/s)"]),
                    "Steady-state system power (W)": system_power_w,
                    "Power / DGX H100": 1.0,
                    "Base device count": 8,
                    "Equal-power device count": 8,
                    "Pipeline parallelism": "",
                    "Tensor parallelism": "",
                    "Active sequence length": context_window,
                    "DGX batch": int(best["batch"]),
                    "Source": display_path(profile_path),
                }
            )
    return rows


def load_cent_rows(
    contexts: list[tuple[str, int]],
    cent_cases: dict[str, dict[str, object]],
    *,
    shard_kv_cache_across_tp: bool = True,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case, config in cent_cases.items():
        df = pd.read_csv(config["csv"])
        if "Context window" not in df.columns:
            raise ValueError(f"{config['csv']} does not record Context window; use midpoint results")
        for model, metadata in MODELS.items():
            model_parallel = df[
                (df["Model"] == model)
                & (df["Pipeline parallelism"] != metadata["layers"])
                & (df.get("DRAM energy model", "legacy") == "legacy")
            ]
            for context_label, context_window in contexts:
                subset = model_parallel[model_parallel["Context window"] == context_window]
                if subset.empty:
                    raise ValueError(f"missing model-parallel row for {case} {model} context={context_window}")
                best = select_capacity_constrained_best(
                    model_parallel,
                    model,
                    context_window,
                    shard_kv_cache_across_tp=shard_kv_cache_across_tp,
                )
                base_devices = int(best["Device number"])
                target_cards = int(config["target_cards"])
                if target_cards % base_devices:
                    raise ValueError(f"{case} target card count {target_cards} is not divisible by {model} base device count {base_devices}")
                scale = target_cards // base_devices
                base_throughput = float(best["Capacity-constrained throughput (tokens/s)"])
                system_power_w = float(best["Token energy (mJ)"]) * base_throughput * scale / 1000.0
                h100_power_w = float(metadata["power_per_gpu_w"]) * 8
                rows.append(
                    {
                        "Model": model,
                        "Context": context_label,
                        "Context window": context_window,
                        "System": f"{case} ({target_cards} PIM cards)",
                        "Base trace throughput (tokens/s)": float(best["Trace throughput (tokens/s)"]),
                        "Base throughput (tokens/s)": base_throughput,
                        "Scale factor": scale,
                        "Equal-power throughput (tokens/s)": base_throughput * scale,
                        "Steady-state system power (W)": system_power_w,
                        "Power / DGX H100": system_power_w / h100_power_w,
                        "Base device count": base_devices,
                        "Equal-power device count": target_cards,
                        "Pipeline parallelism": int(best["Pipeline parallelism"]),
                        "Tensor parallelism": int(best["Tensor parallelism"]),
                        "Active sequence length": int(best["Sequence length"]),
                        "Max resident microbatch": int(best["Max resident microbatch"]),
                        "Microbatch used for throughput": int(best["Microbatch used for throughput"]),
                        "Pipeline fill ratio": float(best["Pipeline fill ratio"]),
                        "DGX batch": "",
                        "Source": display_path(config["csv"]),
                    }
                )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, object]], contexts: list[tuple[str, int]], path: Path) -> None:
    systems = [GPU_CASE, *(f"{case} ({config['target_cards']} PIM cards)" for case, config in CENT_CASES.items())]
    labels = systems
    colors = [GPU_COLOR, *(CENT_CASES[case]["color"] for case in CENT_CASES)]
    hatches = ["", *(CENT_CASES[case]["hatch"] for case in CENT_CASES)]
    models = list(MODELS)
    lookup = {(str(row["Model"]), str(row["Context"]), str(row["System"])): float(row["Equal-power throughput (tokens/s)"]) for row in rows}
    power_ratio = {(str(row["Model"]), str(row["Context"]), str(row["System"])): float(row["Power / DGX H100"]) for row in rows}

    fig, axes = plt.subplots(1, len(models), figsize=(11.6, 4.1), sharey=False)
    x = np.arange(len(contexts))
    width = 0.19
    for ax, model in zip(axes, models):
        for index, prefix in enumerate(systems):
            matching = [row for row in rows if row["Model"] == model and str(row["System"]) == prefix]
            if not matching:
                raise ValueError(f"missing plot rows for {model} {prefix}")
            system = str(matching[0]["System"])
            values = [lookup[(model, context_label, system)] for context_label, _ in contexts]
            offset = (index - (len(systems) - 1) / 2) * width
            ax.bar(x + offset, values, width, label=labels[index], color=colors[index], hatch=hatches[index], edgecolor="black", linewidth=0.55)
        ax.set_title(model)
        ax.set_xticks(x)
        ax.set_xticklabels([label for label, _ in contexts])
        ax.set_ylabel("Equal-power max throughput (tokens/s)")
        ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.65)
        ax.set_axisbelow(True)
        # Card counts are calibrated from 4K. Show how the scaled systems'
        # actual power changes at the two longer windows directly below each
        # PIM bar; the DGX H100 reference is always 100%.
        for context_index, (context_label, _) in enumerate(contexts[1:], start=1):
            for system_index, system in enumerate(systems[1:], start=1):
                offset = (system_index - (len(systems) - 1) / 2) * width
                ax.text(
                    x[context_index] + offset,
                    -0.16,
                    f"{power_ratio[(model, context_label, system)]:.0%}",
                    ha="center",
                    va="top",
                    rotation=90,
                    transform=ax.get_xaxis_transform(),
                    fontsize=8,
                    color=colors[system_index],
                )

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 0.87))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", default="4K=4096,32K=32768,128K=131072")
    parser.add_argument("--h100-profile", type=Path, default=ROOT / "DGX_H100_profile_results.csv")
    parser.add_argument("--gddr6-csv", type=Path)
    parser.add_argument("--lpddr4x-nccd2-csv", type=Path)
    parser.add_argument("--lpddr4x-nccd6-csv", type=Path)
    parser.add_argument(
        "--master-attention",
        action="store_true",
        help="Use master-local KV-cache capacity instead of TP-sharded KV-cache capacity.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "figures/cent_equal_power_model_parallel_throughput.png")
    parser.add_argument("--source-csv", type=Path, default=OUTPUT_ROOT / "figures/cent_equal_power_model_parallel_throughput.csv")
    args = parser.parse_args()

    contexts = parse_contexts(args.contexts)
    cent_cases = {case: dict(config) for case, config in CENT_CASES.items()}
    if args.gddr6_csv:
        cent_cases["GDDR6"]["csv"] = args.gddr6_csv
    if args.lpddr4x_nccd2_csv:
        cent_cases["LPDDR4X nCCD=2"]["csv"] = args.lpddr4x_nccd2_csv
    if args.lpddr4x_nccd6_csv:
        cent_cases["LPDDR4X nCCD=6"]["csv"] = args.lpddr4x_nccd6_csv
    rows = load_h100_rows(args.h100_profile, contexts) + load_cent_rows(
        contexts,
        cent_cases,
        shard_kv_cache_across_tp=not args.master_attention,
    )
    rows.sort(key=lambda row: (list(MODELS).index(str(row["Model"])), [label for label, _ in contexts].index(str(row["Context"])), str(row["System"])))
    write_csv(rows, args.source_csv)
    plot(rows, contexts, args.output)
    print(f"[data] {display_path(args.source_csv)}")
    print(f"[plot] {display_path(args.output)}")
    print(f"[plot] {display_path(args.output.with_suffix('.pdf'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
