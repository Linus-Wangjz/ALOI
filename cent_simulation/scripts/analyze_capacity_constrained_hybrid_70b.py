#!/usr/bin/env python3
"""Apply a BF16 KV-capacity limit to 32-device Llama2-70B hybrid PP/TP results.

The functional/Ramulator traces model one decode token with one resident KV
cache.  The reported throughput therefore assumes a fully filled pipeline but
does not reserve KV cache for every independent request needed to fill it.
This tool turns those token-level measurements into an admission-controlled
estimate: it finds the largest resident microbatch that fits in every PIM
device, then limits the pipeline fill factor accordingly.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cent")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CENT_SIM = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = CENT_SIM / "output"

MODEL = "Llama2-70B"
LAYERS = 80
DIM = 8192
KV_HEADS = 8
HEAD_DIM = 128
FFN_DIM = 28672
VOCAB_SIZE = 32000
BF16_BYTES = 2

DEFAULT_CASES = {
    "GDDR6": OUTPUT_ROOT / "GDDR6/simulation_results_decode_only_long_context_midpoint.csv",
    "LPDDR4X nCCD=2": OUTPUT_ROOT / "LPDDR4X/simulation_results_decode_only_long_context_midpoint_nCCD2.csv",
    "LPDDR4X nCCD=6": OUTPUT_ROOT / "LPDDR4X/simulation_results_decode_only_long_context_midpoint_nCCD6.csv",
}

CASE_STYLE = {
    "GDDR6": ("#3b6ea8", ""),
    "LPDDR4X nCCD=2": ("#2a9d8f", "//"),
    "LPDDR4X nCCD=6": ("#d95f02", "\\\\"),
}


def parse_contexts(raw: str) -> list[tuple[str, int]]:
    contexts: list[tuple[str, int]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        label, value = part.split("=", 1)
        contexts.append((label.strip(), int(value.strip())))
    if not contexts:
        raise ValueError("--contexts must not be empty")
    return contexts


def gibibytes(value: float) -> float:
    return value / (1024**3)


def transformer_layer_weight_bytes() -> int:
    """BF16 weight footprint of one Llama2-70B transformer block.

    This follows the tensors used by ``function_sim.py``: Q, K, V, O, and
    SwiGLU's W1/W3/W2.  RMSNorm vectors are included although tiny.
    """

    kv_dim = KV_HEADS * HEAD_DIM
    elements = (
        DIM * DIM  # Wq
        + 2 * kv_dim * DIM  # Wk, Wv
        + DIM * DIM  # Wo
        + 3 * FFN_DIM * DIM  # W1, W3, W2
        + 2 * DIM  # two RMSNorm vectors
    )
    return elements * BF16_BYTES


LAYER_WEIGHT_BYTES = transformer_layer_weight_bytes()
ENDPOINT_WEIGHT_BYTES = VOCAB_SIZE * DIM * BF16_BYTES


def kv_bytes_per_layer(context_window: int) -> int:
    return 2 * context_window * KV_HEADS * HEAD_DIM * BF16_BYTES


def layer_counts_per_stage(pp: int) -> list[int]:
    base, remainder = divmod(LAYERS, pp)
    return [base + (stage < remainder) for stage in range(pp)]


def capacity_for_layout(
    *,
    pp: int,
    tp: int,
    context_window: int,
    device_capacity_bytes: int,
    reserve_bytes: int,
    shard_kv_cache_across_tp: bool = True,
) -> dict[str, float | int]:
    """Return the limiting per-device capacity for a balanced PP/TP layout.

    The 80 blocks are placed across PP stages as evenly as possible.  Each
    stage is tensor-sharded across TP devices.  The cache is either sharded
    too, or retained by the stage master to reproduce CENT's original
    master-attention mapping. Input embedding lives at the first stage and
    LM-head output embedding at the final stage. Llama2 does not tie these
    two matrices, so each is included.
    """

    layers_per_stage = layer_counts_per_stage(pp)
    per_layer_kv = kv_bytes_per_layer(context_window)
    capacities: list[tuple[int, int, int, int]] = []

    for stage, layer_count in enumerate(layers_per_stage):
        static_weights = layer_count * LAYER_WEIGHT_BYTES
        if stage == 0:
            static_weights += ENDPOINT_WEIGHT_BYTES
        if stage == pp - 1:
            static_weights += ENDPOINT_WEIGHT_BYTES
        static_per_device = math.ceil(static_weights / tp) + reserve_bytes
        kv_bytes_per_stage_request = layer_count * per_layer_kv
        kv_per_request_per_device = (
            math.ceil(kv_bytes_per_stage_request / tp)
            if shard_kv_cache_across_tp
            else kv_bytes_per_stage_request
        )
        available = device_capacity_bytes - static_per_device
        if available < 0:
            max_microbatch = 0
        else:
            max_microbatch = available // kv_per_request_per_device
        capacities.append((max_microbatch, stage, static_per_device, kv_per_request_per_device))

    max_microbatch, bottleneck_stage, static_per_device, kv_per_request = min(capacities)
    return {
        "Max resident microbatch": int(max_microbatch),
        "Bottleneck pipeline stage": int(bottleneck_stage),
        "Layers at bottleneck stage": int(layers_per_stage[bottleneck_stage]),
        "Static model footprint / bottleneck device (GiB)": gibibytes(static_per_device),
        "KV cache / request / bottleneck device (GiB)": gibibytes(kv_per_request),
        "KV cache / request / model (GiB)": gibibytes(LAYERS * per_layer_kv),
        "KV cache mapping": "TP-sharded" if shard_kv_cache_across_tp else "master-local",
    }


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def find_hybrid_candidates(df: pd.DataFrame, context_window: int) -> pd.DataFrame:
    context_column = "Context window" if "Context window" in df.columns else "Sequence length"
    candidates = df[(df["Model"] == MODEL) & (df[context_column] == context_window)].copy()
    candidates = candidates[
        candidates["Pipeline parallelism"] * candidates["Tensor parallelism"]
        == candidates["Device number"]
    ]
    if candidates.empty:
        raise ValueError(f"no PP×TP=device-count hybrid rows for {MODEL} context={context_window}")
    return candidates


def evaluate_case(
    *,
    case: str,
    source_csv: Path,
    contexts: list[tuple[str, int]],
    device_capacity_bytes: int,
    reserve_bytes: int,
    shard_kv_cache_across_tp: bool,
) -> list[dict[str, object]]:
    df = pd.read_csv(source_csv)
    rows: list[dict[str, object]] = []

    for context_label, context_window in contexts:
        candidates = find_hybrid_candidates(df, context_window)
        evaluated: list[dict[str, object]] = []
        for _, candidate in candidates.iterrows():
            pp = int(candidate["Pipeline parallelism"])
            tp = int(candidate["Tensor parallelism"])
            device_count = int(candidate["Device number"])
            if pp * tp != device_count:
                continue
            capacity = capacity_for_layout(
                pp=pp,
                tp=tp,
                context_window=context_window,
                device_capacity_bytes=device_capacity_bytes,
                reserve_bytes=reserve_bytes,
                shard_kv_cache_across_tp=shard_kv_cache_across_tp,
            )
            resident_microbatch = int(capacity["Max resident microbatch"])
            used_microbatch = min(resident_microbatch, pp)
            pipeline_fill = used_microbatch / pp
            unconstrained_throughput = float(candidate["Throughput (tokens/s)"])
            throughput = unconstrained_throughput * pipeline_fill
            token_energy = float(candidate["Token energy (mJ)"])
            system_power = token_energy * throughput / 1000.0
            active_devices = device_count * float(candidate["Device utilization"])

            evaluated.append(
                {
                    "Case": case,
                    "Model": MODEL,
                    "Context": context_label,
                    "Context window": context_window,
                    "Active sequence length": int(candidate["Sequence length"]),
                    "Pipeline parallelism": pp,
                    "Tensor parallelism": tp,
                    "Device count": device_count,
                    "Max resident microbatch": resident_microbatch,
                    "Microbatch used for throughput": used_microbatch,
                    "Pipeline fill ratio": pipeline_fill,
                    "Bottleneck pipeline stage": capacity["Bottleneck pipeline stage"],
                    "Layers at bottleneck stage": capacity["Layers at bottleneck stage"],
                    "Static model footprint / bottleneck device (GiB)": capacity[
                        "Static model footprint / bottleneck device (GiB)"
                    ],
                    "KV cache / request / bottleneck device (GiB)": capacity[
                        "KV cache / request / bottleneck device (GiB)"
                    ],
                    "KV cache / request / model (GiB)": capacity["KV cache / request / model (GiB)"],
                    "KV cache mapping": capacity["KV cache mapping"],
                    "Unconstrained throughput (tokens/s)": unconstrained_throughput,
                    "Capacity-constrained throughput (tokens/s)": throughput,
                    "Token energy (mJ)": token_energy,
                    "System power (W)": system_power,
                    "Power / active PIM device (W)": system_power / active_devices,
                    "Power / deployed PIM device (W)": system_power / device_count,
                    "Device capacity (GiB)": gibibytes(device_capacity_bytes),
                    "Per-device reserve (GiB)": gibibytes(reserve_bytes),
                    "Source CSV": display_path(source_csv),
                }
            )

        if not evaluated:
            raise RuntimeError(f"no valid hybrid candidates for {case} {context_label}")
        # Prefer lower token energy only when capacity-constrained throughput ties.
        rows.append(
            max(
                evaluated,
                key=lambda row: (
                    float(row["Capacity-constrained throughput (tokens/s)"]),
                    -float(row["Token energy (mJ)"]),
                ),
            )
        )
    return rows


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(
    rows: list[dict[str, object]],
    contexts: list[tuple[str, int]],
    cases: list[str],
    output: Path,
) -> None:
    metrics = [
        ("Capacity-constrained throughput (tokens/s)", "Capacity-constrained throughput\n(tokens/s)"),
        ("Token energy (mJ)", "Token energy\n(mJ/token)"),
        ("System power (W)", "System power\n(W)"),
        ("Power / active PIM device (W)", "Power / active PIM device\n(W)"),
    ]
    lookup = {(str(row["Case"]), str(row["Context"])): row for row in rows}
    x = np.arange(len(contexts))
    width = 0.23
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0))

    for axis, (metric, ylabel) in zip(axes.flat, metrics):
        for index, case in enumerate(cases):
            values = [float(lookup[(case, label)][metric]) for label, _ in contexts]
            color, hatch = CASE_STYLE[case]
            offset = (index - (len(cases) - 1) / 2) * width
            axis.bar(x + offset, values, width, label=case, color=color, hatch=hatch, edgecolor="black", linewidth=0.6)
        axis.set_xticks(x)
        axis.set_xticklabels([label for label, _ in contexts])
        axis.set_xlabel("Context window")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.65)
        axis.set_axisbelow(True)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(cases), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", default="4K=4096,32K=32768,128K=131072")
    parser.add_argument("--device-capacity-gib", type=float, default=16.0)
    parser.add_argument("--per-device-reserve-gib", type=float, default=0.0)
    parser.add_argument(
        "--master-attention",
        action="store_true",
        help="Keep each TP stage's KV cache on its master device instead of TP-sharding it.",
    )
    parser.add_argument("--gddr6-csv", type=Path, default=DEFAULT_CASES["GDDR6"])
    parser.add_argument("--lpddr4x-nccd2-csv", type=Path, default=DEFAULT_CASES["LPDDR4X nCCD=2"])
    parser.add_argument("--lpddr4x-nccd6-csv", type=Path, default=DEFAULT_CASES["LPDDR4X nCCD=6"])
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=OUTPUT_ROOT / "figures/cent_capacity_constrained_hybrid_70b.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT / "figures/cent_capacity_constrained_hybrid_70b.png",
    )
    args = parser.parse_args()

    contexts = parse_contexts(args.contexts)
    device_capacity_bytes = int(args.device_capacity_gib * 1024**3)
    reserve_bytes = int(args.per_device_reserve_gib * 1024**3)
    if device_capacity_bytes <= reserve_bytes:
        raise ValueError("--device-capacity-gib must exceed --per-device-reserve-gib")

    case_csvs = {
        "GDDR6": args.gddr6_csv,
        "LPDDR4X nCCD=2": args.lpddr4x_nccd2_csv,
        "LPDDR4X nCCD=6": args.lpddr4x_nccd6_csv,
    }
    missing = [display_path(path) for path in case_csvs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing input CSV(s):\n  - " + "\n  - ".join(missing))

    rows: list[dict[str, object]] = []
    for case, csv_path in case_csvs.items():
        rows.extend(
            evaluate_case(
                case=case,
                source_csv=csv_path,
                contexts=contexts,
                device_capacity_bytes=device_capacity_bytes,
                reserve_bytes=reserve_bytes,
                shard_kv_cache_across_tp=not args.master_attention,
            )
        )
    context_order = {label: index for index, (label, _) in enumerate(contexts)}
    case_order = {case: index for index, case in enumerate(case_csvs)}
    rows.sort(key=lambda row: (case_order[str(row["Case"])], context_order[str(row["Context"])]))
    write_csv(rows, args.source_csv)
    plot(rows, contexts, list(case_csvs), args.output)

    for row in rows:
        print(
            "[selected]"
            f" case={row['Case']} context={row['Context']}"
            f" pp={row['Pipeline parallelism']} tp={row['Tensor parallelism']}"
            f" max_mb={row['Max resident microbatch']}"
            f" throughput={float(row['Capacity-constrained throughput (tokens/s)']):.2f} tok/s"
        )
    print(f"[data] {display_path(args.source_csv)}")
    print(f"[plot] {display_path(args.output)}")
    print(f"[plot] {display_path(args.output.with_suffix('.pdf'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
