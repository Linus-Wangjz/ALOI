#!/usr/bin/env python3
"""Sweep CENT PP/TP/DP layouts under memory and DGX-H100 power limits.

The Ramulator CSVs contain one full-model serial decode latency and one
full-model token energy for every simulated TP degree.  PP only changes how
the model layers are split into pipeline stages, so this script reuses the
per-TP trace and sweeps arbitrary physical PP degrees.  DP is the number of
independent replicas:

    total_cards = PP * TP * DP

Three deployment modes are emitted:

* fixed_cards_exact: use exactly the requested GDDR6/LPDDR4X card count.
* card_cap_and_power: stay within both that card count and the DGX power.
* equal_power_unbounded: enforce only DGX power (useful, but optimistic when
  idle-card power is left at its default of zero).

By default, layer imbalance is modeled explicitly.  For example, 80 layers
on PP=32 have an effective saturated pipeline parallelism of 80/3, because
the bottleneck stages contain three layers.  ``--stage-balance ideal``
reproduces the simulator's ideal PP multiplier instead.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from cent_capacity import (
    MODEL_SPECS,
    _endpoint_weight_bytes,
    _kv_bytes_per_layer,
    _layer_weight_bytes,
)


ROOT = Path(__file__).resolve().parents[2]
CENT_SIM = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = CENT_SIM / "output" / "parallelism_sweep"

CONTEXTS = {
    "4K": 4096,
    "32K": 32768,
    "128K": 131072,
}

MODEL_METADATA = {
    "Llama2-7B": {
        "gpu_model": "Llama-3.1-8B",
        "dgx_power_w": 8 * 581.5,
    },
    "Llama2-70B": {
        "gpu_model": "Llama-3.1-70B",
        "dgx_power_w": 8 * 556.3,
    },
}

MEMORY_CASES = {
    "GDDR6": {
        "fixed_cards": 128,
        "csv": CENT_SIM
        / "output/master_attention/GDDR6/simulation_results_decode_only_long_context_midpoint.csv",
    },
    "LPDDR4X_nCCD2": {
        "fixed_cards": 256,
        "csv": CENT_SIM
        / "output/master_attention/LPDDR4X/simulation_results_decode_only_long_context_midpoint_nCCD2.csv",
    },
}

KV_MAPPINGS = (
    "k_sharded_v_local",
    "fully_tp_sharded",
    "master_local",
)


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gddr6-csv", type=Path)
    parser.add_argument("--lpddr4x-nccd2-csv", type=Path)
    parser.add_argument("--h100-profile", type=Path, default=ROOT / "DGX_H100_profile_results.csv")
    parser.add_argument("--gddr6-cards", type=int, default=128)
    parser.add_argument("--lpddr4x-cards", type=int, default=256)
    parser.add_argument("--device-capacity-gib", type=float, default=16.0)
    parser.add_argument("--reserve-gib", type=float, default=0.0)
    parser.add_argument(
        "--max-pp",
        type=int,
        default=32,
        help="Maximum pipeline stages. CENT's existing topology uses at most 32.",
    )
    parser.add_argument(
        "--stage-balance",
        choices=("discrete", "ideal"),
        default="discrete",
        help="Use discrete layers/stage or the simulator's ideal PP multiplier.",
    )
    parser.add_argument(
        "--kv-mappings",
        nargs="+",
        choices=KV_MAPPINGS,
        default=list(KV_MAPPINGS),
    )
    parser.add_argument(
        "--gddr6-idle-card-power-w",
        type=float,
        default=0.0,
        help="Optional wall-power floor for each idle GDDR6 card.",
    )
    parser.add_argument(
        "--lpddr4x-idle-card-power-w",
        type=float,
        default=0.0,
        help="Optional wall-power floor for each idle LPDDR4X card.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_dgx_best(profile_path: Path) -> dict[tuple[str, int], dict[str, float | int]]:
    df = pd.read_csv(profile_path)
    result: dict[tuple[str, int], dict[str, float | int]] = {}
    for model, metadata in MODEL_METADATA.items():
        for context_window in CONTEXTS.values():
            subset = df[
                (df["model"] == metadata["gpu_model"])
                & (df["num_gpu"] == 8)
                & (df["stage"] == "Decoding")
                & (df["seqlen"] == context_window)
            ]
            if subset.empty:
                raise ValueError(f"missing DGX profile for {model} context={context_window}")
            best = subset.loc[subset["throughput (tokens/s)"].idxmax()]
            result[(model, context_window)] = {
                "throughput": float(best["throughput (tokens/s)"]),
                "batch": int(best["batch"]),
                "power": float(metadata["dgx_power_w"]),
            }
    return result


def load_tp_sources(path: Path, model: str, context_window: int) -> dict[int, pd.Series]:
    df = pd.read_csv(path)
    subset = df[(df["Model"] == model) & (df["Context window"] == context_window)].copy()
    # Hybrid model-parallel rows have PP*TP equal to the simulated device
    # count.  This cleanly excludes the separate pipeline-only trace row.
    subset = subset[
        subset["Pipeline parallelism"] * subset["Tensor parallelism"]
        == subset["Device number"]
    ]
    if subset.empty:
        raise ValueError(f"no model-parallel source rows in {path} for {model} context={context_window}")
    sources: dict[int, pd.Series] = {}
    for _, row in subset.iterrows():
        tp = int(row["Tensor parallelism"])
        if tp in sources:
            raise ValueError(f"duplicate TP={tp} source rows in {path} for {model} context={context_window}")
        sources[tp] = row
    return sources


def kv_bytes_on_bottleneck_device(stage_kv_bytes: int, tp: int, mapping: str) -> int:
    if mapping == "fully_tp_sharded":
        return math.ceil(stage_kv_bytes / tp)
    if mapping == "master_local":
        return stage_kv_bytes
    if mapping == "k_sharded_v_local":
        # K and V contain the same number of elements.  The current no-inter-
        # device-attention trace stripes K over TP devices but leaves V on the
        # attention/master device.
        half = stage_kv_bytes // 2
        return math.ceil(half / tp) + (stage_kv_bytes - half)
    raise ValueError(f"unknown KV mapping: {mapping}")


def capacity_for_candidate(
    model: str,
    pp: int,
    tp: int,
    context_window: int,
    mapping: str,
    device_capacity_bytes: int,
    reserve_bytes: int,
) -> dict[str, float | int]:
    spec = MODEL_SPECS[model]
    base, remainder = divmod(spec["layers"], pp)
    per_layer_weights = _layer_weight_bytes(spec)
    endpoint_weights = _endpoint_weight_bytes(spec)
    per_layer_kv = _kv_bytes_per_layer(spec, context_window)
    stage_results: list[tuple[int, int, int, int, int]] = []
    for stage in range(pp):
        layer_count = base + int(stage < remainder)
        static = layer_count * per_layer_weights
        if stage == 0:
            static += endpoint_weights
        if stage == pp - 1:
            static += endpoint_weights
        static_per_device = math.ceil(static / tp) + reserve_bytes
        kv_per_request = kv_bytes_on_bottleneck_device(layer_count * per_layer_kv, tp, mapping)
        available = device_capacity_bytes - static_per_device
        max_resident = 0 if available < 0 else available // kv_per_request
        stage_results.append((max_resident, stage, layer_count, static_per_device, kv_per_request))
    max_resident, stage, layer_count, static_per_device, kv_per_request = min(stage_results)
    return {
        "Max resident microbatch": int(max_resident),
        "Bottleneck stage": stage,
        "Bottleneck stage layers": layer_count,
        "Static / bottleneck device (GiB)": static_per_device / 2**30,
        "KV / request / bottleneck device (GiB)": kv_per_request / 2**30,
    }


def effective_pipeline_parallelism(layers: int, pp: int, stage_balance: str) -> float:
    if stage_balance == "ideal":
        return float(pp)
    return layers / math.ceil(layers / pp)


def base_candidates(
    memory: str,
    config: dict[str, object],
    model: str,
    context_label: str,
    context_window: int,
    mapping: str,
    max_pp: int,
    stage_balance: str,
    device_capacity_bytes: int,
    reserve_bytes: int,
    idle_card_power_w: float,
) -> list[dict[str, object]]:
    sources = load_tp_sources(Path(config["csv"]), model, context_window)
    layers = MODEL_SPECS[model]["layers"]
    rows: list[dict[str, object]] = []
    for tp, source in sources.items():
        serial_latency_ms = float(source["Token latency (ms)"])
        serial_token_rate = 1000.0 / serial_latency_ms
        token_energy_mj = float(source["Token energy (mJ)"])
        for pp in range(1, min(max_pp, layers) + 1):
            capacity = capacity_for_candidate(
                model,
                pp,
                tp,
                context_window,
                mapping,
                device_capacity_bytes,
                reserve_bytes,
            )
            max_resident = int(capacity["Max resident microbatch"])
            effective_pp = effective_pipeline_parallelism(layers, pp, stage_balance)
            active_concurrency = min(float(max_resident), effective_pp)
            if active_concurrency <= 0:
                continue
            replica_throughput = serial_token_rate * active_concurrency
            work_power = token_energy_mj * replica_throughput / 1000.0
            physical_stage_duty = active_concurrency / pp
            replica_cards = pp * tp
            idle_power = replica_cards * (1.0 - physical_stage_duty) * idle_card_power_w
            replica_power = work_power + idle_power
            rows.append(
                {
                    "Memory": memory,
                    "Model": model,
                    "Context": context_label,
                    "Context window": context_window,
                    "KV mapping": mapping,
                    "Stage balance": stage_balance,
                    "PP": pp,
                    "TP": tp,
                    "Replica cards": replica_cards,
                    "Effective PP": effective_pp,
                    "Max resident microbatch": max_resident,
                    "Active concurrency": active_concurrency,
                    "Physical stage duty": physical_stage_duty,
                    "Replica throughput (tokens/s)": replica_throughput,
                    "Replica work power (W)": work_power,
                    "Replica idle-floor power (W)": idle_power,
                    "Replica power (W)": replica_power,
                    "Token energy (mJ)": token_energy_mj,
                    "Serial token latency (ms)": serial_latency_ms,
                    "Source TP": tp,
                    "Source base devices": int(source["Device number"]),
                    "Source active sequence length": int(source["Sequence length"]),
                    "Source CSV": display_path(Path(config["csv"])),
                    "Mapping validity": (
                        "matches current no-inter-device trace mapping"
                        if mapping == "k_sharded_v_local"
                        else "capacity sensitivity only; latency/energy still come from the current master trace"
                    ),
                    "TP trace caveat": (
                        "complete TP1 trace"
                        if tp == 1
                        else "optimistic: helper K/QK energy and score-gather latency are not modeled"
                    ),
                    **capacity,
                }
            )
    return rows


def deployment_row(
    candidate: dict[str, object],
    mode: str,
    dp: int,
    dgx: dict[str, float | int],
    card_limit: int | None,
) -> dict[str, object]:
    row = dict(candidate)
    total_cards = int(candidate["Replica cards"]) * dp
    throughput = float(candidate["Replica throughput (tokens/s)"]) * dp
    power = float(candidate["Replica power (W)"]) * dp
    row.update(
        {
            "Deployment mode": mode,
            "DP": dp,
            "Total cards": total_cards,
            "Card limit": "" if card_limit is None else card_limit,
            "System throughput (tokens/s)": throughput,
            "System power (W)": power,
            "Power / DGX H100": power / float(dgx["power"]),
            "Throughput / DGX H100": throughput / float(dgx["throughput"]),
            "DGX H100 throughput (tokens/s)": float(dgx["throughput"]),
            "DGX H100 power (W)": float(dgx["power"]),
            "DGX H100 batch": int(dgx["batch"]),
        }
    )
    return row


def choose_best(rows: list[dict[str, object]]) -> dict[str, object]:
    return max(
        rows,
        key=lambda row: (
            float(row["System throughput (tokens/s)"]),
            -float(row["System power (W)"]),
            -int(row["Total cards"]),
            -int(row["TP"]),
        ),
    )


def main() -> int:
    args = parse_args()
    cases = {name: dict(config) for name, config in MEMORY_CASES.items()}
    if args.gddr6_csv:
        cases["GDDR6"]["csv"] = args.gddr6_csv
    if args.lpddr4x_nccd2_csv:
        cases["LPDDR4X_nCCD2"]["csv"] = args.lpddr4x_nccd2_csv
    cases["GDDR6"]["fixed_cards"] = args.gddr6_cards
    cases["LPDDR4X_nCCD2"]["fixed_cards"] = args.lpddr4x_cards
    idle_power = {
        "GDDR6": args.gddr6_idle_card_power_w,
        "LPDDR4X_nCCD2": args.lpddr4x_idle_card_power_w,
    }
    dgx_rows = load_dgx_best(args.h100_profile)
    device_capacity_bytes = int(args.device_capacity_gib * 2**30)
    reserve_bytes = int(args.reserve_gib * 2**30)

    all_deployments: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []
    for memory, config in cases.items():
        card_limit = int(config["fixed_cards"])
        for model in MODEL_METADATA:
            for context_label, context_window in CONTEXTS.items():
                dgx = dgx_rows[(model, context_window)]
                for mapping in args.kv_mappings:
                    candidates = base_candidates(
                        memory,
                        config,
                        model,
                        context_label,
                        context_window,
                        mapping,
                        args.max_pp,
                        args.stage_balance,
                        device_capacity_bytes,
                        reserve_bytes,
                        idle_power[memory],
                    )
                    mode_rows: dict[str, list[dict[str, object]]] = {
                        "fixed_cards_exact": [],
                        "card_cap_and_power": [],
                        "equal_power_unbounded": [],
                    }
                    for candidate in candidates:
                        replica_cards = int(candidate["Replica cards"])
                        replica_power = float(candidate["Replica power (W)"])
                        if card_limit % replica_cards == 0:
                            dp = card_limit // replica_cards
                            if dp >= 1:
                                mode_rows["fixed_cards_exact"].append(
                                    deployment_row(candidate, "fixed_cards_exact", dp, dgx, card_limit)
                                )
                        power_dp = int(float(dgx["power"]) // replica_power)
                        cap_dp = card_limit // replica_cards
                        dp = min(power_dp, cap_dp)
                        if dp >= 1:
                            mode_rows["card_cap_and_power"].append(
                                deployment_row(candidate, "card_cap_and_power", dp, dgx, card_limit)
                            )
                        if power_dp >= 1:
                            mode_rows["equal_power_unbounded"].append(
                                deployment_row(candidate, "equal_power_unbounded", power_dp, dgx, None)
                            )
                    for mode, rows in mode_rows.items():
                        if not rows:
                            continue
                        all_deployments.extend(rows)
                        best_rows.append(choose_best(rows))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_path = output_dir / "parallelism_sweep_all_candidates.csv"
    best_path = output_dir / "parallelism_sweep_best.csv"
    pd.DataFrame(all_deployments).to_csv(all_path, index=False)
    best = pd.DataFrame(best_rows).sort_values(
        ["KV mapping", "Deployment mode", "Memory", "Model", "Context window"]
    )
    best.to_csv(best_path, index=False)

    display_columns = [
        "KV mapping",
        "Deployment mode",
        "Memory",
        "Model",
        "Context",
        "PP",
        "TP",
        "DP",
        "Total cards",
        "Max resident microbatch",
        "System throughput (tokens/s)",
        "System power (W)",
        "Power / DGX H100",
    ]
    print(best[display_columns].to_string(index=False))
    print(f"[data] {display_path(all_path)}")
    print(f"[data] {display_path(best_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
