"""KV-cache admission control for CENT hybrid PP/TP decode simulations.

The simulator measures one decode token while assuming a filled pipeline.  A
long-context serving deployment must additionally keep one complete KV cache
per concurrent request on every participating device.  This module accounts
for that memory admission limit and reports the resulting pipeline fill.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


BF16_BYTES = 2
FP8_BYTES = 1

# Tensor shapes follow function_sim.py.  Llama2's input and output embedding
# matrices are accounted separately because its weights are not tied.
MODEL_SPECS: dict[str, dict[str, int]] = {
    "Llama2-7B": {
        "layers": 32,
        "dim": 4096,
        "kv_heads": 32,
        "head_dim": 128,
        "ffn_dim": 11008,
        "vocab_size": 32000,
    },
    "Llama2-70B": {
        "layers": 80,
        "dim": 8192,
        "kv_heads": 8,
        "head_dim": 128,
        "ffn_dim": 28672,
        "vocab_size": 32000,
    },
}


def gibibytes(value: float) -> float:
    return value / (1024**3)


def _layer_weight_bytes(
    spec: dict[str, int], element_bytes: int = BF16_BYTES
) -> int:
    dim = spec["dim"]
    kv_dim = spec["kv_heads"] * spec["head_dim"]
    elements = (
        dim * dim  # Wq
        + 2 * kv_dim * dim  # Wk, Wv
        + dim * dim  # Wo
        + 3 * spec["ffn_dim"] * dim  # W1, W3, W2
        + 2 * dim  # two RMSNorm vectors
    )
    return elements * element_bytes


def _endpoint_weight_bytes(
    spec: dict[str, int], element_bytes: int = BF16_BYTES
) -> int:
    return spec["vocab_size"] * spec["dim"] * element_bytes


def _kv_bytes_per_layer(
    spec: dict[str, int],
    context_window: int,
    element_bytes: int = BF16_BYTES,
) -> int:
    return 2 * context_window * spec["kv_heads"] * spec["head_dim"] * element_bytes


def capacity_for_layout(
    model: str,
    pp: int,
    tp: int,
    context_window: int,
    device_capacity_bytes: int,
    reserve_bytes: int = 0,
    shard_kv_cache_across_tp: bool = True,
    element_bytes: int = BF16_BYTES,
) -> dict[str, float | int]:
    """Return the limiting per-device capacity for a balanced PP/TP layout.

    ``shard_kv_cache_across_tp=False`` models CENT's paper mapping: FC
    weights are tensor-sharded, while each stage's master keeps the complete
    KV cache and performs attention locally.
    """

    if model not in MODEL_SPECS:
        raise ValueError(f"unknown model '{model}'")
    if pp <= 0 or tp <= 0:
        raise ValueError("PP and TP must be positive")
    if element_bytes <= 0:
        raise ValueError("element_bytes must be positive")
    spec = MODEL_SPECS[model]
    layers = spec["layers"]
    base, remainder = divmod(layers, pp)
    layers_per_stage = [base + (stage < remainder) for stage in range(pp)]
    per_layer_kv = _kv_bytes_per_layer(spec, context_window, element_bytes)
    layer_weights = _layer_weight_bytes(spec, element_bytes)
    endpoint_weights = _endpoint_weight_bytes(spec, element_bytes)

    capacities: list[tuple[int, int, int, int]] = []
    for stage, layer_count in enumerate(layers_per_stage):
        static_weights = layer_count * layer_weights
        if stage == 0:
            static_weights += endpoint_weights
        if stage == pp - 1:
            static_weights += endpoint_weights
        static_per_device = math.ceil(static_weights / tp) + reserve_bytes
        kv_bytes_per_stage_request = layer_count * per_layer_kv
        kv_per_request_per_device = (
            math.ceil(kv_bytes_per_stage_request / tp)
            if shard_kv_cache_across_tp
            else kv_bytes_per_stage_request
        )
        available = device_capacity_bytes - static_per_device
        max_microbatch = 0 if available < 0 else available // kv_per_request_per_device
        capacities.append((max_microbatch, stage, static_per_device, kv_per_request_per_device))

    max_microbatch, bottleneck_stage, static_per_device, kv_per_request = min(capacities)
    return {
        "Max resident microbatch": int(max_microbatch),
        "Bottleneck pipeline stage": int(bottleneck_stage),
        "Layers at bottleneck stage": int(layers_per_stage[bottleneck_stage]),
        "Static model footprint / bottleneck device (GiB)": gibibytes(static_per_device),
        "KV cache / request / bottleneck device (GiB)": gibibytes(kv_per_request),
        "KV cache / request / model (GiB)": gibibytes(layers * per_layer_kv),
        "KV cache mapping": "TP-sharded" if shard_kv_cache_across_tp else "master-local",
    }


def context_window_from_row(row: pd.Series | dict[str, Any]) -> int:
    value = row.get("Context window")
    if value is not None and not pd.isna(value):
        return int(value)
    return int(row["Sequence length"])


def _with_capacity(
    row: pd.Series,
    model: str,
    context_window: int,
    device_capacity_bytes: int,
    reserve_bytes: int,
    shard_kv_cache_across_tp: bool,
) -> dict[str, Any]:
    pp = int(row["Pipeline parallelism"])
    tp = int(row["Tensor parallelism"])
    capacity = capacity_for_layout(
        model,
        pp,
        tp,
        context_window,
        device_capacity_bytes,
        reserve_bytes,
        shard_kv_cache_across_tp,
    )
    max_microbatch = int(capacity["Max resident microbatch"])
    used_microbatch = min(max_microbatch, pp)
    fill_ratio = used_microbatch / pp
    raw_throughput = float(row["Throughput (tokens/s)"])
    constrained_throughput = raw_throughput * fill_ratio

    evaluated = row.to_dict()
    evaluated.update(capacity)
    evaluated.update(
        {
            "Trace throughput (tokens/s)": raw_throughput,
            "Microbatch used for throughput": used_microbatch,
            "Pipeline fill ratio": fill_ratio,
            "Capacity-constrained throughput (tokens/s)": constrained_throughput,
            "Capacity-constrained system power (W)": float(row["Token energy (mJ)"])
            * constrained_throughput
            / 1000.0,
            "Device capacity (GiB)": gibibytes(device_capacity_bytes),
            "Per-device reserve (GiB)": gibibytes(reserve_bytes),
        }
    )
    return evaluated


def select_capacity_constrained_best(
    df: pd.DataFrame,
    model: str,
    context_window: int,
    *,
    device_capacity_gib: float = 16.0,
    per_device_reserve_gib: float = 0.0,
    hybrid_only: bool = True,
    shard_kv_cache_across_tp: bool = True,
) -> dict[str, Any]:
    """Select the highest-throughput layout that can sustain its pipeline.

    Only layouts where PP * TP equals the simulated device count are admitted
    by default.  This removes simulator-only PP rows that represent more
    pipeline stages than physically deployed PIM devices.
    """

    if device_capacity_gib <= per_device_reserve_gib:
        raise ValueError("device capacity must exceed its reserve")
    candidates = df[(df["Model"] == model) & (df.apply(context_window_from_row, axis=1) == context_window)].copy()
    for column in (
        "Pipeline parallelism",
        "Tensor parallelism",
        "Device number",
        "Throughput (tokens/s)",
        "Token energy (mJ)",
    ):
        candidates[column] = pd.to_numeric(candidates[column])
    if hybrid_only:
        candidates = candidates[
            candidates["Pipeline parallelism"] * candidates["Tensor parallelism"] == candidates["Device number"]
        ]
    if candidates.empty:
        scope = "hybrid " if hybrid_only else ""
        raise ValueError(f"no {scope}rows for {model} context={context_window}")

    device_capacity_bytes = int(device_capacity_gib * 1024**3)
    reserve_bytes = int(per_device_reserve_gib * 1024**3)
    evaluated = [
        _with_capacity(
            row,
            model,
            context_window,
            device_capacity_bytes,
            reserve_bytes,
            shard_kv_cache_across_tp,
        )
        for _, row in candidates.iterrows()
    ]
    return max(
        evaluated,
        key=lambda row: (
            float(row["Capacity-constrained throughput (tokens/s)"]),
            -float(row["Token energy (mJ)"]),
        ),
    )
