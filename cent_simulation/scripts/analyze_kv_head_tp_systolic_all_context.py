#!/usr/bin/env python3
"""Combine reused Vector results with the SA4 standard-TP physical mapping.

The Vector architecture is read only from balanced_equal_power_all_contexts.
The Systolic architecture is reconstructed from the new 4x16 KV-head-TP
Ramulator sources.  PP/TP/batch is first fixed by throughput/device inside each
(architecture, memory, model, context) group, then DP alone scales that base
layout to the nearest-integer DGX H100 device-power target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import pandas as pd


CENT_SIM = Path(__file__).resolve().parents[1]
ROOT = CENT_SIM.parent
if str(CENT_SIM) not in sys.path:
    sys.path.insert(0, str(CENT_SIM))

from scripts import analyze_balanced_equal_power as balanced


DEFAULT_ROOT = CENT_SIM / "output/kv_head_tp_systolic_all_context"
DEFAULT_VECTOR_ANALYSIS = CENT_SIM / "output/balanced_equal_power_all_contexts/analysis"
FORBIDDEN_VECTOR_SOURCE = "kv_head_tp_equal_power_all_contexts"
ARCHITECTURES = ("Vector", "Systolic 4x16")
MEMORIES = tuple(balanced.MEMORY_CASES)
SYSTOLIC_BATCH_SIZES = (1, 2, 3, 4)
ARCH_HATCH = {"Vector": "", "Systolic 4x16": "///"}
MEMORY_COLORS = {
    "GDDR6": balanced.SEABORN_COLORBLIND[0],
    "LPDDR4X_nCCD2": balanced.SEABORN_COLORBLIND[1],
    "LPDDR4X_nCCD6": balanced.SEABORN_COLORBLIND[2],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systolic-raw-root",
        type=Path,
        default=DEFAULT_ROOT / "raw/systolic_4x16",
    )
    parser.add_argument(
        "--vector-analysis",
        type=Path,
        default=DEFAULT_VECTOR_ANALYSIS,
        help="Read-only balanced Vector analysis directory.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "analysis")
    parser.add_argument(
        "--h100-profile", type=Path, default=ROOT / "DGX_H100_profile_results.csv"
    )
    parser.add_argument("--device-capacity-gib", type=float, default=16.0)
    parser.add_argument("--reserve-gib", type=float, default=0.0)
    parser.add_argument(
        "--dram-energy-model",
        choices=balanced.cent.DRAM_ENERGY_MODELS,
        default="legacy",
    )
    parser.add_argument(
        "--flash-attention",
        action="store_true",
        help="Require cent_dev-style block FlashAttention source rows.",
    )
    parser.add_argument(
        "--flash-attention-block-size",
        type=int,
        default=1024,
        help="Required FlashAttention block size when --flash-attention is set.",
    )
    parser.add_argument(
        "--pipelined-softmax",
        action="store_true",
        help="Require Systolic source rows using cent_dev-style QK/Softmax overlap.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_forbidden_vector_source(value: object) -> None:
    if FORBIDDEN_VECTOR_SOURCE in str(value):
        raise ValueError(
            "Vector input must come from balanced_equal_power_all_contexts; "
            f"forbidden source found: {value}"
        )


def load_vector_candidates(vector_analysis: Path) -> tuple[pd.DataFrame, Path]:
    source = (vector_analysis / "all_candidates.csv").resolve()
    reject_forbidden_vector_source(source)
    if not source.exists():
        raise FileNotFoundError(f"missing reused Vector candidate table: {source}")
    rows = pd.read_csv(source)
    required = {
        "Memory",
        "Model",
        "Context",
        "PP",
        "TP",
        "Can host one request",
        "Replica power (W)",
        "Replica throughput (tokens/s)",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Vector table is missing columns: {', '.join(missing)}")
    for value in rows.get("Source CSV", pd.Series(dtype=str)).dropna():
        reject_forbidden_vector_source(value)
    rows = rows[
        rows["Memory"].isin(MEMORIES)
        & rows["Model"].isin(balanced.MODEL_CONFIG)
        & rows["Context"].isin(balanced.CONTEXTS)
    ].copy()
    rows["Architecture"] = "Vector"
    rows["Physical mapping"] = "balanced_vector_reused"
    rows["Batch size"] = 1
    rows["Max resident requests"] = rows["Max resident microbatch"]
    rows["Max resident batch groups"] = rows["Max resident microbatch"]
    rows["Batch groups used"] = rows["Microbatch used"]
    rows["Resident requests used"] = rows["Microbatch used"]
    rows["Can host one batch"] = rows["Can host one request"]
    rows["Projection SA row utilization"] = math.nan
    return rows, source


def build_systolic_batch_candidate(
    memory: str,
    dram_impl: str,
    source_path: Path,
    source: pd.Series,
    model: str,
    context: str,
    pp: int,
    tp: int,
    batch_size: int,
    device_capacity_gib: float,
    reserve_gib: float,
) -> dict[str, object]:
    """Apply batch-group admission to the existing systolic KV-head TP row."""

    model_config = balanced.MODEL_CONFIG[model]
    context_config = balanced.CONTEXTS[context]
    layers = balanced.MODEL_SPECS[model]["layers"]
    if layers % pp:
        raise ValueError(f"PP={pp} must exactly divide {layers} transformer blocks")
    capacity = balanced.capacity_for_candidate(
        model,
        int(context_config["window"]),
        pp,
        tp,
        device_capacity_gib,
        reserve_gib,
    )
    resident_requests = int(capacity["Max resident microbatch"])
    resident_groups = resident_requests // batch_size
    used_groups = min(resident_groups, pp)
    fill_ratio = used_groups / pp

    main_pim_ms = float(source["Main PIM latency"])
    helper_pim_ms = float(source["Helper PIM latency"]) if tp > 1 else 0.0
    source_cxl_ms = float(source["CXL latency"])
    has_split_cxl = {
        "TP collective CXL latency",
        "PP handoff CXL latency",
    }.issubset(source.index)
    if has_split_cxl:
        tp_collective_cxl_ms = float(source["TP collective CXL latency"])
        source_pp_handoff_cxl_ms = float(source["PP handoff CXL latency"])
        cxl_ms = tp_collective_cxl_ms + (
            source_pp_handoff_cxl_ms if pp > 1 else 0.0
        )
    else:
        tp_collective_cxl_ms = source_cxl_ms
        source_pp_handoff_cxl_ms = 0.0
        cxl_ms = source_cxl_ms

    main_acc_ms = float(source["Main Acc latency"])
    helper_acc_ms = float(source["Helper Acc latency"])
    acc_ms = float(source["Acc latency"])
    critical_local_ms = max(
        main_pim_ms + main_acc_ms,
        helper_pim_ms + helper_acc_ms,
    )
    source_block_ms = critical_local_ms + source_cxl_ms
    if not math.isclose(
        source_block_ms,
        float(source["TransformerBlock latency"]),
        rel_tol=1.0e-9,
        abs_tol=1.0e-9,
    ):
        raise ValueError(
            f"{memory}/{model}/{context}/TP={tp}/B={batch_size} block latency mismatch"
        )
    source_serial_ms = layers * source_block_ms
    if not math.isclose(
        source_serial_ms,
        float(source["Token latency (ms)"]),
        rel_tol=1.0e-9,
        abs_tol=1.0e-8,
    ):
        raise ValueError(
            f"{memory}/{model}/{context}/TP={tp}/B={batch_size} serial latency mismatch"
        )
    block_ms = critical_local_ms + cxl_ms
    serial_ms = layers * block_ms
    ideal_throughput = 1000.0 * pp * batch_size / serial_ms
    replica_throughput = ideal_throughput * fill_ratio
    replica_devices = pp * tp
    throughput_per_device = replica_throughput / replica_devices

    pp_handoff_removal = 0.0
    if has_split_cxl and pp == 1:
        payload_column = "PP handoff CXL payload (bits/block)"
        if payload_column not in source.index:
            raise ValueError("split CXL source is missing PP handoff payload")
        pp_handoff_group_energy = (
            layers
            * float(source[payload_column])
            * float(balanced.cent.CELLAR_POWER_CALCULATOR.PCIE_ENERGY)
            / 1.0e9
        )
        pp_handoff_removal = pp_handoff_group_energy / batch_size
    base_active_energy = float(source["Token energy (mJ)"]) - pp_handoff_removal
    waiting_floor = balanced.waiting_floor_power_w(dram_impl)
    main_gap_group_energy = layers * max(0.0, block_ms - main_pim_ms) * waiting_floor
    helper_gap_group_energy = (
        layers
        * (tp - 1)
        * max(0.0, block_ms - helper_pim_ms)
        * waiting_floor
    )
    main_gap_energy = main_gap_group_energy / batch_size
    helper_gap_energy = helper_gap_group_energy / batch_size
    gap_energy = main_gap_energy + helper_gap_energy
    work_token_energy = base_active_energy + gap_energy
    work_power = work_token_energy * replica_throughput / 1000.0
    idle_floor_power = replica_devices * (1.0 - fill_ratio) * waiting_floor
    replica_power = work_power + idle_floor_power
    effective_token_energy = (
        replica_power / replica_throughput * 1000.0
        if replica_throughput > 0.0
        else math.nan
    )

    return {
        "Memory": memory,
        "Model": model,
        "Context": context,
        "Context window": int(context_config["window"]),
        "Active sequence length": int(context_config["active"]),
        "Precision": "BF16",
        "Attention mapping": "kv_head",
        "KV mapping": "kv_head_sharded",
        "Stage balance": f"{layers}_layers_exact_divisor",
        "PP": pp,
        "TP": tp,
        "Batch size": batch_size,
        "Layers per stage": layers // pp,
        "Replica devices": replica_devices,
        "Max resident microbatch": resident_requests,
        "Max resident requests": resident_requests,
        "Max resident batch groups": resident_groups,
        "Microbatch used": used_groups,
        "Batch groups used": used_groups,
        "Resident requests used": used_groups * batch_size,
        "Pipeline fill ratio": fill_ratio,
        "Can host one request": resident_requests >= 1,
        "Can host one batch": resident_groups >= 1,
        "Pipeline full": resident_groups >= pp,
        "Projection SA row utilization": batch_size / 4.0,
        "Device capacity (GiB)": device_capacity_gib,
        "Per-device reserve (GiB)": reserve_gib,
        "Bottleneck stage": int(capacity["Bottleneck pipeline stage"]),
        "Static / bottleneck device (GiB)": float(
            capacity["Static model footprint / bottleneck device (GiB)"]
        ),
        "KV / request / bottleneck device (GiB)": float(
            capacity["KV cache / request / bottleneck device (GiB)"]
        ),
        "KV / request / model (GiB)": float(
            capacity["KV cache / request / model (GiB)"]
        ),
        "Main PIM latency (ms/block)": main_pim_ms,
        "Helper PIM latency (ms/block)": helper_pim_ms,
        "Critical PIM latency (ms/block)": max(main_pim_ms, helper_pim_ms),
        "CXL latency (ms/block)": cxl_ms,
        "TP collective CXL latency (ms/block)": tp_collective_cxl_ms,
        "PP handoff CXL latency (ms/block)": (
            source_pp_handoff_cxl_ms if pp > 1 else 0.0
        ),
        "Accelerator latency (ms/block)": acc_ms,
        "Main accelerator latency (ms/block)": main_acc_ms,
        "Helper accelerator latency (ms/block)": helper_acc_ms,
        "Critical local latency (ms/block)": critical_local_ms,
        "Block latency (ms)": block_ms,
        "Serial token latency (ms)": serial_ms,
        "Stage interval (ms)": serial_ms / pp,
        "Ideal full-pipeline throughput (tokens/s)": ideal_throughput,
        "Replica throughput (tokens/s)": replica_throughput,
        "Throughput / device (tokens/s/device)": throughput_per_device,
        "Base active token energy (mJ)": base_active_energy,
        "Removed PP handoff energy (mJ/token)": pp_handoff_removal,
        "Waiting floor power / device (W)": waiting_floor,
        "Main gap energy (mJ/token)": main_gap_energy,
        "Helper gap energy (mJ/token)": helper_gap_energy,
        "Trace-external gap energy (mJ/token)": gap_energy,
        "Work token energy (mJ)": work_token_energy,
        "Work power (W)": work_power,
        "Pipeline idle-floor power (W)": idle_floor_power,
        "Replica power (W)": replica_power,
        "Power / device (W)": replica_power / replica_devices,
        "Effective token energy (mJ)": effective_token_energy,
        "Tokens/J": replica_throughput / replica_power if replica_power > 0.0 else 0.0,
        "Source PP": int(source["Pipeline parallelism"]),
        "Source TP": int(source["Tensor parallelism"]),
        "Source devices": int(source["Device number"]),
        "Source total PCIe lanes": int(model_config["source_pcie_lanes"]),
        "Source PCIe lanes / device": (
            int(model_config["source_pcie_lanes"])
            // int(model_config["source_devices"])
        ),
        "Source CSV": balanced.display_path(source_path),
        "Helper role assumption": "one_symmetric_systolic_rank_times_TP_minus_1",
    }


def load_systolic_candidates(
    raw_root: Path,
    device_capacity_gib: float,
    reserve_gib: float,
    dram_energy_model: str,
    flash_attention: bool,
    flash_attention_block_size: int,
    pipelined_softmax: bool,
) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    envelope = balanced.sweep_envelope(
        device_capacity_gib, reserve_gib, tp_values=(1, 2, 4, 8)
    )
    rows: list[dict[str, object]] = []
    manifest: dict[str, dict[str, str]] = {}
    for memory, config in balanced.MEMORY_CASES.items():
        source_path = (raw_root / Path(config["csv"])).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"missing Systolic source CSV: {source_path}")
        manifest[memory] = {
            "path": balanced.display_path(source_path),
            "sha256": file_sha256(source_path),
        }
        for model in balanced.MODEL_CONFIG:
            for context in balanced.CONTEXTS:
                for batch_size in SYSTOLIC_BATCH_SIZES:
                    sources = balanced.load_tp_sources(
                        source_path,
                        model,
                        context,
                        (1, 2, 4, 8),
                        dram_energy_model,
                        attention_mapping="kv_head",
                        batch_size=batch_size,
                        flash_attention=flash_attention,
                        flash_attention_block_size=flash_attention_block_size,
                        pipelined_softmax=pipelined_softmax,
                        ewmul_pnm_effective=True,
                    )
                    for tp, source in sources.items():
                        if not bool(source.get("Systolic pim", False)):
                            raise ValueError(
                                f"{memory}/{model}/{context}/TP={tp} is not systolic"
                            )
                        if int(source.get("Systolic dim", -1)) != 4:
                            raise ValueError(
                                f"{memory}/{model}/{context}/TP={tp} is not SA=4x16"
                            )
                        if int(source.get("Batch size", -1)) != batch_size:
                            raise ValueError(
                                f"{memory}/{model}/{context}/TP={tp} has wrong batch"
                            )
                        if source.get("EWMUL PNM provenance") != "native":
                            raise ValueError(
                                f"{memory}/{model}/{context}/TP={tp} uses "
                                "incompatible legacy activation provenance"
                            )
                        if not bool(source.get("EWMUL PNM effective", False)):
                            raise ValueError(
                                f"{memory}/{model}/{context}/TP={tp} does not "
                                "use effective EWMUL_PNM"
                            )
                        if "Flash attention" not in source.index:
                            raise ValueError(
                                f"{memory}/{model}/{context}/TP={tp} is missing "
                                "Flash attention provenance"
                            )
                        if bool(source["Flash attention"]) != flash_attention:
                            raise ValueError(
                                f"{memory}/{model}/{context}/TP={tp} has wrong "
                                "Flash attention setting"
                            )
                        if flash_attention and int(
                            source.get("Flash attention block size", -1)
                        ) != flash_attention_block_size:
                            raise ValueError(
                                f"{memory}/{model}/{context}/TP={tp} has wrong "
                                "Flash attention block size"
                            )
                        if "Pipelined softmax" not in source.index:
                            raise ValueError(
                                f"{memory}/{model}/{context}/TP={tp} is missing "
                                "Pipelined softmax provenance"
                            )
                        if bool(source["Pipelined softmax"]) != pipelined_softmax:
                            raise ValueError(
                                f"{memory}/{model}/{context}/TP={tp} has wrong "
                                "Pipelined softmax setting"
                            )
                    workload = envelope[
                        (envelope["Model"] == model)
                        & (envelope["Context"] == context)
                    ]
                    for point in workload.itertuples(index=False):
                        pp = int(point.PP)
                        tp = int(point.TP)
                        candidate = build_systolic_batch_candidate(
                            memory,
                            str(config["dram_impl"]),
                            source_path,
                            sources[tp],
                            model,
                            context,
                            pp,
                            tp,
                            batch_size,
                            device_capacity_gib,
                            reserve_gib,
                        )
                        candidate["Architecture"] = "Systolic 4x16"
                        candidate["Physical mapping"] = "device_channel_group_bank"
                        candidate["Flash attention"] = flash_attention
                        candidate["Flash attention block size"] = (
                            flash_attention_block_size if flash_attention else 0
                        )
                        candidate["Pipelined softmax"] = pipelined_softmax
                        rows.append(candidate)
    return pd.DataFrame(rows), manifest


def add_architecture_ranks(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = candidates.copy()
    rows["Architecture throughput/device rank"] = pd.NA
    rows["Architecture Tokens/J rank"] = pd.NA
    admitted = rows["Can host one batch"].astype(bool)
    group_cols = ["Architecture", "Memory", "Model", "Context"]
    for _, group in rows[admitted].groupby(group_cols):
        rows.loc[group.index, "Architecture throughput/device rank"] = group[
            "Throughput / device (tokens/s/device)"
        ].rank(method="dense", ascending=False)
        rows.loc[group.index, "Architecture Tokens/J rank"] = group["Tokens/J"].rank(
            method="dense", ascending=False
        )
    return rows


def build_all_equal_power_deployments(
    candidates: pd.DataFrame,
    dgx: dict[tuple[str, str], dict[str, float | int]],
) -> pd.DataFrame:
    deployments: list[dict[str, object]] = []
    admitted = candidates[candidates["Can host one batch"].astype(bool)]
    for _, candidate in admitted.iterrows():
        reference = dgx[(str(candidate["Model"]), str(candidate["Context"]))]
        target_power = float(reference["power"])
        replica_power = float(candidate["Replica power (W)"])
        local_rows: list[dict[str, object]] = []
        for rounding, dp in balanced.integer_dp_choices(target_power, replica_power):
            row = candidate.to_dict()
            system_power = replica_power * dp
            system_throughput = float(candidate["Replica throughput (tokens/s)"]) * dp
            row.update(
                {
                    "DP rounding": rounding,
                    "Continuous ideal DP": target_power / replica_power,
                    "DP": dp,
                    "Total devices": int(candidate["Replica devices"]) * dp,
                    "Resident requests in flight": (
                        int(candidate["Resident requests used"]) * dp
                    ),
                    "System throughput (tokens/s)": system_throughput,
                    "System power (W)": system_power,
                    "Power delta vs DGX (W)": system_power - target_power,
                    "Absolute power delta vs DGX (W)": abs(system_power - target_power),
                    "Power / DGX H100": system_power / target_power,
                    "Throughput / DGX H100": system_throughput
                    / float(reference["throughput"]),
                    "DGX H100 throughput (tokens/s)": float(reference["throughput"]),
                    "DGX H100 device-side power (W)": target_power,
                    "DGX H100 batch": int(reference["batch"]),
                    "DGX profile model": str(reference["profile_model"]),
                }
            )
            local_rows.append(row)
        closest = min(float(row["Absolute power delta vs DGX (W)"]) for row in local_rows)
        for row in local_rows:
            row["Candidate closest integer DP"] = math.isclose(
                float(row["Absolute power delta vs DGX (W)"]),
                closest,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            deployments.append(row)
    return pd.DataFrame(deployments)


def select_equal_power(deployments: pd.DataFrame) -> pd.DataFrame:
    """Choose the closest integer DP after PP/TP/batch has been fixed."""

    deployments = deployments.copy()
    if "Batch size" not in deployments:
        deployments["Batch size"] = 1
    winners: list[dict[str, object]] = []
    group_cols = ["Architecture", "Memory", "Model", "Context"]
    for group_key, group in deployments.groupby(group_cols):
        base_layouts = group[["PP", "TP", "Batch size"]].drop_duplicates()
        if len(base_layouts) != 1:
            raise ValueError(
                "equal-power DP scaling received multiple PP/TP/batch bases for "
                f"{group_key}: {base_layouts.to_dict(orient='records')}"
            )
        closest = group[group["Candidate closest integer DP"].astype(bool)]
        if closest.empty:
            raise ValueError(f"no closest integer DP marked for {group_key}")
        ordered = closest.sort_values(
            [
                "Absolute power delta vs DGX (W)",
                "DP",
            ],
            ascending=[True, True],
        )
        winner = ordered.iloc[0].to_dict()
        winner["Selection rule"] = (
            "fixed_best_throughput_per_device_PP_TP_batch_then_nearest_integer_DP_"
            "then_smaller_DP"
        )
        winners.append(winner)
    selected = pd.DataFrame(winners)
    expected = len(deployments[group_cols].drop_duplicates())
    if len(selected) != expected:
        raise ValueError(f"expected {expected} equal-power winners, found {len(selected)}")
    return selected


def select_base_objective(candidates: pd.DataFrame, metric: str) -> pd.DataFrame:
    candidates = candidates.copy()
    if "Can host one batch" not in candidates:
        candidates["Can host one batch"] = candidates["Can host one request"]
    if "Batch size" not in candidates:
        candidates["Batch size"] = 1
    admitted = candidates[candidates["Can host one batch"].astype(bool)]
    winners: list[dict[str, object]] = []
    for _, group in admitted.groupby(["Architecture", "Memory", "Model", "Context"]):
        best = float(group[metric].max())
        tied = group[
            group[metric].map(
                lambda value: math.isclose(float(value), best, rel_tol=1e-12, abs_tol=1e-15)
            )
        ]
        winners.append(
            tied.sort_values(
                ["Replica devices", "PP", "TP", "Batch size"],
                ascending=[True, True, True, True],
            ).iloc[0].to_dict()
        )
    return pd.DataFrame(winners)


def plot_grouped_metric(
    rows: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output: Path,
    *,
    annotation: str | None = None,
    dgx_line: bool = False,
    dgx_reference: dict[tuple[str, str], float] | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.style.use("seaborn-v0_8-whitegrid")
    series = [(arch, memory) for arch in ARCHITECTURES for memory in MEMORIES]
    width = 0.125
    offsets = [(index - (len(series) - 1) / 2) * width for index in range(len(series))]
    for model in balanced.MODEL_CONFIG:
        model_rows = rows[rows["Model"] == model]
        positions = list(range(len(balanced.CONTEXTS)))
        fig, ax = plt.subplots(figsize=(10.8, 5.2))
        reference = None
        if dgx_line:
            if dgx_reference is None:
                reference = (
                    model_rows.drop_duplicates("Context")
                    .set_index("Context")
                    .loc[list(balanced.CONTEXTS), "DGX H100 throughput (tokens/s)"]
                )
            else:
                reference = pd.Series(
                    {
                        context: dgx_reference[(model, context)]
                        for context in balanced.CONTEXTS
                    }
                )
        for offset, (architecture, memory) in zip(offsets, series):
            subset = (
                model_rows[
                    (model_rows["Architecture"] == architecture)
                    & (model_rows["Memory"] == memory)
                ]
                .set_index("Context")
                .loc[list(balanced.CONTEXTS)]
            )
            bars = ax.bar(
                [position + offset for position in positions],
                subset[metric],
                width=width,
                color=MEMORY_COLORS[memory],
                hatch=ARCH_HATCH[architecture],
                edgecolor="black" if architecture != "Vector" else "none",
                linewidth=0.55,
            )
            if annotation:
                for bar, (context, row) in zip(bars, subset.iterrows()):
                    if annotation == "equal_power":
                        label = (
                            f"{float(row['Throughput / DGX H100']):.2f}x\n"
                            f"{int(row['PP'])}/{int(row['TP'])}/"
                            f"B{int(row['Batch size'])}/{int(row['DP'])}"
                        )
                    elif annotation == "tokens_per_joule":
                        if reference is None:
                            raise ValueError("Tokens/J annotations require a DGX reference")
                        label = (
                            f"{float(row[metric]) / float(reference.loc[context]):.2f}x\n"
                            f"{int(row['PP'])}/{int(row['TP'])}/"
                            f"B{int(row['Batch size'])}"
                        )
                    else:
                        label = (
                            f"{int(row['PP'])}/{int(row['TP'])}/"
                            f"B{int(row['Batch size'])}"
                        )
                    ax.annotate(
                        label,
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=6.4,
                        rotation=90,
                        linespacing=0.9,
                    )
        if dgx_line:
            ax.plot(
                positions,
                reference,
                "k--o",
                linewidth=1.4,
                markersize=4,
                label="DGX H100",
            )
        ax.set_xticks(positions, list(balanced.CONTEXTS))
        ax.set_ylabel(ylabel)
        ax.set_title(f"{model}: {title}")
        ax.grid(axis="y", alpha=0.25)
        memory_handles = [
            Patch(facecolor=MEMORY_COLORS[memory], label=memory) for memory in MEMORIES
        ]
        architecture_handles = [
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch=ARCH_HATCH[architecture],
                label=architecture,
            )
            for architecture in ARCHITECTURES
        ]
        handles = memory_handles + architecture_handles
        if dgx_line:
            handles.append(ax.lines[-1])
        ax.legend(handles=handles, frameon=False, ncols=3, fontsize=8)
        top = max(float(model_rows[metric].max()), 1.0)
        if dgx_line:
            top = max(top, float(reference.max()))
        ax.set_ylim(0.0, top * (1.36 if annotation else 1.18))
        fig.tight_layout()
        stem = output.parent / f"{output.name}_{model.replace('Llama2-', '').lower()}"
        for suffix in ("png", "pdf"):
            fig.savefig(stem.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
        plt.close(fig)


def write_plots(
    candidates: pd.DataFrame,
    selected_equal_power: pd.DataFrame,
    dgx: dict[tuple[str, str], dict[str, float | int]],
    output_dir: Path,
) -> None:
    throughput = select_base_objective(
        candidates, "Throughput / device (tokens/s/device)"
    )
    efficiency = select_base_objective(candidates, "Tokens/J")
    dgx_tokens_per_joule = {
        workload: float(reference["throughput"]) / float(reference["power"])
        for workload, reference in dgx.items()
    }
    plot_grouped_metric(
        throughput,
        "Throughput / device (tokens/s/device)",
        "Tokens/s/device",
        "best capacity-admitted PP/TP/batch (labels: PP/TP/B)",
        output_dir / "best_throughput_per_device",
        annotation="pp_tp",
    )
    plot_grouped_metric(
        efficiency,
        "Tokens/J",
        "Tokens/J",
        "best energy efficiency (labels: token/J/DGX; PP/TP/B)",
        output_dir / "best_tokens_per_joule",
        annotation="tokens_per_joule",
        dgx_line=True,
        dgx_reference=dgx_tokens_per_joule,
    )
    plot_grouped_metric(
        throughput,
        "Power / device (W)",
        "Power/device (W)",
        "power of throughput/device-selected layouts",
        output_dir / "selected_power_per_device",
    )
    plot_grouped_metric(
        throughput,
        "Effective token energy (mJ)",
        "Effective energy/token (mJ)",
        "energy of throughput/device-selected layouts",
        output_dir / "selected_token_energy",
    )
    plot_grouped_metric(
        selected_equal_power,
        "System throughput (tokens/s)",
        "System throughput (tokens/s)",
        "DGX H100 equal-power winners (labels: throughput/DGX; PP/TP/B/DP)",
        output_dir / "equal_power_system_throughput",
        annotation="equal_power",
        dgx_line=True,
    )


def write_readme(
    output_dir: Path,
    candidates: pd.DataFrame,
    winners: pd.DataFrame,
    flash_attention: bool,
    flash_attention_block_size: int,
    pipelined_softmax: bool,
) -> None:
    summary = winners.sort_values(
        ["Model", "Context window", "Architecture", "Memory"]
    )[
        [
            "Architecture",
            "Memory",
            "Model",
            "Context",
            "PP",
            "TP",
            "Batch size",
            "DP",
            "System throughput (tokens/s)",
            "System power (W)",
            "Throughput / DGX H100",
        ]
    ]
    flash_command_options = (
        " --flash-attention"
        f" --flash-attention-block-size {flash_attention_block_size}"
        if flash_attention
        else ""
    )
    pipeline_command_option = (
        " --pipelined-softmax" if pipelined_softmax else ""
    )
    text = f"""# KV-head TP Systolic all-context campaign

This directory combines two architectures without rerunning the Vector path:

- **Vector** is read-only reuse of `balanced_equal_power_all_contexts/analysis/all_candidates.csv`.
- **Systolic 4x16** is the standard Device–Channel-group–Bank KV-head TP mapping.

Systolic PIM follows cent_dev and therefore has effective EWMUL_PNM enabled.
W1 `AF`/`RD_AF` commands remain in the PIM trace; RMSNorm, RoPE, and the fused
FFN element-wise multiplies use the analytical PNM VEC_MUL path.

Systolic source rows use FlashAttention: **{flash_attention}**. When enabled,
the context block size is **{flash_attention_block_size}** and score workspace
`W_MEM`/`R_MEM` traffic is absent from each block.

Pipelined Softmax is **{pipelined_softmax}**. When enabled, full Softmax energy
is retained, while only the cent_dev producer-startup fraction remains exposed
on the latency critical path.

Both use BF16, 16 GiB/device, the 4K/32K/128K midpoint samples, and GDDR6 plus
LPDDR4X nCCD2/nCCD6. Vector remains the reused batch-1 baseline; Systolic
4x16 evaluates batch 1/2/3/4. SA=8x16 is intentionally excluded. The Systolic
raw campaign has 288 Ramulator jobs (96 per timing) and 192 functional traces
because both LPDDR timings reuse the same trace set.

Equal power is evaluated independently for every architecture and memory.
First, throughput/device fixes one capacity-admitted PP/TP/batch layout;
equal-score ties use fewer replica devices, then smaller PP, TP, and batch.
Only that fixed layout
is DP-scaled, retaining both integer DP neighbors around the DGX power target
for audit and selecting the nearest one (smaller DP on an exact tie). Therefore
DP packing cannot reselect PP/TP/batch. Each workload has six CENT bars; DGX H100 is
a line, not a bar.

Candidates: {len(candidates)} total; {int(candidates['Can host one batch'].astype(bool).sum())} admitted.
Equal-power winners: {len(winners)} total (six per model/context).

## Equal-power winners

{balanced.markdown_table(summary, list(summary.columns))}

## Reproduction

```bash
/home/linuswang/miniforge3/envs/cent/bin/python scripts/run_cent_memory_cases.py \\
  --kv-head-tp-systolic --models Llama2-7B,Llama2-70B \\
  --cases GDDR6,LPDDR4X_nCCD2,LPDDR4X_nCCD6 \\
  --EWMUL_PNM{flash_command_options}{pipeline_command_option}

/home/linuswang/miniforge3/envs/cent/bin/python \\
  scripts/analyze_kv_head_tp_systolic_all_context.py \\
  {flash_command_options}{pipeline_command_option}
```

For an accelerator-only accounting refresh, the first command reuses valid
Ramulator traces/logs and overwrites the source CSV rows before regenerating
this analysis. Legacy `--activation` provenance is retained for audit but is
excluded from candidates. To physically replace the raw source CSVs too, delete
the three `simulation_results_decode_only_long_context_midpoint*.csv` files
under `raw/systolic_4x16/{{GDDR6,LPDDR4X}}` before the first command.
"""
    (output_dir / "README.md").write_text(text)


def main() -> int:
    args = parse_args()
    if args.device_capacity_gib <= args.reserve_gib:
        raise ValueError("device capacity must exceed reserve")
    raw_root = args.systolic_raw_root.resolve()
    vector_analysis = args.vector_analysis.resolve()
    output_dir = args.output_dir.resolve()
    reject_forbidden_vector_source(vector_analysis)
    output_dir.mkdir(parents=True, exist_ok=True)

    vector, vector_source = load_vector_candidates(vector_analysis)
    systolic, systolic_manifest = load_systolic_candidates(
        raw_root,
        args.device_capacity_gib,
        args.reserve_gib,
        args.dram_energy_model,
        args.flash_attention,
        args.flash_attention_block_size,
        args.pipelined_softmax,
    )
    candidates = pd.concat([vector, systolic], ignore_index=True, sort=False)
    candidates = add_architecture_ranks(candidates).sort_values(
        [
            "Model",
            "Context window",
            "Architecture",
            "Memory",
            "PP",
            "TP",
            "Batch size",
        ]
    )
    best_throughput = select_base_objective(
        candidates, "Throughput / device (tokens/s/device)"
    ).sort_values(["Model", "Context window", "Architecture", "Memory"])
    best_tokens_per_joule = select_base_objective(candidates, "Tokens/J").sort_values(
        ["Model", "Context window", "Architecture", "Memory"]
    )
    dgx = balanced.load_dgx(args.h100_profile.resolve())
    deployments = build_all_equal_power_deployments(best_throughput, dgx).sort_values(
        [
            "Model",
            "Context window",
            "Architecture",
            "Memory",
            "PP",
            "TP",
            "Batch size",
            "DP",
        ]
    )
    winners = select_equal_power(deployments).sort_values(
        ["Model", "Context window", "Architecture", "Memory"]
    )
    expected_winners = (
        len(ARCHITECTURES)
        * len(MEMORIES)
        * len(balanced.MODEL_CONFIG)
        * len(balanced.CONTEXTS)
    )
    if len(winners) != expected_winners:
        raise ValueError(
            f"expected {expected_winners} full-campaign winners, found {len(winners)}"
        )

    outputs = {
        "all_candidates.csv": candidates,
        "equal_power_deployments.csv": deployments,
        "equal_power_selected.csv": winners,
        "best_throughput_per_device.csv": best_throughput,
        "best_tokens_per_joule.csv": best_tokens_per_joule,
    }
    for filename, frame in outputs.items():
        path = output_dir / filename
        frame.to_csv(path, index=False)
        print(f"[data] {balanced.display_path(path)}")

    manifest = {
        "architectures": list(ARCHITECTURES),
        "memories": list(MEMORIES),
        "models": list(balanced.MODEL_CONFIG),
        "contexts": balanced.CONTEXTS,
        "systolic_array": "4x16",
        "ewmul_pnm_effective": True,
        "flash_attention": args.flash_attention,
        "flash_attention_block_size": (
            args.flash_attention_block_size if args.flash_attention else 0
        ),
        "pipelined_softmax": args.pipelined_softmax,
        "batch_sizes": list(SYSTOLIC_BATCH_SIZES),
        "tp_values": [1, 2, 4, 8],
        "ramulator_jobs": 288,
        "functional_traces": 192,
        "vector_jobs": 0,
        "vector_source": {
            "path": balanced.display_path(vector_source),
            "sha256": file_sha256(vector_source),
            "read_only_reuse": True,
        },
        "vector_source_policy": "balanced_equal_power_all_contexts_only",
        "systolic_sources": systolic_manifest,
        "selection_rule": (
            "max_throughput_per_device_then_fewer_replica_devices_smaller_PP_TP_batch_"
            "then_record_floor_and_ceil_DP_and_select_nearest_integer_DP"
        ),
    }
    manifest_text = json.dumps(manifest, indent=2) + "\n"
    reject_forbidden_vector_source(manifest["vector_source"]["path"])
    (output_dir / "manifest.json").write_text(manifest_text)
    write_readme(
        output_dir,
        candidates,
        winners,
        args.flash_attention,
        args.flash_attention_block_size,
        args.pipelined_softmax,
    )

    if not args.no_plots:
        write_plots(candidates, winners, dgx, output_dir)
        print(f"[plots] {balanced.display_path(output_dir)}")

    columns = [
        "Architecture",
        "Memory",
        "Model",
        "Context",
        "PP",
        "TP",
        "Batch size",
        "DP",
        "System throughput (tokens/s)",
        "System power (W)",
        "Throughput / DGX H100",
    ]
    print(winners[columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
