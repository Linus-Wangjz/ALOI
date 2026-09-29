#!/usr/bin/env python3
"""Analyze only the SA4 Device–Channel-group–Bank KV-head-TP mapping.

This script constructs Systolic 4x16 BF16 or 4x32 FP8 candidates from its Ramulator campaign,
selects PP/TP/batch within that architecture, and subsequently applies only
integer DP scaling to reach the DGX H100 device-power reference.  It never
loads, selects, or plots Vector candidates; use
``analyze_kv_head_tp_systolic_vs_vector.py`` for the read-only comparison.
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
from scripts.utility.campaign_selection import (
    build_all_equal_power_deployments,
    select_base_objective,
    select_equal_power,
)
from scripts.utility.campaign_plots import plot_grouped_metric
from scripts.utility.system_energy_breakdown import (
    build_selected_component_breakdown,
    plot_selected_component_breakdown,
)
from scripts.utility.systolic_cycle_breakdown import (
    build_selected_systolic_cycle_breakdown,
    plot_selected_systolic_cycle_breakdown,
    write_cycle_breakdown_csv,
)


DEFAULT_ROOT = CENT_SIM / "output/kv_head_tp_systolic_all_context"
ARCHITECTURES = ("Systolic 4x16",)
MEMORY_CASES = dict(balanced.MEMORY_CASES)
MEMORIES = tuple(MEMORY_CASES)
MODELS = dict(balanced.MODEL_CONFIG)
CONTEXTS = dict(balanced.CONTEXTS)
CONTEXT_CATALOG = {
    **balanced.CONTEXTS,
    "8K": {"window": 8192, "active": 6400},
    "16K": {"window": 16384, "active": 14592},
    "64K": {"window": 65536, "active": 63744},
}
SYSTOLIC_BATCH_SIZES = (1, 2, 3, 4)
ARCH_HATCH = {"Systolic 4x16": "///"}
MEMORY_COLORS = {
    # Vega/Altair Category10's first three colors. These apply only to the
    # scalar memory bars; system-energy components retain their own palette.
    "GDDR6": "#4C78A8",
    "LPDDR4X_nCCD2": "#F58518",
    "LPDDR4X_nCCD6": "#E45756",
}

# Micron uses the same LPDDR4X timing/logs as the standard FP8 campaign.  Its
# source CSVs differ only because energy is refreshed with the Micron IDD table.
FP8_MICRON_MEMORY_CASES = {
    "LPDDR4X_nCCD2": {
        "csv": Path("LPDDR4X/simulation_results_decode_only_long_context_midpoint_nCCD2_micron.csv"),
        "dram_impl": "LPDDR4X_MICRON",
    },
    "LPDDR4X_nCCD6": {
        "csv": Path("LPDDR4X/simulation_results_decode_only_long_context_midpoint_nCCD6_micron.csv"),
        "dram_impl": "LPDDR4X_MICRON",
    },
}


def configure_precision(precision: str, dram_vendor: str) -> int:
    """Select architecture labels and the supported memory cases."""

    global ARCHITECTURES, MEMORY_CASES, MEMORIES, ARCH_HATCH
    if precision == "FP8":
        width = 32
        MEMORY_CASES = (
            dict(FP8_MICRON_MEMORY_CASES)
            if dram_vendor == "micron"
            else dict(balanced.MEMORY_CASES)
        )
        MEMORIES = ("LPDDR4X_nCCD2", "LPDDR4X_nCCD6")
    else:
        width = 16
        if dram_vendor != "winbond":
            raise ValueError("--dram-vendor micron is currently supported for FP8 only")
        MEMORY_CASES = dict(balanced.MEMORY_CASES)
        MEMORIES = tuple(MEMORY_CASES)
    architecture = f"Systolic 4x{width}"
    ARCHITECTURES = (architecture,)
    ARCH_HATCH = {architecture: "///"}
    return width


def configure_scope(
    models: list[str], contexts: list[str], memories: list[str] | None
) -> None:
    """Limit the standard campaign analysis to a complete source subset."""

    global MODELS, CONTEXTS, MEMORIES
    MODELS = {model: balanced.MODEL_CONFIG[model] for model in models}
    CONTEXTS = {context: CONTEXT_CATALOG[context] for context in contexts}
    if memories is not None:
        unsupported = sorted(set(memories) - set(MEMORIES))
        if unsupported:
            raise ValueError(
                "memory case(s) are unsupported for the selected precision: "
                + ", ".join(unsupported)
            )
        MEMORIES = tuple(memories)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=("BF16", "FP8"), default="BF16")
    parser.add_argument(
        "--dram-vendor",
        choices=("winbond", "micron"),
        default="winbond",
        help=(
            "DRAM IDD table for FP8. Micron reuses LPDDR4X timing/traces and "
            "loads its separately refreshed energy CSVs."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(balanced.MODEL_CONFIG),
        default=list(balanced.MODEL_CONFIG),
        help="Model subset to summarize (default: full campaign).",
    )
    parser.add_argument(
        "--contexts",
        nargs="+",
        choices=tuple(CONTEXT_CATALOG),
        default=list(balanced.CONTEXTS),
        help="Context subset to summarize (default: full campaign).",
    )
    parser.add_argument(
        "--memories",
        nargs="+",
        choices=tuple(balanced.MEMORY_CASES),
        help="Memory-case subset; FP8 accepts LPDDR4X cases only.",
    )
    parser.add_argument(
        "--systolic-raw-root",
        type=Path,
    )
    parser.add_argument("--output-dir", type=Path)
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
    parser.add_argument(
        "--cycle-breakdown",
        action="store_true",
        help=(
            "Record missing command sidecars for the throughput/device-selected "
            "Systolic traces, then write issued-command PIM cycle breakdowns."
        ),
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    precision: str = "BF16",
) -> dict[str, object]:
    """Apply batch-group admission to the existing systolic KV-head TP row."""

    model_config = MODELS[model]
    context_config = CONTEXTS[context]
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
        1 if precision == "FP8" else 2,
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
        "Precision": precision,
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
    precision: str = "BF16",
) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    envelope = balanced.sweep_envelope(
        device_capacity_gib,
        reserve_gib,
        tp_values=(1, 2, 4, 8),
        element_bytes=(1 if precision == "FP8" else 2),
        contexts=CONTEXTS,
    )
    envelope = envelope[
        envelope["Model"].isin(MODELS) & envelope["Context"].isin(CONTEXTS)
    ]
    rows: list[dict[str, object]] = []
    manifest: dict[str, dict[str, str]] = {}
    for memory in MEMORIES:
        config = MEMORY_CASES[memory]
        source_path = (raw_root / Path(config["csv"])).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"missing Systolic source CSV: {source_path}")
        manifest[memory] = {
            "path": balanced.display_path(source_path),
            "sha256": file_sha256(source_path),
        }
        for model in MODELS:
            for context in CONTEXTS:
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
                        precision=precision,
                        context_config=CONTEXTS[context],
                    )
                    for tp, source in sources.items():
                        if not bool(source.get("Systolic pim", False)):
                            raise ValueError(
                                f"{memory}/{model}/{context}/TP={tp} is not systolic"
                            )
                        if int(source.get("Systolic dim", -1)) != 4:
                            raise ValueError(
                                f"{memory}/{model}/{context}/TP={tp} does not use "
                                "a four-row systolic array"
                            )
                        expected_width = 32 if precision == "FP8" else 16
                        source_width = source.get("PIM elements per 256-bit word")
                        if (
                            (precision == "FP8" and source_width is None)
                            or (
                                source_width is not None
                                and int(source_width) != expected_width
                            )
                        ):
                            raise ValueError(
                                f"{memory}/{model}/{context}/TP={tp} does not use "
                                f"a 4x{expected_width} systolic array"
                            )
                        if precision == "FP8" and source.get(
                            "Systolic PIM power assumption"
                        ) != "iso_256b_bf16_calibration":
                            raise ValueError(
                                f"{memory}/{model}/{context}/TP={tp} is missing "
                                "the FP8 fixed-256-bit power assumption"
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
                            precision,
                        )
                        candidate["Architecture"] = ARCHITECTURES[0]
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


def write_plots(
    candidates: pd.DataFrame,
    selected_equal_power: pd.DataFrame,
    component_breakdown: pd.DataFrame,
    dgx: dict[tuple[str, str], dict[str, float | int]],
    output_dir: Path,
    cycle_breakdown: pd.DataFrame | None = None,
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
        architectures=ARCHITECTURES,
        memories=MEMORIES,
        contexts=CONTEXTS,
        models=MODELS,
        memory_colors=MEMORY_COLORS,
        architecture_hatches=ARCH_HATCH,
        metric="Throughput / device (tokens/s/device)",
        ylabel="Tokens/s/device",
        title="best capacity-admitted PP/TP/batch (labels: PP/TP/B)",
        output=output_dir / "best_throughput_per_device",
        annotation="pp_tp",
    )
    plot_grouped_metric(
        efficiency,
        architectures=ARCHITECTURES,
        memories=MEMORIES,
        contexts=CONTEXTS,
        models=MODELS,
        memory_colors=MEMORY_COLORS,
        architecture_hatches=ARCH_HATCH,
        metric="Tokens/J",
        ylabel="Tokens/J",
        title="best energy efficiency (labels: token/J/DGX; PP/TP/B)",
        output=output_dir / "best_tokens_per_joule",
        annotation="tokens_per_joule",
        dgx_line=True,
        dgx_reference=dgx_tokens_per_joule,
    )
    plot_selected_component_breakdown(
        component_breakdown,
        architectures=ARCHITECTURES,
        memories=MEMORIES,
        contexts=CONTEXTS,
        models=MODELS,
        metric_suffix="energy (mJ/token)",
        ylabel="Effective token energy (mJ/token)",
        title="system energy breakdown for throughput/device-selected layouts",
        output=output_dir / "selected_token_energy",
    )
    plot_selected_component_breakdown(
        component_breakdown,
        architectures=ARCHITECTURES,
        memories=MEMORIES,
        contexts=CONTEXTS,
        models=MODELS,
        metric_suffix="power / device (W)",
        ylabel="Average power per provisioned device (W/device)",
        title="system power/device breakdown for throughput/device-selected layouts",
        output=output_dir / "selected_power_per_device",
    )
    plot_grouped_metric(
        selected_equal_power,
        architectures=ARCHITECTURES,
        memories=MEMORIES,
        contexts=CONTEXTS,
        models=MODELS,
        memory_colors=MEMORY_COLORS,
        architecture_hatches=ARCH_HATCH,
        metric="System throughput (tokens/s)",
        ylabel="System throughput (tokens/s)",
        title="DGX H100 equal-power winners (labels: throughput/DGX; PP/TP/B/DP)",
        output=output_dir / "equal_power_system_throughput",
        annotation="equal_power",
        dgx_line=True,
    )
    if cycle_breakdown is not None:
        plot_selected_systolic_cycle_breakdown(
            cycle_breakdown,
            output_dir / "selected_pim_cycle_breakdown",
        )


def write_readme(
    output_dir: Path,
    raw_root: Path,
    candidates: pd.DataFrame,
    winners: pd.DataFrame,
    flash_attention: bool,
    flash_attention_block_size: int,
    pipelined_softmax: bool,
    cycle_breakdown_enabled: bool,
    dram_vendor: str,
) -> None:
    precision = str(candidates["Precision"].iloc[0])
    architecture = str(candidates["Architecture"].iloc[0])
    try:
        relative_raw_root = raw_root.relative_to(CENT_SIM / "output")
    except ValueError:
        trace_root_option = ""
    else:
        trace_parts = tuple(part for part in relative_raw_root.parts if part != "raw")
        trace_root = CENT_SIM / "trace" / Path(*trace_parts)
        trace_root_option = f" --trace-root {trace_root}"
    case_list = ",".join(MEMORIES)
    runner_case_list = ",".join(
        f"{memory}_MICRON" if dram_vendor == "micron" else memory
        for memory in MEMORIES
    )
    memory_description = " plus ".join(MEMORIES)
    ramulator_jobs = (
        len(MEMORIES)
        * len(MODELS)
        * len(CONTEXTS)
        * len(SYSTOLIC_BATCH_SIZES)
        * 4
    )
    functional_traces = (
        len({MEMORY_CASES[name]["dram_impl"] for name in MEMORIES})
        * len(MODELS)
        * len(CONTEXTS)
        * len(SYSTOLIC_BATCH_SIZES)
        * 4
    )
    context_description = "/".join(CONTEXTS)
    context_windows = ",".join(
        str(CONTEXTS[context]["window"]) for context in CONTEXTS
    )
    model_list = ",".join(MODELS)
    precision_note = (
        "FP8 PIM energy uses the explicit `iso_256b_bf16_calibration` "
        "assumption: the physical command remains 256 bits and reuses the "
        "measured BF16 array scaling until an FP8 circuit-level table is "
        "supplied. PNM, accumulators, Softmax, and TP/CXL partial sums remain "
        "BF16 in this campaign."
        if precision == "FP8"
        else "The campaign uses the measured BF16 systolic-array power scaling."
    )
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
    cycle_command_option = " --cycle-breakdown" if cycle_breakdown_enabled else ""
    campaign_scope = context_description if len(CONTEXTS) == 1 else "all-context"
    shared_trace_note = (
        " Both LPDDR timings reuse the same trace set."
        if sum(memory.startswith("LPDDR4X") for memory in MEMORIES) > 1
        else ""
    )
    text = f"""# KV-head TP Systolic {campaign_scope} campaign

This directory contains only the standard **{architecture}**
Device–Channel-group–Bank KV-head TP mapping.  It does not load, select, or
plot Vector candidates.  The read-only Vector-versus-Systolic five-metric
comparison is generated separately by
`scripts/analyze_kv_head_tp_systolic_vs_vector.py`.

Systolic PIM follows cent_dev and therefore has effective EWMUL_PNM enabled.
W1 `AF`/`RD_AF` commands remain in the PIM trace; RMSNorm, RoPE, and the fused
FFN element-wise multiplies use the analytical PNM VEC_MUL path.

{precision_note}

Systolic source rows use FlashAttention: **{flash_attention}**. When enabled,
the context block size is **{flash_attention_block_size}** and score workspace
`W_MEM`/`R_MEM` traffic is absent from each block.

Pipelined Softmax is **{pipelined_softmax}**. When enabled, full Softmax energy
is retained, while only the cent_dev producer-startup fraction remains exposed
on the latency critical path.

It uses {precision}, 16 GiB/device, the {context_description} midpoint samples, and
{memory_description}. {architecture} evaluates batch 1/2/3/4. Other array
heights are intentionally excluded. The Systolic raw campaign has
{ramulator_jobs} Ramulator jobs and {functional_traces} functional traces.{shared_trace_note}

DRAM power vendor: **{dram_vendor}**. {'The Micron IDD table is a power-only refresh over the identical LPDDR4X timing/traces.' if dram_vendor == 'micron' else 'The Winbond IDD table is used.'}

`selected_token_energy_*` and `selected_power_per_device_*` are system-energy
breakdowns for the throughput/device-selected layouts.  Their x-axis hierarchy
is **context → {architecture} → memory**; the legend contains only energy
components. The physical terms use the same palette and grouping as CENT's
system energy breakdown: DRAM, I/O/controller, SRAM, accelerator, and PCIe.
`Trace-external waiting` and `Pipeline-bubble waiting` are retained as two
explicit gray terms because they are part of effective token energy but do not
come from a Ramulator command.  The audit CSV
`selected_throughput_energy_component_breakdown.csv` verifies that both the
energy stack and the per-device-power stack reconstruct their pre-existing
scalar values exactly. This Systolic-only campaign reconstructs only its own
source rows; it contains no Vector source-CSV calibration.

{("`selected_pim_cycle_breakdown.csv` and `selected_pim_cycle_breakdown_*` attribute each selected layout's critical PIM trace to issued-command components, using the same classification as AIM Simulator's GEMV cycle breakdown. TraceRecorder sidecars were generated only where they were missing." if cycle_breakdown_enabled else "Issued-command PIM cycle breakdowns are optional: rerun this analysis with `--cycle-breakdown` to record any missing TraceRecorder sidecars and generate them.")}

Equal power is evaluated for each memory. First, throughput/device fixes one
capacity-admitted PP/TP/batch layout; equal-score ties use fewer replica
devices, then smaller PP, TP, and batch. Only that fixed layout is DP-scaled,
retaining both integer DP neighbors around the DGX power target for audit and
selecting the nearest one (smaller DP on an exact tie). Therefore DP packing
cannot reselect PP/TP/batch. Each workload has {len(MEMORIES)} CENT bars; DGX
H100 is a line, not a bar.

Candidates: {len(candidates)} total; {int(candidates['Can host one batch'].astype(bool).sum())} admitted.
Equal-power winners: {len(winners)} total ({len(MEMORIES)} per model/context).

## Equal-power winners

{balanced.markdown_table(summary, list(summary.columns))}

## Reproduction

```bash
{sys.executable} scripts/run_cent_memory_cases.py \\
  --kv-head-tp-systolic --precision {precision.lower()} \\
  --models {model_list} --context-windows {context_windows} --cases {runner_case_list} \\
  --output-root {raw_root}{trace_root_option} \\
  --EWMUL_PNM{flash_command_options}{pipeline_command_option}{' --power-refresh-only' if dram_vendor == 'micron' else ''}

{sys.executable} scripts/analyze_kv_head_tp_systolic_all_context.py \\
  --precision {precision} --models {model_list} --contexts {context_description.replace('/', ' ')} \\
  --memories {case_list.replace(',', ' ')} --dram-vendor {dram_vendor}{flash_command_options}{pipeline_command_option}{cycle_command_option} \\
  --systolic-raw-root {raw_root} --output-dir {output_dir}
```

For an accelerator-only accounting refresh, the first command reuses valid
Ramulator traces/logs and overwrites the source CSV rows before regenerating
this analysis. Legacy `--activation` provenance is retained for audit but is
excluded from candidates. To physically replace the raw source CSVs too, delete
the `simulation_results_decode_only_long_context_midpoint*.csv` files under
`{raw_root}` before the first command.
"""
    (output_dir / "README.md").write_text(text)


def main() -> int:
    args = parse_args()
    systolic_width = configure_precision(args.precision, args.dram_vendor)
    configure_scope(args.models, args.contexts, args.memories)
    if args.device_capacity_gib <= args.reserve_gib:
        raise ValueError("device capacity must exceed reserve")
    default_variant = (
        "systolic_4x32_fp8" if args.precision == "FP8" else "systolic_4x16"
    )
    raw_root = (
        args.systolic_raw_root
        if args.systolic_raw_root is not None
        else DEFAULT_ROOT / "raw" / default_variant
    ).resolve()
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else DEFAULT_ROOT / (
            "analysis_fp8_micron"
            if args.precision == "FP8" and args.dram_vendor == "micron"
            else ("analysis_fp8" if args.precision == "FP8" else "analysis")
        )
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    systolic, systolic_manifest = load_systolic_candidates(
        raw_root,
        args.device_capacity_gib,
        args.reserve_gib,
        args.dram_energy_model,
        args.flash_attention,
        args.flash_attention_block_size,
        args.pipelined_softmax,
        args.precision,
    )
    candidates = add_architecture_ranks(systolic).sort_values(
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
    component_breakdown = build_selected_component_breakdown(
        best_throughput,
        memory_cases={name: MEMORY_CASES[name] for name in MEMORIES},
        model_specs=balanced.MODEL_SPECS,
        project_root=ROOT,
    ).sort_values(
        ["Model", "Context window", "Architecture", "Memory"]
    )
    cycle_breakdown = None
    if args.cycle_breakdown:
        cycle_breakdown = build_selected_systolic_cycle_breakdown(
            best_throughput,
            project_root=ROOT,
        ).sort_values(["Model", "Context", "Memory"])
    dgx = {
        workload: reference
        for workload, reference in balanced.load_dgx(
            args.h100_profile.resolve(), contexts=CONTEXTS
        ).items()
        if workload[0] in MODELS and workload[1] in CONTEXTS
    }
    deployments = build_all_equal_power_deployments(
        best_throughput, dgx, balanced.integer_dp_choices
    ).sort_values(
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
        * len(MODELS)
        * len(CONTEXTS)
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
        "selected_throughput_energy_component_breakdown.csv": component_breakdown,
    }
    for filename, frame in outputs.items():
        path = output_dir / filename
        frame.to_csv(path, index=False)
        print(f"[data] {balanced.display_path(path)}")
    if cycle_breakdown is not None:
        cycle_path = output_dir / "selected_pim_cycle_breakdown.csv"
        write_cycle_breakdown_csv(cycle_breakdown, cycle_path)
        print(f"[data] {balanced.display_path(cycle_path)}")

    manifest = {
        "architectures": list(ARCHITECTURES),
        "memories": list(MEMORIES),
        "models": list(MODELS),
        "contexts": CONTEXTS,
        "precision": args.precision,
        "dram_vendor": args.dram_vendor,
        "systolic_array": f"4x{systolic_width}",
        "ewmul_pnm_effective": True,
        "flash_attention": args.flash_attention,
        "flash_attention_block_size": (
            args.flash_attention_block_size if args.flash_attention else 0
        ),
        "pipelined_softmax": args.pipelined_softmax,
        "cycle_breakdown": args.cycle_breakdown,
        "batch_sizes": list(SYSTOLIC_BATCH_SIZES),
        "tp_values": [1, 2, 4, 8],
        "ramulator_jobs": (
            len(MEMORIES)
            * len(MODELS)
            * len(CONTEXTS)
            * len(SYSTOLIC_BATCH_SIZES)
            * 4
        ),
        "functional_traces": (
            len({MEMORY_CASES[name]["dram_impl"] for name in MEMORIES})
            * len(MODELS)
            * len(CONTEXTS)
            * len(SYSTOLIC_BATCH_SIZES)
            * 4
        ),
        "systolic_sources": systolic_manifest,
        "selection_rule": (
            "max_throughput_per_device_then_fewer_replica_devices_smaller_PP_TP_batch_"
            "then_record_floor_and_ceil_DP_and_select_nearest_integer_DP"
        ),
    }
    manifest_text = json.dumps(manifest, indent=2) + "\n"
    (output_dir / "manifest.json").write_text(manifest_text)
    write_readme(
        output_dir,
        raw_root,
        candidates,
        winners,
        args.flash_attention,
        args.flash_attention_block_size,
        args.pipelined_softmax,
        args.cycle_breakdown,
        args.dram_vendor,
    )

    if not args.no_plots:
        write_plots(
            candidates,
            winners,
            component_breakdown,
            dgx,
            output_dir,
            cycle_breakdown,
        )
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
