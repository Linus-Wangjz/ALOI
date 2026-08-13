#!/usr/bin/env python3
"""Analyze capacity-constrained balanced PP/TP layouts for CENT.

Ramulator supplies one corrected inter-device-attention transformer-block
measurement for each model/context/TP tuple.  This script sweeps exact layer
divisors for PP and powers of two for TP.  A PP sweep starts at TP=1 and keeps
doubling TP through the first point where Bmax > PP.  Bmax=0 is rejected;
1 <= Bmax < PP remains valid with proportional pipeline fill.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd

_CENT_SIM_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(_CENT_SIM_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_CENT_SIM_IMPORT_ROOT))
import cent_power_calculator as cent
from scripts.cent_capacity import MODEL_SPECS, capacity_for_layout


ROOT = Path(__file__).resolve().parents[2]
CENT_SIM = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ROOT = CENT_SIM / "output/balanced_equal_power_all_contexts"
CHANNELS_PER_DEVICE = 32
PRECISION = "BF16"

# Seaborn's colorblind palette.  Keep the exact colors locally so plotting
# does not add a runtime dependency on seaborn.
SEABORN_COLORBLIND = (
    "#0173B2",
    "#DE8F05",
    "#029E73",
    "#D55E00",
    "#CC78BC",
    "#CA9161",
    "#FBAFE4",
    "#949494",
    "#ECE133",
    "#56B4E9",
)

CONTEXTS = {
    "4K": {"window": 4096, "active": 2304},
    "32K": {"window": 32768, "active": 30976},
    "128K": {"window": 131072, "active": 129280},
}

MODEL_CONFIG = {
    "Llama2-7B": {
        "gpu_model": "Llama-3.1-8B",
        "dgx_power_w": 8 * 581.5,
        # TP16 is needed only to close the 128K/PP1 sweep.  Keeping 18 lanes
        # per source device preserves the previous 8-device/144-lane model.
        "source_devices": 16,
        "source_pcie_lanes": 288,
    },
    "Llama2-70B": {
        "gpu_model": "Llama-3.1-70B",
        "dgx_power_w": 8 * 556.3,
        "source_devices": 32,
        "source_pcie_lanes": 144,
    },
}

MEMORY_CASES = {
    "GDDR6": {
        "csv": Path("GDDR6/simulation_results_decode_only_long_context_midpoint.csv"),
        "dram_impl": "GDDR6",
    },
    "LPDDR4X_nCCD2": {
        "csv": Path("LPDDR4X/simulation_results_decode_only_long_context_midpoint_nCCD2.csv"),
        "dram_impl": "LPDDR4X",
    },
    "LPDDR4X_nCCD6": {
        "csv": Path("LPDDR4X/simulation_results_decode_only_long_context_midpoint_nCCD6.csv"),
        "dram_impl": "LPDDR4X",
    },
}


def parse_args(
    *,
    default_experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
    default_attention_mapping: str = "inter_device",
    default_tp_values: tuple[int, ...] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=default_experiment_root / "raw")
    parser.add_argument("--output-dir", type=Path, default=default_experiment_root / "analysis")
    parser.add_argument("--h100-profile", type=Path, default=ROOT / "DGX_H100_profile_results.csv")
    parser.add_argument("--device-capacity-gib", type=float, default=16.0)
    parser.add_argument("--reserve-gib", type=float, default=0.0)
    parser.add_argument("--dram-energy-model", choices=cent.DRAM_ENERGY_MODELS, default="legacy")
    parser.add_argument(
        "--attention-mapping",
        choices=("inter_device", "kv_head"),
        default=default_attention_mapping,
    )
    parser.add_argument(
        "--tp-values",
        type=int,
        nargs="+",
        default=list(default_tp_values) if default_tp_values is not None else None,
        help="Fixed full-grid TP values. Omit to retain the legacy Bmax>PP stopping sweep.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_pp_values(model: str) -> tuple[int, ...]:
    layers = MODEL_SPECS[model]["layers"]
    return tuple(value for value in range(1, layers + 1) if layers % value == 0)


def non_dram_static_power_w(channels_per_device: int = CHANNELS_PER_DEVICE) -> float:
    milliwatts = (
        cent.SRAM_POWER["GB"]["STT"] * channels_per_device
        + cent.SRAM_POWER["SB"]["STT"]
        + cent.SRAM_POWER["IB"]["STT"]
        + sum(
            cent.ACCEL_POWER[name]["STT"] * channels_per_device
            for name in ("RED", "EXP", "VEC", "VEC_MUL")
        )
        + 0.5 * (cent.ACCEL_POWER["CTR"]["STT"] + cent.ACCEL_POWER["CTR"]["DYN"])
    )
    return milliwatts / 1000.0


def waiting_floor_power_w(
    dram_impl: str,
    channels_per_device: int = CHANNELS_PER_DEVICE,
) -> float:
    """Return the precharged waiting floor, including non-DRAM static power."""

    dram_power = cent.CELLAR_POWER_CALCULATOR.dram_power_for_impl(dram_impl)
    dram_floor = dram_power["PRE_STBY"] * channels_per_device / 1000.0
    return dram_floor + non_dram_static_power_w(channels_per_device)


def capacity_for_candidate(
    model: str,
    context_window: int,
    pp: int,
    tp: int,
    device_capacity_gib: float,
    reserve_gib: float,
) -> dict[str, float | int]:
    return capacity_for_layout(
        model,
        pp,
        tp,
        context_window,
        int(device_capacity_gib * 2**30),
        int(reserve_gib * 2**30),
        shard_kv_cache_across_tp=True,
    )


def tp_sweep_for_pp(
    model: str,
    context_window: int,
    pp: int,
    device_capacity_gib: float,
    reserve_gib: float,
    *,
    max_tp: int = 4096,
) -> list[tuple[int, dict[str, float | int]]]:
    """Evaluate TP=1,2,4,... through and including the first Bmax > PP."""

    points: list[tuple[int, dict[str, float | int]]] = []
    tp = 1
    while tp <= max_tp:
        capacity = capacity_for_candidate(
            model, context_window, pp, tp, device_capacity_gib, reserve_gib
        )
        points.append((tp, capacity))
        if int(capacity["Max resident microbatch"]) > pp:
            return points
        tp *= 2
    raise RuntimeError(
        f"TP sweep did not reach Bmax > PP for {model}, context={context_window}, PP={pp}"
    )


def sweep_envelope(
    device_capacity_gib: float,
    reserve_gib: float,
    tp_values: Iterable[int] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in MODEL_CONFIG:
        for context, context_config in CONTEXTS.items():
            window = int(context_config["window"])
            for pp in exact_pp_values(model):
                if tp_values is None:
                    points = tp_sweep_for_pp(
                        model, window, pp, device_capacity_gib, reserve_gib
                    )
                else:
                    points = [
                        (
                            tp,
                            capacity_for_candidate(
                                model,
                                window,
                                pp,
                                tp,
                                device_capacity_gib,
                                reserve_gib,
                            ),
                        )
                        for tp in sorted(set(int(value) for value in tp_values))
                    ]
                for index, (tp, capacity) in enumerate(points):
                    bmax = int(capacity["Max resident microbatch"])
                    used = min(bmax, pp)
                    rows.append(
                        {
                            "Model": model,
                            "Context": context,
                            "Context window": window,
                            "PP": pp,
                            "TP": tp,
                            "Max resident microbatch": bmax,
                            "Microbatch used": used,
                            "Pipeline fill ratio": used / pp,
                            "Can host one request": bmax >= 1,
                            "Pipeline full": bmax >= pp,
                            "TP stopping point": index == len(points) - 1,
                            "Stop condition Bmax > PP": bmax > pp,
                        }
                    )
    return pd.DataFrame(rows)


def required_tps(envelope: pd.DataFrame, model: str, context: str) -> tuple[int, ...]:
    subset = envelope[(envelope["Model"] == model) & (envelope["Context"] == context)]
    return tuple(sorted(int(value) for value in subset["TP"].unique()))


def integer_dp_choices(target_power_w: float, replica_power_w: float) -> list[tuple[str, int]]:
    if target_power_w <= 0.0 or replica_power_w <= 0.0:
        raise ValueError("target and replica power must be positive")
    exact = target_power_w / replica_power_w
    floor_dp = max(1, math.floor(exact))
    ceil_dp = max(1, math.ceil(exact))
    choices = [("floor", floor_dp)]
    if ceil_dp != floor_dp:
        choices.append(("ceil", ceil_dp))
    return choices


def load_dgx(profile_path: Path) -> dict[tuple[str, str], dict[str, float | int]]:
    df = pd.read_csv(profile_path)
    result: dict[tuple[str, str], dict[str, float | int]] = {}
    for model, model_config in MODEL_CONFIG.items():
        for context, context_config in CONTEXTS.items():
            window = int(context_config["window"])
            subset = df[
                (df["model"] == model_config["gpu_model"])
                & (df["num_gpu"] == 8)
                & (df["stage"] == "Decoding")
                & (df["seqlen"] == window)
            ]
            if subset.empty:
                raise ValueError(f"missing 8-GPU DGX profile for {model}, context={window}")
            best = subset.loc[subset["throughput (tokens/s)"].idxmax()]
            result[(model, context)] = {
                "throughput": float(best["throughput (tokens/s)"]),
                "batch": int(best["batch"]),
                "power": float(model_config["dgx_power_w"]),
                "profile_model": str(model_config["gpu_model"]),
            }
    return result


SOURCE_COLUMNS = {
    "Model",
    "Device number",
    "Pipeline parallelism",
    "Tensor parallelism",
    "Sequence length",
    "Context window",
    "Main PIM latency",
    "Helper PIM latency",
    "CXL latency",
    "Acc latency",
    "TransformerBlock latency",
    "Token latency (ms)",
    "Token energy (mJ)",
    "Attention mapping",
    "DRAM energy model",
}


def load_tp_sources(
    path: Path,
    model: str,
    context: str,
    required: Iterable[int],
    dram_energy_model: str,
    attention_mapping: str = "inter_device",
    batch_size: int | None = None,
    flash_attention: bool | None = None,
    flash_attention_block_size: int | None = None,
    pipelined_softmax: bool | None = None,
    ewmul_pnm_effective: bool | None = None,
) -> dict[int, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"missing source CSV: {path}")
    df = pd.read_csv(path)
    missing = sorted(SOURCE_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    model_config = MODEL_CONFIG[model]
    context_config = CONTEXTS[context]
    source_devices = int(model_config["source_devices"])
    subset = df[
        (df["Model"] == model)
        & (df["Device number"] == source_devices)
        & (df["Sequence length"] == int(context_config["active"]))
        & (df["Context window"] == int(context_config["window"]))
        & (df["Attention mapping"] == attention_mapping)
        & (df["DRAM energy model"] == dram_energy_model)
    ].copy()
    if batch_size is not None:
        if "Batch size" not in subset.columns:
            raise ValueError(f"{path} is missing Batch size")
        subset = subset[subset["Batch size"] == batch_size]
    if flash_attention is not None:
        if "Flash attention" not in subset.columns:
            raise ValueError(f"{path} is missing Flash attention provenance")
        subset = subset[subset["Flash attention"].astype(bool) == flash_attention]
        if flash_attention and flash_attention_block_size is not None:
            if "Flash attention block size" not in subset.columns:
                raise ValueError(
                    f"{path} is missing Flash attention block size provenance"
                )
            subset = subset[
                subset["Flash attention block size"] == flash_attention_block_size
            ]
    if pipelined_softmax is not None:
        if "Pipelined softmax" not in subset.columns:
            raise ValueError(f"{path} is missing Pipelined softmax provenance")
        subset = subset[
            subset["Pipelined softmax"].astype(bool) == pipelined_softmax
        ]
    if ewmul_pnm_effective is not None:
        required_ewmul = {"EWMUL PNM effective", "EWMUL PNM provenance"}
        if not required_ewmul.issubset(subset.columns):
            raise ValueError(
                f"{path} uses legacy --activation provenance; regenerate it "
                "with the --EWMUL_PNM model"
            )
        subset = subset[
            (subset["EWMUL PNM effective"] == ewmul_pnm_effective)
            & (subset["EWMUL PNM provenance"] == "native")
        ]
    subset = subset[
        subset["Pipeline parallelism"] * subset["Tensor parallelism"]
        == subset["Device number"]
    ]
    sources: dict[int, pd.Series] = {}
    for tp in required:
        rows = subset[subset["Tensor parallelism"] == tp]
        if len(rows) != 1:
            raise ValueError(
                f"expected one {attention_mapping} {model}/{context}/TP={tp} "
                f"batch={batch_size if batch_size is not None else 'any'} row "
                f"in {path}, found {len(rows)}"
            )
        row = rows.iloc[0]
        expected_pp = source_devices // tp
        if int(row["Pipeline parallelism"]) != expected_pp:
            raise ValueError(
                f"{model}/{context}/TP={tp} source PP is "
                f"{int(row['Pipeline parallelism'])}, expected {expected_pp}"
            )
        sources[tp] = row
    return sources


def build_candidate(
    memory: str,
    dram_impl: str,
    source_path: Path,
    source: pd.Series,
    model: str,
    context: str,
    pp: int,
    tp: int,
    device_capacity_gib: float,
    reserve_gib: float,
    attention_mapping: str = "inter_device",
) -> dict[str, object]:
    model_config = MODEL_CONFIG[model]
    context_config = CONTEXTS[context]
    layers = MODEL_SPECS[model]["layers"]
    capacity = capacity_for_candidate(
        model,
        int(context_config["window"]),
        pp,
        tp,
        device_capacity_gib,
        reserve_gib,
    )
    bmax = int(capacity["Max resident microbatch"])
    used_microbatch = min(bmax, pp)
    fill_ratio = used_microbatch / pp

    main_pim_ms = float(source["Main PIM latency"])
    helper_pim_ms = float(source["Helper PIM latency"]) if tp > 1 else 0.0
    critical_pim_ms = max(main_pim_ms, helper_pim_ms)
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
    acc_ms = float(source["Acc latency"])
    if attention_mapping == "kv_head":
        required_role_columns = {"Main Acc latency", "Helper Acc latency"}
        missing = sorted(required_role_columns - set(source.index))
        if missing:
            raise ValueError(
                f"KV-head source is missing role latency columns: {', '.join(missing)}"
            )
        main_acc_ms = float(source["Main Acc latency"])
        helper_acc_ms = float(source["Helper Acc latency"])
        critical_local_ms = max(
            main_pim_ms + main_acc_ms,
            helper_pim_ms + helper_acc_ms,
        )
        source_block_ms = critical_local_ms + source_cxl_ms
        block_ms = critical_local_ms + cxl_ms
    else:
        main_acc_ms = acc_ms
        helper_acc_ms = acc_ms if tp > 1 else 0.0
        critical_local_ms = critical_pim_ms + acc_ms
        source_block_ms = critical_local_ms + source_cxl_ms
        block_ms = critical_local_ms + cxl_ms
    if not math.isclose(
        source_block_ms,
        float(source["TransformerBlock latency"]),
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{memory}/{model}/{context}/TP={tp} block latency mismatch")
    source_serial_ms = layers * source_block_ms
    if not math.isclose(
        source_serial_ms,
        float(source["Token latency (ms)"]),
        rel_tol=1e-9,
        abs_tol=1e-8,
    ):
        raise ValueError(f"{memory}/{model}/{context}/TP={tp} serial latency mismatch")
    serial_ms = layers * block_ms

    ideal_replica_throughput = 1000.0 * pp / serial_ms
    replica_throughput = ideal_replica_throughput * fill_ratio
    replica_devices = pp * tp
    throughput_per_device = replica_throughput / replica_devices

    main_gap_ms = max(0.0, block_ms - main_pim_ms)
    helper_gap_ms = max(0.0, block_ms - helper_pim_ms) if tp > 1 else 0.0
    base_active_energy_mj = float(source["Token energy (mJ)"])
    pp_handoff_energy_adjustment_mj = 0.0
    if has_split_cxl and pp == 1:
        if "PP handoff CXL payload (bits/block)" not in source.index:
            raise ValueError("split CXL source is missing PP handoff payload")
        pp_handoff_energy_adjustment_mj = (
            layers
            * float(source["PP handoff CXL payload (bits/block)"])
            * float(cent.CELLAR_POWER_CALCULATOR.PCIE_ENERGY)
            / 1.0e9
        )
        base_active_energy_mj -= pp_handoff_energy_adjustment_mj
    waiting_floor_w = waiting_floor_power_w(dram_impl)
    main_gap_energy_mj = layers * main_gap_ms * waiting_floor_w
    helper_gap_energy_mj = layers * (tp - 1) * helper_gap_ms * waiting_floor_w
    gap_energy_mj = main_gap_energy_mj + helper_gap_energy_mj
    work_token_energy_mj = base_active_energy_mj + gap_energy_mj
    work_power_w = work_token_energy_mj * replica_throughput / 1000.0
    pipeline_idle_floor_power_w = replica_devices * (1.0 - fill_ratio) * waiting_floor_w
    replica_power_w = work_power_w + pipeline_idle_floor_power_w
    effective_token_energy_mj = (
        replica_power_w / replica_throughput * 1000.0
        if replica_throughput > 0.0
        else math.nan
    )
    tokens_per_joule = replica_throughput / replica_power_w if replica_power_w > 0.0 else 0.0

    return {
        "Memory": memory,
        "Model": model,
        "Context": context,
        "Context window": int(context_config["window"]),
        "Active sequence length": int(context_config["active"]),
        "Precision": PRECISION,
        "Attention mapping": attention_mapping,
        "KV mapping": "kv_head_sharded" if attention_mapping == "kv_head" else "fully_tp_sharded",
        "Stage balance": f"{layers}_layers_exact_divisor",
        "PP": pp,
        "TP": tp,
        "Layers per stage": layers // pp,
        "Replica devices": replica_devices,
        "Max resident microbatch": bmax,
        "Microbatch used": used_microbatch,
        "Pipeline fill ratio": fill_ratio,
        "Can host one request": bmax >= 1,
        "Pipeline full": bmax >= pp,
        "Stop condition Bmax > PP": bmax > pp,
        "Device capacity (GiB)": device_capacity_gib,
        "Per-device reserve (GiB)": reserve_gib,
        "Bottleneck stage": int(capacity["Bottleneck pipeline stage"]),
        "Static / bottleneck device (GiB)": float(
            capacity["Static model footprint / bottleneck device (GiB)"]
        ),
        "KV / request / bottleneck device (GiB)": float(
            capacity["KV cache / request / bottleneck device (GiB)"]
        ),
        "KV / request / model (GiB)": float(capacity["KV cache / request / model (GiB)"]),
        "Main PIM latency (ms/block)": main_pim_ms,
        "Helper PIM latency (ms/block)": helper_pim_ms,
        "Critical PIM latency (ms/block)": critical_pim_ms,
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
        "Ideal full-pipeline throughput (tokens/s)": ideal_replica_throughput,
        "Replica throughput (tokens/s)": replica_throughput,
        "Throughput / device (tokens/s/device)": throughput_per_device,
        "Base active token energy (mJ)": base_active_energy_mj,
        "Removed PP handoff energy (mJ/token)": pp_handoff_energy_adjustment_mj,
        "Waiting floor power / device (W)": waiting_floor_w,
        "Main gap energy (mJ/token)": main_gap_energy_mj,
        "Helper gap energy (mJ/token)": helper_gap_energy_mj,
        "Trace-external gap energy (mJ/token)": gap_energy_mj,
        "Work token energy (mJ)": work_token_energy_mj,
        "Work power (W)": work_power_w,
        "Pipeline idle-floor power (W)": pipeline_idle_floor_power_w,
        "Replica power (W)": replica_power_w,
        "Power / device (W)": replica_power_w / replica_devices,
        "Effective token energy (mJ)": effective_token_energy_mj,
        "Tokens/J": tokens_per_joule,
        "Source PP": int(source["Pipeline parallelism"]),
        "Source TP": int(source["Tensor parallelism"]),
        "Source devices": int(source["Device number"]),
        "Source total PCIe lanes": int(model_config["source_pcie_lanes"]),
        "Source PCIe lanes / device": int(model_config["source_pcie_lanes"])
        // int(model_config["source_devices"]),
        "Source CSV": display_path(source_path),
        "Helper role assumption": "one_representative_helper_times_TP_minus_1",
    }


def assign_ranks(candidates: pd.DataFrame) -> pd.DataFrame:
    ranked = candidates.copy()
    ranked["Throughput/device rank"] = pd.NA
    ranked["Tokens/J rank"] = pd.NA
    admitted = ranked["Can host one request"]
    group_columns = ["Memory", "Model", "Context"]
    for _, group in ranked[admitted].groupby(group_columns):
        indices = group.index
        ranked.loc[indices, "Throughput/device rank"] = group[
            "Throughput / device (tokens/s/device)"
        ].rank(method="dense", ascending=False)
        ranked.loc[indices, "Tokens/J rank"] = group["Tokens/J"].rank(
            method="dense", ascending=False
        )
    return ranked


def choose_base_configs(candidates: pd.DataFrame) -> pd.DataFrame:
    selected: list[dict[str, object]] = []
    admitted = candidates[candidates["Can host one request"]]
    for (memory, model, context), rows in admitted.groupby(["Memory", "Model", "Context"]):
        for objective, score_column in (
            ("throughput_per_device", "Throughput / device (tokens/s/device)"),
            ("tokens_per_joule", "Tokens/J"),
        ):
            best_score = float(rows[score_column].max())
            tied = rows[
                rows[score_column].map(
                    lambda value: math.isclose(
                        float(value), best_score, rel_tol=1e-12, abs_tol=1e-15
                    )
                )
            ]
            winner = tied.sort_values(
                ["PP", "TP", "Replica devices"],
                ascending=[False, True, True],
            ).iloc[0].to_dict()
            winner.update(
                {
                    "Objective": objective,
                    "Tied best layouts": len(tied),
                    "Tie break": "largest_PP_then_smallest_TP_then_replica_devices",
                }
            )
            selected.append(winner)
    return pd.DataFrame(selected)


def build_equal_power_deployments(
    selected: pd.DataFrame,
    dgx: dict[tuple[str, str], dict[str, float | int]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, base in selected.iterrows():
        reference = dgx[(str(base["Model"]), str(base["Context"]))]
        replica_power_w = float(base["Replica power (W)"])
        exact_dp = float(reference["power"]) / replica_power_w
        local_rows: list[dict[str, object]] = []
        for rounding, dp in integer_dp_choices(float(reference["power"]), replica_power_w):
            row = base.to_dict()
            system_power_w = replica_power_w * dp
            system_throughput = float(base["Replica throughput (tokens/s)"]) * dp
            row.update(
                {
                    "DP rounding": rounding,
                    "Continuous ideal DP": exact_dp,
                    "DP": dp,
                    "Total devices": int(base["Replica devices"]) * dp,
                    "Resident requests in flight": int(base["Microbatch used"]) * dp,
                    "System throughput (tokens/s)": system_throughput,
                    "System power (W)": system_power_w,
                    "Power delta vs DGX (W)": system_power_w - float(reference["power"]),
                    "Absolute power delta vs DGX (W)": abs(
                        system_power_w - float(reference["power"])
                    ),
                    "Power / DGX H100": system_power_w / float(reference["power"]),
                    "Throughput / DGX H100": system_throughput
                    / float(reference["throughput"]),
                    "DGX H100 throughput (tokens/s)": float(reference["throughput"]),
                    "DGX H100 device-side power (W)": float(reference["power"]),
                    "DGX H100 batch": int(reference["batch"]),
                    "DGX profile model": str(reference["profile_model"]),
                }
            )
            local_rows.append(row)
        closest = min(float(row["Absolute power delta vs DGX (W)"]) for row in local_rows)
        for row in local_rows:
            row["Closest integer DP"] = math.isclose(
                float(row["Absolute power delta vs DGX (W)"]), closest, abs_tol=1e-12
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_selected_energy_breakdown(selected: pd.DataFrame) -> pd.DataFrame:
    """Return phase-level effective token energy for throughput-selected layouts."""

    rows = selected[selected["Objective"] == "throughput_per_device"].copy()
    rows["Trace active energy (mJ/token)"] = rows["Base active token energy (mJ)"]
    rows["Trace-external waiting energy (mJ/token)"] = rows[
        "Trace-external gap energy (mJ/token)"
    ]
    rows["Pipeline-bubble waiting energy (mJ/token)"] = (
        rows["Effective token energy (mJ)"] - rows["Work token energy (mJ)"]
    ).clip(lower=0.0)
    reconstructed = (
        rows["Trace active energy (mJ/token)"]
        + rows["Trace-external waiting energy (mJ/token)"]
        + rows["Pipeline-bubble waiting energy (mJ/token)"]
    )
    if not all(
        math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-8)
        for actual, expected in zip(rows["Effective token energy (mJ)"], reconstructed)
    ):
        raise ValueError("selected energy-breakdown reconstruction mismatch")
    return rows


def _workload_order(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["Model order"] = ordered["Model"].map({"Llama2-7B": 0, "Llama2-70B": 1})
    ordered["Context order"] = ordered["Context"].map({"4K": 0, "32K": 1, "128K": 2})
    return ordered.sort_values(["Model order", "Context order", "Memory"])


def write_plots(
    selected: pd.DataFrame,
    deployments: pd.DataFrame,
    energy_breakdown: pd.DataFrame,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    memories = list(MEMORY_CASES)
    colors = {
        "GDDR6": SEABORN_COLORBLIND[0],
        "LPDDR4X_nCCD2": SEABORN_COLORBLIND[1],
        "LPDDR4X_nCCD6": SEABORN_COLORBLIND[2],
    }
    labels = [f"{model.replace('Llama2-', '')}\n{context}" for model in MODEL_CONFIG for context in CONTEXTS]
    x = list(range(len(labels)))
    width = 0.24

    for objective, metric, ylabel, filename in (
        (
            "throughput_per_device",
            "Throughput / device (tokens/s/device)",
            "Tokens/s/device",
            "best_throughput_per_device",
        ),
        ("tokens_per_joule", "Tokens/J", "Tokens/J", "best_tokens_per_joule"),
    ):
        rows = _workload_order(selected[selected["Objective"] == objective])
        fig, ax = plt.subplots(figsize=(11.5, 4.8))
        for memory_index, memory in enumerate(memories):
            subset = rows[rows["Memory"] == memory]
            values = subset[metric].to_numpy()
            bars = ax.bar(
                [position + (memory_index - 1) * width for position in x],
                values,
                width=width,
                color=colors[memory],
                label=memory,
            )
            for bar, (_, row) in zip(bars, subset.iterrows()):
                ax.annotate(
                    f"{int(row['PP'])}/{int(row['TP'])}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.set_title("Best capacity-constrained PP/TP (bar label: PP/TP)")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, ncols=3)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(output_dir / f"{filename}.{suffix}", dpi=220, bbox_inches="tight")
        plt.close(fig)

    throughput_selected = _workload_order(
        selected[selected["Objective"] == "throughput_per_device"]
    )
    for metric, ylabel, filename in (
        ("Power / device (W)", "Power/device (W)", "selected_power_per_device"),
        (
            "Effective token energy (mJ)",
            "Effective energy/token (mJ)",
            "selected_token_energy",
        ),
    ):
        fig, ax = plt.subplots(figsize=(11.5, 4.8))
        for memory_index, memory in enumerate(memories):
            subset = throughput_selected[throughput_selected["Memory"] == memory]
            ax.bar(
                [position + (memory_index - 1) * width for position in x],
                subset[metric],
                width=width,
                color=colors[memory],
                label=memory,
            )
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.set_title("Metrics for the throughput/device-selected PP/TP layouts")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, ncols=3)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(output_dir / f"{filename}.{suffix}", dpi=220, bbox_inches="tight")
        plt.close(fig)

    breakdown_components = [
        "Trace active energy (mJ/token)",
        "Trace-external waiting energy (mJ/token)",
        "Pipeline-bubble waiting energy (mJ/token)",
    ]
    breakdown_colors = [
        SEABORN_COLORBLIND[0],
        SEABORN_COLORBLIND[1],
        SEABORN_COLORBLIND[4],
    ]
    fig, axes = plt.subplots(1, len(memories), figsize=(15, 4.8), sharey=False)
    ordered_breakdown = _workload_order(energy_breakdown)
    for ax, memory in zip(axes, memories):
        subset = ordered_breakdown[ordered_breakdown["Memory"] == memory]
        bottom = [0.0] * len(x)
        for component, color in zip(breakdown_components, breakdown_colors):
            values = subset[component].to_numpy()
            ax.bar(x, values, bottom=bottom, color=color, label=component.replace(" (mJ/token)", ""))
            bottom = [base + value for base, value in zip(bottom, values)]
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_title(memory)
        ax.set_ylabel("Effective energy/token (mJ)")
        ax.grid(axis="y", alpha=0.25)
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    fig.suptitle("Energy breakdown for throughput/device-selected layouts", y=0.99)
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncols=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"selected_throughput_energy_breakdown.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)

    closest = deployments[
        (deployments["Objective"] == "throughput_per_device")
        & deployments["Closest integer DP"]
    ]
    closest = _workload_order(closest)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=False)
    for ax, model in zip(axes, MODEL_CONFIG):
        model_rows = closest[closest["Model"] == model]
        positions = list(range(len(CONTEXTS)))
        for memory_index, memory in enumerate(memories):
            subset = model_rows[model_rows["Memory"] == memory]
            bars = ax.bar(
                [position + (memory_index - 1) * width for position in positions],
                subset["System throughput (tokens/s)"],
                width=width,
                color=colors[memory],
                label=memory,
            )
            for bar, (_, row) in zip(bars, subset.iterrows()):
                ax.annotate(
                    (
                        f"{float(row['Throughput / DGX H100']):.2f}×\n"
                        f"{int(row['PP'])}/{int(row['TP'])}/{int(row['DP'])}"
                    ),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    rotation=90,
                    linespacing=0.95,
                )
        reference = (
            model_rows.drop_duplicates("Context")
            .set_index("Context")
            .loc[list(CONTEXTS), "DGX H100 throughput (tokens/s)"]
        )
        ax.plot(positions, reference, "k--o", linewidth=1.4, markersize=4, label="DGX H100")
        ax.set_xticks(positions, list(CONTEXTS))
        ax.set_title(model)
        ax.set_ylabel("System throughput (tokens/s)")
        ax.grid(axis="y", alpha=0.25)
        tallest = max(
            float(model_rows["System throughput (tokens/s)"].max()),
            float(reference.max()),
        )
        ax.set_ylim(0.0, tallest * 1.32)
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    fig.suptitle(
        "Nearest-integer-DP device-side equal-power throughput\n"
        "bar label: throughput/DGX ×; PP/TP/DP",
        y=0.99,
    )
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncols=4,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"equal_power_system_throughput.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    rows = frame[columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in rows.itertuples(index=False, name=None):
        cells = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in values]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_readme(
    output_dir: Path,
    envelope: pd.DataFrame,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    deployments: pd.DataFrame,
    attention_mapping: str = "inter_device",
) -> None:
    admitted = candidates[candidates["Can host one request"]]
    closest = _workload_order(
        deployments[
            (deployments["Objective"] == "throughput_per_device")
            & deployments["Closest integer DP"]
        ]
    )
    summary = markdown_table(
        closest,
        [
            "Memory",
            "Model",
            "Context",
            "PP",
            "TP",
            "Pipeline fill ratio",
            "DP",
            "Total devices",
            "System throughput (tokens/s)",
            "System power (W)",
        ],
    )
    if attention_mapping == "kv_head":
        rejected = candidates[~candidates["Can host one request"]]
        underfilled = admitted[~admitted["Pipeline full"]]
        text = f"""# KV-head TP equal-power all-context analysis

This is the BF16, 16-GiB/device result for Llama2-7B and Llama2-70B at the
4K, 32K, and 128K midpoint decode samples. Query heads, KV heads, QKV/WO
weights, FFN weights, and the KV cache are tensor-sharded. Every rank performs
local QK, per-head Softmax, and SV; no score tensor is gathered.

TP is the fixed full grid `1,2,4,8`. PP is restricted to exact layer divisors.
Only `Bmax=0` is rejected; underfilled pipelines use `min(Bmax,PP)/PP`.
Equal objective values prefer the largest PP, then the smallest TP and replica.
There are {len(candidates)} memory-specific candidates, {len(admitted)} admitted,
{len(rejected)} rejected, and {len(underfilled)} admitted underfilled points.

Each block has two row-parallel gather/reduce/broadcast phases, after WO and
W2. KV-head TP ranks are compute-symmetric, so one fresh rank trace represents
every rank at a given model/context/TP point. The campaign contains 24
simulations per memory timing and 72 Ramulator jobs across GDDR6, LPDDR4X
nCCD2, and LPDDR4X nCCD6; the two LPDDR4X timings share the same 24 functional
traces, for 48 functional traces total.

## Closest equal-power deployments for the throughput/device objective

{summary}

## Modeling boundary

- One symmetric rank trace is replicated across all TP ranks. Link energy is
  still rank-specific because the gather root and non-root ranks transmit
  different shares of the two collectives.
- QK and SV co-issue all local KV heads and active V context groups on
  disjoint channel masks. Distinct score/query payloads retain separate
  channel-local `WR_GB` commands; identical row/op MACs use one union mask,
  followed by one union-mask blocking `RD_MAC` completion barrier.
- For V heads with more than 128 banks, channels are split into 128-bank
  groups. Each group stores all 128 head dimensions for one context slice,
  broadcasts only its local score slice through the unmodified shared GB, and
  uses `MAC_ABK`. The final cross-group 128-dimension element-wise add is
  analytical PNM postprocessing; K/V cache updates remain explicit `W MEM`
  trace commands. No `MAC_SBK` or bank-private GB is used.
- PP and DP are post-processing. Pipeline fill, PRE_STBY waiting power, and the
  nearest integer DP neighbors around the DGX H100 power target are retained.
- CXL traffic is two full-D gathers plus two full-D broadcasts per block,
  totaling `4*(TP-1)*D` BF16 link values.

## Files

- `sweep_envelope.csv` and `all_candidates.csv`: full PP/TP grid and capacity.
- `rank_throughput_per_device.csv` and `rank_tokens_per_joule.csv`: admitted rankings.
- `selected_base_configs.csv`: independent winners for both objectives.
- `selected_throughput_energy_breakdown.csv`: selected energy decomposition.
- `equal_power_deployments.csv`: DP floor/ceil systems around the DGX target.
"""
        (output_dir / "README.md").write_text(text)
        return
    counts = "7B: 4K=7, 32K=12, 128K=19; 70B: 4K=10, 32K=14, 128K=22"
    text = f"""# Capacity-constrained balanced PP/TP/DP analysis

This is the consolidated BF16, 16-GiB/device result for Llama2-7B and
Llama2-70B at 4K, 32K, and 128K. Attention is distributed across the TP group
and each request's KV cache is fully TP-sharded.

For every exact layer divisor PP, TP starts at 1 and doubles through the first
point where `Bmax > PP`. A point with `Bmax=0` is rejected. A point with
`1 <= Bmax < PP` remains valid and its ideal pipeline throughput is multiplied
by `Bmax/PP`. Thus partial pipeline fill is a performance penalty, not an
admission failure. The admitted candidate counts per memory case are {counts}.

Energy during main/helper traces comes from Cellar. The analysis adds the
precharged waiting floor during trace-external latency and on unoccupied
pipeline slots. Output columns intentionally expose one canonical energy and
power result rather than separate standby sensitivities.

Throughput/device and tokens/J are optimized independently. Ties prefer the
largest PP, then the smallest TP and replica. For each selected base replica, both
positive integer DP neighbors around the eight-GPU DGX device-power target are
kept; `Closest integer DP` marks the nearer one.

## Closest equal-power deployments for the throughput/device objective

{summary}

## Ramulator work

The 84 admitted PP/TP points collapse to model/context/TP source traces:

- 7B: 10 source bundles and 17 main/helper simulations per memory timing.
- 70B: 15 source bundles and 27 simulations per memory timing.
- Total: 44 simulations per timing, or 132 for GDDR6, LPDDR4X nCCD2, and
  LPDDR4X nCCD6. The two LPDDR4X timings reuse the same functional traces.

Reproduce from `cent_simulation/` with:

```bash
/home/linuswang/miniforge3/envs/cent/bin/python scripts/run_cent_memory_cases.py \\
  --models Llama2-7B,Llama2-70B --paper-long-context \\
  --context-windows 4096,32768,131072 --balanced-capacity-envelope \\
  --source-devices-by-model 'Llama2-7B=16;Llama2-70B=32' \\
  --pcie-lanes-by-model 'Llama2-7B=288;Llama2-70B=144' \\
  --skip-pipeline --cases GDDR6,LPDDR4X_nCCD2,LPDDR4X_nCCD6 \\
  --output-root output/balanced_equal_power_all_contexts/raw \\
  --trace-root trace/balanced_equal_power_all_contexts \\
  --trace-workers 1 --run-workers 4

/home/linuswang/miniforge3/envs/cent/bin/python \\
  scripts/analyze_balanced_equal_power.py
```

## Modeling boundary

- PP is restricted to exact divisors: 7B uses divisors of 32 and 70B uses
  divisors of 80, so all stages contain the same number of transformer blocks.
- One representative inter-device helper trace is multiplied by `TP-1`.
- The 7B source uses 16 devices only to include TP16; 288 total PCIe lanes keep
  the prior 18-lane/device source assumption. The 70B source remains 32 devices
  and 144 total lanes (4 lanes/device).
- PP and DP are post-processing. Per-block CXL latency is inherited from the
  model's source topology; additional PP-boundary traffic, switch/host idle
  power, PSU loss, and facility overhead are outside this result.

## Files

- `sweep_envelope.csv`: every evaluated TP point, including Bmax=0 rejects.
- `all_candidates.csv`: all memory-timing candidates and partial-fill power.
- `rank_throughput_per_device.csv` and `rank_tokens_per_joule.csv`: admitted rankings.
- `selected_base_configs.csv`: one deterministic winner per objective/workload.
- `selected_throughput_energy_breakdown.csv`: trace, external-wait, and pipeline-bubble
  energy for the throughput/device-selected layouts.
- `equal_power_deployments.csv`: DP floor/ceil systems around the DGX target.
- PNG/PDF figures show both objectives, power/device, energy/token, the energy
  breakdown, and equal-power throughput.
"""
    (output_dir / "README.md").write_text(text)


def main(
    *,
    default_experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
    default_attention_mapping: str = "inter_device",
    default_tp_values: tuple[int, ...] | None = None,
) -> int:
    args = parse_args(
        default_experiment_root=default_experiment_root,
        default_attention_mapping=default_attention_mapping,
        default_tp_values=default_tp_values,
    )
    if args.device_capacity_gib <= args.reserve_gib:
        raise ValueError("device capacity must exceed the per-device reserve")
    raw_root = args.raw_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.attention_mapping == "kv_head" and tuple(sorted(set(args.tp_values or []))) != (1, 2, 4, 8):
        raise ValueError("KV-head all-context analysis requires TP=1,2,4,8")
    envelope = sweep_envelope(args.device_capacity_gib, args.reserve_gib, args.tp_values)
    dgx = load_dgx(args.h100_profile.resolve())
    rows: list[dict[str, object]] = []
    source_manifest: dict[str, dict[str, str]] = {}
    for memory, memory_config in MEMORY_CASES.items():
        source_path = raw_root / Path(memory_config["csv"])
        source_manifest[memory] = {
            "path": display_path(source_path),
            "sha256": sha256(source_path),
        }
        for model in MODEL_CONFIG:
            for context in CONTEXTS:
                sources = load_tp_sources(
                    source_path,
                    model,
                    context,
                    required_tps(envelope, model, context),
                    args.dram_energy_model,
                    args.attention_mapping,
                )
                workload_envelope = envelope[
                    (envelope["Model"] == model) & (envelope["Context"] == context)
                ]
                for point in workload_envelope.itertuples(index=False):
                    pp = int(point.PP)
                    tp = int(point.TP)
                    rows.append(
                        build_candidate(
                            memory,
                            str(memory_config["dram_impl"]),
                            source_path,
                            sources[tp],
                            model,
                            context,
                            pp,
                            tp,
                            args.device_capacity_gib,
                            args.reserve_gib,
                            args.attention_mapping,
                        )
                    )

    candidates = assign_ranks(pd.DataFrame(rows)).sort_values(
        ["Model", "Context window", "Memory", "PP", "TP"]
    )
    admitted = candidates[candidates["Can host one request"]].copy()
    throughput_rank = admitted.sort_values(
        ["Model", "Context window", "Memory", "Throughput/device rank", "PP", "TP"],
        ascending=[True, True, True, True, False, True],
    )
    tokens_rank = admitted.sort_values(
        ["Model", "Context window", "Memory", "Tokens/J rank", "PP", "TP"],
        ascending=[True, True, True, True, False, True],
    )
    selected = choose_base_configs(candidates).sort_values(
        ["Model", "Context window", "Memory", "Objective"]
    )
    energy_breakdown = build_selected_energy_breakdown(selected).sort_values(
        ["Model", "Context window", "Memory"]
    )
    deployments = build_equal_power_deployments(selected, dgx).sort_values(
        ["Model", "Context window", "Memory", "Objective", "DP"]
    )

    outputs = {
        "sweep_envelope.csv": envelope,
        "all_candidates.csv": candidates,
        "rank_throughput_per_device.csv": throughput_rank,
        "rank_tokens_per_joule.csv": tokens_rank,
        "selected_base_configs.csv": selected,
        "selected_throughput_energy_breakdown.csv": energy_breakdown,
        "equal_power_deployments.csv": deployments,
    }
    for filename, dataframe in outputs.items():
        path = output_dir / filename
        dataframe.to_csv(path, index=False)
        print(f"[data] {display_path(path)}")

    manifest = {
        "precision": PRECISION,
        "device_capacity_gib": args.device_capacity_gib,
        "reserve_gib": args.reserve_gib,
        "attention_mapping": args.attention_mapping,
        "kv_mapping": "kv_head_sharded" if args.attention_mapping == "kv_head" else "fully_tp_sharded",
        "dram_energy_model": args.dram_energy_model,
        "models": MODEL_CONFIG,
        "contexts": CONTEXTS,
        "pp_values": {model: list(exact_pp_values(model)) for model in MODEL_CONFIG},
        "tp_values_by_workload": {
            f"{model}/{context}": list(required_tps(envelope, model, context))
            for model in MODEL_CONFIG
            for context in CONTEXTS
        },
        "sources": source_manifest,
        "dgx": {f"{model}/{context}": value for (model, context), value in dgx.items()},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_readme(
        output_dir,
        envelope,
        candidates,
        selected,
        deployments,
        args.attention_mapping,
    )

    if not args.no_plots:
        write_plots(selected, deployments, energy_breakdown, output_dir)
        print(f"[plots] {display_path(output_dir)}")

    summary_columns = [
        "Memory",
        "Model",
        "Context",
        "Objective",
        "PP",
        "TP",
        "Pipeline fill ratio",
        "DP",
        "Total devices",
        "System throughput (tokens/s)",
        "System power (W)",
        "Throughput / DGX H100",
    ]
    print(
        deployments[deployments["Closest integer DP"]][summary_columns].to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
