#!/usr/bin/env python3
"""Plot CENT token energy breakdowns from existing simulation logs."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CENT_SIM = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = CENT_SIM / "output"
sys.path.insert(0, str(CENT_SIM))

import cent_power_calculator as cent  # noqa: E402
from scripts.cent_capacity import select_capacity_constrained_best  # noqa: E402

n_heads = {"Llama2-7B": 32, "Llama2-70B": 64}
gqa_factor = {"Llama2-7B": 1, "Llama2-70B": 8}
embedding_size = {"Llama2-7B": 4096, "Llama2-70B": 8192}
ffn_size = {"Llama2-7B": 11008, "Llama2-70B": 28672}
TransformerBlock_number = {"Llama2-7B": 32, "Llama2-70B": 80}


BASE_RESULT_CSVS = {
    "GDDR6": {
        "csvs": [
            OUTPUT_ROOT / "GDDR6/simulation_results_decode_only_long_context_midpoint.csv",
        ],
        "log_root": OUTPUT_ROOT / "GDDR6/ramulator_long_context_midpoint",
    },
    "LPDDR4X_nCCD2": {
        "csvs": [
            OUTPUT_ROOT / "LPDDR4X/simulation_results_decode_only_long_context_midpoint_nCCD2.csv",
        ],
        "log_root": OUTPUT_ROOT / "LPDDR4X/ramulator_long_context_midpoint_nCCD2",
    },
    "LPDDR4X_nCCD6": {
        "csvs": [
            OUTPUT_ROOT / "LPDDR4X/simulation_results_decode_only_long_context_midpoint_nCCD6.csv",
        ],
        "log_root": OUTPUT_ROOT / "LPDDR4X/ramulator_long_context_midpoint_nCCD6",
    },
}

MEMORY_CASES = {
    "GDDR6": {
        "label": "GDDR6",
        "short_label": "G6",
        "csvs": BASE_RESULT_CSVS["GDDR6"]["csvs"],
        "log_root": BASE_RESULT_CSVS["GDDR6"]["log_root"],
        "dram_power_impl": "GDDR6",
        "nccd": "",
        "timing_source": "GDDR6_AiM",
    },
    "LPDDR4X_nCCD2": {
        "label": "LPDDR4X nCCD=2",
        "short_label": "X2",
        "csvs": BASE_RESULT_CSVS["LPDDR4X_nCCD2"]["csvs"],
        "log_root": BASE_RESULT_CSVS["LPDDR4X_nCCD2"]["log_root"],
        "dram_power_impl": "LPDDR4X",
        "nccd": "2",
        "timing_source": "LPDDR4_AiM timing nCCD=2 with LPDDR4X power",
    },
    "LPDDR4X_nCCD6": {
        "label": "LPDDR4X nCCD=6",
        "short_label": "X6",
        "csvs": BASE_RESULT_CSVS["LPDDR4X_nCCD6"]["csvs"],
        "log_root": BASE_RESULT_CSVS["LPDDR4X_nCCD6"]["log_root"],
        "dram_power_impl": "LPDDR4X",
        "nccd": "6",
        "timing_source": "LPDDR4_AiM timing nCCD=6 with LPDDR4X power",
    },
}

LEGACY_DRAM_COMPONENTS = ["ACT/PRE", "RD", "WR", "PIM", "ACT_STBY", "PRE_STBY"]
TRACE_BASED_DRAM_COMPONENTS = ["ACT", "PRE", "RD", "WR", "PIM", "ACT_STBY", "PRE_STBY"]


def component_groups(scope: str, dram_energy_model: str) -> dict[str, list[str]]:
    if dram_energy_model == "legacy":
        dram_components = LEGACY_DRAM_COMPONENTS
    elif dram_energy_model == "trace-based":
        dram_components = TRACE_BASED_DRAM_COMPONENTS
    else:
        raise ValueError(f"unknown DRAM energy model: {dram_energy_model}")
    if scope == "dram":
        return {component: [component] for component in dram_components}
    if scope != "system":
        raise ValueError(f"unknown energy scope: {scope}")
    return {
        **{component: [component] for component in dram_components},
        "DQ_IO": ["DQ"],
        "CTRL_PHY": ["MEM_CTR"],
        "SRAM_STT": ["GB_STT", "SB_STT", "IB_STT"],
        "ACCEL_STT": [
            "RED_STT", "EXP_STT", "VEC_ADD_STT", "VEC_MUL_STT", "CTR_STT", "TOPK_STT"
        ],
        "SRAM_DYN": ["GB_RD", "GB_WR", "SB_DYN", "IB_DYN"],
        "ACCEL_DYN": ["RV_DYN", "RED_DYN", "EXP_DYN", "VEC_ADD_DYN", "VEC_MUL_DYN", "DV_CTR"],
        "PCIe": ["PCIe"],
    }


SYSTEM_GROUPS = {
    **{component: [component] for component in LEGACY_DRAM_COMPONENTS},
    "DQ_IO": ["DQ"],
    "CTRL_PHY": ["MEM_CTR"],
    "SRAM_STT": ["GB_STT", "SB_STT", "IB_STT"],
    "ACCEL_STT": [
        "RED_STT", "EXP_STT", "VEC_ADD_STT", "VEC_MUL_STT", "CTR_STT", "TOPK_STT"
    ],
    "SRAM_DYN": ["GB_RD", "GB_WR", "SB_DYN", "IB_DYN"],
    "ACCEL_DYN": ["RV_DYN", "RED_DYN", "EXP_DYN", "VEC_ADD_DYN", "VEC_MUL_DYN", "DV_CTR"],
    "PCIe": ["PCIe"],
}


def parse_csv_list(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("comma-separated list must not be empty")
    return values


def parse_int_list(raw: str) -> list[int]:
    return [int(value) for value in parse_csv_list(raw)]


def parse_context_specs(raw: str) -> list[tuple[str, int]]:
    specs: list[tuple[str, int]] = []
    for part in parse_csv_list(raw):
        if "=" in part:
            label, value = part.split("=", 1)
            specs.append((label.strip(), int(value.strip())))
        else:
            value = int(part)
            specs.append((f"{value // 1024}K", value))
    return specs


def set_cent_channel_count(ch_per_dv: int) -> None:
    cent.set_channel_count(ch_per_dv)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def power_column(component: str) -> str:
    return f"{component.replace('/', '_')}_W"


def per_device_power_column(component: str) -> str:
    return f"{component.replace('/', '_')}_per_device_W"


def latency_normalized_power_column(component: str) -> str:
    return f"{component.replace('/', '_')}_latency_normalized_W"


def parse_path_list(raw: str | None, defaults: list[Path]) -> list[Path]:
    if raw is None:
        return defaults
    return [Path(part) for part in parse_csv_list(raw)]


def load_results(case_name: str, csv_paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    missing = [path for path in csv_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing simulation CSV(s) for {case_name}: {', '.join(str(path) for path in missing)}")
    for csv_path in csv_paths:
        with csv_path.open(newline="") as fh:
            loaded = list(csv.DictReader(fh))
        for row in loaded:
            row["_source_csv"] = display_path(csv_path)
        rows.extend(loaded)
    return rows


def row_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def row_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def row_context_window(row: dict[str, str]) -> int:
    raw = row.get("Context window", "")
    return int(float(raw)) if raw else row_int(row, "Sequence length")


def row_mode(row: dict[str, str], log_root: Path) -> str:
    model = str(row["Model"])
    seqlen = row_int(row, "Sequence length")
    tp = row_int(row, "Tensor parallelism")
    channels_per_block = row_int(row, "Channels per block")

    model_log = log_root / f"model_parallel/{model}/trace_{tp}_FC_devices_seqlen_{seqlen}.txt.log"
    pipeline_log = log_root / f"pipeline_parallel/{model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt.log"
    device_number = row_int(row, "Device number")
    pp = row_int(row, "Pipeline parallelism")
    if pp * tp == device_number and model_log.exists():
        return "model_parallel"
    if pipeline_log.exists():
        return "pipeline_parallel"
    if model_log.exists():
        return "model_parallel"
    raise FileNotFoundError(f"could not infer trace log for row: {model} seqlen={seqlen}, tp={tp}, ch/block={channels_per_block}")


def select_row(
    rows: list[dict[str, str]],
    model: str,
    seqlen: int,
    mode: str,
    dram_energy_model: str,
    device_capacity_gib: float,
    per_device_reserve_gib: float,
    shard_kv_cache_across_tp: bool,
) -> dict[str, str]:
    subset = [
        row for row in rows
        if row["Model"] == model and row_context_window(row) == seqlen
    ]
    if not subset:
        raise ValueError(f"no simulation row for {model} seqlen={seqlen}")
    if any("DRAM energy model" in row for row in subset):
        subset = [row for row in subset if row.get("DRAM energy model", "legacy") == dram_energy_model]
    if not subset:
        raise ValueError(f"no {dram_energy_model} simulation row for {model} seqlen={seqlen}")

    expected_mapping = "inter_device" if shard_kv_cache_across_tp else "master"
    explicit_mappings = {
        row.get("Attention mapping", "")
        for row in subset
        if row.get("Attention mapping", "") in {"inter_device", "master"}
    }
    if explicit_mappings:
        subset = [row for row in subset if row.get("Attention mapping") == expected_mapping]
        if not subset:
            raise ValueError(
                f"no {expected_mapping} attention row for {model} seqlen={seqlen}; "
                f"available mappings: {', '.join(sorted(explicit_mappings))}"
            )

    if mode == "capacity_constrained":
        return select_capacity_constrained_best(
            pd.DataFrame(subset),
            model,
            seqlen,
            device_capacity_gib=device_capacity_gib,
            per_device_reserve_gib=per_device_reserve_gib,
            shard_kv_cache_across_tp=shard_kv_cache_across_tp,
        )
    if mode == "model_parallel":
        subset = sorted(subset, key=lambda row: (row_float(row, "Tensor parallelism"), row_float(row, "Throughput (tokens/s)")), reverse=True)
    elif mode == "pipeline_parallel":
        subset = [row for row in subset if row_int(row, "Tensor parallelism") == 1]
        subset = sorted(subset, key=lambda row: (row_float(row, "Pipeline parallelism"), row_float(row, "Throughput (tokens/s)")), reverse=True)
    elif mode == "max_throughput":
        subset = sorted(subset, key=lambda row: row_float(row, "Throughput (tokens/s)"), reverse=True)
    else:
        raise ValueError(f"unknown mode: {mode}")

    if not subset:
        raise ValueError(f"no {mode} simulation row for {model} seqlen={seqlen}")
    return subset[0]


def trace_logs(row: dict[str, str], mode: str, log_root: Path) -> tuple[Path, Path | None, str]:
    model = str(row["Model"])
    seqlen = row_int(row, "Sequence length")
    tp = row_int(row, "Tensor parallelism")
    channels_per_block = row_int(row, "Channels per block")
    selected_mode = row_mode(row, log_root) if mode in {"max_throughput", "capacity_constrained"} else mode

    if selected_mode == "model_parallel":
        main = log_root / f"model_parallel/{model}/trace_{tp}_FC_devices_seqlen_{seqlen}.txt.log"
        if tp == 1:
            return main, None, selected_mode
        helper_mode = (
            "model_parallel_helper_attention"
            if row.get("Attention mapping") == "inter_device"
            else "model_parallel_FC"
        )
        helper = log_root / f"{helper_mode}/{model}/trace_{tp}_FC_devices_seqlen_{seqlen}.txt.log"
        return main, helper, selected_mode
    if selected_mode == "pipeline_parallel":
        main = log_root / f"pipeline_parallel/{model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt.log"
        return main, None, selected_mode
    raise ValueError(f"unknown selected mode: {selected_mode}")


def require_log(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty log: {path}")


def calculate_component_energy(
    row: dict[str, str],
    mode: str,
    log_root: Path,
    dram_power_impl: str,
    dram_energy_model: str,
) -> tuple[dict[str, float], dict[str, float], str, Path, Path | None]:
    model = str(row["Model"])
    seqlen = row_int(row, "Sequence length")
    ch_per_dv = row_int(row, "Channels per device")
    set_cent_channel_count(ch_per_dv)

    main_log, helper_log, selected_mode = trace_logs(row, mode, log_root)
    require_log(main_log)
    stat_main = cent.command_processor(str(main_log))

    pcie_bits = embedding_size[model]
    if selected_mode == "model_parallel":
        pcie_bits = embedding_size[model] * 10 + ffn_size[model] * 2

    energy_main, latency = cent.power_calculator(
        stat_main,
        pcie_bits,
        n_heads[model],
        embedding_size[model],
        seqlen,
        gqa_factor[model],
        dram_power_impl=dram_power_impl,
        dram_energy_model=dram_energy_model,
        command_trace_prefix=cent.command_trace_prefix_for_log(main_log),
    )

    if selected_mode == "model_parallel":
        tp = row_int(row, "Tensor parallelism")
        if tp == 1:
            energy_token = {
                comp: energy_main[comp] * TransformerBlock_number[model]
                for comp in energy_main
            }
        else:
            if helper_log is None:
                raise ValueError("TP>1 model_parallel mode requires a helper log")
            require_log(helper_log)
            stat_helper = cent.command_processor(str(helper_log))
            inter_device = row.get("Attention mapping") == "inter_device"
            energy_helper, _latency_helper = cent.power_calculator(
                stat_helper,
                pcie_bits,
                n_heads[model],
                embedding_size[model],
                seqlen,
                gqa_factor[model],
                dram_power_impl=dram_power_impl,
                dram_energy_model=dram_energy_model,
                command_trace_prefix=cent.command_trace_prefix_for_log(helper_log),
                device_role=("inter_device_helper" if inter_device else "fc_helper"),
            )
            energy_token = {
                comp: (energy_main[comp] + energy_helper[comp] * (tp - 1)) * TransformerBlock_number[model]
                for comp in energy_main
            }
    else:
        utilized_devices = row_float(row, "Device utilization") * row_float(row, "Device number")
        energy_token = {comp: energy_main[comp] * utilized_devices for comp in energy_main}

    return energy_token, stat_main, selected_mode, main_log, helper_log


def append_row(
    rows: list[dict[str, str]],
    case_name: str,
    case_config: dict[str, object],
    model: str,
    context_label: str,
    seqlen: int,
    selected: dict[str, str],
    selected_mode: str,
    dram_energy_model: str,
    energy: dict[str, float],
    stat: dict[str, float],
    groups: dict[str, list[str]],
    main_log: Path,
    helper_log: Path | None,
) -> None:
    component_energy = {
        component: sum(energy.get(source, 0.00) for source in sources)
        for component, sources in groups.items()
    }
    total_mj = sum(component_energy.values())
    token_latency_ms = row_float(selected, "Token latency (ms)")
    csv_energy = row_float(selected, "Token energy (mJ)")
    device_count = row_int(selected, "Device number")
    utilized_device_count = row_float(selected, "Device utilization") * device_count
    trace_throughput_tokens_s = row_float(selected, "Throughput (tokens/s)")
    throughput_tokens_s = float(selected.get("Capacity-constrained throughput (tokens/s)", trace_throughput_tokens_s))
    # A filled pipeline processes multiple tokens concurrently. Convert the
    # full-model energy per output token into the steady-state system power.
    total_w = total_mj * throughput_tokens_s / 1000.0
    latency_normalized_total_w = total_mj / token_latency_ms
    row = {
        "case": case_name,
        "memory": str(case_config["label"]),
        "nccd": str(case_config["nccd"]),
        "model": model,
        "context": context_label,
        "context_window": str(seqlen),
        "active_sequence_length": str(row_int(selected, "Sequence length")),
        "mode": selected_mode,
        "dram_energy_model": dram_energy_model,
        "timing_source": str(case_config["timing_source"]),
        "source_csv": selected.get("_source_csv", ""),
        "pipeline_parallelism": f"{row_float(selected, 'Pipeline parallelism'):.12g}",
        "tensor_parallelism": f"{row_float(selected, 'Tensor parallelism'):.12g}",
        "channels_per_device": f"{row_float(selected, 'Channels per device'):.12g}",
        "banks_per_device": f"{row_float(selected, 'Banks per device'):.12g}" if "Banks per device" in selected else "",
        "channels_per_block": f"{row_float(selected, 'Channels per block'):.12g}",
        "device_count": str(device_count),
        "utilized_device_count": f"{utilized_device_count:.12g}",
        "memory_system_cycles": f"{stat['cycles']:.0f}",
        "pim_latency_ms": f"{stat['latency']:.12g}",
        "token_latency_ms": f"{token_latency_ms:.12g}",
        "trace_throughput_tokens_s": f"{trace_throughput_tokens_s:.12g}",
        "throughput_tokens_s": f"{throughput_tokens_s:.12g}",
        "max_resident_microbatch": str(selected.get("Max resident microbatch", "")),
        "microbatch_used_for_throughput": str(selected.get("Microbatch used for throughput", "")),
        "pipeline_fill_ratio": f"{float(selected.get('Pipeline fill ratio', 1.0)):.12g}",
        "csv_token_energy_mJ": f"{csv_energy:.12g}",
        "total_mJ": f"{total_mj:.12g}",
        "energy_delta_pct": f"{((total_mj - csv_energy) / csv_energy * 100.0) if csv_energy else 0.0:.12g}",
        "total_W": f"{total_w:.12g}",
        "latency_normalized_total_W": f"{latency_normalized_total_w:.12g}",
        "average_provisioned_device_W": f"{total_w / device_count:.12g}",
        "average_active_device_W": f"{total_w / utilized_device_count:.12g}",
        "main_log": display_path(main_log),
        "helper_log": "" if helper_log is None else display_path(helper_log),
        # Retain the historical column for downstream CSV compatibility.
        "fc_log": "" if helper_log is None else display_path(helper_log),
    }
    for component, value in component_energy.items():
        row[component] = f"{value:.12g}"
        component_w = value * throughput_tokens_s / 1000.0
        row[power_column(component)] = f"{component_w:.12g}"
        row[latency_normalized_power_column(component)] = f"{value / token_latency_ms:.12g}"
        row[per_device_power_column(component)] = f"{component_w / device_count:.12g}"
    rows.append(row)


def csv_paths_for_case(args: argparse.Namespace, case_name: str, config: dict[str, object]) -> list[Path]:
    overrides = {
        "GDDR6": args.gddr6_csvs,
        "LPDDR4X_nCCD2": args.lpddr4x_nccd2_csvs,
        "LPDDR4X_nCCD6": args.lpddr4x_nccd6_csvs,
    }
    return parse_path_list(overrides.get(case_name), list(config["csvs"]))


def log_root_for_case(args: argparse.Namespace, case_name: str, config: dict[str, object]) -> Path:
    overrides = {
        "GDDR6": args.gddr6_log_root,
        "LPDDR4X_nCCD2": args.lpddr4x_nccd2_log_root,
        "LPDDR4X_nCCD6": args.lpddr4x_nccd6_log_root,
    }
    return overrides.get(case_name) or Path(config["log_root"])


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, str]], dict[str, list[str]], list[str]]:
    groups = component_groups(args.energy_scope, args.dram_energy_model)
    models = parse_csv_list(args.models)
    contexts = parse_context_specs(args.contexts)
    case_names = parse_csv_list(args.cases)
    rows: list[dict[str, str]] = []
    warnings: list[str] = []

    for case_name in case_names:
        if case_name not in MEMORY_CASES:
            raise ValueError(f"unknown case '{case_name}'. Expected one of: {', '.join(MEMORY_CASES)}")
        config = MEMORY_CASES[case_name]
        df = load_results(case_name, csv_paths_for_case(args, case_name, config))
        log_root = log_root_for_case(args, case_name, config)
        dram_power_impl = str(config["dram_power_impl"])

        for model in models:
            for context_label, seqlen in contexts:
                try:
                    selected = select_row(
                        df,
                        model,
                        seqlen,
                        args.mode,
                        args.dram_energy_model,
                        args.device_capacity_gib,
                        args.per_device_reserve_gib,
                        not args.master_attention,
                    )
                    energy, stat, selected_mode, main_log, fc_log = calculate_component_energy(
                        selected,
                        args.mode,
                        log_root,
                        dram_power_impl,
                        args.dram_energy_model,
                    )
                    append_row(
                        rows,
                        case_name,
                        config,
                        model,
                        context_label,
                        seqlen,
                        selected,
                        selected_mode,
                        args.dram_energy_model,
                        energy,
                        stat,
                        groups,
                        main_log,
                        fc_log,
                    )
                except (FileNotFoundError, ValueError) as exc:
                    message = f"skipped {case_name} {model} {context_label} (seqlen={seqlen}): {exc}"
                    if args.strict:
                        raise type(exc)(message) from exc
                    warnings.append(message)

    return rows, groups, warnings


def write_csv(rows: list[dict[str, str]], csv_path: Path, groups: dict[str, list[str]]) -> None:
    components = list(groups)
    fieldnames = [
        "case",
        "memory",
        "nccd",
        "model",
        "context",
        "context_window",
        "active_sequence_length",
        "mode",
        "dram_energy_model",
        "timing_source",
        "source_csv",
        "pipeline_parallelism",
        "tensor_parallelism",
        "channels_per_device",
        "banks_per_device",
        "channels_per_block",
        "device_count",
        "utilized_device_count",
        "memory_system_cycles",
        "pim_latency_ms",
        "token_latency_ms",
        "trace_throughput_tokens_s",
        "throughput_tokens_s",
        "max_resident_microbatch",
        "microbatch_used_for_throughput",
        "pipeline_fill_ratio",
        "csv_token_energy_mJ",
        *components,
        "total_mJ",
        *(power_column(component) for component in components),
        "total_W",
        *(latency_normalized_power_column(component) for component in components),
        "latency_normalized_total_W",
        *(per_device_power_column(component) for component in components),
        "average_provisioned_device_W",
        "average_active_device_W",
        "energy_delta_pct",
        "main_log",
        "helper_log",
        "fc_log",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def configure_matplotlib_cache() -> None:
    cache_dir = Path("/tmp/matplotlib-cent")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


def colors() -> dict[str, str]:
    return {
        "ACT/PRE": "#E45756",
        "ACT": "#E45756",
        "PRE": "#FF9DA6",
        "RD": "#4C78A8",
        "WR": "#F58518",
        "PIM": "#54A24B",
        "ACT_STBY": "#72B7B2",
        "PRE_STBY": "#B279A2",
        "DQ_IO": "#FF9DA6",
        "CTRL_PHY": "#9C755F",
        "SRAM_STT": "#8CD17D",
        "ACCEL_STT": "#499894",
        "SRAM_DYN": "#59A14F",
        "ACCEL_DYN": "#AF7AA1",
        "PCIe": "#BAB0AC",
    }


def row_sort_key(row: dict[str, str]) -> tuple[int, int, int]:
    model_order = {"Llama2-7B": 0, "Llama2-70B": 1}
    case_order = {case_name: idx for idx, case_name in enumerate(MEMORY_CASES)}
    return (model_order.get(row["model"], 99), int(row["context_window"]), case_order.get(row["case"], 99))


def write_stacked_plot(
    rows: list[dict[str, str]],
    plot_path: Path,
    groups: dict[str, list[str]],
    metric: str,
    energy_scope: str,
) -> None:
    configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(rows, key=row_sort_key)
    workloads = sorted(
        {(row["model"], row["context"], row["context_window"]) for row in ordered},
        key=lambda item: ({"Llama2-7B": 0, "Llama2-70B": 1}.get(item[0], 99), int(item[2])),
    )
    case_order = {case_name: idx for idx, case_name in enumerate(MEMORY_CASES)}
    cases = sorted({row["case"] for row in ordered}, key=lambda case: case_order.get(case, 99))
    rows_by_key = {(row["model"], row["context"], row["case"]): row for row in ordered}

    bar_spacing = 0.46
    group_stride = max(1.85, len(cases) * bar_spacing + 0.55)
    x_positions = []
    x_labels = []
    bar_rows = []
    for idx, (model, context, _sequence_length) in enumerate(workloads):
        base = idx * group_stride
        for case_idx, case in enumerate(cases):
            key = (model, context, case)
            if key not in rows_by_key:
                continue
            x_positions.append(base + (case_idx - (len(cases) - 1) / 2) * bar_spacing)
            x_labels.append(str(MEMORY_CASES[case]["short_label"]))
            bar_rows.append(rows_by_key[key])

    width = max(11.5, 0.95 * len(workloads) + 3.0)
    fig, ax = plt.subplots(figsize=(width, 6.2))
    palette = colors()
    bottoms = [0.0] * len(bar_rows)
    for component in groups:
        if metric == "energy":
            col = component
        elif metric == "power":
            col = power_column(component)
        elif metric == "per_device_power":
            col = per_device_power_column(component)
        else:
            raise ValueError(f"unknown metric: {metric}")
        values = [float(row[col]) for row in bar_rows]
        ax.bar(
            x_positions,
            values,
            bottom=bottoms,
            label=component,
            color=palette[component],
            width=0.42,
            edgecolor="white",
            linewidth=0.5,
        )
        bottoms = [base + value for base, value in zip(bottoms, values)]

    group_positions = [idx * group_stride for idx in range(len(workloads))]
    group_labels = [f"{model.replace('Llama2-', '')}\n{context}" for model, context, _sequence_length in workloads]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    for x, label in zip(group_positions, group_labels):
        ax.text(x, -0.12, label, ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=9)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_xlabel("case / model-context (X2/X6 = LPDDR4X nCCD 2/6)")
    ax.xaxis.set_label_coords(0.5, -0.20)

    if metric == "energy":
        ylabel = "Token energy (mJ)"
        title_metric = "Energy"
    elif metric == "power":
        ylabel = "Steady-state system power (W)"
        title_metric = "Steady-State Power"
    else:
        ylabel = "Average power per provisioned device (W)"
        title_metric = "Per-Device Power"
    title_scope = "DRAM" if energy_scope == "dram" else "System"
    ax.set_ylabel(ylabel)
    fig.suptitle(f"CENT {title_scope} {title_metric} Breakdown", y=0.98, fontsize=13, fontweight="semibold")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_cols = min(len(groups), 6 if len(groups) <= 6 else 4)
    legend_rows = max(1, math.ceil(len(groups) / legend_cols))
    fig.legend(
        *ax.get_legend_handles_labels(),
        ncols=legend_cols,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        frameon=False,
        fontsize=9,
    )
    fig.subplots_adjust(bottom=0.22, top=max(0.70, 0.86 - 0.045 * (legend_rows - 1)))
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="Llama2-7B,Llama2-70B")
    parser.add_argument("--contexts", default="4K=4096,32K=32768,128K=131072")
    parser.add_argument("--cases", default="GDDR6,LPDDR4X_nCCD2,LPDDR4X_nCCD6")
    parser.add_argument(
        "--mode",
        choices=["model_parallel", "pipeline_parallel", "max_throughput", "capacity_constrained"],
        default="capacity_constrained",
    )
    parser.add_argument("--device-capacity-gib", type=float, default=16.0)
    parser.add_argument("--per-device-reserve-gib", type=float, default=0.0)
    parser.add_argument(
        "--master-attention",
        action="store_true",
        help="Use master-local KV-cache capacity instead of TP-sharded KV-cache capacity.",
    )
    parser.add_argument("--energy-scope", choices=["dram", "system"], default="system")
    parser.add_argument("--dram-energy-model", choices=cent.DRAM_ENERGY_MODELS, default="legacy")
    parser.add_argument("--gddr6-csvs", help="Comma-separated GDDR6 simulation CSV paths.")
    parser.add_argument("--lpddr4x-nccd2-csvs", help="Comma-separated LPDDR4X nCCD=2 simulation CSV paths.")
    parser.add_argument("--lpddr4x-nccd6-csvs", help="Comma-separated LPDDR4X nCCD=6 simulation CSV paths.")
    parser.add_argument("--gddr6-log-root", "--gddr6-trace-root", dest="gddr6_log_root", type=Path)
    parser.add_argument("--lpddr4x-nccd2-log-root", "--lpddr4x-nccd2-trace-root", dest="lpddr4x_nccd2_log_root", type=Path)
    parser.add_argument("--lpddr4x-nccd6-log-root", "--lpddr4x-nccd6-trace-root", dest="lpddr4x_nccd6_log_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "figures/cent_energy_breakdown")
    parser.add_argument("--strict", action="store_true", help="Fail when a requested model/context/case is missing instead of skipping it with a warning.")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    rows, groups, warnings = build_rows(args)
    if not rows:
        raise RuntimeError("no rows were generated; check requested models, contexts, cases, CSVs, and logs")

    model_suffix = args.dram_energy_model.replace("-", "_")
    csv_path = args.output_dir / f"cent_{args.mode}_{args.energy_scope}_{model_suffix}_energy_breakdown.csv"
    write_csv(rows, csv_path, groups)
    print(f"[csv] {display_path(csv_path)}")
    for warning in warnings:
        print(f"[warn] {warning}")

    if not args.no_plot:
        energy_plot = args.output_dir / f"cent_{args.mode}_{args.energy_scope}_{model_suffix}_energy_breakdown_mj.png"
        power_plot = args.output_dir / f"cent_{args.mode}_{args.energy_scope}_{model_suffix}_power_breakdown_w.png"
        per_device_power_plot = args.output_dir / f"cent_{args.mode}_{args.energy_scope}_{model_suffix}_per_device_power_breakdown_w.png"
        try:
            write_stacked_plot(rows, energy_plot, groups, "energy", args.energy_scope)
            write_stacked_plot(rows, power_plot, groups, "power", args.energy_scope)
            write_stacked_plot(rows, per_device_power_plot, groups, "per_device_power", args.energy_scope)
            print(f"[plot] {display_path(energy_plot)}")
            print(f"[plot] {display_path(power_plot)}")
            print(f"[plot] {display_path(per_device_power_plot)}")
        except ModuleNotFoundError as exc:
            if exc.name != "matplotlib":
                raise
            print("[plot skipped] matplotlib is not installed for this Python; rerun in the cent env or pass --no-plot")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
