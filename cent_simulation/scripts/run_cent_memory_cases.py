#!/usr/bin/env python3
"""Generate CENT GDDR6 and LPDDR4X nCCD result CSVs."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CENT_SIM = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = CENT_SIM / "output"
TRACE_ROOT = CENT_SIM / "trace"

NCCD_RE = re.compile(r"^\s*nCCD:\s*(\d+)\s*$", re.MULTILINE)
MODEL_DEVICES = {
    "Llama2-7B": 8,
    "Llama2-70B": 32,
}
MODEL_PCIE_LANES = {
    "Llama2-7B": 144,
    "Llama2-70B": 144,
}
BALANCED_ENVELOPE_SOURCE_DEVICES = {
    "Llama2-7B": 16,
    "Llama2-70B": 32,
}
# The legacy 7B source used 144 total lanes for 8 devices (18 lanes/device).
# Use 288 lanes with the 16-device envelope source to preserve that ratio.
BALANCED_ENVELOPE_PCIE_LANES = {
    "Llama2-7B": 288,
    "Llama2-70B": 144,
}
BALANCED_ENVELOPE_CAPACITY_GIB = 16.0
BALANCED_ENVELOPE_LAYERS = {
    "Llama2-7B": 32,
    "Llama2-70B": 80,
}
DEFAULT_LONG_CONTEXTS = "32768,131072"
PAPER_LONG_CONTEXT_VARIANT = "long_context_midpoint"


def parse_csv_list(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("comma-separated list must not be empty")
    return values


def parse_int_list(raw: str) -> list[int]:
    return [int(value) for value in parse_csv_list(raw)]


def _parse_model_assignments(raw: str) -> list[tuple[str, str]]:
    assignments: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        model, separator, value = item.partition("=")
        model = model.strip()
        value = value.strip()
        if not separator or not model or not value:
            raise argparse.ArgumentTypeError(
                "model maps use semicolon-separated MODEL=VALUE assignments"
            )
        if model in seen:
            raise argparse.ArgumentTypeError(f"duplicate model assignment: {model}")
        seen.add(model)
        assignments.append((model, value))
    if not assignments:
        raise argparse.ArgumentTypeError("model map must not be empty")
    return assignments


def parse_model_int_map(raw: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for model, value in _parse_model_assignments(raw):
        try:
            parsed = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"{model} requires one integer, got: {value}") from error
        if parsed <= 0:
            raise argparse.ArgumentTypeError(f"{model} requires a positive integer, got: {parsed}")
        result[model] = parsed
    return result


def parse_model_int_list_map(raw: str) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for model, value in _parse_model_assignments(raw):
        try:
            parsed = sorted(set(parse_int_list(value)))
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"{model} requires comma-separated integers, got: {value}"
            ) from error
        if any(item <= 0 for item in parsed):
            raise argparse.ArgumentTypeError(f"{model} TP values must all be positive: {parsed}")
        result[model] = parsed
    return result


def parse_workload_int_list_map(raw: str) -> dict[tuple[str, int], list[int]]:
    result: dict[tuple[str, int], list[int]] = {}
    for workload, value in _parse_model_assignments(raw):
        model, separator, context = workload.partition("@")
        if not separator or not model.strip() or not context.strip():
            raise argparse.ArgumentTypeError(
                "workload maps use MODEL@CONTEXT=TP,TP assignments"
            )
        try:
            context_window = int(context)
            parsed = sorted(set(parse_int_list(value)))
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"invalid workload assignment: {workload}={value}"
            ) from error
        if context_window <= 0 or any(tp <= 0 for tp in parsed):
            raise argparse.ArgumentTypeError(
                f"context and TP values must be positive: {workload}={value}"
            )
        result[(model.strip(), context_window)] = parsed
    return result


def factors(value: int) -> list[int]:
    return [candidate for candidate in range(1, value + 1) if value % candidate == 0]


def balanced_envelope_tp_values(
    model: str,
    context_window: int,
    source_devices: int,
    *,
    device_capacity_gib: float = BALANCED_ENVELOPE_CAPACITY_GIB,
) -> list[int]:
    """Return the TP union from first resident request through first Bmax > PP."""

    # Keep pandas-backed capacity analysis lazy so legacy runner modes and
    # ``--help`` retain their original lightweight import behavior.
    from cent_capacity import capacity_for_layout

    pp_values = factors(BALANCED_ENVELOPE_LAYERS[model])
    tp_values = factors(source_devices)
    selected: set[int] = set()
    for pp in pp_values:
        capacities = [
            int(
                capacity_for_layout(
                    model,
                    pp,
                    tp,
                    context_window,
                    int(device_capacity_gib * 2**30),
                    0,
                    shard_kv_cache_across_tp=True,
                )["Max resident microbatch"]
            )
            for tp in tp_values
        ]
        headroom_indices = [index for index, capacity in enumerate(capacities) if capacity > pp]
        if not headroom_indices:
            continue
        first_headroom = min(headroom_indices)
        # Bmax=0 points are evaluated and rejected by post-processing; they do
        # not change the rule that every PP sweep starts at TP=1.
        selected.update(tp_values[: first_headroom + 1])
    if not selected:
        raise ValueError(
            f"no capacity-feasible TP envelope for {model} context={context_window} "
            f"with {source_devices} source devices"
        )
    return sorted(selected)


def resolve_model_run_config(
    models: list[str],
    *,
    global_tp_values: list[int] | None,
    tp_values_by_model: dict[str, list[int]] | None,
    source_devices_by_model: dict[str, int] | None,
    pcie_lanes_by_model: dict[str, int] | None,
) -> dict[str, dict[str, int | list[int] | None]]:
    known_models = set(MODEL_DEVICES)
    supplied_maps = {
        "--tp-values-by-model": tp_values_by_model,
        "--source-devices-by-model": source_devices_by_model,
        "--pcie-lanes-by-model": pcie_lanes_by_model,
    }
    for option, values in supplied_maps.items():
        unknown = sorted(set(values or {}) - known_models)
        if unknown:
            raise ValueError(f"{option} contains unsupported model(s): {', '.join(unknown)}")

    if tp_values_by_model is not None:
        missing = sorted(set(models) - set(tp_values_by_model))
        if missing:
            raise ValueError(
                "--tp-values-by-model must specify every selected model; missing: "
                + ", ".join(missing)
            )

    device_overrides = source_devices_by_model or {}
    lane_overrides = pcie_lanes_by_model or {}
    result: dict[str, dict[str, int | list[int] | None]] = {}
    for model in models:
        devices = int(device_overrides.get(model, MODEL_DEVICES[model]))
        lanes = int(lane_overrides.get(model, MODEL_PCIE_LANES[model]))
        tp_values = (
            list(tp_values_by_model[model])
            if tp_values_by_model is not None
            else (list(global_tp_values) if global_tp_values is not None else None)
        )
        invalid_tp = [tp for tp in (tp_values or []) if tp <= 0 or devices % tp]
        if invalid_tp:
            raise ValueError(
                f"TP values must divide source devices for {model}: "
                f"devices={devices}, invalid={invalid_tp}"
            )
        result[model] = {
            "source_devices": devices,
            "pcie_lanes": lanes,
            "tp_values": tp_values,
        }
    return result


def resolve_workload_tp_values(
    models: list[str],
    context_windows: list[int],
    model_run_configs: dict[str, dict[str, int | list[int] | None]],
    *,
    tp_values_by_workload: dict[tuple[str, int], list[int]] | None,
    balanced_envelope: bool,
) -> dict[tuple[str, int], list[int] | None]:
    selected_workloads = {(model, context) for model in models for context in context_windows}
    if tp_values_by_workload is not None:
        unknown_models = sorted(
            {model for model, _ in tp_values_by_workload} - set(MODEL_DEVICES)
        )
        if unknown_models:
            raise ValueError(
                "--tp-values-by-workload contains unsupported model(s): "
                + ", ".join(unknown_models)
            )
        missing = sorted(selected_workloads - set(tp_values_by_workload))
        extra = sorted(set(tp_values_by_workload) - selected_workloads)
        if missing or extra:
            details = []
            if missing:
                details.append(
                    "missing " + ", ".join(f"{model}@{context}" for model, context in missing)
                )
            if extra:
                details.append(
                    "unselected " + ", ".join(f"{model}@{context}" for model, context in extra)
                )
            raise ValueError("--tp-values-by-workload must exactly cover selected workloads: " + "; ".join(details))

    result: dict[tuple[str, int], list[int] | None] = {}
    for model, context in sorted(selected_workloads):
        devices = int(model_run_configs[model]["source_devices"])
        if balanced_envelope:
            tp_values = balanced_envelope_tp_values(model, context, devices)
        elif tp_values_by_workload is not None:
            tp_values = list(tp_values_by_workload[(model, context)])
        else:
            configured = model_run_configs[model]["tp_values"]
            tp_values = list(configured) if configured is not None else None
        invalid = [tp for tp in (tp_values or []) if tp <= 0 or devices % tp]
        if invalid:
            raise ValueError(
                f"TP values must divide source devices for {model}@{context}: "
                f"devices={devices}, invalid={invalid}"
            )
        result[(model, context)] = tp_values
    return result


def planned_ramulator_jobs(
    models: list[str],
    context_windows: list[int],
    workload_tp_values: dict[tuple[str, int], list[int] | None],
    memory_case_count: int,
    *,
    include_pipeline: bool,
    include_model_parallel: bool = True,
    symmetric_model_parallel: bool = False,
    batch_count: int = 1,
) -> int | None:
    if batch_count < 1:
        raise ValueError("batch_count must be positive")
    if any(workload_tp_values[(model, context)] is None for model in models for context in context_windows):
        return None
    per_memory_case = 0
    for model in models:
        for context in context_windows:
            if include_model_parallel:
                tp_values = workload_tp_values[(model, context)] or []
                per_memory_case += sum(
                    1 if symmetric_model_parallel or tp == 1 else 2
                    for tp in tp_values
                )
            if include_pipeline:
                per_memory_case += 1
    return per_memory_case * memory_case_count * batch_count


def build_contexts(args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.paper_long_context:
        context_windows = sorted(set(parse_int_list(args.context_windows)))
        if any(window < args.decode_tokens for window in context_windows):
            raise ValueError("every --context-window must be at least --decode-tokens")
        return [(window, window - args.decode_tokens // 2) for window in context_windows]
    if args.contexts:
        contexts = sorted(set(parse_int_list(args.contexts)))
        return [(context, context) for context in contexts]
    contexts = list(range(args.seqlen_gap, args.short_max_context + 1, args.seqlen_gap))
    if not contexts or contexts[-1] != args.short_max_context:
        contexts.append(args.short_max_context)
    contexts.extend(parse_int_list(args.long_contexts))
    contexts = sorted(set(contexts))
    return [(context, context) for context in contexts]


def write_lpddr4_yaml_with_nccd(base_yaml: Path, nccd: int, dst: Path) -> None:
    lines = base_yaml.read_text().splitlines(keepends=True)
    output: list[str] = []
    inserted = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("nCCD:"):
            continue
        output.append(line)
        if not inserted and "preset:" in stripped and "LPDDR4_AiM_timing" in stripped:
            indent = re.match(r"^(\s*)", line).group(1)
            output.append(f"{indent}nCCD: {nccd}\n")
            inserted = True

    if not inserted:
        raise RuntimeError(f"could not find LPDDR4_AiM_timing preset in {base_yaml}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(output))
    match = NCCD_RE.search(dst.read_text())
    if not match or int(match.group(1)) != nccd:
        raise RuntimeError(f"nCCD override verification failed for {dst}")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def run(cmd: list[str], *, dry_run: bool = False) -> None:
    print("[run]", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=CENT_SIM, check=True)


def run_model_case(
    args: argparse.Namespace,
    case: dict[str, str | int | Path],
    model: str,
    model_run_config: dict[str, int | list[int] | None],
    workload_tp_values: dict[tuple[str, int], list[int] | None],
    contexts: list[tuple[int, int]],
    model_parallel: bool,
) -> None:
    cmd = [
        sys.executable,
        "run_sim.py",
        "--num_channels", "32",
        "--num_banks", str(case["num_banks"]),
        "--experiment", str(case["name"]),
        "--trace-root", str(case["trace_root"]),
        "--log-root", str(case["log_root"]),
        "--ramulator-config", str(case["config"]),
        "--dram-power-impl", str(case["dram_power_impl"]),
        "--dram-energy-model", args.dram_energy_model,
        "--simulation_result_path", str(case["csv"]),
        "--model", model,
        "--generate_trace",
        "--simulate_trace",
        "--update_csv",
        "--num_devices", str(model_run_config["source_devices"]),
        "--PCIE_lanes", str(model_run_config["pcie_lanes"]),
        "--run_simulation_max_workers", str(args.run_workers),
        "--generate_trace_max_workers", str(args.trace_workers),
    ]
    if args.include_embedding:
        cmd.append("--process_results")
    else:
        cmd.append("--decode-only")
    if model_parallel:
        cmd.append("--model_parallel")
        if args.kv_head_tp or args.kv_head_tp_systolic:
            cmd.append("--kv-head-tp")
        elif not args.master_attention:
            cmd.append("--inter-device-attention")
    if args.paper_long_context:
        for context_window, active_seqlen in contexts:
            batch_sizes = args.batch_sizes if args.kv_head_tp_systolic else [1]
            for batch_size in batch_sizes:
                workload_cmd = list(cmd)
                tp_values = workload_tp_values[(model, context_window)] if model_parallel else None
                if tp_values:
                    workload_cmd.extend(["--tp-values", *(str(value) for value in tp_values)])
                if args.kv_head_tp_systolic:
                    workload_cmd.extend([
                        "--systolic-pim",
                        "--systolic-dim", "4",
                        "--batch-size", str(batch_size),
                    ])
                    if args.ewmul_pnm:
                        workload_cmd.append("--EWMUL_PNM")
                    if args.flash_attention:
                        workload_cmd.extend([
                            "--flash-attention",
                            "--flash-attention-block-size",
                            str(args.flash_attention_block_size),
                        ])
                    if args.pipelined_softmax:
                        workload_cmd.append("--pipelined-softmax")
                run(workload_cmd + [
                    "--seqlen", str(active_seqlen),
                    "--max-seq-len", str(context_window),
                ], dry_run=args.dry_run)
    else:
        if model_parallel:
            tp_values = model_run_config["tp_values"]
            if tp_values:
                cmd.extend(["--tp-values", *(str(value) for value in tp_values)])
        batch_sizes = args.batch_sizes if args.kv_head_tp_systolic else [1]
        for batch_size in batch_sizes:
            workload_cmd = list(cmd)
            if args.kv_head_tp_systolic:
                workload_cmd.extend([
                    "--systolic-pim",
                    "--systolic-dim", "4",
                    "--batch-size", str(batch_size),
                ])
                if args.ewmul_pnm:
                    workload_cmd.append("--EWMUL_PNM")
                if args.flash_attention:
                    workload_cmd.extend([
                        "--flash-attention",
                        "--flash-attention-block-size",
                        str(args.flash_attention_block_size),
                    ])
                if args.pipelined_softmax:
                    workload_cmd.append("--pipelined-softmax")
            run(
                workload_cmd
                + ["--seqlen", *(str(active_seqlen) for _, active_seqlen in contexts)],
                dry_run=args.dry_run,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="Llama2-7B,Llama2-70B")
    parser.add_argument("--contexts", default="4096,32768,131072", help="Explicit comma-separated sequence lengths. Overrides --seqlen-gap/--long-contexts.")
    parser.add_argument("--seqlen-gap", type=int, default=4096)
    parser.add_argument("--short-max-context", type=int, default=4096)
    parser.add_argument("--long-contexts", default=DEFAULT_LONG_CONTEXTS)
    parser.add_argument("--paper-long-context", action="store_true", help="Use fixed 3584-token decode midpoint samples for each --context-window.")
    parser.add_argument("--context-windows", default="4096,32768,131072", help="Total context windows for --paper-long-context.")
    parser.add_argument("--decode-tokens", type=int, default=3584, help="Fixed decode length used to derive long-context midpoint samples.")
    parser.add_argument("--trace-workers", type=int, default=8)
    parser.add_argument("--run-workers", type=int, default=8)
    parser.add_argument(
        "--tp-values",
        type=parse_int_list,
        help="Comma-separated tensor-parallel degrees for model-parallel runs (for example 1,2,4).",
    )
    parser.add_argument(
        "--tp-values-by-model",
        type=parse_model_int_list_map,
        help=(
            "Per-model TP lists. Use semicolon-separated assignments, for example "
            "'Llama2-7B=1,2,4,8;Llama2-70B=1,2,4'. Overrides --tp-values."
        ),
    )
    parser.add_argument(
        "--tp-values-by-workload",
        type=parse_workload_int_list_map,
        help=(
            "Exact per-model/context TP lists for midpoint runs, for example "
            "'Llama2-7B@4096=1;Llama2-7B@32768=1,2'. Requires complete coverage "
            "of selected paper context windows and overrides model/global TP lists."
        ),
    )
    parser.add_argument(
        "--balanced-capacity-envelope",
        "--balanced-envelope",
        dest="balanced_envelope",
        action="store_true",
        help=(
            "Run the 16-GiB fully-sharded TP envelope independently for each model/context, "
            "stopping after the first TP with Bmax > PP. Implies midpoint contexts and "
            "model-parallel-only execution, uses 16 devices/288 total lanes for 7B "
            "(preserving 18 lanes/device), and writes to fresh "
            "balanced_equal_power_all_contexts roots unless overridden. "
            "--balanced-envelope is retained as an alias."
        ),
    )
    parser.add_argument(
        "--source-devices-by-model",
        type=parse_model_int_map,
        help=(
            "Optional per-model source device counts, for example "
            "'Llama2-7B=8;Llama2-70B=32'. Unspecified models keep repository defaults."
        ),
    )
    parser.add_argument(
        "--pcie-lanes-by-model",
        type=parse_model_int_map,
        help=(
            "Optional per-model total PCIe lane counts, for example "
            "'Llama2-7B=144;Llama2-70B=144'. Unspecified models keep repository defaults."
        ),
    )
    parser.add_argument("--include-embedding", action="store_true", help="Include embedding traces/latency. Default is decode-only.")
    parser.add_argument("--dram-energy-model", choices=["legacy", "trace-based"], default="legacy")
    parser.add_argument("--lpddr4-base-yaml", type=Path, default=ROOT / "aim_simulator/test/example_LPDDR4.yaml")
    parser.add_argument("--gddr6-yaml", type=Path, default=ROOT / "aim_simulator/test/example_GDDR6.yaml")
    parser.add_argument("--generated-yaml-dir", type=Path, default=ROOT / "aim_simulator/test/generated")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT, help="Root directory for result CSVs and Ramulator logs.")
    parser.add_argument("--trace-root", type=Path, default=TRACE_ROOT, help="Root directory for generated instruction traces.")
    parser.add_argument(
        "--master-attention",
        action="store_true",
        help=(
            "Keep KV-cache attention on the TP-group master device, matching the "
            "paper's FC-only TP mapping. By default model-parallel runs use "
            "inter-device attention."
        ),
    )
    parser.add_argument(
        "--kv-head-tp",
        action="store_true",
        help=(
            "Run the standard KV-head TP mapping with TP=1,2,4,8. Uses fresh "
            "kv_head_tp_equal_power_all_contexts roots unless overridden."
        ),
    )
    parser.add_argument(
        "--kv-head-tp-systolic",
        action="store_true",
        help=(
            "Run only the standard Device-Channel-group-Bank KV-head TP mapping "
            "with SA=4x16, batch=1,2,3,4, TP=1,2,4,8, and midpoint contexts. Writes "
            "to kv_head_tp_systolic_all_context roots unless overridden."
        ),
    )
    parser.add_argument(
        "--EWMUL_PNM",
        "--EWMUL-PNM",
        "--ewmul-pnm",
        dest="ewmul_pnm",
        action="store_true",
        help=(
            "Explicitly request cent_dev PNM element-wise multiplies. "
            "Systolic PIM enables the same behavior implicitly."
        ),
    )
    parser.add_argument(
        "--flash-attention",
        action="store_true",
        help=(
            "Use the cent_dev-style block FlashAttention trace for "
            "--kv-head-tp-systolic."
        ),
    )
    parser.add_argument(
        "--flash-attention-block-size",
        type=int,
        default=1024,
        help="Context tokens per FlashAttention block (default: 1024).",
    )
    parser.add_argument(
        "--pipelined-softmax",
        action="store_true",
        help=(
            "Use the cent_dev Softmax/QK overlap model for "
            "--kv-head-tp-systolic. This changes analytical latency only."
        ),
    )
    parser.add_argument(
        "--batch-sizes",
        type=parse_int_list,
        help=(
            "Batch sizes for --kv-head-tp-systolic. The default is 1,2,3,4; "
            "values must fit the four-row systolic array."
        ),
    )
    parser.add_argument("--cases", help="Comma-separated subset of cases: GDDR6,LPDDR4X_nCCD2,LPDDR4X_nCCD6")
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument(
        "--include-pipeline",
        action="store_true",
        help="Include pipeline-only sources in balanced capacity-envelope mode (off by default).",
    )
    parser.add_argument("--skip-model-parallel", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact run_sim.py jobs without generating or simulating traces.",
    )
    args = parser.parse_args()
    if args.skip_pipeline and args.include_pipeline:
        raise ValueError("--skip-pipeline and --include-pipeline are mutually exclusive")
    if (args.kv_head_tp or args.kv_head_tp_systolic) and args.master_attention:
        raise ValueError("KV-head TP modes and --master-attention are mutually exclusive")
    if args.kv_head_tp and args.kv_head_tp_systolic:
        raise ValueError("--kv-head-tp and --kv-head-tp-systolic are mutually exclusive")
    if args.flash_attention and not args.kv_head_tp_systolic:
        raise ValueError("--flash-attention requires --kv-head-tp-systolic")
    if args.pipelined_softmax and not args.kv_head_tp_systolic:
        raise ValueError("--pipelined-softmax requires --kv-head-tp-systolic")
    if args.flash_attention_block_size < 1:
        raise ValueError("--flash-attention-block-size must be positive")
    if args.batch_sizes is not None and not args.kv_head_tp_systolic:
        raise ValueError("--batch-sizes is only valid with --kv-head-tp-systolic")
    if (args.kv_head_tp or args.kv_head_tp_systolic) and args.balanced_envelope:
        raise ValueError("KV-head TP modes use a fixed TP=1,2,4,8 sweep; omit --balanced-capacity-envelope")
    if args.kv_head_tp or args.kv_head_tp_systolic:
        conflicting = [
            option
            for option, value in (
                ("--tp-values-by-model", args.tp_values_by_model),
                ("--tp-values-by-workload", args.tp_values_by_workload),
            )
            if value is not None
        ]
        if conflicting:
            raise ValueError("KV-head TP modes cannot be combined with " + ", ".join(conflicting))
        if args.tp_values is not None and sorted(set(args.tp_values)) != [1, 2, 4, 8]:
            raise ValueError("KV-head TP all-context runs require --tp-values 1 2 4 8")
        args.tp_values = [1, 2, 4, 8]
        args.paper_long_context = True
        args.skip_pipeline = True
        if args.kv_head_tp_systolic:
            args.batch_sizes = sorted(set(args.batch_sizes or [1, 2, 3, 4]))
            invalid_batches = [
                batch for batch in args.batch_sizes if batch < 1 or batch > 4
            ]
            if invalid_batches:
                raise ValueError(
                    "SA=4x16 batch sizes must be in [1, 4]: "
                    f"{invalid_batches}"
                )
            if args.output_root == OUTPUT_ROOT:
                args.output_root = (
                    OUTPUT_ROOT
                    / "kv_head_tp_systolic_all_context/raw/systolic_4x16"
                )
            if args.trace_root == TRACE_ROOT:
                args.trace_root = (
                    TRACE_ROOT
                    / "kv_head_tp_systolic_all_context/systolic_4x16"
                )
        else:
            if args.output_root == OUTPUT_ROOT:
                args.output_root = OUTPUT_ROOT / "kv_head_tp_equal_power_all_contexts/raw"
            if args.trace_root == TRACE_ROOT:
                args.trace_root = TRACE_ROOT / "kv_head_tp_equal_power_all_contexts"
    if args.balanced_envelope:
        conflicting = [
            option
            for option, value in (
                ("--tp-values", args.tp_values),
                ("--tp-values-by-model", args.tp_values_by_model),
                ("--tp-values-by-workload", args.tp_values_by_workload),
            )
            if value is not None
        ]
        if conflicting:
            raise ValueError(
                "--balanced-capacity-envelope cannot be combined with " + ", ".join(conflicting)
            )
        args.paper_long_context = True
        args.skip_pipeline = not args.include_pipeline
        if args.output_root == OUTPUT_ROOT:
            args.output_root = OUTPUT_ROOT / "balanced_equal_power_all_contexts/raw"
        if args.trace_root == TRACE_ROOT:
            args.trace_root = TRACE_ROOT / "balanced_equal_power_all_contexts"
    if args.tp_values_by_workload is not None and not args.paper_long_context:
        raise ValueError("--tp-values-by-workload requires --paper-long-context")
    # run_sim.py is launched with cent_simulation as its working directory.
    # Resolve caller-provided roots here so relative CLI paths still designate
    # the intended location rather than cent_simulation/cent_simulation/....
    args.output_root = args.output_root.resolve()
    args.trace_root = args.trace_root.resolve()

    models = parse_csv_list(args.models)
    contexts = build_contexts(args)
    if args.paper_long_context and args.include_embedding:
        raise ValueError("--paper-long-context requires decode-only results; omit --include-embedding")
    unknown = sorted(set(models) - set(MODEL_DEVICES))
    if unknown:
        raise ValueError(f"unsupported model(s): {', '.join(unknown)}")
    source_device_overrides = args.source_devices_by_model
    pcie_lane_overrides = args.pcie_lanes_by_model
    if args.balanced_envelope or args.kv_head_tp or args.kv_head_tp_systolic:
        source_device_overrides = {
            **BALANCED_ENVELOPE_SOURCE_DEVICES,
            **(source_device_overrides or {}),
        }
        pcie_lane_overrides = {
            **BALANCED_ENVELOPE_PCIE_LANES,
            **(pcie_lane_overrides or {}),
        }
    model_run_configs = resolve_model_run_config(
        models,
        global_tp_values=args.tp_values,
        tp_values_by_model=args.tp_values_by_model,
        source_devices_by_model=source_device_overrides,
        pcie_lanes_by_model=pcie_lane_overrides,
    )
    context_windows = [context_window for context_window, _ in contexts]
    workload_tp_values = resolve_workload_tp_values(
        models,
        context_windows,
        model_run_configs,
        tp_values_by_workload=args.tp_values_by_workload,
        balanced_envelope=args.balanced_envelope,
    )
    for model in models:
        config = model_run_configs[model]
        print(
            f"[model] {model} source_devices={config['source_devices']} "
            f"pcie_lanes={config['pcie_lanes']}"
        )
        if args.paper_long_context:
            for context_window in context_windows:
                tp_description = workload_tp_values[(model, context_window)]
                if tp_description is None:
                    tp_description = "all factors"
                print(f"[workload] {model}@{context_window} tp_values={tp_description}")

    lpddr4x_nccd2_yaml = args.generated_yaml_dir / "example_LPDDR4X_nCCD2.yaml"
    lpddr4x_nccd6_yaml = args.generated_yaml_dir / "example_LPDDR4X_nCCD6.yaml"
    write_lpddr4_yaml_with_nccd(args.lpddr4_base_yaml, 2, lpddr4x_nccd2_yaml)
    write_lpddr4_yaml_with_nccd(args.lpddr4_base_yaml, 6, lpddr4x_nccd6_yaml)
    print(f"[yaml] {display_path(lpddr4x_nccd2_yaml)}")
    print(f"[yaml] {display_path(lpddr4x_nccd6_yaml)}")
    if args.paper_long_context:
        for context_window, active_seqlen in contexts:
            print(f"[context] window={context_window} active_decode_seqlen={active_seqlen}")

    trace_variant = Path(PAPER_LONG_CONTEXT_VARIANT) if args.paper_long_context else Path()
    csv_suffix = f"_{PAPER_LONG_CONTEXT_VARIANT}" if args.paper_long_context else ""

    cases = [
        {
            "name": "GDDR6",
            "num_banks": 16,
            "config": args.gddr6_yaml,
            "trace_root": args.trace_root / trace_variant / "GDDR6",
            "log_root": args.output_root / "GDDR6" / f"ramulator{csv_suffix}",
            "csv": args.output_root / "GDDR6" / f"simulation_results_decode_only{csv_suffix}.csv",
            "dram_power_impl": "GDDR6",
        },
        {
            "name": "LPDDR4X_nCCD2",
            "num_banks": 8,
            "config": lpddr4x_nccd2_yaml,
            "trace_root": args.trace_root / trace_variant / "LPDDR4X",
            "log_root": args.output_root / "LPDDR4X" / f"ramulator{csv_suffix}_nCCD2",
            "csv": args.output_root / "LPDDR4X" / f"simulation_results_decode_only{csv_suffix}_nCCD2.csv",
            "dram_power_impl": "LPDDR4X",
        },
        {
            "name": "LPDDR4X_nCCD6",
            "num_banks": 8,
            "config": lpddr4x_nccd6_yaml,
            "trace_root": args.trace_root / trace_variant / "LPDDR4X",
            "log_root": args.output_root / "LPDDR4X" / f"ramulator{csv_suffix}_nCCD6",
            "csv": args.output_root / "LPDDR4X" / f"simulation_results_decode_only{csv_suffix}_nCCD6.csv",
            "dram_power_impl": "LPDDR4X",
        },
    ]
    case_by_name = {str(case["name"]): case for case in cases}
    if args.cases:
        requested_cases = set(parse_csv_list(args.cases))
        known_cases = set(case_by_name)
        unknown_cases = sorted(requested_cases - known_cases)
        if unknown_cases:
            raise ValueError(f"unsupported case(s): {', '.join(unknown_cases)}")
        cases = [case for case in cases if str(case["name"]) in requested_cases]

    job_count = planned_ramulator_jobs(
        models,
        context_windows,
        workload_tp_values,
        len(cases),
        include_pipeline=not args.skip_pipeline,
        include_model_parallel=not args.skip_model_parallel,
        symmetric_model_parallel=(args.kv_head_tp or args.kv_head_tp_systolic),
        batch_count=(len(args.batch_sizes) if args.kv_head_tp_systolic else 1),
    )
    if job_count is not None:
        print(f"[plan] exact Ramulator jobs={job_count}")

    for case in cases:
        print(f"[case] {case['name']}", flush=True)
        for model in models:
            if not args.skip_pipeline:
                run_model_case(
                    args,
                    case,
                    model,
                    model_run_configs[model],
                    workload_tp_values,
                    contexts,
                    model_parallel=False,
                )
            if not args.skip_model_parallel:
                run_model_case(
                    args,
                    case,
                    model,
                    model_run_configs[model],
                    workload_tp_values,
                    contexts,
                    model_parallel=True,
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
