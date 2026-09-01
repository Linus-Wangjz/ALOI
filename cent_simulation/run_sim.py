import os
import math
import pandas as pd
import argparse
import subprocess
import concurrent.futures
import sys
from pathlib import Path
from cxl_latency import (
    gpt_latency,
    kv_head_tp_latency,
    kv_head_tp_pcie_bits,
    llama_latency,
    vector_latency,
)
from cent_power_calculator import DRAM_ENERGY_MODELS, ACCEL_CYCLE, SHARED_BUFFER_CAPACITY_BYTES, SRAM_IO_PARALLEL, add_energy_terms, kv_head_tp_pnm_dynamic_energy, power_calculator, command_processor, command_trace_prefix_for_log, set_channel_count, KILO, FREQ, SB_RD_CYCLE, SB_WR_CYCLE, RV_RMSNorm_CYCLE, RV_ROTEmbed_CYCLE_PIPELINE, RV_SFT_CYCLE_PIPELINE
from systolic_power import SYSTOLIC_PIM_POWER_SCALING
from tp_mapping import KVHeadTPLayout, SystolicTPLayout, kv_head_tp_shape
from utils import InOut_latency, n_heads, gqa_factor, embedding_size, ffn_size, TransformerBlock_number, minimal_channel_per_block, pipeline_parallel_mode_list, model_parallel_mode_list

def get_args():
    parser = argparse.ArgumentParser('run_scripts.py')
    parser.add_argument("--num_channels", type=int, help="Number of channels per device", default=32)
    parser.add_argument("--num_banks", "--num-banks", dest="num_banks", type=int, help="Number of banks per channel", default=16)
    parser.add_argument(
        "--parallel_SRAM",
        "--parallel-sram",
        dest="parallel_sram",
        type=int,
        default=int(SRAM_IO_PARALLEL),
        help="Independent 128-bit shared-buffer banks/ports (cent_dev default: 32)",
    )
    parser.add_argument("--num_devices", type=int, help="Number of CXL devices", default=32)
    parser.add_argument("--PCIE_lanes", type=int, help="Number of PCIE lanes", default=144)
    parser.add_argument("--reuse_size", type=int, help="GB reuse size, depending on register number", default=32)
    parser.add_argument("--generate_trace_max_workers", type=int, help="maximum concurrent threads to generate traces, limited by memory", default=20)
    parser.add_argument("--run_simulation_max_workers", type=int, help="maximum concurrent threads to generate traces, limited by memory", default=4)
    parser.add_argument("--model", choices=["Llama2-7B", "Llama2-13B", "Llama2-70B"], help="LLM Model", required=True)
    parser.add_argument("--generate_trace", action="store_true", help="Generate traces")
    parser.add_argument("--simulate_trace", action="store_true", help="Simulate traces")
    parser.add_argument("--process_results", action="store_true", help="Process results")
    parser.add_argument("--update_csv", action="store_true", help="Update results to csv file")
    parser.add_argument("--simulation_result_path", type=str, help="Path to the result file")
    parser.add_argument("--process_throughputs", action="store_true", help="average throughputs for various seqlen")
    parser.add_argument("--processed_result_path", type=str, help="Path to the final result file")
    parser.add_argument("--phase", choices=["end2end", "prefill", "decoding"], help="Phase of the model", default="end2end")
    parser.add_argument("--prefill", type=int, help="Prefill length", default=512)
    parser.add_argument("--decoding", type=int, help="Decoding length", default=3584)
    parser.add_argument("--seqlen", type=int, nargs='+', help="Sequence list")
    parser.add_argument("--seqlen_gap", type=int, help="Gap between sequence lengths", default=4096)
    parser.add_argument("--max_seq_len", "--max-seq-len", dest="max_seq_len", type=int, help="Maximum sequence length used for memory mapping")
    parser.add_argument("--ramulator_config", "--ramulator-config", dest="ramulator_config", type=str, help="Ramulator YAML config", default="../aim_simulator/test/example_GDDR6.yaml")
    parser.add_argument("--experiment", type=str, help="Experiment name used for default output paths (for example: GDDR6 or LPDDR4X_nCCD2)")
    parser.add_argument("--trace_root", "--trace-root", dest="trace_root", type=str, help="Directory for functional trace files")
    parser.add_argument("--log_root", "--log-root", dest="log_root", type=str, help="Directory for Ramulator logs and sidecar artifacts")
    parser.add_argument("--dram_power_impl", "--dram-power-impl", dest="dram_power_impl", choices=["GDDR6", "LPDDR4", "LPDDR4X"], help="DRAM power table to use when updating CSV energy/power")
    parser.add_argument("--dram_energy_model", "--dram-energy-model", dest="dram_energy_model", choices=DRAM_ENERGY_MODELS, default="legacy", help="DRAM energy model: command-count legacy or TraceRecorder-based activity replay")
    parser.add_argument("--decode_only", "--decode-only", dest="decode_only", action="store_true", help="Skip embedding traces and report decode-only token latency/energy")
    parser.add_argument("--model_parallel", action="store_true", help="Apply model parallelism")
    parser.add_argument("--systolic_pim", "--systolic-pim", dest="systolic_pim", action="store_true", help="Use the systolic PIM trace path")
    parser.add_argument("--systolic_dim", "--systolic-dim", dest="systolic_dim", type=int, choices=sorted(SYSTOLIC_PIM_POWER_SCALING), default=1, help="Systolic array height; width is 16")
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=1, help="Decode batch size; independent of the systolic array height")
    parser.add_argument(
        "--EWMUL_PNM",
        "--EWMUL-PNM",
        dest="ewmul_pnm",
        action="store_true",
        help=(
            "Move RMSNorm, RoPE, and fused-FFN element-wise multiplies to "
            "the PNM VEC_MUL path; cent_dev forces this on for systolic PIM"
        ),
    )
    parser.add_argument(
        "--flash_attention",
        "--flash-attention",
        dest="flash_attention",
        action="store_true",
        help="Use cent_dev-style block FlashAttention without score DRAM workspace",
    )
    parser.add_argument("--flash_attention_block_size", "--flash-attention-block-size", dest="flash_attention_block_size", type=int, default=1024)
    parser.add_argument(
        "--pipelined_softmax",
        "--pipelined-softmax",
        dest="pipelined_softmax",
        action="store_true",
        help=(
            "Overlap Softmax with its QK score producer using the cent_dev "
            "startup-window model; Softmax energy is still charged in full"
        ),
    )
    parser.add_argument("--inter-device-attention", action="store_true")
    parser.add_argument(
        "--kv-head-tp",
        action="store_true",
        help="Use standard query/KV-head tensor parallelism with local attention",
    )
    parser.add_argument(
        "--tp-values",
        type=int,
        nargs="+",
        help=(
            "Optional tensor-parallel degrees to trace/simulate. Values must divide "
            "--num_devices. The default preserves the original all-factor sweep."
        ),
    )
    args = parser.parse_args()
    if args.kv_head_tp and args.inter_device_attention:
        parser.error("--kv-head-tp and --inter-device-attention are mutually exclusive")
    if args.kv_head_tp and not args.model_parallel:
        parser.error("--kv-head-tp requires --model_parallel")
    if args.kv_head_tp and not args.decode_only:
        parser.error("--kv-head-tp currently supports decode-only trace generation")
    if args.flash_attention_block_size < 1:
        parser.error("--flash_attention_block_size must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.parallel_sram < 1:
        parser.error("--parallel-sram must be positive")
    set_channel_count(args.num_channels, args.parallel_sram)
    if args.simulation_result_path is None:
        args.simulation_result_path = default_simulation_result_path(args)
    if args.processed_result_path is None:
        args.processed_result_path = default_processed_result_path(args)
    return args


def factorize(n):
    factors = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            factors.append(i)
            if i != n // i:
                factors.append(n // i)
    return sorted(factors)


def adjust_systolic_energy(energy, args):
    """Apply cent_dev's batch and systolic-array energy scaling."""
    adjusted = dict(energy)
    for component in [
        "IB_DYN",
        "SB_DYN",
        "RV_DYN",
        "RED_DYN",
        "EXP_DYN",
        "VEC_ADD_DYN",
        "VEC_MUL_DYN",
    ]:
        if component in adjusted:
            adjusted[component] *= args.batch_size
    if args.systolic_pim and "PIM" in adjusted:
        adjusted["PIM"] *= SYSTOLIC_PIM_POWER_SCALING[args.systolic_dim]
    return adjusted


def normalize_batch_group_energy(energy, batch_size):
    """Return total energy for one traced group and per emitted token."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    group_energy = sum(energy.values())
    return group_energy, group_energy / batch_size


INTER_DEVICE_HELPER_MODE = "model_parallel_helper_attention"
KV_HEAD_MAIN_MODE = "model_parallel_kv_head_main"
KV_HEAD_HELPER_MODE = "model_parallel_kv_head_helper"


def model_parallel_tp_values(args):
    values = factorize(args.num_devices) if args.tp_values is None else sorted(set(args.tp_values))
    invalid = [value for value in values if value <= 0 or args.num_devices % value]
    if invalid:
        raise ValueError(
            f"Every --tp-values entry must be positive and divide --num_devices={args.num_devices}: {invalid}"
        )
    if getattr(args, "kv_head_tp", False):
        kv_heads = n_heads[args.model] // gqa_factor[args.model]
        invalid = [
            value
            for value in values
            if n_heads[args.model] % value
            or kv_heads % value
            or ffn_size[args.model] % value
            or value > 8
        ]
        if invalid:
            raise ValueError(
                "KV-head TP values must be <=8 and evenly divide query heads, "
                f"KV heads, and FFN dimension for {args.model}: {invalid}"
            )
    return values


def model_parallel_main_mode(args):
    return KV_HEAD_MAIN_MODE if getattr(args, "kv_head_tp", False) else "model_parallel"


def model_parallel_helper_mode(args):
    if getattr(args, "kv_head_tp", False):
        return KV_HEAD_HELPER_MODE
    return INTER_DEVICE_HELPER_MODE if args.inter_device_attention else "model_parallel_FC"


def model_parallel_modes_for_tp(args, tp):
    modes = [model_parallel_main_mode(args)]
    if tp > 1 and not getattr(args, "kv_head_tp", False):
        modes.append(model_parallel_helper_mode(args))
    return modes


def model_parallel_helper_trace_command(args, python, model_flag, tp, seqlen, max_seq_len, trace_file):
    command = [
        python, "function_sim.py", model_flag,
        "--n_heads", str(n_heads[args.model]),
        "--ffn_dim", str(ffn_size[args.model]),
        "--only-trace",
        "--num-channels", str(args.num_channels),
        "--num-banks", str(args.num_banks),
        "--max-seq-len", str(max_seq_len),
        "--FC-devices", str(tp),
        "--model-parallel",
        "--seqlen", str(seqlen),
        "--GEMV", "reuse-GB",
        "--reuse-size", str(args.reuse_size),
        "--trace-file", trace_file,
    ]
    if getattr(args, "kv_head_tp", False):
        command.extend([
            "--kv-head-tp",
            "--tp-device-role", "helper",
            "--trace-fc-kqvo",
            "--trace-attention",
            "--trace-softmax",
            "--trace-fc-ffn",
            "--trace-activation",
        ])
    elif args.inter_device_attention:
        # Helpers execute their local FC and KV-attention shards. Norm,
        # centralized softmax, and activation stay on the main device.
        command.extend([
            "--inter-device-attention",
            "--trace-fc-kqvo",
            "--trace-attention",
            "--trace-fc-ffn",
        ])
    else:
        command.extend(["--only-FC", "--op-trace"])
    return command


def trace_root(args):
    if args.trace_root:
        root = args.trace_root
        return os.path.join(root, systolic_trace_variant(args)) if args.systolic_pim else root
    name = experiment_name(args)
    # LPDDR4X nCCD sweeps reuse one functional trace set; only Ramulator
    # timing/log artifacts differ between nCCD values.
    trace_name = name.split("_nCCD", 1)[0]
    root = os.path.join("trace", trace_name)
    return os.path.join(root, systolic_trace_variant(args)) if args.systolic_pim else root


def log_root(args):
    if args.log_root:
        root = args.log_root
        return os.path.join(root, systolic_trace_variant(args)) if args.systolic_pim else root
    _, nccd = split_experiment_name(experiment_name(args))
    if nccd:
        root = os.path.join(experiment_output_dir(args), f"ramulator_{nccd}")
    else:
        root = os.path.join(experiment_output_dir(args), "ramulator")
    return os.path.join(root, systolic_trace_variant(args)) if args.systolic_pim else root


def systolic_trace_variant(args):
    variant = (
        f"systolic_pim_{args.systolic_dim}_batch_size_{args.batch_size}"
        f"_ewmul_pnm_{int(ewmul_pnm_enabled(args))}"
    )
    if args.flash_attention:
        variant += f"_flash_{args.flash_attention_block_size}"
    return variant


def ewmul_pnm_requested(args):
    """Return the user-requested EWMUL placement."""

    return bool(getattr(args, "ewmul_pnm", getattr(args, "EWMUL_PNM", False)))


def ewmul_pnm_enabled(args):
    """Match cent_dev: every systolic-PIM run uses the PNM EWMUL path."""

    return bool(
        getattr(args, "systolic_pim", False) or ewmul_pnm_requested(args)
    )


def softmax_pipeline_startup_tokens(args):
    """Number of score positions produced before streamed Softmax can hide.

    This is cent_dev's ``pp_init`` model.  A systolic MAC command produces one
    16-element burst per bank, while the vector path produces one element per
    bank.  It is a producer/consumer startup window, not the pipeline-parallel
    stage count.
    """

    producer_width = 16 if args.systolic_pim else 1
    return producer_width * args.num_channels * args.num_banks


def softmax_exposure_factor(args, seqlen):
    """Return the fraction of full Softmax latency exposed on the critical path."""

    if seqlen <= 0:
        raise ValueError("seqlen must be positive")
    if not getattr(args, "pipelined_softmax", False):
        return 1.0
    startup_tokens = softmax_pipeline_startup_tokens(args)
    return min(startup_tokens, seqlen) / float(seqlen)


def experiment_name(args):
    if args.experiment:
        return args.experiment
    if args.dram_power_impl:
        return args.dram_power_impl
    config_name = os.path.basename(args.ramulator_config).upper()
    if "LPDDR" in config_name:
        return "LPDDR4X"
    if "GDDR" in config_name:
        return "GDDR6"
    return "LPDDR4X" if args.num_banks <= 8 else "GDDR6"


def experiment_output_dir(args):
    base_name, _ = split_experiment_name(experiment_name(args))
    return os.path.join("output", base_name)


def split_experiment_name(name):
    if "_nCCD" not in name:
        return name, ""
    base_name, nccd = name.split("_", 1)
    return base_name, nccd


def default_simulation_result_path(args):
    _, nccd = split_experiment_name(experiment_name(args))
    phase_suffix = "_decode_only" if args.decode_only else ""
    nccd_suffix = f"_{nccd}" if nccd else ""
    filename = f"simulation_results{phase_suffix}{nccd_suffix}.csv"
    return os.path.join(experiment_output_dir(args), filename)


def default_processed_result_path(args):
    _, nccd = split_experiment_name(experiment_name(args))
    suffix = f"_{nccd}" if nccd else ""
    return os.path.join(experiment_output_dir(args), f"processed_results{suffix}.csv")


def missing_or_empty(file):
    return not os.path.exists(file) or os.stat(file).st_size == 0


def trace_needs_generation(trace_file):
    path = Path(trace_file)
    if path.is_symlink() and not path.exists():
        # Old LPDDR4X trace links point at the retired LPDDR4 output tree. A
        # full rerun must replace only these dangling links with fresh traces.
        path.unlink()
    return missing_or_empty(path)


def trace_based_artifact_paths(log_file):
    path = Path(log_file)
    base = Path(str(path)[:-4]) if str(path).endswith(".log") else path
    return {
        "config": Path(f"{base}.ramulator.yaml"),
        "timing": Path(f"{base}.timing.yaml"),
        "command_prefix": Path(f"{base}.cmd"),
    }


def timing_artifact_complete(log_file):
    artifacts = trace_based_artifact_paths(log_file)
    return not missing_or_empty(log_file) and not missing_or_empty(artifacts["timing"])


def trace_based_artifacts_complete(log_file):
    artifacts = trace_based_artifact_paths(log_file)
    return timing_artifact_complete(log_file) and any(
        path.stat().st_size > 0
        for path in artifacts["command_prefix"].parent.glob(f"{artifacts['command_prefix'].name}.ch*")
    )


def write_ramulator_config(base_config, destination, timing_path, command_trace_prefix, include_command_trace):
    lines = Path(base_config).read_text().splitlines(keepends=True)
    output = []
    inserted_plugins = False
    for line in lines:
        output.append(line)
        if line.strip() == "plugins:":
            indent = line[:len(line) - len(line.lstrip())]
            output.extend([
                f"{indent}  - ControllerPlugin:\n",
                f"{indent}      impl: DRAMTimingExporter\n",
                f"{indent}      path: {timing_path}\n",
            ])
            if include_command_trace:
                output.extend([
                    f"{indent}  - ControllerPlugin:\n",
                    f"{indent}      impl: TraceRecorder\n",
                    f"{indent}      path: {command_trace_prefix}\n",
                ])
            inserted_plugins = True
    if not inserted_plugins:
        raise RuntimeError(f"Could not find the Controller plugins section in {base_config}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(output))


def needs_simulation(args, log_file):
    if not timing_artifact_complete(log_file):
        return True
    return args.dram_energy_model == "trace-based" and not trace_based_artifacts_complete(log_file)


def ramulator_command(args, trace_file, log_file):
    artifacts = trace_based_artifact_paths(log_file)
    write_ramulator_config(
        args.ramulator_config,
        artifacts["config"],
        artifacts["timing"],
        artifacts["command_prefix"],
        args.dram_energy_model == "trace-based",
    )
    config = artifacts["config"]
    return f"../aim_simulator/build/ramulator2 -f {config} -t {trace_file}"


def trace_max_seq_len(args, seqlen_list):
    requested = list(seqlen_list)
    if not args.decode_only:
        requested.append(args.prefill + args.decoding)
    max_requested_seqlen = max(requested)
    if args.max_seq_len is None:
        return max_requested_seqlen
    if args.max_seq_len < max_requested_seqlen:
        raise ValueError(f"--max-seq-len ({args.max_seq_len}) must be >= maximum requested seqlen ({max_requested_seqlen})")
    return args.max_seq_len


def run_checked_command(command):
    if "--trace-file" in command:
        trace_path = Path(command[command.index("--trace-file") + 1])
        trace_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        command_str = " ".join(command)
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {command_str}\n{result.stdout}\n{result.stderr}")
    return result


def generate_trace(args, seqlen_list):

    print(f"Generating traces for {args.model} with {args.generate_trace_max_workers} threads...")

    if args.model == "GPT3-175B":
        model = "--GPT3-175B"
    elif args.model == "Llama2-70B" or "Llama3" in args.model:
        model = "--Llama-GQA"
    elif "Llama2" in args.model:
        model = "--Llama"

    commands_generate_traces = []
    blocks_per_device = (TransformerBlock_number[args.model] - 1) // args.num_devices + 1
    channels_per_block = args.num_channels // blocks_per_device
    FC_devices_list = model_parallel_tp_values(args)
    trace_root_dir = trace_root(args)
    max_seq_len = trace_max_seq_len(args, seqlen_list)
    python = sys.executable

    # Embedding is skipped for decode-only results.
    seqlen = args.prefill + args.decoding
    if not args.decode_only:
        if args.model_parallel:
            for FC_devices in FC_devices_list:
                if trace_needs_generation(f"{trace_root_dir}/model_parallel_embedding/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt"):
                    commands_generate_traces.append([python, "function_sim.py", model, "--n_heads", str(n_heads[args.model]), "--ffn_dim", str(ffn_size[args.model]), "--embedding", "--only-trace", "--num-channels", str(args.num_channels), "--num-banks", str(args.num_banks), "--max-seq-len", str(max_seq_len), "--FC-devices", str(FC_devices), "--model-parallel", "--seqlen", str(seqlen), "--op-trace", "--GEMV", "reuse-GB", "--reuse-size", str(args.reuse_size), "--trace-file", f"{trace_root_dir}/model_parallel_embedding/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt"])
        else:
            if trace_needs_generation(f"{trace_root_dir}/pipeline_parallel_embedding/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt"):
                commands_generate_traces.append([python, "function_sim.py", model, "--n_heads", str(n_heads[args.model]), "--ffn_dim", str(ffn_size[args.model]), "--embedding", "--only-trace", "--num-channels", str(args.num_channels), "--num-banks", str(args.num_banks), "--max-seq-len", str(max_seq_len), "--channels-per-block", str(channels_per_block), "--pipeline-parallel", "--multi-tb-per-device", "--seqlen", str(seqlen), "--op-trace", "--GEMV", "reuse-GB", "--reuse-size", str(args.reuse_size), "--trace-file", f"{trace_root_dir}/pipeline_parallel_embedding/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt"])

    for seqlen in seqlen_list:
        if args.model_parallel:          
            for FC_devices in FC_devices_list:
                main_mode = model_parallel_main_mode(args)
                main_path = f"{trace_root_dir}/{main_mode}/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt"
                if trace_needs_generation(main_path):
                    commands_generate_traces.append([python, "function_sim.py", model, "--n_heads", str(n_heads[args.model]), "--ffn_dim", str(ffn_size[args.model]), "--only-trace", "--num-channels", str(args.num_channels), "--num-banks", str(args.num_banks), "--max-seq-len", str(max_seq_len), "--FC-devices", str(FC_devices), "--model-parallel", "--seqlen", str(seqlen), "--op-trace", "--GEMV", "reuse-GB", "--reuse-size", str(args.reuse_size), "--trace-file", main_path])
                    if args.inter_device_attention:
                        commands_generate_traces[-1].append("--inter-device-attention")
                    if args.kv_head_tp:
                        commands_generate_traces[-1].extend(["--kv-head-tp", "--tp-device-role", "main"])
                # KV-head TP ranks are compute-symmetric.  One fresh trace is
                # simulated and replicated across all TP ranks in postprocess;
                # only their CXL ingress/egress energy differs.
                if FC_devices == 1 or args.kv_head_tp:
                    continue
                helper_mode = model_parallel_helper_mode(args)
                helper_path = f"{trace_root_dir}/{helper_mode}/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt"
                if trace_needs_generation(helper_path):
                    commands_generate_traces.append(
                        model_parallel_helper_trace_command(
                            args, python, model, FC_devices, seqlen, max_seq_len, helper_path
                        )
                    )
        else:
            if channels_per_block < minimal_channel_per_block[args.model]:
                raise ValueError(f"Channels per block {channels_per_block} is less than minimal channel per block {minimal_channel_per_block[args.model]}")
            if trace_needs_generation(f"{trace_root_dir}/pipeline_parallel/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt"):
                commands_generate_traces.append([python, "function_sim.py", model, "--n_heads", str(n_heads[args.model]), "--ffn_dim", str(ffn_size[args.model]), "--only-trace", "--num-channels", str(args.num_channels), "--num-banks", str(args.num_banks), "--max-seq-len", str(max_seq_len), "--channels-per-block", str(channels_per_block), "--pipeline-parallel", "--multi-tb-per-device", "--seqlen", str(seqlen), "--op-trace", "--GEMV", "reuse-GB", "--reuse-size", str(args.reuse_size), "--trace-file", f"{trace_root_dir}/pipeline_parallel/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt"])

    if ewmul_pnm_enabled(args):
        for command in commands_generate_traces:
            command.append("--EWMUL-PNM")

    if args.systolic_pim:
        systolic_args = [
            "--systolic-pim",
            "--systolic-dim", str(args.systolic_dim),
            "--batch-size", str(args.batch_size),
        ]
        for command in commands_generate_traces:
            command.extend(systolic_args)
            if args.flash_attention:
                command.extend([
                    "--flash-attention",
                    "--flash-attention-block-size", str(args.flash_attention_block_size),
                ])

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.generate_trace_max_workers) as executor:
        futures = [executor.submit(run_checked_command, cmd) for cmd in commands_generate_traces]
        for future in concurrent.futures.as_completed(futures):
            future.result()

def run_command(command, log_file):
    print(command)
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    filtered_output = "\n".join(line for line in result.stdout.splitlines() if not line.startswith('['))
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w") as log:
        if result.returncode == 0:
            log.write(filtered_output)
        else:
            log.write(result.stdout)
            log.write(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {command}\n{result.stdout}\n{result.stderr}")

def simulate_trace(args, seqlen_list):
    commands_simulate_traces = []

    blocks_per_device = (TransformerBlock_number[args.model] - 1) // args.num_devices + 1
    channels_per_block = args.num_channels // blocks_per_device
    FC_devices_list = model_parallel_tp_values(args)
    trace_root_dir = trace_root(args)
    log_root_dir = log_root(args)

    def queue_simulation(trace_file, log_file):
        if not needs_simulation(args, log_file):
            return
        if missing_or_empty(trace_file):
            raise FileNotFoundError(f"Trace file is missing or empty: {trace_file}. Re-run with --generate_trace.")
        commands_simulate_traces.append((ramulator_command(args, trace_file, log_file), log_file))

    # Embedding is skipped for decode-only results.
    seqlen = args.prefill + args.decoding
    if not args.decode_only:
        if args.model_parallel:
            for FC_devices in FC_devices_list:
                log_file = f"{log_root_dir}/model_parallel_embedding/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt.log"
                trace_file = f"{trace_root_dir}/model_parallel_embedding/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt"
                queue_simulation(trace_file, log_file)
        else:
            log_file = f"{log_root_dir}/pipeline_parallel_embedding/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt.log"
            trace_file = f"{trace_root_dir}/pipeline_parallel_embedding/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt"
            queue_simulation(trace_file, log_file)

    for seqlen in seqlen_list:
        if args.model_parallel:
            for FC_devices in FC_devices_list:
                for mode in model_parallel_modes_for_tp(args, FC_devices):
                    log_file = f"{log_root_dir}/{mode}/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt.log"
                    trace_file = f"{trace_root_dir}/{mode}/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt"
                    queue_simulation(trace_file, log_file)
        else:
            for mode in ["pipeline_parallel"]:
                log_file = f"{log_root_dir}/{mode}/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt.log"
                trace_file = f"{trace_root_dir}/{mode}/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt"
                queue_simulation(trace_file, log_file)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.run_simulation_max_workers) as executor:
        futures = [executor.submit(run_command, cmd, log) for cmd, log in commands_simulate_traces]
        for future in concurrent.futures.as_completed(futures):
            future.result()

def process_results(args):
    print("Processing results...")
    if args.decode_only and args.model_parallel:
        mode_list = [model_parallel_main_mode(args)]
        if (
            not args.kv_head_tp
            and any(tp > 1 for tp in model_parallel_tp_values(args))
        ):
            mode_list.append(model_parallel_helper_mode(args))
    elif args.decode_only:
        mode_list = ["pipeline_parallel"]
    else:
        if args.model_parallel:
            mode_list = [model_parallel_main_mode(args), "model_parallel_embedding"]
            if (
                not args.kv_head_tp
                and any(tp > 1 for tp in model_parallel_tp_values(args))
            ):
                mode_list.append(model_parallel_helper_mode(args))
        else:
            mode_list = pipeline_parallel_mode_list
    log_root_dir = log_root(args)
    for mode in mode_list:
        compile_dir = f"{log_root_dir}/{mode}/{args.model}/"
        subprocess.run(["cp", "../trace/compile.sh", compile_dir])
        subprocess.run(["cp", "../trace/compile.py", compile_dir])

        # Run compile.sh and write output to result.txt
        result = subprocess.run(["bash", "compile.sh"], cwd=compile_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        with open(f"{compile_dir}/result.txt", "w") as result_file:
            result_file.write(result.stdout)
            result_file.write(result.stderr)

        # Run compile.py and write output to compiled_results.txt
        result = subprocess.run(["python3", "compile.py", "./result.txt"], cwd=compile_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        with open(f"{compile_dir}/compiled_results.txt", "w") as compiled_results_file:
            compiled_results_file.write(result.stdout)
            compiled_results_file.write(result.stderr)

def calculate_acc_latency(args, seqlen, tp=1, device_role="main"):
    latency = {}
    local_heads = n_heads[args.model] // tp if args.kv_head_tp else n_heads[args.model]
    local_hidden = embedding_size[args.model] // tp if args.kv_head_tp else embedding_size[args.model]
    local_kv_hidden = local_hidden // gqa_factor[args.model]
    local_ffn = ffn_size[args.model] // tp if args.kv_head_tp else ffn_size[args.model]
    rms_hidden = embedding_size[args.model]
    latency["RMSNorm_latency"] = rms_hidden / 16.00 / 16.00 / args.num_channels * ACCEL_CYCLE["VEC_ADD"]
    if ewmul_pnm_enabled(args):
        latency["RMSNorm_latency"] += (
            rms_hidden / 16.00 / args.num_channels
            * ACCEL_CYCLE["VEC_MUL"]
            * 3.00
        )
    latency["RMSNorm_latency"] += SB_RD_CYCLE + SB_WR_CYCLE + 1.00
    latency["RMSNorm_latency"] += RV_RMSNorm_CYCLE
    latency["RMSNorm_latency"] = float(2.00 * latency["RMSNorm_latency"]) / float(FREQ / KILO)
    latency["Softmax_latency"] = seqlen * local_heads / 16.00 / args.num_channels * ACCEL_CYCLE["EXP"]
    latency["Softmax_latency"] += seqlen * local_heads / 16.00 / args.num_channels * ACCEL_CYCLE["VEC_ADD"]
    latency["Softmax_latency"] += seqlen * local_heads / 16.00 / args.num_channels * ACCEL_CYCLE["VEC_MUL"] * 2.00
    latency["Softmax_latency"] += local_heads * 1.00 * SB_RD_CYCLE
    latency["Softmax_latency"] += local_heads * RV_SFT_CYCLE_PIPELINE
    latency["Softmax_latency"] = float(latency["Softmax_latency"]) / float(FREQ / KILO)
    latency["Softmax_latency"] *= softmax_exposure_factor(args, seqlen)
    if getattr(args, "flash_attention", False):
        flash_score_buffer_bytes = (
            args.flash_attention_block_size * local_heads * 2
        )
        if flash_score_buffer_bytes > SHARED_BUFFER_CAPACITY_BYTES:
            raise ValueError(
                "FlashAttention score block requires "
                f"{flash_score_buffer_bytes} B of shared buffer, exceeding the "
                f"cent_dev-compatible {SHARED_BUFFER_CAPACITY_BYTES} B capacity"
            )
        # Keep the cent_dev abstraction: online-softmax block merge work is a
        # single vector-add term, while the score-to-SV stream itself is
        # represented by the adjacent QK/SV trace commands.
        latency["FlashAttention_latency"] = (
            seqlen // args.flash_attention_block_size
            * local_heads / 16.00 / args.num_channels
            * ACCEL_CYCLE["VEC_ADD"]
            / float(FREQ / KILO)
        )
    latency["RotEmbed_latency"] = (
        local_hidden + local_kv_hidden
    ) * RV_ROTEmbed_CYCLE_PIPELINE
    if ewmul_pnm_enabled(args):
        latency["RotEmbed_latency"] += (
            (local_hidden + local_kv_hidden)
            / 16.00
            / args.num_channels
            * ACCEL_CYCLE["VEC_MUL"]
        )
        latency["EWMULActivation_latency"] = (
            local_ffn
            / 16.00
            / args.num_channels
            * 2.00
            * ACCEL_CYCLE["VEC_MUL"]
            / float(FREQ / KILO)
        )
    latency["RotEmbed_latency"] = float(latency["RotEmbed_latency"]) / float(FREQ / KILO)
    if args.kv_head_tp:
        shape = kv_head_tp_shape(
                dim=embedding_size[args.model],
                query_heads=n_heads[args.model],
                kv_heads=n_heads[args.model] // gqa_factor[args.model],
                ffn_dim=ffn_size[args.model],
                tp=tp,
            )
        layout = KVHeadTPLayout(
            shape=shape,
            num_channels=args.num_channels,
            banks_per_channel=args.num_banks,
            max_seq_len=(args.max_seq_len if args.max_seq_len is not None else seqlen),
        )
        if args.systolic_pim:
            systolic_layout = SystolicTPLayout(
                shape=shape,
                num_channels=args.num_channels,
                banks_per_channel=args.num_banks,
                max_seq_len=(
                    args.max_seq_len if args.max_seq_len is not None else seqlen
                ),
                systolic_height=args.systolic_dim,
            )
            reduction_work = systolic_layout.pnm_reduction_adds(seqlen)
            for name in ("q", "kv", "sv"):
                groups_per_channel = math.ceil(
                    reduction_work[name] / 16.0 / args.num_channels
                )
                latency[f"{name.upper()}Reduction_latency"] = (
                    groups_per_channel * ACCEL_CYCLE["VEC_ADD"]
                ) / float(FREQ / KILO)
        else:
            reduction_groups_per_channel = math.ceil(
                layout.v_reduction_adds(seqlen)
                / 16.0
                / args.num_channels
            )
            latency["SVReduction_latency"] = (
                reduction_groups_per_channel * ACCEL_CYCLE["VEC_ADD"]
            ) / float(FREQ / KILO)
    else:
        latency["SVReduction_latency"] = 0.0
    return latency

def load_data_point(args, seqlen, FC_devices, channels_per_block, PCIe_lanes_per_device, blocks_per_device, embedding_latency, utilized_devices, pp, tp):

    log_root_dir = log_root(args)
    if args.model_parallel:
        main_mode = model_parallel_main_mode(args)
        path = f"{log_root_dir}/{main_mode}/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt.log"
    else:
        path = f"{log_root_dir}/pipeline_parallel/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt.log"
    if missing_or_empty(path):
        raise FileNotFoundError(f"Simulation log is missing or empty: {path}. Re-run --generate_trace --simulate_trace before --update_csv.")
    stats = command_processor(path)
    main_pim_latency = stats["latency"]
    helper_path = None
    helper_stats = None
    helper_pim_latency = 0.0
    if args.model_parallel and FC_devices > 1 and not args.kv_head_tp:
        helper_mode = model_parallel_helper_mode(args)
        helper_path = f"{log_root_dir}/{helper_mode}/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt.log"
        if missing_or_empty(helper_path):
            raise FileNotFoundError(
                f"Simulation log is missing or empty: {helper_path}. "
                "Re-run --generate_trace --simulate_trace before --update_csv."
            )
        helper_stats = command_processor(helper_path)
        helper_pim_latency = helper_stats["latency"]
    elif args.kv_head_tp and FC_devices > 1:
        # Every head-TP rank executes the same local tensor shapes and mapping.
        helper_pim_latency = main_pim_latency
    pim_latency = max(main_pim_latency, helper_pim_latency)

    if args.model_parallel:
        if args.kv_head_tp:
            tp_collective_cxl_latency = kv_head_tp_latency(
                embedding_size[args.model] * args.batch_size,
                PCIe_lanes_per_device,
                FC_devices,
                args.num_devices,
            )
            pp_handoff_cxl_latency = (
                vector_latency(
                    embedding_size[args.model] * args.batch_size,
                    PCIe_lanes_per_device,
                )
                if pp > 1
                else 0.0
            )
            cxl_latency = tp_collective_cxl_latency + pp_handoff_cxl_latency
        elif "Llama" in args.model:
            cxl_latency = llama_latency([embedding_size[args.model] * args.batch_size, ffn_size[args.model] * args.batch_size], PCIe_lanes_per_device, FC_devices, args.num_devices)
        else:
            cxl_latency = gpt_latency([embedding_size[args.model] * args.batch_size, ffn_size[args.model] * args.batch_size], PCIe_lanes_per_device, FC_devices, args.num_devices)
        embedding_latency_data = 0.00 if args.decode_only else embedding_latency['model_parallel'][FC_devices]
    else:
        cxl_latency = vector_latency(embedding_size[args.model] * args.batch_size, PCIe_lanes_per_device)
        embedding_latency_data = 0.00 if args.decode_only else embedding_latency['pipeline_parallel'][channels_per_block]
    if not args.kv_head_tp:
        tp_collective_cxl_latency = 0.0
        pp_handoff_cxl_latency = cxl_latency if pp > 1 else 0.0
    main_acc_latency_dict = calculate_acc_latency(args, seqlen, FC_devices, "main")
    helper_acc_latency_dict = calculate_acc_latency(args, seqlen, FC_devices, "helper")
    softmax_pipeline_factor = softmax_exposure_factor(args, seqlen)
    main_acc_latency = sum(main_acc_latency_dict.values()) * blocks_per_device * args.batch_size
    main_exposed_softmax_latency = (
        main_acc_latency_dict["Softmax_latency"]
        * blocks_per_device
        * args.batch_size
    )
    main_full_softmax_latency = (
        main_exposed_softmax_latency / softmax_pipeline_factor
    )
    helper_acc_latency = (
        sum(helper_acc_latency_dict.values()) * blocks_per_device * args.batch_size
        if helper_stats is not None or (args.kv_head_tp and FC_devices > 1)
        else 0.0
    )
    acc_latency = max(main_acc_latency, helper_acc_latency)
    if args.kv_head_tp:
        critical_local_latency = max(
            main_pim_latency + main_acc_latency,
            helper_pim_latency + helper_acc_latency,
        )
        transformer_block_latency = critical_local_latency + cxl_latency
    else:
        critical_local_latency = pim_latency + acc_latency
        transformer_block_latency = critical_local_latency + cxl_latency
    token_latency = transformer_block_latency * TransformerBlock_number[args.model] + embedding_latency_data
    if not args.decode_only:
        token_latency += InOut_latency
    throughput = 1000 / token_latency * pp * args.batch_size

    energy_token = {}
    if args.kv_head_tp:
        main_PCIE, helper_PCIE, system_PCIE = kv_head_tp_pcie_bits(
            embedding_size[args.model] * args.batch_size, FC_devices
        )
        tp_collective_PCIE = system_PCIE
        pp_handoff_PCIE = (
            embedding_size[args.model] * args.batch_size * 16 if pp > 1 else 0
        )
        # Charge the one source-side pipeline transfer once per TP group.  TP
        # helper ranks retain only their two all-reduce broadcast shares.
        main_PCIE += pp_handoff_PCIE
        system_PCIE += pp_handoff_PCIE
        local_heads = n_heads[args.model] // FC_devices
        local_hidden = embedding_size[args.model] // FC_devices
    else:
        PCIE = (embedding_size[args.model] * 10 + ffn_size[args.model] * 2 if args.model_parallel else embedding_size[args.model]) * args.batch_size
        main_PCIE = helper_PCIE = system_PCIE = PCIE
        tp_collective_PCIE = 0
        pp_handoff_PCIE = system_PCIE if pp > 1 else 0
        local_heads = n_heads[args.model]
        local_hidden = embedding_size[args.model]
    energy_main, latency_main = power_calculator(
        stats, main_PCIE, local_heads, local_hidden, seqlen, gqa_factor[args.model],
        dram_power_impl=args.dram_power_impl,
        dram_energy_model=args.dram_energy_model,
        command_trace_prefix=command_trace_prefix_for_log(path),
        rmsnorm_hidden_dim=(embedding_size[args.model] if args.kv_head_tp else None),
    )
    reduction_adds = 0
    local_ffn_elements = ffn_size[args.model]
    if args.kv_head_tp:
        kv_layout = KVHeadTPLayout(
            shape=kv_head_tp_shape(
                dim=embedding_size[args.model],
                query_heads=n_heads[args.model],
                kv_heads=n_heads[args.model] // gqa_factor[args.model],
                ffn_dim=ffn_size[args.model],
                tp=FC_devices,
            ),
            num_channels=args.num_channels,
            banks_per_channel=args.num_banks,
            max_seq_len=(args.max_seq_len if args.max_seq_len is not None else seqlen),
        )
        if args.systolic_pim:
            systolic_layout = SystolicTPLayout(
                shape=kv_layout.shape,
                num_channels=args.num_channels,
                banks_per_channel=args.num_banks,
                max_seq_len=(
                    args.max_seq_len if args.max_seq_len is not None else seqlen
                ),
                systolic_height=args.systolic_dim,
            )
            reduction_adds = sum(
                systolic_layout.pnm_reduction_adds(seqlen).values()
            )
        else:
            reduction_adds = kv_layout.v_reduction_adds(seqlen)
        local_ffn_elements = kv_layout.shape.local_ffn_dim

    ewmul_elements = 0
    if ewmul_pnm_enabled(args):
        # Two RMSNorms, each with x.pow plus two element-wise multiplies;
        # Q/K RoPE; and two multiplies for SiLU(W1) * W3.
        ewmul_elements = (
            6 * embedding_size[args.model]
            + local_hidden
            + local_hidden // gqa_factor[args.model]
            + 2 * local_ffn_elements
        )
    pnm_energy = kv_head_tp_pnm_dynamic_energy(
        stats,
        reduction_adds=reduction_adds,
        ewmul_elements=ewmul_elements,
    )
    if reduction_adds or ewmul_elements:
        energy_main = add_energy_terms(energy_main, pnm_energy)
    energy_main = adjust_systolic_energy(energy_main, args)
    if args.model_parallel:
        if args.kv_head_tp and FC_devices > 1:
            # Reuse the symmetric compute trace, but apply each helper's own
            # share of the two all-reduce payloads.
            energy_helper, _latency_helper = power_calculator(
                stats, helper_PCIE, local_heads, local_hidden, seqlen, gqa_factor[args.model],
                dram_power_impl=args.dram_power_impl,
                dram_energy_model=args.dram_energy_model,
                command_trace_prefix=command_trace_prefix_for_log(path),
                rmsnorm_hidden_dim=embedding_size[args.model],
            )
            if reduction_adds or ewmul_elements:
                energy_helper = add_energy_terms(energy_helper, pnm_energy)
            energy_helper = adjust_systolic_energy(energy_helper, args)
            for comp in energy_main.keys():
                energy_token[comp] = (
                    energy_main[comp] + energy_helper[comp] * (FC_devices - 1)
                ) * TransformerBlock_number[args.model]
        elif helper_stats is None:
            for comp in energy_main.keys():
                energy_token[comp] = energy_main[comp] * TransformerBlock_number[args.model]
        else:
            energy_helper, _latency_helper = power_calculator(
                helper_stats, helper_PCIE, local_heads, local_hidden, seqlen, gqa_factor[args.model],
                dram_power_impl=args.dram_power_impl,
                dram_energy_model=args.dram_energy_model,
                command_trace_prefix=command_trace_prefix_for_log(helper_path),
                device_role=(
                    "kv_head_helper"
                    if args.kv_head_tp
                    else ("inter_device_helper" if args.inter_device_attention else "fc_helper")
                ),
            )
            energy_helper = adjust_systolic_energy(energy_helper, args)
            for comp in energy_main.keys():
                energy_token[comp] = (
                    energy_main[comp] + energy_helper[comp] * (FC_devices - 1)
                ) * TransformerBlock_number[args.model]
    else:
        for comp in energy_main.keys():
            energy_token[comp] = energy_main[comp] * utilized_devices
    batch_group_energy, total_energy = normalize_batch_group_energy(
        energy_token, args.batch_size
    )
    # Energy per output token times sustained output rate gives the steady-
    # state power of all participating devices, including pipeline overlap.
    total_power = total_energy * throughput / 1000.0
    device_utilization = 1.0 * utilized_devices / args.num_devices
                        
    new_result = {
        'Model': args.model,
        'Device number': args.num_devices,
        'Pipeline parallelism': pp,
        'Tensor parallelism': tp,
        'Batch size': args.batch_size,
        'Systolic pim': args.systolic_pim,
        'Systolic dim': args.systolic_dim,
        'EWMUL PNM requested': ewmul_pnm_requested(args),
        'EWMUL PNM effective': ewmul_pnm_enabled(args),
        'EWMUL PNM provenance': 'native',
        'Flash attention': args.flash_attention,
        'Flash attention block size': (
            args.flash_attention_block_size if args.flash_attention else 0
        ),
        'Pipelined softmax': getattr(args, 'pipelined_softmax', False),
        'Softmax pipeline startup tokens': softmax_pipeline_startup_tokens(args),
        'Softmax exposure factor': softmax_pipeline_factor,
        'Parallel SRAM banks': getattr(
            args, 'parallel_sram', int(SRAM_IO_PARALLEL)
        ),
        'Shared buffer capacity (bytes)': SHARED_BUFFER_CAPACITY_BYTES,
        'Channels per device': args.num_channels,
        'Banks per device': args.num_banks,
        'Channels per block': channels_per_block,
        'Sequence length': seqlen,
        'Context window': args.max_seq_len if args.max_seq_len is not None else seqlen,
        'PIM latency': pim_latency,
        'Main PIM latency': main_pim_latency,
        'Helper PIM latency': helper_pim_latency,
        'CXL latency': cxl_latency,
        'TP collective CXL latency': tp_collective_cxl_latency,
        'PP handoff CXL latency': pp_handoff_cxl_latency,
        'Acc latency': acc_latency,
        'Main Acc latency': main_acc_latency,
        'Helper Acc latency': helper_acc_latency,
        'Full softmax latency': main_full_softmax_latency,
        'Exposed softmax latency': main_exposed_softmax_latency,
        'Q reduction latency': main_acc_latency_dict.get('QReduction_latency', 0.0) * blocks_per_device * args.batch_size,
        'KV reduction latency': main_acc_latency_dict.get('KVReduction_latency', 0.0) * blocks_per_device * args.batch_size,
        'SV reduction latency': main_acc_latency_dict.get('SVReduction_latency', 0.0) * blocks_per_device * args.batch_size,
        'EWMUL activation latency': main_acc_latency_dict.get('EWMULActivation_latency', 0.0) * blocks_per_device * args.batch_size,
        'Critical local latency': critical_local_latency,
        'TransformerBlock latency': transformer_block_latency,
        'Embedding latency': embedding_latency_data,
        'Token latency (ms)': token_latency,
        'Batch group latency (ms)': token_latency,
        'Throughput (tokens/s)': throughput,
        'Token energy (mJ)': total_energy,
        'Batch group energy (mJ)': batch_group_energy,
        'Total power (W)': total_power,
        'Device utilization': device_utilization,
        'Attention mapping': (
            'kv_head'
            if args.kv_head_tp
            else ('inter_device' if args.inter_device_attention else ('master' if args.model_parallel else 'pipeline'))
        ),
        'KV cache mapping': 'kv_head_sharded' if args.kv_head_tp else ('fully_tp_sharded' if args.inter_device_attention else 'master_local'),
        'CXL payload (bits/block)': system_PCIE,
        'TP collective CXL payload (bits/block)': tp_collective_PCIE,
        'PP handoff CXL payload (bits/block)': pp_handoff_PCIE,
        'DRAM energy model': args.dram_energy_model,
    }
    new_result_df = pd.DataFrame([new_result])
    return new_result_df


def update_csv(args, seqlen_list):

    print("Updating simulation results to CSV file...")
    log_root_dir = log_root(args)
    os.makedirs(os.path.dirname(args.simulation_result_path) or ".", exist_ok=True)

    if os.path.exists(args.simulation_result_path):
        results_df = pd.read_csv(args.simulation_result_path)
        # ``KV repack latency`` was an obsolete analytical term.  Explicit
        # K/V cache W_MEM commands remain in the trace and are still charged.
        results_df = results_df.drop(
            columns=['KV repack latency', 'Fused activation latency'],
            errors='ignore',
        )
        if 'DRAM energy model' not in results_df.columns:
            results_df['DRAM energy model'] = 'legacy'
        if 'Context window' not in results_df.columns:
            results_df['Context window'] = results_df['Sequence length']
        if 'Main PIM latency' not in results_df.columns:
            results_df['Main PIM latency'] = results_df['PIM latency']
        if 'Helper PIM latency' not in results_df.columns:
            results_df['Helper PIM latency'] = 0.0
        if 'Attention mapping' not in results_df.columns:
            results_df['Attention mapping'] = 'legacy_unspecified'
        if 'Activation placement' in results_df.columns:
            results_df['Legacy activation placement'] = results_df[
                'Activation placement'
            ]
            results_df = results_df.drop(columns=['Activation placement'])
        if 'EWMUL PNM requested' not in results_df.columns:
            results_df['EWMUL PNM requested'] = pd.NA
        if 'EWMUL PNM effective' not in results_df.columns:
            results_df['EWMUL PNM effective'] = pd.NA
        if 'EWMUL PNM provenance' not in results_df.columns:
            results_df['EWMUL PNM provenance'] = (
                'legacy_activation_incompatible'
                if 'Legacy activation placement' in results_df.columns
                else 'legacy_unspecified_incompatible'
            )
        if 'Flash attention' not in results_df.columns:
            results_df['Flash attention'] = False
        if 'Flash attention block size' not in results_df.columns:
            results_df['Flash attention block size'] = 0
        if 'Pipelined softmax' not in results_df.columns:
            results_df['Pipelined softmax'] = False
        if 'Softmax pipeline startup tokens' not in results_df.columns:
            results_df['Softmax pipeline startup tokens'] = 0
        if 'Softmax exposure factor' not in results_df.columns:
            results_df['Softmax exposure factor'] = 1.0
        if 'Parallel SRAM banks' not in results_df.columns:
            results_df['Parallel SRAM banks'] = 1
        if 'Shared buffer capacity (bytes)' not in results_df.columns:
            results_df['Shared buffer capacity (bytes)'] = 512 * 1024
    else:
        columns = ['Model', 'Device number', 'Pipeline parallelism', 'Tensor parallelism', 'Batch size', 'Systolic pim', 'Systolic dim', 'EWMUL PNM requested', 'EWMUL PNM effective', 'EWMUL PNM provenance', 'Flash attention', 'Flash attention block size', 'Pipelined softmax', 'Softmax pipeline startup tokens', 'Softmax exposure factor', 'Parallel SRAM banks', 'Shared buffer capacity (bytes)', 'Channels per device', 'Banks per device', 'Channels per block', 'Sequence length', 'Context window', 'PIM latency', 'Main PIM latency', 'Helper PIM latency', 'CXL latency', 'Full softmax latency', 'Exposed softmax latency', 'TransformerBlock latency', 'Embedding latency', 'Token latency (ms)', 'Throughput (tokens/s)', 'Token energy (mJ)', 'Total power (W)', 'Device utilization', 'Attention mapping', 'DRAM energy model']
        results_df = pd.DataFrame(columns=columns)

    embedding_latency = {'pipeline_parallel': {}, 'model_parallel': {}}

    if args.model_parallel:
        FC_devices_list = model_parallel_tp_values(args)

    if args.decode_only:
        pass
    elif args.model_parallel:
        for FC_devices in FC_devices_list:
            embedding_compile_dir = f"{log_root_dir}/model_parallel_embedding/{args.model}/"
            with open(f"{embedding_compile_dir}/compiled_results.txt", "r") as compiled_results_file:
                lines = compiled_results_file.readlines()
                for line in lines:
                    filename, latency = line.split()[0], line.split()[1]
                    FC_devices = int(filename.split('_')[1])
                    embedding_latency["model_parallel"][FC_devices] = float(latency)
        missing_FC_devices = [FC_devices for FC_devices in FC_devices_list if FC_devices not in embedding_latency["model_parallel"]]
        if missing_FC_devices:
            raise ValueError(
                f"Missing model-parallel embedding latency for FC_devices={missing_FC_devices} in "
                f"{log_root_dir}/model_parallel_embedding/{args.model}/compiled_results.txt. "
                "Check for missing or empty trace/log files and re-run --generate_trace --simulate_trace --process_results."
            )
    else:
        embedding_compile_dir = f"{log_root_dir}/pipeline_parallel_embedding/{args.model}/"
        with open(f"{embedding_compile_dir}/compiled_results.txt", "r") as compiled_results_file:
            lines = compiled_results_file.readlines()
            for line in lines:
                filename, latency = line.split()[0], line.split()[1]
                channels_per_block = int(filename.split('_')[1])
                embedding_latency["pipeline_parallel"][channels_per_block] = float(latency)

    for seqlen in seqlen_list:

        PCIe_lanes_per_device = args.PCIE_lanes // args.num_devices

        if args.model_parallel:
            blocks_per_device = 1
            channels_per_block = args.num_channels // blocks_per_device
            utilized_devices = args.num_devices
            for FC_devices in FC_devices_list:
                pp = args.num_devices // FC_devices
                tp = FC_devices
                new_result_df = load_data_point(args, seqlen, FC_devices, channels_per_block, PCIe_lanes_per_device, blocks_per_device, embedding_latency, utilized_devices, pp, tp)
                results_df = pd.concat([results_df, new_result_df], ignore_index=True)
        else:
            pp = TransformerBlock_number[args.model]
            tp = 1
            pp_per_device = (pp - 1) // args.num_devices + 1
            blocks_per_device =  pp_per_device * (TransformerBlock_number[args.model] // pp)
            channels_per_block = args.num_channels // blocks_per_device
            if channels_per_block < minimal_channel_per_block[args.model]:
                continue
            utilized_devices = (TransformerBlock_number[args.model] - 1) // blocks_per_device + 1
            new_result_df = load_data_point(args, seqlen, 0, channels_per_block, PCIe_lanes_per_device, blocks_per_device, embedding_latency, utilized_devices, pp, tp)
            results_df = pd.concat([results_df, new_result_df], ignore_index=True)

    # Save the DataFrame to a CSV file
    # The SRAM organization is a hardware revision, not an additional sweep
    # dimension.  Reprocessing an existing workload therefore replaces its
    # legacy single-port result with the current banked-SRAM result.
    results_df = results_df.drop_duplicates(subset=['Model', 'Device number', 'Pipeline parallelism', 'Tensor parallelism', 'Batch size', 'Systolic pim', 'Systolic dim', 'EWMUL PNM effective', 'EWMUL PNM provenance', 'Flash attention', 'Flash attention block size', 'Pipelined softmax', 'Channels per device', 'Banks per device', 'Channels per block', 'Sequence length', 'Context window', 'Attention mapping', 'DRAM energy model'], keep='last')
    results_df = results_df.sort_values(by=['Model', 'Device number', 'Pipeline parallelism', 'Tensor parallelism', 'Batch size', 'Systolic pim', 'Systolic dim', 'EWMUL PNM provenance', 'EWMUL PNM effective', 'Flash attention', 'Flash attention block size', 'Pipelined softmax', 'Parallel SRAM banks', 'Channels per device', 'Banks per device', 'Channels per block', 'Context window', 'Sequence length', 'DRAM energy model'])
    results_df.to_csv(args.simulation_result_path, index=False)
    # print(results_df)

def process_throughputs(args):

    print("Processing results to CSV file...")

    if os.path.exists(args.simulation_result_path):
        df_simulation = pd.read_csv(args.simulation_result_path)
    else:
        raise ValueError(f"File {args.simulation_result_path} does not exist. Generate simulation results first.")
    if 'Banks per device' in df_simulation.columns:
        df_simulation = df_simulation[df_simulation['Banks per device'] == args.num_banks]
    if 'DRAM energy model' in df_simulation.columns:
        df_simulation = df_simulation[df_simulation['DRAM energy model'] == args.dram_energy_model]
    if 'Parallel SRAM banks' in df_simulation.columns:
        df_simulation = df_simulation[
            df_simulation['Parallel SRAM banks'] == args.parallel_sram
        ]
    if 'Systolic pim' in df_simulation.columns:
        df_simulation = df_simulation[df_simulation['Systolic pim'] == args.systolic_pim]
    if args.systolic_pim and 'Systolic dim' in df_simulation.columns:
        df_simulation = df_simulation[df_simulation['Systolic dim'] == args.systolic_dim]
    required_ewmul_columns = {
        'EWMUL PNM effective', 'EWMUL PNM provenance'
    }
    if not required_ewmul_columns.issubset(df_simulation.columns):
        raise ValueError(
            "Simulation results use legacy --activation provenance; regenerate "
            "them with the --EWMUL_PNM model"
        )
    df_simulation = df_simulation[
        (df_simulation['EWMUL PNM effective'] == ewmul_pnm_enabled(args))
        & (df_simulation['EWMUL PNM provenance'] == 'native')
    ]
    if 'Pipelined softmax' in df_simulation.columns:
        df_simulation = df_simulation[
            df_simulation['Pipelined softmax']
            == getattr(args, 'pipelined_softmax', False)
        ]
    
    if os.path.exists(args.processed_result_path):
        results_df = pd.read_csv(args.processed_result_path)
    else:
        columns = ['Model', 'Device number', 'Banks per device', 'Seqlen', 'Pipeline parallelism', 'Tensor parallelism', 'Batch size', 'Systolic pim', 'Systolic dim', 'EWMUL PNM effective', 'Pipelined softmax', 'Phase', 'DRAM energy model', 'Total Latency (s)', 'Throughput (tokens/s)', 'Energy per Token (mJ)', 'Total power (W)']
        results_df = pd.DataFrame(columns=columns)


    if args.model_parallel:

        FC_devices_list = model_parallel_tp_values(args)

        for FC_Devices in FC_devices_list:

            pp = args.num_devices // FC_Devices
            tp = FC_Devices
            df = df_simulation[(df_simulation['Model'] == args.model) & (df_simulation['Pipeline parallelism'] == pp) & (df_simulation['Tensor parallelism'] == tp)]
            # print("tp", tp, len(df))

            if args.phase == "prefill":
                df = df[(df['Sequence length'] <= args.prefill)]
                seqlen = args.prefill
            elif args.phase == "decoding":
                df = df[((args.prefill + args.decoding) >= df['Sequence length']) & (df['Sequence length'] > args.prefill)]
                seqlen = args.decoding
            elif args.phase == "end2end":
                df = df[((args.prefill + args.decoding) >= df['Sequence length'])]
                seqlen = args.prefill + args.decoding

            average_throughput = df['Throughput (tokens/s)'].mean()
            average_energy = df['Token energy (mJ)'].mean()
            total_latency = df['Token latency (ms)'].mean() * seqlen / 1000
            total_power = df['Total power (W)'].mean()

            new_result = {
                'Model': args.model,
                'Device number': args.num_devices,
                'Banks per device': args.num_banks,
                'Seqlen': args.prefill + args.decoding,
                'Pipeline parallelism': pp,
                'Tensor parallelism': tp,
                'Batch size': args.batch_size,
                'Systolic pim': args.systolic_pim,
                'Systolic dim': args.systolic_dim,
                'EWMUL PNM effective': ewmul_pnm_enabled(args),
                'Pipelined softmax': getattr(args, 'pipelined_softmax', False),
                'Phase': args.phase,
                'DRAM energy model': args.dram_energy_model,
                'Total Latency (s)': total_latency,
                'Throughput (tokens/s)': average_throughput,
                'Energy per Token (mJ)': average_energy,
                'Total power (W)': total_power
            }
            new_result_df = pd.DataFrame([new_result])
            results_df = pd.concat([results_df, new_result_df], ignore_index=True)

    else:

        df = df_simulation[(df_simulation['Model'] == args.model) & (df_simulation['Pipeline parallelism'] == TransformerBlock_number[args.model]) & (df_simulation['Tensor parallelism'] == 1)]

        if args.phase == "prefill":
            df = df[(df['Sequence length'] <= args.prefill)]
            seqlen = args.prefill
        elif args.phase == "decoding":
            df = df[((args.prefill + args.decoding) >= df['Sequence length']) & (df['Sequence length'] > args.prefill)]
            seqlen = args.decoding
        elif args.phase == "end2end":
            df = df[((args.prefill + args.decoding) >= df['Sequence length'])]
            seqlen = args.prefill + args.decoding

        average_throughput = df['Throughput (tokens/s)'].mean()
        average_energy = df['Token energy (mJ)'].mean()
        total_latency = df['Token latency (ms)'].mean() * seqlen / 1000
        total_power = df['Total power (W)'].mean()

        new_result = {
            'Model': args.model,
            'Device number': args.num_devices,
            'Banks per device': args.num_banks,
            'Seqlen': args.prefill + args.decoding,
            'Pipeline parallelism': TransformerBlock_number[args.model],
            'Tensor parallelism': 1,
            'Batch size': args.batch_size,
            'Systolic pim': args.systolic_pim,
            'Systolic dim': args.systolic_dim,
            'EWMUL PNM effective': ewmul_pnm_enabled(args),
            'Pipelined softmax': getattr(args, 'pipelined_softmax', False),
            'Phase': args.phase,
            'DRAM energy model': args.dram_energy_model,
            'Total Latency (s)': total_latency,
            'Throughput (tokens/s)': average_throughput,
            'Energy per Token (mJ)': average_energy,
            'Total power (W)': total_power
        }
        new_result_df = pd.DataFrame([new_result])
    results_df = pd.concat([results_df, new_result_df], ignore_index=True)
    
    results_df = results_df.drop_duplicates()
    results_df = results_df.sort_values(by=['Model', 'Device number', 'Banks per device', 'Seqlen', 'Pipeline parallelism', 'Tensor parallelism', 'EWMUL PNM effective', 'Pipelined softmax', 'Phase'])
    os.makedirs(os.path.dirname(args.processed_result_path) or ".", exist_ok=True)
    results_df.to_csv(args.processed_result_path, index=False)


if __name__ == "__main__":


    args = get_args()

    if args.seqlen:
        seqlen_list = args.seqlen
    else:
        seqlen_list = [i * args.seqlen_gap for i in range(1, (args.prefill + args.decoding) // args.seqlen_gap + 1)]
        
    runtime_modes = pipeline_parallel_mode_list + model_parallel_mode_list + [INTER_DEVICE_HELPER_MODE]
    for mode in dict.fromkeys(runtime_modes):
        os.makedirs(f"{trace_root(args)}/{mode}/{args.model}", exist_ok=True)
        os.makedirs(f"{log_root(args)}/{mode}/{args.model}", exist_ok=True)

    if args.generate_trace:
        generate_trace(args, seqlen_list)
        
    if args.simulate_trace:
        simulate_trace(args, seqlen_list)

    if args.process_results:
        process_results(args)
        
    if args.update_csv:
        update_csv(args, seqlen_list)
        
    if args.process_throughputs:
        process_throughputs(args)
