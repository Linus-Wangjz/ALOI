import os
import math
import pandas as pd
import argparse
import subprocess
import concurrent.futures
import sys
from pathlib import Path
from cxl_latency import llama_latency, gpt_latency, vector_latency
from cent_power_calculator import DRAM_ENERGY_MODELS, ACCEL_CYCLE, power_calculator, command_processor, command_trace_prefix_for_log, set_channel_count, KILO, FREQ, SB_RD_CYCLE, SB_WR_CYCLE, RV_RMSNorm_CYCLE, RV_ROTEmbed_CYCLE, RV_SFT_CYCLE_PIPELINE
from utils import InOut_latency, n_heads, gqa_factor, embedding_size, ffn_size, TransformerBlock_number, minimal_channel_per_block, pipeline_parallel_mode_list, model_parallel_mode_list

def get_args():
    parser = argparse.ArgumentParser('run_scripts.py')
    parser.add_argument("--num_channels", type=int, help="Number of channels per device", default=32)
    parser.add_argument("--num_banks", "--num-banks", dest="num_banks", type=int, help="Number of banks per channel", default=16)
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
    parser.add_argument("--inter-device-attention", action="store_true")
    args = parser.parse_args()
    set_channel_count(args.num_channels)
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


def trace_root(args):
    if args.trace_root:
        return args.trace_root
    name = experiment_name(args)
    # LPDDR4X nCCD sweeps reuse one functional trace set; only Ramulator
    # timing/log artifacts differ between nCCD values.
    trace_name = name.split("_nCCD", 1)[0]
    return os.path.join("trace", trace_name)


def log_root(args):
    if args.log_root:
        return args.log_root
    _, nccd = split_experiment_name(experiment_name(args))
    if nccd:
        return os.path.join(experiment_output_dir(args), f"ramulator_{nccd}")
    return os.path.join(experiment_output_dir(args), "ramulator")


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
    FC_devices_list = factorize(args.num_devices)
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
                if trace_needs_generation(f"{trace_root_dir}/model_parallel/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt"):
                    commands_generate_traces.append([python, "function_sim.py", model, "--n_heads", str(n_heads[args.model]), "--ffn_dim", str(ffn_size[args.model]), "--only-trace", "--num-channels", str(args.num_channels), "--num-banks", str(args.num_banks), "--max-seq-len", str(max_seq_len), "--FC-devices", str(FC_devices), "--model-parallel", "--seqlen", str(seqlen), "--op-trace", "--GEMV", "reuse-GB", "--reuse-size", str(args.reuse_size), "--trace-file", f"{trace_root_dir}/model_parallel/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt"])
                    if args.inter_device_attention:
                        commands_generate_traces[-1].append("--inter-device-attention")
                if trace_needs_generation(f"{trace_root_dir}/model_parallel_FC/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt"):
                    commands_generate_traces.append([python, "function_sim.py", model, "--n_heads", str(n_heads[args.model]), "--ffn_dim", str(ffn_size[args.model]), "--only-FC", "--only-trace", "--num-channels", str(args.num_channels), "--num-banks", str(args.num_banks), "--max-seq-len", str(max_seq_len), "--FC-devices", str(FC_devices), "--model-parallel", "--seqlen", str(seqlen), "--op-trace", "--GEMV", "reuse-GB", "--reuse-size", str(args.reuse_size), "--trace-file", f"{trace_root_dir}/model_parallel_FC/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt"])
        else:
            if channels_per_block < minimal_channel_per_block[args.model]:
                raise ValueError(f"Channels per block {channels_per_block} is less than minimal channel per block {minimal_channel_per_block[args.model]}")
            if trace_needs_generation(f"{trace_root_dir}/pipeline_parallel/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt"):
                commands_generate_traces.append([python, "function_sim.py", model, "--n_heads", str(n_heads[args.model]), "--ffn_dim", str(ffn_size[args.model]), "--only-trace", "--num-channels", str(args.num_channels), "--num-banks", str(args.num_banks), "--max-seq-len", str(max_seq_len), "--channels-per-block", str(channels_per_block), "--pipeline-parallel", "--multi-tb-per-device", "--seqlen", str(seqlen), "--op-trace", "--GEMV", "reuse-GB", "--reuse-size", str(args.reuse_size), "--trace-file", f"{trace_root_dir}/pipeline_parallel/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt"])

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
    FC_devices_list = factorize(args.num_devices)
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
                for mode in ["model_parallel", "model_parallel_FC"]:
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
        mode_list = ["model_parallel", "model_parallel_FC"]
    elif args.decode_only:
        mode_list = ["pipeline_parallel"]
    else:
        mode_list = model_parallel_mode_list if args.model_parallel else pipeline_parallel_mode_list
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

def calculate_acc_latency(args, seqlen):
    latency = {}
    GQA_factor = 1.00 + 1.00 / gqa_factor[args.model]
    latency["RMSNorm_latency"] =  embedding_size[args.model] / 16.00 / 16.00 / args.num_channels * ACCEL_CYCLE["VEC"]    # EMB /16.00 /16.00 ADD
    latency["RMSNorm_latency"] += SB_RD_CYCLE + SB_WR_CYCLE + 1.00                              # 1 RED
    latency["RMSNorm_latency"] += RV_RMSNorm_CYCLE                                              # 1 RISCV
    latency["RMSNorm_latency"] = float(2.00 * latency["RMSNorm_latency"]) / float(FREQ / KILO)
    latency["Softmax_latency"] =  seqlen * n_heads[args.model] / 16.00 / args.num_channels * ACCEL_CYCLE["EXP"]        # TOK*HEAD /16.00 EXP
    latency["Softmax_latency"] += seqlen * n_heads[args.model] / 16.00 / args.num_channels * ACCEL_CYCLE["VEC"]        # TOK*HEAD /16.00 ADD
    latency["Softmax_latency"] += n_heads[args.model] * 1.00 * SB_RD_CYCLE                                     # HEAD RED
    latency["Softmax_latency"] += n_heads[args.model] * RV_SFT_CYCLE_PIPELINE                                  # HEAD RISCV
    latency["Softmax_latency"] = float(latency["Softmax_latency"]) / float(FREQ / KILO)
    latency["RotEmbed_latency"] = embedding_size[args.model] * RV_ROTEmbed_CYCLE                                 # EMB RISCV
    latency["RotEmbed_latency"] = float(GQA_factor * latency["RotEmbed_latency"]) / float(FREQ / KILO)
    return latency

def load_data_point(args, seqlen, FC_devices, channels_per_block, PCIe_lanes_per_device, blocks_per_device, embedding_latency, utilized_devices, pp, tp):

    log_root_dir = log_root(args)
    if args.model_parallel:
        path = f"{log_root_dir}/model_parallel/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt.log"
    else:
        path = f"{log_root_dir}/pipeline_parallel/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt.log"
    if missing_or_empty(path):
        raise FileNotFoundError(f"Simulation log is missing or empty: {path}. Re-run --generate_trace --simulate_trace before --update_csv.")
    stats = command_processor(path)
    pim_latency = stats["latency"]

    if args.model_parallel:
        if "Llama" in args.model:
            cxl_latency = llama_latency([embedding_size[args.model], ffn_size[args.model]], PCIe_lanes_per_device, FC_devices, args.num_devices)
        else:
            cxl_latency = gpt_latency([embedding_size[args.model], ffn_size[args.model]], PCIe_lanes_per_device, FC_devices, args.num_devices)
        embedding_latency_data = 0.00 if args.decode_only else embedding_latency['model_parallel'][FC_devices]
    else:
        cxl_latency = vector_latency(embedding_size[args.model], PCIe_lanes_per_device)
        embedding_latency_data = 0.00 if args.decode_only else embedding_latency['pipeline_parallel'][channels_per_block]
    acc_latency_dict = calculate_acc_latency(args, seqlen)
    acc_latency = (acc_latency_dict["RMSNorm_latency"] + acc_latency_dict["Softmax_latency"] + acc_latency_dict["RotEmbed_latency"]) * blocks_per_device
    transformer_block_latency = pim_latency + cxl_latency + acc_latency
    token_latency = transformer_block_latency * TransformerBlock_number[args.model] + embedding_latency_data
    if not args.decode_only:
        token_latency += InOut_latency
    throughput = 1000 / token_latency * pp

    energy_token = {}
    PCIE = embedding_size[args.model] * 10 + ffn_size[args.model] * 2 if args.model_parallel else embedding_size[args.model]
    energy_main, latency_main = power_calculator(
        stats, PCIE, n_heads[args.model], embedding_size[args.model], seqlen, gqa_factor[args.model],
        dram_power_impl=args.dram_power_impl,
        dram_energy_model=args.dram_energy_model,
        command_trace_prefix=command_trace_prefix_for_log(path),
    )
    if args.model_parallel:
        pipeline_stages = args.num_devices // FC_devices
        FC_path = f"{log_root_dir}/model_parallel_FC/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt.log"
        if missing_or_empty(FC_path):
            raise FileNotFoundError(f"Simulation log is missing or empty: {FC_path}. Re-run --generate_trace --simulate_trace before --update_csv.")
        stats_FC = command_processor(FC_path)
        energy_FC, latency_FC = power_calculator(
            stats_FC, PCIE, n_heads[args.model], embedding_size[args.model], seqlen, gqa_factor[args.model],
            dram_power_impl=args.dram_power_impl,
            dram_energy_model=args.dram_energy_model,
            command_trace_prefix=command_trace_prefix_for_log(FC_path),
        )
        for comp in energy_main.keys():
            energy_token[comp] = (energy_main[comp] + energy_FC[comp] * (FC_devices - 1)) * TransformerBlock_number[args.model]
    else:
        for comp in energy_main.keys():
            energy_token[comp] = energy_main[comp] * utilized_devices
    total_energy = 0
    for comp in energy_token.keys():
        total_energy += energy_token[comp]
    # Energy per output token times sustained output rate gives the steady-
    # state power of all participating devices, including pipeline overlap.
    total_power = total_energy * throughput / 1000.0
    device_utilization = 1.0 * utilized_devices / args.num_devices
                        
    new_result = {
        'Model': args.model,
        'Device number': args.num_devices,
        'Pipeline parallelism': pp,
        'Tensor parallelism': tp,
        'Channels per device': args.num_channels,
        'Banks per device': args.num_banks,
        'Channels per block': channels_per_block,
        'Sequence length': seqlen,
        'Context window': args.max_seq_len if args.max_seq_len is not None else seqlen,
        'PIM latency': pim_latency,
        'CXL latency': cxl_latency,
        'Acc latency': acc_latency,
        'TransformerBlock latency': transformer_block_latency,
        'Embedding latency': embedding_latency_data,
        'Token latency (ms)': token_latency,
        'Throughput (tokens/s)': throughput,
        'Token energy (mJ)': total_energy,
        'Total power (W)': total_power,
        'Device utilization': device_utilization,
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
        if 'DRAM energy model' not in results_df.columns:
            results_df['DRAM energy model'] = 'legacy'
        if 'Context window' not in results_df.columns:
            results_df['Context window'] = results_df['Sequence length']
    else:
        columns = ['Model', 'Device number', 'Pipeline parallelism', 'Tensor parallelism', 'Channels per device', 'Banks per device', 'Channels per block', 'Sequence length', 'Context window', 'PIM latency', 'CXL latency', 'Acc latency', 'TransformerBlock latency', 'Embedding latency', 'Token latency (ms)', 'Throughput (tokens/s)', 'Token energy (mJ)', 'Total power (W)', 'Device utilization', 'DRAM energy model']
        results_df = pd.DataFrame(columns=columns)

    embedding_latency = {'pipeline_parallel': {}, 'model_parallel': {}}

    if args.model_parallel:
        FC_devices_list = factorize(args.num_devices)

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
    results_df = results_df.drop_duplicates(subset=['Model', 'Device number', 'Pipeline parallelism', 'Tensor parallelism', 'Channels per device', 'Banks per device', 'Channels per block', 'Sequence length', 'Context window', 'DRAM energy model'], keep='last')
    results_df = results_df.sort_values(by=['Model', 'Device number', 'Pipeline parallelism', 'Tensor parallelism', 'Channels per device', 'Banks per device', 'Channels per block', 'Context window', 'Sequence length', 'DRAM energy model'])
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
    
    if os.path.exists(args.processed_result_path):
        results_df = pd.read_csv(args.processed_result_path)
    else:
        columns = ['Model', 'Device number', 'Banks per device', 'Seqlen', 'Pipeline parallelism', 'Tensor parallelism', 'Phase', 'DRAM energy model', 'Total Latency (s)', 'Throughput (tokens/s)', 'Energy per Token (mJ)', 'Total power (W)']
        results_df = pd.DataFrame(columns=columns)


    if args.model_parallel:

        FC_devices_list = factorize(args.num_devices)

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
    results_df = results_df.sort_values(by=['Model', 'Device number', 'Banks per device', 'Seqlen', 'Pipeline parallelism', 'Tensor parallelism', 'Phase'])
    os.makedirs(os.path.dirname(args.processed_result_path) or ".", exist_ok=True)
    results_df.to_csv(args.processed_result_path, index=False)


if __name__ == "__main__":


    args = get_args()

    if args.seqlen:
        seqlen_list = args.seqlen
    else:
        seqlen_list = [i * args.seqlen_gap for i in range(1, (args.prefill + args.decoding) // args.seqlen_gap + 1)]
        
    for mode in pipeline_parallel_mode_list + model_parallel_mode_list:
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
