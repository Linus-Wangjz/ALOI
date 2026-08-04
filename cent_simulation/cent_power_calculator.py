"""CENT adapter for the canonical AiM Cellar power calculator.

All energy accounting is implemented in ``aim_simulator/scripts``.  This
module only supplies CENT's log naming, command-line interface, and the small
amount of model-level aggregation that CENT needs.
"""

import argparse
import importlib
import math
import sys
from pathlib import Path


KILO = 1000
MEGA = 1000000
GIGA = 1000000000
FREQ = 2.00 * GIGA
PIM_FREQ = 1.00 * GIGA
WORD_SIZE = 256
tRC = 44.5
tBL = 1.25
tCCDL = GIGA / PIM_FREQ

_AIM_SIMULATOR_SCRIPTS = Path(__file__).resolve().parents[1] / "aim_simulator" / "scripts"
if not _AIM_SIMULATOR_SCRIPTS.is_dir():
    raise ImportError(f"Cellar scripts were not found at {_AIM_SIMULATOR_SCRIPTS}")
sys.path.insert(0, str(_AIM_SIMULATOR_SCRIPTS))
CELLAR_POWER_CALCULATOR = importlib.import_module("cellar_power_calculator")

DRAM_ENERGY_MODELS = CELLAR_POWER_CALCULATOR.DRAM_ENERGY_MODELS
DRAM_POWER = CELLAR_POWER_CALCULATOR.dram_power_for_impl("GDDR6")
ACCEL_POWER = CELLAR_POWER_CALCULATOR.ACCEL_POWER
SRAM_POWER = CELLAR_POWER_CALCULATOR.SRAM_POWER
CTRL_POWER = CELLAR_POWER_CALCULATOR.CTRL_POWER
commands = list(CELLAR_POWER_CALCULATOR.commands)
isrs = list(CELLAR_POWER_CALCULATOR.isrs)

CH_PER_DV = float(CELLAR_POWER_CALCULATOR.CH_PER_DV)
RV_COUNT = CELLAR_POWER_CALCULATOR.RV_COUNT
SB_RD_CYCLE = CELLAR_POWER_CALCULATOR.SB_RD_CYCLE
SB_WR_CYCLE = CELLAR_POWER_CALCULATOR.SB_WR_CYCLE
EXP_LANE_CYCLE = CELLAR_POWER_CALCULATOR.EXP_LANE_CYCLE
RV_RMSNorm_CYCLE = CELLAR_POWER_CALCULATOR.RV_RMSNorm_CYCLE
RV_ROTEmbed_CYCLE = CELLAR_POWER_CALCULATOR.RV_ROTEmbed_CYCLE
RV_SFT_CYCLE_PIPELINE = CELLAR_POWER_CALCULATOR.RV_SFT_CYCLE_PIPELINE
RV_SFT_CYCLE_SINGLE = CELLAR_POWER_CALCULATOR.RV_SFT_CYCLE_SINGLE
ACCEL_CYCLE = dict(CELLAR_POWER_CALCULATOR.ACCEL_CYCLE)


def set_channel_count(channels_per_device):
    """Keep CENT's exported constants and Cellar's mutable globals aligned."""
    global CH_PER_DV
    CH_PER_DV = float(channels_per_device)
    ACCEL_CYCLE.clear()
    ACCEL_CYCLE.update({
        "EXP": CH_PER_DV * SB_RD_CYCLE + EXP_LANE_CYCLE + SB_WR_CYCLE,
        "VEC": CH_PER_DV * 2.00 * SB_RD_CYCLE + 1.00 + SB_WR_CYCLE,
    })
    CELLAR_POWER_CALCULATOR.CH_PER_DV = CH_PER_DV
    CELLAR_POWER_CALCULATOR.ACCEL_CYCLE = dict(ACCEL_CYCLE)


def default_timing_path_for_log(stat_path):
    path = Path(stat_path)
    base = str(path)[:-4] if str(path).endswith(".log") else str(path)
    return Path(f"{base}.timing.yaml")


def command_trace_prefix_for_log(stat_path):
    path = Path(stat_path)
    base = str(path)[:-4] if str(path).endswith(".log") else str(path)
    return Path(f"{base}.cmd")


def command_processor(stat_path, timing_path=None):
    """Read one Ramulator run through Cellar's timing-resolved parser."""
    result_path = Path(stat_path)
    if not result_path.exists() or result_path.stat().st_size == 0:
        raise FileNotFoundError(f"Ramulator log is missing or empty: {result_path}")
    resolved_timing_path = Path(timing_path) if timing_path else default_timing_path_for_log(result_path)
    if not resolved_timing_path.exists() or resolved_timing_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Resolved timing export is missing: {resolved_timing_path}. "
            "Re-run Ramulator through run_sim.py so DRAMTimingExporter is enabled."
        )
    stat = CELLAR_POWER_CALCULATOR.read_run_statistics(result_path, resolved_timing_path)
    stat["timing_path"] = str(resolved_timing_path)
    stat["command_trace_prefix"] = str(command_trace_prefix_for_log(result_path))
    return stat


def power_calculator(
    stat,
    PCIE_bits,
    Head,
    HiddenDim,
    Tokens,
    GQA,
    dram_power_impl=None,
    dram_energy_model="legacy",
    command_trace_prefix=None,
):
    """Call Cellar's canonical legacy or TraceRecorder-based energy model."""
    if dram_energy_model not in DRAM_ENERGY_MODELS:
        raise ValueError(f"Unknown DRAM energy model '{dram_energy_model}'")
    set_channel_count(CH_PER_DV)
    if command_trace_prefix is None:
        command_trace_prefix = stat.get("command_trace_prefix")
    return CELLAR_POWER_CALCULATOR.calculate_energy_and_latency(
        stat,
        PCIE_bits,
        Head,
        HiddenDim,
        Tokens,
        GQA,
        dram_power_impl=dram_power_impl,
        dram_energy_model=dram_energy_model,
        command_trace_prefix=command_trace_prefix,
    )


def get_args():
    parser = argparse.ArgumentParser(description="CENT Power Calculator (Cellar adapter)")
    parser.add_argument("--mlog", type=str, required=True, help="Main Ramulator log")
    parser.add_argument("--mtiming", type=str, help="Matching DRAMTimingExporter YAML")
    parser.add_argument("--mcmd-trace", type=str, help="Matching TraceRecorder prefix")
    parser.add_argument("--plog", type=str, help="Additional PIM-stage Ramulator log")
    parser.add_argument("--ptiming", type=str, help="Additional PIM-stage timing YAML")
    parser.add_argument("--pcmd-trace", type=str, help="Additional PIM-stage command-trace prefix")
    parser.add_argument("--dram-energy-model", choices=DRAM_ENERGY_MODELS, default="legacy")
    parser.add_argument("--dram-power-impl", choices=["GDDR6", "LPDDR4", "LPDDR4X"])
    parser.add_argument("--head", type=int, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--fc", type=int, required=True)
    parser.add_argument("--token", type=int, required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--ch_per_bl", type=int, required=True)
    parser.add_argument("--dv", type=int, default=32)
    parser.add_argument("--ch_per_dv", type=int, default=32)
    parser.add_argument("--gqa", type=int, default=1)
    return parser.parse_args()


def main():
    args = get_args()
    set_channel_count(args.ch_per_dv)
    if args.dram_energy_model == "trace-based" and not args.mcmd_trace:
        raise ValueError("--mcmd-trace is required for --dram-energy-model trace-based")

    stat_main = command_processor(args.mlog, args.mtiming)
    pcie_bits = args.hidden if args.ch_per_bl <= CH_PER_DV else args.hidden * 10 + args.fc * 2.00
    energy_main, latency_main = power_calculator(
        stat_main,
        pcie_bits,
        args.head,
        args.hidden,
        args.token,
        args.gqa,
        dram_power_impl=args.dram_power_impl,
        dram_energy_model=args.dram_energy_model,
        command_trace_prefix=args.mcmd_trace,
    )

    if args.ch_per_bl > CH_PER_DV:
        if not args.plog:
            raise ValueError("--plog is required when --ch_per_bl > --ch_per_dv")
        if args.dram_energy_model == "trace-based" and not args.pcmd_trace:
            raise ValueError("--pcmd-trace is required for trace-based PIM energy")
        devices_per_block = math.ceil(args.ch_per_bl / CH_PER_DV)
        if args.dv % devices_per_block:
            raise ValueError("--dv must be divisible by the devices required per block")
        stat_pim = command_processor(args.plog, args.ptiming)
        energy_pim, _ = power_calculator(
            stat_pim,
            pcie_bits,
            args.head,
            args.hidden,
            args.token,
            args.gqa,
            dram_power_impl=args.dram_power_impl,
            dram_energy_model=args.dram_energy_model,
            command_trace_prefix=args.pcmd_trace,
        )
        device_count = args.block * devices_per_block
        pipeline_stages = args.dv / devices_per_block
        energy_token = {
            component: (energy_main[component] + energy_pim[component] * (devices_per_block - 1)) * args.block
            for component in energy_main
        }
        power_all_devices = {
            component: (energy_main[component] + energy_pim[component] * (devices_per_block - 1)) * pipeline_stages / stat_main["latency"]
            for component in energy_main
        }
    else:
        blocks_per_device = int(CH_PER_DV / args.ch_per_bl)
        device_count = math.ceil(args.block / blocks_per_device)
        energy_token = {component: energy_main[component] * device_count for component in energy_main}
        power_all_devices = {component: energy_token[component] / stat_main["latency"] for component in energy_main}
        for component in latency_main:
            latency_main[component] *= blocks_per_device

    total_acc_latency = sum(latency_main.values())
    total_latency = stat_main["latency"] + total_acc_latency
    print(f"{stat_main['latency']},{latency_main['RMSNorm_latency']},{latency_main['Softmax_latency']},{latency_main['RotEmbed_latency']},{total_acc_latency},{total_latency},{stat_main['utilization']}")
    print(",\nenergy 1 token detailed (mJ):")
    print(",".join(energy_token))
    print(",".join(str(energy_token[component]) for component in energy_token))
    print(sum(energy_token.values()))
    print(sum(power_all_devices.values()))
    print(f"devices used: {device_count}")


if __name__ == "__main__":
    main()
