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
DEVICE_ROLES = ("main", "inter_device_helper", "fc_helper", "kv_head_helper")
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
SRAM_IO_PARALLEL = float(CELLAR_POWER_CALCULATOR.SRAM_IO_PARALLEL)
SHARED_BUFFER_CAPACITY_BYTES = int(
    CELLAR_POWER_CALCULATOR.SHARED_BUFFER_CAPACITY_BYTES
)
ACCEL_CYCLE = dict(CELLAR_POWER_CALCULATOR.ACCEL_CYCLE)


def set_channel_count(channels_per_device, sram_io_parallel=None):
    """Keep CENT's channel count and banked-SRAM model aligned with Cellar."""
    global CH_PER_DV, SRAM_IO_PARALLEL
    CH_PER_DV = float(channels_per_device)
    if sram_io_parallel is not None:
        if sram_io_parallel <= 0:
            raise ValueError("sram_io_parallel must be positive")
        SRAM_IO_PARALLEL = float(sram_io_parallel)
    ACCEL_CYCLE.clear()
    ACCEL_CYCLE.update({
        "EXP": (
            CH_PER_DV / SRAM_IO_PARALLEL * SB_RD_CYCLE
            + EXP_LANE_CYCLE
            + SB_WR_CYCLE
        ),
        "VEC": (
            CH_PER_DV / SRAM_IO_PARALLEL * 2.00 * SB_RD_CYCLE
            + 1.00
            + SB_WR_CYCLE
        ),
    })
    ACCEL_CYCLE["VEC_ADD"] = ACCEL_CYCLE["VEC"]
    ACCEL_CYCLE["VEC_MUL"] = ACCEL_CYCLE["VEC"]
    CELLAR_POWER_CALCULATOR.CH_PER_DV = CH_PER_DV
    CELLAR_POWER_CALCULATOR.SRAM_IO_PARALLEL = SRAM_IO_PARALLEL
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


def _analytical_dynamic_energy_by_operation(
    stat,
    Head,
    HiddenDim,
    Tokens,
    GQA,
    *,
    rmsnorm_hidden_dim=None,
):
    """Reproduce Cellar's non-trace accelerator terms, split by operation.

    Cellar historically adds RMSNorm, softmax, and rotary-embedding energy to
    every trace.  CENT's helper traces intentionally execute only a subset of
    those operations, so the adapter needs the same formulas split by phase in
    order to remove only the operations that are absent from a helper role.
    Returned energies are in mJ, matching ``calculate_energy_and_latency``.
    """

    tck_ps = stat["tCK_ps"]
    gqa_factor = 1.0 + 1.0 / GQA
    rms_hidden = HiddenDim if rmsnorm_hidden_dim is None else rmsnorm_hidden_dim
    scale = tck_ps / 1e12
    sb = SRAM_POWER["SB"]
    ib_read = SRAM_POWER["IB"]["RD"]

    rmsnorm = {
        "SB_DYN": 2.0
        * (rms_hidden / 16.0 / 16.0 + 2.0)
        * (2.0 * sb["RD"] + sb["WR"])
        * scale,
        "IB_DYN": 2.0 * (rms_hidden / 16.0 / 16.0 + 2.0) * ib_read * scale,
        "RV_DYN": 2.0 * RV_RMSNorm_CYCLE * ACCEL_POWER["RV"] * scale,
        "RED_DYN": 2.0 * ACCEL_POWER["RED"]["DYN"] * scale,
        "EXP_DYN": 0.0,
        "VEC_DYN": 2.0 * rms_hidden / 16.0 / 16.0 * ACCEL_POWER["VEC"]["DYN"] * scale,
        "VEC_MUL_DYN": 0.0,
    }
    softmax = {
        "SB_DYN": (
            (Tokens * Head / 16.0 * 3.0 + Head * 2.0) * sb["RD"]
            + (Tokens * Head / 16.0 * 2.0 + Head * 2.0) * sb["WR"]
        )
        * scale,
        "IB_DYN": (Tokens * Head / 16.0 * 2.0 + Head * 2.0) * ib_read * scale,
        "RV_DYN": Head * RV_SFT_CYCLE_SINGLE * ACCEL_POWER["RV"] * scale,
        "RED_DYN": Head * ACCEL_POWER["RED"]["DYN"] * scale,
        "EXP_DYN": Tokens * Head / 16.0 * ACCEL_POWER["EXP"]["DYN"] * scale,
        "VEC_DYN": Tokens * Head / 16.0 * ACCEL_POWER["VEC"]["DYN"] * scale,
        "VEC_MUL_DYN": (
            Tokens
            * Head
            / 16.0
            * ACCEL_POWER["VEC_MUL"]["DYN"]
            * 2.0
            * scale
        ),
    }
    rotary = {
        "SB_DYN": gqa_factor
        * HiddenDim
        / 16.0
        * (sb["RD"] + 2.0 * sb["WR"])
        * scale,
        "IB_DYN": gqa_factor * HiddenDim * ib_read * scale,
        "RV_DYN": gqa_factor
        * HiddenDim
        * RV_ROTEmbed_CYCLE
        * ACCEL_POWER["RV"]
        * scale,
        "RED_DYN": 0.0,
        "EXP_DYN": 0.0,
        "VEC_DYN": 0.0,
        "VEC_MUL_DYN": 0.0,
    }
    return {
        "RMSNorm": rmsnorm,
        "Softmax": softmax,
        "RotEmbed": rotary,
    }


def kv_head_tp_pnm_dynamic_energy(
    stat, *, reduction_adds, ewmul_elements=0
):
    """Analytical energy for device-local TP reduction and PNM EWMUL.

    These device-local PNM operations intentionally do not appear in the
    Ramulator trace.  Reduction combines the channel-group partials with the
    16-lane vector-add unit. ``ewmul_elements`` counts scalar element-wise
    multiplies moved from PIM to PNM across RMSNorm, RoPE, and
    ``SiLU(W1) * W3``. Every 16 elements issue one VEC_MUL operation with two
    shared-buffer reads and one write. K/V cache writes are represented by
    their explicit ``W MEM`` trace commands.
    Returned terms use Cellar's mJ convention and can be added directly to one
    rank's ``power_calculator`` result.
    """

    if reduction_adds < 0 or ewmul_elements < 0:
        raise ValueError("PNM work counts cannot be negative")
    scale = stat["tCK_ps"] / 1e12
    reduction_groups = math.ceil(reduction_adds / 16.0)
    ewmul_groups = math.ceil(ewmul_elements / 16.0)
    sb = SRAM_POWER["SB"]
    return {
        "SB_DYN": (
            reduction_groups * (2.0 * sb["RD"] + sb["WR"])
            + ewmul_groups * (2.0 * sb["RD"] + sb["WR"])
        ) * scale,
        "IB_DYN": (reduction_groups + ewmul_groups)
        * SRAM_POWER["IB"]["RD"]
        * scale,
        "EXP_DYN": 0.0,
        "VEC_DYN": (
            reduction_groups * ACCEL_POWER["VEC_ADD"]["DYN"]
            * scale
        ),
        "VEC_MUL_DYN": (
            ewmul_groups * ACCEL_POWER["VEC_MUL"]["DYN"] * scale
        ),
    }


def add_energy_terms(energy, additions):
    """Return a copy of an energy breakdown with additive component terms."""

    result = dict(energy)
    for component, value in additions.items():
        if component not in result:
            raise KeyError(f"Unknown Cellar energy component: {component}")
        result[component] += value
    return result


def _filter_energy_for_device_role(energy, latency, stat, Head, HiddenDim, Tokens, GQA, device_role):
    if device_role == "main":
        return energy, latency

    excluded_operations = {
        # Inter-device helpers rotate their local Q/K shard, but normalization
        # and the current centralized softmax remain on the main device.
        "inter_device_helper": ("RMSNorm", "Softmax"),
        # The paper-compatible FC-only helper performs none of these phases.
        "fc_helper": ("RMSNorm", "Softmax", "RotEmbed"),
        # Standard head-TP helpers execute local RoPE, Softmax, and activation;
        # the main alone performs the two global residual/RMSNorm phases.
        "kv_head_helper": ("RMSNorm",),
    }[device_role]
    filtered_energy = dict(energy)
    filtered_latency = dict(latency)
    contributions = _analytical_dynamic_energy_by_operation(stat, Head, HiddenDim, Tokens, GQA)
    latency_keys = {
        "RMSNorm": "RMSNorm_latency",
        "Softmax": "Softmax_latency",
        "RotEmbed": "RotEmbed_latency",
    }
    for operation in excluded_operations:
        for component, value in contributions[operation].items():
            filtered_energy[component] -= value
            if filtered_energy[component] < 0.0:
                tolerance = max(1e-15, abs(value) * 1e-9)
                if filtered_energy[component] < -tolerance:
                    raise ValueError(
                        f"Filtering {operation} for {device_role} made {component} negative: "
                        f"{filtered_energy[component]} mJ"
                    )
                filtered_energy[component] = 0.0
        filtered_latency[latency_keys[operation]] = 0.0
    return filtered_energy, filtered_latency


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
    device_role="main",
    rmsnorm_hidden_dim=None,
):
    """Call Cellar's canonical legacy or TraceRecorder-based energy model."""
    if dram_energy_model not in DRAM_ENERGY_MODELS:
        raise ValueError(f"Unknown DRAM energy model '{dram_energy_model}'")
    if device_role not in DEVICE_ROLES:
        raise ValueError(f"Unknown device role '{device_role}'. Expected one of: {', '.join(DEVICE_ROLES)}")
    set_channel_count(CH_PER_DV)
    if command_trace_prefix is None:
        command_trace_prefix = stat.get("command_trace_prefix")
    energy, latency = CELLAR_POWER_CALCULATOR.calculate_energy_and_latency(
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
    if rmsnorm_hidden_dim is not None and rmsnorm_hidden_dim != HiddenDim:
        old_rms = _analytical_dynamic_energy_by_operation(
            stat, Head, HiddenDim, Tokens, GQA
        )["RMSNorm"]
        new_rms = _analytical_dynamic_energy_by_operation(
            stat,
            Head,
            HiddenDim,
            Tokens,
            GQA,
            rmsnorm_hidden_dim=rmsnorm_hidden_dim,
        )["RMSNorm"]
        for component in old_rms:
            energy[component] += new_rms[component] - old_rms[component]
        tck_ps = stat["tCK_ps"]
        rms_cycles = rmsnorm_hidden_dim / 16.0 / 16.0 / CH_PER_DV * ACCEL_CYCLE["VEC"]
        rms_cycles += SB_RD_CYCLE + SB_WR_CYCLE + 1.0 + RV_RMSNorm_CYCLE
        latency["RMSNorm_latency"] = 2.0 * rms_cycles * tck_ps / GIGA
    return _filter_energy_for_device_role(
        energy,
        latency,
        stat,
        Head,
        HiddenDim,
        Tokens,
        GQA,
        device_role,
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
