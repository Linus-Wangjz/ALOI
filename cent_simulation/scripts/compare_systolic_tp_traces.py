"""Compare the imported systolic trace with the standard-TP physical mapping."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace


CENT_SIMULATION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CENT_SIMULATION))

from cent_power_calculator import (
    add_energy_terms,
    command_processor,
    kv_head_tp_pnm_dynamic_energy,
    power_calculator,
)
import run_sim
from tp_mapping import SystolicTPLayout, kv_head_tp_shape


REPORT_COMMANDS = (
    "MEM",
    "R_MEM",
    "W_MEM",
    "WR_BIAS",
    "WR_GB",
    "MAC_ABK",
    "RD_MAC",
    "AF",
    "RD_AF",
    "SYNC",
    "EWADD",
)


def trace_stats(path: Path) -> dict:
    commands = Counter()
    mac_cycles = 0
    with path.open() as trace:
        for line in trace:
            fields = line.split()
            if len(fields) >= 2 and fields[0] in ("R", "W") and fields[1] == "MEM":
                commands["MEM"] += 1
                commands[f"{fields[0]}_MEM"] += 1
                continue
            if len(fields) < 2 or fields[0] != "AiM":
                continue
            commands[fields[1]] += 1
            if fields[1] == "MAC_ABK":
                mac_cycles += int(fields[2])
    sidecar_path = Path(str(path) + ".systolic.json")
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else None
    return {
        "path": str(path),
        "commands": dict(commands),
        "total_commands": sum(commands.values()) - commands["MEM"],
        "mac_cycles": mac_cycles,
        "sidecar": sidecar,
    }


def percent_delta(original: int | float, mapped: int | float) -> float | None:
    if original == 0:
        return None
    return 100.0 * (mapped - original) / original


def comparison(original: dict, mapped: dict, blocks: int) -> dict:
    metrics = {
        "total_commands": (
            original["total_commands"],
            mapped["total_commands"],
        ),
        "mac_cycles": (original["mac_cycles"], mapped["mac_cycles"]),
    }
    for command in REPORT_COMMANDS:
        metrics[command] = (
            original["commands"].get(command, 0),
            mapped["commands"].get(command, 0),
        )
    rows = {}
    for name, (old_value, new_value) in metrics.items():
        rows[name] = {
            "original_per_block": old_value,
            "standard_tp_per_block": new_value,
            "delta_percent": percent_delta(old_value, new_value),
            "original_pp_total": old_value * blocks,
            "standard_tp_pp_total": new_value * blocks,
        }
    mapped_sidecar = mapped.get("sidecar") or {}
    return {
        "pipeline_blocks": blocks,
        "metrics": rows,
        "standard_tp": {
            "array": mapped_sidecar.get("array"),
            "qk": mapped_sidecar.get("qk"),
            "sv": mapped_sidecar.get("sv"),
            "pipeline_cycles": mapped_sidecar.get("pipeline_cycles"),
            "kernel_pipeline_cycles": mapped_sidecar.get(
                "kernel_pipeline_cycles"
            ),
            "pnm_reduction_adds": mapped_sidecar.get("pnm_reduction_adds"),
            "physical_rows_per_bank": mapped_sidecar.get(
                "physical_rows_per_bank"
            ),
            "fused_row_aliases": mapped_sidecar.get("fused_row_aliases"),
        },
    }


def simulated_comparison(args: argparse.Namespace) -> dict:
    original_stat = command_processor(args.original_log)
    mapped_stat = command_processor(args.standard_tp_log)
    pp_handoff_bits = args.dim * 16 if args.blocks > 1 else 0
    common_power_args = (
        pp_handoff_bits,
        args.query_heads,
        args.dim,
        args.context,
        args.gqa,
    )
    original_energy, _ = power_calculator(original_stat, *common_power_args)
    mapped_energy, _ = power_calculator(mapped_stat, *common_power_args)

    shape = kv_head_tp_shape(
        dim=args.dim,
        query_heads=args.query_heads,
        kv_heads=args.kv_heads,
        ffn_dim=args.ffn_dim,
        tp=1,
    )
    layout = SystolicTPLayout(
        shape=shape,
        num_channels=args.channels,
        banks_per_channel=args.banks,
        max_seq_len=args.context,
        systolic_height=args.sa_height,
    )
    mapped_energy = add_energy_terms(
        mapped_energy,
        kv_head_tp_pnm_dynamic_energy(
            mapped_stat,
            repack_elements=2 * shape.local_kv_dim,
            reduction_adds=sum(layout.pnm_reduction_adds(args.context).values()),
            activation_elements=shape.local_ffn_dim,
        ),
    )
    scale_args = SimpleNamespace(
        batch_size=1, systolic_pim=True, systolic_dim=args.sa_height
    )
    original_energy = run_sim.adjust_systolic_energy(original_energy, scale_args)
    mapped_energy = run_sim.adjust_systolic_energy(mapped_energy, scale_args)

    common_latency_args = {
        "model": "Llama2-70B",
        "num_channels": args.channels,
        "num_banks": args.banks,
        "max_seq_len": args.context,
        "systolic_pim": True,
        "systolic_dim": args.sa_height,
    }
    original_acc = run_sim.calculate_acc_latency(
        SimpleNamespace(kv_head_tp=False, **common_latency_args), args.context
    )
    mapped_acc = run_sim.calculate_acc_latency(
        SimpleNamespace(kv_head_tp=True, **common_latency_args), args.context, 1
    )
    pp_handoff_latency = (
        run_sim.vector_latency(args.dim, args.pcie_lanes_per_device)
        if args.blocks > 1
        else 0.0
    )
    original_block_latency = (
        original_stat["latency"] + sum(original_acc.values()) + pp_handoff_latency
    )
    mapped_block_latency = (
        mapped_stat["latency"] + sum(mapped_acc.values()) + pp_handoff_latency
    )
    original_block_energy = sum(original_energy.values())
    mapped_block_energy = sum(mapped_energy.values())

    return {
        "latency_ms": {
            "original_pim_per_block": original_stat["latency"],
            "standard_tp_pim_per_block": mapped_stat["latency"],
            "original_acc_per_block": sum(original_acc.values()),
            "standard_tp_acc_per_block": sum(mapped_acc.values()),
            "common_pp_handoff_per_block": pp_handoff_latency,
            "original_total_per_block": original_block_latency,
            "standard_tp_total_per_block": mapped_block_latency,
            "delta_percent": percent_delta(
                original_block_latency, mapped_block_latency
            ),
            "original_serial_pp_total": original_block_latency * args.blocks,
            "standard_tp_serial_pp_total": mapped_block_latency * args.blocks,
        },
        "energy_mj": {
            "original_per_block": original_block_energy,
            "standard_tp_per_block": mapped_block_energy,
            "delta_percent": percent_delta(
                original_block_energy, mapped_block_energy
            ),
            "original_per_token_pp_total": original_block_energy * args.blocks,
            "standard_tp_per_token_pp_total": mapped_block_energy * args.blocks,
        },
        "steady_pipeline_power_w": {
            "original": args.blocks
            * original_block_energy
            / original_block_latency,
            "standard_tp": args.blocks
            * mapped_block_energy
            / mapped_block_latency,
        },
        "system_throughput_tokens_s": {
            "original": 1000.0 / original_block_latency,
            "standard_tp": 1000.0 / mapped_block_latency,
        },
        "standard_tp_acc_breakdown_ms": mapped_acc,
        "energy_components_mj_per_block": {
            "original": original_energy,
            "standard_tp": mapped_energy,
        },
        "scope": (
            "TP=1 has no Wo/W2 all-reduce.  The identical one-hidden-vector "
            "PP handoff is included on both sides."
        ),
    }
def print_markdown(report: dict) -> None:
    print(
        "metric | original/block | standard TP/block | delta | "
        f"original x{report['pipeline_blocks']} | standard TP x{report['pipeline_blocks']}"
    )
    print("--- | ---: | ---: | ---: | ---: | ---:")
    for name, values in report["metrics"].items():
        delta = values["delta_percent"]
        delta_text = "n/a" if delta is None else f"{delta:+.2f}%"
        print(
            f"{name} | {values['original_per_block']} | "
            f"{values['standard_tp_per_block']} | {delta_text} | "
            f"{values['original_pp_total']} | "
            f"{values['standard_tp_pp_total']}"
        )
    standard_tp = report["standard_tp"]
    if standard_tp["sv"]:
        print()
        print(
            "SV packing: "
            f"{standard_tp['sv']['mode']}, "
            f"height utilization={standard_tp['sv']['height_utilization']:.2%}, "
            f"width utilization={standard_tp['sv']['width_utilization']:.2%}"
        )
    if standard_tp["qk"]:
        print(
            "QK utilization: "
            f"height={standard_tp['qk']['height_utilization']:.2%}, "
            f"width={standard_tp['qk']['width_utilization']:.2%}"
        )
    simulated = report.get("simulation")
    if simulated:
        latency = simulated["latency_ms"]
        energy = simulated["energy_mj"]
        power = simulated["steady_pipeline_power_w"]
        print()
        print(
            "Ramulator + Cellar local latency/block: "
            f"{latency['original_total_per_block']:.6f} ms -> "
            f"{latency['standard_tp_total_per_block']:.6f} ms "
            f"({latency['delta_percent']:+.2f}%)"
        )
        print(
            "Energy/block: "
            f"{energy['original_per_block']:.6f} mJ -> "
            f"{energy['standard_tp_per_block']:.6f} mJ "
            f"({energy['delta_percent']:+.2f}%)"
        )
        print(
            f"Steady x{report['pipeline_blocks']} pipeline power: "
            f"{power['original']:.3f} W -> {power['standard_tp']:.3f} W"
        )


def comparison_plot_rows(report: dict) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = [
        {
            "metric": "Trace commands",
            "filename": "trace_commands",
            "unit": "commands/block",
            "direct_port": report["metrics"]["total_commands"]["original_per_block"],
            "mapped": report["metrics"]["total_commands"]["standard_tp_per_block"],
        },
        {
            "metric": "MAC cycles",
            "filename": "mac_cycles",
            "unit": "cycles/block",
            "direct_port": report["metrics"]["mac_cycles"]["original_per_block"],
            "mapped": report["metrics"]["mac_cycles"]["standard_tp_per_block"],
        },
    ]
    simulated = report.get("simulation")
    if simulated:
        rows.extend(
            [
                {
                    "metric": "Block latency incl. PP handoff",
                    "filename": "block_latency",
                    "unit": "ms/block",
                    "direct_port": simulated["latency_ms"]["original_total_per_block"],
                    "mapped": simulated["latency_ms"]["standard_tp_total_per_block"],
                },
                {
                    "metric": "System throughput",
                    "filename": "system_throughput",
                    "unit": "tokens/s",
                    "direct_port": simulated["system_throughput_tokens_s"]["original"],
                    "mapped": simulated["system_throughput_tokens_s"]["standard_tp"],
                },
                {
                    "metric": "Token energy",
                    "filename": "token_energy",
                    "unit": "mJ/token",
                    "direct_port": simulated["energy_mj"]["original_per_token_pp_total"],
                    "mapped": simulated["energy_mj"]["standard_tp_per_token_pp_total"],
                },
                {
                    "metric": "Steady pipeline power",
                    "filename": "steady_pipeline_power",
                    "unit": "W",
                    "direct_port": simulated["steady_pipeline_power_w"]["original"],
                    "mapped": simulated["steady_pipeline_power_w"]["standard_tp"],
                },
            ]
        )
    for row in rows:
        row["delta_percent"] = percent_delta(
            float(row["direct_port"]), float(row["mapped"])
        )
    return rows


def write_artifacts(report: dict, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    rows = comparison_plot_rows(report)
    with (output_dir / "comparison_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "metric",
                "unit",
                "direct_port",
                "mapped",
                "delta_percent",
            ),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = ("#949494", "#0173B2")
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.2), constrained_layout=True)
    panel_labels = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")
    for panel, (ax, row) in enumerate(zip(axes.flat, rows)):
        values = [float(row["direct_port"]), float(row["mapped"])]
        bars = ax.bar(
            ["Direct Port", "Device–Channel–\nBank"],
            values,
            color=colors,
            width=0.60,
        )
        for index, (bar, value) in enumerate(zip(bars, values)):
            if row["filename"] in ("trace_commands", "mac_cycles"):
                label = f"{value:,.0f}"
            elif row["filename"] == "block_latency":
                label = f"{value:.4f}"
            else:
                label = f"{value:,.0f}"
            if index == 1:
                label += f"\n({float(row['delta_percent']):+.2f}%)"
            ax.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
        ax.set_ylabel(str(row["unit"]))
        ax.set_title(f"{panel_labels[panel]} {row['metric']}", fontsize=11)
        ax.set_ylim(0.0, max(values) * 1.22)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        (
            "Direct Port vs. Device–Channel–Bank Systolic Mapping\n"
            "Llama2-70B · PP=80 · TP=1 · Context=4K · SA=4x16 · batch=1 · GDDR6"
        ),
        fontsize=15,
        fontweight="semibold",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"comparison_metrics.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("original", type=Path)
    parser.add_argument("standard_tp", type=Path)
    parser.add_argument("--blocks", type=int, default=80)
    parser.add_argument("--original-log", type=Path)
    parser.add_argument("--standard-tp-log", type=Path)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=8192)
    parser.add_argument("--ffn-dim", type=int, default=28672)
    parser.add_argument("--query-heads", type=int, default=64)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--gqa", type=int, default=8)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--banks", type=int, default=16)
    parser.add_argument("--sa-height", type=int, default=4)
    parser.add_argument("--pcie-lanes-per-device", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write comparison tables and one multi-panel comparison figure.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.blocks < 1:
        parser.error("--blocks must be positive")
    if bool(args.original_log) != bool(args.standard_tp_log):
        parser.error("--original-log and --standard-tp-log must be provided together")
    return args


def main() -> None:
    args = parse_args()
    report = comparison(
        trace_stats(args.original), trace_stats(args.standard_tp), args.blocks
    )
    if args.original_log:
        report["simulation"] = simulated_comparison(args)
    if args.output_dir:
        write_artifacts(report, args.output_dir)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_markdown(report)


if __name__ == "__main__":
    main()
