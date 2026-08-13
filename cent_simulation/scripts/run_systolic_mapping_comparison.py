#!/usr/bin/env python3
"""Run and plot the PP80/TP1 direct-port versus physical-mapping comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


CENT_SIM = Path(__file__).resolve().parents[1]
ROOT = CENT_SIM.parent
DEFAULT_OUTPUT = CENT_SIM / "output/kv_head_tp_systolic_all_context/comparison"
DEFAULT_TRACE = CENT_SIM / "trace/kv_head_tp_systolic_all_context/comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE)
    parser.add_argument(
        "--gddr6-yaml",
        type=Path,
        default=ROOT / "aim_simulator/test/example_GDDR6.yaml",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command: list[str], *, dry_run: bool) -> None:
    print("[run]", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=CENT_SIM, check=True)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    trace_dir = args.trace_dir.resolve()
    python = sys.executable
    direct_variant = "systolic_pim_4_batch_size_1_ewmul_pnm_1"
    mapped_variant = direct_variant
    common = [
        python,
        "run_sim.py",
        "--model", "Llama2-70B",
        "--num_channels", "32",
        "--num_banks", "16",
        "--num_devices", "80",
        "--PCIE_lanes", "144",
        "--model_parallel",
        "--tp-values", "1",
        "--decode-only",
        "--seqlen", "4096",
        "--max-seq-len", "4096",
        "--systolic-pim",
        "--systolic-dim", "4",
        "--batch-size", "1",
        "--EWMUL_PNM",
        "--generate_trace",
        "--simulate_trace",
        "--update_csv",
        "--generate_trace_max_workers", str(args.workers),
        "--run_simulation_max_workers", str(args.workers),
        "--ramulator-config", str(args.gddr6_yaml.resolve()),
        "--dram-power-impl", "GDDR6",
        "--dram-energy-model", "legacy",
    ]
    direct_trace_root = trace_dir / "direct_port"
    mapped_trace_root = trace_dir / "device_channel_bank"
    direct_log_root = output_dir / "raw/direct_port"
    mapped_log_root = output_dir / "raw/device_channel_bank"
    direct_csv = direct_log_root / "simulation_results.csv"
    mapped_csv = mapped_log_root / "simulation_results.csv"

    run(
        common
        + [
            "--experiment", "GDDR6_direct_port",
            "--trace-root", str(direct_trace_root),
            "--log-root", str(direct_log_root),
            "--simulation_result_path", str(direct_csv),
        ],
        dry_run=args.dry_run,
    )
    run(
        common
        + [
            "--kv-head-tp",
            "--experiment", "GDDR6_device_channel_bank",
            "--trace-root", str(mapped_trace_root),
            "--log-root", str(mapped_log_root),
            "--simulation_result_path", str(mapped_csv),
        ],
        dry_run=args.dry_run,
    )

    direct_trace = (
        direct_trace_root
        / direct_variant
        / "model_parallel/Llama2-70B/trace_1_FC_devices_seqlen_4096.txt"
    )
    mapped_trace = (
        mapped_trace_root
        / mapped_variant
        / "model_parallel_kv_head_main/Llama2-70B/trace_1_FC_devices_seqlen_4096.txt"
    )
    direct_log = (
        direct_log_root
        / direct_variant
        / "model_parallel/Llama2-70B/trace_1_FC_devices_seqlen_4096.txt.log"
    )
    mapped_log = (
        mapped_log_root
        / mapped_variant
        / "model_parallel_kv_head_main/Llama2-70B/trace_1_FC_devices_seqlen_4096.txt.log"
    )
    compare = [
        python,
        "scripts/compare_systolic_tp_traces.py",
        str(direct_trace),
        str(mapped_trace),
        "--blocks", "80",
        "--original-log", str(direct_log),
        "--standard-tp-log", str(mapped_log),
        "--context", "4096",
        "--sa-height", "4",
        "--pcie-lanes-per-device", "1",
        "--output-dir", str(output_dir),
    ]
    run(compare, dry_run=args.dry_run)

    if not args.dry_run:
        manifest = {
            "model": "Llama2-70B",
            "context": 4096,
            "pipeline_parallelism": 80,
            "tensor_parallelism": 1,
            "batch_size": 1,
            "systolic_array": "4x16",
            "ewmul_pnm_requested": True,
            "ewmul_pnm_effective": True,
            "memory": "GDDR6",
            "direct_port": "trace_only_systolic_PIM",
            "mapped": "trace_only_systolic_TP",
            "pp_handoff_included": True,
            "pcie_lanes_per_device": 1,
            "direct_trace": str(direct_trace),
            "mapped_trace": str(mapped_trace),
            "direct_log": str(direct_log),
            "mapped_log": str(mapped_log),
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
