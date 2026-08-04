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
DEFAULT_LONG_CONTEXTS = "32768,131072"
PAPER_LONG_CONTEXT_VARIANT = "long_context_midpoint"


def parse_csv_list(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("comma-separated list must not be empty")
    return values


def parse_int_list(raw: str) -> list[int]:
    return [int(value) for value in parse_csv_list(raw)]


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


def run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=CENT_SIM, check=True)


def run_model_case(
    args: argparse.Namespace,
    case: dict[str, str | int | Path],
    model: str,
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
        "--num_devices", str(MODEL_DEVICES[model]),
        "--run_simulation_max_workers", str(args.run_workers),
        "--generate_trace_max_workers", str(args.trace_workers),
    ]
    if args.include_embedding:
        cmd.append("--process_results")
    else:
        cmd.append("--decode-only")
    if model_parallel:
        cmd.append("--model_parallel")
        if not args.master_attention:
            cmd.append("--inter-device-attention")
    if args.paper_long_context:
        for context_window, active_seqlen in contexts:
            run(cmd + [
                "--seqlen", str(active_seqlen),
                "--max-seq-len", str(context_window),
            ])
    else:
        run(cmd + ["--seqlen", *(str(active_seqlen) for _, active_seqlen in contexts)])


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
    parser.add_argument("--cases", help="Comma-separated subset of cases: GDDR6,LPDDR4X_nCCD2,LPDDR4X_nCCD6")
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument("--skip-model-parallel", action="store_true")
    args = parser.parse_args()
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

    for case in cases:
        print(f"[case] {case['name']}", flush=True)
        for model in models:
            if not args.skip_pipeline:
                run_model_case(args, case, model, contexts, model_parallel=False)
            if not args.skip_model_parallel:
                run_model_case(args, case, model, contexts, model_parallel=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
