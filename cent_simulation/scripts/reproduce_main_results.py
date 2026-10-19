#!/usr/bin/env python3
"""Reproduce the historical ALOI spreadsheet without changing the working tree.

Exports the original Python/power-model revision into the output directory,
generates fresh traces, runs the installed Ramulator, and compares the results.
FP8 uses the historical baseline; BF16 enables FlashAttention/pipelined Softmax.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[2]
ALOI_REVISION = "637537c3bc4f7a02f6d2f5fa50ce9de6c18e55a0"
AIM_REVISION = "22041f2b59446cd034963322335f08f1a1f13ab8"
MODEL = "Llama2-70B"
CASES = (
    ("fp8_8K_tp1", "fp8", 8192, 1),
    ("fp8_16K_tp1", "fp8", 16384, 1),
    ("fp8_32K_tp2", "fp8", 32768, 2),
    ("fp8_64K_tp4", "fp8", 65536, 4),
    ("fp8_32K_tp1", "fp8", 32768, 1),
    ("bf16_32K_tp2", "bf16", 32768, 2),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_revision(repo: Path, revision: str, destination: Path, paths=()) -> None:
    archive = subprocess.check_output(["git", "-C", str(repo), "archive", revision, *paths])
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        stream.extractall(destination, filter="data")


def run_case(source: Path, output: Path, case, python: str, env: dict,
             *, micron: bool = False) -> dict:
    name, precision, window, tp = case
    run_name = name + ("_micron" if micron else "")
    target = output / run_name
    target.mkdir(parents=True, exist_ok=True)
    command = [
        python, "run_sim.py", "--model", MODEL, "--precision", precision,
        "--num_channels", "32", "--num_banks", "8", "--num_devices", "32",
        "--PCIE_lanes", "144", "--model_parallel", "--kv-head-tp",
        "--tp-values", str(tp), "--decode-only", "--systolic-pim",
        "--systolic-dim", "4", "--batch-size", "4", "--EWMUL_PNM",
        "--seqlen", str(window - 1792), "--max-seq-len", str(window),
        "--experiment", "LPDDR4X_nCCD2", "--dram-power-impl",
        "LPDDR4X_MICRON" if micron else "LPDDR4X",
        "--ramulator-config", str(output / "LPDDR4X_nCCD2.yaml"),
        "--trace-root", str(output / "traces" / name),
        "--log-root", str(output / name / "ramulator"),
        "--simulation_result_path", str(target / "simulation_results.csv"),
        "--update_csv",
    ]
    if not micron:
        command.extend(["--generate_trace", "--simulate_trace"])
    if precision == "bf16":
        command.extend(["--flash-attention", "--flash-attention-block-size", "1024",
                        "--pipelined-softmax"])
    print(f"Running {run_name}", flush=True)
    with (target / "driver.log").open("w") as stream:
        subprocess.run(command, cwd=source / "cent_simulation", env=env,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    return {"case": run_name, "cwd": str(source / "cent_simulation"), "argv": command}


def summarize(source: Path, output: Path, spreadsheet: Path) -> list[dict]:
    # Import only the exported historical analysis and its matching power model.
    sys.path.insert(0, str(source / "cent_simulation"))
    from scripts import analyze_kv_head_tp_systolic_all_context as analysis
    import pandas as pd
    import yaml

    analysis.configure_scope([MODEL], ["8K", "16K", "32K", "64K"], ["LPDDR4X_nCCD2"])
    targets = list(csv.DictReader(spreadsheet.open(newline="")))[:7]
    comparisons, candidates = [], []
    for index, target in enumerate(targets):
        if not target["System"].startswith("ALOI"):
            continue
        context, precision = target["Context"], target["Precision"]
        vendor = "micron" if "Micron" in target["System"] else "winbond"
        reported_pp, reported_tp = int(target["PP"]), int(target["TP"])
        # Also evaluate literal spreadsheet options when a value was transcribed incorrectly.
        tps = [reported_tp, 2] if vendor == "micron" else [reported_tp]
        for tp in dict.fromkeys(tps):
            name = f"{precision.lower()}_{context}_tp{tp}"
            run_name = name + ("_micron" if vendor == "micron" else "")
            csv_path = output / run_name / "simulation_results.csv"
            rows = pd.read_csv(csv_path)
            assert len(rows) == 1, csv_path
            row = rows.iloc[0]
            optimized = precision == "BF16"
            assert bool(row["Flash attention"]) == optimized
            assert bool(row["Pipelined softmax"]) == optimized
            assert bool(row["EWMUL PNM effective"])
            if optimized:
                assert int(row["Flash attention block size"]) == 1024
                assert float(row["Exposed softmax latency"]) < float(row["Full softmax latency"])
            else:
                assert float(row["Softmax exposure factor"]) == 1.0
                assert float(row["Full softmax latency"]) == float(row["Exposed softmax latency"])
            configs = list((output / name / "ramulator").rglob("*.ramulator.yaml"))
            assert len(configs) == 1, configs
            config = yaml.safe_load(configs[0].read_text())
            assert config["Frontend"].get("issue_width", 1) == 1
            assert config["MemorySystem"].get("host_issue_width", 1) == 1
            impl = "LPDDR4X_MICRON" if vendor == "micron" else "LPDDR4X"
            for pp in analysis.balanced.exact_pp_values(MODEL):
                candidate = analysis.build_systolic_batch_candidate(
                    "LPDDR4X_nCCD2", impl, csv_path, row, MODEL, context,
                    pp, tp, 4, 16.0, 0.0, precision,
                )
                candidate.update({"Vendor": vendor, "Spreadsheet row": index + 2})
                candidates.append(candidate)
                if pp != reported_pp:
                    continue
                actual = {
                    "Energy Efficiency (Tokens/J)": candidate["Tokens/J"],
                    "Throughput/Device (Tokens/s)": candidate["Throughput / device (tokens/s/device)"],
                    "Device Power (W)": candidate["Power / device (W)"],
                }
                result = {
                    "Spreadsheet row": index + 2, "System": target["System"],
                    "Reported model": target["Model"], "Simulator model": MODEL,
                    "Context": context, "Active sequence length": int(row["Sequence length"]),
                    "Precision": precision, "Vendor": vendor, "PP": pp,
                    "Reported TP": reported_tp, "Reproduced TP": tp, "Batch": 4,
                    "Flash attention": optimized, "Online/pipelined softmax": optimized,
                    "Flash attention block size": 1024 if optimized else 0,
                    "Channel concurrency": False, "Frontend issue width": 1,
                    "Memory host issue width": 1, "EWMUL PNM": True,
                    "DRAM energy model": str(row["DRAM energy model"]),
                    "Source CSV": str(csv_path),
                }
                matches = []
                for metric, value in actual.items():
                    expected = float(target[metric])
                    relative = abs(value / expected - 1.0)
                    result[f"Reported {metric}"] = expected
                    result[f"Reproduced {metric}"] = value
                    result[f"Relative error {metric}"] = relative
                    matches.append(relative < 1e-8)
                result["Matches reported numbers"] = all(matches)
                result["Status"] = (
                    "matches after correcting Micron TP to 2" if all(matches) and tp != reported_tp
                    else "matches" if all(matches)
                    else "literal Micron TP does not match" if vendor == "micron"
                    else "throughput matches; power/energy differ"
                )
                comparisons.append(result)
    for filename, rows in (("comparison.csv", comparisons), ("candidates.csv", candidates)):
        with (output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "cent_simulation/output/main_results_reproduction/historical")
    parser.add_argument("--spreadsheet", type=Path,
                        default=ROOT / "docs/ALOI General Information - Main Results.csv")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--ramulator", type=Path, default=ROOT / "aim_simulator/build/ramulator2")
    parser.add_argument("--library", type=Path, default=ROOT / "aim_simulator/libramulator.so")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh --output-root; refusing to reuse simulation logs: {output}")
    source = output / "source"
    export_revision(ROOT, ALOI_REVISION, source, ("cent_simulation", "DGX_H100_profile_results.csv"))
    export_revision(ROOT / "aim_simulator", AIM_REVISION, source / "aim_simulator",
                    ("scripts", "test", "src", "CMakeLists.txt"))
    binary = source / "aim_simulator/build/ramulator2"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.symlink_to(args.ramulator.resolve())
    (source / "aim_simulator/libramulator.so").symlink_to(args.library.resolve())
    # Explicitly preserve the original width-one admission settings.
    (output / "LPDDR4X_nCCD2.yaml").write_text(
        "Frontend:\n  impl: AiMTrace\n  clock_ratio: 1\n"
        "  Translation:\n    impl: NoTranslation\n    max_addr: 2147483648\n"
        "MemorySystem:\n  impl: AiMDRAM\n  clock_ratio: 1\n"
        "  DRAM:\n    impl: LPDDR4\n    org:\n      preset: LPDDR4_AiM_org\n"
        "    timing:\n      preset: LPDDR4_AiM_timing\n      nCCD: 2\n"
        "  Controller:\n    impl: AiM\n    clock_ratio: 1\n"
        "    Scheduler:\n      impl: FRFCFS\n    RefreshManager:\n      impl: AllBank\n"
        "    plugins:\n  AddrMapper:\n    impl: RoBaRaCoCh\n"
    )
    env = dict(os.environ)
    env["MPLCONFIGDIR"] = str(output / "mpl_cache")
    env["LD_LIBRARY_PATH"] = str(source / "aim_simulator") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    manifest = {
        "aloi_revision": ALOI_REVISION, "aim_python_revision": AIM_REVISION,
        "ramulator_binary": str(args.ramulator.resolve()),
        "ramulator_sha256": sha256(args.ramulator),
        "library": str(args.library.resolve()), "library_sha256": sha256(args.library),
        "python": sys.version, "input_sha256": sha256(args.spreadsheet),
        "frontend_issue_width": 1, "memory_host_issue_width": 1,
        "note": "Historical Python and power tables with the installed backend in legacy width-one mode; no backend rebuild.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        commands = list(executor.map(lambda case: run_case(source, output, case, sys.executable, env), CASES))
    for case in CASES:
        if case[0] in ("fp8_32K_tp1", "fp8_32K_tp2"):
            commands.append(run_case(source, output, case, sys.executable, env, micron=True))
    (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    comparisons = summarize(source, output, args.spreadsheet)
    fp8_matches = [r for r in comparisons if r["Precision"] == "FP8" and r["Matches reported numbers"]]
    bf16 = next(r for r in comparisons if r["Precision"] == "BF16")
    if len(fp8_matches) != 5 or bf16["Relative error Throughput/Device (Tokens/s)"] >= 1e-8:
        raise RuntimeError("Unexpected reproduction difference; inspect comparison.csv")
    for row in comparisons:
        print(f"{row['Vendor']} {row['Precision']} {row['Context']} PP{row['PP']} "
              f"TP{row['Reproduced TP']}: {row['Reproduced Throughput/Device (Tokens/s)']:.9f} "
              f"tokens/s/device; {row['Reproduced Device Power (W)']:.9f} W; "
              f"{row['Reproduced Energy Efficiency (Tokens/J)']:.9f} tokens/J; "
              f"match={row['Matches reported numbers']}", flush=True)
    print(f"Results: {output / 'comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
