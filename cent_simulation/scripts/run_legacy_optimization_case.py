#!/usr/bin/env python3
"""Reproduce or retain the FP8 TP2/batch4 LPDDR4 nCCD2 baseline/full pair."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

CENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CENT))
from cycle_breakdown import analyze_case, read_case, render_summary, write_report  # noqa: E402

VARIANTS = {
    "baseline": [],
    "parallel": ["--channel-concurrency"],
    "flash": ["--flash-attention"],
    "full": ["--channel-concurrency", "--flash-attention", "--pipelined-softmax"],
}


def publish_comparison(root, comparison, commands):
    """Publish an index plus complete kernel details at stable result paths."""
    root = Path(root)
    reports = {}
    for item in comparison:
        report_path = Path(item["summary"]).with_name("breakdown.json").resolve()
        report = json.loads(report_path.read_text())
        reports[item["variant"]] = report
        folder = root / item["variant"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "summary.txt").write_text(render_summary(report))
        for filename in ("breakdown.json", "breakdown.csv", "run_manifest.json"):
            link = folder / filename
            target = report_path.parent / filename
            if link.exists() or link.is_symlink():
                if not link.is_symlink():
                    raise ValueError(f"Refusing to replace a regular report file: {link}")
                link.unlink()
            link.symlink_to(target.relative_to(folder))
        item["summary"] = str((folder / "summary.txt").resolve())
    with (root / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    (root / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    lines = ["Llama2-70B FP8 SA4x32; LPDDR4 nCCD2; TP2; batch4; window32768; active30976",
             "Scope: one representative rank, one Transformer block, one batch group.",
             "Baseline: FlashAttention OFF; pipelined Softmax OFF; channel concurrency OFF.",
             "EWMUL_PNM remains ON in the baseline because the systolic implementation forces it on; this is not an all-four-off result.",
             "Full: EWMUL_PNM ON; FlashAttention ON (block1024); pipelined Softmax ON; channel concurrency ON.",
             "PIM totals are simulated; kernel PIM attribution uses issued gaps. PNM/CXL and overlap use legacy analytical formulas.",
             "", "variant       PIM cycles        PIM(us)       PNM(us)       CXL(us)     Block(us)   ScoreTransfer(us)"]
    for item in comparison:
        lines.append(f"{item['variant']:<12} {int(item['pim_cycles']):>11,} "
                     + " ".join(f"{float(item[key]):>13.6f}" for key in ("pim_us", "pnm_us", "cxl_us", "block_us"))
                     + f" {float(item['score_transfer_cycles']) * 0.000625:>19.6f}")
    for item in comparison:
        lines.extend(["", "=" * 78, f"{item['variant']} — complete kernel cycle details",
                      f"Stable summary: {item['summary']}", "=" * 78,
                      render_summary(reports[item["variant"]]).rstrip()])
    (root / "summary.txt").write_text("\n".join(lines) + "\n")


def retain_pair(root, trace_root, *, clean_legacy_v00=False):
    """Validate both retained runs before deleting redundant experiment outputs."""
    root, trace_root = Path(root).resolve(), Path(trace_root).resolve()
    with (root / "comparison.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected, retained, deletions = [], {}, []
    raw_keys = ("source_csv", "trace", "systolic_metadata", "pim_log", "resolved_timing",
                "backend_config", "issued_command_trace", "source_provenance")
    for variant in ("baseline", "full"):
        matches = [row for row in rows if row["variant"] == variant]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one indexed {variant} result")
        row = matches[0]
        report_path = Path(row["summary"]).with_name("breakdown.json").resolve()
        report = json.loads(report_path.read_text())
        artifacts = report["run_identity"]["artifacts"]

        def artifact_path(key):
            path = Path(artifacts[key]["path"])
            return (path if path.is_absolute() else CENT.parent / path).resolve()

        run_folder = artifact_path("source_csv").parent
        trace_folder = trace_root / variant / run_folder.name
        if (run_folder.parent != root / variant
                or report_path.parent.parent != run_folder / "breakdown"
                or report_path.parent.name != report["run_id"]):
            raise ValueError(f"Retained {variant} artifacts are outside the expected experiment layout")
        for key in raw_keys:
            path = artifact_path(key)
            path.relative_to(trace_folder if key in ("trace", "systolic_metadata") else run_folder)
            if hashlib.sha256(path.read_bytes()).hexdigest() != artifacts[key]["sha256"]:
                raise ValueError(f"Retained {variant} input changed: {key}")
        for filename in ("breakdown.csv", "summary.txt", "run_manifest.json"):
            if not (report_path.parent / filename).is_file():
                raise ValueError(f"Retained {variant} report is incomplete: {filename}")
        case = report["case"]
        if not case.get("ewmul_pnm") or any(case[k] != (variant == "full") for k in
                                           ("flash_attention", "pipelined_softmax", "channel_concurrency")):
            raise ValueError(f"Unexpected optimization settings in {variant}")
        details = [r["pim_issued_gap_details"] for r in report["regions"] if r.get("pim_issued_gap_details")]
        if (sum(d["total_gap_cycles"] for d in details) != report["full_run_totals"]["pim_native_cycles"]
                or any(abs(value) > 1e-6 for value in report["reconciliation_residuals_ns"].values())):
            raise ValueError(f"Retained {variant} cycle details do not reconcile")
        retained[variant] = {"run": run_folder, "report": report_path.parent, "trace": trace_folder}
        selected.append(row)
    aliases = {"summary.txt", "breakdown.json", "breakdown.csv", "run_manifest.json"}
    for variant in VARIANTS:
        for base in (root, trace_root):
            folder = base / variant
            if not folder.exists():
                continue
            if variant not in retained:
                deletions.append(folder)
                continue
            keep = retained[variant]["run" if base == root else "trace"]
            deletions.extend(p for p in folder.iterdir()
                             if p != keep and not (base == root and p.name in aliases))
        if variant in retained:
            report_dir = retained[variant]["report"]
            deletions.extend(p for p in report_dir.parent.iterdir() if p != report_dir)
    if clean_legacy_v00:
        legacy = CENT / "output/legacy_v0.0/LPDDR4_nCCD2_TP2_32K_batch4"
        if legacy.exists():
            for path in legacy.rglob("breakdown.json"):
                old = json.loads(path.read_text())
                if (old["case"]["tp"], old["case"]["batch_size"], old["full_run_totals"]["pim_native_cycles"]) != (2, 4, 653953):
                    raise ValueError(f"Unexpected legacy result; leaving cleanup untouched: {path}")
            deletions.append(legacy)
    commands = json.loads((root / "commands.json").read_text())
    if any(variant not in commands for variant in retained):
        raise ValueError("Missing reproduction command for a retained result")
    removed_bytes = sum(p.stat().st_size for target in deletions
                        for p in (target.rglob("*") if target.is_dir() else [target])
                        if p.is_file() and not p.is_symlink())
    publish_comparison(root, selected, {variant: commands[variant] for variant in retained})
    for path in deletions:
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    (root / "cleanup_manifest.json").write_text(json.dumps({
        "retained": {variant: {key: str(value) for key, value in paths.items()}
                     for variant, paths in retained.items()},
        "deleted": [str(path) for path in deletions], "removed_bytes": removed_bytes,
    }, indent=2) + "\n")
    return removed_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=CENT / "output/legacy_optimizations/LPDDR4_nCCD2_TP2_32K_batch4")
    parser.add_argument("--trace-root", type=Path, default=CENT / "trace/legacy_optimizations/LPDDR4_nCCD2_TP2_32K_batch4")
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=["baseline", "full"])
    parser.add_argument("--retain-only", action="store_true",
                        help="Keep the indexed baseline/full pair and remove redundant runs; no simulation")
    parser.add_argument("--clean-legacy-v00", action="store_true",
                        help="Also remove the duplicate legacy_v0.0 TP2/batch4 reports")
    args = parser.parse_args()
    root = args.output_root.resolve()
    if args.retain_only:
        removed = retain_pair(root, args.trace_root, clean_legacy_v00=args.clean_legacy_v00)
        print(f"Retained baseline/full; removed {removed / 1024**3:.3f} GiB. Summary: {root / 'summary.txt'}")
        return
    if args.clean_legacy_v00:
        parser.error("--clean-legacy-v00 requires --retain-only")
    root.mkdir(parents=True, exist_ok=True)
    commands, comparison = {}, []
    tracked_sources = ["cent_simulation/run_sim.py", "cent_simulation/TransformerBlock.py",
                       "cent_simulation/aim_sim.py", "cent_simulation/tp_mapping.py",
                       "aim_simulator/src/frontend/impl/memory_trace/aim_trace.cpp",
                       "aim_simulator/src/memory_system/impl/aim_DRAM_system.cpp",
                       "aim_simulator/build/ramulator2", "aim_simulator/libramulator.so"]
    manifest = {"sha256": {name: hashlib.sha256((CENT.parent / name).read_bytes()).hexdigest()
                            for name in tracked_sources}, "python": sys.version}
    manifest_text = json.dumps(manifest, indent=2) + "\n"
    (root / "BUILD_MANIFEST.json").write_text(manifest_text)
    provenance = root / "RUN_PROVENANCE.md"
    provenance.write_text(
        "# FP8 LPDDR4 nCCD2 TP2 batch4 optimization comparison\n\n"
        "Window=32768; active decode length=30976; array=4x32; 32 channels; EWMUL_PNM=true.\n"
        "FP8 geometry and legacy analytical formulas restored from ALOI 637537c.\n"
        "PNM/CXL and streamed Softmax remain legacy analytical models; FlashAttention is a timing-trace abstraction.\n"
        "Concurrent mode accepts at most one request/channel/cycle; DRAM timing and SYNC/EOC remain active.\n"
        "WR_GB retains multicast semantics. Native full-run PIM cycles are simulated; kernel attribution uses issued gaps.\n"
        "Reproduce: python scripts/run_legacy_optimization_case.py\n"
        f"BUILD_MANIFEST.json SHA256: {hashlib.sha256(manifest_text.encode()).hexdigest()}\n"
    )
    for variant in args.variants:
        input_fingerprint = hashlib.sha256(json.dumps(
            {"build": manifest, "options": VARIANTS[variant]}, sort_keys=True).encode()).hexdigest()[:16]
        out = root / variant / input_fingerprint
        trace_root = args.trace_root.resolve() / variant / input_fingerprint
        out.mkdir(parents=True, exist_ok=True)
        run_provenance = out / "RUN_PROVENANCE.md"
        run_provenance.write_text(provenance.read_text())
        (out / "BUILD_MANIFEST.json").write_text(manifest_text)
        source = out / "simulation_results.csv"
        command = [sys.executable, "run_sim.py", "--model", "Llama2-70B",
                   "--precision", "fp8", "--num_channels", "32", "--num_banks", "8",
                   "--num_devices", "32", "--PCIE_lanes", "144", "--model_parallel",
                   "--kv-head-tp", "--tp-values", "2", "--decode-only", "--systolic-pim",
                   "--systolic-dim", "4", "--batch-size", "4", "--EWMUL_PNM",
                   "--seqlen", "30976", "--max-seq-len", "32768",
                   "--experiment", "LPDDR4X_nCCD2", "--dram-power-impl", "LPDDR4X",
                   "--ramulator-config", str(CENT.parent / "aim_simulator/test/generated/example_LPDDR4X_nCCD2.yaml"),
                   "--trace-root", str(trace_root), "--log-root", str(out / "ramulator"),
                   "--simulation_result_path", str(source), "--record-command-trace",
                   "--generate_trace", "--simulate_trace", "--update_csv", *VARIANTS[variant]]
        commands[variant] = command
        print(f"Running {variant}", flush=True)
        with (out / "driver.log").open("w") as stream:
            subprocess.run(command, cwd=CENT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        trace = next(trace_root.rglob("trace_2_FC_devices_seqlen_30976.txt"))
        log = next((out / "ramulator").rglob("trace_2_FC_devices_seqlen_30976.txt.log"))
        stem = str(log).removesuffix(".log")
        row = read_case(source, model="Llama2-70B", tp=2, batch=4, context_window=32768, sequence_length=30976)
        report = analyze_case(row, source_csv=source, trace=trace, log=log,
                              timing=stem + ".timing.yaml", metadata_path=str(trace) + ".systolic.json",
                              config=stem + ".ramulator.yaml", provenance=run_provenance)
        report_dir = write_report(report, out / "breakdown")
        totals = report["full_run_totals"]
        comparison.append({"variant": variant, "pim_cycles": totals["pim_native_cycles"],
                           "pim_us": totals["pim_latency_ns"] / 1000,
                           "pnm_us": totals["pnm_exposed_latency_ns"] / 1000,
                           "cxl_us": totals["cxl_latency_ns"] / 1000,
                           "block_us": totals["block_latency_ns"] / 1000,
                           "score_transfer_cycles": next(r["backends"]["pim"]["reference_equivalent_cycles"]
                                                         for r in report["regions"] if r["region_id"] == "aux.score_transfer"),
                           "summary": str(report_dir / "summary.txt")})
        print(f"{variant}: {comparison[-1]['block_us']:.6f} us; {report_dir}", flush=True)
    if (root / "comparison.csv").exists():
        with (root / "comparison.csv").open(newline="") as stream:
            prior = list(csv.DictReader(stream))
        comparison = [r for r in prior if r["variant"] not in args.variants] + comparison
    previous_commands = json.loads((root / "commands.json").read_text()) if (root / "commands.json").exists() else {}
    previous_commands.update(commands)
    publish_comparison(root, comparison, {item["variant"]: previous_commands[item["variant"]] for item in comparison})
    print(f"Comparison: {root / 'comparison.csv'}")


if __name__ == "__main__":
    main()
