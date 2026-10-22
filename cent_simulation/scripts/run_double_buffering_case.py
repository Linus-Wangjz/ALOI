#!/usr/bin/env python3
"""Replay retained FP8 TP2/batch4 traces to isolate async PU and GB overlap."""
import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import yaml

CENT = Path(__file__).resolve().parents[1]
REPO = CENT.parent
sys.path.insert(0, str(CENT))
from cycle_breakdown import read_result_stats, trace_work, result_cycles, verify_trace_counts
from pim_double_buffering import read_observer, validate_events, account_global_phases

REFERENCE = CENT / "output/legacy_optimizations/LPDDR4_nCCD2_TP2_32K_batch4"
OUTPUT = CENT / "output/legacy_double_buffering/LPDDR4_nCCD2_TP2_32K_batch4"
PROFILES = {"legacy": None, "async-single": (1, False),
            "async-serialized": (2, True), "double": (2, False)}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact(report, key):
    p = Path(report["run_identity"]["artifacts"][key]["path"])
    return p if p.is_absolute() else REPO / p


def render(report):
    total, case = report["full_run_totals"], report["case"]
    lines = ["Llama2-70B FP8 SA4x32; TP2; batch4; window32768; active30976",
             "LPDDR4 x16 x32 channels; prefetch16; 32 B/burst; nBL8; nCCD2; 625 ps/clock.",
             f"Profile: {case['execution_profile']}; Flash={case['flash_attention']}; "
             f"pipelined Softmax={case['pipelined_softmax']}; channel concurrency={case['channel_concurrency']}; EWMUL_PNM=ON.",
             f"PIM={total['pim_native_cycles']:,} cycles ({total['pim_latency_ns']/1000:.6f} us); "
             f"PNM={total['pnm_exposed_latency_ns']/1000:.6f} us; CXL={total['cxl_latency_ns']/1000:.6f} us; "
             f"block={total['block_latency_ns']/1000:.6f} us.",
             "PIM totals simulated; PNM/CXL/Softmax exposure retain the reference analytical formulas.",
             "Two slots means 2 x 2 KiB/channel (4 KiB total). Extra GB area/power is uncalibrated; no energy claim.",
             "Accounting: global consecutive host phases retire in order at their last all-channel completion.",
             "States show one reference channel; other-channel/host waits are explicit. Issued counts may overlap; do not sum command spans as latency.",
             "Legacy kernel estimates remain in the reference report; this report uses completion-aware observer events.",
             "Residual EWADD arithmetic remains unmodeled; its zero ledger entry is not a hardware latency prediction.",
             "", "Kernel                              PIM clocks      PNM(us)      CXL(us)       Total(us)"]
    for r in report["regions"]:
        back = r["backends"]
        lines.append(f"{r['display_name']:<35} {back['pim']['accounted_cycles']:>10,} "
                     f"{back['pnm']['legacy_ledger_duration_ns']/1000:>12.6f} "
                     f"{back['cxl']['legacy_ledger_duration_ns']/1000:>12.6f} "
                     f"{r['legacy_ledger_duration_ns']/1000:>14.6f}")
    for r in report["regions"]:
        lines += ["", r["display_name"], f"  PIM exclusive wall cycles: {r['backends']['pim']['accounted_cycles']:,}"]
        for state, count in sorted(r["pim_wall_states"].items()):
            lines.append(f"    {state:<32} {count:>10,} cycles")
        for cmd, count in sorted(r["pim_issued_command_counts"].items()):
            lines.append(f"    issued {cmd:<25} {count:>10,} events")
        mac = r["pim_issued_command_counts"].get("MAC8", 0)
        gb = r["pim_issued_command_counts"].get("WRGB", 0)
        lines.append(f"  II-weighted work budgets (not elapsed latency): MAC8={mac:,} x 2 = {mac*2:,}; WRGB={gb:,} x 8 = {gb*8:,} clocks")
        s = r["pim_observed_span"]
        lines.append("  Observer endpoints: " + ", ".join(f"{k}={v}" for k, v in s.items()))
    lines += ["", "All-channel lifecycle/resource validation: PASS",
              f"Reference channel: {report['estimator']['reference_channel']}",
              f"Reference-channel simultaneous PU-step / WRGB-issue clocks: {report['validation']['reference_channel']['simultaneous_pu_gb_cycles']:,}",
              "Reconciliation residuals: " + str(report["reconciliation_residuals_ns"])]
    return "\n".join(lines) + "\n"


def analyze(reference, folder, stats, owners):
    total_cycles = int(result_cycles(stats))
    channels, host_completions = {}, {}
    for ch in range(32):
        path = folder / f"tasks.ch{ch}.csv"
        # Stream all channels for lifecycle checks; retain one full event stream
        # only when selected for detailed accounting.
        channels[ch] = validate_events(read_observer(path), buffers=reference["case"]["gb_buffer_count"], host_completions=host_completions)
    # Different channel masks and serialized host admission can make CH0 finish
    # early. Choose by actual last completion, with the lower channel as tie-break.
    finish = {ch: value["last_completion_cycle"] for ch, value in channels.items()}
    channel = max(finish, key=lambda ch: (finish[ch], -ch))
    events = list(read_observer(folder / f"tasks.ch{channel}.csv"))
    allocations, details, commands, spans = account_global_phases(events, owners, host_completions, total_cycles)
    report = copy.deepcopy(reference)
    report.update(method="completion_observer_accounting", legacy_framework_version="0.11",
                  profile_schema_version="0.11")
    report["estimator"] = {"id": "async_phase_completion_wall_v1", "reference_channel": channel,
                           "completion_observed": True,
                           "policy": "Consecutive host phases retire in order using all-channel completion watermarks; reference-channel state detail; command counts separate"}
    report["validation"] = {"all_channels": channels, "reference_channel": channels[channel]}
    period_ns = report["reference_clock_period_ps"] / 1000
    for r in report["regions"]:
        rid = r["region_id"]
        r["pim_issued_gap_details"] = None
        r["pim_wall_states"] = dict(details[rid])
        r["pim_issued_command_counts"] = dict(commands[rid])
        r["pim_observed_span"] = spans[rid]
        ns = allocations[rid] * period_ns
        r["backends"]["pim"].update(duration_ns=ns, legacy_ledger_duration_ns=ns,
            full_work_duration_ns=ns, reference_equivalent_cycles=allocations[rid],
            accounted_cycles=allocations[rid], measurement_kind="observed_exclusive_wall_allocation",
            coverage="observed", accept_cycle=spans[rid]["accept"],
            first_issue_cycle=spans[rid]["first_issue"], last_issue_cycle=spans[rid]["last_issue"],
            completion_cycle=spans[rid]["complete"])
        if rid.endswith("residual"):
            r["backends"]["pim"].update(duration_ns=None, full_work_duration_ns=None,
                reference_equivalent_cycles=None, measurement_kind="unmodeled", coverage="unmodeled")
        r["legacy_ledger_duration_ns"] = sum(b["legacy_ledger_duration_ns"] for b in r["backends"].values())
        r["reference_equivalent_cycles"] = r["legacy_ledger_duration_ns"] / period_ns
    totals = report["full_run_totals"]
    totals.update(pim_native_cycles=total_cycles, pim_latency_ns=total_cycles * period_ns)
    totals["block_latency_ns"] = sum(totals[k] for k in ("pim_latency_ns", "pnm_exposed_latency_ns", "cxl_latency_ns"))
    totals["reference_equivalent_cycles"] = totals["block_latency_ns"] / period_ns
    totals["per_output_token_amortized_ns"] = totals["block_latency_ns"] / report["case"]["batch_size"]
    report["reconciliation_residuals_ns"] = {"pim": sum(allocations.values())*period_ns-totals["pim_latency_ns"],
        "block": sum(r["legacy_ledger_duration_ns"] for r in report["regions"])-totals["block_latency_ns"]}
    if any(abs(x) > 1e-6 for x in report["reconciliation_residuals_ns"].values()):
        raise ValueError("Kernel timeline reconciliation failed")
    report["diagnostics"] = ["Concurrent profile: legacy FIFO issued-gap estimator disabled.",
        "Payloads are timing-only in AiM; small numerical ownership tests exercise two physical GB payload arrays.",
        "PNM/CXL/streaming-Softmax remain legacy analytical formulas; no new cross-kernel prefetch.",
        "Extra GB energy/area uncalibrated. No token energy is published."]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--profiles", choices=PROFILES, nargs="+", default=list(PROFILES))
    parser.add_argument("--variants", choices=("baseline", "full"), nargs="+", default=["baseline", "full"])
    parser.add_argument("--reuse", action="store_true", help="Reanalyze completed runs only when build/config/trace hashes match")
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_files = [REPO / p for p in (
        "aim_simulator/src/dram_controller/impl/aim_async_controller.cpp", "aim_simulator/src/dram/node.h",
        "aim_simulator/src/dram/dram.h", "aim_simulator/src/dram/impl/LPDDR4.cpp",
        "aim_simulator/src/dram_controller/controller.h", "aim_simulator/src/dram_controller/impl/plugin/trace_recorder.cpp",
        "aim_simulator/src/memory_system/impl/aim_DRAM_system.cpp", "aim_simulator/build/ramulator2",
        "aim_simulator/libramulator.so", "cent_simulation/pim_double_buffering.py", str(Path(__file__).relative_to(REPO)))]
    build = {str(p.relative_to(REPO)): sha(p) for p in source_files}
    analysis_paths = {"cent_simulation/pim_double_buffering.py", str(Path(__file__).relative_to(REPO))}
    simulation_build = {p: digest for p, digest in build.items() if p not in analysis_paths}
    comparison = []
    kernel_comparison = []
    for variant in args.variants:
        refpath = (args.reference_root / variant / "breakdown.json").resolve()
        original = json.loads(refpath.read_text())
        trace = artifact(original, "trace")
        if sha(trace) != original["run_identity"]["artifacts"]["trace"]["sha256"]:
            raise ValueError("Retained trace hash changed")
        reference_stats = read_result_stats(artifact(original, "pim_log"))
        owners = {}
        _, _, counts, _ = trace_work(trace, json.loads(artifact(original, "systolic_metadata").read_text()), 0, host_regions=owners)
        for profile in args.profiles:
            folder = root / variant / profile
            folder.mkdir(parents=True, exist_ok=True)
            config = yaml.safe_load(artifact(original, "backend_config").read_text())
            controller = config["MemorySystem"]["Controller"]
            controller["plugins"] = [{"ControllerPlugin": {"impl": "DRAMTimingExporter", "path": str(folder / "timing.yaml")}},
                                      {"ControllerPlugin": {"impl": "TraceRecorder", "path": str(folder / "issued")}}]
            if PROFILES[profile] is not None:
                slots, serialized = PROFILES[profile]
                controller.update(impl="AiMAsync", gb_buffer_count=slots, serialize_loads=serialized,
                                  descriptor_depth=64, observer_path=str(folder / "tasks"))
            config_path, log = folder / "config.yaml", folder / "result.log"
            config_text = yaml.safe_dump(config, sort_keys=False)
            manifest = {"build_sha256": simulation_build, "trace": str(trace), "trace_sha256": sha(trace),
                        "reference_report": str(refpath), "reference_sha256": sha(refpath), "config": config_text}
            manifest_path = folder / "manifest.json"
            previous_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            if previous_manifest:
                previous_manifest["build_sha256"] = {p: digest for p, digest in previous_manifest["build_sha256"].items()
                                                      if p not in analysis_paths}
            reuse = args.reuse and log.exists() and previous_manifest == manifest
            config_path.write_text(config_text)
            command = [str(REPO / "aim_simulator/build/ramulator2"), "-f", str(config_path), "-t", str(trace)]
            print(f"{variant}/{profile}: {'reanalyzing' if reuse else 'simulating'}", flush=True)
            if not reuse:
                with log.open("w") as stream:
                    subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, check=True)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            (folder / "command.json").write_text(json.dumps(command, indent=2) + "\n")
            stats = read_result_stats(log)
            verify_trace_counts(counts, stats)
            resolved = yaml.safe_load((folder / "timing.yaml").read_text())
            if (resolved["impl"] != "LPDDR4" or resolved["timing"]["nBL"] != 8 or resolved["timing"]["nCCD"] != 2):
                raise ValueError("Unexpected resolved DRAM profile")
            for ch in range(32):
                for cmd in ("MAC8", "WRGB", "WRMAC8", "RDMAC8", "RD", "WR", "AF8", "RDAF8"):
                    key = f"CH{ch}_num_{cmd}_commands"
                    if stats.get(key, 0) != reference_stats.get(key, 0):
                        raise ValueError(f"Changed terminal work count: {key}")
            if profile == "legacy":
                if int(result_cycles(stats)) != original["full_run_totals"]["pim_native_cycles"]:
                    raise ValueError("Default legacy controller changed timing")
                report = copy.deepcopy(original)
                report["case"].update(execution_profile=profile, gb_buffer_count=1)
                (folder / "summary.txt").write_text("Replayed legacy controller: exact full-run cycle match to reference.\n" +
                    (args.reference_root / variant / "summary.txt").read_text())
            else:
                reference = copy.deepcopy(original)
                reference["case"].update(execution_profile=profile, gb_buffer_count=slots,
                                         serialize_loads=serialized, gb_bytes_per_slot=2048)
                report = analyze(reference, folder, stats, owners)
                reference_identity = report.pop("run_identity")
                report["reference_run_identity"] = reference_identity
                raw = {str(path.relative_to(folder)): sha(path) for path in folder.iterdir()
                       if path.name.startswith(("tasks.", "issued.", "timing.", "config.", "result."))}
                report["run_identity"] = {"execution_manifest": manifest,
                    "raw_sha256": raw, "analysis_sha256": {p: build[p] for p in analysis_paths}}
                report["run_id"] = hashlib.sha256(json.dumps(report["run_identity"], sort_keys=True).encode()).hexdigest()[:20]
                (folder / "summary.txt").write_text(render(report))
            report["execution_manifest"] = manifest
            report["execution_artifacts"] = {"log": str(log), "config": str(config_path), "folder": str(folder)}
            (folder / "breakdown.json").write_text(json.dumps(report, indent=2) + "\n")
            totals = report["full_run_totals"]
            old = original["full_run_totals"]
            row = {"variant": variant, "profile": profile, "pim_cycles": totals["pim_native_cycles"],
                   "pim_us": totals["pim_latency_ns"]/1000, "pnm_us": totals["pnm_exposed_latency_ns"]/1000,
                   "cxl_us": totals["cxl_latency_ns"]/1000, "block_us": totals["block_latency_ns"]/1000,
                   "pim_reduction_percent": (1-totals["pim_native_cycles"]/old["pim_native_cycles"])*100,
                   "block_reduction_percent": (1-totals["block_latency_ns"]/old["block_latency_ns"])*100,
                   "summary": str(folder / "summary.txt")}
            comparison.append(row)
            for region in report["regions"]:
                r = {"variant": variant, "profile": profile, "region_id": region["region_id"],
                     "kernel": region["display_name"],
                     "pim_cycles": region["backends"]["pim"]["reference_equivalent_cycles"],
                     "pnm_us": region["backends"]["pnm"]["legacy_ledger_duration_ns"]/1000,
                     "cxl_us": region["backends"]["cxl"]["legacy_ledger_duration_ns"]/1000,
                     "total_us": region["legacy_ledger_duration_ns"]/1000}
                kernel_comparison.append(r)
            print(f"  PIM {row['pim_cycles']:,}; block {row['block_us']:.6f} us", flush=True)
    with (root / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    with (root / "kernel_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(kernel_comparison[0]))
        writer.writeheader()
        writer.writerows(kernel_comparison)
    (root / "BUILD_MANIFEST.json").write_text(json.dumps(build, indent=2) + "\n")
    lines = ["FP8 SA4x32 / LPDDR4 nCCD2 / TP2 / batch4 / window32768 / active30976",
             "Baseline keeps EWMUL_PNM on; Flash, pipelined Softmax and channel concurrency off.",
             "Full enables all four. Each variant replays its original host trace byte-for-byte.",
             "L=legacy, A1=async single GB, A2=async two GB serialized, A3=async two GB overlapping.",
             "A3 vs A2 isolates overlap; A1/A2 vs L also changes the command/mode execution profile.",
             "PNM/CXL/Softmax exposure unchanged analytical formulas; extra GB energy uncalibrated.", "",
             "Variant    Profile                PIM cycles   Block(us)  PIM reduction  Block reduction"]
    for r in comparison:
        lines.append(f"{r['variant']:<10} {r['profile']:<21} {r['pim_cycles']:>10,} {r['block_us']:>11.6f} "
                     f"{r['pim_reduction_percent']:>12.2f}% {r['block_reduction_percent']:>14.2f}%")
    for variant in args.variants:
        values = {r['profile']:r for r in comparison if r['variant']==variant}
        if 'double' in values and 'async-serialized' in values:
            a, b = values['async-serialized'], values['double']
            lines.append(f"{variant}: overlap alone reduces PIM {(1-b['pim_cycles']/a['pim_cycles'])*100:.2f}%; "
                         f"block {(1-b['block_us']/a['block_us'])*100:.2f}%; {a['pim_cycles']-b['pim_cycles']:,} PIM clocks.")
    for r in comparison:
        lines += ["", "="*78, r['variant'] + '/' + r['profile'], "="*78, Path(r['summary']).read_text()]
    (root / "summary.txt").write_text("\n".join(lines))
    print(root / "summary.txt")


if __name__ == "__main__": main()
