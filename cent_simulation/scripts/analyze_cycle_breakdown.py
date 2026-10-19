#!/usr/bin/env python3
"""Legacy v0.0: analyze an existing Llama2-70B systolic KV-head TP case."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

CENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CENT_ROOT))

from cycle_breakdown import analyze_case, read_case, write_report  # noqa: E402


def plot_report(report, run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in report["regions"] if r["legacy_ledger_duration_ns"] > 0]
    fig, axis = plt.subplots(figsize=(11, max(5, len(rows) * 0.38)))
    starts = [0.0] * len(rows)
    for backend, color, label in (("pim", "#4c78a8", "PIM estimate"),
                                  ("pnm", "#f58518", "PNM legacy formula"),
                                  ("cxl", "#54a24b", "CXL legacy formula")):
        values = [r["backends"][backend]["legacy_ledger_duration_ns"] / 1000 for r in rows]
        axis.barh(range(len(rows)), values, left=starts, label=label, color=color)
        starts = [x + y for x, y in zip(starts, values)]
    axis.set_yticks(range(len(rows)), [r["display_name"] for r in rows])
    axis.invert_yaxis()
    case = report["case"]
    axis.set_title(f"{case['model']} · {case['dram_timing_impl']} nCCD={case['nCCD']} · TP={case['tp']} · B={case['batch_size']}\n"
                   f"{case['precision']} SA{case['array_height']}×{case['array_width']}; window={case['context_window']}, active={case['sequence_length']}")
    axis.set_xlabel("Legacy block batch-group latency contribution (µs)")
    axis.legend(loc="lower right")
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(Path(run_dir) / "breakdown.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--timing", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--command-trace", type=Path)
    parser.add_argument("--model", default="Llama2-70B")
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--context-window", type=int, required=True)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--expected-dram", choices=("GDDR6", "LPDDR4", "LPDDR4X"))
    parser.add_argument("--expected-nccd", type=int)
    parser.add_argument("--estimator", choices=("auto", "issued-gap", "service-work"), default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    timing = args.timing or Path(str(args.log).removesuffix(".log") + ".timing.yaml")
    metadata = args.metadata
    if metadata is None and args.trace is not None:
        candidate = Path(str(args.trace) + ".systolic.json")
        if candidate.exists():
            metadata = candidate
    config = args.config
    if config is None:
        candidate = Path(str(args.log).removesuffix(".log") + ".ramulator.yaml")
        if candidate.exists():
            config = candidate
    try:
        row = read_case(args.source_csv, model=args.model, tp=args.tp, batch=args.batch_size,
                        context_window=args.context_window, sequence_length=args.sequence_length)
        report = analyze_case(row, log=args.log, timing=timing, source_csv=args.source_csv,
                              trace=args.trace, metadata_path=metadata, config=config,
                              provenance=args.provenance, command_trace=args.command_trace,
                              estimator=args.estimator)
        if args.expected_dram is not None and report["case"]["dram_timing_impl"] != args.expected_dram:
            raise ValueError("Resolved DRAM implementation does not match --expected-dram")
        if args.expected_nccd is not None and report["case"]["nCCD"] != args.expected_nccd:
            raise ValueError("Resolved nCCD does not match --expected-nccd")
        output = write_report(report, args.output_dir)
        if args.plot:
            plot_report(report, output)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    case, totals = report["case"], report["full_run_totals"]
    print(f"Report: {output}")
    print(f"{case['precision']} SA{case['array_height']}x{case['array_width']}; "
          f"window={case['context_window']}, active={case['sequence_length']}, TP={case['tp']}, B={case['batch_size']}")
    print(f"PIM: {totals['pim_native_cycles']:,} measured cycles; "
          f"legacy block: {totals['block_latency_ns'] / 1000:.6f} us")
    print(f"PIM attribution: {report['estimator']['id']}; completion timestamps unavailable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
