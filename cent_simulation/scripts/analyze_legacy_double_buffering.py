#!/usr/bin/env python3
"""Analytical WR_GB overlap deltas on immutable legacy single-buffer results.

This does not run Ramulator, alter its command counts, or predict energy.
Each load can overlap only the immediately preceding compute wave: two GBs,
one current operand and one prefetched operand. PIM/PNM non-WRGB costs stay
fixed; CXL bandwidth units are explicit for the single/double-buffer comparison.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

from legacy_cxl_units import apply_cxl_units
from double_buffer_kernel_table import write_double_buffer_kernel_table


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "cent_simulation/output/legacy_optimizations/LPDDR4_nCCD2_TP2_32K_batch4_flash2048_pnm_residual"
REGIONS = {
    "q": "attention.q_projection", "fused_kv": "attention.kv_projection",
    "qk": "attention.qk", "sv_paired_kv_heads": "attention.sv",
    "wo": "attention.o_projection", "fused_w1_w3": "ffn.up_gate_projection",
    "w2": "ffn.down_projection",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def resolve_artifact(record, search_root):
    """Accept a relocated artifact only when its complete SHA256 matches."""
    original = Path(record["path"])
    candidate = original if original.is_absolute() else REPO / original
    if candidate.is_file() and digest(candidate) == record["sha256"]:
        return candidate
    for candidate in sorted(Path(search_root).rglob(original.name)):
        if candidate.is_file() and digest(candidate) == record["sha256"]:
            return candidate
    raise ValueError(f"No exact hash-matching copy of {record['path']}")


@dataclass
class Wave:
    index: int
    phase: int
    kernel: str
    trace_line: int
    bursts: int
    mac_commands: int = 0
    barrier_before: bool = False


def decode_waves(trace, metadata, channel):
    phases = metadata["command_phases"]
    phase, remaining = -1, 0
    waves, counts = [], defaultdict(Counter)
    barrier = True
    active_wave = None
    eoc = 0
    with Path(trace).open() as stream:
        for line_number, line in enumerate(stream, 1):
            f = line.split()
            if not f or f[0] == "#":
                continue
            if f[0] in ("R", "W"):
                barrier = True
                continue
            if f[0] != "AiM":
                raise ValueError(f"Unsupported trace line {line_number}")
            op = f[1].removeprefix("ISR_")
            if op in ("SYNC", "EOC", "EWADD", "AF", "RD_AF"):
                barrier = True
                eoc += op == "EOC"
                continue
            if op == "WR_BIAS" and remaining == 0:
                phase += 1
                if phase >= len(phases):
                    raise ValueError("Trace has more phases than metadata")
                remaining = phases[phase]["mac_work"]
                active_wave = None
            if phase < 0 or op not in ("WR_BIAS", "WR_GB", "MAC_ABK", "RD_MAC"):
                raise ValueError(f"Unsupported phase protocol at line {line_number}")
            kernel = phases[phase]["kernel"]
            mask = int(f[4] if op == "WR_GB" else f[3], 0)
            active = bool(mask & (1 << (31 - channel)))
            work = int(f[2]) if op in ("WR_GB", "MAC_ABK") else 1
            if active:
                counts[kernel][op] += work
            if op == "WR_GB":
                active_wave = None
                if active:
                    active_wave = Wave(len(waves), phase, kernel, line_number, work,
                                       barrier_before=barrier)
                    waves.append(active_wave)
                    barrier = False
            elif op == "MAC_ABK":
                remaining -= work
                if remaining < 0:
                    raise ValueError("MAC work exceeds phase budget")
                if active:
                    if active_wave is None:
                        raise ValueError("MAC has no matching GB load")
                    active_wave.mac_commands += work
    if phase != len(phases) - 1 or remaining or eoc != 1:
        raise ValueError("Incomplete trace or phase metadata")
    if not waves or any(w.bursts <= 0 or w.mac_commands <= 0 for w in waves):
        raise ValueError("Empty load/compute wave")
    return waves, counts


def readiness(previous, current, ideal=False):
    if previous is None:
        return False, "first_load"
    if ideal:
        return True, "ideal_inputs_already_ready"
    if current.barrier_before:
        return False, "memory_or_sync_boundary"
    if current.phase == previous.phase:
        return True, "same_accumulator_phase"
    if current.kernel == previous.kernel:
        return True, "next_tile_or_query_wave_inputs_ready"
    if (previous.kernel, current.kernel) == ("q", "fused_kv"):
        return True, "Q_and_KV_share_already_available_hidden_input"
    if (previous.kernel, current.kernel) == ("sv_paired_kv_heads", "qk"):
        return True, "next_context_or_batch_query_already_available"
    if (previous.kernel, current.kernel) == ("qk", "sv_paired_kv_heads"):
        return False, "whole_chunk_QK_then_Softmax_dependency"
    return False, "producer_PNM_or_collective_dependency"


def overlap_rows(waves, nbl, nccd, ideal=False):
    rows = []
    previous = None
    for wave in waves:
        ready, reason = readiness(previous, wave, ideal)
        load = wave.bursts * nbl
        window = previous.mac_commands * nccd if previous is not None and ready else 0
        hidden = min(load, window)
        rows.append({**asdict(wave), "region_id": REGIONS[wave.kernel],
                     "previous_wave": previous.index if previous else None,
                     "previous_kernel": previous.kernel if previous else None,
                     "readiness": reason, "wrgb_cycles": load,
                     "mac_cycles": wave.mac_commands * nccd,
                     "eligible_compute_cycles": window, "hidden_wrgb_cycles": hidden,
                     "exposed_wrgb_cycles": load - hidden})
        previous = wave
    return rows


def issued_wrgb_gaps(waves, path, channel):
    """Align WRGB issue gaps with loads, using the same ownership as the source.

    A gap belongs to the following command. In particular CASWRGB setup and
    gaps after the last WRGB belong to other commands, not to this budget.
    """
    gaps, previous_clock = [], 0
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            fields = [f.strip() for f in line.split(",")]
            clock = int(fields[0])
            if clock < previous_clock or int(fields[2]) != channel:
                raise ValueError("Issued trace is not ordered on the reference channel")
            if fields[1] == "WRGB":
                gaps.append(clock - previous_clock)
            previous_clock = clock
    if len(gaps) != sum(w.bursts for w in waves):
        raise ValueError("Issued WRGB events do not match decoded load bursts")
    grouped, start = [], 0
    for wave in waves:
        grouped.append(gaps[start:start + wave.bursts])
        start += wave.bursts
    return grouped


def credited_wrgb_overlap(nominal_hidden, nbl, gaps):
    """Credit only the overlapped prefix's WRGB-owned gap, at most nBL/burst.

    Never borrow a later, unhidden burst's gap or any CAS/control budget to
    make up the difference between nominal wire time and issued-gap time.
    """
    if nbl <= 0 or not 0 <= nominal_hidden <= len(gaps) * nbl or any(g < 0 for g in gaps):
        raise ValueError("Invalid WRGB overlap/gap budget")
    return sum(min(max(nominal_hidden - index * nbl, 0), nbl, gap)
               for index, gap in enumerate(gaps))


def validate_counts(source, metadata, counts):
    by_region = {r["region_id"]: r for r in source["regions"]}
    for kernel, observed in counts.items():
        region = by_region[REGIONS[kernel]]
        issued = {x["command"]: x["issued_event_count"]
                  for x in region["pim_issued_gap_details"]["commands"]}
        for host, command in (("WR_GB", "WRGB"), ("MAC_ABK", "MAC8"),
                              ("WR_BIAS", "WRMAC8"), ("RD_MAC", "RDMAC8")):
            if observed[host] != issued[command]:
                raise ValueError(f"{kernel}: {host} differs from saved issued counts")
        if observed["MAC_ABK"] != sum(metadata["kernel_pipeline_cycles"][kernel].values()):
            raise ValueError(f"{kernel}: metadata MAC stages do not reconcile")


def model_report(source, waves, ideal, wave_gaps):
    case = source["case"]
    rows = overlap_rows(waves, case["nBL"], case["nCCD"], ideal)
    if len(rows) != len(wave_gaps):
        raise ValueError("Missing per-wave issued WRGB gaps")
    for row, gaps in zip(rows, wave_gaps):
        if len(gaps) != row["bursts"]:
            raise ValueError("Wave burst count does not match issued WRGB gaps")
        row["nominal_hidden_wrgb_cycles"] = row["hidden_wrgb_cycles"]
        row["nominal_exposed_wrgb_cycles"] = row["exposed_wrgb_cycles"]
        row["old_wrgb_gap_cycles"] = sum(gaps)
        row["first_wrgb_gap_cycles"] = gaps[0]
        row["issued_wrgb_gap_histogram"] = json.dumps(dict(sorted(Counter(gaps).items())))
        row["hidden_wrgb_cycles"] = credited_wrgb_overlap(
            row["nominal_hidden_wrgb_cycles"], case["nBL"], gaps)
        row["exposed_wrgb_cycles"] = row["old_wrgb_gap_cycles"] - row["hidden_wrgb_cycles"]
        row["uncredited_nominal_overlap_cycles"] = (
            row["nominal_hidden_wrgb_cycles"] - row["hidden_wrgb_cycles"])
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["region_id"]].append(row)
    period_ns = source["reference_clock_period_ps"] / 1000
    kernels = []
    for region in source["regions"]:
        wr = grouped[region["region_id"]]
        hidden = sum(w["hidden_wrgb_cycles"] for w in wr)
        before = region["backends"]["pim"]["reference_equivalent_cycles"]
        old_ns = region["legacy_ledger_duration_ns"]
        new_ns = old_ns - hidden * period_ns
        if before is not None and before < hidden:
            raise ValueError("Overlap exceeds the saved region budget")
        commands = (region.get("pim_issued_gap_details") or {}).get("commands", [])
        attributed = sum(c["gap_cycles"] for c in commands if c["command"] == "WRGB")
        cas = sum(c["gap_cycles"] for c in commands if c["command"] == "CASWRGB")
        if sum(w["old_wrgb_gap_cycles"] for w in wr) != attributed:
            raise ValueError(f"{region['region_id']}: issued WRGB gaps differ from source report")
        if not 0 <= hidden <= attributed:
            raise ValueError("Credited overlap exceeds WRGB-only attribution")
        kernels.append({
            "region_id": region["region_id"], "kernel": region["display_name"],
            "gb_load_waves": len(wr), "accumulator_phases": len({w["phase"] for w in wr}),
            "wrgb_bursts": sum(w["bursts"] for w in wr),
            "wrgb_wire_cycles": sum(w["wrgb_cycles"] for w in wr),
            "old_wrgb_gap_cycles": attributed,
            "caswrgb_gap_cycles_unchanged": cas,
            "non_wrgb_pim_cycles_unchanged": None if before is None else before - attributed,
            "nominal_hidden_wrgb_cycles": sum(w["nominal_hidden_wrgb_cycles"] for w in wr),
            "nominal_exposed_wrgb_cycles": sum(w["nominal_exposed_wrgb_cycles"] for w in wr),
            "uncredited_nominal_overlap_cycles": sum(w["uncredited_nominal_overlap_cycles"] for w in wr),
            "hidden_wrgb_cycles": hidden,
            "exposed_wrgb_cycles": sum(w["exposed_wrgb_cycles"] for w in wr),
            "cold_loads": sum(w["eligible_compute_cycles"] == 0 for w in wr),
            "partly_hidden_loads": sum(0 < w["nominal_hidden_wrgb_cycles"] < w["wrgb_cycles"] for w in wr),
            "fully_hidden_loads": sum(w["nominal_hidden_wrgb_cycles"] == w["wrgb_cycles"] for w in wr),
            "old_pim_cycles": before,
            "new_pim_cycles_estimate": None if before is None else before - hidden,
            "pnm_equivalent_cycles": region["backends"]["pnm"]["reference_equivalent_cycles"],
            "cxl_equivalent_cycles": region["backends"]["cxl"]["reference_equivalent_cycles"],
            "old_block_contribution_cycles": old_ns / period_ns,
            "new_block_contribution_cycles_estimate": new_ns / period_ns,
            "old_total_us": old_ns / 1000, "new_total_us_estimate": new_ns / 1000,
        })
    hidden = sum(r["hidden_wrgb_cycles"] for r in rows)
    old = source["full_run_totals"]
    pim = old["pim_native_cycles"] - hidden
    block_ns = old["block_latency_ns"] - hidden * period_ns
    totals = {
        "old_pim_cycles_measured": old["pim_native_cycles"],
        "new_pim_cycles_analytical": pim, "hidden_wrgb_cycles": hidden,
        "wrgb_wire_cycles": sum(r["wrgb_cycles"] for r in rows),
        "old_wrgb_gap_cycles": sum(r["old_wrgb_gap_cycles"] for r in rows),
        "nominal_hidden_wrgb_cycles": sum(r["nominal_hidden_wrgb_cycles"] for r in rows),
        "nominal_exposed_wrgb_cycles": sum(r["nominal_exposed_wrgb_cycles"] for r in rows),
        "uncredited_nominal_overlap_cycles": sum(r["uncredited_nominal_overlap_cycles"] for r in rows),
        "exposed_wrgb_cycles": sum(r["exposed_wrgb_cycles"] for r in rows),
        "old_block_equivalent_cycles": old["reference_equivalent_cycles"],
        "new_block_equivalent_cycles_analytical": block_ns / period_ns,
        "old_pim_us": old["pim_latency_ns"] / 1000, "new_pim_us_analytical": pim * period_ns / 1000,
        "pnm_us_frozen": old["pnm_exposed_latency_ns"] / 1000,
        "cxl_us_frozen": old["cxl_latency_ns"] / 1000,
        "old_block_us": old["block_latency_ns"] / 1000, "new_block_us_analytical": block_ns / 1000,
        "saved_us": hidden * period_ns / 1000,
        "pim_reduction_percent": 100 * hidden / old["pim_native_cycles"],
        "block_reduction_percent": 100 * hidden * period_ns / old["block_latency_ns"],
        "block_speedup": old["block_latency_ns"] / block_ns,
        "per_output_token_amortized_us": block_ns / case["batch_size"] / 1000,
    }
    assert math.isclose(sum(k["new_pim_cycles_estimate"] or 0 for k in kernels), pim)
    assert math.isclose(sum(k["new_total_us_estimate"] for k in kernels), block_ns / 1000)
    assert totals["wrgb_wire_cycles"] == totals["nominal_hidden_wrgb_cycles"] + totals["nominal_exposed_wrgb_cycles"]
    assert totals["old_wrgb_gap_cycles"] == totals["exposed_wrgb_cycles"] + hidden
    for kernel in kernels:
        assert kernel["exposed_wrgb_cycles"] >= 0
        if kernel["new_pim_cycles_estimate"] is not None:
            assert math.isclose(kernel["new_pim_cycles_estimate"],
                                kernel["non_wrgb_pim_cycles_unchanged"] + kernel["exposed_wrgb_cycles"])
    return {"totals": totals, "kernels": kernels, "waves": rows}


ASSUMPTIONS = [
    "Analytical delta estimate, NOT a double-buffer Ramulator run or measured speedup.",
    "Two independent 2 KiB GBs/channel; WR_GB may execute during MAC using the other GB.",
    "Each transfer: L_i = WRGB_bursts_i * nBL(8). Each compute window: C_i = MAC8_count_i * nCCD(2), including fill/drain.",
    "Only the immediately next load can overlap: H_i=min(L_i,C_(i-1)) when its input is ready; otherwise H_i=0. First load is cold.",
    "Main readiness policy: within a phase, same-kernel tiles/waves, Q->KV, and SV->next QK permit prefetch. MEM/SYNC and other producer boundaries stop it.",
    "QK->SV is conservatively treated as a whole-chunk producer/Softmax barrier; no speculative sub-head score readiness is assumed.",
    "The ideal_ready_inputs scenario removes input-readiness/barrier constraints ONLY as an optimistic sensitivity, not an implementable schedule.",
    "Accounting v2: nominal H_i selects the overlapped prefix. Credit each overlapped burst by min(overlapped nominal time, its actual WRGB-owned issue gap). Never subtract nominal time directly from a different issued-gap ledger.",
    "T_new = saved T_old - sum(credited WRGB gaps). Each kernel's remaining WRGB gap is nonnegative. CASWRGB is excluded from the deduction budget; all non-WRGB costs remain fixed.",
    "Two separate identities: nominal wire = nominal hidden + nominal exposed; old WRGB issue gaps = credited hidden + remaining WRGB gaps. The exposed_wrgb_cycles field is the latter ledger remainder, not nominal bus time.",
    "PNM, CXL and legacy Softmax exposed time stay exactly as in each source. This does not recompute overlap, refresh schedule, contention, energy or device throughput.",
    "All source GB writes, Q reloads, array mappings and simplified legacy RD_MAC counts are retained. No Q-residency or proposed same-V dual-SA optimization is added.",
    "Results describe one representative TP rank, one Transformer block, one batch group. Do not multiply latency by 32 channels or by TP.",
    "Raw issued gaps contain waiting and are not independent service times. A cycle-accurate implementation may realize a different saving.",
]


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_summary(result):
    source = result["source"]
    case = source["case"]
    lines = [
        "PIM double buffering — analytical WR_GB overlap estimate (accounting v2)",
        f"Case: {result['label']}; FP8 SA4x32; TP2; batch4; active {case['sequence_length']}; Flash chunk={case['flash_attention_block_size']}.",
        f"Source run: {source['run_id']}; clock={source['reference_clock_period_ps']:g} ps; reference channel={source['estimator']['reference_channel']}.",
        "Scope: one representative rank / one Transformer block / one batch group.",
        "No simulator run. Existing artifacts were read only; all new cycles are analytical estimates.",
        "", "Method and assumptions:", *[f"- {a}" for a in result["assumptions"]],
    ]
    for name, model in result["models"].items():
        t = model["totals"]
        lines += ["", f"Scenario: {name}",
                  f"PIM cycles: {t['old_pim_cycles_measured']:,} -> {t['new_pim_cycles_analytical']:,}; hidden WR_GB={t['hidden_wrgb_cycles']:,}; PIM reduction={t['pim_reduction_percent']:.4f}%.",
                  f"Block equivalent cycles: {t['old_block_equivalent_cycles']:.6f} -> {t['new_block_equivalent_cycles_analytical']:.6f}.",
                  f"Block latency: {t['old_block_us']:.6f} -> {t['new_block_us_analytical']:.6f} us; reduction={t['block_reduction_percent']:.4f}%; speedup={t['block_speedup']:.6f}x.",
                  f"New PIM={t['new_pim_us_analytical']:.6f} us; unchanged PNM={t['pnm_us_frozen']:.6f} us; unchanged CXL={t['cxl_us_frozen']:.6f} us.",
                  f"Nominal WR_GB wire budget={t['wrgb_wire_cycles']:,}; nominal hidden={t['nominal_hidden_wrgb_cycles']:,}; nominal exposed={t['nominal_exposed_wrgb_cycles']:,} cycles.",
                  f"Issued-gap ledger: old WRGB={t['old_wrgb_gap_cycles']:,}; credited hidden={t['hidden_wrgb_cycles']:,}; remaining WRGB={t['exposed_wrgb_cycles']:,}; uncredited nominal overlap={t['uncredited_nominal_overlap_cycles']:,} cycles. CASWRGB is unchanged.",
                  "", "kernel                         loads   old PIM   old WRGB gap  hidden gap  remaining gap   new PIM   old total(us)   new total(us)"]
        for k in model["kernels"]:
            old_pim = "unmodeled" if k["old_pim_cycles"] is None else f"{k['old_pim_cycles']:,.0f}"
            new_pim = "unmodeled" if k["new_pim_cycles_estimate"] is None else f"{k['new_pim_cycles_estimate']:,.0f}"
            lines.append(f"{k['kernel']:<29} {k['gb_load_waves']:>5} {old_pim:>10} {k['old_wrgb_gap_cycles']:>14,} {k['hidden_wrgb_cycles']:>11,} {k['exposed_wrgb_cycles']:>14,} {new_pim:>10} {k['old_total_us']:>15.6f} {k['new_total_us_estimate']:>15.6f}")
        lines += ["", "Remaining WRGB issue-gap reasons (already included in the table):"]
        reasons = Counter()
        for w in model["waves"]:
            reasons[(w["kernel"], w["readiness"])] += w["exposed_wrgb_cycles"]
        lines.extend(f"  {kernel}: {reason}: {cycles:,}" for (kernel, reason), cycles in reasons.items() if cycles)
    return "\n".join(lines) + "\n"


def analyze(source_path, label, output, search_root, cxl_bandwidth_unit="GB/s"):
    source = json.loads(Path(source_path).read_text())
    case = source["case"]
    if (case["model"], case["tp"], case["batch_size"], case["array_height"], case["array_width"],
            case["nBL"], case["nCCD"], case["flash_attention"], case["channel_concurrency"]) != (
            "Llama2-70B", 2, 4, 4, 32, 8, 2, True, True):
        raise ValueError("This analytical readiness policy is scoped to the retained full FP8 SA4x32 TP2 batch4 case")
    artifacts = source["run_identity"]["artifacts"]
    trace = resolve_artifact(artifacts["trace"], search_root)
    metadata_path = resolve_artifact(artifacts["systolic_metadata"], search_root)
    issued_path = resolve_artifact(artifacts["issued_command_trace"],
                                   REPO / "cent_simulation/output/legacy_optimizations")
    metadata = json.loads(metadata_path.read_text())
    waves, counts = decode_waves(trace, metadata, source["estimator"]["reference_channel"])
    validate_counts(source, metadata, counts)
    wave_gaps = issued_wrgb_gaps(waves, issued_path, source["estimator"]["reference_channel"])
    models = {name: model_report(source, waves, ideal, wave_gaps)
              for name, ideal in (("dependency_aware", False), ("ideal_ready_inputs", True))}
    assert models["dependency_aware"]["totals"]["hidden_wrgb_cycles"] <= models["ideal_ready_inputs"]["totals"]["hidden_wrgb_cycles"]
    result = {"method": "analytical_wrgb_issue_gap_credit_v2", "label": label,
              "source": {k: source[k] for k in ("run_id", "case", "reference_clock_period_ps", "estimator", "full_run_totals")},
              "assumptions": list(ASSUMPTIONS),
              "provenance": {"report": describe(source_path), "trace": describe(trace),
                             "metadata": describe(metadata_path), "issued_command_trace": describe(issued_path),
                             "analyzer": describe(__file__),
                             "python_executable": sys.executable, "python_version": sys.version},
              "validation": {"trace_and_metadata_sha256_match_source_manifest": True,
                             "all_kernel_WRGB_MAC_WRMAC_RDMAC_counts_match": True,
                             "all_kernel_MAC_stages_match_metadata": True,
                             "region_totals_and_hidden_exposed_budgets_reconcile": True,
                             "WRGB_gap_totals_match_source_for_every_kernel": True,
                             "all_remaining_WRGB_gap_budgets_nonnegative": True,
                             "CASWRGB_and_all_non_WRGB_PIM_costs_unchanged": True},
              "models": models}
    apply_cxl_units(result, source["run_identity"]["source_row"], metadata["shape"], cxl_bandwidth_unit)
    result["provenance"]["cxl_unit_postprocessor"] = describe(Path(__file__).with_name("legacy_cxl_units.py"))
    result["provenance"]["kernel_csv_renderer"] = describe(Path(__file__).with_name("double_buffer_kernel_table.py"))
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "summary.txt").write_text(render_summary(result))
    for name, model in models.items():
        write_double_buffer_kernel_table(result, name, output / f"{name}_kernels.csv", source)
        write_csv(output / f"{name}_waves.csv", model["waves"])
    return result


def plot_results(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 4.5), squeeze=False)
    for axis, result in zip(axes[0], results):
        rows = [r for r in result["models"]["dependency_aware"]["kernels"] if r["gb_load_waves"]]
        positions = list(range(len(rows)))
        axis.barh([p - .18 for p in positions], [r["old_pim_cycles"] for r in rows], height=.36, label="Single buffer (saved attribution)")
        axis.barh([p + .18 for p in positions], [r["new_pim_cycles_estimate"] for r in rows], height=.36, label="Double buffer (analytical)")
        axis.set_yticks(positions, [r["kernel"] for r in rows])
        axis.invert_yaxis()
        axis.set_title(f"Flash chunk {result['source']['case']['flash_attention_block_size']}")
        axis.set_xlabel("PIM clock cycles (0.625 ns/cycle)")
        axis.legend(fontsize=8)
        axis.grid(axis="x", alpha=.2)
    fig.suptitle("Dependency-aware WR_GB overlap estimate — not a new simulation")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "full/breakdown.json")
    parser.add_argument("--supplement-source", type=Path, default=ROOT / "full_flash1024/breakdown.json")
    parser.add_argument("--skip-supplement", action="store_true")
    parser.add_argument("--trace-search-root", type=Path, default=REPO / "cent_simulation/trace/legacy_optimizations")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "double buffering")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--cxl-bandwidth-unit", choices=("GB/s", "GiB/s"), default="GB/s",
                        help="GB/s uses decimal bandwidth; GiB/s reproduces the saved legacy CXL units")
    args = parser.parse_args(argv)
    source_paths = [args.source]
    if not args.skip_supplement:
        source_paths.append(args.supplement_source)
    sources = [(f"full_flash{json.loads(path.read_text())['case']['flash_attention_block_size']}", path)
               for path in source_paths]
    if len({label for label, _ in sources}) != len(sources):
        parser.error("Primary and supplement must use different FlashAttention chunk sizes")
    results = [analyze(source, label, args.output_dir / label, args.trace_search_root, args.cxl_bandwidth_unit)
               for label, source in sources]
    comparisons = [{"case": r["label"], "scenario": name, **m["totals"]}
                   for r in results for name, m in r["models"].items()]
    write_csv(args.output_dir / "comparison.csv", comparisons)
    notes = ["PIM double buffering — analytical results index (accounting v2)", "",
             "主结果采用 dependency_aware：按数据依赖决定是否预取，先求名义传输重叠，再按实际 WRGB issue-gap 归属逐 burst 扣减。所有新周期都是 analytical estimate，未重跑 Ramulator。",
             "定义：一个 GB load wave 是一条 host WR_GB 展开的全部 bursts，以及随后使用该 GB 的 MAC 工作。不是一个 query-head wave，也不是整个 kernel。",
             "公式：L_i = bursts_i × 8；C_i = MAC8_count_i × 2；若下一输入已就绪，hidden_i = min(L_i, C_(i-1))；首轮或数据未就绪时 hidden_i = 0。",
             "账本扣减：对重叠前缀中的每个 burst，credit=min(该 burst 被覆盖的名义时间, 原 WRGB issue gap)。CASWRGB 不参与 credit；不得从 CAS/控制项借用周期。",
             "修复原因：KV 原 WRGB gap=1,017，但 v1 扣了名义 1,024，按原分项展开会得到 -7；v2 最多扣 1,017，剩余 WRGB=0，CASWRGB=32 保留。",
             "名义传输层面：Q/WO/Up-Gate/Down 各保留 512 cycles 首轮；KV 首轮可预取到 Q 末尾。账本实际可扣减量另列。",
             "名义传输层面：QK 在 3 个 batch 边界有短尾，未覆盖量=128+3×84=380 cycles；SV 因整块 QK/Softmax 依赖保留 15,488 cycles。这些数不直接减原始分项账本。",
             "两套守恒分别校验：名义 wire=名义 hidden+名义 exposed；原 WRGB gap=credited hidden+剩余 WRGB gap。表里的 hidden/remaining 使用第二套口径，均不得为负。",
             "每 kernel 的新 PIM 周期 = 原 PIM 周期 − credited hidden。所有非 WRGB 的 PIM 开销（含 CAS）、PNM 和原 Softmax exposed 公式均保留。",
             f"CXL 带宽单位：{args.cxl_bandwidth_unit}；单缓冲/双缓冲列统一采用该单位，保留 multicast 50% 折扣、180 ns 固定开销及原 PP handoff 计费。原始 source 文件未修改。",
             "",
             f"Primary: {results[0]['label']}; source={args.source.resolve()}.",
             *[f"Supplement: {result['label']}; source={result['provenance']['report']['path']}." for result in results[1:]],
             *[f"{result['label']} source PNM residual arithmetic: "
               f"{sum(k['old_total_us'] for k in result['models']['dependency_aware']['kernels'] if k['region_id'] in ('aux.attention_residual', 'aux.ffn_residual')):.6f} us; retained unchanged."
               for result in results],
             "Main result = dependency_aware. ideal_ready_inputs is an optimistic input-readiness sensitivity only.",
             "Command counts were reconciled; original reports and simulator outputs were not changed.",
             "", "case / scenario                 old PIM    new PIM    hidden WRGB   old block(us)  new block(us)  reduction"]
    for r in comparisons:
        notes.append(f"{r['case']}/{r['scenario']}: {r['old_pim_cycles_measured']:,} -> {r['new_pim_cycles_analytical']:,}; hidden={r['hidden_wrgb_cycles']:,}; block {r['old_block_us']:.6f} -> {r['new_block_us_analytical']:.6f} us ({r['block_reduction_percent']:.4f}%).")
    correction_path = args.output_dir / "wrgb_accounting_correction_reference.json"
    if correction_path.is_file():
        superseded = json.loads(correction_path.read_text())["cases"]
        corrections = []
        notes += ["", "WRGB 口径修正：以下 v1 数值已作废，仅保留用于审计；当前结果使用 v2。"]
        for r in comparisons:
            if r["case"] not in superseded:
                continue
            old = superseded[r["case"]]["models"][r["scenario"]]["totals"]
            corrections.append({"case": r["case"], "scenario": r["scenario"],
                                "superseded_v1_pim_cycles": old["new_pim_cycles_analytical"],
                                "corrected_v2_pim_cycles": r["new_pim_cycles_analytical"],
                                "restored_cycles": r["new_pim_cycles_analytical"] - old["new_pim_cycles_analytical"],
                                "superseded_v1_block_us": old["new_block_us_analytical"],
                                "corrected_v2_block_us": r["new_block_us_analytical"]})
            notes.append(f"{r['case']}/{r['scenario']}: PIM {old['new_pim_cycles_analytical']:,} -> {r['new_pim_cycles_analytical']:,}; block {old['new_block_us_analytical']:.6f} -> {r['new_block_us_analytical']:.6f} us.")
        if corrections:
            write_csv(args.output_dir / "wrgb_accounting_correction.csv", corrections)
    previous_path = args.output_dir / "previous_location_reference.json"
    if previous_path.is_file():
        previous = json.loads(previous_path.read_text())["cases"]
        migration = []
        notes += ["", "与迁移前同 chunk 结果的比较（previous_location_reference.json 保存旧数值）："]
        for r in comparisons:
            if r["case"] not in previous:
                continue
            old = previous[r["case"]]["model_totals"][r["scenario"]]
            delta = r["new_block_us_analytical"] - old["new_block_us_analytical"]
            migration.append({"case": r["case"], "scenario": r["scenario"],
                              "previous_new_pim_cycles": old["new_pim_cycles_analytical"],
                              "current_new_pim_cycles": r["new_pim_cycles_analytical"],
                              "previous_new_block_us": old["new_block_us_analytical"],
                              "current_new_block_us": r["new_block_us_analytical"],
                              "delta_block_us": delta,
                              "delta_pnm_us": r["pnm_us_frozen"] - old["pnm_us_frozen"]})
            notes.append(f"{r['case']}/{r['scenario']}: PIM {old['new_pim_cycles_analytical']:,} -> {r['new_pim_cycles_analytical']:,}; block {old['new_block_us_analytical']:.6f} -> {r['new_block_us_analytical']:.6f} us; delta={delta:+.6f} us.")
        if migration:
            write_csv(args.output_dir / "migration_comparison.csv", migration)
    notes += ["", "Reproduce from repository root:",
              f"  python cent_simulation/scripts/analyze_legacy_double_buffering.py --cxl-bandwidth-unit {args.cxl_bandwidth_unit} --plot", "",
              "Files per case: summary.txt, analysis.json, dependency_aware_kernels.csv, dependency_aware_waves.csv, ideal_ready_inputs_kernels.csv, ideal_ready_inputs_waves.csv.",
              "Kernel CSV: two header rows, 19 standard cycle-breakdown columns, and Block total. Zero-cost rows are omitted. WRGB is the remaining issued-gap budget; PNM native cycles use the source PNM clock. Detailed overlap fields remain in analysis.json.",
              "Detailed method and assumptions:", *[f"- {a}" for a in results[0]["assumptions"]]]
    (args.output_dir / "summary.txt").write_text("\n".join(notes) + "\n")
    if args.plot:
        plot_results(results, args.output_dir / "kernel_cycles.png")
    for r in comparisons:
        print(f"{r['case']}/{r['scenario']}: PIM {r['old_pim_cycles_measured']:,} -> {r['new_pim_cycles_analytical']:,}; block {r['old_block_us']:.6f} -> {r['new_block_us_analytical']:.6f} us.")
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
