"""Render derived double-buffer kernels in the standard 19-column table."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cycle_breakdown import write_kernel_table


def check(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-8):
        raise ValueError(f"{label}: {actual} != {expected}")


def write_double_buffer_kernel_table(result, scenario, path, original=None):
    if original is None:
        record = result["provenance"]["report"]
        source_path = Path(record["path"])
        if not source_path.is_absolute():
            source_path = Path(__file__).resolve().parents[2] / source_path
        content = source_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise ValueError(f"Original command breakdown changed: {source_path}")
        original = json.loads(content)
    source_regions = {r["region_id"]: r for r in original["regions"]}
    model = result["models"][scenario]
    period_ps = result["source"]["reference_clock_period_ps"]
    period_ns = period_ps / 1000.0
    regions = []
    for kernel in model["kernels"]:
        # Omit empty/unsupported rows rather than presenting unmodeled SYNC as
        # a measured zero. Nonzero costs remain fully reconciled below.
        if kernel["new_total_us_estimate"] == 0:
            continue
        region_id = kernel["region_id"]
        source = source_regions.get(region_id)
        if source is None and not region_id.startswith("communication."):
            raise ValueError(f"Missing original command breakdown: {region_id}")
        details = (source or {}).get("pim_issued_gap_details") or {"commands": []}
        commands = [{"command": c["command"], "gap_cycles": c["gap_cycles"]}
                    for c in details["commands"]]
        wrgb = sum(c["gap_cycles"] for c in commands if c["command"] == "WRGB")
        check(wrgb, kernel["old_wrgb_gap_cycles"], f"{region_id} source WRGB")
        remaining_credit = kernel["hidden_wrgb_cycles"]
        for command in commands:
            if command["command"] == "WRGB":
                credit = min(remaining_credit, command["gap_cycles"])
                command["gap_cycles"] -= credit
                remaining_credit -= credit
        check(remaining_credit, 0, f"{region_id} WRGB credit")
        check(sum(c["gap_cycles"] for c in commands if c["command"] == "WRGB"),
              kernel["exposed_wrgb_cycles"], f"{region_id} remaining WRGB")
        pim = kernel["new_pim_cycles_estimate"]
        if pim is None:
            raise ValueError(f"Nonzero kernel has unavailable PIM contribution: {region_id}")
        check(sum(c["gap_cycles"] for c in commands), pim, f"{region_id} PIM buckets")
        pnm_eq = kernel["pnm_equivalent_cycles"]
        if source:
            pnm = copy.deepcopy(source["backends"]["pnm"])
            check(pnm["reference_equivalent_cycles"], pnm_eq, f"{region_id} PNM preserved")
            if pnm_eq:
                check(pnm["formula_cycles"] * pnm["clock_period_ps"] / period_ps,
                      pnm_eq, f"{region_id} PNM native conversion")
        else:
            pnm = {"formula_cycles": 0, "reference_equivalent_cycles": 0}
            check(pnm_eq, 0, f"{region_id} communication-only PNM")
        equivalent = kernel["new_block_contribution_cycles_estimate"]
        check(pim + pnm_eq + kernel["cxl_equivalent_cycles"], equivalent,
              f"{region_id} backend sum")
        check(equivalent * period_ns / 1000, kernel["new_total_us_estimate"],
              f"{region_id} latency")
        regions.append({
            "region_id": region_id, "display_name": kernel["kernel"],
            "pim_issued_gap_details": {"commands": commands},
            "backends": {
                "pim": {"reference_equivalent_cycles": pim, "duration_ns": pim * period_ns},
                "pnm": pnm,
                "cxl": {"reference_equivalent_cycles": kernel["cxl_equivalent_cycles"]},
            },
            "reference_equivalent_cycles": equivalent,
            "legacy_ledger_duration_ns": kernel["new_total_us_estimate"] * 1000,
        })
    totals = model["totals"]
    check(sum(r["legacy_ledger_duration_ns"] for r in regions),
          totals["new_block_us_analytical"] * 1000, "block latency")
    for backend, expected in (("pim", totals["new_pim_cycles_analytical"]),
                              ("pnm", totals["pnm_us_frozen"] * 1000 / period_ns),
                              ("cxl", totals["cxl_us_frozen"] * 1000 / period_ns)):
        check(sum(r["backends"][backend]["reference_equivalent_cycles"] for r in regions),
              expected, f"{backend} block total")
    report = {
        "regions": regions, "reference_clock_period_ps": period_ps,
        "full_run_totals": {
            "block_latency_ns": totals["new_block_us_analytical"] * 1000,
            "pim_latency_ns": totals["new_pim_us_analytical"] * 1000,
            "reference_equivalent_cycles": totals["new_block_equivalent_cycles_analytical"],
        },
    }
    write_kernel_table({f"{result['label']}_{scenario}": report}, path)
