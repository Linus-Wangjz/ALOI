"""Explicit bandwidth-unit conversion for derived legacy CXL reports.

Retains the legacy 180 ns transfer overhead, multicast bandwidth discount,
BF16 wire payload, flit packing, and per-block PP handoff accounting.
"""

import math


def cxl_costs_us(hidden_dim, batch, tp, pp, lanes, bytes_per_bandwidth_unit):
    flits = math.ceil(hidden_dim * batch * 2 / 192)
    wire_us = flits * 256 / (lanes * 8 * 0.5 * bytes_per_bandwidth_unit) * 1e6
    vector_us = 0.180 + wire_us
    reduction_us = (0.180 + (tp - 1) * wire_us + vector_us) if tp > 1 else 0.0
    return {
        "attention.o_projection": reduction_us,
        "ffn.down_projection": reduction_us,
        "aux.pp_handoff": vector_us if pp > 1 else 0.0,
    }


def apply_cxl_units(result, source_row, shape, unit):
    """Rebase both single/double-buffer comparisons; preserve raw provenance."""
    if "cxl_bandwidth_model" in result:
        raise ValueError("CXL bandwidth units already applied")
    if unit not in ("GB/s", "GiB/s"):
        raise ValueError(f"Unknown bandwidth unit: {unit}")
    case = result["source"]["case"]
    # This analytical report is scoped to the retained --PCIE_lanes 144 case.
    devices = int(source_row["Device number"])
    lanes = 144 // devices
    old_costs = cxl_costs_us(shape["dim"], case["batch_size"], case["tp"], case["pp"], lanes, 2**30)
    scale = 10**9 if unit == "GB/s" else 2**30
    new_costs = cxl_costs_us(shape["dim"], case["batch_size"], case["tp"], case["pp"], lanes, scale)
    old_us, new_us = sum(old_costs.values()), sum(new_costs.values())
    period_us = result["source"]["reference_clock_period_ps"] / 1e6
    delta_us = new_us - old_us
    for model in result["models"].values():
        totals = model["totals"]
        if not math.isclose(totals["cxl_us_frozen"], old_us, abs_tol=1e-9):
            raise ValueError("Legacy CXL parameters do not reproduce saved total")
        for row in model["kernels"]:
            old_row_us = row["cxl_equivalent_cycles"] * period_us
            expected = old_costs.get(row["region_id"], 0.0)
            if not math.isclose(old_row_us, expected, abs_tol=1e-9):
                raise ValueError(f"Legacy CXL allocation differs: {row['region_id']}")
            if row["region_id"] not in new_costs or scale == 2**30:
                continue
            new_row_us = new_costs[row["region_id"]]
            delta = new_row_us - old_row_us
            row["cxl_equivalent_cycles"] = new_row_us / period_us
            for key in ("old_total_us", "new_total_us_estimate"):
                row[key] += delta
            for key in ("old_block_contribution_cycles", "new_block_contribution_cycles_estimate"):
                row[key] += delta / period_us
        if scale != 2**30:
            totals["cxl_us_frozen"] = new_us
            for key in ("old_block_us", "new_block_us_analytical"):
                totals[key] += delta_us
            totals["old_block_equivalent_cycles"] = totals["old_block_us"] / period_us
            totals["new_block_equivalent_cycles_analytical"] = totals["new_block_us_analytical"] / period_us
            totals["block_reduction_percent"] = 100 * totals["saved_us"] / totals["old_block_us"]
            totals["block_speedup"] = totals["old_block_us"] / totals["new_block_us_analytical"]
            totals["per_output_token_amortized_us"] = totals["new_block_us_analytical"] / case["batch_size"]
        for row_key, total_key in (("old_total_us", "old_block_us"),
                                   ("new_total_us_estimate", "new_block_us_analytical")):
            if not math.isclose(sum(r[row_key] for r in model["kernels"]), totals[total_key], abs_tol=1e-9):
                raise ValueError(f"Adjusted CXL rows do not reconcile: {total_key}")
    result["cxl_bandwidth_model"] = {
        "unit": unit, "bytes_per_bandwidth_unit": scale,
        "total_pcie_lanes": 144, "devices": devices, "lanes_per_device": lanes,
        "raw_bandwidth_per_lane": 8, "multicast_bandwidth_factor": 0.5,
        "effective_bandwidth_per_device": lanes * 4,
        "fixed_transfer_overhead_ns": 180, "wire_bytes_per_value": 2,
        "payload_bytes_per_flit": 192, "wire_bytes_per_flit": 256,
        "original_source_cxl_us": old_us, "selected_cxl_us": new_us,
        "block_delta_us_from_original_source": delta_us,
        "original_region_cxl_us": old_costs, "selected_region_cxl_us": new_costs,
        "pp_handoff_policy": "legacy_one_transfer_per_block_when_pp_gt_1",
        "scope": "Derived reports only; original simulation sources remain immutable.",
    }
    result["validation"]["legacy_cxl_formula_reproduces_source"] = True
    result["validation"]["cxl_unit_adjustment_applied_to_both_buffer_baselines"] = True
    note = (f"CXL bandwidth: {lanes} lanes/device x 8 x 0.5 multicast discount = {lanes * 4} {unit}; "
            f"original CXL={old_us:.9f} us, selected CXL={new_us:.9f} us. "
            "Both single/double-buffer columns use the selected units. The 180 ns overhead, BF16 wire payload, "
            "flit packing and per-block PP handoff remain unchanged; source/provenance retain original measurements.")
    result["assumptions"] = [
        ("PNM and legacy Softmax exposed time stay as in the source. CXL stays fixed across the "
         "single/double-buffer comparison at the explicitly selected bandwidth units.")
        if a.startswith("PNM, CXL and legacy Softmax") else a
        for a in result["assumptions"]
    ] + [note]
    return result
