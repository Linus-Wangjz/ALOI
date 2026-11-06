#!/usr/bin/env python3
"""Convert saved double-buffer block estimates to legacy system throughput.

Uses run_sim.py's full-pipeline, decode-only throughput convention. It does
not rerun Ramulator or change the WR_GB overlap, PNM, or CXL latency models.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path

from analyze_legacy_double_buffering import REPO, ROOT, describe, digest, write_csv


def close(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-9):
        raise ValueError(f"{label}: {actual} != {expected}")


def model_layers(model):
    # Read the same model table as run_sim without importing torch or pandas.
    tree = ast.parse((REPO / "cent_simulation/utils.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "TransformerBlock_number"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)[model]
    raise ValueError("TransformerBlock_number table missing")


def verified_json(record):
    path = Path(record["path"])
    if not path.is_absolute():
        path = REPO / path
    if digest(path) != record["sha256"]:
        raise ValueError(f"Source changed since analytical estimate: {path}")
    return json.loads(path.read_text())


def throughput(block_us, layers, batch, pp, devices):
    token_ms = block_us * layers / 1000.0
    system = 1000.0 * pp * batch / token_ms
    return {
        "block_us": block_us,
        "stage_service_ms": token_ms / pp,
        "token_latency_ms": token_ms,
        "system_tokens_per_s": system,
        "tokens_per_s_per_device": system / devices,
    }


def process_case(path, separated_path):
    analysis = json.loads(path.read_text())
    source = verified_json(analysis["provenance"]["report"])
    metadata = verified_json(analysis["provenance"]["metadata"])
    row, case = source["run_identity"]["source_row"], source["case"]
    layers = model_layers(case["model"])
    devices, pp, tp, batch = (int(row[key]) for key in (
        "Device number", "Pipeline parallelism", "Tensor parallelism", "Batch size"))
    if devices != pp * tp or layers % pp:
        raise ValueError("Expected balanced PP stages and one TP group per stage")
    for key, value in (("pp", pp), ("tp", tp), ("batch_size", batch)):
        close(case[key], value, f"source {key}")
    if analysis["source"]["case"] != case:
        raise ValueError("Analytical and original cases differ")
    # These saved cases are decode-only; reject embedding/host overhead instead
    # of silently omitting it from a different workload.
    close(float(row["Embedding latency"]), 0, "decode-only embedding")
    original_base = throughput(float(row["TransformerBlock latency"]) * 1000,
                               layers, batch, pp, devices)
    close(original_base["token_latency_ms"], float(row["Token latency (ms)"]), "source token latency")
    close(original_base["system_tokens_per_s"], float(row["Throughput (tokens/s)"]), "source throughput")
    cxl = analysis.get("cxl_bandwidth_model", {})
    delta_us = cxl.get("block_delta_us_from_original_source", 0.0)
    base = throughput(original_base["block_us"] + delta_us, layers, batch, pp, devices)
    separated = json.loads(separated_path.read_text())
    if separated["source"] != analysis["source"]:
        raise ValueError("CXL-separated view has a different source")

    config = {
        **case, "devices": devices, "layers": layers,
        "layers_per_pipeline_stage": layers // pp,
        "batch_size_per_microbatch": batch,
        "full_pipeline_microbatches": pp,
        "full_pipeline_sequences": pp * batch,
        "channels_per_device": int(row["Channels per device"]),
        # The legacy CSV column is misnamed; --num_banks is per channel.
        "banks_per_channel": int(row["Banks per device"]),
        "pim_clock_period_ps": source["reference_clock_period_ps"],
        "parallel_sram_banks": int(row["Parallel SRAM banks"]),
        "pnm_shared_buffer_bytes": int(row["Shared buffer capacity (bytes)"]),
        "double_buffer_count_per_channel": 2,
        "double_buffer_bytes_each": 2048,
        "attention_mapping": row["Attention mapping"],
        "kv_cache_mapping": row["KV cache mapping"],
        "model_shape": metadata["shape"],
        "decode_only": True,
        "cxl_bandwidth_model": cxl,
    }
    common = {
        "case": analysis["label"], "model": case["model"], "devices": devices,
        "tp": tp, "pp": pp, "batch_size": batch, "layers": layers,
        "sequence_length": case["sequence_length"],
        "flash_chunk_size": case["flash_attention_block_size"],
    }
    rows = [{**common, "scenario": "single_buffer", "estimate": bool(delta_us),
             "pim_cycles": source["full_run_totals"]["pim_native_cycles"],
             **base, "throughput_gain_percent": 0.0, "latency_reduction_percent": 0.0}]
    for name, model in analysis["models"].items():
        totals = model["totals"]
        if totals != separated["models"][name]["totals"]:
            raise ValueError(f"CXL split changes totals: {name}")
        close(totals["old_block_us"], base["block_us"], f"{name} baseline")
        new_us = totals["new_block_us_analytical"]
        close(new_us, totals["new_pim_us_analytical"] + totals["pnm_us_frozen"]
              + totals["cxl_us_frozen"], f"{name} backend sum")
        close(new_us, sum(k["new_total_us_estimate"] for k in
                         separated["models"][name]["kernels"]), f"{name} CXL split sum")
        result = throughput(new_us, layers, batch, pp, devices)
        # Independent steady-state form: one microbatch per stage service time,
        # then normalize by TP * PP physical devices.
        close(result["tokens_per_s_per_device"],
              batch * 1e6 / (layers * new_us * tp), f"{name} device normalization")
        rows.append({**common, "scenario": name, "estimate": True,
                     "pim_cycles": totals["new_pim_cycles_analytical"], **result,
                     "throughput_gain_percent": (result["system_tokens_per_s"]
                                                  / base["system_tokens_per_s"] - 1) * 100,
                     "latency_reduction_percent": (1 - new_us / base["block_us"]) * 100})
    return {
        "label": analysis["label"], "configuration": config, "results": rows,
        "original_source_single_buffer": original_base,
        "provenance": {"analysis": describe(path), "cxl_separated_analysis": describe(separated_path),
                       "original_report": analysis["provenance"]["report"],
                       "metadata": analysis["provenance"]["metadata"]},
        "validation": {"source_throughput_reproduced": True,
                       "source_token_latency_reproduced": True,
                       "cxl_separation_preserves_totals": True,
                       "device_normalization_checked": True},
    }


def render(cases):
    main = cases[0]
    c = main["configuration"]
    base, new = main["results"][:2]
    lines = [
        "Double buffering — system throughput postprocessing", "",
        f"Primary case: {main['label']}; scenario: {new['scenario']}",
        f"Saved model label: {c['model']}; {c['layers']} Transformer blocks.",
        "The source label is Llama2-70B; this does not constitute a separate Llama3.1 run.",
        f"System: {c['devices']} devices = TP{c['tp']} x PP{c['pp']}; "
        f"{c['layers_per_pipeline_stage']} blocks per PP stage, sharded over {c['tp']} devices.",
        f"Batch: {c['batch_size_per_microbatch']} per microbatch; full pipeline = "
        f"{c['full_pipeline_microbatches']} microbatches / {c['full_pipeline_sequences']} sequences.",
        f"PIM: {c['precision']}, {c['array_height']}x{c['array_width']} SA; "
        f"{c['channels_per_device']} channels/device, {c['banks_per_channel']} banks/channel.",
        f"DRAM: {c['dram_timing_impl']}, x{c['dram_organization']['channel_width_bits']}/channel; "
        f"{c['dram_organization']['transaction_bytes']} B/burst; nBL={c['nBL']}, "
        f"nCCD={c['nCCD']}, tCK={c['pim_clock_period_ps']:g} ps.",
        "Double buffering: two independent 2 KiB GBs/channel; existing dependency-aware WR_GB credits retained.",
        f"PNM: {c['parallel_sram_banks']} SRAM banks, "
        f"{c['pnm_shared_buffer_bytes'] // 1048576} MiB shared buffer; EWMUL/residual PNM enabled in source.",
        f"Context window: {c['context_window']}; evaluated decode sequence length: {c['sequence_length']}.",
        f"Flash chunk: {c['flash_attention_block_size']}; pipelined softmax and channel concurrency enabled.",
        "CXL: O/Down TP collectives and PP handoff included; separation changes presentation only.",
        (f"CXL bandwidth unit: {c['cxl_bandwidth_model'].get('unit', 'GiB/s')}; "
         "single/double-buffer comparison uses the same units; original source values are preserved in analysis.json."),
        "", "Formula (same as run_sim.load_data_point):",
        "  token_latency_ms = block_us * layers / 1000  (decode-only)",
        "  system_tokens_per_s = 1000 * PP * batch / token_latency_ms",
        "  tokens_per_s_per_device = system_tokens_per_s / (TP * PP)",
        f"  = 1e6 * {c['batch_size']} / ({c['layers']} * {new['block_us']:.12f} * {c['tp']})",
        f"  = {new['tokens_per_s_per_device']:.9f} tokens/s/device", "",
        f"Primary result: {base['tokens_per_s_per_device']:.6f} -> "
        f"{new['tokens_per_s_per_device']:.6f} tokens/s/device "
        f"(+{new['throughput_gain_percent']:.6f}%).",
        f"Block latency: {base['block_us']:.6f} -> {new['block_us']:.6f} us "
        f"(-{new['latency_reduction_percent']:.6f}%).", "",
        "case             scenario             block(us) token(ms) system(tok/s) device(tok/s) gain(%)",
    ]
    for item in cases:
        for r in item["results"]:
            lines.append(f"{r['case']:<16} {r['scenario']:<20} {r['block_us']:>10.6f} "
                         f"{r['token_latency_ms']:>9.6f} {r['system_tokens_per_s']:>13.6f} "
                         f"{r['tokens_per_s_per_device']:>13.6f} {r['throughput_gain_percent']:>8.4f}")
    lines += ["", "Scope and validation:",
              "- Throughput assumes steady-state full PP occupancy. Batch=4 is per microbatch, not four sequences total.",
              "- With only one batch=4 group in flight, divide the reported system/device throughput by PP=16.",
              "- Decode Transformer blocks only: embedding, output head, CPU sampling and pipeline fill/drain are excluded.",
              "- This is the sequence-length=30976 point, not a context-length average.",
              "- PNM/softmax exposure stay at source values. CXL uses the selected bandwidth units, retaining the existing per-block PP handoff charge.",
              "- dependency_aware is the main analytical result; ideal_ready_inputs is an optimistic readiness sensitivity.",
              "- Original source throughput/latency reproduced; CXL-separated totals and device normalization checked.",
              "- No new Ramulator run, power estimate, or energy estimate is implied."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "double buffering")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    paths = sorted(args.source_root.glob("full_flash*/analysis.json"),
                   key=lambda p: json.loads(p.read_text())["source"]["case"]["flash_attention_block_size"],
                   reverse=True)
    if not paths:
        parser.error("No saved double-buffer analysis found")
    cases = [process_case(p, args.source_root / "cxl_separated" / p.parent.name / p.name) for p in paths]
    output = args.output_dir or args.source_root / "throughput"
    output.mkdir(parents=True, exist_ok=True)
    report = {"method": "legacy_decode_full_pipeline_throughput_v1", "cases": cases,
              "provenance": {"postprocessor": describe(__file__),
                             "throughput_formula": describe(REPO / "cent_simulation/run_sim.py"),
                             "model_layers": describe(REPO / "cent_simulation/utils.py")}}
    (output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    write_csv(output / "comparison.csv", [r for c in cases for r in c["results"]])
    (output / "summary.txt").write_text(render(cases))
    print(render(cases))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
