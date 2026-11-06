#!/usr/bin/env python3
"""Create a separate CXL view of existing analytical double-buffer results.

Move O/Down TP-collective costs into separate rows without changing any
compute work, communication model, overlap, or block total.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from analyze_legacy_double_buffering import ROOT, describe, digest, write_csv
from double_buffer_kernel_table import write_double_buffer_kernel_table


SPLITS = {
    "attention.o_projection": (
        "communication.o_projection_tp_all_reduce", "CXL TP All-Reduce (after O)"),
    "ffn.down_projection": (
        "communication.down_projection_tp_all_reduce", "CXL TP All-Reduce (after Down)"),
}


def close(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-8):
        raise ValueError(f"{label}: {actual} != {expected}")


def split_model(model, period_us):
    rows, seen = [], set()
    for original in model["kernels"]:
        row = dict(original)
        if row["region_id"] not in SPLITS:
            rows.append(row)
            continue
        seen.add(row["region_id"])
        cycles = row["cxl_equivalent_cycles"]
        if cycles < 0:
            raise ValueError("CXL contribution must be nonnegative")
        duration_us = cycles * period_us
        row["cxl_equivalent_cycles"] = 0.0
        row["kernel"] += " (compute)"
        for key in ("old_block_contribution_cycles", "new_block_contribution_cycles_estimate"):
            row[key] -= cycles
        for key in ("old_total_us", "new_total_us_estimate"):
            row[key] -= duration_us
        # The source schema has two string identity fields; all other fields
        # are numeric. A communication-only row has no PIM work or GB traffic.
        communication = {key: 0 for key in original}
        region_id, name = SPLITS[original["region_id"]]
        communication.update({
            "region_id": region_id, "kernel": name,
            "cxl_equivalent_cycles": cycles,
            "old_block_contribution_cycles": cycles,
            "new_block_contribution_cycles_estimate": cycles,
            "old_total_us": duration_us, "new_total_us_estimate": duration_us,
        })
        rows.extend((row, communication))
        for key in ("old_total_us", "new_total_us_estimate",
                    "old_block_contribution_cycles", "new_block_contribution_cycles_estimate"):
            close(row[key] + communication[key], original[key], f"{original['region_id']} {key}")
    if seen != set(SPLITS):
        raise ValueError("Expected one O Projection and one Down Projection to split")
    if len({r["region_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate communication/compute region IDs")
    for key in model["kernels"][0]:
        if key in ("region_id", "kernel"):
            continue
        close(sum(r[key] or 0 for r in rows), sum(r[key] or 0 for r in model["kernels"]),
              f"conservation of {key}")
    totals = model["totals"]
    for key, total_key in (("old_total_us", "old_block_us"),
                           ("new_total_us_estimate", "new_block_us_analytical"),
                           ("old_pim_cycles", "old_pim_cycles_measured"),
                           ("new_pim_cycles_estimate", "new_pim_cycles_analytical")):
        close(sum(r[key] or 0 for r in rows), totals[total_key], total_key)
    close(sum(r["cxl_equivalent_cycles"] for r in rows) * period_us,
          totals["cxl_us_frozen"], "total CXL")
    return {**copy.deepcopy(model), "kernels": rows}


def make_view(source, source_path):
    result = copy.deepcopy(source)
    period_us = source["source"]["reference_clock_period_ps"] / 1e6
    result["models"] = {name: split_model(model, period_us)
                        for name, model in source["models"].items()}
    result["presentation"] = {
        "id": "separate_projection_cxl_v1", "source_analysis": describe(source_path),
        "postprocessor": describe(__file__),
        "kernel_csv_renderer": describe(Path(__file__).with_name("double_buffer_kernel_table.py")),
        "splits": {key: {"region_id": value[0], "name": value[1]} for key, value in SPLITS.items()},
        "semantics": "Move CXL from O/Down totals into separate communication rows. PP Handoff stays separate. All totals and wave schedules are unchanged.",
    }
    result["validation"]["cxl_split_all_numeric_column_sums_preserved"] = True
    result["validation"]["cxl_split_block_and_backend_totals_preserved"] = True
    return result


def render(result):
    lines = ["Double buffering — CXL separated view", "",
             f"Case: {result['label']}; 原结果: {result['presentation']['source_analysis']['path']}",
             "范围：一个代表 TP rank、一个 Transformer block、一个 batch=4 group。",
             "O Projection 和 Down Projection 行只保留 PIM/PNM；各自的 TP CXL 通信紧随其后单列。",
             "PP Handoff 原本已独立，保持其归属。这里只调整展示归属，没有新增、删除或隐藏通信时间。",
             "原始单缓冲和双缓冲 analytical 的总周期、总延迟、GB waves 及 overlap 均保持不变。",
             "所有等效周期按 0.625 ns/cycle；CXL 的等效周期不是 CXL 原生时钟周期。"]
    for name, model in result["models"].items():
        t = model["totals"]
        lines += ["", f"Scenario: {name}",
                  f"PIM cycles: {t['old_pim_cycles_measured']:,} -> {t['new_pim_cycles_analytical']:,}",
                  f"Block latency: {t['old_block_us']:.6f} -> {t['new_block_us_analytical']:.6f} us (unchanged by this split)",
                  f"New block equivalent cycles: {t['new_block_equivalent_cycles_analytical']:.6f}",
                  f"New PIM={t['new_pim_us_analytical']:.6f} us; PNM={t['pnm_us_frozen']:.6f} us; CXL={t['cxl_us_frozen']:.6f} us.",
                  "", "component                              old total(us)  new PIM cycles  PNM eq cycles  CXL eq cycles  new total(us)"]
        for row in model["kernels"]:
            pim = "unmodeled" if row["new_pim_cycles_estimate"] is None else f"{row['new_pim_cycles_estimate']:,.0f}"
            lines.append(f"{row['kernel']:<38} {row['old_total_us']:>13.6f} {pim:>15} {row['pnm_equivalent_cycles']:>14.6f} {row['cxl_equivalent_cycles']:>14.6f} {row['new_total_us_estimate']:>14.6f}")
        lines += [f"{'TOTAL':<38} {t['old_block_us']:>13.6f} {t['new_pim_cycles_analytical']:>15,} "
                  f"{t['pnm_us_frozen'] / .000625:>14.6f} {t['cxl_us_frozen'] / .000625:>14.6f} {t['new_block_us_analytical']:>14.6f}"]
    lines += ["", "Original analytical assumptions (unchanged):",
              *[f"- {a}" for a in result["assumptions"]]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "double buffering")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    output = args.output_dir or args.source_root / "cxl_separated"
    if output.resolve() == args.source_root.resolve():
        parser.error("Use a separate output directory to preserve the original view")
    paths = sorted(args.source_root.glob("full_flash*/analysis.json"),
                   key=lambda p: json.loads(p.read_text())["source"]["case"]["flash_attention_block_size"],
                   reverse=True)
    if not paths:
        parser.error("No double-buffer analysis.json files found")
    views, comparison, cxl_rows = [], [], []
    for path in paths:
        before_hash = digest(path)
        source = json.loads(path.read_text())
        view = make_view(source, path)
        target = output / view["label"]
        target.mkdir(parents=True, exist_ok=True)
        (target / "analysis.json").write_text(json.dumps(view, indent=2) + "\n")
        (target / "summary.txt").write_text(render(view))
        for name, model in view["models"].items():
            write_double_buffer_kernel_table(view, name, target / f"{name}_kernels.csv")
            comparison.append({"case": view["label"], "scenario": name, **model["totals"]})
            for row in model["kernels"]:
                if row["cxl_equivalent_cycles"]:
                    cxl_rows.append({"case": view["label"], "scenario": name,
                                     "component": row["kernel"],
                                     "cxl_equivalent_cycles": row["cxl_equivalent_cycles"],
                                     "cxl_us": row["new_total_us_estimate"]})
        if digest(path) != before_hash:
            raise ValueError("Source analysis changed during separation")
        views.append(view)
    write_csv(output / "comparison.csv", comparison)
    write_csv(output / "cxl_breakdown.csv", cxl_rows)
    primary = views[0]
    lines = ["Double buffering — CXL separated results", "",
             "单独展示版本：从 O Projection / Down Projection 中拆出 TP CXL，原版本保持不变。",
             f"主配置：{primary['label']}；其余配置保留为对照。",
             "PNM/PIM 计算、CXL 总量、double-buffer overlap 和 block 总延迟均未改变。", ""]
    if "cxl_bandwidth_model" in primary:
        cxl = primary["cxl_bandwidth_model"]
        lines += [f"本版本 CXL：{cxl['effective_bandwidth_per_device']} {cxl['unit']}/device；multicast 折扣=0.5。",
                  "拆分保持输入 analysis 的总量；原始模拟 source 仍保留 legacy GiB/s 数值。", ""]
    for view in views:
        t = view["models"]["dependency_aware"]["totals"]
        lines.append(f"{view['label']}/dependency_aware: PIM={t['new_pim_cycles_analytical']:,} cycles; block={t['new_block_us_analytical']:.6f} us; CXL={t['cxl_us_frozen']:.6f} us.")
    lines += ["", "主配置 dependency_aware 的拆分结果：",
              "component                                PIM cycles      CXL eq cycles    total(us)"]
    for row in primary["models"]["dependency_aware"]["kernels"]:
        if row["region_id"] in SPLITS or row["cxl_equivalent_cycles"]:
            lines.append(f"{row['kernel']:<40} {row['new_pim_cycles_estimate']:>10,.0f} {row['cxl_equivalent_cycles']:>18.6f} {row['new_total_us_estimate']:>12.6f}")
    lines += ["", "每个配置的 summary.txt 展示全部 kernel、两个 analytical scenarios，以及单缓冲/双缓冲合计。",
              "CSV/JSON 包含同样的拆分结果；waves 未变，保存在 analysis.json 内。",
              "Kernel CSV 使用 PIM/PNM/CXL 分组双行表头，仅保留 19 列和 Block total；省略零开销行。WRGB 为双缓冲后剩余 issue-gap 周期，详细 overlap 数据保留在 JSON。",
              "验证：拆分前后所有数值列的总和、PIM/PNM/CXL 总量与 block 总延迟守恒。",
              "", "Reproduce from repository root:",
              "  python cent_simulation/scripts/separate_legacy_cxl.py"]
    (output / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:18]))
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
