"""Issued-command cycle breakdowns for selected Systolic PIM layouts.

The categorization and critical-channel policy are deliberately shared with
``aim_simulator/scripts/plot_gemv_cycle_breakdown.py``.  CENT campaign logs
usually retain only Ramulator statistics, so this module can add a
TraceRecorder sidecar for a selected trace without changing the source CSV.
"""

from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys
import tempfile

import pandas as pd

from scripts.utility.system_energy_breakdown import (
    _bool,
    load_selected_source_row,
    source_log_root,
)


CENT_SIM = Path(__file__).resolve().parents[2]
PROJECT_ROOT = CENT_SIM.parent
AIM_SCRIPTS = PROJECT_ROOT / "aim_simulator" / "scripts"
if str(AIM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(AIM_SCRIPTS))

from aim_analysis.commands import STACK_LABELS, command_component  # noqa: E402
from aim_analysis.ramulator import (  # noqa: E402
    command_trace_files,
    parse_command_trace,
    read_result_stats,
    result_cycles,
)


CYCLE_COMPONENT_COLORS = {
    "WR_GB": "#4C78A8",
    "WR_BIAS": "#F58518",
    "ACT_PRE": "#E45756",
    "MAC_ABK": "#54A24B",
    "RD_MAC": "#B279A2",
    "TMOD": "#72B7B2",
    "Other": "#BAB0AC",
}


def systolic_trace_variant(source: pd.Series) -> str:
    """Return the source trace variant used by a Systolic campaign row."""

    variant = (
        f"systolic_pim_{int(source['Systolic dim'])}"
        f"_batch_size_{int(source.get('Batch size', 1))}"
        f"_ewmul_pnm_{int(_bool(source.get('EWMUL PNM effective', False)))}"
    )
    if _bool(source.get("Flash attention", False)):
        variant += f"_flash_{int(source['Flash attention block size'])}"
    if str(source.get("Precision", "BF16")).upper() == "FP8":
        variant += "_fp8"
    return variant


def selected_systolic_paths(
    candidate: pd.Series,
    source_path: Path,
    source: pd.Series,
    *,
    trace_root: Path,
) -> tuple[Path, Path]:
    """Resolve the matching source trace and existing Ramulator log."""

    if not _bool(source.get("Systolic pim", False)):
        raise ValueError("cycle breakdown requires a Systolic PIM source row")
    if str(source.get("Attention mapping", "")) != "kv_head":
        raise ValueError("cycle breakdown currently supports KV-head Systolic mappings")

    memory = str(candidate["Memory"])
    model = str(candidate["Model"])
    tp = int(source["Tensor parallelism"])
    seqlen = int(source["Sequence length"])
    variant = systolic_trace_variant(source)
    filename = f"trace_{tp}_FC_devices_seqlen_{seqlen}.txt"
    log = (
        source_log_root(source_path, memory)
        / variant
        / "model_parallel_kv_head_main"
        / model
        / f"{filename}.log"
    )

    output_root = CENT_SIM / "output"
    try:
        source_parts = source_path.parent.relative_to(output_root).parts
    except ValueError as error:
        raise ValueError(
            f"source CSV must reside under {output_root}: {source_path}"
        ) from error
    if len(source_parts) < 2:
        raise ValueError(f"cannot infer trace campaign from source CSV: {source_path}")
    # Campaigns written under ``raw/<array>/<memory>`` mirror their trace tree
    # without ``raw``.  Older FP8 campaigns directly use ``<campaign>/<memory>``.
    memory_dir = source_parts[-1]
    campaign_parts = list(source_parts[:-1])
    if "raw" in campaign_parts:
        raw_index = campaign_parts.index("raw")
        campaign_parts.pop(raw_index)
    trace = (
        trace_root
        / Path(*campaign_parts)
        / "long_context_midpoint"
        / memory_dir
        / variant
        / "model_parallel_kv_head_main"
        / model
        / filename
    )
    return trace, log


def trace_recorder_config(config: Path, command_prefix: Path) -> str:
    """Add a TraceRecorder plugin to an existing resolved Ramulator config."""

    output: list[str] = []
    inserted = False
    for line in config.read_text().splitlines(keepends=True):
        output.append(line)
        if line.strip() == "plugins:":
            indent = line[: len(line) - len(line.lstrip())]
            output.extend(
                [
                    f"{indent}  - ControllerPlugin:\n",
                    f"{indent}      impl: TraceRecorder\n",
                    f"{indent}      path: {command_prefix}\n",
                ]
            )
            inserted = True
    if not inserted:
        raise ValueError(f"Ramulator config has no plugins section: {config}")
    return "".join(output)


def ensure_command_traces(trace: Path, log: Path, ramulator: Path) -> Path:
    """Reuse or record TraceRecorder sidecars for one immutable source trace."""

    command_prefix = Path(str(log)[:-4] if str(log).endswith(".log") else str(log))
    command_prefix = Path(f"{command_prefix}.cmd")
    if command_trace_files(command_prefix):
        return command_prefix
    if not trace.exists() or trace.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty Systolic source trace: {trace}")
    if not log.exists() or log.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty Systolic Ramulator log: {log}")
    config = Path(str(log)[:-4] if str(log).endswith(".log") else str(log))
    config = Path(f"{config}.ramulator.yaml")
    if not config.exists() or config.stat().st_size == 0:
        raise FileNotFoundError(f"missing resolved Ramulator config: {config}")
    if not ramulator.is_file():
        raise FileNotFoundError(f"Ramulator binary not found: {ramulator}")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="cent-cycle-", delete=False
    ) as handle:
        temp_config = Path(handle.name)
        handle.write(trace_recorder_config(config, command_prefix))
    try:
        completed = subprocess.run(
            [str(ramulator), "-f", str(temp_config), "-t", str(trace)],
            cwd=CENT_SIM,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "TraceRecorder Ramulator run failed for "
                f"{trace}:\n{completed.stdout}\n{completed.stderr}"
            )
    finally:
        temp_config.unlink(missing_ok=True)
    if not command_trace_files(command_prefix):
        raise RuntimeError(f"TraceRecorder produced no channel traces for {trace}")
    return command_prefix


def select_critical_command_trace(command_prefix: Path) -> Path:
    """Match AIM's longest-issued-channel choice for a multi-channel trace."""

    def last_issued_clock(path: Path) -> int:
        """Read the final issued-command timestamp without parsing the full file."""

        block_size = 8192
        with path.open("rb") as handle:
            handle.seek(0, 2)
            remaining = handle.tell()
            trailing = b""
            while remaining:
                read_size = min(block_size, remaining)
                remaining -= read_size
                handle.seek(remaining)
                lines = (handle.read(read_size) + trailing).splitlines()
                trailing = lines[0] if lines else b""
                for line in reversed(lines[1:] if remaining else lines):
                    clock_text = line.split(b",", 1)[0].strip()
                    if clock_text.isdigit():
                        return int(clock_text)
        raise ValueError(f"no issued commands found in {path}")

    def issued_command_count(path: Path) -> int:
        """Count TraceRecorder rows using block reads instead of object parsing."""

        count = 0
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                count += block.count(b"\n")
        return count

    candidates: list[tuple[int, int, Path]] = []
    for path in command_trace_files(command_prefix):
        candidates.append((last_issued_clock(path), issued_command_count(path), path))
    if not candidates:
        raise RuntimeError(f"no issued commands found for {command_prefix}")
    return max(candidates)[2]


def command_cycle_components(
    command_trace: Path, memory_system_cycles: int
) -> dict[str, int]:
    """Attribute intervals exactly as AIM's GEMV command-component plot does."""

    if memory_system_cycles <= 0:
        raise ValueError("memory_system_cycles must be positive")
    values = {component: 0 for component in STACK_LABELS}
    issued = parse_command_trace(command_trace)
    for previous, current in zip(issued, issued[1:]):
        interval = current.clock - previous.clock
        if interval < 0:
            raise ValueError(f"command trace is not monotonic: {command_trace}")
        values[command_component(current.command)] += interval
    component_sum = sum(values.values())
    values["Other"] += max(memory_system_cycles - component_sum, 0)
    if sum(values.values()) != memory_system_cycles:
        raise ValueError(
            f"cycle components exceed memory_system_cycles for {command_trace}: "
            f"{sum(values.values())} > {memory_system_cycles}"
        )
    return values


def build_selected_systolic_cycle_breakdown(
    selected: pd.DataFrame,
    *,
    project_root: Path,
    trace_root: Path | None = None,
    ramulator: Path | None = None,
) -> pd.DataFrame:
    """Record and break down one critical PIM trace per selected layout."""

    trace_root = trace_root or CENT_SIM / "trace"
    ramulator = ramulator or PROJECT_ROOT / "aim_simulator" / "build" / "ramulator2"
    raw_cache: dict[Path, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    for _, candidate in selected.iterrows():
        source_path, source = load_selected_source_row(candidate, raw_cache, project_root)
        trace, log = selected_systolic_paths(
            candidate, source_path, source, trace_root=trace_root
        )
        command_prefix = ensure_command_traces(trace, log, ramulator)
        command_trace = select_critical_command_trace(command_prefix)
        memory_system_cycles = result_cycles(read_result_stats(log))
        components = command_cycle_components(command_trace, memory_system_cycles)
        row = {
            "Architecture": candidate.get("Architecture", "Systolic"),
            "Memory": candidate["Memory"],
            "Model": candidate["Model"],
            "Context": candidate["Context"],
            "PP": int(candidate["PP"]),
            "TP": int(candidate["TP"]),
            "Batch size": int(candidate.get("Batch size", 1)),
            "Source PIM latency (ms/block)": float(source["Main PIM latency"]),
            "memory_system_cycles": memory_system_cycles,
            "issued_commands": len(parse_command_trace(command_trace)),
            "Source trace": str(trace.relative_to(project_root)),
            "Source log": str(log.relative_to(project_root)),
            "Critical command trace": str(command_trace.relative_to(project_root)),
        }
        row.update(components)
        row["component_sum"] = sum(components.values())
        rows.append(row)
    return pd.DataFrame(rows)


def write_cycle_breakdown_csv(rows: pd.DataFrame, path: Path) -> None:
    """Write the audit table with a stable, human-readable column order."""

    fieldnames = [
        "Architecture",
        "Memory",
        "Model",
        "Context",
        "PP",
        "TP",
        "Batch size",
        "Source PIM latency (ms/block)",
        "memory_system_cycles",
        "issued_commands",
        *STACK_LABELS,
        "component_sum",
        "Source trace",
        "Source log",
        "Critical command trace",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows.to_dict("records"))


def plot_selected_systolic_cycle_breakdown(rows: pd.DataFrame, output: Path) -> None:
    """Draw absolute and normalized critical-channel PIM cycle stacks."""

    import matplotlib.pyplot as plt

    for model in sorted(rows["Model"].unique()):
        model_rows = rows[rows["Model"] == model].sort_values(["Context", "Memory"])
        count = len(model_rows)
        figure_width = max(6.4, 4.9 + 0.55 * count)
        positions = list(range(count))
        labels = [f"{row['Context']}\n{row['Memory'].replace('LPDDR4X_nCCD', 'X')}" for _, row in model_rows.iterrows()]
        fig, (absolute, percentage) = plt.subplots(
            2,
            1,
            figsize=(figure_width, 7.9),
            sharex=True,
            gridspec_kw={"height_ratios": [1.35, 1.0]},
        )
        for axis, normalized in ((absolute, False), (percentage, True)):
            bottoms = [0.0] * count
            for component in STACK_LABELS:
                raw_values = [float(row[component]) for _, row in model_rows.iterrows()]
                values = (
                    [100.0 * value / float(row["memory_system_cycles"])
                     for value, (_, row) in zip(raw_values, model_rows.iterrows())]
                    if normalized
                    else raw_values
                )
                axis.bar(
                    positions,
                    values,
                    bottom=bottoms,
                    width=0.14 if count == 1 else min(0.44, 0.72 / count),
                    label=component,
                    color=CYCLE_COMPONENT_COLORS[component],
                    edgecolor="white",
                    linewidth=0.5,
                )
                bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
            axis.grid(axis="y", alpha=0.25)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.set_xlim(-0.25, max(positions) + 0.25)
        absolute.set_ylabel("memory-system cycles / block")
        absolute.set_title("Absolute cycles", loc="left", fontsize=11, fontweight="semibold")
        percentage.set_ylabel("share of cycles (%)")
        percentage.set_ylim(0.0, 100.0)
        percentage.set_title("Percentage", loc="left", fontsize=11, fontweight="semibold")
        percentage.set_xticks(positions, labels)
        percentage.tick_params(axis="x", length=0, pad=5)
        handles, labels_legend = absolute.get_legend_handles_labels()
        fig.suptitle(
            f"{model}: Systolic PIM cycle breakdown from issued commands",
            y=0.985,
            fontsize=13,
        )
        fig.legend(
            handles,
            labels_legend,
            ncols=4,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.945),
            frameon=False,
            fontsize=8,
        )
        fig.subplots_adjust(left=0.13, right=0.98, bottom=0.12, top=0.84, hspace=0.27)
        stem = output.parent / f"{output.name}_{model.replace('Llama2-', '').lower()}"
        for suffix in ("png", "pdf"):
            fig.savefig(stem.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
        plt.close(fig)
