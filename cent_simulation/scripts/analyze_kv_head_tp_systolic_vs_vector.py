#!/usr/bin/env python3
"""Compare existing Vector and Systolic all-context candidate tables.

This is post-processing only: it does not generate traces or run a simulator.
Vector candidates are read from ``kv_head_tp_vector_all_context`` and
Systolic candidates are read from ``kv_head_tp_systolic_all_context``.  Each
architecture independently selects its best PP/TP/batch per memory/workload;
only then is the chosen replica DP-scaled to the nearest DGX H100 power.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import pandas as pd


CENT_SIM = Path(__file__).resolve().parents[1]
ROOT = CENT_SIM.parent
if str(CENT_SIM) not in sys.path:
    sys.path.insert(0, str(CENT_SIM))

from scripts import analyze_balanced_equal_power as balanced
from scripts.utility.campaign_plots import plot_grouped_metric
from scripts.utility.campaign_selection import (
    build_all_equal_power_deployments,
    select_base_objective,
    select_equal_power,
)
from scripts.utility.system_energy_breakdown import (
    build_selected_component_breakdown,
    plot_selected_component_breakdown,
    plot_selected_component_breakdown_context_subplots,
)


DEFAULT_ROOT = CENT_SIM / "output/kv_head_tp_systolic_vs_vector"
DEFAULT_SYSTOLIC_ANALYSIS = (
    CENT_SIM / "output/kv_head_tp_systolic_all_context/analysis"
)
DEFAULT_VECTOR_ANALYSIS = (
    CENT_SIM / "output/kv_head_tp_vector_all_context/analysis"
)
ARCHITECTURES = ("Vector", "Systolic 4x16")
MEMORIES = tuple(balanced.MEMORY_CASES)
ARCH_HATCH = {"Vector": "", "Systolic 4x16": "///"}
MEMORY_COLORS = {
    # Vega/Altair Category10's first three colors. These apply only to the
    # scalar memory bars; system-energy components retain their own palette.
    "GDDR6": "#4C78A8",
    "LPDDR4X_nCCD2": "#F58518",
    "LPDDR4X_nCCD6": "#E45756",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systolic-analysis",
        type=Path,
        default=DEFAULT_SYSTOLIC_ANALYSIS,
        help="Existing Systolic-only analysis directory; no simulation is run.",
    )
    parser.add_argument(
        "--vector-analysis",
        type=Path,
        default=DEFAULT_VECTOR_ANALYSIS,
        help="Existing KV-head-TP Vector analysis directory; no simulation is run.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "analysis")
    parser.add_argument(
        "--h100-profile", type=Path, default=ROOT / "DGX_H100_profile_results.csv"
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_csv(analysis_dir: Path, label: str) -> Path:
    path = (analysis_dir / "all_candidates.csv").resolve()
    if not path.exists():
        raise FileNotFoundError(f"missing {label} candidate table: {path}")
    return path


def _check_required_columns(rows: pd.DataFrame, label: str) -> None:
    required = {
        "Memory",
        "Model",
        "Context",
        "PP",
        "TP",
        "Replica devices",
        "Replica power (W)",
        "Replica throughput (tokens/s)",
        "Throughput / device (tokens/s/device)",
        "Effective token energy (mJ)",
        "Tokens/J",
        "Source CSV",
        "Source PP",
        "Source TP",
        "Trace-external gap energy (mJ/token)",
        "Work token energy (mJ)",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"{label} table is missing columns: {', '.join(missing)}")


def load_systolic_candidates(analysis_dir: Path) -> tuple[pd.DataFrame, Path]:
    """Read Systolic selections only; never rebuild the candidate sweep."""

    source = _candidate_csv(analysis_dir, "Systolic")
    rows = pd.read_csv(source)
    _check_required_columns(rows, "Systolic")
    required_values = {
        "Architecture": "Systolic 4x16",
        "Physical mapping": "device_channel_group_bank",
        "Flash attention": True,
        "Pipelined softmax": True,
    }
    for column, expected in required_values.items():
        if column not in rows:
            raise ValueError(f"Systolic table is missing provenance column: {column}")
        if not rows[column].map(_as_bool if isinstance(expected, bool) else str).eq(expected).all():
            raise ValueError(
                f"Systolic table must contain only {column}={expected!r} rows"
            )
    if "Systolic pim" in rows and not rows["Systolic pim"].map(_as_bool).all():
        raise ValueError("Systolic candidate table contains a non-systolic row")
    rows = _campaign_scope(rows)
    if rows.empty:
        raise ValueError("Systolic candidate table has no campaign-scope rows")
    return rows, source


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _campaign_scope(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[
        rows["Memory"].isin(MEMORIES)
        & rows["Model"].isin(balanced.MODEL_CONFIG)
        & rows["Context"].isin(balanced.CONTEXTS)
    ].copy()


def load_vector_candidates(analysis_dir: Path) -> tuple[pd.DataFrame, Path]:
    """Read the prescribed balanced Vector candidates and normalize metadata."""

    source = _candidate_csv(analysis_dir, "Vector")
    if "kv_head_tp_vector_all_context" not in str(source):
        raise ValueError(
            "Vector input must be output/kv_head_tp_vector_all_context/analysis"
        )
    rows = pd.read_csv(source)
    _check_required_columns(rows, "Vector")
    for value in rows["Source CSV"].dropna():
        if "kv_head_tp_vector_all_context" not in str(value):
            raise ValueError(
                "Vector candidate source must remain kv_head_tp_vector_all_context"
            )
    manifest_path = analysis_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing KV-head Vector manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("attention_mapping") != "kv_head"
        or manifest.get("kv_mapping") != "kv_head_sharded"
    ):
        raise ValueError("Vector source manifest does not describe KV-head TP")
    rows = _campaign_scope(rows)
    if rows.empty:
        raise ValueError("Vector candidate table has no campaign-scope rows")
    rows["Architecture"] = "Vector"
    if not (rows["Attention mapping"] == "kv_head").all():
        raise ValueError("Vector candidates must retain KV-head attention mapping")
    if not (rows["KV mapping"] == "kv_head_sharded").all():
        raise ValueError("Vector candidates must retain sharded KV mapping")
    rows["Physical mapping"] = "kv_head_vector_reused"
    rows["Batch size"] = 1
    rows["Max resident requests"] = rows["Max resident microbatch"]
    rows["Max resident batch groups"] = rows["Max resident microbatch"]
    rows["Batch groups used"] = rows["Microbatch used"]
    rows["Resident requests used"] = rows["Microbatch used"]
    rows["Can host one batch"] = rows["Can host one request"]
    rows["Projection SA row utilization"] = math.nan
    return rows, source


def add_architecture_ranks(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = candidates.copy()
    rows["Architecture throughput/device rank"] = pd.NA
    rows["Architecture Tokens/J rank"] = pd.NA
    admitted = rows["Can host one batch"].map(_as_bool)
    groups = ["Architecture", "Memory", "Model", "Context"]
    for _, group in rows[admitted].groupby(groups):
        rows.loc[group.index, "Architecture throughput/device rank"] = group[
            "Throughput / device (tokens/s/device)"
        ].rank(method="dense", ascending=False)
        rows.loc[group.index, "Architecture Tokens/J rank"] = group["Tokens/J"].rank(
            method="dense", ascending=False
        )
    return rows


def write_plots(
    candidates: pd.DataFrame,
    selected_equal_power: pd.DataFrame,
    component_breakdown: pd.DataFrame,
    dgx: dict[tuple[str, str], dict[str, float | int]],
    output_dir: Path,
) -> None:
    throughput = select_base_objective(
        candidates, "Throughput / device (tokens/s/device)"
    )
    efficiency = select_base_objective(candidates, "Tokens/J")
    dgx_tokens_per_joule = {
        workload: float(reference["throughput"]) / float(reference["power"])
        for workload, reference in dgx.items()
    }
    common = dict(
        architectures=ARCHITECTURES,
        memories=MEMORIES,
        contexts=balanced.CONTEXTS,
        models=balanced.MODEL_CONFIG,
        memory_colors=MEMORY_COLORS,
        architecture_hatches=ARCH_HATCH,
    )
    plot_grouped_metric(
        throughput,
        **common,
        metric="Throughput / device (tokens/s/device)",
        ylabel="Tokens/s/device",
        title="best capacity-admitted PP/TP/batch (labels: PP/TP/B)",
        output=output_dir / "best_throughput_per_device",
        annotation="pp_tp",
    )
    plot_grouped_metric(
        efficiency,
        **common,
        metric="Tokens/J",
        ylabel="Tokens/J",
        title="best energy efficiency (labels: token/J/DGX; PP/TP/B)",
        output=output_dir / "best_tokens_per_joule",
        annotation="tokens_per_joule",
        dgx_line=True,
        dgx_reference=dgx_tokens_per_joule,
    )
    plot_selected_component_breakdown_context_subplots(
        component_breakdown,
        architectures=ARCHITECTURES,
        memories=MEMORIES,
        contexts=balanced.CONTEXTS,
        models=balanced.MODEL_CONFIG,
        metric_suffix="energy (mJ/token)",
        ylabel="Effective token energy (mJ/token)",
        title="system energy breakdown for throughput/device-selected layouts",
        output=output_dir / "selected_token_energy",
        external_title_legend=True,
    )
    plot_selected_component_breakdown(
        component_breakdown,
        architectures=ARCHITECTURES,
        memories=MEMORIES,
        contexts=balanced.CONTEXTS,
        models=balanced.MODEL_CONFIG,
        metric_suffix="power / device (W)",
        ylabel="Average power per provisioned device (W/device)",
        title="system power/device breakdown for throughput/device-selected layouts",
        output=output_dir / "selected_power_per_device",
        external_title_legend=True,
    )
    plot_grouped_metric(
        selected_equal_power,
        **common,
        metric="System throughput (tokens/s)",
        ylabel="System throughput (tokens/s)",
        title="DGX H100 equal-power winners (labels: throughput/DGX; PP/TP/B/DP)",
        output=output_dir / "equal_power_system_throughput",
        annotation="equal_power",
        dgx_line=True,
        external_title_legend=True,
    )


def write_readme(
    output_dir: Path,
    candidates: pd.DataFrame,
    winners: pd.DataFrame,
) -> None:
    summary = winners.sort_values(
        ["Model", "Context window", "Architecture", "Memory"]
    )[
        [
            "Architecture",
            "Memory",
            "Model",
            "Context",
            "PP",
            "TP",
            "Batch size",
            "DP",
            "System throughput (tokens/s)",
            "System power (W)",
            "Throughput / DGX H100",
        ]
    ]
    text = f"""# Vector vs Systolic all-context comparison

This directory is analysis-only: it neither generates traces nor runs Vector
or Systolic simulations. It reads exactly these candidate tables:

- Vector: `output/kv_head_tp_vector_all_context/analysis/all_candidates.csv`
- Systolic 4x16: `output/kv_head_tp_systolic_all_context/analysis/all_candidates.csv`

For every **architecture × memory × model × context**, PP/TP/batch is selected
independently by maximum throughput/device. Equal-score ties choose fewer
replica devices and then smaller PP, TP, and batch. After that selection is
fixed, only integer DP is scaled near DGX H100 power; it cannot reselect a
different PP/TP/batch layout. This is the same five-metric bundle for both
architectures, not the separate trace-level mapping microcomparison.

Each scalar figure uses architecture hatching; both selected energy figures
use the hierarchy **context → Vector/Systolic → G6/X2/X6**. They reconstruct
the physical CENT energy categories plus explicit trace-external and pipeline
waiting terms. `selected_token_energy` is a three-panel context figure
(4K/32K/128K), with an independent y-axis in every panel. The figures are:

1. `best_throughput_per_device_{'{'}7b,70b{'}'}`
2. `best_tokens_per_joule_{'{'}7b,70b{'}'}`
3. `selected_token_energy_{'{'}7b,70b{'}'}`
4. `selected_power_per_device_{'{'}7b,70b{'}'}`
5. `equal_power_system_throughput_{'{'}7b,70b{'}'}`

Thus this directory contains ten PNG and ten PDF figures.

Candidates: {len(candidates)} total; {int(candidates['Can host one batch'].map(_as_bool).sum())} admitted.
Equal-power winners: {len(winners)} total (six per model/context).

## Equal-power winners

{balanced.markdown_table(summary, list(summary.columns))}

## Reproduction

```bash
/home/linuswang/miniforge3/envs/cent/bin/python \\
  scripts/analyze_kv_head_tp_systolic_vs_vector.py
```
"""
    (output_dir / "README.md").write_text(text)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    vector, vector_source = load_vector_candidates(args.vector_analysis.resolve())
    systolic, systolic_source = load_systolic_candidates(args.systolic_analysis.resolve())
    candidates = pd.concat([vector, systolic], ignore_index=True, sort=False)
    candidates = add_architecture_ranks(candidates).sort_values(
        ["Model", "Context window", "Architecture", "Memory", "PP", "TP", "Batch size"]
    )
    best_throughput = select_base_objective(
        candidates, "Throughput / device (tokens/s/device)"
    ).sort_values(["Model", "Context window", "Architecture", "Memory"])
    best_tokens_per_joule = select_base_objective(candidates, "Tokens/J").sort_values(
        ["Model", "Context window", "Architecture", "Memory"]
    )
    component_breakdown = build_selected_component_breakdown(
        best_throughput,
        memory_cases=balanced.MEMORY_CASES,
        model_specs=balanced.MODEL_SPECS,
        project_root=ROOT,
    ).sort_values(["Model", "Context window", "Architecture", "Memory"])
    dgx = balanced.load_dgx(args.h100_profile.resolve())
    deployments = build_all_equal_power_deployments(
        best_throughput, dgx, balanced.integer_dp_choices
    ).sort_values(
        [
            "Model",
            "Context window",
            "Architecture",
            "Memory",
            "PP",
            "TP",
            "Batch size",
            "DP",
        ]
    )
    winners = select_equal_power(deployments).sort_values(
        ["Model", "Context window", "Architecture", "Memory"]
    )
    expected_winners = (
        len(ARCHITECTURES)
        * len(MEMORIES)
        * len(balanced.MODEL_CONFIG)
        * len(balanced.CONTEXTS)
    )
    if len(winners) != expected_winners:
        raise ValueError(
            f"expected {expected_winners} full-campaign winners, found {len(winners)}"
        )

    outputs = {
        "all_candidates.csv": candidates,
        "equal_power_deployments.csv": deployments,
        "equal_power_selected.csv": winners,
        "best_throughput_per_device.csv": best_throughput,
        "best_tokens_per_joule.csv": best_tokens_per_joule,
        "selected_throughput_energy_component_breakdown.csv": component_breakdown,
    }
    for filename, frame in outputs.items():
        path = output_dir / filename
        frame.to_csv(path, index=False)
        print(f"[data] {balanced.display_path(path)}")

    manifest = {
        "architectures": list(ARCHITECTURES),
        "memories": list(MEMORIES),
        "models": list(balanced.MODEL_CONFIG),
        "contexts": balanced.CONTEXTS,
        "systolic_source": {
            "path": balanced.display_path(systolic_source),
            "sha256": file_sha256(systolic_source),
            "read_only_reuse": True,
        },
        "vector_source": {
            "path": balanced.display_path(vector_source),
            "sha256": file_sha256(vector_source),
            "read_only_reuse": True,
        },
        "simulation_runs": 0,
        "trace_generations": 0,
        "selection_rule": (
            "per_architecture_max_throughput_per_device_then_fewer_replica_devices_"
            "smaller_PP_TP_batch_then_fixed_layout_nearest_integer_DP"
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_readme(output_dir, candidates, winners)

    if not args.no_plots:
        write_plots(candidates, winners, component_breakdown, dgx, output_dir)
        print(f"[plots] {balanced.display_path(output_dir)}")

    columns = [
        "Architecture",
        "Memory",
        "Model",
        "Context",
        "PP",
        "TP",
        "Batch size",
        "DP",
        "System throughput (tokens/s)",
        "System power (W)",
        "Throughput / DGX H100",
    ]
    print(winners[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
