"""Systolic-style five-metric presentation for the existing Vector campaign."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

import pandas as pd

from scripts.utility.campaign_plots import plot_grouped_metric
from scripts.utility.system_energy_breakdown import (
    build_selected_component_breakdown,
    plot_selected_component_breakdown,
)


ARCHITECTURES = ("Vector",)
MEMORY_COLORS = {
    "GDDR6": "#4C78A8",
    "LPDDR4X_nCCD2": "#F58518",
    "LPDDR4X_nCCD6": "#E45756",
}
ARCHITECTURE_HATCHES = {"Vector": ""}
LEGACY_PLOT_STEMS = (
    "best_throughput_per_device",
    "best_tokens_per_joule",
    "selected_power_per_device",
    "selected_token_energy",
    "selected_throughput_energy_breakdown",
    "equal_power_system_throughput",
)


def select_existing_equal_power_winners(deployments: pd.DataFrame) -> pd.DataFrame:
    """Retain the Vector campaign's pre-existing objective and DP semantics."""

    rows = deployments[
        (deployments["Objective"] == "throughput_per_device")
        & deployments["Closest integer DP"].astype(bool)
    ].copy()
    winners: list[dict[str, object]] = []
    groups = ["Memory", "Model", "Context"]
    for key, group in rows.groupby(groups):
        winner = group.sort_values(
            ["Absolute power delta vs DGX (W)", "DP"], ascending=[True, True]
        ).iloc[0]
        winners.append(winner.to_dict())
    selected = pd.DataFrame(winners)
    expected = len(rows[groups].drop_duplicates())
    if len(selected) != expected:
        raise ValueError(f"expected {expected} Vector equal-power winners, found {len(selected)}")
    return selected


def _remove_legacy_plot_files(output_dir: Path) -> None:
    for stem in LEGACY_PLOT_STEMS:
        for suffix in ("png", "pdf"):
            path = output_dir / f"{stem}.{suffix}"
            if path.exists():
                path.unlink()


def _write_readme(
    output_dir: Path,
    throughput: pd.DataFrame,
    winners: pd.DataFrame,
) -> None:
    text = f"""# KV-head TP Vector all-context campaign

This directory retains the existing Vector KV-head TP candidate generation and
selection semantics. PP/TP ties, the throughput/device and Tokens/J objectives,
and the pre-existing closest-integer-DP equal-power selection are unchanged.
Only the presentation is aligned with the Systolic 4x16 campaign.

The five metrics are emitted separately for 7B and 70B:

1. `best_throughput_per_device_{{7b,70b}}`
2. `best_tokens_per_joule_{{7b,70b}}`
3. `selected_token_energy_{{7b,70b}}`
4. `selected_power_per_device_{{7b,70b}}`
5. `equal_power_system_throughput_{{7b,70b}}`

Scalar bars use Vega/Altair G6/X2/X6 colors. The energy and power figures use
the CENT system-energy palette and context → Vector → G6/X2/X6 hierarchy.
`selected_throughput_energy_component_breakdown.csv` audits that each physical
stack reconstructs the selected effective token energy and device power.

Throughput/device selections: {len(throughput)}. Equal-power winners:
{len(winners)}.
"""
    (output_dir / "README.md").write_text(text)


def write_vector_systolic_style_presentation(
    *,
    selected: pd.DataFrame,
    deployments: pd.DataFrame,
    dgx: Mapping[tuple[str, str], Mapping[str, float | int]],
    output_dir: Path,
    memory_cases: Mapping[str, Mapping[str, object]],
    model_specs: Mapping[str, Mapping[str, int]],
    contexts: Mapping[str, object],
    models: Mapping[str, object],
    project_root: Path,
) -> None:
    """Write the Vector campaign's Systolic-style data views and figures.

    This function intentionally consumes the existing Vector winners without
    augmenting their candidate schema. Plot helpers use a declared implicit
    single architecture, and the source CSVs remain mapping-neutral.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    throughput = selected[selected["Objective"] == "throughput_per_device"].copy()
    efficiency = selected[selected["Objective"] == "tokens_per_joule"].copy()
    winners = select_existing_equal_power_winners(deployments)
    expected = len(memory_cases) * len(models) * len(contexts)
    if len(throughput) != expected or len(efficiency) != expected:
        raise ValueError(
            f"expected {expected} Vector winners per objective, found "
            f"{len(throughput)} throughput and {len(efficiency)} efficiency"
        )

    component_breakdown = build_selected_component_breakdown(
        throughput,
        memory_cases=memory_cases,
        model_specs=model_specs,
        project_root=project_root,
    ).sort_values(["Model", "Context window", "Memory"])
    outputs = {
        "best_throughput_per_device.csv": throughput.sort_values(
            ["Model", "Context window", "Memory"]
        ),
        "best_tokens_per_joule.csv": efficiency.sort_values(
            ["Model", "Context window", "Memory"]
        ),
        "equal_power_selected.csv": winners.sort_values(
            ["Model", "Context window", "Memory"]
        ),
        "selected_throughput_energy_component_breakdown.csv": component_breakdown,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)

    _remove_legacy_plot_files(output_dir)
    memories = tuple(memory_cases)
    dgx_tokens_per_joule = {
        workload: float(reference["throughput"]) / float(reference["power"])
        for workload, reference in dgx.items()
    }
    common = dict(
        architectures=ARCHITECTURES,
        memories=memories,
        contexts=contexts,
        models=models,
        memory_colors=MEMORY_COLORS,
        architecture_hatches=ARCHITECTURE_HATCHES,
        implicit_architecture="Vector",
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
    plot_selected_component_breakdown(
        component_breakdown,
        architectures=ARCHITECTURES,
        memories=memories,
        contexts=contexts,
        models=models,
        metric_suffix="energy (mJ/token)",
        ylabel="Effective token energy (mJ/token)",
        title="system energy breakdown for throughput/device-selected layouts",
        output=output_dir / "selected_token_energy",
        implicit_architecture="Vector",
    )
    plot_selected_component_breakdown(
        component_breakdown,
        architectures=ARCHITECTURES,
        memories=memories,
        contexts=contexts,
        models=models,
        metric_suffix="power / device (W)",
        ylabel="Average power per provisioned device (W/device)",
        title="system power/device breakdown for throughput/device-selected layouts",
        output=output_dir / "selected_power_per_device",
        implicit_architecture="Vector",
    )
    plot_grouped_metric(
        winners,
        **common,
        metric="System throughput (tokens/s)",
        ylabel="System throughput (tokens/s)",
        title="DGX H100 equal-power winners (labels: throughput/DGX; PP/TP/B/DP)",
        output=output_dir / "equal_power_system_throughput",
        annotation="equal_power",
        dgx_line=True,
    )

    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["presentation"] = "systolic_style_five_metrics"
        manifest["physical_energy_component_breakdown"] = True
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    _write_readme(output_dir, throughput, winners)
