#!/usr/bin/env python3
"""Render focused figures for the Winbond report from existing analysis CSVs."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


CENT_SIM = Path(__file__).resolve().parents[1]
if str(CENT_SIM) not in sys.path:
    sys.path.insert(0, str(CENT_SIM))


EQUAL_POWER_SOURCE = (
    CENT_SIM
    / "output/kv_head_tp_systolic_vs_vector/analysis/equal_power_selected.csv"
)
SELECTED_POWER_SOURCE = (
    CENT_SIM
    / "output/kv_head_tp_systolic_vs_vector/analysis/"
    "selected_throughput_energy_component_breakdown.csv"
)
BEST_TOKENS_PER_JOULE_SOURCE = (
    CENT_SIM / "output/kv_head_tp_systolic_vs_vector/analysis/best_tokens_per_joule.csv"
)
DGX_PROFILE = CENT_SIM.parent / "DGX_H100_profile_results.csv"
OUTPUT_DIR = CENT_SIM / "output/winbond_report"
MEMORIES = ("GDDR6", "LPDDR4X_nCCD2", "LPDDR4X_nCCD6")
ARCHITECTURES = ("Vector", "Systolic 4x16")
MEMORY_LABELS = {
    "GDDR6": "G6",
    "LPDDR4X_nCCD2": "X2",
    "LPDDR4X_nCCD6": "X6",
}
MEMORY_COLORS = {
    "GDDR6": "#4C78A8",
    "LPDDR4X_nCCD2": "#F58518",
    "LPDDR4X_nCCD6": "#E45756",
}
ARCHITECTURE_HATCHES = {"Vector": "", "Systolic 4x16": "///"}
TCO_PER_HOUR = {
    "DGX-H100 (Our Estimation)": 9.96,
    "DGX-H100 (Azure Rental Price)": 18.17,
    "Vector PIM": 1.45,
    "Systolic PIM": 1.37,
}
TCO_SYSTEM_COLORS = {
    "DGX-H100 (Our Estimation)": "#4C78A8",
    "DGX-H100 (Azure Rental Price)": "#4C78A8",
    "Vector PIM": "#F58518",
    "Systolic PIM": "#54A24B",
}
TCO_SYSTEM_HATCHES = {
    "DGX-H100 (Our Estimation)": "",
    "DGX-H100 (Azure Rental Price)": "///",
    "Vector PIM": "",
    "Systolic PIM": "",
}


def write_equal_power_tco_cost_per_token(rows: pd.DataFrame) -> pd.DataFrame:
    """Render the Llama2-7B/4K hourly-system-TCO cost/token comparison.

    The TCO inputs are whole-system hourly costs.  Consequently every bar is
    ``TCO ($/h) / (equal-power system throughput (token/s) * 3600)`` and does
    not scale the hourly cost again by the selected DP/device count.
    """

    import matplotlib.pyplot as plt

    model = "Llama2-7B"
    context = "4K"
    systems = (
        "DGX-H100 (Our Estimation)",
        "DGX-H100 (Azure Rental Price)",
        "Vector PIM",
        "Systolic PIM",
    )
    dgx_systems = systems[:2]
    source_memory = "LPDDR4X_nCCD2"
    source_architectures = {
        "Vector PIM": "Vector",
        "Systolic PIM": "Systolic 4x16",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    context_rows = rows[
        (rows["Model"] == model)
        & (rows["Context"] == context)
        & (rows["Memory"] == source_memory)
    ]
    if len(context_rows) != len(source_architectures):
        raise ValueError(
            f"expected Vector and Systolic nCCD2 rows for {model}/{context}, "
            f"found {len(context_rows)}"
        )
    dgx_throughput = float(context_rows["DGX H100 throughput (tokens/s)"].iloc[0])
    throughput_by_system = {system: dgx_throughput for system in dgx_systems}
    for system, architecture in source_architectures.items():
        match = context_rows[context_rows["Architecture"] == architecture]
        if len(match) != 1:
            raise ValueError(
                f"expected one {system} row for {model}/{context}, found {len(match)}"
            )
        throughput_by_system[system] = float(
            match.iloc[0]["System throughput (tokens/s)"]
        )
    costs = pd.DataFrame(
        {
            "Model": model,
            "Context": context,
            "System": systems,
            "Memory": ("N/A", "N/A", "LPDDR4X_nCCD2", "LPDDR4X_nCCD2"),
            "TCO ($/h)": [TCO_PER_HOUR[system] for system in systems],
            "Equal-power system throughput (tokens/s)": [
                throughput_by_system[system] for system in systems
            ],
        }
    )
    costs["Cost per generated token ($/token)"] = costs.apply(
        lambda row: float(row["TCO ($/h)"])
        / (float(row["Equal-power system throughput (tokens/s)"]) * 3600.0),
        axis=1,
    )
    stem = OUTPUT_DIR / "equal_power_tco_cost_per_token_7b_4k"
    costs.to_csv(stem.with_suffix(".csv"), index=False)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    positions = (0.0, 0.70, 1.95, 2.80)
    display_costs = costs["Cost per generated token ($/token)"] * 1.0e6
    bars = ax.bar(
        positions,
        display_costs,
        color=[TCO_SYSTEM_COLORS[system] for system in systems],
        hatch=[TCO_SYSTEM_HATCHES[system] for system in systems],
        edgecolor=["black" if TCO_SYSTEM_HATCHES[system] else "none" for system in systems],
        linewidth=0.55,
        width=0.52,
    )
    dgx_cost_per_token = float(
        costs.loc[
            costs["System"] == "DGX-H100 (Our Estimation)",
            "Cost per generated token ($/token)",
        ].iloc[0]
    )
    for bar, (_, row) in zip(bars, costs.iterrows()):
        cost_per_token = float(row["Cost per generated token ($/token)"])
        display_cost = cost_per_token * 1.0e6
        label = f"${display_cost:.2g}"
        if row["System"] not in dgx_systems:
            label += f"\n({cost_per_token / dgx_cost_per_token:.1%} of DGX)"
        ax.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(
        positions,
        ("Our Estimation", "Azure Rental Price", "Vector", "Systolic"),
        fontsize=8,
    )
    ax.tick_params(axis="x", length=0, pad=3)
    ax.text(
        sum(positions[:2]) / 2,
        -0.10,
        "DGX-H100",
        ha="center",
        va="top",
        transform=ax.get_xaxis_transform(),
        fontsize=10,
        fontweight="semibold",
    )
    ax.text(
        sum(positions[2:]) / 2,
        -0.10,
        "PIM",
        ha="center",
        va="top",
        transform=ax.get_xaxis_transform(),
        fontsize=10,
        fontweight="semibold",
    )
    ax.set_xlim(-0.42, 3.22)
    ax.set_ylabel("Cost ($/1M tokens)")
    ax.set_ylim(0.0, float(display_costs.max()) * 1.18)
    ax.grid(axis="y", alpha=0.25)
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Llama2-7B, 4K Context, Cost per 1M Token")
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.25, top=0.88)
    for suffix in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(stem.with_suffix(".png"))
    return costs


def write_equal_power_tokens_per_dollar(costs: pd.DataFrame) -> None:
    """Render the reciprocal Token/$ view for the same four TCO cases."""

    import matplotlib.pyplot as plt

    systems = tuple(costs["System"])
    dgx_systems = systems[:2]
    positions = (0.0, 0.70, 1.95, 2.80)
    tokens_per_dollar = 1.0 / costs["Cost per generated token ($/token)"]
    display_tokens_per_dollar = tokens_per_dollar / 1.0e6
    plotted = costs.copy()
    plotted["Tokens per dollar"] = tokens_per_dollar
    stem = OUTPUT_DIR / "equal_power_tokens_per_dollar_7b_4k"
    plotted.to_csv(stem.with_suffix(".csv"), index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bars = ax.bar(
        positions,
        display_tokens_per_dollar,
        color=[TCO_SYSTEM_COLORS[system] for system in systems],
        hatch=[TCO_SYSTEM_HATCHES[system] for system in systems],
        edgecolor=["black" if TCO_SYSTEM_HATCHES[system] else "none" for system in systems],
        linewidth=0.55,
        width=0.52,
    )
    dgx_tokens_per_dollar = float(display_tokens_per_dollar.iloc[0])
    for bar, value in zip(bars, display_tokens_per_dollar):
        ax.annotate(
            f"{float(value):.3g}M\n({float(value) / dgx_tokens_per_dollar:.2g}×)",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(
        positions,
        ("Our Estimation", "Azure Rental Price", "Vector", "Systolic"),
        fontsize=8,
    )
    ax.tick_params(axis="x", length=0, pad=3)
    ax.text(
        sum(positions[:2]) / 2,
        -0.10,
        "DGX-H100",
        ha="center",
        va="top",
        transform=ax.get_xaxis_transform(),
        fontsize=10,
        fontweight="semibold",
    )
    ax.text(
        sum(positions[2:]) / 2,
        -0.10,
        "PIM",
        ha="center",
        va="top",
        transform=ax.get_xaxis_transform(),
        fontsize=10,
        fontweight="semibold",
    )
    ax.set_xlim(-0.42, 3.22)
    ax.set_ylabel("Tokens/$ (M)")
    ax.set_ylim(0.0, float(display_tokens_per_dollar.max()) * 1.18)
    ax.grid(axis="y", alpha=0.25)
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Llama2-7B, 4K Context, Token/$")
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.25, top=0.88)
    for suffix in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(stem.with_suffix(".png"))


def main() -> int:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scripts import analyze_balanced_equal_power as balanced
    from scripts.utility.system_energy_breakdown import (
        ENERGY_COMPONENTS,
        SYSTEM_ENERGY_COLORS,
    )

    if not EQUAL_POWER_SOURCE.exists():
        raise FileNotFoundError(
            f"missing equal-power selection table: {EQUAL_POWER_SOURCE}"
        )
    rows = pd.read_csv(EQUAL_POWER_SOURCE)
    cost_rows = write_equal_power_tco_cost_per_token(rows)
    write_equal_power_tokens_per_dollar(cost_rows)
    rows = rows[(rows["Model"] == "Llama2-7B") & (rows["Context"] == "4K")].copy()
    expected = len(ARCHITECTURES) * len(MEMORIES)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} 7B/4K rows, found {len(rows)}")

    plt.style.use("seaborn-v0_8-whitegrid")
    width = 0.22
    memory_step = 0.27
    group_stride = 1.24
    positions: list[float] = []
    ordered_rows: list[pd.Series] = []
    labels: list[str] = []
    group_centers: list[tuple[float, str]] = []
    for group_index, architecture in enumerate(ARCHITECTURES):
        base = group_index * group_stride
        group_centers.append((base + memory_step, architecture.replace(" 4x16", "")))
        for memory_index, memory in enumerate(MEMORIES):
            match = rows[
                (rows["Architecture"] == architecture) & (rows["Memory"] == memory)
            ]
            if len(match) != 1:
                raise ValueError(
                    f"expected one row for {architecture}/{memory}, found {len(match)}"
                )
            positions.append(base + memory_index * memory_step)
            labels.append(MEMORY_LABELS[memory])
            ordered_rows.append(match.iloc[0])

    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    for position, row in zip(positions, ordered_rows):
        architecture = str(row["Architecture"])
        memory = str(row["Memory"])
        bar = ax.bar(
            position,
            float(row["System throughput (tokens/s)"]),
            width=width,
            color=MEMORY_COLORS[memory],
            hatch=ARCHITECTURE_HATCHES[architecture],
            edgecolor="black" if ARCHITECTURE_HATCHES[architecture] else "none",
            linewidth=0.55,
        )[0]
        ax.annotate(
            f"{float(row['Throughput / DGX H100']):.2f}x",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
            linespacing=0.9,
        )

    dgx = float(ordered_rows[0]["DGX H100 throughput (tokens/s)"])
    ax.axhline(dgx, color="black", linestyle="--", linewidth=1.4, label="DGX H100")
    ax.set_xticks(positions, labels)
    ax.tick_params(axis="x", length=0, pad=2)
    for center, label in group_centers:
        ax.text(
            center,
            -0.12,
            label,
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=9,
            fontweight="semibold",
        )
    ax.set_xlim(-0.26, group_stride * len(ARCHITECTURES) - 0.44)
    ax.set_ylim(0.0, max(float(row["System throughput (tokens/s)"]) for row in ordered_rows) * 1.27)
    ax.set_ylabel("System throughput (tokens/s)")
    ax.grid(axis="y", alpha=0.25)
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    memory_handles = [
        Patch(facecolor=MEMORY_COLORS[memory], label=memory) for memory in MEMORIES
    ]
    architecture_handles = [
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch=ARCHITECTURE_HATCHES[architecture],
            label=architecture,
        )
        for architecture in ARCHITECTURES
    ]
    fig.suptitle(
        "Llama2-7B, 4K: DGX H100 equal-power system throughput",
        y=0.98,
        fontsize=13,
    )
    fig.legend(
        handles=memory_handles + architecture_handles + [ax.lines[-1]],
        frameon=False,
        ncols=3,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.20, top=0.76)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = pd.DataFrame(ordered_rows)
    selected.to_csv(OUTPUT_DIR / "equal_power_system_throughput_7b_4k.csv", index=False)
    for suffix in ("png", "pdf"):
        fig.savefig(
            OUTPUT_DIR / f"equal_power_system_throughput_7b_4k.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)
    print(OUTPUT_DIR / "equal_power_system_throughput_7b_4k.png")

    if not SELECTED_POWER_SOURCE.exists():
        raise FileNotFoundError(
            f"missing selected-power component table: {SELECTED_POWER_SOURCE}"
        )
    power_rows = pd.read_csv(SELECTED_POWER_SOURCE)
    power_rows = power_rows[
        (power_rows["Model"] == "Llama2-7B") & (power_rows["Context"] == "4K")
    ].copy()
    if len(power_rows) != expected:
        raise ValueError(f"expected {expected} selected power rows, found {len(power_rows)}")
    ordered_power_rows: list[pd.Series] = []
    for architecture in ARCHITECTURES:
        for memory in MEMORIES:
            match = power_rows[
                (power_rows["Architecture"] == architecture)
                & (power_rows["Memory"] == memory)
            ]
            if len(match) != 1:
                raise ValueError(
                    f"expected one selected power row for {architecture}/{memory}, "
                    f"found {len(match)}"
                )
            ordered_power_rows.append(match.iloc[0])

    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    bottoms = [0.0] * len(ordered_power_rows)
    for component in ENERGY_COMPONENTS:
        values = [
            float(row[f"{component} power / device (W)"])
            for row in ordered_power_rows
        ]
        ax.bar(
            positions,
            values,
            bottom=bottoms,
            width=width,
            color=SYSTEM_ENERGY_COLORS[component],
            edgecolor="white",
            linewidth=0.45,
            label=component,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    ax.set_xticks(positions, labels)
    ax.tick_params(axis="x", length=0, pad=2)
    for center, label in group_centers:
        ax.text(
            center,
            -0.13,
            label,
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=9,
            fontweight="semibold",
        )
    ax.set_xlim(-0.26, group_stride * len(ARCHITECTURES) - 0.44)
    ax.set_ylim(0.0, max(bottoms) * 1.12)
    ax.set_ylabel("Average device power (W)")
    ax.grid(axis="y", alpha=0.25)
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle(
        "Llama2-7B, 4K context, device power breakdown",
        y=0.98,
        fontsize=13,
    )
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncols=5,
        frameon=False,
        fontsize=7.5,
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.20, top=0.73)
    selected_power = pd.DataFrame(ordered_power_rows)
    selected_power.to_csv(OUTPUT_DIR / "selected_power_per_device_7b_4k.csv", index=False)
    for suffix in ("png", "pdf"):
        fig.savefig(
            OUTPUT_DIR / f"selected_power_per_device_7b_4k.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)
    print(OUTPUT_DIR / "selected_power_per_device_7b_4k.png")

    if not BEST_TOKENS_PER_JOULE_SOURCE.exists():
        raise FileNotFoundError(
            f"missing best-Tokens/J table: {BEST_TOKENS_PER_JOULE_SOURCE}"
        )
    token_efficiency = pd.read_csv(BEST_TOKENS_PER_JOULE_SOURCE)
    contexts = ("4K", "32K", "128K")
    dgx = balanced.load_dgx(DGX_PROFILE)
    x = list(range(len(contexts)))
    for model in ("Llama2-7B", "Llama2-70B"):
        model_efficiency = token_efficiency[token_efficiency["Model"] == model]
        selected_rows: list[pd.Series] = []
        for context in contexts:
            for memory in MEMORIES:
                candidates = model_efficiency[
                    (model_efficiency["Context"] == context)
                    & (model_efficiency["Memory"] == memory)
                ]
                if len(candidates) != len(ARCHITECTURES):
                    raise ValueError(
                        f"expected Vector and Systolic best-Tokens/J rows for "
                        f"{model}/{context}/{memory}, found {len(candidates)}"
                    )
                # Exact ties are deterministic: preserve Vector's lower-complexity
                # order only if its Tokens/J is numerically identical to Systolic.
                selected_rows.append(
                    candidates.sort_values(
                        ["Tokens/J", "Architecture"], ascending=[False, True]
                    ).iloc[0]
                )
        best_token_efficiency = pd.DataFrame(selected_rows)
        best_token_efficiency["DGX H100 Tokens/J"] = best_token_efficiency.apply(
            lambda row: float(
                dgx[(str(row["Model"]), str(row["Context"]))]["throughput"]
            )
            / float(dgx[(str(row["Model"]), str(row["Context"]))]["power"]),
            axis=1,
        )
        best_token_efficiency["Tokens/J / DGX H100"] = (
            best_token_efficiency["Tokens/J"]
            / best_token_efficiency["DGX H100 Tokens/J"]
        )
        model_stem = model.replace("Llama2-", "").lower()
        best_token_efficiency.to_csv(
            OUTPUT_DIR / f"best_tokens_per_joule_relative_dgx_{model_stem}.csv",
            index=False,
        )

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for memory in MEMORIES:
            subset = (
                best_token_efficiency[best_token_efficiency["Memory"] == memory]
                .set_index("Context")
                .loc[list(contexts)]
            )
            ax.plot(
                x,
                subset["Tokens/J / DGX H100"],
                color=MEMORY_COLORS[memory],
                marker="o",
                linewidth=2.2,
                markersize=6,
                label=memory,
            )
        ax.axhline(
            1.0, color="black", linestyle="--", linewidth=1.2, label="DGX H100"
        )
        ax.set_xticks(x, contexts)
        ax.set_xlim(-0.12, len(contexts) - 0.88)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("Context")
        ax.set_ylabel("Token/J relative to DGX H100")
        ax.set_title(f"{model}: performance scaling with context")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, ncols=4, fontsize=8)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(
                OUTPUT_DIR / f"best_tokens_per_joule_relative_dgx_{model_stem}.{suffix}",
                dpi=220,
                bbox_inches="tight",
            )
        plt.close(fig)
        print(OUTPUT_DIR / f"best_tokens_per_joule_relative_dgx_{model_stem}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
