"""Common five-metric campaign plots for one or more architectures."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd


def plot_grouped_metric(
    rows: pd.DataFrame,
    *,
    architectures: tuple[str, ...],
    memories: tuple[str, ...],
    contexts: Mapping[str, object],
    models: Mapping[str, object],
    memory_colors: Mapping[str, str],
    architecture_hatches: Mapping[str, str],
    metric: str,
    ylabel: str,
    title: str,
    output: Path,
    annotation: str | None = None,
    dgx_line: bool = False,
    dgx_reference: Mapping[tuple[str, str], float] | None = None,
    implicit_architecture: str | None = None,
    external_title_legend: bool = False,
) -> None:
    """Plot one campaign metric, producing a 7B and a 70B figure.

    Bars remain memory-colored and architecture-hatched. ``implicit_architecture``
    lets a single-architecture campaign preserve its original candidate schema
    without adding an Architecture column just for plotting. This is used for the
    three scalar metrics in the standard five-plot bundle; system-energy bars
    use ``system_energy_breakdown.plot_selected_component_breakdown`` instead.
    """

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if "Architecture" not in rows:
        if implicit_architecture is None:
            raise ValueError("rows without Architecture require implicit_architecture")
        if architectures != (implicit_architecture,):
            raise ValueError("implicit architecture must be the only plotted architecture")
    elif implicit_architecture is not None:
        raise ValueError("implicit_architecture is only valid without Architecture")
    plt.style.use("seaborn-v0_8-whitegrid")
    series = [(arch, memory) for arch in architectures for memory in memories]
    width = min(0.24, 0.75 / max(len(series), 1))
    offsets = [(index - (len(series) - 1) / 2) * width for index in range(len(series))]
    for model in models:
        model_rows = rows[rows["Model"] == model]
        positions = list(range(len(contexts)))
        fig, ax = plt.subplots(figsize=(10.8, 5.2))
        reference = None
        if dgx_line:
            if dgx_reference is None:
                reference = (
                    model_rows.drop_duplicates("Context")
                    .set_index("Context")
                    .loc[list(contexts), "DGX H100 throughput (tokens/s)"]
                )
            else:
                reference = pd.Series(
                    {context: dgx_reference[(model, context)] for context in contexts}
                )
        for offset, (architecture, memory) in zip(offsets, series):
            mask = model_rows["Memory"] == memory
            if "Architecture" in model_rows:
                mask &= model_rows["Architecture"] == architecture
            subset = model_rows[mask].set_index("Context").loc[list(contexts)]
            bars = ax.bar(
                [position + offset for position in positions],
                subset[metric],
                width=width,
                color=memory_colors[memory],
                hatch=architecture_hatches[architecture],
                edgecolor="black" if architecture_hatches[architecture] else "none",
                linewidth=0.55,
            )
            if annotation:
                for bar, (context, row) in zip(bars, subset.iterrows()):
                    batch_size = int(row.get("Batch size", 1))
                    if annotation == "equal_power":
                        label = (
                            f"{float(row['Throughput / DGX H100']):.2f}x\n"
                            f"{int(row['PP'])}/{int(row['TP'])}/"
                            f"B{batch_size}/{int(row['DP'])}"
                        )
                    elif annotation == "tokens_per_joule":
                        if reference is None:
                            raise ValueError("Tokens/J annotations require a DGX reference")
                        label = (
                            f"{float(row[metric]) / float(reference.loc[context]):.2f}x\n"
                            f"{int(row['PP'])}/{int(row['TP'])}/"
                            f"B{batch_size}"
                        )
                    else:
                        label = (
                            f"{int(row['PP'])}/{int(row['TP'])}/"
                            f"B{batch_size}"
                        )
                    ax.annotate(
                        label,
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=6.4,
                        rotation=90,
                        linespacing=0.9,
                    )
        if dgx_line:
            ax.plot(positions, reference, "k--o", linewidth=1.4, markersize=4, label="DGX H100")
        ax.set_xticks(positions, list(contexts))
        ax.set_ylabel(ylabel)
        if not external_title_legend:
            ax.set_title(f"{model}: {title}")
        ax.grid(axis="y", alpha=0.25)
        memory_handles = [Patch(facecolor=memory_colors[memory], label=memory) for memory in memories]
        architecture_handles = [
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch=architecture_hatches[architecture],
                label=architecture,
            )
            for architecture in architectures
        ]
        handles = memory_handles + architecture_handles
        if dgx_line:
            handles.append(ax.lines[-1])
        if external_title_legend:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.suptitle(f"{model}: {title}", y=0.98, fontsize=13)
            fig.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.93),
                ncols=3,
                frameon=False,
                fontsize=8,
            )
            fig.subplots_adjust(left=0.10, right=0.99, bottom=0.16, top=0.82)
        else:
            ax.legend(handles=handles, frameon=False, ncols=3, fontsize=8)
        top = max(float(model_rows[metric].max()), 1.0)
        if dgx_line:
            top = max(top, float(reference.max()))
        ax.set_ylim(0.0, top * (1.36 if annotation else 1.18))
        if not external_title_legend:
            fig.tight_layout()
        stem = output.parent / f"{output.name}_{model.replace('Llama2-', '').lower()}"
        for suffix in ("png", "pdf"):
            fig.savefig(stem.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
        plt.close(fig)
