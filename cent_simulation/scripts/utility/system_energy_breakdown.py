"""CENT system-energy reconstruction and hierarchical breakdown plotting."""

from __future__ import annotations

from collections import OrderedDict
import math
from pathlib import Path
from typing import Mapping

import pandas as pd

import run_sim as simulator
from tp_mapping import (
    KVHeadTPLayout,
    SystolicTPLayout,
    kv_head_tp_shape,
    pim_precision_geometry,
)
from utils import embedding_size, ffn_size, gqa_factor, n_heads


# This is the same Vega/Altair-style system energy palette as
# plot_cent_energy_breakdown.py, kept locally so campaign post-processing does
# not import a CLI plotting script.
SYSTEM_ENERGY_COLORS = {
    "ACT/PRE": "#E45756",
    "ACT": "#E45756",
    "PRE": "#FF9DA6",
    "RD": "#4C78A8",
    "WR": "#F58518",
    "PIM": "#54A24B",
    "ACT_STBY": "#72B7B2",
    "PRE_STBY": "#B279A2",
    "DQ_IO": "#FF9DA6",
    "CTRL_PHY": "#9C755F",
    "SRAM_STT": "#8CD17D",
    "ACCEL_STT": "#499894",
    "SRAM_DYN": "#59A14F",
    "ACCEL_DYN": "#AF7AA1",
    "PCIe": "#BAB0AC",
    "Trace-external waiting": "#D0D0D0",
    "Pipeline-bubble waiting": "#8E8E8E",
}

SYSTEM_ENERGY_GROUPS = OrderedDict(
    [
        ("ACT/PRE", ("ACT/PRE",)),
        ("RD", ("RD",)),
        ("WR", ("WR",)),
        ("PIM", ("PIM",)),
        ("ACT_STBY", ("ACT_STBY",)),
        ("PRE_STBY", ("PRE_STBY",)),
        ("DQ_IO", ("DQ",)),
        ("CTRL_PHY", ("MEM_CTR",)),
        ("SRAM_STT", ("GB_STT", "SB_STT", "IB_STT")),
        (
            "ACCEL_STT",
            ("RED_STT", "EXP_STT", "VEC_ADD_STT", "VEC_MUL_STT", "CTR_STT", "TOPK_STT"),
        ),
        ("SRAM_DYN", ("GB_RD", "GB_WR", "SB_DYN", "IB_DYN")),
        (
            "ACCEL_DYN",
            (
                "RV_DYN",
                "RED_DYN",
                "EXP_DYN",
                "VEC_ADD_DYN",
                "VEC_MUL_DYN",
                "DV_CTR",
            ),
        ),
        ("PCIe", ("PCIe",)),
    ]
)
WAITING_COMPONENTS = ("Trace-external waiting", "Pipeline-bubble waiting")
ENERGY_COMPONENTS = tuple(SYSTEM_ENERGY_GROUPS) + WAITING_COMPONENTS


def component_breakdown_figure_width(bar_count: int) -> float:
    """Size a stacked-breakdown canvas to its actual number of bars.

    The previous 12.8-inch width targeted the largest campaign (18 bars).
    It made one-bar FP8 plots unnecessarily wide and shrank their labels.
    """

    return max(6.4, 5.6 + 0.4 * bar_count)


def component_breakdown_bar_width(bars_per_context: int) -> float:
    """Keep a one-bar component stack visually proportional to its group."""

    return min(0.20, 0.12 + 0.02 * bars_per_context)


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def resolve_source_csv(value: object, project_root: Path) -> Path:
    """Resolve the project-relative raw CSV saved in candidate provenance."""

    path = Path(str(value))
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"missing component-energy source CSV: {path}")
    return path


def load_selected_source_row(
    candidate: pd.Series,
    raw_cache: dict[Path, pd.DataFrame],
    project_root: Path,
) -> tuple[Path, pd.Series]:
    """Find the exact source simulation row for one selected layout."""

    source_path = resolve_source_csv(candidate["Source CSV"], project_root)
    if source_path not in raw_cache:
        raw_cache[source_path] = pd.read_csv(source_path)
    raw = raw_cache[source_path]
    mask = (
        (raw["Model"] == candidate["Model"])
        & (raw["Sequence length"].astype(int) == int(candidate["Active sequence length"]))
        & (raw["Pipeline parallelism"].astype(int) == int(candidate["Source PP"]))
        & (raw["Tensor parallelism"].astype(int) == int(candidate["Source TP"]))
    )
    if "Batch size" in raw:
        mask &= raw["Batch size"].astype(int) == int(candidate.get("Batch size", 1))
    matches = raw[mask]
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one raw source row for "
            f"{candidate.get('Architecture', 'single_architecture')}/{candidate['Memory']}/"
            f"{candidate['Model']}/{candidate['Context']} (found {len(matches)})"
        )
    return source_path, matches.iloc[0]


def source_log_root(source_path: Path, memory: str) -> Path:
    suffix = {
        "GDDR6": "",
        "LPDDR4X_nCCD2": "_nCCD2",
        "LPDDR4X_nCCD6": "_nCCD6",
    }[memory]
    return source_path.parent / f"ramulator_long_context_midpoint{suffix}"


def _energy_args(*, batch_size: int, systolic: bool, systolic_dim: int):
    return type(
        "EnergyArgs",
        (),
        {
            "batch_size": batch_size,
            "systolic_pim": systolic,
            "systolic_dim": systolic_dim,
        },
    )()


def _kv_head_energy(
    source: pd.Series,
    main_stat: dict,
    main_log: Path,
    *,
    model: str,
    tp: int,
    batch_size: int,
    seqlen: int,
    dram_impl: str,
    source_pp: int,
) -> dict[str, float]:
    """Rebuild run_sim's symmetric-rank KV-head TP energy aggregation."""

    is_systolic = _bool(source.get("Systolic pim", False))
    geometry = pim_precision_geometry(str(source.get("Precision", "BF16")))
    systolic_dim = int(source.get("Systolic dim", 1))
    channel_count = int(source["Channels per device"])
    local_heads = n_heads[model] // tp
    local_hidden = embedding_size[model] // tp
    main_pcie, helper_pcie, _system_pcie = simulator.kv_head_tp_pcie_bits(
        embedding_size[model] * batch_size, tp
    )
    if source_pp > 1:
        main_pcie += embedding_size[model] * batch_size * 16
    calc_args = dict(
        dram_power_impl=dram_impl,
        dram_energy_model=str(source["DRAM energy model"]),
        command_trace_prefix=simulator.command_trace_prefix_for_log(main_log),
        rmsnorm_hidden_dim=embedding_size[model],
    )
    main_energy, _ = simulator.power_calculator(
        main_stat,
        main_pcie,
        local_heads,
        local_hidden,
        seqlen,
        gqa_factor[model],
        **calc_args,
    )
    shape = kv_head_tp_shape(
        dim=embedding_size[model],
        query_heads=n_heads[model],
        kv_heads=n_heads[model] // gqa_factor[model],
        ffn_dim=ffn_size[model],
        tp=tp,
    )
    if is_systolic:
        layout = SystolicTPLayout(
            shape=shape,
            num_channels=channel_count,
            banks_per_channel=int(source["Banks per device"]),
            max_seq_len=int(source["Context window"]),
            systolic_height=systolic_dim,
            dram_columns=geometry["dram_columns"],
            burst_length=geometry["burst_length"],
        )
        reduction_adds = sum(layout.pnm_reduction_adds(seqlen).values())
        ewmul_enabled = _bool(source.get("EWMUL PNM effective", False))
    else:
        layout = KVHeadTPLayout(
            shape=shape,
            num_channels=channel_count,
            banks_per_channel=int(source["Banks per device"]),
            max_seq_len=int(source["Context window"]),
            dram_columns=geometry["dram_columns"],
            burst_length=geometry["burst_length"],
            k_contexts_per_row=geometry["dram_columns"] // shape.head_dim,
        )
        reduction_adds = layout.v_reduction_adds(seqlen)
        ewmul_enabled = _bool(source.get("EWMUL PNM effective", False))
    ewmul_elements = 0
    if ewmul_enabled:
        ewmul_elements = (
            6 * embedding_size[model]
            + local_hidden
            + local_hidden // gqa_factor[model]
            + 2 * layout.shape.local_ffn_dim
        )
    pnm_energy = simulator.kv_head_tp_pnm_dynamic_energy(
        main_stat,
        reduction_adds=reduction_adds,
        ewmul_elements=ewmul_elements,
    )
    main_energy = simulator.add_energy_terms(main_energy, pnm_energy)
    main_energy = simulator.adjust_systolic_energy(
        main_energy,
        _energy_args(
            batch_size=batch_size, systolic=is_systolic, systolic_dim=systolic_dim
        ),
    )
    if tp == 1:
        return main_energy
    # Both legacy Vector KV-head TP and the current systolic campaign reuse
    # one symmetric trace.  run_sim evaluates a helper with its own PCIe share
    # and otherwise the same local operation model.
    helper_energy, _ = simulator.power_calculator(
        main_stat,
        helper_pcie,
        local_heads,
        local_hidden,
        seqlen,
        gqa_factor[model],
        **calc_args,
    )
    helper_energy = simulator.add_energy_terms(helper_energy, pnm_energy)
    helper_energy = simulator.adjust_systolic_energy(
        helper_energy,
        _energy_args(
            batch_size=batch_size, systolic=is_systolic, systolic_dim=systolic_dim
        ),
    )
    return {
        component: main_energy[component] + helper_energy[component] * (tp - 1)
        for component in main_energy
    }


def rebuild_physical_energy(
    candidate: pd.Series,
    source_path: Path,
    source: pd.Series,
    *,
    memory_cases: Mapping[str, Mapping[str, object]],
    model_specs: Mapping[str, Mapping[str, int]],
    project_root: Path,
) -> tuple[dict[str, float], float, float, str]:
    """Rebuild full-model per-token physical energy from existing trace logs."""

    model = str(candidate["Model"])
    memory = str(candidate["Memory"])
    tp = int(source["Tensor parallelism"])
    batch_size = int(source.get("Batch size", 1))
    source_pp = int(source["Pipeline parallelism"])
    seqlen = int(source["Sequence length"])
    channel_count = int(source["Channels per device"])
    parallel_sram = int(source.get("Parallel SRAM banks", 32))
    simulator.set_channel_count(channel_count, parallel_sram)
    dram_impl = str(memory_cases[memory]["dram_impl"])
    log_root = source_log_root(source_path, memory)
    is_systolic = _bool(source.get("Systolic pim", False))
    is_kv_head = str(source["Attention mapping"]) == "kv_head"

    if is_systolic:
        variant = (
            f"systolic_pim_{int(source['Systolic dim'])}_batch_size_{batch_size}"
            f"_ewmul_pnm_{int(_bool(source.get('EWMUL PNM effective', False)))}"
        )
        if _bool(source.get("Flash attention", False)):
            variant += f"_flash_{int(source['Flash attention block size'])}"
        if str(source.get("Precision", "BF16")) == "FP8":
            variant += "_fp8"
        main_log = (
            log_root / variant / "model_parallel_kv_head_main" / model
            / f"trace_{tp}_FC_devices_seqlen_{seqlen}.txt.log"
        )
    elif is_kv_head:
        main_log = (
            log_root / "model_parallel_kv_head_main" / model
            / f"trace_{tp}_FC_devices_seqlen_{seqlen}.txt.log"
        )
    else:
        main_log = (
            log_root / "model_parallel" / model
            / f"trace_{tp}_FC_devices_seqlen_{seqlen}.txt.log"
        )
    if not main_log.exists() or main_log.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty source main log: {main_log}")
    main_stat = simulator.command_processor(str(main_log))
    helper_log_text = ""

    if is_kv_head:
        full_energy = _kv_head_energy(
            source,
            main_stat,
            main_log,
            model=model,
            tp=tp,
            batch_size=batch_size,
            seqlen=seqlen,
            dram_impl=dram_impl,
            source_pp=source_pp,
        )
    else:
        pcie_bits = (embedding_size[model] * 10 + ffn_size[model] * 2) * batch_size
        calc_args = dict(
            dram_power_impl=dram_impl,
            dram_energy_model=str(source["DRAM energy model"]),
            command_trace_prefix=simulator.command_trace_prefix_for_log(main_log),
        )
        main_energy, _ = simulator.power_calculator(
            main_stat, pcie_bits, n_heads[model], embedding_size[model], seqlen,
            gqa_factor[model], **calc_args
        )
        # The non-KV fallback remains available for generic campaign reuse.
        main_energy = simulator.adjust_systolic_energy(
            main_energy, _energy_args(batch_size=batch_size, systolic=False, systolic_dim=1)
        )
        if tp == 1:
            full_energy = main_energy
        else:
            helper_mode = (
                "model_parallel_helper_attention"
                if str(source["Attention mapping"]) == "inter_device"
                else "model_parallel_FC"
            )
            helper_log = (
                log_root / helper_mode / model
                / f"trace_{tp}_FC_devices_seqlen_{seqlen}.txt.log"
            )
            if not helper_log.exists() or helper_log.stat().st_size == 0:
                raise FileNotFoundError(f"missing or empty source helper log: {helper_log}")
            helper_log_text = _display_path(helper_log, project_root)
            helper_energy, _ = simulator.power_calculator(
                simulator.command_processor(str(helper_log)),
                pcie_bits,
                n_heads[model],
                embedding_size[model],
                seqlen,
                gqa_factor[model],
                dram_power_impl=dram_impl,
                dram_energy_model=str(source["DRAM energy model"]),
                command_trace_prefix=simulator.command_trace_prefix_for_log(helper_log),
                device_role=(
                    "inter_device_helper"
                    if str(source["Attention mapping"]) == "inter_device"
                    else "fc_helper"
                ),
            )
            helper_energy = simulator.adjust_systolic_energy(
                helper_energy,
                _energy_args(batch_size=batch_size, systolic=False, systolic_dim=1),
            )
            full_energy = {
                component: main_energy[component] + helper_energy[component] * (tp - 1)
                for component in main_energy
            }

    layers = int(model_specs[model]["layers"])
    per_token = {
        component: float(value) * layers / batch_size
        for component, value in full_energy.items()
    }
    main_log_text = _display_path(main_log, project_root)
    logs = main_log_text if not helper_log_text else f"{main_log_text}; {helper_log_text}"
    return per_token, float(source["Token energy (mJ)"]), float(sum(per_token.values())), logs


def group_system_energy(energy: Mapping[str, float]) -> dict[str, float]:
    """Group Cellar physical terms into the CENT system-energy legend."""

    grouped = {
        component: sum(float(energy.get(term, 0.0)) for term in terms)
        for component, terms in SYSTEM_ENERGY_GROUPS.items()
    }
    accounted_terms = {term for terms in SYSTEM_ENERGY_GROUPS.values() for term in terms}
    ungrouped = {
        component: float(value)
        for component, value in energy.items()
        if component not in accounted_terms and not math.isclose(float(value), 0.0, abs_tol=1e-12)
    }
    if ungrouped:
        raise ValueError(f"unmapped source energy components: {ungrouped}")
    return grouped


def build_selected_component_breakdown(
    selected: pd.DataFrame,
    *,
    memory_cases: Mapping[str, Mapping[str, object]],
    model_specs: Mapping[str, Mapping[str, int]],
    project_root: Path,
) -> pd.DataFrame:
    """Build physical and waiting components for effective-energy selections."""

    rows: list[dict[str, object]] = []
    raw_cache: dict[Path, pd.DataFrame] = {}
    for _, candidate in selected.iterrows():
        source_path, source = load_selected_source_row(candidate, raw_cache, project_root)
        raw_energy, source_total, reconstructed_total, source_logs = rebuild_physical_energy(
            candidate,
            source_path,
            source,
            memory_cases=memory_cases,
            model_specs=model_specs,
            project_root=project_root,
        )
        grouped = group_system_energy(raw_energy)
        calibration = source_total - reconstructed_total
        # The read-only Vector source CSVs precede the current Cellar PIM
        # accounting.  Selected legacy rows drift by at most 2.3%; permit a
        # small provenance calibration but reject a materially different log.
        tolerance = max(0.02, source_total * 0.025)
        if abs(calibration) > tolerance:
            raise ValueError(
                f"source-energy reconstruction mismatch for {source_path}: "
                f"CSV={source_total:.12g} mJ, logs={reconstructed_total:.12g} mJ"
            )
        grouped["PIM"] += calibration
        if grouped["PIM"] < 0.0:
            raise ValueError("PIM source-accounting calibration made PIM energy negative")
        removed_value = candidate.get("Removed PP handoff energy (mJ/token)", 0.0)
        removed_pp_handoff = 0.0 if pd.isna(removed_value) else float(removed_value)
        grouped["PCIe"] = max(0.0, grouped["PCIe"] - removed_pp_handoff)
        grouped["Trace-external waiting"] = float(
            candidate["Trace-external gap energy (mJ/token)"]
        )
        grouped["Pipeline-bubble waiting"] = max(
            0.0,
            float(candidate["Effective token energy (mJ)"])
            - float(candidate["Work token energy (mJ)"]),
        )
        effective_energy = float(candidate["Effective token energy (mJ)"])
        reconstructed_effective = sum(grouped.values())
        if not math.isclose(
            reconstructed_effective, effective_energy, rel_tol=1.0e-10, abs_tol=1.0e-8
        ):
            raise ValueError(
                "selected component energy does not reconstruct effective token energy: "
                f"{candidate.get('Architecture', 'single_architecture')}/{candidate['Memory']}/"
                f"{candidate['Model']}/{candidate['Context']} "
                f"({reconstructed_effective} vs {effective_energy})"
            )
        replica_throughput = float(candidate["Replica throughput (tokens/s)"])
        replica_devices = float(candidate["Replica devices"])
        powers = {
            component: energy * replica_throughput / 1000.0 / replica_devices
            for component, energy in grouped.items()
        }
        reconstructed_power = sum(powers.values())
        expected_power = float(candidate["Power / device (W)"])
        if not math.isclose(
            reconstructed_power, expected_power, rel_tol=1.0e-10, abs_tol=1.0e-8
        ):
            raise ValueError(
                "selected component power does not reconstruct per-device power: "
                f"{reconstructed_power} vs {expected_power}"
            )
        row = candidate.to_dict()
        row.update(
            {
                "Component source CSV": _display_path(source_path, project_root),
                "Component source main/helper log": source_logs,
                "Raw source token energy (mJ)": source_total,
                "Raw-log reconstructed token energy (mJ)": reconstructed_total,
                "PIM source-accounting calibration (mJ/token)": calibration,
                "Component effective energy (mJ/token)": reconstructed_effective,
                "Component power / device (W)": reconstructed_power,
            }
        )
        for component in ENERGY_COMPONENTS:
            row[f"{component} energy (mJ/token)"] = grouped[component]
            row[f"{component} power / device (W)"] = powers[component]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_selected_component_breakdown(
    breakdown: pd.DataFrame,
    *,
    architectures: tuple[str, ...],
    memories: tuple[str, ...],
    contexts: Mapping[str, object],
    models: Mapping[str, object],
    metric_suffix: str,
    ylabel: str,
    title: str,
    output: Path,
    implicit_architecture: str | None = None,
    external_title_legend: bool = False,
) -> None:
    """Draw context → architecture → memory hierarchical stacked bars.

    A single-architecture table may omit its Architecture column and use
    ``implicit_architecture`` so the source candidates remain unmodified.
    """

    import matplotlib.pyplot as plt

    if "Architecture" not in breakdown:
        if implicit_architecture is None:
            raise ValueError("breakdown without Architecture requires implicit_architecture")
        if architectures != (implicit_architecture,):
            raise ValueError("implicit architecture must be the only plotted architecture")
    elif implicit_architecture is not None:
        raise ValueError("implicit_architecture is only valid without Architecture")
    short_memory = {"GDDR6": "G6", "LPDDR4X_nCCD2": "X2", "LPDDR4X_nCCD6": "X6"}
    plt.style.use("seaborn-v0_8-whitegrid")
    memory_step, architecture_stride, context_stride = 0.24, 1.02, 2.65
    bar_width = component_breakdown_bar_width(len(architectures) * len(memories))
    for model in models:
        model_rows = breakdown[breakdown["Model"] == model]
        x_positions: list[float] = []
        labels: list[str] = []
        ordered_rows: list[pd.Series] = []
        architecture_centers: list[tuple[float, str]] = []
        context_centers: list[tuple[float, str]] = []
        for context_index, context in enumerate(contexts):
            context_base = context_index * context_stride
            for architecture_index, architecture in enumerate(architectures):
                architecture_base = context_base + architecture_index * architecture_stride
                architecture_centers.append(
                    (
                        architecture_base
                        + (len(memories) - 1) * memory_step / 2,
                        architecture.split(" 4x", 1)[0],
                    )
                )
                for memory_index, memory in enumerate(memories):
                    mask = (model_rows["Context"] == context) & (
                        model_rows["Memory"] == memory
                    )
                    if "Architecture" in model_rows:
                        mask &= model_rows["Architecture"] == architecture
                    match = model_rows[mask]
                    if len(match) != 1:
                        raise ValueError(
                            f"expected one selected component row for {model}/{context}/"
                            f"{architecture}/{memory}, found {len(match)}"
                        )
                    x_positions.append(architecture_base + memory_index * memory_step)
                    labels.append(short_memory[memory])
                    ordered_rows.append(match.iloc[0])
            context_centers.append(
                (
                    context_base
                    + (
                        (len(architectures) - 1) * architecture_stride
                        + (len(memories) - 1) * memory_step
                    )
                    / 2,
                    context,
                )
            )
        figure_width = component_breakdown_figure_width(len(ordered_rows))
        fig, ax = plt.subplots(figsize=(figure_width, 6.8))
        bottoms = [0.0] * len(ordered_rows)
        for component in ENERGY_COMPONENTS:
            values = [float(row[f"{component} {metric_suffix}"]) for row in ordered_rows]
            ax.bar(
                x_positions, values, bottom=bottoms, width=bar_width, label=component,
                color=SYSTEM_ENERGY_COLORS[component], edgecolor="white", linewidth=0.45,
            )
            bottoms = [base + value for base, value in zip(bottoms, values)]
        ax.set_xticks(x_positions, labels, fontsize=8)
        ax.tick_params(axis="x", length=0, pad=2)
        horizontal_margin = max(0.25, bar_width * 1.4)
        ax.set_xlim(
            min(x_positions) - horizontal_margin,
            max(x_positions) + horizontal_margin,
        )
        for center, label in architecture_centers:
            ax.text(center, -0.105, label, ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=8)
        for center, label in context_centers:
            ax.text(center, -0.205, label, ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=9, fontweight="semibold")
        ax.set_ylabel(ylabel)
        if not external_title_legend:
            ax.set_title(f"{model}: {title}")
        ax.grid(axis="y", alpha=0.25)
        ax.xaxis.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        handles, legend_labels = ax.get_legend_handles_labels()
        legend_cols = 3 if figure_width < 8.0 else 5
        legend_rows = math.ceil(len(ENERGY_COMPONENTS) / legend_cols)
        if external_title_legend:
            fig.suptitle(f"{model}: {title}", y=0.98, fontsize=13)
            fig.legend(
                handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.91),
                ncols=legend_cols,
                frameon=False,
                fontsize=8,
            )
            fig.subplots_adjust(
                left=0.08,
                right=0.99,
                bottom=0.25,
                top=max(0.54, 0.96 - 0.05 * legend_rows),
            )
        else:
            fig.legend(
                handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.98),
                ncols=legend_cols,
                frameon=False,
                fontsize=8,
            )
            fig.subplots_adjust(
                left=0.08,
                right=0.99,
                bottom=0.25,
                top=max(0.54, 0.92 - 0.05 * legend_rows),
            )
        stem = output.parent / f"{output.name}_{model.replace('Llama2-', '').lower()}"
        for suffix in ("png", "pdf"):
            fig.savefig(stem.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_selected_component_breakdown_context_subplots(
    breakdown: pd.DataFrame,
    *,
    architectures: tuple[str, ...],
    memories: tuple[str, ...],
    contexts: Mapping[str, object],
    models: Mapping[str, object],
    metric_suffix: str,
    ylabel: str,
    title: str,
    output: Path,
    external_title_legend: bool = False,
) -> None:
    """Draw one independent-y stacked-bar subplot per context.

    The bar hierarchy inside every panel remains architecture → memory.  This
    is useful when the three contexts differ by orders of magnitude and a
    common y-axis would obscure the low-energy cases.
    """

    import matplotlib.pyplot as plt

    short_memory = {"GDDR6": "G6", "LPDDR4X_nCCD2": "X2", "LPDDR4X_nCCD6": "X6"}
    plt.style.use("seaborn-v0_8-whitegrid")
    bar_width, memory_step, architecture_stride = 0.20, 0.24, 1.02
    context_names = list(contexts)
    for model in models:
        model_rows = breakdown[breakdown["Model"] == model]
        fig, axes = plt.subplots(
            1,
            len(context_names),
            figsize=(5.1 * len(context_names), 5.8),
            sharey=False,
            squeeze=False,
        )
        legend_handles = None
        legend_labels = None
        for ax, context in zip(axes[0], context_names):
            x_positions: list[float] = []
            labels: list[str] = []
            ordered_rows: list[pd.Series] = []
            architecture_centers: list[tuple[float, str]] = []
            for architecture_index, architecture in enumerate(architectures):
                architecture_base = architecture_index * architecture_stride
                architecture_centers.append(
                    (architecture_base + memory_step, architecture.split(" 4x", 1)[0])
                )
                for memory_index, memory in enumerate(memories):
                    match = model_rows[
                        (model_rows["Context"] == context)
                        & (model_rows["Architecture"] == architecture)
                        & (model_rows["Memory"] == memory)
                    ]
                    if len(match) != 1:
                        raise ValueError(
                            f"expected one selected component row for {model}/{context}/"
                            f"{architecture}/{memory}, found {len(match)}"
                        )
                    x_positions.append(architecture_base + memory_index * memory_step)
                    labels.append(short_memory[memory])
                    ordered_rows.append(match.iloc[0])
            bottoms = [0.0] * len(ordered_rows)
            for component in ENERGY_COMPONENTS:
                values = [float(row[f"{component} {metric_suffix}"]) for row in ordered_rows]
                ax.bar(
                    x_positions,
                    values,
                    bottom=bottoms,
                    width=bar_width,
                    label=component,
                    color=SYSTEM_ENERGY_COLORS[component],
                    edgecolor="white",
                    linewidth=0.45,
                )
                bottoms = [base + value for base, value in zip(bottoms, values)]
            ax.set_xticks(x_positions, labels, fontsize=8)
            ax.tick_params(axis="x", length=0, pad=2)
            for center, label in architecture_centers:
                ax.text(
                    center,
                    -0.13,
                    label,
                    ha="center",
                    va="top",
                    transform=ax.get_xaxis_transform(),
                    fontsize=8,
                )
            panel_max = max(sum(float(row[f"{component} {metric_suffix}"]) for component in ENERGY_COMPONENTS) for row in ordered_rows)
            ax.set_ylim(0.0, max(panel_max, 1.0) * 1.12)
            ax.set_xlim(-0.23, architecture_stride * len(architectures) - 0.31)
            ax.set_ylabel(ylabel)
            ax.set_title(context, fontsize=10, fontweight="semibold")
            ax.grid(axis="y", alpha=0.25)
            ax.xaxis.grid(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
        if external_title_legend:
            fig.suptitle(f"{model}: {title}", y=0.98, fontsize=13)
            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.91),
                ncols=5,
                frameon=False,
                fontsize=8,
            )
            fig.subplots_adjust(
                left=0.06, right=0.99, bottom=0.25, top=0.78, wspace=0.28
            )
        else:
            fig.suptitle(f"{model}: {title}", y=0.75, fontsize=12)
            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.98),
                ncols=5,
                frameon=False,
                fontsize=8,
            )
            fig.subplots_adjust(
                left=0.06, right=0.99, bottom=0.25, top=0.60, wspace=0.28
            )
        stem = output.parent / f"{output.name}_{model.replace('Llama2-', '').lower()}"
        for suffix in ("png", "pdf"):
            fig.savefig(stem.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
        plt.close(fig)
