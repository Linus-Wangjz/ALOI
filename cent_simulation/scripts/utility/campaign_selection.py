"""Objective and equal-power selection shared by analysis campaigns."""

from __future__ import annotations

import math

import pandas as pd


def select_base_objective(candidates: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Select one capacity-admitted PP/TP/batch layout per workload.

    Ties intentionally prefer fewer replica devices, then smaller PP, TP, and
    batch.  Equal-power scaling is a later step and must not reselect a base
    layout.
    """

    rows = candidates.copy()
    if "Can host one batch" not in rows:
        rows["Can host one batch"] = rows["Can host one request"]
    if "Batch size" not in rows:
        rows["Batch size"] = 1
    admitted = rows[rows["Can host one batch"].astype(bool)]
    winners: list[dict[str, object]] = []
    for _, group in admitted.groupby(["Architecture", "Memory", "Model", "Context"]):
        best = float(group[metric].max())
        tied = group[
            group[metric].map(
                lambda value: math.isclose(
                    float(value), best, rel_tol=1.0e-12, abs_tol=1.0e-15
                )
            )
        ]
        winners.append(
            tied.sort_values(
                ["Replica devices", "PP", "TP", "Batch size"],
                ascending=[True, True, True, True],
            ).iloc[0].to_dict()
        )
    return pd.DataFrame(winners)


def build_all_equal_power_deployments(
    candidates: pd.DataFrame,
    dgx: dict[tuple[str, str], dict[str, float | int]],
    integer_dp_choices,
) -> pd.DataFrame:
    """Scale a preselected layout only by integer DP near DGX power."""

    deployments: list[dict[str, object]] = []
    admitted = candidates[candidates["Can host one batch"].astype(bool)]
    for _, candidate in admitted.iterrows():
        reference = dgx[(str(candidate["Model"]), str(candidate["Context"]))]
        target_power = float(reference["power"])
        replica_power = float(candidate["Replica power (W)"])
        local_rows: list[dict[str, object]] = []
        for rounding, dp in integer_dp_choices(target_power, replica_power):
            row = candidate.to_dict()
            system_power = replica_power * dp
            system_throughput = float(candidate["Replica throughput (tokens/s)"]) * dp
            row.update(
                {
                    "DP rounding": rounding,
                    "Continuous ideal DP": target_power / replica_power,
                    "DP": dp,
                    "Total devices": int(candidate["Replica devices"]) * dp,
                    "Resident requests in flight": (
                        int(candidate["Resident requests used"]) * dp
                    ),
                    "System throughput (tokens/s)": system_throughput,
                    "System power (W)": system_power,
                    "Power delta vs DGX (W)": system_power - target_power,
                    "Absolute power delta vs DGX (W)": abs(system_power - target_power),
                    "Power / DGX H100": system_power / target_power,
                    "Throughput / DGX H100": system_throughput
                    / float(reference["throughput"]),
                    "DGX H100 throughput (tokens/s)": float(reference["throughput"]),
                    "DGX H100 device-side power (W)": target_power,
                    "DGX H100 batch": int(reference["batch"]),
                    "DGX profile model": str(reference["profile_model"]),
                }
            )
            local_rows.append(row)
        closest = min(float(row["Absolute power delta vs DGX (W)"]) for row in local_rows)
        for row in local_rows:
            row["Candidate closest integer DP"] = math.isclose(
                float(row["Absolute power delta vs DGX (W)"]),
                closest,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            deployments.append(row)
    return pd.DataFrame(deployments)


def select_equal_power(deployments: pd.DataFrame) -> pd.DataFrame:
    """Select the nearest DP while proving PP/TP/batch stayed fixed."""

    rows = deployments.copy()
    if "Batch size" not in rows:
        rows["Batch size"] = 1
    winners: list[dict[str, object]] = []
    group_cols = ["Architecture", "Memory", "Model", "Context"]
    for group_key, group in rows.groupby(group_cols):
        base_layouts = group[["PP", "TP", "Batch size"]].drop_duplicates()
        if len(base_layouts) != 1:
            raise ValueError(
                "equal-power DP scaling received multiple PP/TP/batch bases for "
                f"{group_key}: {base_layouts.to_dict(orient='records')}"
            )
        closest = group[group["Candidate closest integer DP"].astype(bool)]
        if closest.empty:
            raise ValueError(f"no closest integer DP marked for {group_key}")
        winner = closest.sort_values(
            ["Absolute power delta vs DGX (W)", "DP"], ascending=[True, True]
        ).iloc[0].to_dict()
        winner["Selection rule"] = (
            "fixed_best_throughput_per_device_PP_TP_batch_then_nearest_integer_DP_"
            "then_smaller_DP"
        )
        winners.append(winner)
    selected = pd.DataFrame(winners)
    expected = len(rows[group_cols].drop_duplicates())
    if len(selected) != expected:
        raise ValueError(f"expected {expected} equal-power winners, found {len(selected)}")
    return selected
