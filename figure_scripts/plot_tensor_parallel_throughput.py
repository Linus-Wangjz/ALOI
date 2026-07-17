import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt


MODELS = ["Llama2-7B", "Llama2-13B", "Llama2-70B"]
MEMORY_CONFIGS = [
    {
        "name": "LPDDR4",
        "banks": 8,
        "path": "cent_simulation/simulation_results_32_channels_8_banks_per_device.csv",
        "color": "#4C78A8",
    },
    {
        "name": "GDDR6",
        "banks": 16,
        "path": "cent_simulation/simulation_results_32_channels_16_banks_per_device.csv",
        "color": "#F58518",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot max-pipeline-parallel token throughput for Llama2 models."
    )
    parser.add_argument("--seqlens", type=int, nargs="+", default=[1024, 2048, 3072, 4096])
    parser.add_argument("--lpddr4-csv", default=MEMORY_CONFIGS[0]["path"])
    parser.add_argument("--gddr6-csv", default=MEMORY_CONFIGS[1]["path"])
    parser.add_argument(
        "--output",
        default="figures/llama_max_pipeline_throughput.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--csv-output",
        default="figure_source_data/llama_max_pipeline_throughput.csv",
        help="Output source data path.",
    )
    return parser.parse_args()


def to_int(row, key):
    return int(float(row[key]))


def to_float(row, key):
    return float(row[key])


def load_input_rows(config, seqlens):
    path = config["path"]
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    rows = []
    with open(path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row["Model"] not in MODELS:
                continue
            if to_int(row, "Sequence length") not in seqlens:
                continue
            if to_int(row, "Channels per device") != 32:
                continue
            if "Banks per device" in row and row["Banks per device"]:
                if to_int(row, "Banks per device") != config["banks"]:
                    continue
            rows.append(
                {
                    "Memory": config["name"],
                    "Model": row["Model"],
                    "Sequence length": to_int(row, "Sequence length"),
                    "Pipeline parallelism": to_int(row, "Pipeline parallelism"),
                    "Tensor parallelism": to_int(row, "Tensor parallelism"),
                    "Token latency (ms)": to_float(row, "Token latency (ms)"),
                    "Throughput (tokens/s)": to_float(row, "Throughput (tokens/s)"),
                }
            )
    return rows


def geomean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values))


def context_label(seqlen):
    if seqlen % 1024 == 0:
        return f"{seqlen // 1024}K"
    return str(seqlen)


def select_max_pipeline_rows(rows, seqlens):
    selected_rows = []
    by_memory_model = defaultdict(list)
    for row in rows:
        by_memory_model[(row["Memory"], row["Model"])].append(row)

    for (memory, model), model_rows in by_memory_model.items():
        max_pp = max(row["Pipeline parallelism"] for row in model_rows)
        rows_by_seqlen = {
            row["Sequence length"]: row
            for row in model_rows
            if row["Pipeline parallelism"] == max_pp
        }
        throughput_values = []
        for seqlen in seqlens:
            row = rows_by_seqlen.get(seqlen)
            if row is None:
                raise ValueError(f"Missing {memory} {model} PP={max_pp} seqlen={seqlen}")
            row = dict(row)
            row["Context length"] = context_label(seqlen)
            row["Is geomean"] = False
            selected_rows.append(row)
            throughput_values.append(row["Throughput (tokens/s)"])

        selected_rows.append(
            {
                "Memory": memory,
                "Model": model,
                "Context length": "Geomean",
                "Sequence length": "",
                "Pipeline parallelism": max_pp,
                "Tensor parallelism": rows_by_seqlen[seqlens[0]]["Tensor parallelism"],
                "Token latency (ms)": "",
                "Throughput (tokens/s)": geomean(throughput_values),
                "Is geomean": True,
            }
        )

    return selected_rows


def write_source_data(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "Memory",
        "Model",
        "Context length",
        "Sequence length",
        "Pipeline parallelism",
        "Tensor parallelism",
        "Token latency (ms)",
        "Throughput (tokens/s)",
        "Is geomean",
    ]
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, output_path, seqlens):
    labels = [context_label(seqlen) for seqlen in seqlens] + ["Geomean"]
    by_model = defaultdict(dict)
    for row in rows:
        model = row["Model"]
        memory = row["Memory"]
        label = row["Context length"]
        by_model[model].setdefault(memory, {})[label] = row["Throughput (tokens/s)"]

    fig, axes = plt.subplots(1, len(MODELS), figsize=(13, 4.2), constrained_layout=True)
    bar_width = 0.36

    for ax, model in zip(axes, MODELS):
        positions = list(range(len(labels)))

        for index, config in enumerate(MEMORY_CONFIGS):
            offset = (index - 0.5) * bar_width
            values = [
                by_model[model].get(config["name"], {}).get(label, 0.0)
                for label in labels
            ]
            ax.bar(
                [pos + offset for pos in positions],
                values,
                width=bar_width,
                label=f'{config["name"]} (32ch/{config["banks"]} banks)',
                color=config["color"],
                edgecolor="black",
                linewidth=0.6,
            )

        ax.set_title(model)
        ax.set_xlabel("Context length")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Token throughput (tokens/s)")
    axes[-1].legend(loc="upper right", fontsize=9)
    fig.suptitle("Max-pipeline-parallel token throughput", fontsize=13)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300)
    print(f"Wrote {output_path}")


def main():
    args = parse_args()
    MEMORY_CONFIGS[0]["path"] = args.lpddr4_csv
    MEMORY_CONFIGS[1]["path"] = args.gddr6_csv

    rows = []
    for config in MEMORY_CONFIGS:
        rows.extend(load_input_rows(config, args.seqlens))
    rows = select_max_pipeline_rows(rows, args.seqlens)

    rows.sort(
        key=lambda row: (
            MODELS.index(row["Model"]),
            args.seqlens.index(row["Sequence length"])
            if row["Sequence length"] in args.seqlens
            else len(args.seqlens),
            row["Memory"],
        )
    )
    write_source_data(rows, args.csv_output)
    plot(rows, args.output, args.seqlens)
    print(f"Wrote {args.csv_output}")


if __name__ == "__main__":
    main()
