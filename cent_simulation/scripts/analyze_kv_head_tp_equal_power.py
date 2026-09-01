#!/usr/bin/env python3
"""Generate the fixed TP=1,2,4,8 KV-head Vector all-context results."""

from pathlib import Path

from analyze_balanced_equal_power import main
from scripts.utility.vector_systolic_style_presentation import (
    write_vector_systolic_style_presentation,
)


CENT_SIM = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = CENT_SIM / "output/kv_head_tp_vector_all_context"
VEGA_ALTAIR_FIRST_THREE = {
    "GDDR6": "#4C78A8",
    "LPDDR4X_nCCD2": "#F58518",
    "LPDDR4X_nCCD6": "#E45756",
}


if __name__ == "__main__":
    raise SystemExit(
        main(
            default_experiment_root=EXPERIMENT_ROOT,
            default_attention_mapping="kv_head",
            default_tp_values=(1, 2, 4, 8),
            scalar_memory_colors=VEGA_ALTAIR_FIRST_THREE,
            presentation_writer=write_vector_systolic_style_presentation,
        )
    )
