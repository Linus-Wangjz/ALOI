#!/usr/bin/env python3
"""Generate the fixed TP=1,2,4,8 KV-head equal-power all-context results."""

from pathlib import Path

from analyze_balanced_equal_power import main


CENT_SIM = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = CENT_SIM / "output/kv_head_tp_equal_power_all_contexts"


if __name__ == "__main__":
    raise SystemExit(
        main(
            default_experiment_root=EXPERIMENT_ROOT,
            default_attention_mapping="kv_head",
            default_tp_values=(1, 2, 4, 8),
        )
    )
