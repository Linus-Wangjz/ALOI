import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


CENT_SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CENT_SIM))

import run_sim  # noqa: E402


def args(*, inter_device_attention=True, tp_values=None):
    return SimpleNamespace(
        model="Llama2-70B",
        num_channels=32,
        num_banks=8,
        num_devices=32,
        reuse_size=32,
        inter_device_attention=inter_device_attention,
        tp_values=tp_values,
    )


class HelperTraceCommandTest(unittest.TestCase):
    def test_inter_device_helper_has_local_attention_without_main_only_ops(self):
        command = run_sim.model_parallel_helper_trace_command(
            args(), "python", "--Llama-GQA", 4, 129280, 131072, "/tmp/helper.txt"
        )

        for flag in (
            "--inter-device-attention",
            "--trace-fc-kqvo",
            "--trace-attention",
            "--trace-fc-ffn",
        ):
            self.assertIn(flag, command)
        self.assertNotIn("--only-FC", command)
        self.assertNotIn("--op-trace", command)

    def test_master_attention_keeps_fc_only_helper(self):
        command = run_sim.model_parallel_helper_trace_command(
            args(inter_device_attention=False),
            "python",
            "--Llama-GQA",
            4,
            129280,
            131072,
            "/tmp/helper.txt",
        )

        self.assertIn("--only-FC", command)
        self.assertIn("--op-trace", command)
        self.assertNotIn("--trace-attention", command)

    def test_tp1_skips_helper_mode(self):
        self.assertEqual(run_sim.model_parallel_modes_for_tp(args(), 1), ["model_parallel"])
        self.assertEqual(
            run_sim.model_parallel_modes_for_tp(args(), 4),
            ["model_parallel", run_sim.INTER_DEVICE_HELPER_MODE],
        )

    def test_tp_filter_is_sorted_unique_and_validated(self):
        self.assertEqual(run_sim.model_parallel_tp_values(args(tp_values=[4, 1, 2, 4])), [1, 2, 4])
        with self.assertRaises(ValueError):
            run_sim.model_parallel_tp_values(args(tp_values=[3]))


if __name__ == "__main__":
    unittest.main()
