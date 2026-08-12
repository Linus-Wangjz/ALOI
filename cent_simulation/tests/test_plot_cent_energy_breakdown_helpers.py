import tempfile
import unittest
from pathlib import Path

from scripts import plot_cent_energy_breakdown as plot


def model_row(tp=4, mapping="inter_device"):
    return {
        "Model": "Llama2-70B",
        "Device number": "32",
        "Pipeline parallelism": str(32 // tp),
        "Tensor parallelism": str(tp),
        "Channels per block": str(32 * tp),
        "Sequence length": "129280",
        "Context window": "131072",
        "Throughput (tokens/s)": "1.0",
        "Token energy (mJ)": "1.0",
        "Attention mapping": mapping,
        "DRAM energy model": "legacy",
    }


class EnergyBreakdownHelperPathTest(unittest.TestCase):
    def test_inter_device_uses_attention_helper(self):
        root = Path("/tmp/log-root")
        main, helper, selected = plot.trace_logs(model_row(), "model_parallel", root)
        self.assertEqual(selected, "model_parallel")
        self.assertIn("model_parallel_helper_attention", str(helper))
        self.assertNotIn("model_parallel_FC", str(helper))
        self.assertIn("model_parallel", str(main))

    def test_master_uses_fc_helper(self):
        _, helper, _ = plot.trace_logs(model_row(mapping="master"), "model_parallel", Path("/tmp/log-root"))
        self.assertIn("model_parallel_FC", str(helper))

    def test_tp1_has_no_helper_and_row_mode_prefers_model_parallel(self):
        row = model_row(tp=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_log = root / "model_parallel/Llama2-70B/trace_1_FC_devices_seqlen_129280.txt.log"
            pipeline_log = root / "pipeline_parallel/Llama2-70B/trace_32_channels_per_block_seqlen_129280.txt.log"
            model_log.parent.mkdir(parents=True)
            pipeline_log.parent.mkdir(parents=True)
            model_log.touch()
            pipeline_log.touch()
            self.assertEqual(plot.row_mode(row, root), "model_parallel")
            _, helper, _ = plot.trace_logs(row, "model_parallel", root)
            self.assertIsNone(helper)

    def test_select_row_filters_attention_mapping(self):
        rows = [model_row(mapping="master"), model_row(mapping="inter_device")]
        selected = plot.select_row(rows, "Llama2-70B", 131072, "model_parallel", "legacy", 16.0, 0.0, True)
        self.assertEqual(selected["Attention mapping"], "inter_device")


if __name__ == "__main__":
    unittest.main()
